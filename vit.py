import os
import torch
import numpy as np
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
from typing import Dict, List, Optional
import math
import argparse
from torch.utils.tensorboard import SummaryWriter
import functools
import time
from dataclasses import dataclass, field
from collections import defaultdict
import wandb
# 시각화를 위한 라이브러리 추가
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Hugging Face datasets 라이브러리 import
from datasets import load_dataset
from PIL import Image
# timm 라이브러리 import
try:
    from timm.data import create_transform, Mixup
except ImportError:
    print("ERROR: timm library not found. Please install it using 'pip install timm'")
    exit(1)

# Shampoo 옵티마이저 라이브러리 import
from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    SGDGraftingConfig,
    AdamGraftingConfig,
    DDPShampooConfig,
    CommunicationDType
)
from optimizers.distributed_shampoo.utils.shampoo_preconditioner_list import ShampooPreconditionerList
from optimizers.matrix_functions import matrix_inverse_root, check_diagonal

# ========================================
# Enhanced Wall Clock Time Measurement Classes
# ========================================

@dataclass
class TimingStats:
    """타이밍 통계를 저장하는 클래스"""
    total_time: float = 0.0
    count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    
    # Per-epoch tracking
    epoch_time: float = 0.0
    epoch_count: int = 0
    epoch_min_time: float = float('inf')
    epoch_max_time: float = 0.0
    
    def update(self, time_delta: float):
        """새로운 시간 측정값 추가"""
        self.total_time += time_delta
        self.count += 1
        self.min_time = min(self.min_time, time_delta)
        self.max_time = max(self.max_time, time_delta)
        
        self.epoch_time += time_delta
        self.epoch_count += 1
        self.epoch_min_time = min(self.epoch_min_time, time_delta)
        self.epoch_max_time = max(self.epoch_max_time, time_delta)
    
    def reset_epoch_stats(self):
        """Reset per-epoch statistics while keeping cumulative"""
        self.epoch_time = 0.0
        self.epoch_count = 0
        self.epoch_min_time = float('inf')
        self.epoch_max_time = 0.0
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def epoch_avg_time(self) -> float:
        return self.epoch_time / self.epoch_count if self.epoch_count > 0 else 0.0

class EnhancedWallClockProfiler:
    """Enhanced Wall clock time profiler with per-epoch tracking"""
    
    def __init__(self, use_cuda_sync: bool = True):
        self.use_cuda_sync = use_cuda_sync
        self.timers = defaultdict(TimingStats)
        self.active_timers = {}
        self.epoch_start_time = None
        self.training_start_time = None
        self.current_epoch = -1
        
    def start_timer(self, name: str):
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.active_timers[name] = time.perf_counter()
    
    def end_timer(self, name: str) -> float:
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if name not in self.active_timers:
            return 0.0
        
        elapsed = time.perf_counter() - self.active_timers[name]
        self.timers[name].update(elapsed)
        del self.active_timers[name]
        return elapsed
    
    def get_stats(self, name: str) -> TimingStats:
        return self.timers[name]
    
    def reset_epoch_timers(self, epoch: int):
        self.current_epoch = epoch
        self.epoch_start_time = time.perf_counter()
        for stats in self.timers.values():
            stats.reset_epoch_stats()
    
    def get_epoch_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for name, stats in self.timers.items():
            if stats.epoch_count > 0:
                summary[name] = {
                    'epoch_time': stats.epoch_time,
                    'epoch_avg_time': stats.epoch_avg_time,
                    'epoch_count': stats.epoch_count
                }
        return summary
    
    def get_cumulative_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for name, stats in self.timers.items():
            summary[name] = {
                'total_time': stats.total_time,
                'avg_time': stats.avg_time,
                'count': stats.count
            }
        return summary

wall_clock_profiler = EnhancedWallClockProfiler(use_cuda_sync=True)

# ========================================
# Monitoring Classes & Functions
# ========================================

class EighMonitor:
    def __init__(self):
        self.epoch_count = 0
        self.total_count = 0
        self.original_eigh = torch.linalg.eigh

    def eigh_wrapper(self, A, UPLO='L', *, out=None):
        self.epoch_count += 1
        self.total_count += 1
        return self.original_eigh(A, UPLO, out=out)

    def reset_epoch(self):
        self.epoch_count = 0

def count_total_shampoo_factors(optimizer):
    total_count = 0
    if hasattr(optimizer, '_per_group_state_lists'):
        for state_lists in optimizer._per_group_state_lists:
            if 'shampoo_preconditioner_list' in state_lists:
                preconditioner_list = state_lists['shampoo_preconditioner_list']
                for kronecker_factors in preconditioner_list._masked_kronecker_factors_list:
                    total_count += len(kronecker_factors.factor_matrices)
    return total_count

class ShampooMonitor:
    """Shampoo 업데이트 통계 및 DryShampoo 핵심 지표 시각화 클래스"""
    def __init__(self, save_dir, rank, world_size):
        self.save_dir = save_dir
        self.rank = rank
        self.world_size = world_size
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.epochs = []
        # 에폭별 소요 시간 저장
        self.epoch_times = []

        # DryShampoo Stats
        self.epoch_stats = defaultdict(lambda: {'L_updated': 0, 'L_total': 0, 'R_updated': 0, 'R_total': 0})
        self.epoch_eigh_times = {}
        self.rc_history = defaultdict(list)
        # Epsilon을 L/R로 구분하여 저장
        self.epsilon_history = defaultdict(lambda: {'L': [], 'R': []})
        self.param_index_to_name = {}

        self.block_history = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: {'updated' : 0 , 'total' : 0})))

    def register_param_names(self, model, optimizer):
        optim_params = optimizer.param_groups[0]['params']
        param_id_to_index = {id(p): i for i, p in enumerate(optim_params)}
        for name, param in model.named_parameters():
            if id(param) in param_id_to_index:
                idx = param_id_to_index[id(param)]
                self.param_index_to_name[str(idx)] = name

    def log_metric(self, epoch, train_loss, val_loss, val_acc, epoch_time):
        if self.rank != 0: return
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.epoch_times.append(epoch_time)

    def log_update(self, epoch, block_id, param_idx, dim_idx, updated, rc_value, epsilon_value):
        # 모든 Rank가 자신의 통계를 기록
        key_prefix = 'L' if dim_idx == 0 else 'R'
        
        self.epoch_stats[epoch][f'{key_prefix}_total'] += 1
        if updated:
            self.epoch_stats[epoch][f'{key_prefix}_updated'] += 1
        
        stats = self.block_history[block_id][dim_idx][epoch]
        stats['total'] += 1
        if updated:
            stats['updated'] += 1
            
        if rc_value is not None:
            self.rc_history[epoch].append(rc_value)
        
        if epsilon_value is not None:
            self.epsilon_history[epoch][key_prefix].append(epsilon_value)

    def log_eigh_time(self, epoch, time_sec):
        if self.rank != 0: return
        self.epoch_eigh_times[epoch] = time_sec

    def get_local_epoch_data(self, epoch):
        """현재 Rank의 에폭 데이터를 Tensor로 반환 (All-Reduce용)"""
        stats = self.epoch_stats[epoch]
        # 1. Update Counts
        l_updated = stats['L_updated']
        r_updated = stats['R_updated']
        
        # 2. Epsilon Sums and Counts
        eps_dict = self.epsilon_history[epoch]
        l_eps_list = eps_dict['L']
        r_eps_list = eps_dict['R']
        
        sum_eps_l = sum(l_eps_list)
        cnt_eps_l = len(l_eps_list)
        sum_eps_r = sum(r_eps_list)
        cnt_eps_r = len(r_eps_list)
        
        # [L_updated, R_updated, Sum_Eps_L, Count_Eps_L, Sum_Eps_R, Count_Eps_R]
        return torch.tensor([
            l_updated, r_updated, 
            sum_eps_l, cnt_eps_l, 
            sum_eps_r, cnt_eps_r
        ], dtype=torch.float64)

    def gather_stats(self):
        """Gather stats from all ranks to rank 0 (Training 완료 후 호출)"""
        if self.world_size <= 1:
            return

        # [수정] 중첩된 defaultdict를 재귀적으로 일반 dict로 변환하는 헬퍼 함수
        def recursive_to_dict(d):
            if isinstance(d, defaultdict):
                return {k: recursive_to_dict(v) for k, v in d.items()}
            return d

        # 1. Epoch Stats
        # epoch_stats는 1단계 깊이라서 dict()만 해도 내부 값이 일반 dict이므로 문제 없음
        local_epoch_stats = dict(self.epoch_stats)
        gathered_epoch_stats = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_epoch_stats, local_epoch_stats)

        # 2. Block History
        # [수정] 중첩된 lambda를 제거하기 위해 재귀적으로 dict로 변환
        local_block_history = recursive_to_dict(self.block_history)
        gathered_block_history = [None for _ in range(self.world_size)]
        
        # 이제 모든 내부 객체가 일반 dict이므로 pickle 에러가 발생하지 않음
        dist.all_gather_object(gathered_block_history, local_block_history)

        if self.rank == 0:
            # Merge Epoch Stats (Sum)
            merged_epoch_stats = defaultdict(lambda: {'L_updated': 0, 'L_total': 0, 'R_updated': 0, 'R_total': 0})
            for stats in gathered_epoch_stats:
                for epoch, counts in stats.items():
                    for k in counts:
                        merged_epoch_stats[epoch][k] += counts[k]
            self.epoch_stats = merged_epoch_stats

            # Merge Block History (Update)
            merged_block_history = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'updated': 0, 'total': 0})))
            for history in gathered_block_history:
                for block_id, dim_dict in history.items():
                    for dim_idx, epoch_dict in dim_dict.items():
                        merged_block_history[block_id][dim_idx].update(epoch_dict)
            self.block_history = merged_block_history

    def save_heatmap_plots(self):
        if self.rank != 0: return
        if not self.block_history:
            return

        epochs = sorted(list(self.epoch_stats.keys()))
        
        def sort_key_wrapper(x):
            parts = x.split('.')
            try:
                param_idx = int(parts[0])
            except ValueError:
                param_idx = parts[0]
            
            block_name = parts[1] if len(parts) > 1 else ""
            block_num = -1
            if block_name and block_name[-1].isdigit():
                num_str = ''
                for char in reversed(block_name):
                    if char.isdigit():
                        num_str = char + num_str
                    else:
                        break
                if num_str:
                    block_num = int(num_str)
            
            return (param_idx, block_name, block_num)

        block_ids = sorted(self.block_history.keys(), key=sort_key_wrapper)

        def generate_and_save(dim_idx, dim_name):
            data = []
            row_labels = []

            for bid in block_ids:
                param_idx = bid.split('.')[0]
                param_name = self.param_index_to_name.get(param_idx, f"Param{param_idx}")
                full_label = f"[{bid}] {param_name}"

                row_data = []
                has_data = False

                if dim_idx not in self.block_history[bid]:
                    continue

                for e in epochs:
                    stats = self.block_history[bid][dim_idx].get(e, {'updated': 0, 'total': 0})
                    if stats['total'] > 0:
                        pct = (stats['updated'] / stats['total']) * 100.0
                        has_data = True
                    else:
                        pct = np.nan
                    row_data.append(pct)
                
                if has_data:
                    data.append(row_data)
                    row_labels.append(full_label)

            if not data:
                return

            df = pd.DataFrame(data, columns=epochs,index = row_labels)

            chunk_size = 30
            num_chunks = math.ceil(len(df) / chunk_size)

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i+1) * chunk_size, len(df))
                df_subset = df.iloc[start_idx:end_idx]

                if df_subset.empty:continue

                plt.figure(figsize = (15, 12))
                sns.heatmap(df_subset, annot = True, fmt='.0f', cmap='YlGnBu', vmin=0, vmax= 100,
                            cbar_kws = {'label' : 'Update %'})
                plt.title(f"EIGH Update Frequency ({dim_name}) - Part {i+1}/{num_chunks}")
                plt.xlabel("Epoch")
                plt.ylabel("Parameter Block")
                plt.tight_layout()
                
                filename = f"heatmap_eigh_{dim_name}_part{i+1}.png"
                plt.savefig(os.path.join(self.save_dir, filename))
                plt.close()

        generate_and_save(0, "L-Factor")
        generate_and_save(1, "R-Factor")

    def save_plots(self):
        """[Updated] Heatmap을 제외한 다른 Plot 생성 제거"""
        if self.rank != 0: return
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Only Heatmap
        self.save_heatmap_plots()

    
def patch_shampoo_optimizer(optimizer, monitor, current_epoch_fn, eigh_monitor):
    
    def patched_compute_single_root_inverse(
        self,
        factor_matrix,
        inv_factor_matrix,
        is_factor_matrix_diagonal,
        factor_matrix_index,
        root,
        epsilon_value,
        kronecker_factors,
        factor_idx
    ):
        start_eigh_count = eigh_monitor.total_count
        
        bias_corrected_factor_matrix = factor_matrix / self._bias_correction2
        prev_Q = kronecker_factors.eigenvectors[factor_idx]
        prev_D = kronecker_factors.eigenvalues[factor_idx]
        
        current_epsilon = kronecker_factors.adaptive_epsilons[factor_idx]
        if current_epsilon is None:
            current_epsilon = epsilon_value

        should_recompute_eigen = True
        rc_val_to_log = None 

        if prev_Q is not None and prev_D is not None and self._matrix_root_inv_threshold > 0.0:
            try:
                rc_t = self._compute_relative_condition_number(
                    bias_corrected_factor_matrix, prev_Q, prev_D, current_epsilon
                )
                rc_val_to_log = rc_t.item()

                # 2. Calculate Alpha
                inv_root_exponent = -self._exponent_multiplier / root
                h_eigenvalues = (prev_D + current_epsilon).pow(inv_root_exponent)
                
                spectral_norm = h_eigenvalues.abs().max()
                frobenius_norm = torch.norm(h_eigenvalues, p=2)
                alpha = spectral_norm / (frobenius_norm + 1e-25)

                # 3. Propose New Epsilon
                new_epsilon = current_epsilon * ((rc_t * alpha) / self._matrix_root_inv_threshold)
                
                # 4. Check Condition
                if (rc_t * alpha) >= self._matrix_root_inv_threshold:
                    # Unstable
                    if new_epsilon < self._max_epsilon:
                        
                        current_epsilon = float(new_epsilon)
                        should_recompute_eigen = False
                        
                        alpha_pow = -self._exponent_multiplier / root
                        
                        eig_term = (prev_D + current_epsilon).pow(alpha_pow)
                        computed_inv_factor_matrix = prev_Q * eig_term.unsqueeze(0) @ prev_Q.T
                        
                        computed_inv_factor_matrix = computed_inv_factor_matrix.to(dtype=inv_factor_matrix.dtype)
                        inv_factor_matrix.copy_(computed_inv_factor_matrix)
                        kronecker_factors.adaptive_epsilons[factor_idx] = current_epsilon
                    else:
                        # Slow Update (Reset)
                        current_epsilon = epsilon_value
                        should_recompute_eigen = True 
                else:
                    # Stable -> Fast Update
                    current_epsilon = float(new_epsilon)
                    should_recompute_eigen = False
                    
                    alpha_pow = -self._exponent_multiplier / root
                    eig_term = (prev_D + current_epsilon).pow(alpha_pow)
                    computed_inv_factor_matrix = prev_Q * eig_term.unsqueeze(0) @ prev_Q.T
                    
                    computed_inv_factor_matrix = computed_inv_factor_matrix.to(dtype=inv_factor_matrix.dtype)
                    inv_factor_matrix.copy_(computed_inv_factor_matrix)
                    kronecker_factors.adaptive_epsilons[factor_idx] = current_epsilon 
                    
            except Exception:
                should_recompute_eigen = True

        if should_recompute_eigen:
            if is_factor_matrix_diagonal and not check_diagonal(factor_matrix):
                is_factor_matrix_diagonal.copy_(torch.tensor(False))

            try:
                result = matrix_inverse_root(
                    A=bias_corrected_factor_matrix,
                    root=root,
                    epsilon=current_epsilon,
                    exponent_multiplier=self._exponent_multiplier,
                    is_diagonal=is_factor_matrix_diagonal,
                    retry_double_precision=self._use_protected_eigh,
                )
                
                computed_inv_factor_matrix, used_epsilon, L, Q = result
                
                if L is not None and Q is not None:
                    raw_eigenvalues = L - used_epsilon
                    kronecker_factors.eigenvalues[factor_idx] = raw_eigenvalues.to(dtype=factor_matrix.dtype)
                    kronecker_factors.eigenvectors[factor_idx] = Q.to(dtype=factor_matrix.dtype)
                    kronecker_factors.adaptive_epsilons[factor_idx] = current_epsilon
                
                computed_inv_factor_matrix = computed_inv_factor_matrix.to(dtype=inv_factor_matrix.dtype)
                inv_factor_matrix.copy_(computed_inv_factor_matrix)

            except Exception:
                pass

        
        end_eigh_count = eigh_monitor.total_count
        performed_eigh = (end_eigh_count > start_eigh_count)
        
        try:
            parts = factor_matrix_index.split('.')
            
            if len(parts) >= 2:
                block_id = f"{parts[0]}.{parts[1]}"
            else:
                block_id = parts[0]
                
            d_idx = int(parts[-1])
            epoch = current_epoch_fn()
            
            monitor.log_update(epoch, block_id, None, d_idx, performed_eigh, rc_val_to_log, current_epsilon)
        except Exception:
            pass

    from optimizers.distributed_shampoo.utils.shampoo_preconditioner_list import ShampooPreconditionerList
    ShampooPreconditionerList._compute_single_root_inverse = patched_compute_single_root_inverse

def validate_on_trainset(model, train_loader, criterion, device, rank, world_size):
    """Full-batch training loss 계산 (에폭 전체 데이터 사용)"""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    # 훈련 셋 전체를 평가 모드로 순회
    with torch.no_grad():
        for batch in train_loader:
            if isinstance(batch, dict):
                images = batch['pixel_values'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
            else:
                images, labels = batch
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            num_batches += 1
            
    # 분산 환경 집계
    total_loss_tensor = torch.tensor(running_loss).to(device)
    total_batches_tensor = torch.tensor(num_batches).to(device)
    
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_batches_tensor, op=dist.ReduceOp.SUM)
    
    if total_batches_tensor.item() > 0:
        full_train_loss = total_loss_tensor.item() / total_batches_tensor.item()
    else:
        full_train_loss = 0.0
        
    return full_train_loss

# ========================================
# ViT Model Components
# ========================================

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim: int = 384, num_heads: int = 6, attn_dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == self.embedding_dim, "embedding_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, need_weights=False):
        batch_size = query.shape[0]
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embedding_dim)
        output = self.out_proj(attn_output)
        return output, None

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int = 384, num_heads: int = 6, mlp_dim: int = 1536, 
                 attn_dropout: float = 0.0, mlp_dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.attn = CustomMultiheadAttention(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = MLPBlock(embedding_dim=embedding_dim, mlp_dim=mlp_dim, dropout=mlp_dropout)
        
    def forward(self, x):
        norm_x = self.norm1(x)
        attn_output, _ = self.attn(query=norm_x, key=norm_x, value=norm_x, need_weights=False)
        x = x + attn_output
        norm_x_mlp = self.norm2(x)
        mlp_output = self.mlp(norm_x_mlp)
        x = x + mlp_output
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size: int = 224, in_channels: int = 3, patch_size: int = 16, 
                 num_classes: int = 1000, embedding_dim: int = 384, depth: int = 12, 
                 num_heads: int = 6, mlp_dim: int = 1536, attn_dropout: float = 0.0, 
                 mlp_dropout: float = 0.1, embedding_dropout: float = 0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, 
                                        kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_dim=mlp_dim, 
                                   attn_dropout=attn_dropout, mlp_dropout=mlp_dropout) 
            for _ in range(depth)
        ])
        self.classifier_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.classifier_head = nn.Linear(in_features=embedding_dim, out_features=num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x).flatten(2, 3).permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.classifier_norm(x)
        cls_token_final = x[:, 0]
        logits = self.classifier_head(cls_token_final)
        return logits

# ========================================
# Utility Functions
# ========================================

def gather_optimizer_state_from_all_ranks(optimizer, model, global_rank, world_size):
    local_state = optimizer.distributed_state_dict(
        key_to_param=model.module.named_parameters()
    )
    all_states = [None] * world_size
    dist.all_gather_object(all_states, local_state)
    
    if global_rank == 0:
        merged_state = {'state': {}, 'param_groups': all_states[0].get('param_groups', [])}
        for param_key in all_states[0]['state'].keys():
            merged_state['state'][param_key] = {}
            param_state_keys = set()
            for state in all_states:
                if 'state' in state and param_key in state['state']:
                    param_state_keys.update(state['state'][param_key].keys())
            
            for state_key in param_state_keys:
                merged_value = None
                is_factor_matrix = 'factor_matrices' in str(state_key)
                for rank, state in enumerate(all_states):
                    if ('state' in state and param_key in state['state'] and 
                        state_key in state['state'][param_key]):
                        value = state['state'][param_key][state_key]
                        if isinstance(value, torch.Tensor):
                            if hasattr(value, '_local_tensor'): value = value._local_tensor
                            if is_factor_matrix:
                                if value.numel() > 0:
                                    if merged_value is None: merged_value = value.clone()
                                    elif merged_value.numel() == 0: merged_value = value.clone()
                            else:
                                if merged_value is None or (merged_value.numel() == 0 and value.numel() > 0):
                                    merged_value = value.clone()
                        else:
                            if merged_value is None: merged_value = value
                if merged_value is not None:
                    merged_state['state'][param_key][state_key] = merged_value
        return merged_state
    return None

def get_warmup_cosine_decay_lr(current_step: int, base_lr: float, num_steps: int, warmup_steps: int) -> float:
    if current_step < warmup_steps:
        return base_lr * (current_step / warmup_steps)
    else:
        progress = (current_step - warmup_steps) / (num_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return base_lr * cosine_decay

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
def cleanup():
    dist.destroy_process_group()

def apply_transforms(examples: Dict[str, List[Image.Image]], transform) -> Dict[str, List[torch.Tensor]]:
    examples['pixel_values'] = [transform(image.convert("RGB")) for image in examples['image']]
    return examples

def get_epsilon_config(args):
    presets = {
        'default': {
            'epsilon': 1e-09,
            'epsilon_left': None,
            'epsilon_right': None,
            'use_adaptive_epsilon': False,
            'condition_thresholds': None
        },
        'asymmetric': {
            'epsilon': 1e-09,
            'epsilon_left': 1e-08,
            'epsilon_right': 1e-08,
            'use_adaptive_epsilon': False,
            'condition_thresholds': None
        },
    }
    if args.epsilon_preset in presets:
        return presets[args.epsilon_preset]
    else:
        # Fallback for manual config
        config = {'epsilon': 1e-08, 'epsilon_left': None, 'epsilon_right': None, 'use_adaptive_epsilon': False}
        if args.epsilon is not None: config['epsilon'] = args.epsilon
        return config

class EighFallbackCounter(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0
        self.setLevel(logging.WARNING)
    def emit(self, record):
        try:
            if "Retrying in double precision" in record.getMessage(): self.count += 1
        except Exception: self.handleError(record)

def set_seed_distributed(seed: int = 42, rank: int = 0):
    rank_seed = seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank_seed)
        torch.cuda.manual_seed_all(rank_seed)
    os.environ['PYTHONHASHSEED'] = str(rank_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# ========================================
# Main Training Function
# ========================================

def train(args: argparse.Namespace):
    setup()
    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    set_seed_distributed(seed = args.seed, rank = global_rank)
    
    if global_rank == 0:
        print(f"Running DDP training. Global Rank: {global_rank}, Local Rank: {local_rank}, World Size: {world_size}")
        wandb.init(
            project = args.project,
            entity = args.entity,
            config = vars(args),
            dir=args.log_dir    
        )

    writer = SummaryWriter(log_dir=args.log_dir) if global_rank == 0 else None
    wall_clock_profiler.training_start_time = time.perf_counter()
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_transform = create_transform(
        input_size=224,
        is_training=True,
        auto_augment='rand-m15-n2-mstd0.5',
        interpolation='bicubic',
    )
    val_transform = create_transform(
        input_size=224,
        is_training=False,
        interpolation='bicubic',
    )

    mixup_fn = None
    if args.mixup > 0 or args.label_smoothing > 0:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=0.0, label_smoothing=args.label_smoothing, num_classes=1000)

    if global_rank == 0:
        load_dataset("imagenet-1k", cache_dir=args.data_path)
    dist.barrier()

    dataset = load_dataset("imagenet-1k", cache_dir=args.data_path)
    train_dataset = dataset['train']
    train_dataset.set_transform(functools.partial(apply_transforms, transform=train_transform))
    val_dataset = dataset['validation']
    val_dataset.set_transform(functools.partial(apply_transforms, transform=val_transform))
    
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'label': torch.tensor([x['label'] for x in batch], dtype=torch.long)
        }

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                             num_workers=args.workers, pin_memory=True, collate_fn=collate_fn, 
                             worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed))
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, 
                           num_workers=args.workers, pin_memory=True, collate_fn=collate_fn, 
                           worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed))

    model = VisionTransformer(img_size=224, patch_size=16, embedding_dim=384, depth=12,
                              num_heads=6, mlp_dim=1536, num_classes=1000).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss().to(local_rank)
    
    epsilon_config = get_epsilon_config(args)

    eigh_monitor = EighMonitor()
    eigh_monitor.original_eigh = torch.linalg.eigh
    def monitored_timed_eigh(A, UPLO='L', *, out=None):
        wall_clock_profiler.start_timer("eigendecomposition")
        try:
            return eigh_monitor.eigh_wrapper(A, UPLO, out=out)
        finally:
            wall_clock_profiler.end_timer("eigendecomposition")
    torch.linalg.eigh = monitored_timed_eigh

    from optimizers.distributed_shampoo.utils.shampoo_preconditioner_list import ShampooPreconditionerList
    original_compute_root_inverse = ShampooPreconditionerList.compute_root_inverse
    def timed_compute_root_inverse_wrapper(self):
        wall_clock_profiler.start_timer("compute_root_inverse_total")
        res = original_compute_root_inverse(self)
        wall_clock_profiler.end_timer("compute_root_inverse_total")
        return res
    ShampooPreconditionerList.compute_root_inverse = timed_compute_root_inverse_wrapper

    distributed_config = DDPShampooConfig(
        communication_dtype=CommunicationDType.FP32,
        num_trainers_per_group=world_size,
        communicate_params=False
    )
    
    optimizer = DistributedShampoo(
        params=model.parameters(),
        lr=args.base_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        **epsilon_config,
        momentum=0.0,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        grafting_config=AdamGraftingConfig(beta2=args.adam_grafting_beta2, epsilon=args.grafting_epsilon),
        use_decoupled_weight_decay=True,
        inv_root_override=2,
        exponent_multiplier=1,
        distributed_config=distributed_config,
        preconditioner_dtype=torch.float32,
        matrix_root_inv_threshold=args.matrix_root_inv_threshold,
        max_epsilon=args.max_epsilon
    )

    monitor = ShampooMonitor(save_dir=args.log_dir, rank=global_rank, world_size=world_size) # Modified: passed world_size
    monitor.register_param_names(model, optimizer)
    current_epoch_ref = {'epoch': 0}
    patch_shampoo_optimizer(optimizer, monitor, lambda: current_epoch_ref['epoch'], eigh_monitor)

    eigh_fallback_handler = EighFallbackCounter()
    matrix_logger = logging.getLogger('optimizers.matrix_functions')
    matrix_logger.setLevel(logging.WARNING)
    matrix_logger.addHandler(eigh_fallback_handler)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=f'cuda:{local_rank}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_distributed_state_dict(checkpoint['optimizer_state_dict'], key_to_param=model.module.named_parameters())
        start_epoch = checkpoint['epoch'] + 1

    total_steps = len(train_loader) * args.epochs
    
    for epoch in range(start_epoch, args.epochs):
        current_epoch_ref['epoch'] = epoch
        wall_clock_profiler.reset_epoch_timers(epoch)
        eigh_monitor.reset_epoch()
        
        epoch_start_time = time.perf_counter()
        train_sampler.set_epoch(epoch)
        eigh_fallback_handler.count = 0
        model.train()
        
        for i, batch in enumerate(train_loader):
            current_step = epoch * len(train_loader) + i
            images = batch['pixel_values'].to(local_rank, non_blocking=True)
            labels = batch['label'].to(local_rank, non_blocking=True)
            if mixup_fn: images, labels = mixup_fn(images, labels)

            new_lr = get_warmup_cosine_decay_lr(current_step, args.base_lr, total_steps, args.warmup_steps)
            for param_group in optimizer.param_groups: param_group['lr'] = new_lr

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            loss_for_logging = loss.clone().detach()
            dist.all_reduce(loss_for_logging, op=dist.ReduceOp.SUM)
            avg_loss = loss_for_logging.item() / world_size
            if global_rank == 0 and (i + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{i+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
            
        # Feature 4: Eigh Time Logging
        epoch_eigendecomp_time = wall_clock_profiler.get_stats("eigendecomposition").epoch_time
        dist_eigh_time = torch.tensor(epoch_eigendecomp_time).to(local_rank)
        dist.all_reduce(dist_eigh_time, op=dist.ReduceOp.SUM)
        avg_eigh_time = dist_eigh_time.item() / world_size
        if global_rank == 0:
            monitor.log_eigh_time(epoch, avg_eigh_time)

        # [NOTE] Epoch Wall-clock Time ends here
        epoch_duration = time.perf_counter() - epoch_start_time

        # Feature 3: Full-batch Training Loss Calculation
        full_train_loss = 0.0
        if epoch == args.epochs - 1:
            if global_rank == 0:
                print(f"Calculating Full-batch Training Loss for Epoch {epoch+1}...")
            full_train_loss = validate_on_trainset(model, train_loader, criterion, local_rank, global_rank, world_size)
        else:
            if global_rank == 0:
                pass
        
        # Validation
        model.eval()
        correct = 0; total = 0; val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['pixel_values'].to(local_rank, non_blocking=True)
                labels = batch['label'].to(local_rank, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        total_t = torch.tensor(total).to(local_rank)
        correct_t = torch.tensor(correct).to(local_rank)
        val_loss_t = torch.tensor(val_loss_sum).to(local_rank)
        dist.all_reduce(total_t); dist.all_reduce(correct_t); dist.all_reduce(val_loss_t)
        
        val_acc = 100 * correct_t.item() / total_t.item() if total_t.item() > 0 else 0.0
        avg_val_loss = val_loss_t.item() / world_size / len(val_loader)

        # -----------------------------------------------------------
        # [MODIFIED] Aggregate Eigh Counts & Epsilon & Rates across ALL ranks
        # -----------------------------------------------------------
        
        # [L_updated, R_updated, Sum_Eps_L, Count_Eps_L, Sum_Eps_R, Count_Eps_R]
        local_stats = monitor.get_local_epoch_data(epoch)
        local_stats = local_stats.to(local_rank)
        
        # 2. All-Reduce SUM
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
        
        
        global_l_up = int(local_stats[0].item())      
        global_r_up = int(local_stats[1].item())      
        
        global_l_total = int(local_stats[3].item())   
        global_r_total = int(local_stats[5].item())   
        
        # 4.Global Update Percentage
        l_update_pct = (global_l_up / global_l_total * 100) if global_l_total > 0 else 0.0
        r_update_pct = (global_r_up / global_r_total * 100) if global_r_total > 0 else 0.0
        
        # 5. Epsilon
        avg_eps_l = local_stats[2].item() / max(1, global_l_total)
        avg_eps_r = local_stats[4].item() / max(1, global_r_total)

        if global_rank == 0:
            print(f"Epoch {epoch+1}: Train Loss (Full) {full_train_loss:.4f}, Val Acc {val_acc:.2f}%, Val Loss {avg_val_loss:.4f}, Time {epoch_duration:.2f}s")
            print(f"  Total L/R Eigh Counts: {global_l_up} / {global_r_up}")
            print(f"  Total L/R Blocks: {global_l_total} / {global_r_total}")  
            print(f"  Global Update % (L/R): {l_update_pct:.2f}% / {r_update_pct:.2f}%") 
            print(f"  Avg Epsilon L/R: {avg_eps_l:.2e} / {avg_eps_r:.2e}")
            
            monitor.log_metric(epoch + 1, full_train_loss, avg_val_loss, val_acc, epoch_duration)
            
            # WandB Logging
            wandb.log({
                'train/iteration_loss': avg_loss,
                'train_loss': full_train_loss,
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'epoch': epoch,
                'learning_rate': new_lr,
                'shampoo/avg_eigh_time': avg_eigh_time,
                'shampoo/epoch_time': epoch_duration,
                'shampoo/L_update_pct': l_update_pct,       # Global %
                'shampoo/R_update_pct': r_update_pct,       # Global %
                'shampoo/total_L_eigh_count': global_l_up,
                'shampoo/total_R_eigh_count': global_r_up,
                'shampoo/avg_epsilon_L': avg_eps_l,
                'shampoo/avg_epsilon_R': avg_eps_r
            })

        # Checkpoint
        if (epoch + 1) % args.save_interval == 0:
            merged_state = gather_optimizer_state_from_all_ranks(optimizer, model, global_rank, world_size)
            if global_rank == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': merged_state,
                    'accuracy': val_acc
                }, os.path.join(args.save_dir, f"vit_epoch_{epoch+1}.pth"))
            dist.barrier()

    if global_rank == 0:
        print("Gathering statistics from all ranks...")
    
    # Gather stats from all ranks (for heatmap plots)
    monitor.gather_stats()

    if global_rank == 0:
        monitor.save_plots()
        wandb.finish()
        if writer: writer.close()
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT-Base Training with DryShampoo Monitoring')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--base-lr', type=float, default=1e-3)
    parser.add_argument('--warmup-steps', type=int, default=10000)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--beta1', type=float, default=0.95)
    parser.add_argument('--beta2', type=float, default=0.995)
    parser.add_argument('--adam-grafting-beta2', type=float, default=0.995)
    parser.add_argument('--grafting-epsilon', type=float, default=1e-09)
    parser.add_argument('--max-preconditioner-dim', type=int, default=1024)
    parser.add_argument('--precondition-frequency', type=int, default=15)
    parser.add_argument('--start-preconditioning-step', type=int, default=15)
    parser.add_argument('--mixup', type=float, default=0.2)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--matrix-root-inv-threshold', type=float, default=0.0)
    parser.add_argument('--max-epsilon', type=float, default=1e-07)
    parser.add_argument('--project', type=str, default='DryShampoo_Experiment_ViT')
    parser.add_argument('--entity', type=str, default = 'Kyunghun')

    parser.add_argument('--epsilon-preset', type=str, default='default', choices=['default', 'asymmetric'])
    parser.add_argument('--epsilon', type=float, default=None)
    parser.add_argument('--epsilon-left', type=float, default=None)
    parser.add_argument('--epsilon-right', type=float, default=None)
    parser.add_argument('--use-adaptive-epsilon', action='store_true')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
