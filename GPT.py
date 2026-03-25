import os
import torch
import numpy as np
import torch.nn as nn
import random
import torch.nn.functional as F
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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Hugging Face 라이브러리 import
from datasets import load_dataset
from transformers import AutoTokenizer

# Shampoo 옵티마이저 라이브러리 import
try:
    from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
    from optimizers.distributed_shampoo.shampoo_types import (
        AdamGraftingConfig,
        DDPShampooConfig,
        CommunicationDType
    )
    from optimizers.distributed_shampoo.utils.shampoo_preconditioner_list import ShampooPreconditionerList
    from optimizers.matrix_functions import matrix_inverse_root, check_diagonal
except ImportError:
    print("ERROR: optimizers.distributed_shampoo not found. Make sure it is in your python path.")
    exit(1)

# ========================================
# Enhanced Wall Clock Time Measurement Classes
# ========================================

@dataclass
class TimingStats:
    total_time: float = 0.0
    count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    epoch_time: float = 0.0
    epoch_count: int = 0
    epoch_min_time: float = float('inf')
    epoch_max_time: float = 0.0
    
    def update(self, time_delta: float):
        self.total_time += time_delta
        self.count += 1
        self.min_time = min(self.min_time, time_delta)
        self.max_time = max(self.max_time, time_delta)
        self.epoch_time += time_delta
        self.epoch_count += 1
        self.epoch_min_time = min(self.epoch_min_time, time_delta)
        self.epoch_max_time = max(self.epoch_max_time, time_delta)
    
    def reset_epoch_stats(self):
        self.epoch_time = 0.0
        self.epoch_count = 0
        self.epoch_min_time = float('inf')
        self.epoch_max_time = 0.0
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0

class EnhancedWallClockProfiler:
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

class ShampooMonitor:
    def __init__(self, save_dir, rank, world_size):
        self.save_dir = save_dir
        self.rank = rank
        self.world_size = world_size
        self.epochs = []
        self.epoch_stats = defaultdict(lambda: {'L_updated': 0, 'L_total': 0, 'R_updated': 0, 'R_total': 0})
        self.epoch_eigh_times = {}
        self.rc_history = defaultdict(list)
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

    def log_metric(self, epoch, train_loss, val_loss, val_ppl, epoch_time):
        if self.rank != 0: return
        self.epochs.append(epoch)

    def log_update(self, epoch, block_id, param_idx, dim_idx, updated, rc_value, epsilon_value):
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
        stats = self.epoch_stats[epoch]
        l_updated, r_updated = stats['L_updated'], stats['R_updated']
        eps_dict = self.epsilon_history[epoch]
        l_eps_list, r_eps_list = eps_dict['L'], eps_dict['R']
        return torch.tensor([
            l_updated, r_updated, sum(l_eps_list), len(l_eps_list), sum(r_eps_list), len(r_eps_list)
        ], dtype=torch.float64)

    def gather_stats(self):
        if self.world_size <= 1: return
        def recursive_to_dict(d):
            if isinstance(d, defaultdict): return {k: recursive_to_dict(v) for k, v in d.items()}
            return d
        local_epoch_stats = dict(self.epoch_stats)
        gathered_epoch_stats = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_epoch_stats, local_epoch_stats)
        local_block_history = recursive_to_dict(self.block_history)
        gathered_block_history = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_block_history, local_block_history)

        if self.rank == 0:
            merged_epoch_stats = defaultdict(lambda: {'L_updated': 0, 'L_total': 0, 'R_updated': 0, 'R_total': 0})
            for stats in gathered_epoch_stats:
                for epoch, counts in stats.items():
                    for k in counts: merged_epoch_stats[epoch][k] += counts[k]
            self.epoch_stats = merged_epoch_stats

            merged_block_history = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'updated': 0, 'total': 0})))
            for history in gathered_block_history:
                for block_id, dim_dict in history.items():
                    for dim_idx, epoch_dict in dim_dict.items():
                        merged_block_history[block_id][dim_idx].update(epoch_dict)
            self.block_history = merged_block_history

    def save_plots(self):
        pass # 필요 시 heatmap 로직 추가 (생략)

def patch_shampoo_optimizer(optimizer, monitor, current_epoch_fn, eigh_monitor):
    def patched_compute_single_root_inverse(
        self, factor_matrix, inv_factor_matrix, is_factor_matrix_diagonal,
        factor_matrix_index, root, epsilon_value, kronecker_factors, factor_idx
    ):
        start_eigh_count = eigh_monitor.total_count
        bias_corrected_factor_matrix = factor_matrix / self._bias_correction2
        prev_Q = kronecker_factors.eigenvectors[factor_idx]
        prev_D = kronecker_factors.eigenvalues[factor_idx]
        current_epsilon = kronecker_factors.adaptive_epsilons[factor_idx] or epsilon_value
        should_recompute_eigen = True
        rc_val_to_log = None 

        if prev_Q is not None and prev_D is not None and self._matrix_root_inv_threshold > 0.0:
            try:
                rc_t = self._compute_relative_condition_number(bias_corrected_factor_matrix, prev_Q, prev_D, current_epsilon)
                rc_val_to_log = rc_t.item()
                inv_root_exponent = -self._exponent_multiplier / root
                h_eigenvalues = (prev_D + current_epsilon).pow(inv_root_exponent)
                spectral_norm = h_eigenvalues.abs().max()
                frobenius_norm = torch.norm(h_eigenvalues, p=2)
                alpha = spectral_norm / (frobenius_norm + 1e-25)
                new_epsilon = current_epsilon * ((rc_t * (alpha / root)) / self._matrix_root_inv_threshold)
                
                if (rc_t * (alpha / root)) >= self._matrix_root_inv_threshold:
                    if new_epsilon < self._max_epsilon:
                        current_epsilon = float(new_epsilon)
                        should_recompute_eigen = False
                        eig_term = (prev_D + current_epsilon).pow(-self._exponent_multiplier / root)
                        inv_factor_matrix.copy_((prev_Q * eig_term.unsqueeze(0) @ prev_Q.T).to(dtype=inv_factor_matrix.dtype))
                        kronecker_factors.adaptive_epsilons[factor_idx] = current_epsilon
                    else:
                        should_recompute_eigen = True 
                else:
                    current_epsilon = max(float(new_epsilon), float(epsilon_value))
                    should_recompute_eigen = False
                    eig_term = (prev_D + current_epsilon).pow(-self._exponent_multiplier / root)
                    inv_factor_matrix.copy_((prev_Q * eig_term.unsqueeze(0) @ prev_Q.T).to(dtype=inv_factor_matrix.dtype))
                    kronecker_factors.adaptive_epsilons[factor_idx] = current_epsilon 
            except Exception:
                should_recompute_eigen = True

        if should_recompute_eigen:
            if is_factor_matrix_diagonal and not check_diagonal(factor_matrix):
                is_factor_matrix_diagonal.copy_(torch.tensor(False))
            try:
                result = matrix_inverse_root(
                    A=bias_corrected_factor_matrix, root=root, epsilon=current_epsilon,
                    exponent_multiplier=self._exponent_multiplier, is_diagonal=is_factor_matrix_diagonal,
                    retry_double_precision=self._use_protected_eigh,
                )
                computed_inv_factor_matrix, used_epsilon, L, Q = result
                if L is not None and Q is not None:
                    kronecker_factors.eigenvalues[factor_idx] = (L - used_epsilon).to(dtype=factor_matrix.dtype)
                    kronecker_factors.eigenvectors[factor_idx] = Q.to(dtype=factor_matrix.dtype)
                    kronecker_factors.adaptive_epsilons[factor_idx] = current_epsilon
                inv_factor_matrix.copy_(computed_inv_factor_matrix.to(dtype=inv_factor_matrix.dtype))
            except Exception: pass
        
        end_eigh_count = eigh_monitor.total_count
        performed_eigh = (end_eigh_count > start_eigh_count)
        try:
            parts = factor_matrix_index.split('.')
            block_id = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else parts[0]
            monitor.log_update(current_epoch_fn(), block_id, None, int(parts[-1]), performed_eigh, rc_val_to_log, current_epsilon)
        except Exception: pass

    ShampooPreconditionerList._compute_single_root_inverse = patched_compute_single_root_inverse

# ========================================
# Small-GPT Model Components
# ========================================

class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, attn_dropout: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.c_attn = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.c_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(attn_dropout)
        self.n_head = num_heads
        self.embedding_dim = embedding_dim

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embedding_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Causal mask is handled efficiently via PyTorch's scaled_dot_product_attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class GPTBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class SmallGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 512, embedding_dim: int = 512, 
                 depth: int = 8, num_heads: int = 8, mlp_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, embedding_dim)
        self.wpe = nn.Embedding(block_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([GPTBlock(embedding_dim, num_heads, mlp_dim, dropout) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight # Weight tying

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

# ========================================
# Utility Functions
# ========================================

def get_warmup_cosine_decay_lr(current_step: int, base_lr: float, num_steps: int, warmup_steps: int) -> float:
    if current_step < warmup_steps: return base_lr * (current_step / warmup_steps)
    progress = (current_step - warmup_steps) / (num_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def set_seed_distributed(seed: int = 42, rank: int = 0):
    rank_seed = seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank_seed)
        torch.cuda.manual_seed_all(rank_seed)
    torch.backends.cudnn.benchmark = True

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
        print(f"Running DDP training on Wikitext-103. Global Rank: {global_rank}, World Size: {world_size}")
        wandb.init(project=args.project, entity=args.entity, config=vars(args), dir=args.log_dir)

    writer = SummaryWriter(log_dir=args.log_dir) if global_rank == 0 else None
    
    # 1. Dataset & DataLoader Preparation (Wikitext-103)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    if global_rank == 0:
        load_dataset("wikitext", "wikitext-103-v1", cache_dir=args.data_path)
    dist.barrier()

    dataset = load_dataset("wikitext", "wikitext-103-v1", cache_dir=args.data_path)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    
    def group_texts(examples):
        block_size = args.block_size
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=4)

    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    
    def collate_fn(batch):
        input_ids = torch.tensor([x['input_ids'] for x in batch], dtype=torch.long)
        labels = torch.tensor([x['labels'] for x in batch], dtype=torch.long)
        return {'input_ids': input_ids, 'labels': labels}

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                              num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, 
                            num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    # 2. Model & Optimizer Setup
    model = SmallGPT(vocab_size=len(tokenizer), block_size=args.block_size).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss().to(local_rank)

    eigh_monitor = EighMonitor()
    eigh_monitor.original_eigh = torch.linalg.eigh
    def monitored_timed_eigh(A, UPLO='L', *, out=None):
        wall_clock_profiler.start_timer("eigendecomposition")
        try: return eigh_monitor.eigh_wrapper(A, UPLO, out=out)
        finally: wall_clock_profiler.end_timer("eigendecomposition")
    torch.linalg.eigh = monitored_timed_eigh

    distributed_config = DDPShampooConfig(
        communication_dtype=CommunicationDType.FP32, num_trainers_per_group=world_size, communicate_params=False
    )
    
    optimizer = DistributedShampoo(
        params=model.parameters(), lr=args.base_lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
        epsilon=args.epsilon, momentum=0.0, max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency, start_preconditioning_step=args.start_preconditioning_step,
        grafting_config=AdamGraftingConfig(beta2=args.adam_grafting_beta2, epsilon=args.grafting_epsilon),
        use_decoupled_weight_decay=True, inv_root_override=2, exponent_multiplier=1,
        distributed_config=distributed_config, preconditioner_dtype=torch.float32,
        matrix_root_inv_threshold=args.matrix_root_inv_threshold, max_epsilon=args.max_epsilon
    )

    monitor = ShampooMonitor(save_dir=args.log_dir, rank=global_rank, world_size=world_size)
    monitor.register_param_names(model, optimizer)
    current_epoch_ref = {'epoch': 0}
    patch_shampoo_optimizer(optimizer, monitor, lambda: current_epoch_ref['epoch'], eigh_monitor)

    total_steps = len(train_loader) * args.epochs
    
    # 3. Training Loop
    for epoch in range(args.epochs):
        current_epoch_ref['epoch'] = epoch
        wall_clock_profiler.reset_epoch_timers(epoch)
        eigh_monitor.reset_epoch()
        epoch_start_time = time.perf_counter()
        
        train_sampler.set_epoch(epoch)
        model.train()
        
        for i, batch in enumerate(train_loader):
            current_step = epoch * len(train_loader) + i
            input_ids = batch['input_ids'].to(local_rank, non_blocking=True)
            labels = batch['labels'].to(local_rank, non_blocking=True)

            new_lr = get_warmup_cosine_decay_lr(current_step, args.base_lr, total_steps, args.warmup_steps)
            for param_group in optimizer.param_groups: param_group['lr'] = new_lr

            optimizer.zero_grad()
            logits = model(input_ids)
            
            # 언어 모델의 Shift loss 계산
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            if global_rank == 0 and (i + 1) % args.log_interval == 0:
                dist.all_reduce(loss.clone().detach(), op=dist.ReduceOp.SUM)
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
        epoch_eigendecomp_time = wall_clock_profiler.get_stats("eigendecomposition").epoch_time
        dist_eigh_time = torch.tensor(epoch_eigendecomp_time).to(local_rank)
        dist.all_reduce(dist_eigh_time, op=dist.ReduceOp.SUM)
        avg_eigh_time = dist_eigh_time.item() / world_size
        if global_rank == 0: monitor.log_eigh_time(epoch, avg_eigh_time)

        epoch_duration = time.perf_counter() - epoch_start_time

        # 4. Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(local_rank, non_blocking=True)
                labels = batch['labels'].to(local_rank, non_blocking=True)
                logits = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                val_loss_sum += loss.item()

        val_loss_t = torch.tensor(val_loss_sum).to(local_rank)
        dist.all_reduce(val_loss_t)
        avg_val_loss = val_loss_t.item() / world_size / len(val_loader)
        val_ppl = math.exp(avg_val_loss)

        local_stats = monitor.get_local_epoch_data(epoch).to(local_rank)
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
        
        if global_rank == 0:
            print(f"Epoch {epoch+1}: Val PPL {val_ppl:.2f}, Val Loss {avg_val_loss:.4f}, Time {epoch_duration:.2f}s")
            monitor.log_metric(epoch + 1, 0.0, avg_val_loss, val_ppl, epoch_duration)
            wandb.log({
                'val_perplexity': val_ppl,
                'val_loss': avg_val_loss,
                'epoch': epoch,
                'learning_rate': new_lr,
                'shampoo/avg_eigh_time': avg_eigh_time,
                'shampoo/epoch_time': epoch_duration,
            })

    monitor.gather_stats()
    if global_rank == 0:
        monitor.save_plots()
        wandb.finish()
        if writer: writer.close()
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Small-GPT Training with Shampoo Monitoring')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--block-size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--base-lr', type=float, default=3e-4)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--adam-grafting-beta2', type=float, default=0.99)
    parser.add_argument('--grafting-epsilon', type=float, default=1e-8)
    parser.add_argument('--max-preconditioner-dim', type=int, default=1024)
    parser.add_argument('--precondition-frequency', type=int, default=100)
    parser.add_argument('--start-preconditioning-step', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--matrix-root-inv-threshold', type=float, default=0.5)
    parser.add_argument('--max-epsilon', type=float, default=5e-07)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--project', type=str, default='Shampoo_GPT_Wikitext103')
    parser.add_argument('--entity', type=str, default='Kyunghun')
    
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
