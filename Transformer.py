import os
import time
import math
import random
import argparse
import logging
import functools
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 시각화 및 로깅
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Hugging Face 라이브러리
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
except ImportError:
    print("ERROR: transformers or datasets library not found.")
    print("Please install via 'pip install transformers datasets'")
    exit(1)

# Shampoo Optimizer Imports
# (경로는 실제 환경에 맞춰 수정 필요, 제공된 파일 구조 기준)
from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    DDPShampooConfig,
    CommunicationDType
)
from optimizers.distributed_shampoo.utils.shampoo_preconditioner_list import ShampooPreconditionerList
from optimizers.matrix_functions import matrix_inverse_root, check_diagonal

# ========================================
# 1. Profiling & Monitoring Classes (Derived from vit.py)
# ========================================

@dataclass
class TimingStats:
    total_time: float = 0.0
    count: int = 0
    epoch_time: float = 0.0
    epoch_count: int = 0
    
    def update(self, time_delta: float):
        self.total_time += time_delta
        self.count += 1
        self.epoch_time += time_delta
        self.epoch_count += 1
    
    def reset_epoch_stats(self):
        self.epoch_time = 0.0
        self.epoch_count = 0

class EnhancedWallClockProfiler:
    def __init__(self, use_cuda_sync: bool = True):
        self.use_cuda_sync = use_cuda_sync
        self.timers = defaultdict(TimingStats)
        self.active_timers = {}
        self.training_start_time = None
        
    def start_timer(self, name: str):
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.active_timers[name] = time.perf_counter()
    
    def end_timer(self, name: str) -> float:
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        if name not in self.active_timers: return 0.0
        elapsed = time.perf_counter() - self.active_timers[name]
        self.timers[name].update(elapsed)
        del self.active_timers[name]
        return elapsed
    
    def get_stats(self, name: str) -> TimingStats:
        return self.timers[name]
    
    def reset_epoch_timers(self, epoch: int):
        for stats in self.timers.values():
            stats.reset_epoch_stats()

wall_clock_profiler = EnhancedWallClockProfiler(use_cuda_sync=True)

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
        
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
        # DryShampoo Stats
        self.epoch_stats = defaultdict(lambda: {'L_updated': 0, 'L_total': 0, 'R_updated': 0, 'R_total': 0})
        self.epoch_eigh_times = {}
        self.rc_history = defaultdict(list)
        self.epsilon_history = defaultdict(lambda: {'L': [], 'R': []})
        self.param_index_to_name = {}

    def register_param_names(self, model, optimizer):
        optim_params = optimizer.param_groups[0]['params']
        param_id_to_index = {id(p): i for i, p in enumerate(optim_params)}
        for name, param in model.named_parameters():
            if id(param) in param_id_to_index:
                idx = param_id_to_index[id(param)]
                self.param_index_to_name[str(idx)] = name

    def log_metric(self, epoch, train_loss, val_loss, epoch_time):
        if self.rank != 0: return
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def log_update(self, epoch, block_id, dim_idx, updated, rc_value, epsilon_value):
        key_prefix = 'L' if dim_idx == 0 else 'R'
        self.epoch_stats[epoch][f'{key_prefix}_total'] += 1
        if updated:
            self.epoch_stats[epoch][f'{key_prefix}_updated'] += 1
        
        if rc_value is not None:
            self.rc_history[epoch].append(rc_value)
        if epsilon_value is not None:
            self.epsilon_history[epoch][key_prefix].append(epsilon_value)

    def log_eigh_time(self, epoch, time_sec):
        if self.rank != 0: return
        self.epoch_eigh_times[epoch] = time_sec

    def get_local_epoch_data(self, epoch):
        stats = self.epoch_stats[epoch]
        eps_dict = self.epsilon_history[epoch]
        
        # Safe sum/len handling for empty lists
        sum_l = sum(eps_dict['L']) if eps_dict['L'] else 0.0
        len_l = len(eps_dict['L'])
        sum_r = sum(eps_dict['R']) if eps_dict['R'] else 0.0
        len_r = len(eps_dict['R'])

        return torch.tensor([
            stats['L_updated'], stats['R_updated'],
            sum_l, len_l,
            sum_r, len_r,
            stats['L_total'], stats['R_total'] 
        ], dtype=torch.float64)

    def save_plots(self):
        if self.rank != 0: return
        os.makedirs(self.save_dir, exist_ok=True)
        # 여기에 필요한 Plotting 로직 추가 (RC Distribution 등)

# ========================================
# 2. DryShampoo Logic Patch (Adaptive Epsilon)
# ========================================

def patch_shampoo_optimizer(optimizer, monitor, current_epoch_fn, eigh_monitor):
    """DistributedShampoo에 DryShampoo 로직(적응형 Epsilon 등)을 주입"""
    
    def patched_compute_single_root_inverse(
        self, factor_matrix, inv_factor_matrix, is_factor_matrix_diagonal,
        factor_matrix_index, root, epsilon_value, kronecker_factors, factor_idx
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
                # 1. Compute RC (Relative Condition Number)
                rc_t = self._compute_relative_condition_number(
                    bias_corrected_factor_matrix, prev_Q, prev_D, current_epsilon
                )
                rc_val_to_log = rc_t.item()

                # 2. Calculate Alpha (Scaling factor based on eigenvalues)
                inv_root_exponent = -self._exponent_multiplier / root
                h_eigenvalues = (prev_D + current_epsilon).pow(inv_root_exponent)
                spectral_norm = h_eigenvalues.abs().max()
                frobenius_norm = torch.norm(h_eigenvalues, p=2)
                alpha = spectral_norm / (frobenius_norm + 1e-25)

                # 3. Propose New Epsilon
                new_epsilon = current_epsilon * ((rc_t * alpha) / self._matrix_root_inv_threshold)
                
                # 4. Check Condition (Fast vs Slow Update)
                if (rc_t * alpha) < self._matrix_root_inv_threshold:
                    # Stable state: Fast Update (Reuse Q, update D with new epsilon)
                    current_epsilon = float(new_epsilon)
                    should_recompute_eigen = False
                    
                    alpha_pow = -self._exponent_multiplier / root
                    eig_term = (prev_D + current_epsilon).pow(alpha_pow)
                    computed_inv_factor_matrix = prev_Q * eig_term.unsqueeze(0) @ prev_Q.T
                    
                    computed_inv_factor_matrix = computed_inv_factor_matrix.to(dtype=inv_factor_matrix.dtype)
                    inv_factor_matrix.copy_(computed_inv_factor_matrix)
                    kronecker_factors.adaptive_epsilons[factor_idx] = current_epsilon
                
                elif new_epsilon < self._max_epsilon:
                     # Unstable but epsilon is within bounds -> Fast Update with higher epsilon
                    current_epsilon = float(new_epsilon)
                    should_recompute_eigen = False
                    
                    alpha_pow = -self._exponent_multiplier / root
                    eig_term = (prev_D + current_epsilon).pow(alpha_pow)
                    computed_inv_factor_matrix = prev_Q * eig_term.unsqueeze(0) @ prev_Q.T
                    
                    computed_inv_factor_matrix = computed_inv_factor_matrix.to(dtype=inv_factor_matrix.dtype)
                    inv_factor_matrix.copy_(computed_inv_factor_matrix)
                    kronecker_factors.adaptive_epsilons[factor_idx] = current_epsilon
                else:
                    # Very Unstable -> Slow Update (Full Eigendecomposition with base epsilon)
                    current_epsilon = epsilon_value # Reset to base
                    should_recompute_eigen = True

            except Exception:
                should_recompute_eigen = True

        if should_recompute_eigen:
            if is_factor_matrix_diagonal and not check_diagonal(factor_matrix):
                is_factor_matrix_diagonal.copy_(torch.tensor(False))
            try:
                # Perform full Eigendecomposition
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
                    # Store Raw Eigenvalues (L - epsilon) for next step
                    raw_eigenvalues = L - used_epsilon
                    kronecker_factors.eigenvalues[factor_idx] = raw_eigenvalues.to(dtype=factor_matrix.dtype)
                    kronecker_factors.eigenvectors[factor_idx] = Q.to(dtype=factor_matrix.dtype)
                    kronecker_factors.adaptive_epsilons[factor_idx] = current_epsilon
                
                computed_inv_factor_matrix = computed_inv_factor_matrix.to(dtype=inv_factor_matrix.dtype)
                inv_factor_matrix.copy_(computed_inv_factor_matrix)
            except Exception:
                pass

        # Logging to Monitor
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
            monitor.log_update(epoch, block_id, d_idx, performed_eigh, rc_val_to_log, current_epsilon)
        except Exception:
            pass

    # Patch the class method
    ShampooPreconditionerList._compute_single_root_inverse = patched_compute_single_root_inverse

# ========================================
# 3. Model Architecture (Transformer-Big as per Algoperf)
# ========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerBig(nn.Module):
    """
    Algoperf WMT Benchmark Spec (Transformer-Big):
    - d_model: 1024
    - dim_feedforward: 4096
    - nhead: 16
    - num_layers: 6 (encoder/decoder)
    - Shared Embeddings (Source, Target, Output Projection)
    """
    def __init__(self, ntoken, d_model=1024, nhead=16, d_hid=4096, nlayers=6, dropout=0.1):
        super(TransformerBig, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        
        # Shared Embedding
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True # [Batch, Seq, Feature]
        )
        
        self.output_projection = nn.Linear(d_model, ntoken)

        self._init_weights()
        self._tie_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)

    def _tie_weights(self):
        # Share Embedding & Output Projection Weights
        self.output_projection.weight = self.embedding.weight

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # src, tgt: [Batch, Seq]
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb.permute(1, 0, 2)).permute(1, 0, 2) # Back to batch first for Transformer
        
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb.permute(1, 0, 2)).permute(1, 0, 2)

        output = self.transformer(
            src_emb, tgt_emb, 
            src_mask=src_mask, tgt_mask=tgt_mask, 
            src_key_padding_mask=src_padding_mask, 
            tgt_key_padding_mask=tgt_padding_mask
        )
        output = self.output_projection(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# ========================================
# 4. Utilities
# ========================================

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def set_seed_distributed(seed: int = 42, rank: int = 0):
    rank_seed = seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    torch.backends.cudnn.benchmark = True

def gather_optimizer_state_from_all_ranks(optimizer, model, global_rank, world_size):
    # Checkpointing utility
    local_state = optimizer.distributed_state_dict(key_to_param=model.module.named_parameters())
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
                for state in all_states:
                    if ('state' in state and param_key in state['state'] and 
                        state_key in state['state'][param_key]):
                        value = state['state'][param_key][state_key]
                        if isinstance(value, torch.Tensor):
                            if hasattr(value, '_local_tensor'): value = value._local_tensor
                            if merged_value is None or (merged_value.numel() == 0 and value.numel() > 0):
                                merged_value = value.clone()
                        else:
                            if merged_value is None: merged_value = value
                if merged_value is not None:
                    merged_state['state'][param_key][state_key] = merged_value
        return merged_state
    return None

class EighFallbackCounter(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0
        self.setLevel(logging.WARNING)
    def emit(self, record):
        if "Retrying in double precision" in record.getMessage(): self.count += 1

# ========================================
# 5. Main Training Function
# ========================================

def train(args):
    local_rank = setup()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    set_seed_distributed(args.seed, global_rank)

    if global_rank == 0:
        print(f"Starting WMT Transformer-Big Training (DryShampoo). World Size: {world_size}")
        wandb.init(project=args.project, entity=args.entity, config=vars(args), dir=args.log_dir)

    # 1. Tokenizer & Dataset
    # Algoperf uses SentencePiece (32k vocab). We use a standard pre-trained equivalent for reproducibility.
    # 'Helsinki-NLP/opus-mt-de-en' uses roughly this config.
    tokenizer_name = "Helsinki-NLP/opus-mt-de-en" 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size

    if global_rank == 0:
        print("Loading dataset (WMT14 de-en)...")
        # Algoperf uses WMT17 train, but WMT14 is standard for validation/test.
        # We load a subset for quick demo, remove `split` arg for full training.
        load_dataset("wmt14", "de-en", cache_dir=args.data_path) 
    dist.barrier()
    
    dataset = load_dataset("wmt14", "de-en", cache_dir=args.data_path)
    train_ds = dataset['train']
    val_ds = dataset['validation']

    def preprocess_function(examples):
        inputs = [ex['de'] for ex in examples['translation']]
        targets = [ex['en'] for ex in examples['translation']]
        
        # Algoperf filters max_len=256
        model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Tokenize
    tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(preprocess_function, batched=True, remove_columns=val_ds.column_names)
    
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_sampler = DistributedSampler(tokenized_train, num_replicas=world_size, rank=global_rank, shuffle=True)
    train_loader = DataLoader(tokenized_train, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(tokenized_val, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    # 2. Model: Transformer-small (Algoperf Spec)
    model = TransformerBig(
        ntoken=vocab_size, 
        d_model=args.d_model,    # args 값 사용
        nhead=args.nhead,        # args 값 사용
        d_hid=args.d_hid,        # args 값 사용
        nlayers=args.nlayers,    # args 값 사용
        dropout=args.dropout     # args 값 사용
    ).to(local_rank)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers = False)

    # Eigh Monitoring Setup
    eigh_monitor = EighMonitor()
    eigh_monitor.original_eigh = torch.linalg.eigh
    def monitored_eigh(A, UPLO='L', *, out=None):
        wall_clock_profiler.start_timer("eigendecomposition")
        try:
            return eigh_monitor.eigh_wrapper(A, UPLO, out=out)
        finally:
            wall_clock_profiler.end_timer("eigendecomposition")
    torch.linalg.eigh = monitored_eigh

    # Distributed Shampoo Config
    distributed_config = DDPShampooConfig(
        communication_dtype=CommunicationDType.FP32,
        num_trainers_per_group=world_size,
        communicate_params=False
    )
    
    # Epsilon Config
    epsilon_config = {'epsilon': 1e-10}
    if args.epsilon_preset == 'asymmetric':
        epsilon_config = {'epsilon': 1e-10, 'epsilon_left': 1e-8, 'epsilon_right': 1e-8}

    # Optimizer: DryShampoo Config
    # WMT Transformer typically uses inv_root_override=2
    optimizer = DistributedShampoo(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        **epsilon_config,
        momentum=0.0,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        grafting_config=AdamGraftingConfig(beta2=args.beta2, epsilon=1e-8),
        use_decoupled_weight_decay=True,
        inv_root_override=2, 
        exponent_multiplier=1,
        distributed_config=distributed_config,
        preconditioner_dtype=torch.float32,
        matrix_root_inv_threshold=args.matrix_root_inv_threshold,
        max_epsilon=args.max_epsilon
    )

    # Monitor & Patch
    monitor = ShampooMonitor(save_dir=args.log_dir, rank=global_rank, world_size=world_size)
    monitor.register_param_names(model, optimizer)
    current_epoch_ref = {'epoch': 0}
    patch_shampoo_optimizer(optimizer, monitor, lambda: current_epoch_ref['epoch'], eigh_monitor)

    eigh_fallback_handler = EighFallbackCounter()
    matrix_logger = logging.getLogger('optimizers.matrix_functions')
    matrix_logger.setLevel(logging.WARNING)
    matrix_logger.addHandler(eigh_fallback_handler)

    # Algoperf uses Label Smoothing=0.1
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    
    # 3. Training Loop
    total_steps = len(train_loader) * args.epochs
    
    for epoch in range(args.epochs):
        current_epoch_ref['epoch'] = epoch
        wall_clock_profiler.reset_epoch_timers(epoch)
        eigh_monitor.reset_epoch()
        eigh_fallback_handler.count = 0
        train_sampler.set_epoch(epoch)
        
        epoch_start = time.perf_counter()
        
        model.train()
        total_loss = 0
        
        for i, batch in enumerate(train_loader):
            step = epoch * len(train_loader) + i
            
            # LR Scheduler (Linear Warmup + Cosine Decay)
            if step < args.warmup_steps:
                lr = args.lr * (step / args.warmup_steps)
            else:
                progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
            
            for pg in optimizer.param_groups: pg['lr'] = lr

            src = batch['input_ids'].to(local_rank)
            tgt = batch['labels'].to(local_rank)
            
            # Decoder Input: Shift right (remove last token, prepend start token implicitly handled by tokenizer structure or custom logic)
            # Standard causal masking setup:
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            tgt_mask = model.module.generate_square_subsequent_mask(tgt_input.size(1)).to(local_rank)
            src_padding_mask = (src == tokenizer.pad_token_id).to(local_rank)
            tgt_padding_mask = (tgt_input == tokenizer.pad_token_id).to(local_rank)

            optimizer.zero_grad()
            output = model(src, tgt_input, tgt_mask=tgt_mask, 
                           src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
            
            # Reshape for loss
            loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.detach().item()
            
            if i % args.log_interval == 0 and global_rank == 0:
                print(f"Epoch {epoch} | Step {i} | Loss {loss.detach().item():.4f} | LR {lr:.6f}")
                wandb.log({"train_loss": loss.detach().item(), "lr": lr})

        epoch_duration = time.perf_counter() - epoch_start
        
        # Aggregate Eigh Stats & Logs
        local_stats = monitor.get_local_epoch_data(epoch).to(local_rank)
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
        
        if global_rank == 0:
            avg_train_loss = total_loss / len(train_loader)
            
            l_up, r_up = int(local_stats[0]), int(local_stats[1])
            l_tot, r_tot = int(local_stats[6]), int(local_stats[7])
            
            l_pct = (l_up / l_tot * 100) if l_tot > 0 else 0
            r_pct = (r_up / r_tot * 100) if r_tot > 0 else 0
            
            # Avg Epsilon
            avg_eps_l = local_stats[2] / max(1, local_stats[3])
            avg_eps_r = local_stats[4] / max(1, local_stats[5])

            print(f"==> Epoch {epoch} Summary: Avg Loss {avg_train_loss:.4f}, Time {epoch_duration:.2f}s")
            print(f"    DryShampoo Updates (L/R): {l_pct:.1f}% / {r_pct:.1f}%")
            print(f"    Eigh Fallbacks: {eigh_fallback_handler.count}")
            
            wandb.log({
                "epoch_loss": avg_train_loss,
                "epoch_time": epoch_duration,
                "shampoo/L_update_pct": l_pct,
                "shampoo/R_update_pct": r_pct,
                "shampoo/avg_epsilon_L": avg_eps_l,
                "shampoo/avg_epsilon_R": avg_eps_r,
                "epoch": epoch
            })
            
            # Simple Validation (Loss Only)
            model.eval()
            val_loss_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    src = batch['input_ids'].to(local_rank)
                    tgt = batch['labels'].to(local_rank)
                    tgt_input = tgt[:, :-1]
                    tgt_output = tgt[:, 1:]
                    
                    tgt_mask = model.module.generate_square_subsequent_mask(tgt_input.size(1)).to(local_rank)
                    src_padding_mask = (src == tokenizer.pad_token_id).to(local_rank)
                    tgt_padding_mask = (tgt_input == tokenizer.pad_token_id).to(local_rank)

                    output = model(src, tgt_input, tgt_mask=tgt_mask, 
                                   src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
                    loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
                    val_loss_sum += loss.item()
            
            avg_val_loss = val_loss_sum / len(val_loader)
            wandb.log({"val_loss": avg_val_loss})

        # Save Checkpoint
        if (epoch + 1) % args.save_interval == 0:
            merged_state = gather_optimizer_state_from_all_ranks(optimizer, model, global_rank, world_size)
            if global_rank == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': merged_state,
                    'loss': avg_train_loss
                }, os.path.join(args.save_dir, f"transformer_epoch_{epoch+1}.pth"))
            dist.barrier()

    if global_rank == 0:
        monitor.save_plots() # Heatmap 저장 등
        wandb.finish()
        
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WMT14 Transformer-Big Training with DryShampoo')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--log-dir', type=str, default='logs_transformer')
    parser.add_argument('--save-dir', type=str, default='checkpoints_transformer')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32) # Global batch size should be around 280k tokens, adjust per GPU
    parser.add_argument('--lr', type=float, default=0.002) # Algoperf base LR
    parser.add_argument('--warmup-steps', type=int, default=4000)
    
    # Model Params (Transformer Big)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--nlayers', type=int, default=4)
    parser.add_argument('--d-hid', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Shampoo Params
    parser.add_argument('--max-preconditioner-dim', type=int, default=1024)
    parser.add_argument('--precondition-frequency', type=int, default=10)
    parser.add_argument('--start-preconditioning-step', type=int, default=10)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    
    # DryShampoo Params
    parser.add_argument('--matrix-root-inv-threshold', type=float, default=0.1)
    parser.add_argument('--max-epsilon', type=float, default=1e-6)
    parser.add_argument('--epsilon-preset', type=str, default='default', choices=['default', 'asymmetric'])
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--save-interval', type=int, default=5)
    parser.add_argument('--project', type=str, default='DryShampoo-Transformer')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
