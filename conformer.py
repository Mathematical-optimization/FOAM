import os
import math
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
import torchaudio
import logging
from torch.utils.data import ConcatDataset
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Shampoo Optimizer Imports
from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    DDPShampooConfig,
    CommunicationDType
)
from optimizers.matrix_functions import matrix_inverse_root, check_diagonal

# ==========================================
# 1. Utilities & Setup & Monitoring
# ==========================================

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

class EighFallbackCounter(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0
        self.setLevel(logging.WARNING)

    def emit(self, record):
        try:
            message = record.getMessage()
            if "Retrying in double precision" in message:
                self.count += 1
        except Exception:
            self.handleError(record)

class ShampooMonitor:
    def __init__(self, save_dir, rank):
        self.save_dir = save_dir
        self.rank = rank
        
        self.train_losses = []
        self.epochs = []
        
        # DryShampoo Stats
        self.epoch_stats = defaultdict(lambda: {'L_updated': 0, 'L_total': 0, 'R_updated': 0, 'R_total': 0})
        self.epoch_eigh_times = {}
        self.rc_history = defaultdict(list)
        self.epsilon_history = defaultdict(list)
        self.param_index_to_name = {}

    def register_param_names(self, model, optimizer):
        optim_params = optimizer.param_groups[0]['params']
        param_id_to_index = {id(p): i for i, p in enumerate(optim_params)}
        for name, param in model.named_parameters():
            if id(param) in param_id_to_index:
                idx = param_id_to_index[id(param)]
                self.param_index_to_name[str(idx)] = name

    def log_metric(self, epoch, train_loss):
        if self.rank != 0: return
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)

    def log_update(self, epoch, param_idx, dim_idx, updated, rc_value, epsilon_value):
        if self.rank != 0: return
        key_prefix = 'L' if dim_idx == 0 else 'R'
        self.epoch_stats[epoch][f'{key_prefix}_total'] += 1
        if updated:
            self.epoch_stats[epoch][f'{key_prefix}_updated'] += 1
        if rc_value is not None:
            self.rc_history[epoch].append(rc_value)
        if epsilon_value is not None:
            self.epsilon_history[epoch].append(epsilon_value)

    def log_eigh_time(self, epoch, time_sec):
        if self.rank != 0: return
        self.epoch_eigh_times[epoch] = time_sec

    def save_plots(self):
        if self.rank != 0: return
        os.makedirs(self.save_dir, exist_ok=True)

        if self.epochs:
            plt.figure(figsize=(8, 6))
            plt.plot(self.epochs, self.train_losses, label='Train Loss', marker='.')
            plt.title('Loss Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
            plt.close()

        if self.epoch_stats:
            epochs = sorted(self.epoch_stats.keys())
            l_ratios = [100.0 * self.epoch_stats[e]['L_updated'] / max(1, self.epoch_stats[e]['L_total']) for e in epochs]
            r_ratios = [100.0 * self.epoch_stats[e]['R_updated'] / max(1, self.epoch_stats[e]['R_total']) for e in epochs]
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, l_ratios, label='L-Factor', marker='o')
            plt.plot(epochs, r_ratios, label='R-Factor', marker='s')
            plt.title('Preconditioner Update Percentage (Freshness)')
            plt.xlabel('Epoch')
            plt.ylabel('Update %')
            plt.ylim(-5, 105)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, 'shampoo_update_percentage.png'))
            plt.close()

        epochs = sorted(self.rc_history.keys())
        if self.rc_history:
            plt.figure(figsize=(12, 6))
            data = []
            for e in epochs:
                if e in self.rc_history:
                    data.extend([{'Epoch': e, 'RC': val} for val in self.rc_history[e]])
            if data:
                df_rc = pd.DataFrame(data)
                sns.boxplot(x='Epoch', y='RC', data=df_rc, showfliers=False)
                plt.title('Distribution of Relative Condition (RC) Numbers per Epoch')
                plt.grid(True, axis='y')
                plt.savefig(os.path.join(self.save_dir, 'rc_distribution.png'))
            plt.close()

        if self.epsilon_history:
            avg_eps = [sum(self.epsilon_history[e])/len(self.epsilon_history[e]) for e in epochs if e in self.epsilon_history]
            valid_epochs = [e for e in epochs if e in self.epsilon_history]
            plt.figure(figsize=(10, 6))
            plt.plot(valid_epochs, avg_eps, marker='^', color='purple')
            plt.title('Average Adaptive Epsilon per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Epsilon')
            plt.yscale('log')
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, 'epsilon_evolution.png'))
            plt.close()

        if self.epoch_eigh_times:
            eigh_epochs = sorted(self.epoch_eigh_times.keys())
            times = [self.epoch_eigh_times[e] for e in eigh_epochs]
            plt.figure(figsize=(10, 6))
            plt.plot(eigh_epochs, times, marker='*', color='red', linestyle='-')
            plt.title('Average Eigendecomposition Wall-Clock Time per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Time (seconds)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'eigh_time_evolution.png'))
            plt.close()

def patch_shampoo_optimizer(optimizer, monitor, current_epoch_fn, eigh_monitor):
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
                rc_t = self._compute_relative_condition_number(
                    bias_corrected_factor_matrix, prev_Q, prev_D, current_epsilon
                )
                rc_val_to_log = rc_t.item()
                new_epsilon = current_epsilon * (rc_t / self._matrix_root_inv_threshold)
                
                if rc_t >= self._matrix_root_inv_threshold:
                    if new_epsilon < self._max_epsilon:
                        current_epsilon = float(new_epsilon)
                        should_recompute_eigen = False
                        alpha = -self._exponent_multiplier / root
                        eig_term = (prev_D + current_epsilon).pow(alpha)
                        computed_inv_factor_matrix = prev_Q * eig_term.unsqueeze(0) @ prev_Q.T
                        computed_inv_factor_matrix = computed_inv_factor_matrix.to(dtype=inv_factor_matrix.dtype)
                        inv_factor_matrix.copy_(computed_inv_factor_matrix)
                        kronecker_factors.adaptive_epsilons[factor_idx] = current_epsilon
                    else:
                        current_epsilon = epsilon_value
                        should_recompute_eigen = True 
                else:
                    current_epsilon = float(new_epsilon)
                    should_recompute_eigen = False
                    alpha = -self._exponent_multiplier / root
                    eig_term = (prev_D + current_epsilon).pow(alpha)
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
                computed_inv_factor_matrix, _, L, Q = result
                if L is not None and Q is not None:
                    kronecker_factors.eigenvalues[factor_idx] = L.to(dtype=factor_matrix.dtype)
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
            p_idx = parts[0]
            d_idx = int(parts[-1])
            epoch = current_epoch_fn()
            monitor.log_update(epoch, p_idx, d_idx, performed_eigh, rc_val_to_log, current_epsilon)
        except Exception:
            pass

    from optimizers.distributed_shampoo.utils.shampoo_preconditioner_list import ShampooPreconditionerList
    ShampooPreconditionerList._compute_single_root_inverse = patched_compute_single_root_inverse

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

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def get_lr_schedule(current_step, warmup_steps, base_lr, total_steps):
    if current_step < warmup_steps:
        return base_lr * (current_step / max(1, warmup_steps))
    else:
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# ==========================================
# 2. Tokenizer & Dataset
# ==========================================

class TextTransform:
    def __init__(self):
        self.char_map = {"'": 0, " ": 1}
        self.index_map = {0: "'", 1: " "}
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz", start=2):
            self.char_map[c] = i
            self.index_map[i] = c
            
    def text_to_int(self, text):
        targets = []
        for c in text.lower():
            if c in self.char_map:
                targets.append(self.char_map[c])
        return torch.tensor(targets, dtype=torch.long)

    def int_to_text(self, labels):
        string = []
        for i in labels:
            string.append(self.index_map.get(i.item(), ""))
        return "".join(string)

    def __len__(self):
        return len(self.char_map) + 1 

class AlgoPerfLibriSpeech(Dataset):
    def __init__(self, root, url="train-clean-100", download=True, train=True, args=None):
        if not os.path.isdir(root):
            os.makedirs(root, exist_ok=True)
        
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=url, download=download
        )
        self.train = train
        self.args = args
        
        if self.train:
            ds1 = torchaudio.datasets.LIBRISPEECH(root=root, url="train-clean-100", download=download)
            ds2 = torchaudio.datasets.LIBRISPEECH(root=root, url="train-clean-360", download=download)
            ds3 = torchaudio.datasets.LIBRISPEECH(root=root, url="train-other-500", download=download)
            self.dataset = ConcatDataset([ds1, ds2, ds3])
        else:
            self.dataset = torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=download)
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels
        )
        
        self.spec_augment = nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=35),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=27)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, transcript, _, _, _ = self.dataset[idx]
        
        if waveform.shape[1] > self.args.max_audio_length:
            return None
            
        spec = self.melspec(waveform)
        spec = torch.log(spec + 1e-9)
        
        if self.train:
            spec = self.spec_augment(spec)
        
        return spec.squeeze(0).transpose(0, 1), transcript

text_transform = TextTransform()

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None, None

    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for spec, transcript in batch:
        spectrograms.append(spec)
        label = text_transform.text_to_int(transcript)
        labels.append(label)
        input_lengths.append(spec.shape[0]) 
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, torch.tensor(input_lengths), torch.tensor(label_lengths)

# ==========================================
# 3. Model
# ==========================================

class ConformerAlgoPerf(nn.Module):
    def __init__(self, num_classes, input_dim=80, encoder_dim=512, num_layers=4, num_heads=8, depthwise_kernel_size=31):
        super(ConformerAlgoPerf, self).__init__()
        
        self.conformer = torchaudio.models.Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=encoder_dim * 4,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_kernel_size
        )
        
        self.fc = nn.Linear(input_dim, num_classes) 
        self.input_projection = nn.Linear(input_dim, input_dim) 
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, input_lengths):
        x = self.input_projection(x)
        out, out_lengths = self.conformer(x, input_lengths)
        out = self.fc(out)
        out = F.log_softmax(out, dim=2)
        out = out.transpose(0, 1) 
        return out, out_lengths

# ==========================================
# 4. Main Training Loop
# ==========================================

def main(args):
    local_rank = setup()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    set_seed_distributed(args.seed, global_rank)
    wall_clock_profiler.training_start_time = time.perf_counter()

    if global_rank == 0:
        print(f"Training Config: {args}")
        print(f"World Size: {world_size}, Local Rank: {local_rank}")
        print(f"Model Architecture: Conformer {args.num_layers} layers (AlgoPerf Spec)")

    if global_rank == 0:
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)
        print("Downloading Librispeech datasets...")
        torchaudio.datasets.LIBRISPEECH(root=args.data_path, url="train-clean-100", download=True)
        torchaudio.datasets.LIBRISPEECH(root=args.data_path, url="train-clean-360", download=True)
        torchaudio.datasets.LIBRISPEECH(root=args.data_path, url="train-other-500", download=True)  
    dist.barrier()

    train_dataset = AlgoPerfLibriSpeech(
        root=args.data_path, 
        url="train-clean-100", 
        download=False,
        train=True,
        args=args
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    model = ConformerAlgoPerf(
        num_classes=len(text_transform),
        input_dim=args.n_mels,
        encoder_dim=args.encoder_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        depthwise_kernel_size=args.depthwise_kernel_size
    ).to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Logging Setup
    eigh_fallback_handler = EighFallbackCounter()
    matrix_logger = logging.getLogger('optimizers.matrix_functions')
    matrix_logger.setLevel(logging.WARNING)
    matrix_logger.addHandler(eigh_fallback_handler)

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
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        epsilon=1e-10,
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
        use_protected_eigh=True,
        matrix_root_inv_threshold=0.0,
    )

    # Monitor Setup
    monitor = ShampooMonitor(save_dir='logs_conformer', rank=global_rank)
    monitor.register_param_names(model, optimizer)
    current_epoch_ref = {'epoch': 0}
    patch_shampoo_optimizer(optimizer, monitor, lambda: current_epoch_ref['epoch'], eigh_monitor)

    criterion = nn.CTCLoss(blank=len(text_transform)-1).to(local_rank)
    
    total_steps = len(train_loader) * args.epochs
    global_step = 0

    if global_rank == 0:
        print("Start Training...")

    model.train()
    for epoch in range(args.epochs):
        current_epoch_ref['epoch'] = epoch
        wall_clock_profiler.reset_epoch_timers(epoch)
        eigh_monitor.reset_epoch()
        eigh_fallback_handler.count = 0

        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        valid_batches = 0
        
        for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(train_loader):
            if spectrograms is None: continue
            
            spectrograms = spectrograms.to(local_rank, non_blocking=True)
            labels = labels.to(local_rank, non_blocking=True)
            input_lengths = input_lengths.to(local_rank, non_blocking=True)
            label_lengths = label_lengths.to(local_rank, non_blocking=True)
            
            current_lr = get_lr_schedule(global_step, args.warmup_steps, args.lr, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()
            
            outputs, output_lengths = model(spectrograms, input_lengths)
            loss = criterion(outputs, labels, output_lengths, label_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                if global_rank == 0:
                    print(f"Warning: NaN/Inf loss at step {global_step}. Skipping.")
                continue

            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            valid_batches += 1
            global_step += 1
            
            if batch_idx % args.log_interval == 0 and global_rank == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

        # Epoch End
        if global_rank == 0:
            avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0.0
            print(f"==> Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")
            print(f"    Eigh Fallbacks (Epoch): {eigh_fallback_handler.count}")
            
            # Monitor Metrics
            monitor.log_metric(epoch + 1, avg_loss)
            
            # Eigh Time Logging
            epoch_eigendecomp_time = wall_clock_profiler.get_stats("eigendecomposition").epoch_time
            dist_eigh_time = torch.tensor(epoch_eigendecomp_time).to(local_rank)
            dist.all_reduce(dist_eigh_time)
            avg_eigh_time = dist_eigh_time.item() / world_size
            monitor.log_eigh_time(epoch + 1, avg_eigh_time)

        if (epoch + 1) % args.save_interval == 0:
            if global_rank == 0:
                save_path = os.path.join(args.checkpoint_dir, f"conformer_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.distributed_state_dict(key_to_param=model.module.named_parameters()),
                    'loss': avg_loss,
                }, save_path)
                print(f"Checkpoint saved to {save_path}")
            dist.barrier()

    if global_rank == 0:
        monitor.save_plots()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conformer Training with Distributed Shampoo (AlgoPerf)')
    
    # Data Params
    parser.add_argument('--data-path', type=str, default='./data', help='Dataset path')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--n-fft', type=int, default=400)
    parser.add_argument('--hop-length', type=int, default=160)
    parser.add_argument('--n-mels', type=int, default=80)
    parser.add_argument('--max-audio-length', type=int, default=320000)
    
    # Model Params
    parser.add_argument('--encoder-dim', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=4, help="AlgoPerf uses 4 layers") 
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--depthwise-kernel-size', type=int, default=32)
    
    # Training Params
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=50, help='Per-GPU batch size')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--warmup-steps', type=int, default=569400)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98) 
    
    # Shampoo Params
    parser.add_argument('--max-preconditioner-dim', type=int, default=1024)
    parser.add_argument('--precondition-frequency', type=int, default=100)
    parser.add_argument('--start-preconditioning-step', type=int, default=100)
    
    # Logistics
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--save-interval', type=int, default=5)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    main(args)
