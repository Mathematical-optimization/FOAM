
#!/usr/bin/env python3
"""
ViT / ImageNet-1K experiment suite for:
  A. Optimizer overhead decomposition
  B. Stronger baseline comparison (stale / FOAM / residual-trigger / SOAP)
  F. QR cold-start claim validation
  G. Dimension-scaling microbenchmark

This file upgrades the user's original monolithic ViT training script into a single
entrypoint with subcommands:
  - train
  - qr-benchmark
  - dim-scaling

Notes
-----
1) Shampoo-based baselines require the user's local `optimizers.distributed_shampoo`
   package, matching the original training environment.
2) SOAP uses the exact implementation provided by the user in `soap_exact.py`.
3) Single-GPU / single-process execution is supported. DDP is enabled automatically
   when WORLD_SIZE > 1 and LOCAL_RANK is set.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import functools
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from timm.data import Mixup, create_transform
except Exception:
    Mixup = None
    create_transform = None

try:
    import wandb
except Exception:
    wandb = None

# Local exact SOAP implementation provided by the user.
try:
    from soap_exact import SOAP
except Exception:
    SOAP = None

# Shampoo imports are optional at import-time so that qr/dim-scaling subcommands
# remain usable without the optimizer package installed.
try:
    from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
    from optimizers.distributed_shampoo.shampoo_types import (
        AdamGraftingConfig,
        CommunicationDType,
        DDPShampooConfig,
    )
    from optimizers.distributed_shampoo.utils.shampoo_preconditioner_list import (
        ShampooPreconditionerList,
    )
    from optimizers.matrix_functions import check_diagonal, matrix_inverse_root
    _HAVE_SHAMPOO = True
except Exception:
    DistributedShampoo = None
    AdamGraftingConfig = None
    CommunicationDType = None
    DDPShampooConfig = None
    ShampooPreconditionerList = None
    check_diagonal = None
    matrix_inverse_root = None
    _HAVE_SHAMPOO = False


# ---------------------------------------------------------------------------
# Timing / monitoring
# ---------------------------------------------------------------------------

@dataclass
class TimingStats:
    total_time: float = 0.0
    count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0
    epoch_time: float = 0.0
    epoch_count: int = 0

    def update(self, dt: float) -> None:
        self.total_time += dt
        self.count += 1
        self.min_time = min(self.min_time, dt)
        self.max_time = max(self.max_time, dt)
        self.epoch_time += dt
        self.epoch_count += 1

    def reset_epoch(self) -> None:
        self.epoch_time = 0.0
        self.epoch_count = 0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count else 0.0

    @property
    def epoch_avg_time(self) -> float:
        return self.epoch_time / self.epoch_count if self.epoch_count else 0.0


class ExperimentProfiler:
    def __init__(self, use_cuda_sync: bool = True):
        self.use_cuda_sync = use_cuda_sync
        self.timers: DefaultDict[str, TimingStats] = defaultdict(TimingStats)
        self.active: Dict[str, float] = {}
        self.training_start_time: Optional[float] = None
        self.current_epoch: int = -1

    def _sync(self) -> None:
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self, name: str) -> None:
        self._sync()
        self.active[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        self._sync()
        if name not in self.active:
            return 0.0
        dt = time.perf_counter() - self.active.pop(name)
        self.timers[name].update(dt)
        return dt

    @contextlib.contextmanager
    def profile(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def reset_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        for stats in self.timers.values():
            stats.reset_epoch()

    def timer_dict(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for k, v in self.timers.items():
            out[k] = {
                "total_time": v.total_time,
                "count": v.count,
                "avg_time": v.avg_time,
                "min_time": 0.0 if math.isinf(v.min_time) else v.min_time,
                "max_time": v.max_time,
                "epoch_time": v.epoch_time,
                "epoch_count": v.epoch_count,
                "epoch_avg_time": v.epoch_avg_time,
            }
        return out


class SnapshotWriter:
    def __init__(
        self,
        root: Path,
        enabled: bool,
        max_snapshots: int = 32,
        min_dim: int = 128,
        max_dim: int = 4096,
    ):
        self.root = Path(root)
        self.enabled = enabled
        self.max_snapshots = max_snapshots
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.count = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def maybe_save(
        self,
        *,
        epoch: int,
        step: int,
        factor_index: str,
        metric_name: str,
        metric_value: float,
        epsilon: float,
        factor_matrix: torch.Tensor,
        prev_q: Optional[torch.Tensor],
        prev_d: Optional[torch.Tensor],
        root: int,
        exponent_multiplier: float,
        policy: str,
    ) -> Optional[Path]:
        if not self.enabled or self.count >= self.max_snapshots:
            return None
        dim = int(factor_matrix.shape[0])
        if dim < self.min_dim or dim > self.max_dim:
            return None
        if prev_q is None or prev_d is None:
            return None
        path = self.root / f"snap_{self.count:04d}_{policy}_e{epoch:03d}_s{step:08d}_{factor_index.replace('.', '_')}.pt"
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "factor_index": factor_index,
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "epsilon": float(epsilon),
                "factor_matrix": factor_matrix.detach().cpu(),
                "prev_q": prev_q.detach().cpu(),
                "prev_d": prev_d.detach().cpu(),
                "root": int(root),
                "exponent_multiplier": float(exponent_multiplier),
                "policy": policy,
            },
            path,
        )
        self.count += 1
        return path


class ExperimentMonitor:
    def __init__(self, out_dir: Path, rank: int):
        self.out_dir = Path(out_dir)
        self.rank = rank
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_rows: List[Dict[str, Any]] = []
        self.factor_rows: List[Dict[str, Any]] = []
        self.dim_rows: List[Dict[str, Any]] = []

    def log_epoch(self, row: Dict[str, Any]) -> None:
        if self.rank == 0:
            self.epoch_rows.append(dict(row))

    def log_factor(self, row: Dict[str, Any]) -> None:
        if self.rank == 0:
            self.factor_rows.append(dict(row))

    def log_dim_timing(self, row: Dict[str, Any]) -> None:
        if self.rank == 0:
            self.dim_rows.append(dict(row))

    def _write_csv(self, path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save(self) -> None:
        if self.rank != 0:
            return
        self._write_csv(self.out_dir / "epoch_metrics.csv", self.epoch_rows)
        self._write_csv(self.out_dir / "factor_events.csv", self.factor_rows)
        self._write_csv(self.out_dir / "real_dimension_timings.csv", self.dim_rows)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def dist_rank() -> int:
    return dist.get_rank() if is_dist() else 0

def dist_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def dist_barrier() -> None:
    if is_dist():
        dist.barrier()

def reduce_scalar(value: float, device: torch.device, op=dist.ReduceOp.SUM) -> float:
    if not is_dist():
        return float(value)
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(t, op=op)
    return float(t.item())

def reduce_tensor(t: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(t, op=op)
    return t

def maybe_init_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and not args.cpu
    if world_size > 1:
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return world_size > 1, dist_rank(), dist_world_size(), device

def maybe_cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim: int = 384, num_heads: int = 6, attn_dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, need_weights: bool = False):
        bsz = query.shape[0]
        q = self.q_proj(query).reshape(bsz, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(bsz, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(bsz, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(bsz, -1, self.embedding_dim)
        return self.out_proj(out), None


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 384,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = CustomMultiheadAttention(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim=embedding_dim, mlp_dim=mlp_dim, dropout=mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nx = self.norm1(x)
        attn_out, _ = self.attn(nx, nx, nx)
        x = x + attn_out
        mx = self.norm2(x)
        x = x + self.mlp(mx)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        num_classes: int = 1000,
        embedding_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    attn_dropout=attn_dropout,
                    mlp_dropout=mlp_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.classifier_norm = nn.LayerNorm(embedding_dim)
        self.classifier_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.patch_embedding(x).flatten(2, 3).permute(0, 2, 1)
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.position_embedding
        x = self.embedding_dropout(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.classifier_norm(x)
        return self.classifier_head(x[:, 0])


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class SyntheticImageNet(Dataset):
    def __init__(self, size: int = 4096, num_classes: int = 1000, image_size: int = 224, seed: int = 42):
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        # deterministic by idx for reproducibility
        rng = np.random.default_rng(int(idx))
        img = rng.integers(0, 256, size=(3, self.image_size, self.image_size), dtype=np.uint8)
        img = torch.from_numpy(img.astype(np.float32) / 255.0)
        label = int(rng.integers(0, self.num_classes))
        return {"pixel_values": img, "label": label}


def apply_transforms(examples: Dict[str, List[Image.Image]], transform) -> Dict[str, List[torch.Tensor]]:
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "label": torch.tensor([int(x["label"]) for x in batch], dtype=torch.long),
    }


def make_dataloaders(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    distributed: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if args.synthetic_data:
        train_dataset = SyntheticImageNet(size=args.synthetic_train_samples, num_classes=args.num_classes, image_size=args.image_size, seed=args.seed)
        clean_train_dataset = SyntheticImageNet(size=args.synthetic_eval_samples, num_classes=args.num_classes, image_size=args.image_size, seed=args.seed + 1)
        val_dataset = SyntheticImageNet(size=args.synthetic_eval_samples, num_classes=args.num_classes, image_size=args.image_size, seed=args.seed + 2)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if distributed else None
        clean_sampler = DistributedSampler(clean_train_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
        clean_loader = DataLoader(clean_train_dataset, batch_size=args.eval_batch_size, sampler=clean_sampler, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, sampler=val_sampler, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
        return train_loader, clean_loader, val_loader

    if load_dataset is None or create_transform is None:
        raise RuntimeError("datasets and timm are required for ImageNet-1K training mode.")

    train_transform = create_transform(
        input_size=args.image_size,
        is_training=True,
        auto_augment=args.auto_augment,
        interpolation=args.interpolation,
    )
    val_transform = create_transform(
        input_size=args.image_size,
        is_training=False,
        interpolation=args.interpolation,
    )

    if rank == 0:
        load_dataset("imagenet-1k", cache_dir=args.data_path)
    dist_barrier()
    dataset = load_dataset("imagenet-1k", cache_dir=args.data_path)

    train_dataset = dataset["train"]
    train_dataset.set_transform(functools.partial(apply_transforms, transform=train_transform))
    clean_train_dataset = dataset["train"].with_transform(functools.partial(apply_transforms, transform=val_transform))
    val_dataset = dataset["validation"]
    val_dataset.set_transform(functools.partial(apply_transforms, transform=val_transform))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if distributed else None
    clean_sampler = DistributedSampler(clean_train_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    clean_loader = DataLoader(
        clean_train_dataset,
        batch_size=args.eval_batch_size,
        sampler=clean_sampler,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, clean_loader, val_loader


# ---------------------------------------------------------------------------
# Math helpers used by residual / QR / dimension scaling
# ---------------------------------------------------------------------------

def offdiag_residual(a: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    qt_a_q = q.T @ a @ q
    diag = torch.diag(torch.diag(qt_a_q))
    return torch.norm(qt_a_q - diag, p="fro") / (torch.norm(a, p="fro") + 1e-25)

def estimate_eigs_in_basis(a: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.diag(q.T @ a @ q)

def reconstruct_inverse_root(
    q: torch.Tensor,
    d: torch.Tensor,
    epsilon: float,
    root: int,
    exponent_multiplier: float = 1.0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    exponent = -exponent_multiplier / root
    eig_term = (d + epsilon).pow(exponent)
    out = q * eig_term.unsqueeze(0) @ q.T
    return out.to(dtype=dtype or out.dtype)

def warm_start_qr_refine(a: torch.Tensor, q0: torch.Tensor, n_iters: int) -> torch.Tensor:
    q = q0
    for _ in range(n_iters):
        q, _ = torch.linalg.qr(a @ q)
    return q

def relative_inverse_root_error(
    a: torch.Tensor,
    q_approx: torch.Tensor,
    epsilon: float,
    root: int,
    exponent_multiplier: float = 1.0,
) -> torch.Tensor:
    eigvals_true, q_true = torch.linalg.eigh(a)
    p_true = reconstruct_inverse_root(q_true, eigvals_true, epsilon, root, exponent_multiplier)
    d_est = estimate_eigs_in_basis(a, q_approx)
    p_est = reconstruct_inverse_root(q_approx, d_est, epsilon, root, exponent_multiplier)
    return torch.norm(p_est - p_true, p="fro") / (torch.norm(p_true, p="fro") + 1e-25)

def foam_style_proxy_from_basis(
    a: torch.Tensor,
    q: torch.Tensor,
    d: torch.Tensor,
    epsilon: float,
    root: int,
    exponent_multiplier: float = 1.0,
) -> torch.Tensor:
    # A self-contained proxy for microbenchmarking: diagonalization drift scaled by
    # inverse-root sensitivity. This matches the computational structure of FOAM.
    qt_a_q = q.T @ a @ q
    drift = qt_a_q - torch.diag(d)
    rc = torch.norm(drift, p="fro") / (torch.norm(torch.diag(d), p="fro") + 1e-25)
    inv_root_exponent = -exponent_multiplier / root
    h_eigs = (d + epsilon).pow(inv_root_exponent)
    alpha = h_eigs.abs().max() / (torch.norm(h_eigs, p=2) + 1e-25)
    return rc * (alpha / root)

def make_random_spd(dim: int, dtype: torch.dtype = torch.float32, device: str = "cpu", jitter: float = 1e-4) -> torch.Tensor:
    x = torch.randn(dim, dim, dtype=dtype, device=device)
    a = (x @ x.T) / max(dim, 1)
    a = a + jitter * torch.eye(dim, dtype=dtype, device=device)
    return a

def make_stale_basis_from_spd(a: torch.Tensor, noise_scale: float = 0.02) -> Tuple[torch.Tensor, torch.Tensor]:
    d, q = torch.linalg.eigh(a)
    noise = noise_scale * torch.randn_like(q)
    q_noisy, _ = torch.linalg.qr(q + noise)
    d_noisy = torch.clamp(d + noise_scale * d.abs().mean() * torch.randn_like(d), min=1e-12)
    return q_noisy, d_noisy


# ---------------------------------------------------------------------------
# Optimizer helpers / monkey-patching
# ---------------------------------------------------------------------------

def get_warmup_cosine_decay_lr(current_step: int, base_lr: float, num_steps: int, warmup_steps: int) -> float:
    if current_step < warmup_steps:
        return base_lr * (current_step / max(warmup_steps, 1))
    progress = (current_step - warmup_steps) / max(num_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * cosine_decay

class EighCounter:
    def __init__(self):
        self.total_count = 0
        self.epoch_count = 0
        self.original = torch.linalg.eigh

    def wrap(self, a, UPLO='L', *, out=None):
        self.total_count += 1
        self.epoch_count += 1
        return self.original(a, UPLO=UPLO, out=out)

    def reset_epoch(self):
        self.epoch_count = 0

def install_global_linalg_wrappers(profiler: ExperimentProfiler, eigh_counter: EighCounter):
    orig_eigh = torch.linalg.eigh
    orig_qr = torch.linalg.qr

    def timed_eigh(a, UPLO='L', *, out=None):
        with profiler.profile("eigendecomposition"):
            return eigh_counter.wrap(a, UPLO=UPLO, out=out)

    def timed_qr(a, mode='reduced'):
        with profiler.profile("qr_refine"):
            return orig_qr(a, mode=mode)

    torch.linalg.eigh = timed_eigh
    torch.linalg.qr = timed_qr
    return orig_eigh, orig_qr

def restore_global_linalg_wrappers(orig_eigh, orig_qr):
    torch.linalg.eigh = orig_eigh
    torch.linalg.qr = orig_qr

def wrap_soap_optimizer(optimizer: Any, profiler: ExperimentProfiler) -> None:
    if SOAP is None or not isinstance(optimizer, SOAP):
        return

    orig_step = optimizer.step

    def wrapped_step(closure=None):
        with profiler.profile("optimizer_step_total"):
            return orig_step(closure)

    optimizer.step = wrapped_step

def gather_optimizer_state_from_all_ranks(optimizer, model, rank: int, world_size: int):
    if not is_dist() or world_size <= 1:
        if hasattr(optimizer, "distributed_state_dict"):
            return optimizer.distributed_state_dict(key_to_param=model.module.named_parameters() if isinstance(model, DDP) else model.named_parameters())
        return optimizer.state_dict()

    if not hasattr(optimizer, "distributed_state_dict"):
        return optimizer.state_dict() if rank == 0 else None

    local_state = optimizer.distributed_state_dict(
        key_to_param=model.module.named_parameters() if isinstance(model, DDP) else model.named_parameters()
    )
    all_states = [None] * world_size
    dist.all_gather_object(all_states, local_state)
    if rank != 0:
        return None

    merged_state = {"state": {}, "param_groups": all_states[0].get("param_groups", [])}
    for param_key in all_states[0]["state"].keys():
        merged_state["state"][param_key] = {}
        param_state_keys = set()
        for state in all_states:
            if "state" in state and param_key in state["state"]:
                param_state_keys.update(state["state"][param_key].keys())

        for state_key in param_state_keys:
            merged_value = None
            is_factor_matrix = "factor_matrices" in str(state_key)
            for state in all_states:
                if "state" in state and param_key in state["state"] and state_key in state["state"][param_key]:
                    value = state["state"][param_key][state_key]
                    if isinstance(value, torch.Tensor):
                        if hasattr(value, "_local_tensor"):
                            value = value._local_tensor
                        if is_factor_matrix:
                            if value.numel() > 0:
                                if merged_value is None or merged_value.numel() == 0:
                                    merged_value = value.clone()
                        else:
                            if merged_value is None or (merged_value.numel() == 0 and value.numel() > 0):
                                merged_value = value.clone()
                    else:
                        if merged_value is None:
                            merged_value = value
            if merged_value is not None:
                merged_state["state"][param_key][state_key] = merged_value
    return merged_state


def install_shampoo_policy_patch(
    *,
    args: argparse.Namespace,
    profiler: ExperimentProfiler,
    monitor: ExperimentMonitor,
    snapshot_writer: SnapshotWriter,
    epoch_ref: Dict[str, int],
    step_ref: Dict[str, int],
    rank: int,
    eigh_counter: EighCounter,
):
    if not _HAVE_SHAMPOO:
        raise RuntimeError("Shampoo optimizer package is required for stale/foam/residual baselines.")

    orig_compute_root_inverse = ShampooPreconditionerList.compute_root_inverse
    orig_single = ShampooPreconditionerList._compute_single_root_inverse

    def timed_compute_root_inverse(self):
        with profiler.profile("compute_root_inverse_total"):
            return orig_compute_root_inverse(self)

    def patched_compute_single_root_inverse(
        self,
        factor_matrix,
        inv_factor_matrix,
        is_factor_matrix_diagonal,
        factor_matrix_index,
        root,
        epsilon_value,
        kronecker_factors,
        factor_idx,
    ):
        start_eigh_count = eigh_counter.total_count
        epoch = int(epoch_ref["epoch"])
        step = int(step_ref["step"])

        bias_corrected = factor_matrix / self._bias_correction2
        prev_q = kronecker_factors.eigenvectors[factor_idx]
        prev_d = kronecker_factors.eigenvalues[factor_idx]
        current_epsilon = kronecker_factors.adaptive_epsilons[factor_idx]
        if current_epsilon is None:
            current_epsilon = epsilon_value

        dim = int(bias_corrected.shape[0])
        metric_name = ""
        metric_value = None
        action = "refresh_eigh"
        should_recompute = True

        def stale_reuse_with_basis(new_eps: float):
            alpha_pow = -self._exponent_multiplier / root
            eig_term = (prev_d + new_eps).pow(alpha_pow)
            computed = prev_q * eig_term.unsqueeze(0) @ prev_q.T
            computed = computed.to(dtype=inv_factor_matrix.dtype)
            inv_factor_matrix.copy_(computed)
            kronecker_factors.adaptive_epsilons[factor_idx] = float(new_eps)

        proxy_time = 0.0
        residual_time = 0.0

        if args.optimizer in {"foam", "residual"} and prev_q is not None and prev_d is not None:
            if args.optimizer == "foam":
                t0 = time.perf_counter()
                try:
                    rc_t = self._compute_relative_condition_number(bias_corrected, prev_q, prev_d, current_epsilon)
                    inv_root_exponent = -self._exponent_multiplier / root
                    h_eigs = (prev_d + current_epsilon).pow(inv_root_exponent)
                    alpha = h_eigs.abs().max() / (torch.norm(h_eigs, p=2) + 1e-25)
                    proxy = rc_t * (alpha / root)
                    proxy_time = time.perf_counter() - t0
                    profiler.timers["proxy_compute"].update(proxy_time)
                    metric_name = "foam_proxy"
                    metric_value = float(proxy.item())
                    monitor.log_dim_timing(
                        {
                            "epoch": epoch,
                            "step": step,
                            "dim": dim,
                            "policy": args.optimizer,
                            "op_name": "proxy_compute",
                            "seconds": proxy_time,
                            "source": "train",
                            "factor_index": factor_matrix_index,
                        }
                    )
                    new_epsilon = current_epsilon * (float(proxy.item()) / max(args.matrix_root_inv_threshold, 1e-25))
                    if float(proxy.item()) >= args.matrix_root_inv_threshold:
                        if new_epsilon < args.max_epsilon:
                            current_epsilon = float(new_epsilon)
                            should_recompute = False
                            action = "reuse_stale_adaptive_eps"
                            stale_reuse_with_basis(current_epsilon)
                        else:
                            current_epsilon = epsilon_value
                            should_recompute = True
                            action = "refresh_eigh_epsmax"
                    else:
                        current_epsilon = max(float(new_epsilon), float(epsilon_value))
                        should_recompute = False
                        action = "reuse_stale_low_proxy"
                        stale_reuse_with_basis(current_epsilon)
                except Exception as exc:
                    metric_name = "foam_proxy_error"
                    metric_value = math.nan
                    should_recompute = True
                    action = f"refresh_eigh_exception"
            else:
                t0 = time.perf_counter()
                try:
                    res_t = offdiag_residual(bias_corrected, prev_q)
                    residual_time = time.perf_counter() - t0
                    profiler.timers["proxy_compute"].update(residual_time)
                    metric_name = "diag_residual"
                    metric_value = float(res_t.item())
                    monitor.log_dim_timing(
                        {
                            "epoch": epoch,
                            "step": step,
                            "dim": dim,
                            "policy": args.optimizer,
                            "op_name": "residual_compute",
                            "seconds": residual_time,
                            "source": "train",
                            "factor_index": factor_matrix_index,
                        }
                    )
                    if float(res_t.item()) < args.residual_threshold:
                        should_recompute = False
                        action = "reuse_stale_low_residual"
                        stale_reuse_with_basis(current_epsilon)
                    else:
                        should_recompute = True
                        action = "refresh_eigh_triggered"
                        if rank == 0:
                            snapshot_writer.maybe_save(
                                epoch=epoch,
                                step=step,
                                factor_index=factor_matrix_index,
                                metric_name=metric_name,
                                metric_value=float(res_t.item()),
                                epsilon=float(current_epsilon),
                                factor_matrix=bias_corrected,
                                prev_q=prev_q,
                                prev_d=prev_d,
                                root=root,
                                exponent_multiplier=self._exponent_multiplier,
                                policy=args.optimizer,
                            )
                except Exception:
                    metric_name = "diag_residual_error"
                    metric_value = math.nan
                    should_recompute = True
                    action = "refresh_eigh_exception"

        elif args.optimizer == "stale":
            should_recompute = True
            action = "refresh_eigh_fixed_schedule"

        if should_recompute:
            local_t0 = time.perf_counter()
            if is_factor_matrix_diagonal is not None and is_factor_matrix_diagonal is not False and check_diagonal is not None:
                try:
                    if bool(is_factor_matrix_diagonal) and not check_diagonal(factor_matrix):
                        is_factor_matrix_diagonal.copy_(torch.tensor(False, device=factor_matrix.device))
                except Exception:
                    pass
            try:
                result = matrix_inverse_root(
                    A=bias_corrected,
                    root=root,
                    epsilon=current_epsilon,
                    exponent_multiplier=self._exponent_multiplier,
                    is_diagonal=is_factor_matrix_diagonal,
                    retry_double_precision=self._use_protected_eigh,
                )
                computed_inv, used_epsilon, L, Q = result
                if L is not None and Q is not None:
                    raw_eigenvalues = L - used_epsilon
                    kronecker_factors.eigenvalues[factor_idx] = raw_eigenvalues.to(dtype=factor_matrix.dtype)
                    kronecker_factors.eigenvectors[factor_idx] = Q.to(dtype=factor_matrix.dtype)
                    kronecker_factors.adaptive_epsilons[factor_idx] = float(current_epsilon)
                inv_factor_matrix.copy_(computed_inv.to(dtype=inv_factor_matrix.dtype))
            except Exception:
                action = "matrix_inverse_root_failed"
            local_dt = time.perf_counter() - local_t0
            monitor.log_dim_timing(
                {
                    "epoch": epoch,
                    "step": step,
                    "dim": dim,
                    "policy": args.optimizer,
                    "op_name": "matrix_inverse_root_total",
                    "seconds": local_dt,
                    "source": "train",
                    "factor_index": factor_matrix_index,
                }
            )
        else:
            # stale reuse reconstruction time for real-dimension logs
            # use a tiny timing around the already-done reconstruction if metric branch used it
            pass

        end_eigh_count = eigh_counter.total_count
        performed_eigh = int(end_eigh_count > start_eigh_count)
        monitor.log_factor(
            {
                "epoch": epoch,
                "step": step,
                "policy": args.optimizer,
                "factor_index": factor_matrix_index,
                "dim": dim,
                "action": action,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "epsilon": float(current_epsilon),
                "performed_eigh": performed_eigh,
            }
        )

    ShampooPreconditionerList.compute_root_inverse = timed_compute_root_inverse
    ShampooPreconditionerList._compute_single_root_inverse = patched_compute_single_root_inverse

    return orig_compute_root_inverse, orig_single

def restore_shampoo_patch(orig_compute_root_inverse, orig_single):
    if _HAVE_SHAMPOO:
        ShampooPreconditionerList.compute_root_inverse = orig_compute_root_inverse
        ShampooPreconditionerList._compute_single_root_inverse = orig_single


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def set_seed(seed: int, rank: int = 0):
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(full_seed)
        torch.cuda.manual_seed_all(full_seed)
    os.environ["PYTHONHASHSEED"] = str(full_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def maybe_build_wandb(args, rank: int):
    if rank != 0 or args.no_wandb or wandb is None:
        return None
    return wandb.init(project=args.project, entity=args.entity, dir=str(args.out_dir), config=vars(args))

def validate(model, loader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += float(loss.item())
            pred = outputs.argmax(dim=1)
            total += int(labels.size(0))
            correct += int((pred == labels).sum().item())

    if is_dist():
        loss_t = torch.tensor(loss_sum, device=device, dtype=torch.float64)
        total_t = torch.tensor(total, device=device, dtype=torch.float64)
        correct_t = torch.tensor(correct, device=device, dtype=torch.float64)
        dist.all_reduce(loss_t)
        dist.all_reduce(total_t)
        dist.all_reduce(correct_t)
        loss_sum = float(loss_t.item())
        total = int(total_t.item())
        correct = int(correct_t.item())

    avg_loss = loss_sum / max(len(loader) * dist_world_size(), 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc

def validate_on_trainset(model, loader, criterion, device: torch.device) -> float:
    model.eval()
    running = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running += float(loss.item())
            batches += 1
    if is_dist():
        loss_t = torch.tensor(running, device=device, dtype=torch.float64)
        batches_t = torch.tensor(batches, device=device, dtype=torch.float64)
        dist.all_reduce(loss_t)
        dist.all_reduce(batches_t)
        running = float(loss_t.item())
        batches = int(batches_t.item())
    return running / max(batches, 1)

def build_model(args: argparse.Namespace) -> nn.Module:
    return VisionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        num_classes=args.num_classes,
        attn_dropout=args.attn_dropout,
        mlp_dropout=args.mlp_dropout,
        embedding_dropout=args.embedding_dropout,
    )

def build_optimizer(
    args: argparse.Namespace,
    model: nn.Module,
    world_size: int,
) -> Any:
    params = model.parameters()
    if args.optimizer == "soap":
        if SOAP is None:
            raise RuntimeError("SOAP implementation is not available.")
        return SOAP(
            params=params,
            lr=args.base_lr,
            betas=(args.beta1, args.beta2),
            eps=args.grafting_epsilon,
            weight_decay=args.weight_decay,
            precondition_frequency=args.precondition_frequency,
            max_precond_dim=args.max_preconditioner_dim,
            merge_dims=False,
            precondition_1d=False,
            normalize_grads=False,
            data_format="channels_first",
            correct_bias=True,
        )

    if args.optimizer in {"foam", "stale", "residual"}:
        if not _HAVE_SHAMPOO:
            raise RuntimeError("DistributedShampoo package is required for foam/stale/residual baselines.")
        distributed_config = None
        if world_size > 1:
            distributed_config = DDPShampooConfig(
                communication_dtype=CommunicationDType.FP32,
                num_trainers_per_group=world_size,
                communicate_params=False,
            )
        optimizer = DistributedShampoo(
            params=params,
            lr=args.base_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            epsilon=args.epsilon,
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
            max_epsilon=args.max_epsilon,
        )
        return optimizer

    raise ValueError(f"Unsupported optimizer: {args.optimizer}")

def save_overhead_summary(out_dir: Path, profiler: ExperimentProfiler, rank: int) -> None:
    if rank != 0:
        return
    raw = profiler.timer_dict()
    total_train = time.perf_counter() - profiler.training_start_time if profiler.training_start_time else 0.0
    optimizer_total = raw.get("optimizer_step_total", {}).get("total_time", 0.0)
    summary = {
        "total_training_time_sec": total_train,
        "optimizer_step_total_sec": optimizer_total,
        "eigendecomposition_sec": raw.get("eigendecomposition", {}).get("total_time", 0.0),
        "compute_root_inverse_total_sec": raw.get("compute_root_inverse_total", {}).get("total_time", 0.0),
        "proxy_compute_sec": raw.get("proxy_compute", {}).get("total_time", 0.0),
        "qr_refine_sec": raw.get("qr_refine", {}).get("total_time", 0.0),
        "forward_sec": raw.get("forward", {}).get("total_time", 0.0),
        "backward_sec": raw.get("backward", {}).get("total_time", 0.0),
        "validation_sec": raw.get("validation", {}).get("total_time", 0.0),
        "optimizer_overhead_ratio_vs_total": optimizer_total / max(total_train, 1e-12),
    }
    with (out_dir / "overhead_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "timers.json").open("w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)

    if plt is not None:
        keys = [
            ("eigendecomposition", "EVD"),
            ("compute_root_inverse_total", "Root inverse"),
            ("proxy_compute", "Proxy/residual"),
            ("qr_refine", "QR"),
        ]
        labels = []
        vals = []
        for k, label in keys:
            vals.append(raw.get(k, {}).get("total_time", 0.0))
            labels.append(label)
        plt.figure(figsize=(8, 4))
        plt.bar(labels, vals)
        plt.ylabel("Seconds")
        plt.title("Optimizer overhead decomposition")
        plt.tight_layout()
        plt.savefig(out_dir / "overhead_breakdown.png")
        plt.close()

def train_command(args: argparse.Namespace) -> None:
    distributed, rank, world_size, device = maybe_init_distributed(args)
    set_seed(args.seed, rank)
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    profiler = ExperimentProfiler(use_cuda_sync=torch.cuda.is_available() and not args.cpu)
    profiler.training_start_time = time.perf_counter()
    monitor = ExperimentMonitor(args.out_dir, rank=rank)
    snapshot_writer = SnapshotWriter(
        args.out_dir / "factor_snapshots",
        enabled=args.save_factor_snapshots,
        max_snapshots=args.max_snapshots,
        min_dim=args.snapshot_min_dim,
        max_dim=args.snapshot_max_dim,
    )

    wandb_run = maybe_build_wandb(args, rank)
    train_loader, clean_loader, val_loader = make_dataloaders(args, rank, world_size, distributed)

    model = build_model(args).to(device)
    if distributed:
        ddp_device_ids = [device.index] if device.type == "cuda" else None
        model = DDP(model, device_ids=ddp_device_ids)
    criterion = nn.CrossEntropyLoss().to(device)

    mixup_fn = None
    if not args.synthetic_data and (args.mixup > 0 or args.label_smoothing > 0):
        if Mixup is None:
            raise RuntimeError("timm is required when mixup/label smoothing is enabled.")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=0.0,
            label_smoothing=args.label_smoothing,
            num_classes=args.num_classes,
        )

    optimizer = build_optimizer(args, model, world_size)

    # Global linear algebra timers capture EVD / QR for both Shampoo and SOAP.
    eigh_counter = EighCounter()
    orig_eigh, orig_qr = install_global_linalg_wrappers(profiler, eigh_counter)

    orig_compute_root_inverse = None
    orig_single = None
    epoch_ref = {"epoch": 0}
    step_ref = {"step": 0}

    if args.optimizer in {"foam", "stale", "residual"}:
        orig_compute_root_inverse, orig_single = install_shampoo_policy_patch(
            args=args,
            profiler=profiler,
            monitor=monitor,
            snapshot_writer=snapshot_writer,
            epoch_ref=epoch_ref,
            step_ref=step_ref,
            rank=rank,
            eigh_counter=eigh_counter,
        )
    elif args.optimizer == "soap":
        wrap_soap_optimizer(optimizer, profiler)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        target_model = model.module if isinstance(model, DDP) else model
        target_model.load_state_dict(ckpt["model_state_dict"])
        if hasattr(optimizer, "load_distributed_state_dict") and "optimizer_state_dict" in ckpt:
            optimizer.load_distributed_state_dict(
                ckpt["optimizer_state_dict"],
                key_to_param=target_model.named_parameters(),
            )
        elif "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1

    total_steps = len(train_loader) * args.epochs
    best_acc = -float("inf")

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_ref["epoch"] = epoch
            profiler.reset_epoch(epoch)
            eigh_counter.reset_epoch()
            epoch_start = time.perf_counter()

            if distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            last_avg_loss = 0.0

            for i, batch in enumerate(train_loader):
                global_step = epoch * len(train_loader) + i
                step_ref["step"] = global_step

                images = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                if mixup_fn is not None:
                    images, labels = mixup_fn(images, labels)

                lr = get_warmup_cosine_decay_lr(global_step, args.base_lr, total_steps, args.warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                if hasattr(optimizer, "zero_grad"):
                    optimizer.zero_grad(set_to_none=True)

                with profiler.profile("forward"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                with profiler.profile("backward"):
                    loss.backward()

                if args.optimizer != "soap":
                    with profiler.profile("optimizer_step_total"):
                        optimizer.step()
                else:
                    optimizer.step()

                loss_val = float(loss.detach().item())
                if is_dist():
                    t = torch.tensor(loss_val, device=device, dtype=torch.float64)
                    dist.all_reduce(t)
                    loss_val = float(t.item()) / world_size
                last_avg_loss = loss_val

                if rank == 0 and (i + 1) % args.log_interval == 0:
                    print(
                        f"[Epoch {epoch+1}/{args.epochs}] "
                        f"step {i+1}/{len(train_loader)} "
                        f"loss={loss_val:.4f} lr={lr:.3e}"
                    )

                if args.max_steps > 0 and global_step + 1 >= args.max_steps:
                    break

            epoch_time = time.perf_counter() - epoch_start

            with profiler.profile("validation"):
                val_loss, val_acc = validate(model, val_loader, criterion, device)

            clean_train_loss = None
            if args.clean_train_eval and (epoch + 1) >= args.clean_train_eval_start_epoch:
                with profiler.profile("validation"):
                    clean_train_loss = validate_on_trainset(model, clean_loader, criterion, device)

            raw_timers = profiler.timer_dict()
            epoch_row = {
                "epoch": epoch + 1,
                "optimizer": args.optimizer,
                "train_iteration_loss": last_avg_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "clean_train_loss": clean_train_loss if clean_train_loss is not None else "",
                "epoch_time_sec": epoch_time,
                "learning_rate": lr,
                "epoch_eigh_count": eigh_counter.epoch_count,
                "eigendecomposition_sec_epoch": raw_timers.get("eigendecomposition", {}).get("epoch_time", 0.0),
                "compute_root_inverse_total_sec_epoch": raw_timers.get("compute_root_inverse_total", {}).get("epoch_time", 0.0),
                "proxy_compute_sec_epoch": raw_timers.get("proxy_compute", {}).get("epoch_time", 0.0),
                "qr_refine_sec_epoch": raw_timers.get("qr_refine", {}).get("epoch_time", 0.0),
                "forward_sec_epoch": raw_timers.get("forward", {}).get("epoch_time", 0.0),
                "backward_sec_epoch": raw_timers.get("backward", {}).get("epoch_time", 0.0),
                "optimizer_step_total_sec_epoch": raw_timers.get("optimizer_step_total", {}).get("epoch_time", 0.0),
            }
            monitor.log_epoch(epoch_row)

            if rank == 0:
                msg = (
                    f"Epoch {epoch+1}: val_acc={val_acc:.2f}% "
                    f"val_loss={val_loss:.4f} time={epoch_time:.2f}s "
                    f"eigh_epoch={eigh_counter.epoch_count}"
                )
                if clean_train_loss is not None:
                    msg += f" clean_train_loss={clean_train_loss:.4f}"
                print(msg)

                if wandb_run is not None:
                    wb = dict(epoch_row)
                    if clean_train_loss is None:
                        wb.pop("clean_train_loss", None)
                    wandb.log(wb)

            # DDP-safe checkpointing: every rank must participate in optimizer-state
            # gathering for Shampoo-family optimizers because it uses all_gather_object.
            # Only the actual file write is restricted to rank 0.
            save_best = val_acc > best_acc
            if save_best:
                best_acc = val_acc
                if args.optimizer == "soap":
                    opt_state = optimizer.state_dict()
                else:
                    opt_state = gather_optimizer_state_from_all_ranks(optimizer, model, rank, world_size)
                if rank == 0:
                    target_model = model.module if isinstance(model, DDP) else model
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": target_model.state_dict(),
                            "optimizer_state_dict": opt_state,
                            "val_acc": val_acc,
                            "args": vars(args),
                        },
                        args.out_dir / "best.pt",
                    )

            save_periodic = args.save_interval > 0 and (epoch + 1) % args.save_interval == 0
            if save_periodic:
                if args.optimizer == "soap":
                    opt_state = optimizer.state_dict()
                else:
                    opt_state = gather_optimizer_state_from_all_ranks(optimizer, model, rank, world_size)
                if rank == 0:
                    target_model = model.module if isinstance(model, DDP) else model
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": target_model.state_dict(),
                            "optimizer_state_dict": opt_state,
                            "val_acc": val_acc,
                            "args": vars(args),
                        },
                        args.out_dir / f"epoch_{epoch+1}.pt",
                    )

            if args.max_steps > 0 and (epoch + 1) * len(train_loader) >= args.max_steps:
                break

        monitor.save()
        save_overhead_summary(args.out_dir, profiler, rank)

        if rank == 0:
            summary = {
                "optimizer": args.optimizer,
                "best_val_acc": best_acc,
                "snapshot_count": snapshot_writer.count,
                "world_size": world_size,
                "device": str(device),
                "output_dir": str(args.out_dir),
            }
            with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    finally:
        if orig_compute_root_inverse is not None:
            restore_shampoo_patch(orig_compute_root_inverse, orig_single)
        restore_global_linalg_wrappers(orig_eigh, orig_qr)
        if wandb_run is not None:
            wandb.finish()
        maybe_cleanup_distributed()


# ---------------------------------------------------------------------------
# QR benchmark (Experiment F)
# ---------------------------------------------------------------------------

def qr_benchmark_command(args: argparse.Namespace) -> None:
    snapshot_dir = Path(args.snapshot_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(snapshot_dir.glob("*.pt"))
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"No snapshot files found in {snapshot_dir}")

    rows = []
    for path in paths:
        item = torch.load(path, map_location="cpu")
        a = item["factor_matrix"].float()
        q0 = item["prev_q"].float()
        eps = float(item["epsilon"])
        root = int(item["root"])
        exponent_multiplier = float(item.get("exponent_multiplier", 1.0))
        dim = int(a.shape[0])

        # baseline metrics at stale basis
        stale_res = float(offdiag_residual(a, q0).item())

        t0 = time.perf_counter()
        d_true, q_true = torch.linalg.eigh(a)
        eigh_time = time.perf_counter() - t0

        for n_iter in args.qr_iters:
            t1 = time.perf_counter()
            q_refined = warm_start_qr_refine(a, q0, n_iter)
            qr_time = time.perf_counter() - t1
            res_after = float(offdiag_residual(a, q_refined).item())
            rel_err = float(relative_inverse_root_error(a, q_refined, eps, root, exponent_multiplier).item())
            rows.append(
                {
                    "snapshot": path.name,
                    "dim": dim,
                    "epoch": item["epoch"],
                    "step": item["step"],
                    "metric_name": item.get("metric_name", ""),
                    "metric_value": item.get("metric_value", ""),
                    "stale_residual": stale_res,
                    "method": f"warm_start_qr_{n_iter}",
                    "n_iters": int(n_iter),
                    "runtime_sec": qr_time,
                    "runtime_over_eigh": qr_time / max(eigh_time, 1e-12),
                    "residual_after": res_after,
                    "relative_inverse_root_error": rel_err,
                }
            )

        rows.append(
            {
                "snapshot": path.name,
                "dim": dim,
                "epoch": item["epoch"],
                "step": item["step"],
                "metric_name": item.get("metric_name", ""),
                "metric_value": item.get("metric_value", ""),
                "stale_residual": stale_res,
                "method": "direct_eigh",
                "n_iters": 0,
                "runtime_sec": eigh_time,
                "runtime_over_eigh": 1.0,
                "residual_after": 0.0,
                "relative_inverse_root_error": 0.0,
            }
        )

    csv_path = out_dir / "qr_benchmark.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    by_method: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    for method, group in by_method.items():
        summary[method] = {
            "median_runtime_sec": float(np.median([g["runtime_sec"] for g in group])),
            "median_runtime_over_eigh": float(np.median([g["runtime_over_eigh"] for g in group])),
            "median_residual_after": float(np.median([g["residual_after"] for g in group])),
            "median_relative_inverse_root_error": float(np.median([g["relative_inverse_root_error"] for g in group])),
            "count": len(group),
        }

    with (out_dir / "qr_benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if plt is not None:
        methods = [f"warm_start_qr_{i}" for i in args.qr_iters] + ["direct_eigh"]
        medians = [summary[m]["median_runtime_over_eigh"] for m in methods if m in summary]
        plt.figure(figsize=(8, 4))
        plt.bar([m for m in methods if m in summary], medians)
        plt.ylabel("Median runtime / direct eigh")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(out_dir / "qr_runtime_ratio.png")
        plt.close()


# ---------------------------------------------------------------------------
# Dimension-scaling microbenchmark (Experiment G)
# ---------------------------------------------------------------------------

def dim_scaling_command(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    def benchmark_single(a: torch.Tensor, q_stale: torch.Tensor, d_stale: torch.Tensor, epsilon: float, dim: int, source: str):
        # proxy
        t0 = time.perf_counter()
        proxy = float(foam_style_proxy_from_basis(a, q_stale, d_stale, epsilon, root=args.root, exponent_multiplier=args.exponent_multiplier).item())
        proxy_dt = time.perf_counter() - t0

        # residual
        t0 = time.perf_counter()
        residual = float(offdiag_residual(a, q_stale).item())
        residual_dt = time.perf_counter() - t0

        # root reconstruction
        t0 = time.perf_counter()
        _ = reconstruct_inverse_root(q_stale, d_stale, epsilon, args.root, args.exponent_multiplier)
        root_dt = time.perf_counter() - t0

        # direct eigh
        t0 = time.perf_counter()
        _ = torch.linalg.eigh(a)
        eigh_dt = time.perf_counter() - t0

        rows.append(
            {
                "source": source,
                "dim": dim,
                "proxy_value": proxy,
                "residual_value": residual,
                "proxy_sec": proxy_dt,
                "residual_sec": residual_dt,
                "root_reconstruct_sec": root_dt,
                "eigh_sec": eigh_dt,
                "qr_iters": 0,
                "qr_sec": 0.0,
            }
        )

        for n_iter in args.qr_iters:
            t0 = time.perf_counter()
            _ = warm_start_qr_refine(a, q_stale, n_iter)
            qr_dt = time.perf_counter() - t0
            rows.append(
                {
                    "source": source,
                    "dim": dim,
                    "proxy_value": proxy,
                    "residual_value": residual,
                    "proxy_sec": proxy_dt,
                    "residual_sec": residual_dt,
                    "root_reconstruct_sec": root_dt,
                    "eigh_sec": eigh_dt,
                    "qr_iters": int(n_iter),
                    "qr_sec": qr_dt,
                }
            )

    if args.mode == "synthetic":
        for dim in args.dims:
            for _ in range(args.repeats):
                a = make_random_spd(dim, dtype=torch.float32, device="cpu", jitter=args.jitter)
                q_stale, d_stale = make_stale_basis_from_spd(a, noise_scale=args.noise_scale)
                benchmark_single(a, q_stale, d_stale, args.epsilon, dim, "synthetic")

    elif args.mode == "snapshots":
        snap_dir = Path(args.snapshot_dir)
        paths = sorted(snap_dir.glob("*.pt"))
        if args.max_files > 0:
            paths = paths[: args.max_files]
        if not paths:
            raise FileNotFoundError(f"No snapshot files found in {snap_dir}")
        for path in paths:
            item = torch.load(path, map_location="cpu")
            a = item["factor_matrix"].float()
            q_stale = item["prev_q"].float()
            d_stale = item["prev_d"].float()
            benchmark_single(a, q_stale, d_stale, float(item["epsilon"]), int(a.shape[0]), "snapshot")
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    csv_path = out_dir / "dimension_scaling.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate medians per dim for plotting.
    agg: Dict[int, Dict[str, float]] = {}
    dims = sorted({int(r["dim"]) for r in rows})
    for dim in dims:
        sub = [r for r in rows if int(r["dim"]) == dim and int(r["qr_iters"]) == 0]
        agg[dim] = {
            "proxy_sec": float(np.median([r["proxy_sec"] for r in sub])),
            "residual_sec": float(np.median([r["residual_sec"] for r in sub])),
            "root_reconstruct_sec": float(np.median([r["root_reconstruct_sec"] for r in sub])),
            "eigh_sec": float(np.median([r["eigh_sec"] for r in sub])),
        }
    with (out_dir / "dimension_scaling_summary.json").open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    if plt is not None and agg:
        xs = sorted(agg.keys())
        plt.figure(figsize=(8, 5))
        plt.plot(xs, [agg[x]["proxy_sec"] for x in xs], marker="o", label="proxy")
        plt.plot(xs, [agg[x]["residual_sec"] for x in xs], marker="o", label="diag residual")
        plt.plot(xs, [agg[x]["root_reconstruct_sec"] for x in xs], marker="o", label="root reconstruct")
        plt.plot(xs, [agg[x]["eigh_sec"] for x in xs], marker="o", label="direct eigh")
        plt.xlabel("Matrix dimension")
        plt.ylabel("Median seconds")
        plt.title(f"Dimension-scaling benchmark ({args.mode})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "dimension_scaling.png")
        plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ViT/ImageNet experiment suite for FOAM baselines")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train ViT and log overhead / stronger baselines")
    train.add_argument("--data-path", type=str, default="./data")
    train.add_argument("--out-dir", type=str, default="./runs/vit")
    train.add_argument("--resume", type=str, default="")
    train.add_argument("--cpu", action="store_true")
    train.add_argument("--synthetic-data", action="store_true")
    train.add_argument("--synthetic-train-samples", type=int, default=4096)
    train.add_argument("--synthetic-eval-samples", type=int, default=1024)

    train.add_argument("--epochs", type=int, default=90)
    train.add_argument("--max-steps", type=int, default=-1)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--eval-batch-size", type=int, default=256)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--base-lr", type=float, default=1e-3)
    train.add_argument("--warmup-steps", type=int, default=10000)
    train.add_argument("--weight-decay", type=float, default=0.05)
    train.add_argument("--beta1", type=float, default=0.95)
    train.add_argument("--beta2", type=float, default=0.995)
    train.add_argument("--adam-grafting-beta2", type=float, default=0.995)
    train.add_argument("--grafting-epsilon", type=float, default=1e-9)
    train.add_argument("--epsilon", type=float, default=1e-9)

    train.add_argument("--optimizer", choices=["stale", "foam", "residual", "soap"], default="foam")
    train.add_argument("--matrix-root-inv-threshold", type=float, default=0.5, help="FOAM proxy threshold")
    train.add_argument("--max-epsilon", type=float, default=5e-7, help="FOAM epsilon cap")
    train.add_argument("--residual-threshold", type=float, default=0.05, help="Residual trigger threshold")
    train.add_argument("--precondition-frequency", type=int, default=20)
    train.add_argument("--start-preconditioning-step", type=int, default=20)
    train.add_argument("--max-preconditioner-dim", type=int, default=1024)

    train.add_argument("--save-factor-snapshots", action="store_true")
    train.add_argument("--max-snapshots", type=int, default=32)
    train.add_argument("--snapshot-min-dim", type=int, default=128)
    train.add_argument("--snapshot-max-dim", type=int, default=4096)

    train.add_argument("--mixup", type=float, default=0.2)
    train.add_argument("--label-smoothing", type=float, default=0.1)
    train.add_argument("--auto-augment", type=str, default="rand-m15-n2-mstd0.5")
    train.add_argument("--interpolation", type=str, default="bicubic")

    train.add_argument("--image-size", type=int, default=224)
    train.add_argument("--patch-size", type=int, default=16)
    train.add_argument("--embedding-dim", type=int, default=384)
    train.add_argument("--depth", type=int, default=12)
    train.add_argument("--num-heads", type=int, default=6)
    train.add_argument("--mlp-dim", type=int, default=1536)
    train.add_argument("--attn-dropout", type=float, default=0.0)
    train.add_argument("--mlp-dropout", type=float, default=0.1)
    train.add_argument("--embedding-dropout", type=float, default=0.1)
    train.add_argument("--num-classes", type=int, default=1000)

    train.add_argument("--clean-train-eval", action="store_true")
    train.add_argument("--clean-train-eval-start-epoch", type=int, default=81)
    train.add_argument("--log-interval", type=int, default=30)
    train.add_argument("--save-interval", type=int, default=45)
    train.add_argument("--seed", type=int, default=42)

    train.add_argument("--project", type=str, default="ViT_FOAM_Experiments")
    train.add_argument("--entity", type=str, default="")
    train.add_argument("--no-wandb", action="store_true")

    qr = sub.add_parser("qr-benchmark", help="Offline QR cold-start benchmark from saved factor snapshots")
    qr.add_argument("--snapshot-dir", type=str, required=True)
    qr.add_argument("--out-dir", type=str, required=True)
    qr.add_argument("--max-files", type=int, default=-1)
    qr.add_argument("--qr-iters", nargs="+", type=int, default=[1, 2, 4, 8, 16])

    dim = sub.add_parser("dim-scaling", help="Dimension-scaling microbenchmark")
    dim.add_argument("--mode", choices=["synthetic", "snapshots"], default="synthetic")
    dim.add_argument("--snapshot-dir", type=str, default="")
    dim.add_argument("--out-dir", type=str, required=True)
    dim.add_argument("--dims", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096])
    dim.add_argument("--repeats", type=int, default=5)
    dim.add_argument("--max-files", type=int, default=-1)
    dim.add_argument("--qr-iters", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    dim.add_argument("--epsilon", type=float, default=1e-9)
    dim.add_argument("--root", type=int, default=2)
    dim.add_argument("--exponent-multiplier", type=float, default=1.0)
    dim.add_argument("--jitter", type=float, default=1e-4)
    dim.add_argument("--noise-scale", type=float, default=0.02)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "qr-benchmark":
        qr_benchmark_command(args)
    elif args.command == "dim-scaling":
        dim_scaling_command(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
