import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import math
import time
import logging
import random
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional
from torch.utils.tensorboard import SummaryWriter

# Shampoo imports
from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    DDPShampooConfig,
    CommunicationDType
)

# ========================================
# Helper Classes (Ported from vit.py)
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
            
    def get_cumulative_summary(self) -> Dict:
        return {name: {'total': s.total_time, 'avg': s.total_time/s.count if s.count else 0} 
                for name, s in self.timers.items()}

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
            if "Retrying in double precision" in record.getMessage():
                self.count += 1
        except Exception:
            self.handleError(record)

def gather_optimizer_state_from_all_ranks(optimizer, model, global_rank, world_size):
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

# ==========================================
# 1. Setup & Utility
# ==========================================
def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

# ==========================================
# 2. Ghost Batch Normalization & ResNet
# ==========================================
class GhostBatchNorm2d(nn.Module):
    def __init__(self, num_features, virtual_batch_size=64, momentum=0.1, eps=1e-5):
        super(GhostBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm2d(num_features, momentum=momentum, eps=eps)

    def forward(self, x):
        if not self.training or x.size(0) <= self.virtual_batch_size:
            return self.bn(x)
        chunks = x.chunk(int(torch.ceil(torch.tensor(x.size(0) / self.virtual_batch_size))), 0)
        res = [self.bn(chunk) for chunk in chunks]
        return torch.cat(res, dim=0)

class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None, virtual_batch_size=64):
        super(ResNet, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, GhostBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_algoperf_resnet50(virtual_batch_size=64):
    norm_layer = lambda num_features: GhostBatchNorm2d(
        num_features, virtual_batch_size=virtual_batch_size
    )
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        zero_init_residual=True,
        norm_layer=norm_layer,
        virtual_batch_size=virtual_batch_size
    )

# ==========================================
# 4. Dataset & Loader
# ==========================================
def get_dataloaders(data_path, batch_size, workers, world_size, global_rank):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Train Data
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=train_sampler
    )
    
    # Val Data (if available)
    valdir = os.path.join(data_path, 'val')
    val_loader = None
    if os.path.exists(valdir):
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, sampler=val_sampler
        )
    
    return train_loader, val_loader

# ==========================================
# 5. Main
# ==========================================
def main(args):
    local_rank = setup()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Setup Logging
    writer = SummaryWriter(log_dir=args.log_dir) if global_rank == 0 else None
    wall_clock_profiler.training_start_time = time.perf_counter()
    
    if global_rank == 0:
        print(f"Starting ResNet-50 training with Distributed Shampoo")
        print(f"World Size: {world_size}, Batch Size per GPU: {args.batch_size}")

    # Create Model
    model = create_algoperf_resnet50(virtual_batch_size=args.batch_size).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Setup Monitors
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

    # Distributed Shampoo Optimizer
    distributed_config = DDPShampooConfig(
        communication_dtype=CommunicationDType.FP32,
        num_trainers_per_group=world_size,
        communicate_params=False
    )

    optimizer = DistributedShampoo(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        epsilon=1e-10,
        weight_decay=args.weight_decay,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        use_decoupled_weight_decay=True,
        inv_root_override=2,
        exponent_multiplier=1,
        grafting_config=AdamGraftingConfig(beta2=args.beta2, epsilon=1e-8),
        distributed_config=distributed_config,
        use_protected_eigh=True,
        matrix_root_inv_threshold=0.0
    )

    criterion = nn.CrossEntropyLoss().to(local_rank)
    train_loader, val_loader = get_dataloaders(args.data_path, args.batch_size, args.workers, world_size, global_rank)
    
    cumulative_fallback_count = 0

    for epoch in range(args.epochs):
        wall_clock_profiler.reset_epoch_timers(epoch)
        eigh_monitor.reset_epoch()
        eigh_fallback_handler.count = 0
        
        epoch_start = time.perf_counter()
        train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        
        batch_start = time.perf_counter()
        
        for i, (images, target) in enumerate(train_loader):
            wall_clock_profiler.timers["data_loading"].update(time.perf_counter() - batch_start)
            
            images = images.to(local_rank, non_blocking=True)
            target = target.to(local_rank, non_blocking=True)

            wall_clock_profiler.start_timer("forward")
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, target)
            wall_clock_profiler.end_timer("forward")
            
            wall_clock_profiler.start_timer("backward")
            loss.backward()
            wall_clock_profiler.end_timer("backward")
            
            wall_clock_profiler.start_timer("optimizer")
            optimizer.step()
            wall_clock_profiler.end_timer("optimizer")

            running_loss += loss.item()
            
            if i % args.log_interval == 0 and global_rank == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{i}] Loss: {loss.item():.4f}")
                if writer:
                    writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + i)
            
            batch_start = time.perf_counter()

        # End Epoch Stats
        epoch_duration = time.perf_counter() - epoch_start
        cumulative_fallback_count += eigh_fallback_handler.count
        
        # Validation
        val_acc = 0.0
        if val_loader:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, target in val_loader:
                    images = images.to(local_rank, non_blocking=True)
                    target = target.to(local_rank, non_blocking=True)
                    output = model(images)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            # Reduce validation stats
            total_t = torch.tensor(total).to(local_rank)
            correct_t = torch.tensor(correct).to(local_rank)
            dist.all_reduce(total_t)
            dist.all_reduce(correct_t)
            val_acc = 100. * correct_t.item() / total_t.item()

        # Stats Gathering
        avg_loss = running_loss / len(train_loader)
        avg_loss_t = torch.tensor(avg_loss).to(local_rank)
        dist.all_reduce(avg_loss_t)
        avg_loss = avg_loss_t.item() / world_size
        
        # Timing Stats
        stats_vec = torch.tensor([
            epoch_duration,
            wall_clock_profiler.get_stats("forward").epoch_time,
            wall_clock_profiler.get_stats("backward").epoch_time,
            wall_clock_profiler.get_stats("optimizer").epoch_time,
            wall_clock_profiler.get_stats("eigendecomposition").epoch_time
        ]).to(local_rank)
        dist.all_reduce(stats_vec)
        stats_vec /= world_size
        
        if global_rank == 0:
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Avg Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Time: {stats_vec[0]:.2f}s (Opt: {stats_vec[3]:.2f}s, Eigh: {stats_vec[4]:.2f}s)")
            print(f"  Fallbacks: {eigh_fallback_handler.count}")
            
            if writer:
                writer.add_scalar('val/accuracy', val_acc, epoch)
                writer.add_scalar('timing/epoch', stats_vec[0], epoch)
                writer.add_scalar('timing/optimizer', stats_vec[3], epoch)

        # Save Checkpoint
        if (epoch + 1) % args.save_interval == 0:
            merged_state = gather_optimizer_state_from_all_ranks(optimizer, model, global_rank, world_size)
            if global_rank == 0:
                save_path = os.path.join(args.checkpoint_dir, f"resnet_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': merged_state,
                    'loss': avg_loss,
                    'val_acc': val_acc,
                    'fallbacks': cumulative_fallback_count
                }, save_path)
                print(f"Checkpoint saved: {save_path}")
            dist.barrier()

    if writer: writer.close()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ResNet50 ImageNet Training with Shampoo')
    parser.add_argument('--data-path', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    
    # Shampoo Params
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--max-preconditioner-dim', type=int, default=1024)
    parser.add_argument('--precondition-frequency', type=int, default=100)
    parser.add_argument('--start-preconditioning-step', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=10)

    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    main(args)
