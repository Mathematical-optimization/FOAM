import os
from xml.parsers.expat import model
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
from typing import Dict, List
import math
import argparse
from torch.utils.tensorboard import SummaryWriter
import functools
import time
from dataclasses import dataclass, field
from collections import defaultdict

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
        # Update cumulative stats
        self.total_time += time_delta
        self.count += 1
        self.min_time = min(self.min_time, time_delta)
        self.max_time = max(self.max_time, time_delta)
        
        # Update per-epoch stats
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
        """평균 시간 계산 (cumulative)"""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def epoch_avg_time(self) -> float:
        """평균 시간 계산 (per-epoch)"""
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
        """타이머 시작"""
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.active_timers[name] = time.perf_counter()
    
    def end_timer(self, name: str) -> float:
        """타이머 종료 및 시간 반환"""
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if name not in self.active_timers:
            return 0.0
        
        elapsed = time.perf_counter() - self.active_timers[name]
        self.timers[name].update(elapsed)
        del self.active_timers[name]
        return elapsed
    
    def get_stats(self, name: str) -> TimingStats:
        """특정 타이머의 통계 반환"""
        return self.timers[name]
    
    def reset_epoch_timers(self, epoch: int):
        """새로운 에폭 시작 시 에폭별 타이머 리셋"""
        self.current_epoch = epoch
        self.epoch_start_time = time.perf_counter()
        for stats in self.timers.values():
            stats.reset_epoch_stats()
    
    def get_epoch_summary(self) -> Dict[str, Dict[str, float]]:
        """현재 에폭의 타이밍 요약 반환"""
        summary = {}
        for name, stats in self.timers.items():
            if stats.epoch_count > 0:
                summary[name] = {
                    'epoch_time': stats.epoch_time,
                    'epoch_avg_time': stats.epoch_avg_time,
                    'epoch_min_time': stats.epoch_min_time,
                    'epoch_max_time': stats.epoch_max_time,
                    'epoch_count': stats.epoch_count
                }
        return summary
    
    def get_cumulative_summary(self) -> Dict[str, Dict[str, float]]:
        """전체 누적 타이밍 요약 반환"""
        summary = {}
        for name, stats in self.timers.items():
            summary[name] = {
                'total_time': stats.total_time,
                'avg_time': stats.avg_time,
                'min_time': stats.min_time,
                'max_time': stats.max_time,
                'count': stats.count
            }
        return summary

# Global profiler instance
wall_clock_profiler = EnhancedWallClockProfiler(use_cuda_sync=True)

# ========================================
# Modified Shampoo Components with Timing
# ========================================

# Monkey-patching을 위한 래퍼 함수들은 직접 train() 함수 내에서 정의

# ========================================
# ViT Model Components (unchanged)
# ========================================

def set_seed(seed : int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

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
    print(f"Rank {rank}: Seed set to {rank_seed} (base seed: {seed})")

class EighFallbackCounter(logging.Handler):
    """'eigh' 연산이 float64로 재시도될 때 발생하는 경고를 카운트합니다."""
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
    """Q, K, V가 분리된 커스텀 Multi-Head Attention 모듈"""
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
# Utility Functions (keeping essential ones)
# ========================================

def validate_checkpoint_completeness(optimizer_state, model):
    """체크포인트가 모든 필요한 정보를 포함하는지 검증"""
    expected_qkv_params = 0
    found_qkv_factor_matrices = 0
    
    for name, param in model.named_parameters():
        if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj']) and 'weight' in name:
            expected_qkv_params += 1
            if name in optimizer_state['state']:
                param_state = optimizer_state['state'][name]
                for key in param_state:
                    if isinstance(key, str) and 'factor_matrices' in key:
                        if isinstance(param_state[key], torch.Tensor) and param_state[key].numel() > 0:
                            found_qkv_factor_matrices += 1
    
    print(f"\n=== 체크포인트 완전성 검증 ===")
    print(f"예상 Q/K/V 파라미터 수: {expected_qkv_params}")
    print(f"Factor matrices를 가진 Q/K/V 파라미터: {found_qkv_factor_matrices // 2}")
    
    if found_qkv_factor_matrices < expected_qkv_params * 2:
        print("⚠️  경고: 일부 factor matrices가 누락되었을 수 있습니다!")
    else:
        print("✅ 모든 factor matrices가 정상적으로 수집되었습니다.")

def gather_optimizer_state_from_all_ranks(optimizer, model, global_rank, world_size):
    """모든 랭크에서 옵티마이저 상태를 수집하고 통합합니다."""
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
                is_factor_matrix = False
                
                if isinstance(state_key, str) and 'factor_matrices' in state_key:
                    is_factor_matrix = True
                
                for rank, state in enumerate(all_states):
                    if ('state' in state and 
                        param_key in state['state'] and 
                        state_key in state['state'][param_key]):
                        
                        value = state['state'][param_key][state_key]
                        
                        if isinstance(value, torch.Tensor):
                            if hasattr(value, '_local_tensor'):
                                value = value._local_tensor
                            
                            if is_factor_matrix:
                                if value.numel() > 0:
                                    if merged_value is None:
                                        merged_value = value.clone()
                                    elif merged_value.numel() == 0:
                                        merged_value = value.clone()
                            else:
                                if merged_value is None or (merged_value.numel() == 0 and value.numel() > 0):
                                    merged_value = value.clone()
                        else:
                            if merged_value is None:
                                merged_value = value
                
                if merged_value is not None:
                    merged_state['state'][param_key][state_key] = merged_value
        
        print("\n=== Factor Matrices 수집 검증 ===")
        total_factor_matrices = 0
        non_empty_factor_matrices = 0
        qkv_factor_matrices = 0
        
        for param_key, param_state in merged_state['state'].items():
            for state_key, value in param_state.items():
                if isinstance(state_key, str) and 'factor_matrices' in state_key:
                    total_factor_matrices += 1
                    if isinstance(value, torch.Tensor) and value.numel() > 0:
                        non_empty_factor_matrices += 1
                        if any(proj in param_key for proj in ['q_proj', 'k_proj', 'v_proj']):
                            qkv_factor_matrices += 1
        
        print(f"총 Factor Matrices: {total_factor_matrices}")
        print(f"비어있지 않은 Factor Matrices: {non_empty_factor_matrices}")
        print(f"Q/K/V Projection의 Factor Matrices: {qkv_factor_matrices}")
        
        return merged_state
    
    return None

def get_warmup_cosine_decay_lr(current_step: int, base_lr: float, num_steps: int, warmup_steps: int) -> float:
    """Warmup + Cosine Decay 학습률 스케줄러"""
    if current_step < warmup_steps:
        return base_lr * (current_step / warmup_steps)
    else:
        progress = (current_step - warmup_steps) / (num_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return base_lr * cosine_decay

def setup():
    """분산 학습 초기화"""
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
def cleanup():
    """분산 학습 종료"""
    dist.destroy_process_group()

def apply_transforms(examples: Dict[str, List[Image.Image]], transform) -> Dict[str, List[torch.Tensor]]:
    """데이터셋에 변환 적용"""
    examples['pixel_values'] = [transform(image.convert("RGB")) for image in examples['image']]
    return examples

def get_epsilon_config(args):
    """Epsilon 설정을 프리셋 또는 커스텀 값으로부터 생성"""
    config = {}
    
    presets = {
        'default': {
            'epsilon': 1e-08,
            'epsilon_left': None,
            'epsilon_right': None,
            'use_adaptive_epsilon': False,
            'condition_thresholds': None
        },
        'asymmetric': {
            'epsilon': 1e-10,
            'epsilon_left': 1e-08,
            'epsilon_right': 1e-05,
            'use_adaptive_epsilon': False,
            'condition_thresholds': None
        },
        'adaptive': {
            'epsilon': 1e-10,
            'epsilon_left': None,
            'epsilon_right': None,
            'use_adaptive_epsilon': True,
            'condition_thresholds': {1e6: 1e-5, 1e8: 1e-4}
        }
    }
    
    if args.epsilon_preset in presets:
        config = presets[args.epsilon_preset]
    else:
        config = presets['default']
        if args.epsilon is not None:
            config['epsilon'] = args.epsilon
        if args.epsilon_left is not None:
            config['epsilon_left'] = args.epsilon_left
        if args.epsilon_right is not None:
            config['epsilon_right'] = args.epsilon_right
        config['use_adaptive_epsilon'] = args.use_adaptive_epsilon
    
    return config

# ========================================
# Modified Training Function with Enhanced Timing
# ========================================

def train(args: argparse.Namespace):
    setup()
    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    set_seed_distributed(seed = args.seed, rank = global_rank)
    
    print(f"Running DDP training. Global Rank: {global_rank}, Local Rank: {local_rank}, World Size: {world_size}")

    writer = SummaryWriter(log_dir=args.log_dir) if global_rank == 0 else None
    
    # Start total training timer
    wall_clock_profiler.training_start_time = time.perf_counter()
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed()
        np_seed = worker_seed % (2**32)
        np.random.seed(np_seed)
        random.seed(worker_seed)

    # 데이터 변환 설정
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

    # Mixup 설정
    mixup_fn = None
    if args.mixup > 0 or args.label_smoothing > 0:
        mixup_args = {
            'mixup_alpha': args.mixup,
            'cutmix_alpha': 0.0,
            'label_smoothing': args.label_smoothing,
            'num_classes': 1000
        }
        mixup_fn = Mixup(**mixup_args)

    # 데이터셋 로드
    if global_rank == 0:
        print("Hugging Face Hub에서 ImageNet-1k 데이터셋을 다운로드 및 캐싱합니다...")
        load_dataset("imagenet-1k", cache_dir=args.data_path)
    dist.barrier()

    print(f"Rank {global_rank}에서 캐시된 ImageNet-1k 데이터셋을 로딩합니다...")
    dataset = load_dataset("imagenet-1k", cache_dir=args.data_path)

    train_dataset = dataset['train']
    train_dataset.set_transform(functools.partial(apply_transforms, transform=train_transform))
    
    val_dataset = dataset['validation']
    val_dataset.set_transform(functools.partial(apply_transforms, transform=val_transform))
    
    def collate_fn(batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'label': torch.tensor([x['label'] for x in batch], dtype=torch.long)
        }

    # DataLoader 설정
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True, seed= args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                             num_workers=args.workers, pin_memory=True, collate_fn=collate_fn, 
                             worker_init_fn = seed_worker, generator = torch.Generator().manual_seed(args.seed))
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False, seed = args.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, 
                           num_workers=args.workers, pin_memory=True, collate_fn=collate_fn, 
                           worker_init_fn = seed_worker, generator = torch.Generator().manual_seed(args.seed))

    torch.manual_seed(args.seed)

    # 모델 설정
    vit_params = {
        'img_size': 224, 'patch_size': 16, 'embedding_dim': 384, 'depth': 12,
        'num_heads': 6, 'mlp_dim': 1536, 'num_classes': 1000
    }
    # 모델 생성
    model = VisionTransformer(**vit_params).to(local_rank)
    
    # 모델 크기 정보 출력
    model = DDP(model, device_ids=[local_rank])

    # 손실 함수
    criterion = nn.CrossEntropyLoss().to(local_rank)
    epsilon_config = get_epsilon_config(args)

    # Monkey-patch torch.linalg.eigh for timing
    original_torch_eigh = torch.linalg.eigh
    def timed_torch_eigh(A, UPLO='L', *, out = None):
        wall_clock_profiler.start_timer("eigendecomposition")
        try:
            result = original_torch_eigh(A, UPLO, out= out)
        finally:
            wall_clock_profiler.end_timer("eigendecomposition")
        return result
    
    torch.linalg.eigh = timed_torch_eigh
    
    if global_rank == 0:
        print("torch.linalg.eigh has been patched for timing measurement")

    # Monkey-patch Shampoo compute_root_inverse for timing
    from optimizers.distributed_shampoo.utils.shampoo_preconditioner_list import ShampooPreconditionerList
    original_compute_root_inverse = ShampooPreconditionerList.compute_root_inverse
    
    def timed_compute_root_inverse_wrapper(self):
        wall_clock_profiler.start_timer("compute_root_inverse_total")
        result = original_compute_root_inverse(self)
        wall_clock_profiler.end_timer("compute_root_inverse_total")
        return result
    
    ShampooPreconditionerList.compute_root_inverse = timed_compute_root_inverse_wrapper

    # 옵티마이저 설정
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
        momentum=False,
        use_nadam=False,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        grafting_config=AdamGraftingConfig(beta2=args.adam_grafting_beta2, epsilon=args.grafting_epsilon),
        use_normalized_grafting=False,
        use_decoupled_weight_decay=True,
        inv_root_override=2,
        exponent_multiplier=1,
        distributed_config=distributed_config,
        preconditioner_dtype=torch.float32,
        matrix_root_inv_threshold=args.matrix_root_inv_threshold,
    )

    # Eigh Fallback Counter 설정
    start_epoch = 0
    cumulative_fallback_count = 0
    
    eigh_fallback_handler = EighFallbackCounter()
    matrix_logger = logging.getLogger('optimizers.matrix_functions')
    matrix_logger.setLevel(logging.WARNING)
    matrix_logger.addHandler(eigh_fallback_handler)

    # 체크포인트 로드 (simplified)
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=f'cuda:{local_rank}')
            
            if 'model_state_dict' in checkpoint:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_distributed_state_dict(
                        checkpoint['optimizer_state_dict'],
                        key_to_param=model.module.named_parameters()
                    )
                    print("=> loaded optimizer state")
                except Exception as e:
                    print(f"Error loading optimizer state: {e}")
            
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            
            if 'cumulative_fallback_count' in checkpoint:
                cumulative_fallback_count = checkpoint['cumulative_fallback_count']

    total_steps = len(train_loader) * args.epochs
    
    # Training loop with enhanced timing
    for epoch in range(start_epoch, args.epochs):
        # Reset per-epoch timing stats
        wall_clock_profiler.reset_epoch_timers(epoch)
        epoch_start_time = time.perf_counter()
        
        train_sampler.set_epoch(epoch)
        eigh_fallback_handler.count = 0
        
        model.train()
        running_loss = 0.0
        
        # Epoch-level timing stats
        epoch_forward_time = 0.0
        epoch_backward_time = 0.0
        epoch_optimizer_time = 0.0
        epoch_data_loading_time = 0.0
        
        batch_start_time = time.perf_counter()
        
        for i, batch in enumerate(train_loader):
            # Data loading time
            data_loading_time = time.perf_counter() - batch_start_time
            epoch_data_loading_time += data_loading_time
            
            current_step = epoch * len(train_loader) + i
            images = batch['pixel_values'].to(local_rank, non_blocking=True)
            labels = batch['label'].to(local_rank, non_blocking=True)
            
            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            # 학습률 스케줄링
            new_lr = get_warmup_cosine_decay_lr(current_step, args.base_lr, total_steps, args.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            # Forward pass timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_start = time.perf_counter()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_time = time.perf_counter() - forward_start
            epoch_forward_time += forward_time
            
            # Backward pass timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_start = time.perf_counter()
            
            loss.backward()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_time = time.perf_counter() - backward_start
            epoch_backward_time += backward_time
            
            # Optimizer step timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            optimizer_start = time.perf_counter()
            
            optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            optimizer_time = time.perf_counter() - optimizer_start
            epoch_optimizer_time += optimizer_time
            
            running_loss += loss.item()
            
            if global_rank == 0 and (i + 1) % args.log_interval == 0:
                # Get both epoch and cumulative stats
                eigendecomp_stats = wall_clock_profiler.get_stats("eigendecomposition")
                root_inv_stats = wall_clock_profiler.get_stats("compute_root_inverse_total")
                
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"LR: {new_lr:.6f}, Loss: {loss.item():.4f}")
                
                # Show per-epoch stats in iteration logs
                if eigendecomp_stats.epoch_count > 0:
                    print(f"  Epoch Eigendecomp: {eigendecomp_stats.epoch_time:.2f}s "
                          f"(Count: {eigendecomp_stats.epoch_count}, Avg: {eigendecomp_stats.epoch_avg_time*1000:.2f}ms)")
                if root_inv_stats.epoch_count > 0:
                    print(f"  Epoch Root Inverse: {root_inv_stats.epoch_time:.2f}s "
                          f"(Count: {root_inv_stats.epoch_count}, Avg: {root_inv_stats.epoch_avg_time*1000:.2f}ms)")
                
                if writer:
                    writer.add_scalar('learning_rate', new_lr, current_step)
            
            batch_start_time = time.perf_counter()
        
        # Epoch timing summary
        epoch_total_time = time.perf_counter() - epoch_start_time
        
        # Get epoch-specific stats from profiler
        eigendecomp_stats = wall_clock_profiler.get_stats("eigendecomposition")
        root_inv_stats = wall_clock_profiler.get_stats("compute_root_inverse_total")
        
        # Gather timing stats across all ranks
        timing_tensors = torch.tensor([
            epoch_total_time,
            epoch_forward_time,
            epoch_backward_time, 
            epoch_optimizer_time,
            epoch_data_loading_time,
            eigendecomp_stats.epoch_time,  # Current epoch time
            root_inv_stats.epoch_time,     # Current epoch time
            eigendecomp_stats.total_time,  # Cumulative time
            root_inv_stats.total_time      # Cumulative time
        ]).to(local_rank)
        
        dist.all_reduce(timing_tensors, op=dist.ReduceOp.SUM)
        timing_tensors /= world_size
        
        # 에폭별 Fallback 카운트 집계
        local_fallback_count = torch.tensor(eigh_fallback_handler.count).to(local_rank)
        epoch_fallback_count = local_fallback_count.clone()
        dist.all_reduce(epoch_fallback_count, op=dist.ReduceOp.SUM)
        cumulative_fallback_count += epoch_fallback_count.item()
        
        # 학습 손실 집계
        total_loss_tensor = torch.tensor(running_loss).to(local_rank)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        avg_epoch_loss = total_loss_tensor.item() / world_size / len(train_loader)
        
        # Validation
        val_start_time = time.perf_counter()
        model.eval()
        correct = 0 
        total = 0
        val_loss_sum = 0.0
        num_batches = len(val_loader)

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

        val_time = time.perf_counter() - val_start_time
        
        # Validation 결과 집계
        total_tensor = torch.tensor(total).to(local_rank)
        correct_tensor = torch.tensor(correct).to(local_rank)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        
        local_avg_loss = val_loss_sum / num_batches if num_batches > 0 else 0.0
        avg_loss_tensor = torch.tensor(local_avg_loss).to(local_rank)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        
        avg_val_loss = avg_loss_tensor.item() / world_size
        accuracy = 100 * correct_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0.0

        # Calculate total elapsed time
        total_elapsed = time.perf_counter() - wall_clock_profiler.training_start_time
        
        # Enhanced results output with proper timing
        if global_rank == 0:
            print(f"\n{'='*80}")
            print(f"Epoch [{epoch+1}/{args.epochs}] Summary:")
            print(f"  Training Loss: {avg_epoch_loss:.4f}")
            print(f"  Validation Accuracy: {accuracy:.2f}%")
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            
            # Epoch-level timing
            print(f"\nTiming Statistics for this Epoch (Averaged across {world_size} GPUs):")
            print(f"  Epoch Total Time: {timing_tensors[0].item():.2f}s")
            
            forward_pct = (timing_tensors[1].item()/timing_tensors[0].item()*100) if timing_tensors[0].item() > 0 else 0
            backward_pct = (timing_tensors[2].item()/timing_tensors[0].item()*100) if timing_tensors[0].item() > 0 else 0
            optimizer_pct = (timing_tensors[3].item()/timing_tensors[0].item()*100) if timing_tensors[0].item() > 0 else 0
            data_pct = (timing_tensors[4].item()/timing_tensors[0].item()*100) if timing_tensors[0].item() > 0 else 0
            val_pct = (val_time/timing_tensors[0].item()*100) if timing_tensors[0].item() > 0 else 0
            
            print(f"    - Forward Pass: {timing_tensors[1].item():.2f}s ({forward_pct:.1f}%)")
            print(f"    - Backward Pass: {timing_tensors[2].item():.2f}s ({backward_pct:.1f}%)")
            print(f"    - Optimizer Step: {timing_tensors[3].item():.2f}s ({optimizer_pct:.1f}%)")
            print(f"    - Data Loading: {timing_tensors[4].item():.2f}s ({data_pct:.1f}%)")
            print(f"    - Validation: {val_time:.2f}s ({val_pct:.1f}%)")
            
            # Current epoch Shampoo statistics
            epoch_eigendecomp = timing_tensors[5].item()
            epoch_root_inv = timing_tensors[6].item()
            
            print(f"\nShampoo Statistics for Current Epoch:")
            print(f"  Eigendecomp Time: {epoch_eigendecomp:.2f}s ({epoch_eigendecomp/timing_tensors[0].item()*100:.1f}% of epoch)")
            print(f"  Root Inverse Time: {epoch_root_inv:.2f}s ({epoch_root_inv/timing_tensors[0].item()*100:.1f}% of epoch)")
            if epoch_root_inv > 0:
                print(f"  Eigendecomp/Root-Inv Ratio: {epoch_eigendecomp/epoch_root_inv*100:.1f}%")
            
            # Note: The epoch Shampoo times should be part of Optimizer Step
            if epoch_eigendecomp + epoch_root_inv > 0:
                shampoo_pct_of_optimizer = (epoch_eigendecomp + epoch_root_inv) / timing_tensors[3].item() * 100
                print(f"  Shampoo operations: {shampoo_pct_of_optimizer:.1f}% of Optimizer Step")
            
            # Cumulative statistics
            cumulative_eigendecomp = timing_tensors[7].item()
            cumulative_root_inv = timing_tensors[8].item()
            
            print(f"\nCumulative Shampoo Statistics (across all {epoch+1} epochs):")
            print(f"  Total Eigendecomp Time: {cumulative_eigendecomp:.2f}s")
            print(f"  Total Root Inverse Time: {cumulative_root_inv:.2f}s")
            print(f"  Average per Epoch: {cumulative_eigendecomp/(epoch+1):.2f}s eigendecomp, "
                  f"{cumulative_root_inv/(epoch+1):.2f}s root inverse")
            print(f"  Percentage of Total Training Time: {(cumulative_eigendecomp+cumulative_root_inv)/total_elapsed*100:.1f}%")
            
            print(f"\nTotal Training Elapsed Time: {total_elapsed/3600:.2f} hours")
            print(f"  Epoch FP32→FP64 Fallbacks: {epoch_fallback_count.item()}")
            print(f"  Cumulative Fallbacks: {cumulative_fallback_count}")
            print(f"{'='*80}\n")
            
            if writer:
                writer.add_scalar('validation_accuracy', accuracy, epoch)
                writer.add_scalar('validation_loss', avg_val_loss, epoch)
                writer.add_scalar('Timing/epoch_eigendecomp', epoch_eigendecomp, epoch)
                writer.add_scalar('Timing/epoch_root_inv', epoch_root_inv, epoch)
                writer.add_scalar('Timing/cumulative_eigendecomp', cumulative_eigendecomp, epoch)
                writer.add_scalar('Timing/cumulative_root_inv', cumulative_root_inv, epoch)
        
        # Save checkpoint (simplified)
        if (epoch + 1) % args.save_interval == 0:
            
            print(f"Saving checkpoint...")
            merged_optimizer_state = gather_optimizer_state_from_all_ranks(optimizer, model, global_rank, world_size)
            
            
            if global_rank==0:
                save_path = os.path.join(args.save_dir, f"vit_checkpoint_epoch_{epoch+1}.pth")
                print(f"Saving checkpoint to {save_path}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': merged_optimizer_state,
                    'cumulative_fallback_count': cumulative_fallback_count,
                    'accuracy': accuracy,
                    'val_loss': avg_val_loss,
                    'timing_stats': {
                        'cumulative': wall_clock_profiler.get_cumulative_summary(),
                        'last_epoch': wall_clock_profiler.get_epoch_summary()
                    },
                    'total_elapsed_time': total_elapsed,
                }, save_path)
                print(f"Checkpoint saved!\n")
            
            dist.barrier()
    
    # Final timing report
    if global_rank == 0:
        total_training_time = time.perf_counter() - wall_clock_profiler.training_start_time
        cumulative_summary = wall_clock_profiler.get_cumulative_summary()
        
        print("\n" + "="*80)
        print("FINAL TRAINING TIMING REPORT")
        print("="*80)
        print(f"Total Training Time: {total_training_time/3600:.2f} hours")
        print(f"\nDetailed Timing Breakdown:")
        
        for name, stats in cumulative_summary.items():
            print(f"\n{name}:")
            print(f"  Total: {stats['total_time']:.2f}s ({stats['total_time']/total_training_time*100:.1f}%)")
            print(f"  Average: {stats['avg_time']*1000:.2f}ms")
            print(f"  Min: {stats['min_time']*1000:.2f}ms")
            print(f"  Max: {stats['max_time']*1000:.2f}ms")
            print(f"  Count: {stats['count']}")
        
        # Special focus on Shampoo operations
        if 'eigendecomposition' in cumulative_summary and 'compute_root_inverse_total' in cumulative_summary:
            eigen_stats = cumulative_summary['eigendecomposition']
            root_stats = cumulative_summary['compute_root_inverse_total']
            shampoo_total = eigen_stats['total_time'] + root_stats['total_time']
            
            print(f"\n{'='*40}")
            print("SHAMPOO OPTIMIZER ANALYSIS")
            print(f"{'='*40}")
            print(f"Total time in Shampoo operations: {shampoo_total:.2f}s")
            print(f"  - Eigendecomposition: {eigen_stats['total_time']:.2f}s ({eigen_stats['count']} calls)")
            print(f"  - Root Inverse: {root_stats['total_time']:.2f}s ({root_stats['count']} calls)")
            print(f"Percentage of total training time: {shampoo_total/total_training_time*100:.2f}%")
            print(f"Average time per preconditioner update:")
            print(f"  - Eigendecomp: {eigen_stats['avg_time']*1000:.2f}ms")
            print(f"  - Root Inverse: {root_stats['avg_time']*1000:.2f}ms")
        
        print("="*80 + "\n")
    
    if writer:
        writer.close()
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT-Base Training with Enhanced Shampoo Timing')
    parser.add_argument('--data-path', type=str, required=True, help='Path to cache Hugging Face datasets')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for TensorBoard logs')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size per GPU')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--base-lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--warmup-steps', type=int, default=10000, help='Number of warmup steps')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.95, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.99, help='Shampoo beta2')
    parser.add_argument('--adam-grafting-beta2', type=float, default=0.99, help='Adam grafting beta2')
    parser.add_argument('--grafting-epsilon', type=float, default=1e-10, help='Grafting epsilon')
    parser.add_argument('--max-preconditioner-dim', type=int, default=1024, help='Max preconditioner dimension')
    parser.add_argument('--precondition-frequency', type=int, default=10, help='Preconditioning frequency')
    parser.add_argument('--start-preconditioning-step', type=int, default=10, help='Start preconditioning step')
    parser.add_argument('--mixup', type=float, default=0.2, help='Mixup alpha')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--log-interval', type=int, default=20, help='Logging interval')
    parser.add_argument('--save-interval', type=int, default=10, help='Checkpoint save interval')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--matrix-root-inv-threshold', type=float, default=0.0, help='Matrix root inverse threshold')
    
    # Epsilon configuration options
    parser.add_argument('--epsilon-preset', type=str, default='default',
                        choices=['default', 'asymmetric'],
                        help='Epsilon configuration preset')
    parser.add_argument('--epsilon', type=float, default=None, help='Custom epsilon value')
    parser.add_argument('--epsilon-left', type=float, default=None, help='Custom left epsilon')
    parser.add_argument('--epsilon-right', type=float, default=None, help='Custom right epsilon')
    parser.add_argument('--use-adaptive-epsilon', action='store_true', help='Use adaptive epsilon')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    train(args)
