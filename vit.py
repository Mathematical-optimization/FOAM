import os
from xml.parsers.expat import model
import torch
import torch.nn as nn
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

class EighFallbackCounter(logging.Handler):
    """'eigh' 연산이 float64로 재시도될 때 발생하는 경고를 카운트합니다."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0
        self.setLevel(logging.WARNING)  # WARNING 레벨만 처리

    def emit(self, record):
        try:
            # record.getMessage()를 사용하여 포맷된 메시지 가져오기
            message = record.getMessage()
            if "Retrying in double precision" in message:
                self.count += 1
        except Exception:
            # 에러 발생 시 조용히 무시 (Handler의 일반적인 패턴)
            self.handleError(record)

# --- ViT 모델 코드 ---

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

        # Q, K, V를 위한 별도의 Linear 레이어
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, need_weights=False):
        batch_size = query.shape[0]

        # 1. Linear 프로젝션
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. Multi-head를 위해 reshape 및 전치
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3. Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 4. Attention 적용
        attn_output = torch.matmul(attn_probs, v)

        # 5. Concat 및 Output 프로젝션
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

# --- 유틸리티 함수들 ---

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
    """
    모든 랭크에서 옵티마이저 상태를 수집하고 통합합니다.
    Distributed Shampoo의 분산된 factor matrices를 올바르게 처리합니다.
    """
    # 로컬 상태 수집
    local_state = optimizer.distributed_state_dict(
        key_to_param=model.module.named_parameters()
    )
    
    # 모든 랭크에서 상태 수집
    all_states = [None] * world_size
    dist.all_gather_object(all_states, local_state)
    
    if global_rank == 0:
        merged_state = {
            'state': {},
            'param_groups': local_state['param_groups']
        }
        
        # 모든 파라미터 키 수집
        all_param_keys = set()
        for state in all_states:
            if 'state' in state:
                all_param_keys.update(state['state'].keys())
        
        print(f"총 {len(all_param_keys)}개의 파라미터 발견")
        
        # 각 파라미터에 대해 상태 통합
        for param_key in all_param_keys:
            merged_state['state'][param_key] = {}
            param_state_keys = set()
            
            # 모든 랭크에서 이 파라미터의 모든 state key 수집
            for state in all_states:
                if 'state' in state and param_key in state['state']:
                    param_state_keys.update(state['state'][param_key].keys())
            
            # 각 state key에 대해 처리
            for state_key in param_state_keys:
                merged_value = None
                is_factor_matrix = False
                
                # state_key가 factor_matrices를 포함하는지 확인
                if isinstance(state_key, str) and 'factor_matrices' in state_key:
                    is_factor_matrix = True
                
                # 모든 랭크에서 이 state_key의 값을 수집
                for rank, state in enumerate(all_states):
                    if ('state' in state and 
                        param_key in state['state'] and 
                        state_key in state['state'][param_key]):
                        
                        value = state['state'][param_key][state_key]
                        
                        # Tensor 처리
                        if isinstance(value, torch.Tensor):
                            # DTensor 처리 (만약 DTensor가 사용되는 경우)
                            if hasattr(value, '_local_tensor'):
                                value = value._local_tensor
                            
                            # Factor matrix의 경우 특별 처리
                            if is_factor_matrix:
                                # 빈 텐서가 아닌 경우만 저장
                                if value.numel() > 0:
                                    if merged_value is None:
                                        merged_value = value.clone()
                                    elif merged_value.numel() == 0:
                                        # 이전에 빈 텐서였다면 교체
                                        merged_value = value.clone()
                            else:
                                # Factor matrix가 아닌 경우
                                if merged_value is None or (merged_value.numel() == 0 and value.numel() > 0):
                                    merged_value = value.clone()
                        else:
                            # Tensor가 아닌 경우
                            if merged_value is None:
                                merged_value = value
                
                # 최종 값 저장
                if merged_value is not None:
                    merged_state['state'][param_key][state_key] = merged_value
        
        # 검증: factor matrices 수집 상태 확인
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
                        # Q/K/V 파라미터인지 확인
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
    """
    Epsilon 설정을 프리셋 또는 커스텀 값으로부터 생성
    
    Returns:
        dict: Shampoo optimizer에 전달할 epsilon 관련 파라미터
    """
    config = {}
    
    # 프리셋 설정
    presets = {
        'default': {
            'epsilon': 1e-08,
            'epsilon_left': None,  # None이면 epsilon 값 사용
            'epsilon_right': None,
            'use_adaptive_epsilon': False,
            'condition_thresholds': None
        },
        'asymmetric': {
            'epsilon': 1e-08,
            'epsilon_left': 1e-8,
            'epsilon_right': 5e-5,
            'use_adaptive_epsilon': False,
            'condition_thresholds': None
        },
        'adaptive': {
            'epsilon': 1e-08,
            'epsilon_left': None,
            'epsilon_right': None,
            'use_adaptive_epsilon': True,
            'condition_thresholds': {1e6: 1e-5, 1e8: 5e-5}
        },
        'adaptive_asymmetric': {
            'epsilon': 1e-8,
            'epsilon_left': 1e-8,
            'epsilon_right': 1e-4,
            'use_adaptive_epsilon': True,
            'condition_thresholds': {5e6: 5e-3, 1e6: 1e-3}
        }
    }
    
    if args.epsilon_preset == 'custom':
        # 커스텀 설정: 명령행 인자 사용
        config['epsilon'] = args.epsilon
        config['epsilon_left'] = args.epsilon_left
        config['epsilon_right'] = args.epsilon_right
        config['use_adaptive_epsilon'] = args.use_adaptive_epsilon
        
        # condition_thresholds 파싱
        if args.use_adaptive_epsilon and args.condition_thresholds:
            thresholds = {}
            for pair in args.condition_thresholds.split(','):
                threshold, epsilon = pair.split(':')
                thresholds[float(threshold)] = float(epsilon)
            config['condition_thresholds'] = thresholds
        else:
            config['condition_thresholds'] = None
    else:
        # 프리셋 사용
        config = presets[args.epsilon_preset].copy()
    
    # 로깅
    if dist.get_rank() == 0:
        print("\n" + "="*60)
        print("EPSILON CONFIGURATION")
        print("="*60)
        print(f"Preset: {args.epsilon_preset}")
        print(f"Base epsilon (1D): {config['epsilon']}")
        print(f"Left matrix epsilon: {config.get('epsilon_left', 'Same as base')}")
        print(f"Right matrix epsilon: {config.get('epsilon_right', 'Same as base')}")
        print(f"Adaptive epsilon: {config['use_adaptive_epsilon']}")
        if config.get('condition_thresholds'):
            print(f"Condition thresholds: {config['condition_thresholds']}")
        print("="*60 + "\n")
    
    return config

# --- 메인 학습 함수 ---
        
def train(args: argparse.Namespace):
    setup()
    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    
    print(f"Running DDP training. Global Rank: {global_rank}, Local Rank: {local_rank}, World Size: {world_size}")

    writer = SummaryWriter(log_dir=args.log_dir) if global_rank == 0 else None

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
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                             num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, 
                           num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    # 모델 생성
    vit_s16_params = {
        'img_size': 224, 'patch_size': 16, 'embedding_dim': 384, 'depth': 12,
        'num_heads': 6, 'mlp_dim': 1536, 'num_classes': 1000
    }
    model = VisionTransformer(**vit_s16_params).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # 손실 함수
    criterion = nn.CrossEntropyLoss().to(local_rank)
    epsilon_config = get_epsilon_config(args)

    # 옵티마이저 설정
    optimizer = DistributedShampoo(
        model.parameters(),
        lr=args.base_lr,
        betas=(args.beta1, 0.99),
        epsilon = epsilon_config['epsilon'], # 1D Tensor용.
        epsilon_left = epsilon_config.get('epsilon_left'), #L matrix epsilon
        epsilon_right = epsilon_config.get('epsilon_right'),  # R matrix epsilon
        use_adaptive_epsilon = epsilon_config['use_adaptive_epsilon'],
        condition_thresholds=epsilon_config.get('condition_thresholds'),
        momentum=False,
        weight_decay=args.weight_decay,
        max_preconditioner_dim=1024,
        precondition_frequency=20,
        use_normalized_grafting=False,
        inv_root_override=2,
        exponent_multiplier=1,
        start_preconditioning_step=20,
        use_nadam=False,
        use_decoupled_weight_decay=True,
        grafting_config=AdamGraftingConfig(beta2=0.99, epsilon=1e-07),
        distributed_config=DDPShampooConfig(
            communication_dtype=CommunicationDType.FP32,
            num_trainers_per_group=-1,
            communicate_params=False
        )
    )
    
    # Eigh Fallback Counter 설정
    start_epoch = 0
    cumulative_fallback_count = 0  # 전체 누적 카운트
    
    eigh_fallback_handler = EighFallbackCounter()
    matrix_logger = logging.getLogger('optimizers.matrix_functions')
    matrix_logger.setLevel(logging.WARNING)
    matrix_logger.addHandler(eigh_fallback_handler)

    # 체크포인트 로드
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=f'cuda:{local_rank}')
            
            if 'model_state_dict' in checkpoint:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.module.load_state_dict(checkpoint)
            
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
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            
            # 누적 fallback 카운트 복원
            if 'cumulative_fallback_count' in checkpoint:
                cumulative_fallback_count = checkpoint['cumulative_fallback_count']
                print(f"=> restored cumulative fallback count: {cumulative_fallback_count}")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    total_steps = len(train_loader) * args.epochs
    
    # 학습 루프
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        
        # 에폭별 카운트를 위해 리셋
        eigh_fallback_handler.count = 0
        
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            current_step = epoch * len(train_loader) + i
            images = batch['pixel_values'].to(local_rank, non_blocking=True)
            labels = batch['label'].to(local_rank, non_blocking=True)
            
            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            # 학습률 스케줄링
            new_lr = get_warmup_cosine_decay_lr(current_step, args.base_lr, total_steps, args.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            # Forward & Backward
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if global_rank == 0 and (i + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"LR: {new_lr:.6f}, Loss: {loss.item():.4f}")
                if writer:
                    writer.add_scalar('learning_rate', new_lr, current_step)
        
        # 에폭별 Fallback 카운트 집계
        local_fallback_count = torch.tensor(eigh_fallback_handler.count).to(local_rank)
        epoch_fallback_count = local_fallback_count.clone()
        dist.all_reduce(epoch_fallback_count, op=dist.ReduceOp.SUM)
        
        # 누적 카운트 업데이트
        cumulative_fallback_count += epoch_fallback_count.item()
        
        # 학습 손실 집계
        total_loss_tensor = torch.tensor(running_loss).to(local_rank)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        avg_epoch_loss = total_loss_tensor.item() / world_size / len(train_loader)
        
        if global_rank == 0:
            if writer:
                writer.add_scalar('training_loss', avg_epoch_loss, epoch)
                
        # Validation
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

        # 결과 출력 및 로깅
        if global_rank == 0:
            print(f"\n{'='*80}")
            print(f"Epoch [{epoch+1}/{args.epochs}] Summary:")
            print(f"  Training Loss: {avg_epoch_loss:.4f}")
            print(f"  Validation Accuracy: {accuracy:.2f}%")
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            print(f"  Epoch FP32→FP64 Fallbacks: {epoch_fallback_count.item()}")
            print(f"  Cumulative Fallbacks: {cumulative_fallback_count}")
            print(f"{'='*80}\n")
            
            if writer:
                writer.add_scalar('validation_accuracy', accuracy, epoch)
                writer.add_scalar('validation_loss', avg_val_loss, epoch)
                writer.add_scalar('Diagnostics/Eigh_Fallback_Per_Epoch', epoch_fallback_count.item(), epoch)
                writer.add_scalar('Diagnostics/Eigh_Fallback_Cumulative', cumulative_fallback_count, epoch)
        
        # 체크포인트 저장
        if (epoch + 1) % args.save_interval == 0:
            print(f"Rank {global_rank}: Gathering optimizer states for checkpoint...")
            merged_optimizer_state = gather_optimizer_state_from_all_ranks(optimizer, model, global_rank, world_size)
            
            if global_rank == 0:
                # 체크포인트 검증
                validate_checkpoint_completeness(merged_optimizer_state, model.module)
        
                save_path = os.path.join(args.save_dir, f"vit_checkpoint_epoch_{epoch+1}.pth")
                print(f"Saving checkpoint to {save_path}")
                print(f"  Cumulative FP32→FP64 fallbacks: {cumulative_fallback_count}")
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': merged_optimizer_state,
                    'cumulative_fallback_count': cumulative_fallback_count,
                    'accuracy': accuracy,
                    'val_loss': avg_val_loss,
                }, save_path)
                print(f"Checkpoint saved successfully!\n")
                
            dist.barrier()
    
    if writer:
        writer.close()
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vision Transformer on ImageNet with DDP, Shampoo, and Algoperf Augmentations')
    parser.add_argument('--data-path', type=str, required=True, help='Path to cache Hugging Face datasets')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for TensorBoard logs')
    parser.add_argument('--epochs', type=int, default=90, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size per GPU')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--log-interval', type=int, default=300, help='Logging frequency')
    parser.add_argument('--save-interval', type=int, default=10, help='Checkpoint saving frequency')
    parser.add_argument('--base-lr', type=float, default=0.0013, help='Base learning rate')
    parser.add_argument('--warmup-steps', type=int, default=6000, help='Number of warmup steps')
    parser.add_argument('--mixup', type=float, default=0.2, help='Mixup alpha (default: 0.2). Set 0 to disable.')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay (default: 0.0005)')
    parser.add_argument('--beta1', type=float, default=0.95, help='Beta1/Momentum (default: 0.95)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    parser.add_argument('--epsilon-preset', type = str, default = 'default', choices = ['default', 'asymmetric', 'adpative', 'adaptive_asymmetric' ], help = 'Epsilon configuration preset')
    parser.add_argument('--epsilon', type= float, default = 1e-08)
    parser.add_argument('--epsilon-left', type = float, default = None)
    parser.add_argument('--epsilon-right', type = float, default = None)
    parser.add_argument('--use-adaptive-epsilon', action = 'store_true')
    parser.add_argument('--condition-thresholds', type = str, default= '1e6:1e-5, 1e8:5e-5')
    args = parser.parse_args()
    train(args)
