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
# Shampoo Optimizer Imports
from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    DDPShampooConfig,
    CommunicationDType
)

# ==========================================
# 1. Utilities & Setup
# ==========================================

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
    """분산 학습 초기화"""
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def get_lr_schedule(current_step, warmup_steps, base_lr, total_steps):
    """Warmup + Cosine Decay Scheduler"""
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
        
        # Filter long audio
        if waveform.shape[1] > self.args.max_audio_length:
            return None
            
        spec = self.melspec(waveform)
        spec = torch.log(spec + 1e-9)
        
        if self.train:
            spec = self.spec_augment(spec)
        
        # (Channel, n_mels, Time) -> (Time, n_mels)
        return spec.squeeze(0).transpose(0, 1), transcript

# 전역 TextTransform 인스턴스 (worker_init_fn 문제 방지용)
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
        out = out.transpose(0, 1) # (Time, Batch, Classes) for CTC
        return out, out_lengths

# ==========================================
# 4. Main Training Loop
# ==========================================

def main(args):
    local_rank = setup()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    set_seed_distributed(args.seed, global_rank)

    if global_rank == 0:
        print(f"Training Config: {args}")
        print(f"World Size: {world_size}, Local Rank: {local_rank}")
        print(f"Model Architecture: Conformer {args.num_layers} layers (AlgoPerf Spec)")

    # --- Data Preparation ---
    if global_rank == 0:
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)
        # 다운로드 (메인 프로세스만)
        print("Downloading Librispeech datasets...")
        torchaudio.datasets.LIBRISPEECH(root=args.data_path, url="train-clean-100", download=True)
        torchaudio.datasets.LIBRISPEECH(root=args.data_path, url="train-clean-360", download=True)
        torchaudio.datasets.LIBRISPEECH(root=args.data_path, url="train-other-500", download=True)  
    dist.barrier()

    train_dataset = AlgoPerfLibriSpeech(
        root=args.data_path, 
        url="train-clean-100", 
        download=False, # 위에서 이미 다운로드 함
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

    # --- Model Setup ---
    model = ConformerAlgoPerf(
        num_classes=len(text_transform),
        input_dim=args.n_mels,
        encoder_dim=args.encoder_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        depthwise_kernel_size=args.depthwise_kernel_size
    ).to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # --- Optimizer Setup (Shampoo) ---
    # Eigh fallback logging setup
    eigh_fallback_handler = EighFallbackCounter()
    matrix_logger = logging.getLogger('optimizers.matrix_functions')
    matrix_logger.setLevel(logging.WARNING)
    matrix_logger.addHandler(eigh_fallback_handler)

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
        epsilon=1e-12,
        momentum=0.0,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        grafting_config=AdamGraftingConfig(beta2=args.beta2, epsilon=1e-8),
        use_decoupled_weight_decay=True,
        distributed_config=distributed_config,
        preconditioner_dtype=torch.float32,
        use_protected_eigh=True
    )

    criterion = nn.CTCLoss(blank=len(text_transform)-1).to(local_rank)
    
    total_steps = len(train_loader) * args.epochs
    global_step = 0

    if global_rank == 0:
        print("Start Training...")

    model.train()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        valid_batches = 0
        
        for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(train_loader):
            if spectrograms is None: continue
            
            spectrograms = spectrograms.to(local_rank, non_blocking=True)
            labels = labels.to(local_rank, non_blocking=True)
            input_lengths = input_lengths.to(local_rank, non_blocking=True)
            label_lengths = label_lengths.to(local_rank, non_blocking=True)
            
            # Learning Rate Schedule
            current_lr = get_lr_schedule(global_step, args.warmup_steps, args.lr, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()
            
            # Forward
            outputs, output_lengths = model(spectrograms, input_lengths)
            
            # Loss
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

        # Epoch Summary
        if global_rank == 0:
            avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0.0
            print(f"==> Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")
            print(f"    Eigh Fallbacks (Epoch): {eigh_fallback_handler.count}")
            eigh_fallback_handler.count = 0 # Reset counter

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
    
    # Model Params (Updated for AlgoPerf)
    parser.add_argument('--encoder-dim', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=4, help="AlgoPerf uses 4 layers") 
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--depthwise-kernel-size', type=int, default=31)
    
    # Training Params
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=50, help='Per-GPU batch size (adjusted for A6000)')
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
