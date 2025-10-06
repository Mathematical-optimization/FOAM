# Transformer.py (최종 개선 버전)

import os
import math
import argparse
import functools
from typing import Optional, List, Dict, Any
import time
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Hugging Face 라이브러리
from datasets import load_dataset
from transformers import AutoTokenizer

# Shampoo optimizer
from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    DDPShampooConfig,
    CommunicationDType
)

# 평가 메트릭
import sacrebleu
from tqdm import tqdm


# ============= Eigh Fallback Counter =============

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


# ============= Transformer 모델 구현 =============

class PositionalEncoding(nn.Module):
    """Positional Encoding with sinusoidal functions"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [seq_len, batch_size, d_model]"""
        return x + self.pe[:x.size(0)]


class CustomMultiheadAttention(nn.Module):
    """Multi-Head Attention with separate Q,K,V projections"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len, batch_size = query.size(0), query.size(1)
        Q = self.q_proj(query).view(seq_len, batch_size, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(-1, batch_size, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(-1, batch_size, self.n_heads, self.d_k).transpose(1, 2)
        Q, K, V = Q.transpose(0, 2), K.transpose(0, 2), V.transpose(0, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        context = context.transpose(0, 1)
        return self.out_proj(context)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ffn(x2))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, n_heads, dropout)
        self.cross_attn = CustomMultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2, tgt_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(x2, encoder_output, encoder_output, src_mask))
        x2 = self.norm3(x)
        x = x + self.dropout3(self.ffn(x2))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_encoder_layers: int,
                 n_decoder_layers: int, d_ff: int, max_seq_len: int, dropout: float, pad_idx: int):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.encoder_pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.decoder_pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_decoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return ~mask

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.shared_embedding(src) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.encoder_pos_encoding(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x)

    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.shared_embedding(tgt) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.decoder_pos_encoding(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return self.decoder_norm(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.create_padding_mask(src)
        tgt_seq_len = tgt.size(1)
        tgt_pad_mask = self.create_padding_mask(tgt)
        tgt_la_mask = self.create_look_ahead_mask(tgt_seq_len, tgt.device)
        tgt_mask = tgt_pad_mask & tgt_la_mask.unsqueeze(0).unsqueeze(0)
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        return F.linear(decoder_output.transpose(0, 1), self.shared_embedding.weight)


# ============= 데이터 처리 =============

class WMT17Dataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]['translation']
        src_text, tgt_text = item['de'], item['en']
        src_encoding = self.tokenizer(src_text, max_length=self.max_seq_len, padding='max_length',
                                     truncation=True, return_tensors='pt')
        tgt_encoding = self.tokenizer(tgt_text, max_length=self.max_seq_len, padding='max_length',
                                     truncation=True, return_tensors='pt')
        return {'src': src_encoding['input_ids'].squeeze(0), 'tgt': tgt_encoding['input_ids'].squeeze(0),
                'src_text': src_text, 'tgt_text': tgt_text}


def create_data_loaders(args, tokenizer, global_rank, world_size):
    if global_rank == 0:
        print("Loading WMT datasets...")
    dataset = load_dataset("wmt17", "de-en", cache_dir=args.data_path)
    val_dataset_raw = load_dataset("wmt14", "de-en", cache_dir=args.data_path)
    train_data = dataset['train']
    if args.max_train_samples:
        train_data = train_data.select(range(min(args.max_train_samples, len(train_data))))
    train_dataset = WMT17Dataset(train_data, tokenizer, args.max_seq_len)
    val_dataset = WMT17Dataset(val_dataset_raw['validation'], tokenizer, args.max_seq_len)
    if global_rank == 0:
        print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                            num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                          num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader


# ============= 학습 함수 =============

def label_smoothed_cross_entropy(logits, targets, epsilon=0.1, ignore_index=-100):
    """Label smoothing cross entropy loss"""
    vocab_size = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    smooth_target = torch.full_like(log_probs, epsilon / (vocab_size - 1))
    smooth_target.scatter_(-1, targets.unsqueeze(-1), 1.0 - epsilon)
    mask = targets != ignore_index
    smooth_target.masked_fill_(~mask.unsqueeze(-1), 0)
    loss = -(smooth_target * log_probs).sum(dim=-1)
    return loss.masked_select(mask).mean()


def train_epoch(model, dataloader, optimizer, device, args, global_step, writer, global_rank):
    """Training epoch"""
    model.train()
    total_loss, total_tokens = 0, 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {args.current_epoch+1} Training",
                       disable=(dist.get_rank() != 0))

    for batch_idx, batch in enumerate(progress_bar):
        global_step += 1
        lr = get_lr_schedule(global_step, args.warmup_steps, args.d_model)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        src, tgt = batch['src'].to(device), batch['tgt'].to(device)
        tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        output = model(src, tgt_input)
        loss = label_smoothed_cross_entropy(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1),
            epsilon=args.label_smoothing,
            ignore_index=tokenizer.pad_token_id
        )
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        
        with torch.no_grad():
            num_tokens = (tgt_output != tokenizer.pad_token_id).sum().item()
            loss_value = loss.item()
            total_loss += loss_value * num_tokens
            total_tokens += num_tokens

        # Condition number 로깅
        if global_rank == 0 and batch_idx % args.log_condition_interval == 0:
            log_condition_numbers(optimizer, writer, global_step)

        if dist.get_rank() == 0:
            progress_bar.set_postfix({'loss': loss_value, 'lr': lr})

    return total_loss / total_tokens if total_tokens > 0 else 0, global_step


def evaluate(model, dataloader, device, tokenizer):
    """Evaluation"""
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0)):
            src, tgt = batch['src'].to(device), batch['tgt'].to(device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            output = model(src, tgt_input)
            loss = F.cross_entropy(
                output.reshape(-1, output.size(-1)),
                tgt_output.reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )

            num_tokens = (tgt_output != tokenizer.pad_token_id).sum().item()
            loss_value = loss.item()
            total_loss += loss_value * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0


def greedy_decode(model, src, max_len, device, tokenizer):
    """Greedy decoding for translation"""
    model.eval()
    with torch.no_grad():
        src_mask = model.module.create_padding_mask(src)
        encoder_output = model.module.encode(src, src_mask)
        tgt = torch.full((src.size(0), 1), tokenizer.cls_token_id, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_seq_len = tgt.size(1)
            tgt_pad_mask = model.module.create_padding_mask(tgt)
            tgt_la_mask = model.module.create_look_ahead_mask(tgt_seq_len, device)
            tgt_mask = tgt_pad_mask & tgt_la_mask.unsqueeze(0).unsqueeze(0)
            decoder_output = model.module.decode(tgt, encoder_output, tgt_mask, src_mask)
            logits = F.linear(decoder_output[-1, :, :], model.module.shared_embedding.weight)
            next_token = logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if (next_token == tokenizer.sep_token_id).all():
                break
    return tgt


def compute_bleu(model, dataloader, device, tokenizer, max_decode_len):
    """Compute BLEU score"""
    model.eval()
    hypotheses, references = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing BLEU", disable=(dist.get_rank() != 0)):
            src, tgt_texts = batch['src'].to(device), batch['tgt_text']
            translations = greedy_decode(model, src, max_decode_len, device, tokenizer)
            for i in range(translations.size(0)):
                hyp_ids = [tid for tid in translations[i].cpu().tolist()
                          if tid not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]]
                hypotheses.append(tokenizer.decode(hyp_ids))
                references.append(tgt_texts[i])
    return sacrebleu.corpus_bleu(hypotheses, [references]).score if hypotheses else 0.0


def get_lr_schedule(step, warmup_steps, d_model):
    """Transformer learning rate schedule"""
    step = max(1, step)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)


def log_condition_numbers(optimizer, writer, global_step):
    """Log condition numbers of Shampoo preconditioners"""
    try:
        if hasattr(optimizer, '_per_group_state_lists'):
            for group_idx, state_lists in enumerate(optimizer._per_group_state_lists):
                if 'SHAMPOO_PRECONDITIONER_LIST' in state_lists:
                    preconditioner_list = state_lists['SHAMPOO_PRECONDITIONER_LIST']

                    for block_idx, kronecker_factors in enumerate(
                        preconditioner_list._masked_kronecker_factors_list[:10]
                    ):
                        for factor_idx, factor_matrix in enumerate(kronecker_factors.factor_matrices):
                            if factor_matrix.shape[0] == factor_matrix.shape[1] and factor_matrix.shape[0] > 1:
                                try:
                                    cond_num = torch.linalg.cond(factor_matrix.double()).item()
                                    if not math.isnan(cond_num) and not math.isinf(cond_num):
                                        writer.add_scalar(
                                            f'ConditionNumber/Group{group_idx}/Block{block_idx}/Factor{factor_idx}',
                                            cond_num,
                                            global_step
                                        )
                                except Exception:
                                    pass
    except Exception as e:
        print(f"Warning: Failed to log condition numbers: {e}")


def gather_optimizer_state_from_all_ranks(optimizer, model, world_size):
    """모든 랭크에서 옵티마이저 상태를 수집하고 통합"""
    local_state = optimizer.distributed_state_dict(
        key_to_param=model.module.named_parameters()
    )

    all_states = [None] * world_size
    dist.all_gather_object(all_states, local_state)

    if dist.get_rank() == 0:
        merged_state = {
            'state': {},
            'param_groups': all_states[0]['param_groups']
        }

        all_param_keys = set()
        for state in all_states:
            if state and 'state' in state:
                all_param_keys.update(state['state'].keys())

        for param_key in all_param_keys:
            merged_state['state'][param_key] = {}
            
            all_state_keys = set()
            for state in all_states:
                if state and 'state' in state and param_key in state['state']:
                    all_state_keys.update(state['state'][param_key].keys())

            for state_key in all_state_keys:
                for state in all_states:
                    if state and 'state' in state and param_key in state['state'] and state_key in state['state'][param_key]:
                        value = state['state'][param_key][state_key]
                        if isinstance(value, torch.Tensor):
                            if value.numel() > 0:
                                merged_state['state'][param_key][state_key] = value
                                break
                        else:
                            merged_state['state'][param_key][state_key] = value
                            break
        return merged_state
        
    return None


def save_checkpoint(model, optimizer, epoch, step, best_bleu, checkpoint_path, 
                   global_rank, world_size, args, cumulative_fallback_count):
    """체크포인트 저장"""
    merged_optimizer_state = gather_optimizer_state_from_all_ranks(optimizer, model, world_size)

    if global_rank == 0 and merged_optimizer_state is not None:
        actual_model = model.module if hasattr(model, 'module') else model

        shampoo_config = {
            'beta2': args.beta2,
            'epsilon': 1e-8,
            'max_preconditioner_dim': args.max_preconditioner_dim,
            'precondition_frequency': args.precondition_frequency,
            'start_preconditioning_step': args.start_preconditioning_step,
        }

        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': actual_model.state_dict(),
            'optimizer_state_dict': merged_optimizer_state,
            'best_bleu': best_bleu,
            'cumulative_fallback_count': cumulative_fallback_count,
            'shampoo_config': shampoo_config,
            'args': vars(args),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        print(f"  Cumulative FP32→FP64 fallbacks: {cumulative_fallback_count}")


def setup():
    """분산 학습 초기화"""
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    """분산 학습 종료"""
    dist.destroy_process_group()


def main(args):
    local_rank = setup()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Running DDP on rank {global_rank}/{world_size}, local rank {local_rank}")

    # Eigh fallback 카운터 설정
    eigh_fallback_handler = EighFallbackCounter()
    matrix_logger = logging.getLogger('optimizers.matrix_functions')
    matrix_logger.setLevel(logging.WARNING)
    matrix_logger.addHandler(eigh_fallback_handler)
    
    # 누적 카운트 추적
    cumulative_fallback_count = 0
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir) if global_rank == 0 else None

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    train_loader, val_loader = create_data_loaders(args, tokenizer, global_rank, world_size)

    model = Transformer(
        vocab_size=len(tokenizer), d_model=args.d_model, n_heads=args.n_heads,
        n_encoder_layers=args.n_layers, n_decoder_layers=args.n_layers, d_ff=args.d_ff,
        max_seq_len=args.max_seq_len, dropout=args.dropout,
        pad_idx=tokenizer.pad_token_id
    ).to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    optimizer = DistributedShampoo(
        model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
        epsilon=1e-8, weight_decay=args.weight_decay,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        use_decoupled_weight_decay=True,
        grafting_config=AdamGraftingConfig(beta2=args.beta2, epsilon=1e-7),
        distributed_config=DDPShampooConfig(
            communication_dtype=CommunicationDType.FP32,
            num_trainers_per_group=-1,
            communicate_params=False
        ),
        use_protected_eigh=True
    )

    global_step, best_bleu = 0, 0
    
    # 학습 루프
    for epoch in range(args.epochs):
        args.current_epoch = epoch
        
        # 에폭별 카운트를 위해 리셋
        eigh_fallback_handler.count = 0
        
        if global_rank == 0:
            print(f"\n{'='*80}\nEpoch {epoch+1}/{args.epochs}\n{'='*80}")
        
        train_loader.sampler.set_epoch(epoch)

        start_time = time.time()
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, local_rank, args, global_step, writer, global_rank
        )
        train_time = time.time() - start_time
        
        val_loss = evaluate(model, val_loader, device=local_rank, tokenizer=tokenizer)
        
        bleu_score = 0.0
        if (epoch + 1) % args.bleu_interval == 0:
            bleu_score = compute_bleu(model, val_loader, local_rank, tokenizer, args.max_seq_len)
        
        # Fallback 카운트 집계
        local_fallback_count = torch.tensor(eigh_fallback_handler.count).to(local_rank)
        epoch_fallback_count = local_fallback_count.clone()
        dist.all_reduce(epoch_fallback_count, op=dist.ReduceOp.SUM)
        
        # 누적 카운트 업데이트
        cumulative_fallback_count += epoch_fallback_count.item()
        
        # 메트릭 집계
        metrics_tensor = torch.tensor([train_loss, val_loss, bleu_score], device=local_rank)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        metrics_tensor /= world_size
        train_loss, val_loss, bleu_score = metrics_tensor.tolist()

        # 결과 출력 및 로깅
        if global_rank == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch [{epoch+1}/{args.epochs}] Summary:")
            print(f"  Training Loss: {train_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  BLEU Score: {bleu_score:.2f}")
            print(f"  Learning Rate: {lr:.6f}")
            print(f"  Training Time: {train_time:.1f}s")
            print(f"  Epoch FP32→FP64 Fallbacks: {epoch_fallback_count.item()}")
            print(f"  Cumulative Fallbacks: {cumulative_fallback_count}")
            print(f"{'='*80}\n")
            
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/validation', val_loss, epoch)
                writer.add_scalar('Diagnostics/Eigh_Fallback_Per_Epoch', epoch_fallback_count.item(), epoch)
                writer.add_scalar('Diagnostics/Eigh_Fallback_Cumulative', cumulative_fallback_count, epoch)
                if (epoch + 1) % args.bleu_interval == 0:
                    writer.add_scalar('Performance/BLEU', bleu_score, epoch)
                writer.add_scalar('LearningRate/lr', lr, global_step)

        # Best model 저장
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            if global_rank == 0:
                print(f"*** New best BLEU: {best_bleu:.2f} ***")
            save_checkpoint(model, optimizer, epoch, global_step, best_bleu,
                          os.path.join(args.checkpoint_dir, "best_model.pth"),
                          global_rank, world_size, args, cumulative_fallback_count)

        # Condition analysis 체크포인트
        if (epoch + 1) % args.condition_analysis_interval == 0:
            save_checkpoint(model, optimizer, epoch, global_step, best_bleu,
                          os.path.join(args.checkpoint_dir, f"condition_epoch_{epoch+1}.pth"),
                          global_rank, world_size, args, cumulative_fallback_count)

        # 정기 체크포인트
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, global_step, best_bleu,
                          os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"),
                          global_rank, world_size, args, cumulative_fallback_count)

    # 학습 완료
    if global_rank == 0:
        print(f"\n{'='*80}")
        print(f"Training completed!")
        print(f"  Best BLEU score: {best_bleu:.2f}")
        print(f"  Total FP32→FP64 fallbacks: {cumulative_fallback_count}")
        print(f"{'='*80}\n")
        if writer:
            writer.add_text('Summary/Best_BLEU', str(best_bleu))
            writer.add_text('Summary/Total_Fallbacks', str(cumulative_fallback_count))
            
    if writer:
        writer.close()
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlgoPerf Transformer + WMT with Distributed Shampoo')

    # Model arguments
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of encoder/decoder layers')
    parser.add_argument('--d-ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--max-seq-len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.0013, help='Base learning rate')
    parser.add_argument('--warmup-steps', type=int, default=26400, help='Warmup steps')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.99, help='Adam beta2')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')

    # Shampoo arguments
    parser.add_argument('--max-preconditioner-dim', type=int, default=1024,
                       help='Max preconditioner dimension')
    parser.add_argument('--precondition-frequency', type=int, default=20,
                       help='Preconditioning frequency')
    parser.add_argument('--start-preconditioning-step', type=int, default=20,
                       help='Step to start preconditioning')

    # Data arguments
    parser.add_argument('--data-path', type=str, default='./wmt_data',
                       help='Path to cache datasets')
    parser.add_argument('--workers', type=int, default=6, help='Number of data workers')
    parser.add_argument('--max-train-samples', type=int, default=None,
                       help='Max training samples for debugging')

    # Logging arguments
    parser.add_argument('--log-dir', type=str, default='logs/transformer_shampoo',
                       help='TensorBoard log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/transformer_shampoo',
                       help='Checkpoint directory')
    parser.add_argument('--save-interval', type=int, default=1, help='Save checkpoint interval')
    parser.add_argument('--bleu-interval', type=int, default=5, help='BLEU evaluation interval')
    parser.add_argument('--condition-analysis-interval', type=int, default=1,
                       help='Interval for condition number analysis checkpoints')
    parser.add_argument('--log-condition-interval', type=int, default=100,
                       help='Interval for logging condition numbers')

    args = parser.parse_args()
    main(args)
