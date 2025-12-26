#!/bin/bash

# --- 스크립트 설정 ---
# set -e: 명령어가 실패하면 즉시 스크립트를 중단합니다.
# set -x: 실행되는 명령어를 터미널에 출력하여 디버깅을 돕습니다.
set -e
set -x

# --- 사용자 설정 변수 ---
# 사용할 GPU ID 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 사용할 GPU 개수
N_GPUS=4

# WandB 설정 (필요시 수정)
export WANDB_MODE=online
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"

# 경로 설정
# 데이터셋 경로 (LibriSpeech 데이터가 저장될 위치)
DATA_PATH="/workspace/datasets"
# SentencePiece 모델 경로 (conformer.py 실행 전 생성 필요)
SPM_MODEL_PATH="spm_librispeech_1024.model"

# 로그 및 체크포인트를 저장할 기본 경로
OUTPUT_DIR="./training_output_conformer"

# Python 스크립트 파일 이름
SCRIPT_NAME="conformer.py"

# 학습 기본 하이퍼파라미터 (Conformer Small 설정 기준)
EPOCHS=90
BATCH_SIZE_PER_GPU=256  # GPU 메모리에 맞춰 조절
WORKERS=4              # 데이터 로딩 워커 수

# 옵티마이저 및 스케줄러 하이퍼파라미터
LR=0.001
WARMUP_STEPS=1250
WEIGHT_DECAY=5e-5
BETA1=0.975
BETA2=0.99

# Shampoo 하이퍼파라미터
MAX_PRECOND_DIM=1024
PRECOND_FREQ=50
START_PRECOND_STEP=50

# 실행 이름 설정 (날짜 포함)
RUN_NAME="conformer_shampoo_LR${LR}_WD${WEIGHT_DECAY}_B1${BETA1}_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="$OUTPUT_DIR/$RUN_NAME/checkpoints"

# 디렉토리 생성
mkdir -p "$CHECKPOINT_DIR"

# --- 정보 출력 ---
echo "========================================================"
echo "Conformer Small (LibriSpeech) Training with Distributed Shampoo"
echo "========================================================"
echo "Hardware Configuration:"
echo "  GPUs: $N_GPUS"
echo "  Batch per GPU: $BATCH_SIZE_PER_GPU"
echo "  Total Batch Size: $(($N_GPUS * $BATCH_SIZE_PER_GPU))"
echo "========================================================"
echo "Optimizer Settings:"
echo "  Learning Rate: $LR"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Betas: ($BETA1, $BETA2)"
echo "  Warmup Steps: $WARMUP_STEPS"
echo "========================================================"
echo "Shampoo Settings:"
echo "  Max Precond Dim: $MAX_PRECOND_DIM"
echo "  Precond Frequency: $PRECOND_FREQ"
echo "  Start Precond Step: $START_PRECOND_STEP"
echo "========================================================"
echo "Paths:"
echo "  Data Path: $DATA_PATH"
echo "  SPM Model: $SPM_MODEL_PATH"
echo "  Checkpoint Dir: $CHECKPOINT_DIR"
echo "========================================================"

# 사용자 확인 대기
echo "Starting training in 3 seconds..."
sleep 3

# --- 학습 실행 ---
# torchrun을 사용하여 분산 학습 시작
torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS $SCRIPT_NAME \
    --data-path "$DATA_PATH" \
    --spm-model-path "$SPM_MODEL_PATH" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE_PER_GPU \
    --workers $WORKERS \
    --lr $LR \
    --warmup-steps $WARMUP_STEPS \
    --weight-decay $WEIGHT_DECAY \
    --beta1 $BETA1 \
    --beta2 $BETA2 \
    --max-preconditioner-dim $MAX_PRECOND_DIM \
    --precondition-frequency $PRECOND_FREQ \
    --start-preconditioning-step $START_PRECOND_STEP \
    --run-name "$RUN_NAME" \
    --log-interval 50 \
    --save-interval 5

echo ""
echo "========================================================"
echo "Training finished successfully."
echo "Results saved to: $CHECKPOINT_DIR"
echo "========================================================"
