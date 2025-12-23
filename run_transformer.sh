#!/bin/bash

# --- 스크립트 설정 ---
# set -e: 명령어가 실패하면 즉시 스크립트를 중단합니다.
# set -x: 실행되는 명령어를 터미널에 출력하여 디버깅을 돕습니다.
set -e
set -x

# --- 사용자 설정 변수 ---
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=disabled
# 사용할 GPU 개수
N_GPUS=2

# 데이터셋을 캐시할 경로 (절대 경로 권장)
DATA_PATH="$HOME/.cache/huggingface/datasets"

# TensorBoard 로그 및 모델 체크포인트를 저장할 기본 경로
OUTPUT_DIR="./training_output_transformer"

# Python 스크립트 파일 이름 (Transformer용으로 변경)
SCRIPT_NAME="Transformer.py"

EPOCHS=100                  
BATCH_SIZE_PER_GPU=256      
WORKERS=6                  

BASE_LR=0.002              
WARMUP_STEPS=4000          
WEIGHT_DECAY=0.0001        
BETA1=0.9                  

EPSILON_PRESET="default"  

RESUME_FROM=""

echo ""
echo "========================================================"
echo "EPSILON CONFIGURATION VERIFICATION"
echo "========================================================"

if [ "$EPSILON_PRESET" == "default" ]; then
    echo "Selected Preset: DEFAULT"
    echo "  - All matrices use same epsilon: 1e-10"
    echo "  - Adaptive Mode: DISABLED"
    EPSILON_DESC="default"
elif [ "$EPSILON_PRESET" == "asymmetric" ]; then
    echo "Selected Preset: ASYMMETRIC"
    echo "  - Different epsilon for L and R matrices"
    echo "  - Adaptive Mode: DISABLED"
    EPSILON_DESC="asymmetric"
else
    echo "ERROR: Invalid EPSILON_PRESET value: $EPSILON_PRESET"
    exit 1
fi

echo "========================================================"
echo ""

# --- 실행 설정 ---
# 로그 및 체크포인트 저장을 위한 디렉토리 생성
RUN_NAME="transformer_shampoo_${EPSILON_DESC}_LR${BASE_LR}_WD${WEIGHT_DECAY}_B1${BETA1}_$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$OUTPUT_DIR/$RUN_NAME/logs"
SAVE_DIR="$OUTPUT_DIR/$RUN_NAME/checkpoints"
mkdir -p $LOG_PATH
mkdir -p $SAVE_DIR

# Resume 옵션 설정
RESUME_OPTION=""
if [ ! -z "$RESUME_FROM" ]; then
    RESUME_OPTION="--resume $RESUME_FROM"
    echo "Resuming training from: $RESUME_FROM"
fi

# Epsilon 관련 옵션
EPSILON_OPTIONS="--epsilon-preset $EPSILON_PRESET"

# --- 분산 학습 실행 ---
echo "========================================================"
echo "WMT14 Transformer-small Training (DryShampoo)"
echo "========================================================"
echo "Hardware Configuration:"
echo "  GPUs: $N_GPUS"
echo "  Batch per GPU: $BATCH_SIZE_PER_GPU"
echo "  Total Batch Size: $(($N_GPUS * $BATCH_SIZE_PER_GPU))"
echo "========================================================"
echo "Optimizer Settings:"
echo "  Learning Rate: $BASE_LR"
echo "  Weight Decay: $WEIGHT_DECAY" 
echo "  Beta1: $BETA1"
echo "  Warmup Steps: $WARMUP_STEPS"
echo "========================================================"
echo "Epsilon Configuration:"
echo "  Preset: $EPSILON_PRESET"
echo "========================================================"
echo "Output Paths:"
echo "  Log Directory: $LOG_PATH"
echo "  Checkpoint Directory: $SAVE_DIR"
echo "========================================================"

# 사용자 확인 대기
echo "Starting training in 3 seconds..."
sleep 3

# 실행 명령어 (ViT 전용 인자 mixup, label-smoothing 제거됨)
torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS $SCRIPT_NAME \
    --data-path $DATA_PATH \
    --log-dir $LOG_PATH \
    --save-dir $SAVE_DIR \
    $RESUME_OPTION \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE_PER_GPU \
    --workers $WORKERS \
    --lr $BASE_LR \
    --warmup-steps $WARMUP_STEPS \
    --weight-decay $WEIGHT_DECAY \
    --beta1 $BETA1 \
    $EPSILON_OPTIONS \
    --log-interval 50 \
    --save-interval 5

echo ""
echo "========================================================"
echo "Training finished successfully."
echo "Results saved to: $OUTPUT_DIR/$RUN_NAME"
echo "========================================================"
