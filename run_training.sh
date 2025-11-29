#!/bin/bash

# --- 스크립트 설정 ---
# set -e: 명령어가 실패하면 즉시 스크립트를 중단합니다.
# set -x: 실행되는 명령어를 터미널에 출력하여 디버깅을 돕습니다.
set -e
set -x

# --- 사용자 설정 변수 ---
# 이 부분의 값들을 필요에 맞게 수정하여 사용하세요. (하이퍼파라미터 튜닝 시 이 부분을 변경)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 사용할 GPU 개수 (시스템에 맞게 설정)
N_GPUS=4

# 데이터셋을 캐시할 경로 (절대 경로 권장)
DATA_PATH="$HOME/.cache/huggingface/datasets"

# TensorBoard 로그 및 모델 체크포인트를 저장할 기본 경로
OUTPUT_DIR="./training_output_1128"

# Python 스크립트 파일 이름
SCRIPT_NAME="vit.py"

# 학습 기본 하이퍼파라미터
EPOCHS=90
BATCH_SIZE_PER_GPU=256 # GPU 메모리에 맞춰 조절
WORKERS=4              # 데이터 로딩에 사용할 CPU 워커 수

# 옵티마이저 및 스케줄러 하이퍼파라미터
BASE_LR=0.00145
WARMUP_STEPS=5634
WEIGHT_DECAY=0.0005
BETA1=0.95

# 데이터 증강 하이퍼파라미터
MIXUP=0.2
LABEL_SMOOTHING=0.1

# ========================================
# EPSILON 설정 (주요 실험 변수)
# ========================================

# Epsilon 프리셋 선택
# 옵션: 'default' 또는 'asymmetric'만 사용
EPSILON_PRESET="default"  # 'default' 또는 'asymmetric' 중 선택

# 체크포인트에서 재개 (필요시 설정, 빈 값이면 새로 시작)
RESUME_FROM=""

# --- Epsilon 설정 검증 및 출력 ---
echo ""
echo "========================================================"
echo "EPSILON CONFIGURATION VERIFICATION"
echo "========================================================"

if [ "$EPSILON_PRESET" == "default" ]; then
    echo "Selected Preset: DEFAULT"
    echo "  - All matrices use same epsilon: 1e-10"
    echo "  - Non-adaptive (fixed epsilon)"
    echo "  - Left Matrix Epsilon: 1e-08"
    echo "  - Right Matrix Epsilon: 1e-08"
    echo "  - Adaptive Mode: DISABLED"
    EPSILON_DESC="default"
elif [ "$EPSILON_PRESET" == "asymmetric" ]; then
    echo "Selected Preset: ASYMMETRIC"
    echo "  - Different epsilon for L and R matrices"
    echo "  - Non-adaptive (fixed epsilon)"
    echo "  - Left Matrix Epsilon: 1e-08"
    echo "  - Right Matrix Epsilon: 5e-05"
    echo "  - Adaptive Mode: DISABLED"
    EPSILON_DESC="asymmetric"
else
    echo "ERROR: Invalid EPSILON_PRESET value: $EPSILON_PRESET"
    echo "Please use either 'default' or 'asymmetric'"
    exit 1
fi

echo "========================================================"
echo ""

# --- 실행 설정 ---
# 로그 및 체크포인트 저장을 위한 디렉토리 생성
RUN_NAME="vit_shampoo_${EPSILON_DESC}_LR${BASE_LR}_WD${WEIGHT_DECAY}_B1${BETA1}_$(date +%Y%m%d_%H%M%S)"
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

# Epsilon 관련 옵션 구성 (단순화)
EPSILON_OPTIONS="--epsilon-preset $EPSILON_PRESET"

# --- 분산 학습 실행 ---
echo "========================================================"
echo "Vision Transformer on ImageNet-1k Training (Algoperf Spec)"
echo "Optimizer: Distributed Shampoo"
echo "========================================================"
echo "Hardware Configuration:"
echo "  GPUs: $N_GPUS"
echo "  Total Batch Size: $(($N_GPUS * $BATCH_SIZE_PER_GPU))"
echo "========================================================"
echo "Optimizer Settings:"
echo "  Learning Rate: $BASE_LR"
echo "  Weight Decay: $WEIGHT_DECAY" 
echo "  Beta1 (momentum): $BETA1"
echo "  Beta2 (Shampoo): 0.95 (default)"
echo "========================================================"
echo "Epsilon Configuration:"
echo "  Preset: $EPSILON_PRESET"
if [ "$EPSILON_PRESET" == "default" ]; then
    echo "  ├─ Epsilon (all dims): 1e-10"
    echo "  └─ Adaptive: NO"
elif [ "$EPSILON_PRESET" == "asymmetric" ]; then
    echo "  ├─ Epsilon Left: 1e-08"
    echo "  ├─ Epsilon Right: 5e-05"
    echo "  └─ Adaptive: NO"
fi
echo "========================================================"
echo "Data Augmentations:"
echo "  RandAugment: m15-n2"
echo "  Mixup Alpha: $MIXUP"
echo "  Label Smoothing: $LABEL_SMOOTHING"
echo "========================================================"
echo "Output Paths:"
echo "  Log Directory: $LOG_PATH"
echo "  Checkpoint Directory: $SAVE_DIR"
echo "  Run Name: $RUN_NAME"
echo "========================================================"

# Python 스크립트에 전달되는 실제 인자 출력
echo ""
echo "PYTHON SCRIPT ARGUMENTS:"
echo "------------------------"
echo "torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS $SCRIPT_NAME \\"
echo "    --data-path $DATA_PATH \\"
echo "    --log-dir $LOG_PATH \\"
echo "    --save-dir $SAVE_DIR \\"
if [ ! -z "$RESUME_FROM" ]; then
    echo "    --resume $RESUME_FROM \\"
fi
echo "    --epochs $EPOCHS \\"
echo "    --batch-size $BATCH_SIZE_PER_GPU \\"
echo "    --workers $WORKERS \\"
echo "    --base-lr $BASE_LR \\"
echo "    --warmup-steps $WARMUP_STEPS \\"
echo "    --weight-decay $WEIGHT_DECAY \\"
echo "    --beta1 $BETA1 \\"
echo "    --mixup $MIXUP \\"
echo "    --label-smoothing $LABEL_SMOOTHING \\"
echo "    --epsilon-preset $EPSILON_PRESET \\"
echo "    --log-interval 200 \\"
echo "    --save-interval 10"
echo "========================================================"
echo ""

# 사용자 확인 대기 (선택사항)
echo "Starting training in 3 seconds..."
echo "Press Ctrl+C to cancel"
sleep 3

# 실행 명령어
torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS $SCRIPT_NAME \
    --data-path $DATA_PATH \
    --log-dir $LOG_PATH \
    --save-dir $SAVE_DIR \
    $RESUME_OPTION \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE_PER_GPU \
    --workers $WORKERS \
    --base-lr $BASE_LR \
    --warmup-steps $WARMUP_STEPS \
    --weight-decay $WEIGHT_DECAY \
    --beta1 $BETA1 \
    --mixup $MIXUP \
    --label-smoothing $LABEL_SMOOTHING \
    $EPSILON_OPTIONS \
    --log-interval 200 \
    --save-interval 10

echo ""
echo "========================================================"
echo "Training finished successfully."
echo "Results saved to: $OUTPUT_DIR/$RUN_NAME"
echo "========================================================"
