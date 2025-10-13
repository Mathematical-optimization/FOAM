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
OUTPUT_DIR="./training_output_1013"

# Python 스크립트 파일 이름
SCRIPT_NAME="vit.py"

# 학습 기본 하이퍼파라미터
EPOCHS=90
BATCH_SIZE_PER_GPU=256 # GPU 메모리에 맞춰 조절
WORKERS=4              # 데이터 로딩에 사용할 CPU 워커 수

# 옵티마이저 및 스케줄러 하이퍼파라미터
BASE_LR=0.0013
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
# 옵션: 'default', 'asymmetric'
EPSILON_PRESET="default"

# 프리셋별 설명:
# - default: 모든 matrix에 동일한 epsilon (1e-10) 사용, 비적응형
# - asymmetric: L과 R matrix에 서로 다른 epsilon 사용 (L:1e-8, R:5e-5), 비적응형
# - adaptive: 조건수 기반 적응형 epsilon, L과 R 동일
# - adaptive_asymmetric: L과 R에 서로 다른 epsilon + 조건수 기반 적응형                   
# - custom: 아래 개별 설정 사용

# Custom 설정 시 사용할 값들 (EPSILON_PRESET='custom'일 때만 적용)
EPSILON_BASE=1e-8           # 기본 epsilon (1D tensor용)
EPSILON_LEFT=1e-8            # L matrix epsilon (비워두면 EPSILON_BASE 사용)
EPSILON_RIGHT=5e-5           # R matrix epsilon (비워두면 EPSILON_BASE 사용)
USE_ADAPTIVE_EPSILON=true   # true/false - 적응형 epsilon 사용 여부
CONDITION_THRESHOLDS="5e7:5e-4"  # 조건수:epsilon 매핑

# 체크포인트에서 재개 (필요시 설정, 빈 값이면 새로 시작)
RESUME_FROM=""

# --- 실행 설정 ---
# 로그 및 체크포인트 저장을 위한 디렉토리 생성
# RUN_NAME에 epsilon 설정 정보 포함
if [ "$EPSILON_PRESET" == "custom" ]; then
    EPSILON_DESC="custom_${EPSILON_LEFT}_${EPSILON_RIGHT}"
    if [ "$USE_ADAPTIVE_EPSILON" == "true" ]; then
        EPSILON_DESC="${EPSILON_DESC}_adaptive"
    fi
else
    EPSILON_DESC="${EPSILON_PRESET}"
fi

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

# Epsilon 관련 옵션 구성
EPSILON_OPTIONS="--epsilon-preset $EPSILON_PRESET"

if [ "$EPSILON_PRESET" == "custom" ]; then
    EPSILON_OPTIONS="$EPSILON_OPTIONS --epsilon $EPSILON_BASE"
    
    if [ ! -z "$EPSILON_LEFT" ]; then
        EPSILON_OPTIONS="$EPSILON_OPTIONS --epsilon-left $EPSILON_LEFT"
    fi
    
    if [ ! -z "$EPSILON_RIGHT" ]; then
        EPSILON_OPTIONS="$EPSILON_OPTIONS --epsilon-right $EPSILON_RIGHT"
    fi
    
    if [ "$USE_ADAPTIVE_EPSILON" == "true" ]; then
        EPSILON_OPTIONS="$EPSILON_OPTIONS --use-adaptive-epsilon"
        
        if [ ! -z "$CONDITION_THRESHOLDS" ]; then
            EPSILON_OPTIONS="$EPSILON_OPTIONS --condition-thresholds $CONDITION_THRESHOLDS"
        fi
    fi
fi

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
echo "  LR: $BASE_LR, WD: $WEIGHT_DECAY, Beta1: $BETA1"
echo "  Epsilon Preset: $EPSILON_PRESET"
if [ "$EPSILON_PRESET" == "custom" ]; then
    echo "  Custom Epsilon Settings:"
    echo "    Base: $EPSILON_BASE"
    echo "    Left: ${EPSILON_LEFT:-same as base}"
    echo "    Right: ${EPSILON_RIGHT:-same as base}"
    echo "    Adaptive: $USE_ADAPTIVE_EPSILON"
    if [ "$USE_ADAPTIVE_EPSILON" == "true" ]; then
        echo "    Thresholds: $CONDITION_THRESHOLDS"
    fi
fi
echo "========================================================"
echo "Augmentations:"
echo "  RandAugment(m15-n2), Mixup($MIXUP), LS($LABEL_SMOOTHING)"
echo "========================================================"
echo "Output:"
echo "  Log Path: $LOG_PATH"
echo "  Checkpoint Path: $SAVE_DIR"
echo "========================================================"

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

echo "Training finished successfully."
