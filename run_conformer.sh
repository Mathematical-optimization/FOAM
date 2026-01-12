#!/bin/bash

# --- 스크립트 설정 ---
# set -e: 명령어가 실패하면 즉시 스크립트를 중단합니다. (첫 실험 실패 시 이후 실험 중단됨)
# set -x: 실행되는 명령어를 터미널에 출력하여 디버깅을 돕습니다.
set -e
set -x

# --- 사용자 설정 변수 ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS=4

# WandB 설정
export WANDB_MODE=online
# API Key는 보안상 별도 관리하거나 환경변수로 빼는 것이 좋지만, 여기서는 그대로 유지
export WANDB_API_KEY="37d143039a17dfe05b4e0e5314eccaae6016efd0"

# 경로 설정
DATA_PATH="/workspace/datasets"
SPM_MODEL_PATH="spm_librispeech_1024.model"
OUTPUT_DIR="./training_output_conformer"
SCRIPT_NAME="conformer.py"

# --- 고정 하이퍼파라미터 (모든 실험 공통) ---
EPOCHS=90
BATCH_SIZE_PER_GPU=256
WORKERS=4
LR=0.002
WARMUP_STEPS=1250
WEIGHT_DECAY=5e-5
BETA1=0.975
BETA2=0.99
MAX_PRECOND_DIM=1024
PRECOND_FREQ=50
START_PRECOND_STEP=50

# --- 실험 계획 (Matrix Root Inv Threshold, Max Epsilon) ---
EXPERIMENTS=(
    "0.4 3e-7"
    "0.4 5e-7"
    "0.4 7e-7"
)

# ========================================================
# 실험 루프 시작
# ========================================================

EXP_COUNT=0

for config in "${EXPERIMENTS[@]}"; do
    # 공백을 기준으로 문자열 분리
    read -r THRESH EPS <<< "$config"
    EXP_COUNT=$((EXP_COUNT+1))

    # 실행 이름 동적 생성 (파라미터 포함)
    RUN_ID="Exp${EXP_COUNT}_Thresh${THRESH}_Eps${EPS}"
    RUN_NAME="conformer_shampoo_${RUN_ID}_$(date +%Y%m%d_%H%M)"
    CHECKPOINT_DIR="$OUTPUT_DIR/$RUN_NAME/checkpoints"

    # 디렉토리 생성
    mkdir -p "$CHECKPOINT_DIR"

    echo ""
    echo "################################################################"
    echo "Starting Experiment $EXP_COUNT / ${#EXPERIMENTS[@]}"
    echo "RUN_NAME: $RUN_NAME"
    echo "Config => Matrix Threshold: $THRESH, Max Epsilon: $EPS"
    echo "################################################################"
    echo ""

    # --- 학습 실행 ---
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
        --matrix-root-inv-threshold $THRESH \
        --max-epsilon $EPS \
        --run-name "$RUN_NAME" \
        --log-interval 50 \
        --save-interval 5

    echo "----------------------------------------------------------------"
    echo "Experiment $EXP_COUNT finished. Results in $CHECKPOINT_DIR"
    echo "Sleeping for 30 seconds to clear GPU memory and sync WandB..."
    echo "----------------------------------------------------------------"
    
    # 다음 실험을 위해 잠시 대기 (GPU 메모리 정리 및 WandB 업로드 시간 확보)
    sleep 15

done

echo ""
echo "========================================================"
echo "All $EXP_COUNT experiments completed successfully."
echo "========================================================"
