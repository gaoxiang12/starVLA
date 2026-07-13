#!/usr/bin/env bash
# E1.5: adapt the best DINOv2-base LIBERO-10 checkpoint to shorter replanning
# horizons while preserving full 8-step supervision.

set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"

export RUN_ID=${RUN_ID:-lewm_oft_libero_dinov2b_random_prefix_from160k_20k}
export PRETRAINED_CKPT=${PRETRAINED_CKPT:-playground/Checkpoints/lewm_oft_libero_dinov2b_spatial4x4_trainenc1e6_statecond_ema09_200k_fullstate_v2/checkpoints/steps_160000_pytorch_model.pt}
export BASE_WM=${BASE_WM:-facebook/dinov2-base}
export STEPS=${STEPS:-20000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
export WARMUP=${WARMUP:-500}
export BASE_LR=${BASE_LR:-1e-5}
export ACTION_LR=${ACTION_LR:-1e-5}
export RANDOM_PREFIX_LOSS_WEIGHT=${RANDOM_PREFIX_LOSS_WEIGHT:-1.0}
export ACTION_WEIGHT=${ACTION_WEIGHT:-0.0}
export DELTA_WEIGHT=${DELTA_WEIGHT:-0.0}
export TOKEN_DIVERSITY_WEIGHT=${TOKEN_DIVERSITY_WEIGHT:-0.0}
export TOKEN_VARIANCE_WEIGHT=${TOKEN_VARIANCE_WEIGHT:-0.0}
export TRAIN_ENCODER=false
export USE_STATE_COND=true
export INCLUDE_STATE=true
export FREEZE_MODULES=${FREEZE_MODULES:-backbone,world_model,visual_token_pooler,task_embedding,action_query_proj,future_action_context_proj}
export IS_RESUME=false
export CUDA_DEVS=${CUDA_DEVS:-0,1,2,3}
export NUM_PROCESSES=${NUM_PROCESSES:-4}
export MAIN_PORT=${MAIN_PORT:-29980}
export ACCELERATE_BIN=${ACCELERATE_BIN:-.venv/bin/accelerate}
export WANDB_MODE=${WANDB_MODE:-offline}

exec bash examples/LIBERO/train_files/run_lewm_oft_dino_visual_token_train.sh