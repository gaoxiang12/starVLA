#!/usr/bin/env bash
# E3: isolate two-frame visual history while keeping the baseline 8-step
# action chunk and two future latent targets unchanged.

set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"

export RUN_ID=${RUN_ID:-lewm_oft_libero_dinov2b_ctx2_h8_statecond_200k}
export BASE_WM=${BASE_WM:-facebook/dinov2-base}
export DATA_MIX=${DATA_MIX:-libero_all_wm_ctx2_h8}
export CTX_LEN=2
export N_FUTURE=2
export ACTION_HORIZON=8
export STEPS=${STEPS:-200000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
export WARMUP=${WARMUP:-2000}
export BASE_LR=${BASE_LR:-1e-4}
export ACTION_LR=${ACTION_LR:-1e-4}
export ENCODER_LR=${ENCODER_LR:-1e-6}
export TRAIN_ENCODER=false
export USE_STATE_COND=true
export INCLUDE_STATE=true
export LATENT_STATS_MOMENTUM=${LATENT_STATS_MOMENTUM:-0.99}
export RANDOM_PREFIX_LOSS_WEIGHT=0.0
export FREEZE_MODULES=${FREEZE_MODULES:-backbone.encoder}
export CUDA_DEVS=${CUDA_DEVS:-4,5,6,7}
export NUM_PROCESSES=${NUM_PROCESSES:-4}
export MAIN_PORT=${MAIN_PORT:-29984}
export ACCELERATE_BIN=${ACCELERATE_BIN:-.venv/bin/accelerate}
export WANDB_MODE=${WANDB_MODE:-offline}

exec bash examples/LIBERO/train_files/run_lewm_oft_dino_visual_token_train.sh