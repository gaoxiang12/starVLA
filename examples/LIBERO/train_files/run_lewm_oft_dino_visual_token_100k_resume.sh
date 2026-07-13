#!/bin/bash
# Continue the stable spatial-token run from 20k to 100k. The action head uses
# the original peak LR while the world model and token pooler restart more
# conservatively to preserve the non-collapsed representation.

set -e

export RUN_ID=${RUN_ID:-lewm_oft_libero_dinov2b_spatial4x4_nosigreg_noactionflow_2k}
export STEPS=${STEPS:-100000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
export WARMUP=${WARMUP:-2000}
export LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-100}
export BASE_LR=${BASE_LR:-3e-5}
export ACTION_LR=${ACTION_LR:-1e-4}
export DELTA_SIGREG_WEIGHT=0
export ACTION_WEIGHT=0
export IS_RESUME=true
export CUDA_DEVS=${CUDA_DEVS:-4,5,6,7}
export NUM_PROCESSES=${NUM_PROCESSES:-4}
export MAIN_PORT=${MAIN_PORT:-29593}
export ACCELERATE_BIN=${ACCELERATE_BIN:-.venv/bin/accelerate}
export WANDB_MODE=${WANDB_MODE:-offline}

exec bash examples/LIBERO/train_files/run_lewm_oft_dino_visual_token_diag.sh
