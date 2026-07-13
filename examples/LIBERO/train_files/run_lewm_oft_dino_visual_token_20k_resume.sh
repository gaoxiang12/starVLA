#!/bin/bash
# Resume the successful no-SIGReg/no-action-flow diagnostic from 2k to 20k.
# By default this shares physical GPUs 4-7 with the existing long-running job.

set -e

export RUN_ID=${RUN_ID:-lewm_oft_libero_dinov2b_spatial4x4_nosigreg_noactionflow_2k}
export STEPS=${STEPS:-20000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
export WARMUP=${WARMUP:-200}
export LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-100}
export DELTA_SIGREG_WEIGHT=0
export ACTION_WEIGHT=0
export IS_RESUME=true
export CUDA_DEVS=${CUDA_DEVS:-4,5,6,7}
export NUM_PROCESSES=${NUM_PROCESSES:-4}
export MAIN_PORT=${MAIN_PORT:-29592}
export ACCELERATE_BIN=${ACCELERATE_BIN:-.venv/bin/accelerate}
export WANDB_MODE=${WANDB_MODE:-offline}

exec bash examples/LIBERO/train_files/run_lewm_oft_dino_visual_token_diag.sh
