#!/bin/bash
# Short diagnostic run for visual/world-model collapse. This starts from
# scratch and logs content-only representation metrics every 20 optimizer steps.

set -e

export RUN_ID=${RUN_ID:-lewm_oft_libero_dinov2b_spatial4x4_collapse_diag_2k}
export STEPS=${STEPS:-2000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-500}
export WARMUP=${WARMUP:-200}
export LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-20}
export ACCELERATE_BIN=${ACCELERATE_BIN:-.venv/bin/accelerate}
export WANDB_MODE=${WANDB_MODE:-offline}
unset PRETRAINED_CKPT
export IS_RESUME=${IS_RESUME:-false}

exec bash examples/LIBERO/train_files/run_lewm_oft_dino_visual_token_train.sh
