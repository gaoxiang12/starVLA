#!/usr/bin/env bash
set -euo pipefail

# Five-suite LIBERO training:
# spatial + object + goal + augmented LIBERO-10 + LIBERO-90.
#
# With the default three GPUs and per-device batch 8, 360k optimizer steps
# expose approximately the same number of samples from each pre-existing suite
# as the four-GPU, 220k-step augmented-L10 run.

export CUDA_DEVS="${CUDA_DEVS:-5,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-3}"
export ACCELERATE_BIN="${ACCELERATE_BIN:-.venv/bin/accelerate}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

export DATA_MIX="${DATA_MIX:-libero_all_wm_l10_augmented_l90}"
export RUN_ID="${RUN_ID:-lewm_oft_libero_dinov3b_l10aug489_l90_spatial4x4_trainenc1e6_statecond_360k}"
export BATCH="${BATCH:-8}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export STEPS="${STEPS:-360000}"
export WARMUP="${WARMUP:-3000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export MAIN_PORT="${MAIN_PORT:-29595}"

exec bash examples/LIBERO/train_files/run_lewm_oft_dinov3_visual_token_train.sh
