#!/usr/bin/env bash
# Launch the joint reconstructive compact-latent world model on RoboTwin Clean.
# This wrapper prepares arguments only; use the repository's detached-job
# policy (nohup + setsid + persistent log) for a formal run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG="${CONFIG:-${SCRIPT_DIR}/starvla_reconstructive_latent_wm_robotwin_clean.yaml}"
export DATA_MIX="${DATA_MIX:-robotwin_clean_reconstructive_wm}"
export RUN_ID="${RUN_ID:-lewm_oft_robotwin_dinov3b_clean_reconstructive_latent_joint_50k}"
export BASE_WM="${BASE_WM:-dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"
export TRAIN_ENCODER=false
export CUDA_DEVS="${CUDA_DEVS:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export BATCH="${BATCH:-8}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export STEPS="${STEPS:-50000}"
export WARMUP="${WARMUP:-1000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export BASE_LR="${BASE_LR:-1e-4}"
export LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-50}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000000}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

exec bash "${SCRIPT_DIR}/run_lewm_oft_dinov2b_train.sh"
