#!/usr/bin/env bash
# Targeted fine-tuning of the canonical Clean-50 DINOv3-B checkpoint on the
# locally collected 1,000-episode click_bell dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

export DATA_ROOT="${DATA_ROOT:-playground/Datasets/RoboTwinClickBellClean1000}"
export DATA_MIX="${DATA_MIX:-robotwin_click_bell_clean1000_wm}"
export RUN_ID="${RUN_ID:-lewm_oft_robotwin_dinov3b_clean50_canonical_click_bell_clean1000_basestats_ft20k}"
export BASE_WM="${BASE_WM:-dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${STARVLA_DIR}/playground/Checkpoints/lewm_oft_robotwin_dinov3b_clean50_canonical_tasktext_fromscratch_200k/checkpoints/steps_200000_pytorch_model.pt}"
export NORMALIZATION_STATISTICS_PATH="${NORMALIZATION_STATISTICS_PATH:-${STARVLA_DIR}/playground/Checkpoints/lewm_oft_robotwin_dinov3b_clean50_canonical_tasktext_fromscratch_200k/dataset_statistics.json}"
# Reuse the warm-start run's resolved architecture as well as its weights.  The
# current generic YAML has since changed the action/language heads, which would
# otherwise make this more than a normalization-statistics-only ablation.
export CONFIG="${CONFIG:-${STARVLA_DIR}/playground/Checkpoints/lewm_oft_robotwin_dinov3b_clean50_canonical_tasktext_fromscratch_200k/config.full.yaml}"

export CUDA_DEVS="${CUDA_DEVS:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export MAIN_PORT="${MAIN_PORT:-29641}"
export BATCH="${BATCH:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export STEPS="${STEPS:-20000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export WARMUP="${WARMUP:-500}"

# A conservative 10x LR reduction preserves the already strong 200k policy
# while allowing the action/world-model heads and encoder to adapt.
export BASE_LR="${BASE_LR:-1e-5}"
export ACTION_LR="${ACTION_LR:-1e-5}"
export ENCODER_LR="${ENCODER_LR:-1e-7}"
export TASK_LANGUAGE_MODE="dataset_name"
export TRAIN_ENCODER="true"
export ACCELERATE_BIN="${ACCELERATE_BIN:-${STARVLA_DIR}/.venv/bin/accelerate}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

exec bash "${SCRIPT_DIR}/run_lewm_oft_dinov2b_train.sh"
