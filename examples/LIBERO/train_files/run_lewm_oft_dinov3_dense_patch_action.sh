#!/usr/bin/env bash
# Patch Policy-inspired dense-current action residual on the validated DINOv3
# LIBERO baseline. The compact 4x4 world model remains frozen and deployed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE_RUN="${BASELINE_RUN:-playground/Checkpoints/lewm_oft_libero_dinov3b_spatial4x4_trainenc1e6_statecond_ema09_200k_fullstate}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BASELINE_RUN}/checkpoints/steps_200000_pytorch_model.pt}"

if [[ ! -f "${BASELINE_CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${BASELINE_CHECKPOINT}" >&2
  exit 2
fi

# Only the new dense adapter and the existing OFT action model are optimized.
# The adapter sees 2 x 14 x 14 current-frame patches; true future patches stay
# exclusive to the frozen compact world-model target path.
env \
  RUN_ID="${RUN_ID:-lewm_oft_dinov3b_densepatch_action_zeroout_from200k}" \
  PRETRAINED_CKPT="${PRETRAINED_CKPT:-${BASELINE_CHECKPOINT}}" \
  STEPS="${STEPS:-20000}" \
  SAVE_INTERVAL="${SAVE_INTERVAL:-2000}" \
  WARMUP="${WARMUP:-500}" \
  TRANSITION_MODE=off \
  USE_DENSE_PATCH_ACTION=true \
  DENSE_PATCH_HIDDEN_DIM="${DENSE_PATCH_HIDDEN_DIM:-384}" \
  DENSE_PATCH_HEADS="${DENSE_PATCH_HEADS:-6}" \
  DENSE_PATCH_GRID_SIZE="${DENSE_PATCH_GRID_SIZE:-14}" \
  DENSE_PATCH_GATE_INIT="${DENSE_PATCH_GATE_INIT:-1.0}" \
  DENSE_PATCH_FREEZE_BASE=true \
  DENSE_PATCH_LR="${DENSE_PATCH_LR:-1e-4}" \
  ACTION_LR="${ACTION_LR:-1e-5}" \
  BASE_LR="${BASE_LR:-1e-4}" \
  TRAIN_ENCODER=false \
  LATENT_WEIGHT=0.0 \
  TOKEN_DIVERSITY_WEIGHT=0.0 \
  TOKEN_VARIANCE_WEIGHT=0.0 \
  RESIDUAL_SIGREG_WEIGHT=0.0 \
  LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}" \
  "$@" \
  bash "${SCRIPT_DIR}/run_lewm_oft_dinov3_visual_token_train.sh"
