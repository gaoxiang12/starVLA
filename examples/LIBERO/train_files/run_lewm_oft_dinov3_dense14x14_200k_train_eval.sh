#!/usr/bin/env bash
set -euo pipefail

# Strict spatial-resolution control against the validated 4x4 DINOv3 baseline.
# Both runs start from the raw DINOv3 pretrained encoder and random downstream
# modules, train the complete model for 200k optimizer steps, and use the same
# four-suite LIBERO closed-loop evaluation. The intentional model difference is
# 14x14 unpooled patch tokens per view instead of 4x4 pooled tokens per view.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

env \
  RUN_ID="${RUN_ID:-lewm_oft_libero_dinov3b_dense14x14_latent1_sigreg0_trainenc1e6_statecond_200k}" \
  STEPS="${STEPS:-200000}" \
  SAVE_INTERVAL="${SAVE_INTERVAL:-10000}" \
  WARMUP="${WARMUP:-2000}" \
  BATCH="${BATCH:-8}" \
  VISUAL_TOKENS_PER_VIEW=196 \
  VISUAL_TOKEN_DIM=384 \
  VISUAL_DIAGNOSTICS=false \
  USE_DENSE_PATCH_ACTION=false \
  TRANSITION_MODE=off \
  TRAIN_ENCODER=true \
  LATENT_WEIGHT=1.0 \
  TOKEN_DIVERSITY_WEIGHT=0.02 \
  TOKEN_VARIANCE_WEIGHT=0.02 \
  RESIDUAL_SIGREG_WEIGHT=0.0 \
  BASE_LR="${BASE_LR:-1e-4}" \
  ACTION_LR="${ACTION_LR:-1e-4}" \
  ENCODER_LR="${ENCODER_LR:-1e-6}" \
  USE_STATE_COND=true \
  INCLUDE_STATE=true \
  LATENT_STATS_MOMENTUM=0.99 \
  PRETRAINED_CKPT= \
  WANDB_MODE="${WANDB_MODE:-disabled}" \
  "$@" \
  bash "${SCRIPT_DIR}/run_lewm_oft_dinov3_200k_train_eval.sh"
