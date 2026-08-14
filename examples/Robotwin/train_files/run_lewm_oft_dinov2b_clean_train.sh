#!/usr/bin/env bash
# Train the current DINOv3-B LeWM-OFT recipe on all 50 RoboTwin Clean tasks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_MIX="${DATA_MIX:-robotwin_clean_smooth_latent_wm}"
export RUN_ID="${RUN_ID:-lewm_oft_robotwin_dinov3b_clean50_act_smooth_spatial48x384_200k}"
export BASE_WM="${BASE_WM:-dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"
export STEPS="${STEPS:-200000}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"

exec bash "${SCRIPT_DIR}/run_lewm_oft_dinov2b_train.sh"
