#!/usr/bin/env bash
# Gate a control latent grounded by inverse dynamics on real latent transitions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RUN_ID="${RUN_ID:-lewm_oft_libero_dinov2b_control8x128_inverse001_ctx2_h8_from160k_5k}"
export CONTROL_STATE_PROBE=false
export CONTROL_ACTION_PROBE=true
export CONTROL_ACTION_WEIGHT="${CONTROL_ACTION_WEIGHT:-0.001}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"

exec "${SCRIPT_DIR}/run_lewm_oft_dinov2b_control_latent_wm_5k.sh"