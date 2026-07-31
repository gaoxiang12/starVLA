#!/usr/bin/env bash
# Gate a control latent grounded by real-frame proprioception only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RUN_ID="${RUN_ID:-lewm_oft_libero_dinov2b_control8x128_state01_ctx2_h8_from160k_5k}"
export CONTROL_STATE_PROBE=true
export CONTROL_STATE_WEIGHT="${CONTROL_STATE_WEIGHT:-0.001}"

exec "${SCRIPT_DIR}/run_lewm_oft_dinov2b_control_latent_wm_5k.sh"