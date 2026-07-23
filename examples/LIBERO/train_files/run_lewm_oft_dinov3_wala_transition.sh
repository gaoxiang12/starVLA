#!/usr/bin/env bash
# Train WALA-style transition supervision on top of the validated DINOv3
# LeWM-OFT baseline. By default, teacher, student, and deployed action modules
# are optimized together in one run. The deployed inference path is unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE_RUN="${BASELINE_RUN:-playground/Checkpoints/lewm_oft_libero_dinov3b_spatial4x4_trainenc1e6_statecond_ema09_200k_fullstate}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BASELINE_RUN}/checkpoints/steps_200000_pytorch_model.pt}"
TRANSITION_MODE="${TRANSITION_MODE:-combined}"

case "${TRANSITION_MODE}" in
  combined)
    default_run_id="lewm_oft_dinov3b_wala_transition_combined_from200k"
    default_checkpoint="${BASELINE_CHECKPOINT}"
    default_steps=20000
    default_detach=false
    ;;
  teacher)
    default_run_id="lewm_oft_dinov3b_wala_transition_teacher_from200k"
    default_checkpoint="${BASELINE_CHECKPOINT}"
    default_steps=20000
    default_detach=true
    ;;
  student)
    default_run_id="lewm_oft_dinov3b_wala_transition_student"
    default_checkpoint=""
    default_steps=20000
    default_detach=true
    ;;
  joint)
    default_run_id="lewm_oft_dinov3b_wala_transition_joint"
    default_checkpoint=""
    default_steps=20000
    default_detach=false
    ;;
  *)
    echo "TRANSITION_MODE must be teacher, student, or joint; got ${TRANSITION_MODE}" >&2
    exit 2
    ;;
esac

PRETRAINED_CKPT="${PRETRAINED_CKPT:-${default_checkpoint}}"
if [[ -z "${PRETRAINED_CKPT}" ]]; then
  echo "${TRANSITION_MODE} mode requires PRETRAINED_CKPT from the preceding stage" >&2
  exit 2
fi
if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
  echo "Checkpoint not found: ${PRETRAINED_CKPT}" >&2
  exit 2
fi

env \
  RUN_ID="${RUN_ID:-${default_run_id}}" \
  PRETRAINED_CKPT="${PRETRAINED_CKPT}" \
  STEPS="${STEPS:-${default_steps}}" \
  TRANSITION_MODE="${TRANSITION_MODE}" \
  TRANSITION_DETACH_ACTION_QUERIES="${TRANSITION_DETACH_ACTION_QUERIES:-${default_detach}}" \
  TRAIN_ENCODER="${TRAIN_ENCODER:-false}" \
  "$@" \
  bash "${SCRIPT_DIR}/run_lewm_oft_dinov3_visual_token_train.sh"
