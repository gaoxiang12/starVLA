#!/usr/bin/env bash
set -euo pipefail

# Hand LIBERO failure cases over to a strong EXPERT policy (e.g. Qwen-OFT) and
# export the successful recoveries as a training-ready LeRobot v2.0 dataset.
#
# This is the CLIENT process. It needs TWO things running:
#
#   1) The EXPERT policy server (the ~98%% model), on its own GPU/port:
#
#        CKPT=/path/to/qwen_oft/checkpoints/steps_XXXXX_pytorch_model.pt \
#          GPU_ID=1 PORT=6700 bash examples/LIBERO/eval_files/run_policy_server.sh
#
#   2) The failure cases produced earlier by collect_failures.sh (FAILURES_DIR).
#
# Then run this collector against the expert server:
#
#   EXPERT_CKPT=/path/to/qwen_oft/.../steps_XXXXX_pytorch_model.pt \
#     LIBERO_HOME=$PWD/playground/LIBERO PORT=6700 \
#     FAILURES_DIR=$PWD/playground/Checkpoints/lewm_oft_libero_wm_vfuse/failure_cases/libero_goal/steps_80000_pytorch_model \
#     TASK_SUITE_NAME=libero_goal HANDOVER_MODE=scan \
#     bash examples/LIBERO/eval_files/collect_expert_recovery.sh

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6700}"
FAILURES_DIR="${FAILURES_DIR:-}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
HANDOVER_MODE="${HANDOVER_MODE:-deepest}"
HANDOVER_FRACTIONS="${HANDOVER_FRACTIONS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7}"
KEEP_ALL_RECOVERIES="${KEEP_ALL_RECOVERIES:-False}"
OUT_DATASET_DIR="${OUT_DATASET_DIR:-}"
MAX_STEPS="${MAX_STEPS:--1}"
LIMIT_EPISODES="${LIMIT_EPISODES:--1}"
SAVE_DEBUG_VIDEO="${SAVE_DEBUG_VIDEO:-False}"
EXPERT_CKPT="${EXPERT_CKPT:-}"
UNNORM_KEY="${UNNORM_KEY:-}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  echo "Example: LIBERO_HOME=/path/to/LIBERO PORT=6700 FAILURES_DIR=... bash $0"
  exit 1
fi
if [[ -z "${FAILURES_DIR}" ]]; then
  echo "FAILURES_DIR is required (output directory of collect_failures.sh)."
  exit 1
fi

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"

CMD=(
  "${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/collect_expert_recovery.py
  --args.host "${HOST}"
  --args.port "${PORT}"
  --args.failures-dir "${FAILURES_DIR}"
  --args.task-suite-name "${TASK_SUITE_NAME}"
  --args.handover-mode "${HANDOVER_MODE}"
  --args.handover-fractions "${HANDOVER_FRACTIONS}"
  --args.max-steps "${MAX_STEPS}"
  --args.limit-episodes "${LIMIT_EPISODES}"
  --args.expert-ckpt "${EXPERT_CKPT}"
)

# tyro exposes boolean fields as --flag / --no-flag (no explicit value).
case "${SAVE_DEBUG_VIDEO,,}" in
  1|true|yes|on) CMD+=(--args.save-debug-video) ;;
  *)             CMD+=(--args.no-save-debug-video) ;;
esac

case "${KEEP_ALL_RECOVERIES,,}" in
  1|true|yes|on) CMD+=(--args.keep-all-recoveries) ;;
  *)             CMD+=(--args.no-keep-all-recoveries) ;;
esac

if [[ -n "${OUT_DATASET_DIR}" ]]; then
  CMD+=(--args.out-dataset-dir "${OUT_DATASET_DIR}")
fi
if [[ -n "${UNNORM_KEY}" ]]; then
  CMD+=(--args.unnorm-key "${UNNORM_KEY}")
fi

"${CMD[@]}"
