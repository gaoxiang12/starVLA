#!/usr/bin/env bash
set -euo pipefail

# Collect failure cases of YOUR policy on (a subset of) LIBERO tasks.
#
# This is the CLIENT process. Start the policy server first (same as normal
# eval), pointing it at the model you want to probe:
#
#   CKPT=$PWD/playground/Checkpoints/lewm_oft_libero_wm_vfuse/checkpoints/steps_80000_pytorch_model.pt \
#     PORT=6699 bash examples/LIBERO/eval_files/run_policy_server.sh
#
# Then run this collector against it:
#
#   CKPT=$PWD/playground/Checkpoints/lewm_oft_libero_wm_vfuse/checkpoints/steps_80000_pytorch_model.pt \
#     LIBERO_HOME=$PWD/playground/LIBERO PORT=6699 \
#     TASK_SUITE_NAME=libero_goal NUM_TRIALS_PER_TASK=50 \
#     TASK_FILTER="open the top drawer and put the bowl inside" \
#     bash examples/LIBERO/eval_files/collect_failures.sh

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/lewm_oft_libero_wm_vfuse/checkpoints/steps_80000_pytorch_model.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6699}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
TASK_FILTER="${TASK_FILTER:-open the top drawer and put the bowl inside}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SAVE_VIDEO="${SAVE_VIDEO:-True}"
OUT_PATH="${OUT_PATH:-}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  echo "Example: LIBERO_HOME=/path/to/LIBERO PORT=6699 bash $0"
  exit 1
fi

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"

CMD=(
  "${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/collect_failures.py
  --args.pretrained-path "${CKPT}"
  --args.host "${HOST}"
  --args.port "${PORT}"
  --args.task-suite-name "${TASK_SUITE_NAME}"
  --args.task-filter "${TASK_FILTER}"
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}"
)

# tyro exposes boolean fields as --flag / --no-flag (no explicit value).
case "${SAVE_VIDEO,,}" in
  1|true|yes|on) CMD+=(--args.save-video) ;;
  *)             CMD+=(--args.no-save-video) ;;
esac

if [[ -n "${OUT_PATH}" ]]; then
  CMD+=(--args.out-path "${OUT_PATH}")
fi

"${CMD[@]}"
