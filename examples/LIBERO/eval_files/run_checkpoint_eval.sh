#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "Usage: $0 CKPT GPU_ID PORT [NUM_TRIALS_PER_TASK=10] [SEED=7] [EXECUTE_HORIZON=model]"
  exit 2
fi

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
CKPT="$1"
GPU_ID="$2"
PORT="$3"
NUM_TRIALS_PER_TASK="${4:-10}"
SEED="${5:-7}"
EXECUTE_HORIZON="${6:-}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
TASK_START="${TASK_START:-0}"
TEMPORAL_ACTION_ENSEMBLE="${TEMPORAL_ACTION_ENSEMBLE:-false}"
ADAPTIVE_ENSEMBLE_ALPHA="${ADAPTIVE_ENSEMBLE_ALPHA:-0.0}"
PROGRESS_MODE="${PROGRESS_MODE:-learned}"
FIXED_PROGRESS="${FIXED_PROGRESS:-0.5}"
PROGRESS_EMA="${PROGRESS_EMA:-}"
RUN_VARIANT="${RUN_VARIANT:-}"
LIBERO_HOME="${LIBERO_HOME:-${STARVLA_DIR}/playground/LIBERO}"
PYTHON="${STARVLA_PYTHON:-${STARVLA_DIR}/.venv/bin/python}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}"
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found: ${PYTHON}"
  exit 1
fi

CKPT="$(realpath "${CKPT}")"
MODEL_ROOT="${CKPT%%/checkpoints/*}"
CKPT_NAME="$(basename "${CKPT}" .pt)"
EXECUTE_TAG="${EXECUTE_HORIZON:-model}"
TASK_TAG=""
if [[ "${TASK_START}" != "0" ]]; then
  TASK_TAG="_fromtask${TASK_START}"
fi
ENSEMBLE_TAG=""
if [[ "${TEMPORAL_ACTION_ENSEMBLE}" == "true" ]]; then
  ENSEMBLE_TAG="_temporalens_a${ADAPTIVE_ENSEMBLE_ALPHA}"
fi
PROGRESS_TAG="_progress${PROGRESS_MODE}"
if [[ "${PROGRESS_MODE}" == "fixed" ]]; then
  PROGRESS_TAG="${PROGRESS_TAG}${FIXED_PROGRESS}"
fi
if [[ -n "${PROGRESS_EMA}" ]]; then
  PROGRESS_TAG="${PROGRESS_TAG}_ema${PROGRESS_EMA}"
fi
if [[ -n "${RUN_VARIANT}" ]]; then
  PROGRESS_TAG="${PROGRESS_TAG}_${RUN_VARIANT}"
fi
RUN_NAME="${CKPT_NAME}_${TASK_SUITE_NAME}_${NUM_TRIALS_PER_TASK}x10_seed${SEED}_exec${EXECUTE_TAG}${PROGRESS_TAG}${ENSEMBLE_TAG}${TASK_TAG}"
LOG_DIR="${MODEL_ROOT}/logs/${TASK_SUITE_NAME}_10x10"
VIDEO_DIR="${MODEL_ROOT}/results/${TASK_SUITE_NAME}_eval${NUM_TRIALS_PER_TASK}ep/${RUN_NAME}"
SERVER_LOG="${LOG_DIR}/${RUN_NAME}_server.log"
EVAL_LOG="${LOG_DIR}/${RUN_NAME}.log"

mkdir -p "${LOG_DIR}" "${VIDEO_DIR}"
cd "${STARVLA_DIR}"

export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
unset DEBUG

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[eval] ckpt=${CKPT} gpu=${GPU_ID} port=${PORT} trials_per_task=${NUM_TRIALS_PER_TASK} seed=${SEED} execute_horizon=${EXECUTE_TAG} progress_mode=${PROGRESS_MODE} fixed_progress=${FIXED_PROGRESS} progress_ema=${PROGRESS_EMA:-checkpoint}"
SERVER_CMD=(
  "${PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}" \
  --port "${PORT}"
  --use_bf16
  --progress-mode "${PROGRESS_MODE}"
  --fixed-progress "${FIXED_PROGRESS}"
)
if [[ -n "${PROGRESS_EMA}" ]]; then
  SERVER_CMD+=(--progress-ema "${PROGRESS_EMA}")
fi
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${SERVER_CMD[@]}" >"${SERVER_LOG}" 2>&1 &
server_pid=$!

EVAL_CMD=(
  "${PYTHON}" examples/LIBERO/eval_files/eval_libero.py
  --args.pretrained-path "${CKPT}"
  --args.host 127.0.0.1
  --args.port "${PORT}"
  --args.task-suite-name "${TASK_SUITE_NAME}"
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}"
  --args.start-task "${TASK_START}"
  --args.max-tasks 10
  --args.seed "${SEED}"
  --args.adaptive-ensemble-alpha "${ADAPTIVE_ENSEMBLE_ALPHA}"
  --args.job-name "${RUN_NAME}"
  --args.video-out-path "${VIDEO_DIR}"
)
if [[ "${TEMPORAL_ACTION_ENSEMBLE}" == "true" ]]; then
  EVAL_CMD+=(--args.temporal-action-ensemble)
fi
if [[ -n "${EXECUTE_HORIZON}" ]]; then
  EVAL_CMD+=(--args.execute-horizon "${EXECUTE_HORIZON}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${EVAL_CMD[@]}" 2>&1 | tee "${EVAL_LOG}"

echo "[eval] complete: ${EVAL_LOG}"
