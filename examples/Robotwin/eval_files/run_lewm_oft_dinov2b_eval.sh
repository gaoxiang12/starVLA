#!/usr/bin/env bash
# Evaluate a DINOv2-base LeWM-OFT checkpoint on RoboTwin clean/randomized modes.
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"

if [[ -z "${ROBOTWIN_PATH:-}" && -d "${STARVLA_DIR}/thirdparty/RoboTwin" ]]; then
  export ROBOTWIN_PATH="${STARVLA_DIR}/thirdparty/RoboTwin"
fi

usage() {
  cat >&2 <<'EOF'
Usage:
  bash examples/Robotwin/eval_files/run_lewm_oft_dinov2b_eval.sh <checkpoint> [task ...]

Examples:
  MODES=demo_clean bash examples/Robotwin/eval_files/run_lewm_oft_dinov2b_eval.sh \
    playground/Checkpoints/<run>/checkpoints/steps_200000_pytorch_model.pt click_bell

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    bash examples/Robotwin/eval_files/run_lewm_oft_dinov2b_eval.sh \
    playground/Checkpoints/<run>/checkpoints/steps_200000_pytorch_model.pt all

Environment:
  MODES                 Comma-separated modes (default: demo_clean,demo_randomized)
  ROBOTWIN_PATH         Required path to the RoboTwin checkout
  ROBOTWIN_PYTHON       Python executable in the RoboTwin environment
  STARVLA_PYTHON        Python executable in the StarVLA environment
  SEED                   Evaluation seed (default: 0)
  EPISODES               Valid rollouts per task (default: 100)
  JOBS_PER_GPU           Concurrent tasks per GPU (default: 1)
  BASE_PORT              First policy-server port (default: 5694)
  SERVER_TIMEOUT         Policy-server startup timeout in seconds (default: 600)
  ROBOTWIN_EVAL_VIDEO_LOG Encode every policy rollout as MP4 (default: 0)
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

CKPT_PATH="$1"
shift
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ -z "${ROBOTWIN_PATH:-}" || ! -d "${ROBOTWIN_PATH}" ]]; then
  echo "Set ROBOTWIN_PATH to a valid RoboTwin checkout." >&2
  exit 1
fi

if [[ -z "${STARVLA_PYTHON:-}" && -x "${STARVLA_DIR}/.venv/bin/python" ]]; then
  export STARVLA_PYTHON="${STARVLA_DIR}/.venv/bin/python"
fi

tasks=("$@")
if (( ${#tasks[@]} == 0 )); then
  tasks=(all)
fi

run_dir="$(dirname "$(dirname "${CKPT_PATH}")")"
checkpoint_stem="$(basename "${CKPT_PATH}" .pt)"
checkpoint_tag="${checkpoint_stem%_pytorch_model}"
POLICY_NAME="${POLICY_NAME:-$(basename "${run_dir}")_dinov2b_lewm_oft_${checkpoint_tag}}"
IFS=',' read -r -a modes <<<"${MODES:-demo_clean,demo_randomized}"

base_port="${BASE_PORT:-5694}"
for index in "${!modes[@]}"; do
  mode="${modes[$index]}"
  if [[ "${mode}" != "demo_clean" && "${mode}" != "demo_randomized" ]]; then
    echo "Unsupported RoboTwin mode: ${mode}" >&2
    exit 1
  fi

  mode_port=$((base_port + index * 100))
  echo "[INFO] Evaluating ${mode} from ${CKPT_PATH}"
  bash examples/Robotwin/eval_files/start_eval.sh \
    --mode "${mode}" \
    --name "${POLICY_NAME}" \
    --ckpt "${CKPT_PATH}" \
    --seed "${SEED:-0}" \
    --episodes "${EPISODES:-100}" \
    --jobs-per-gpu "${JOBS_PER_GPU:-1}" \
    --base-port "${mode_port}" \
    --server-timeout "${SERVER_TIMEOUT:-600}" \
    "${tasks[@]}"
done
