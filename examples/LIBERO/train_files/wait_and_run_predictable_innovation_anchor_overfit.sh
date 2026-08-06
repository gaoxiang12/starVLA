#!/usr/bin/env bash
# Wait for an approved GPU, then run a tiny smoke overfit followed by the formal
# 128-anchor diagnostic. GPU 4 sharing is hard-disabled unless the caller sets
# ALLOW_GPU4_SHARING=1 after explicit authorization.
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"

PIPELINE_DIR="${PIPELINE_DIR:-playground/Checkpoints/lewm_oft_libero_dinov3b_localinc_stageb_anchor_overfit_pipeline}"
SMOKE_DIR="${SMOKE_DIR:-playground/Checkpoints/lewm_oft_libero_dinov3b_localinc_stageb_overfit16_fp32_seed42_1k}"
FORMAL_DIR="${FORMAL_DIR:-playground/Checkpoints/lewm_oft_libero_dinov3b_localinc_stageb_overfit128_fp32_seed42_2k}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1 2 3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
MAX_UTIL="${MAX_UTIL:-10}"
POLL_SECONDS="${POLL_SECONDS:-30}"
ALLOW_GPU4_SHARING="${ALLOW_GPU4_SHARING:-0}"

CHECKPOINT_ROOT="$(realpath -m "${STARVLA_DIR}/playground/Checkpoints")"
resolve_output_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    realpath -m "${value}"
  else
    realpath -m "${STARVLA_DIR}/${value}"
  fi
}
PIPELINE_DIR="$(resolve_output_path "${PIPELINE_DIR}")"
SMOKE_DIR="$(resolve_output_path "${SMOKE_DIR}")"
FORMAL_DIR="$(resolve_output_path "${FORMAL_DIR}")"
for output_path in "${PIPELINE_DIR}" "${SMOKE_DIR}" "${FORMAL_DIR}"; do
  if [[ "${output_path}" != "${CHECKPOINT_ROOT}/"* ]]; then
    echo "Output must stay below ${CHECKPOINT_ROOT}: ${output_path}" >&2
    exit 2
  fi
done
if [[ "${PIPELINE_DIR}" == "${SMOKE_DIR}" \
  || "${PIPELINE_DIR}" == "${FORMAL_DIR}" \
  || "${SMOKE_DIR}" == "${FORMAL_DIR}" ]]; then
  echo "Pipeline, smoke, and formal directories must be distinct" >&2
  exit 2
fi
if [[ ! "${MIN_FREE_MIB}" =~ ^[0-9]+$ \
  || ! "${MAX_UTIL}" =~ ^[0-9]+$ \
  || ! "${POLL_SECONDS}" =~ ^[0-9]+$ \
  || "${POLL_SECONDS}" -lt 5 ]]; then
  echo "Invalid GPU polling thresholds" >&2
  exit 2
fi
if [[ "${ALLOW_GPU4_SHARING}" != "0" && "${ALLOW_GPU4_SHARING}" != "1" ]]; then
  echo "ALLOW_GPU4_SHARING must be 0 or 1" >&2
  exit 2
fi
candidate_count=0
for gpu in ${GPU_CANDIDATES}; do
  candidate_count=$((candidate_count + 1))
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU candidate ${gpu}" >&2
    exit 2
  fi
  if [[ "${gpu}" == "4" && "${ALLOW_GPU4_SHARING}" != "1" ]]; then
    echo "GPU 4 requires explicit ALLOW_GPU4_SHARING=1" >&2
    exit 2
  fi
done
if (( candidate_count == 0 )); then
  echo "GPU_CANDIDATES must contain at least one index" >&2
  exit 2
fi

mkdir -p "${PIPELINE_DIR}"
if [[ -e "${PIPELINE_DIR}/STATUS.complete" \
  || -e "${PIPELINE_DIR}/STATUS.failed" \
  || -e "${PIPELINE_DIR}/STATUS.waiting" \
  || -e "${PIPELINE_DIR}/STATUS.running" ]]; then
  echo "Pipeline already has an active or terminal status: ${PIPELINE_DIR}" >&2
  exit 2
fi
if ! mkdir "${PIPELINE_DIR}/watcher.lock" 2>/dev/null; then
  echo "Another watcher owns ${PIPELINE_DIR}/watcher.lock" >&2
  exit 2
fi

active_child=""
signal_name=""
transition_status() {
  local target="$1"
  local source
  for source in waiting running; do
    if [[ -e "${PIPELINE_DIR}/STATUS.${source}" ]]; then
      mv "${PIPELINE_DIR}/STATUS.${source}" "${PIPELINE_DIR}/STATUS.${target}"
      return 0
    fi
  done
  return 0
}
handle_signal() {
  signal_name="$1"
  if [[ -n "${active_child}" ]] && kill -0 "${active_child}" 2>/dev/null; then
    kill -TERM "${active_child}" 2>/dev/null || true
    wait "${active_child}" 2>/dev/null || true
    active_child=""
  fi
  exit 143
}
cleanup() {
  local rc=$?
  if [[ -n "${active_child}" ]] && kill -0 "${active_child}" 2>/dev/null; then
    kill -TERM "${active_child}" 2>/dev/null || true
    wait "${active_child}" 2>/dev/null || true
  fi
  if [[ -n "${signal_name}" ]]; then
    transition_status stopped
    echo "[$(date -Is)] watcher stopped by ${signal_name}" >&2
  elif (( rc != 0 )) \
    && [[ ! -e "${PIPELINE_DIR}/STATUS.failed" \
      && ! -e "${PIPELINE_DIR}/STATUS.complete" ]]; then
    transition_status failed
  fi
  rmdir "${PIPELINE_DIR}/watcher.lock" 2>/dev/null || true
}
trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT
trap 'handle_signal HUP' HUP
trap cleanup EXIT

echo "pid=$$" > "${PIPELINE_DIR}/STATUS.waiting"

gpu_available() {
  local gpu="$1" line used free util compute_pids
  line="$(nvidia-smi -i "${gpu}" \
    --query-gpu=memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -n "${line}" ]] || return 1
  IFS=',' read -r used free util <<< "${line}"
  used="${used//[[:space:]]/}"
  free="${free//[[:space:]]/}"
  util="${util//[[:space:]]/}"
  compute_pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  compute_pids="${compute_pids//[[:space:]]/}"
  [[ "${free}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] || return 1
  if [[ "${gpu}" == "4" && "${ALLOW_GPU4_SHARING}" == "1" ]]; then
    # Explicit shared-card mode: preserve a large memory margin but allow the
    # already-running RoboTwin compute process and its utilization.
    (( free >= MIN_FREE_MIB ))
  else
    [[ -z "${compute_pids}" ]] \
      && (( free >= MIN_FREE_MIB && util <= MAX_UTIL ))
  fi
}

select_gpu() {
  local gpu
  for gpu in ${GPU_CANDIDATES}; do
    if gpu_available "${gpu}"; then
      echo "${gpu}"
      return 0
    fi
  done
  return 1
}

wait_for_gpu() {
  local candidate="" selected
  while true; do
    selected="$(select_gpu || true)"
    if [[ -n "${selected}" && "${selected}" == "${candidate}" ]]; then
      echo "${selected}"
      return 0
    fi
    candidate="${selected}"
    if [[ -n "${candidate}" ]]; then
      echo "[$(date -Is)] GPU ${candidate} passed first availability poll; confirming" >&2
    else
      echo "[$(date -Is)] waiting for idle compute-free GPU in: ${GPU_CANDIDATES}" >&2
    fi
    sleep "${POLL_SECONDS}"
  done
}

run_diagnostic() {
  local output_dir="$1"
  local anchors="$2"
  local steps="$3"
  local eval_interval="$4"
  local train_batch="$5"
  local log_path="${output_dir}.log"
  if [[ -e "${output_dir}" ]] \
    && [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to reuse non-empty diagnostic: ${output_dir}" >&2
    return 3
  fi
  if [[ -e "${log_path}" ]]; then
    echo "Refusing to overwrite diagnostic log: ${log_path}" >&2
    return 3
  fi
  echo "[$(date -Is)] launching ${anchors}-anchor diagnostic"
  mkdir -p "$(dirname "${output_dir}")"
  CUDA_DEVS="${gpu}" \
  OUTPUT_DIR="${output_dir}" \
  ANCHORS="${anchors}" \
  STEPS="${steps}" \
  EVAL_INTERVAL="${eval_interval}" \
  ENCODE_BATCH=4 \
  TRAIN_BATCH="${train_batch}" \
  EVAL_BATCH=32 \
  LEARNING_RATE=1e-3 \
  MAX_GRAD_NORM=10 \
  RAW_LOSS_WEIGHT=0 \
    bash examples/LIBERO/train_files/run_predictable_innovation_anchor_overfit.sh \
      > "${log_path}" 2>&1 &
  active_child=$!
  set +e
  wait "${active_child}"
  local rc=$?
  set -e
  active_child=""
  return "${rc}"
}

gpu="$(wait_for_gpu)"
transition_status running
echo "[$(date -Is)] selected physical GPU ${gpu} for smoke"
printf '{"stage":"smoke","gpu":%s,"time":"%s"}\n' \
  "${gpu}" "$(date -Is)" >> "${PIPELINE_DIR}/selected_gpus.jsonl"
if ! run_diagnostic "${SMOKE_DIR}" 16 1000 25 16; then
  transition_status failed
  echo "[$(date -Is)] smoke diagnostic process failed" >&2
  exit 4
fi
if [[ ! -e "${SMOKE_DIR}/GATE.pass" ]]; then
  transition_status failed
  echo "[$(date -Is)] smoke overfit gate failed; formal diagnostic not launched" >&2
  exit 5
fi

echo "[$(date -Is)] smoke gate passed"
gpu="$(wait_for_gpu)"
echo "[$(date -Is)] selected physical GPU ${gpu} for formal"
printf '{"stage":"formal","gpu":%s,"time":"%s"}\n' \
  "${gpu}" "$(date -Is)" >> "${PIPELINE_DIR}/selected_gpus.jsonl"
if ! run_diagnostic "${FORMAL_DIR}" 128 2000 50 16; then
  transition_status failed
  echo "[$(date -Is)] formal diagnostic process failed" >&2
  exit 6
fi
if [[ ! -e "${FORMAL_DIR}/GATE.pass" ]]; then
  transition_status failed
  echo "[$(date -Is)] formal overfit gate failed" >&2
  exit 7
fi

transition_status complete
echo "[$(date -Is)] smoke and formal fixed-anchor overfit gates passed"
