#!/usr/bin/env bash
# Safely run the offline raw-feature predictive-subspace probe while sharing
# physical GPU 4 with the existing RoboTwin training process. This launcher
# never sends a terminating signal to RoboTwin; its traps only own the probe's
# separate process group.
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"

export PATH="${STARVLA_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled

GPU_ID="${GPU_ID:-4}"
ALLOW_GPU4_SHARING="${ALLOW_GPU4_SHARING:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"
ENCODE_BATCH="${ENCODE_BATCH:-4}"
PROGRESS_POLL_SECONDS="${PROGRESS_POLL_SECONDS:-10}"
PROGRESS_TIMEOUT_SECONDS="${PROGRESS_TIMEOUT_SECONDS:-120}"
METRICS_MAX_AGE_SECONDS="${METRICS_MAX_AGE_SECONDS:-180}"

ROBOTWIN_RUN_ID="${ROBOTWIN_RUN_ID:-lewm_oft_robotwin_dinov3b_clean50_spatial4x4_current_200k}"
ROBOTWIN_RUN_DIR="${ROBOTWIN_RUN_DIR:-playground/Checkpoints/${ROBOTWIN_RUN_ID}}"
ROBOTWIN_STATUS="${ROBOTWIN_STATUS:-${ROBOTWIN_RUN_DIR}/STATUS.running}"
ROBOTWIN_METRICS="${ROBOTWIN_METRICS:-${ROBOTWIN_RUN_DIR}/metrics.jsonl}"

PROBE_SCRIPT="${PROBE_SCRIPT:-examples/LIBERO/train_files/diagnose_raw_predictive_subspace.py}"
CONFIG="${CONFIG:-examples/LIBERO/train_files/starvla_lewm_oft_dinov3_libero_wm_only_predictable_innovation_10k.yaml}"
CHECKPOINT="${CHECKPOINT:-playground/Checkpoints/lewm_oft_libero_dinov3b_wmonly_localbasis_m4r64_fixedmean_2k/checkpoints/steps_2000_pytorch_model.pt}"
TRAIN_ANCHORS="${TRAIN_ANCHORS:-640}"
VAL_ANCHORS="${VAL_ANCHORS:-192}"
TEST_ANCHORS="${TEST_ANCHORS:-256}"
SEED="${SEED:-42}"

RUN_ID="${RUN_ID:-lewm_oft_libero_dinov3b_raw_predictive_subspace_probe_train640_val192_test256_seed42}"
OUTPUT_DIR="${OUTPUT_DIR:-playground/Checkpoints/${RUN_ID}}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}.log}"
LAUNCH_DIR="${LAUNCH_DIR:-${OUTPUT_DIR}.launcher}"

fail() {
  echo "ERROR: $*" >&2
  return 1
}

if [[ "${ALLOW_GPU4_SHARING}" != "1" ]]; then
  fail "GPU 4 sharing requires explicit ALLOW_GPU4_SHARING=1" || exit 2
fi
if [[ "${GPU_ID}" != "4" ]]; then
  fail "this guarded launcher is restricted to physical GPU 4" || exit 2
fi
if [[ ! "${MIN_FREE_MIB}" =~ ^[0-9]+$ ]] || (( MIN_FREE_MIB < 24000 )); then
  fail "MIN_FREE_MIB must be an integer at least 24000" || exit 2
fi
if [[ "${ENCODE_BATCH}" != "4" ]]; then
  fail "ENCODE_BATCH is safety-fixed at 4 while sharing GPU 4" || exit 2
fi
for value in \
  "${PROGRESS_POLL_SECONDS}" \
  "${PROGRESS_TIMEOUT_SECONDS}" \
  "${METRICS_MAX_AGE_SECONDS}" \
  "${TRAIN_ANCHORS}" \
  "${VAL_ANCHORS}" \
  "${TEST_ANCHORS}" \
  "${SEED}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    fail "polling, split, and seed values must be non-negative integers" || exit 2
  fi
done
if (( PROGRESS_POLL_SECONDS < 5 \
  || PROGRESS_TIMEOUT_SECONDS < PROGRESS_POLL_SECONDS \
  || METRICS_MAX_AGE_SECONDS < PROGRESS_POLL_SECONDS )); then
  fail "invalid RoboTwin metrics polling window" || exit 2
fi
if (( TRAIN_ANCHORS == 0 || VAL_ANCHORS == 0 || TEST_ANCHORS == 0 )); then
  fail "train/validation/test anchor counts must all be positive" || exit 2
fi

for source_path in "${PROBE_SCRIPT}" "${CONFIG}" "${CHECKPOINT}"; do
  if [[ ! -f "${source_path}" ]]; then
    fail "required input is missing: ${source_path}" || exit 2
  fi
done

CHECKPOINT_ROOT="$(realpath -m "${STARVLA_DIR}/playground/Checkpoints")"
resolve_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    realpath -m "${value}"
  else
    realpath -m "${STARVLA_DIR}/${value}"
  fi
}

OUTPUT_DIR="$(resolve_path "${OUTPUT_DIR}")"
LOG_PATH="$(resolve_path "${LOG_PATH}")"
LAUNCH_DIR="$(resolve_path "${LAUNCH_DIR}")"
ROBOTWIN_RUN_DIR="$(resolve_path "${ROBOTWIN_RUN_DIR}")"
ROBOTWIN_STATUS="$(resolve_path "${ROBOTWIN_STATUS}")"
ROBOTWIN_METRICS="$(resolve_path "${ROBOTWIN_METRICS}")"
PROBE_SCRIPT="$(resolve_path "${PROBE_SCRIPT}")"
CONFIG="$(resolve_path "${CONFIG}")"
CHECKPOINT="$(resolve_path "${CHECKPOINT}")"

for output_path in "${OUTPUT_DIR}" "${LOG_PATH}" "${LAUNCH_DIR}"; do
  if [[ "${output_path}" != "${CHECKPOINT_ROOT}/"* ]]; then
    fail "probe output must stay below ${CHECKPOINT_ROOT}: ${output_path}" || exit 2
  fi
  case "${output_path}" in
    "${ROBOTWIN_RUN_DIR}"|"${ROBOTWIN_RUN_DIR}/"*)
      fail "probe paths must not overlap the RoboTwin run directory" || exit 2
      ;;
  esac
done
if [[ "${OUTPUT_DIR}" == "${LOG_PATH}" \
  || "${OUTPUT_DIR}" == "${LAUNCH_DIR}" \
  || "${LOG_PATH}" == "${LAUNCH_DIR}" ]]; then
  fail "output, log, and launcher-state paths must be distinct" || exit 2
fi
if [[ -e "${OUTPUT_DIR}" || -e "${LOG_PATH}" || -e "${LAUNCH_DIR}" ]]; then
  fail "refusing to reuse an existing output, log, or launcher state path" || exit 2
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"
mkdir "${LAUNCH_DIR}"
if ! (set -o noclobber; : > "${LOG_PATH}") 2>/dev/null; then
  fail "could not reserve non-overwriting log path: ${LOG_PATH}" || exit 2
fi

status_path="${LAUNCH_DIR}/STATUS.preflight"
printf 'status=preflight\nlauncher_pid=%s\nstarted_at=%s\n' \
  "$$" "$(date -Is)" > "${status_path}"

active_child=""
signal_name=""
transition_status() {
  local target="$1"
  local source
  for source in preflight running; do
    if [[ -e "${LAUNCH_DIR}/STATUS.${source}" ]]; then
      mv "${LAUNCH_DIR}/STATUS.${source}" "${LAUNCH_DIR}/STATUS.${target}"
      printf 'terminal_status=%s\nfinished_at=%s\n' \
        "${target}" "$(date -Is)" >> "${LAUNCH_DIR}/STATUS.${target}"
      status_path="${LAUNCH_DIR}/STATUS.${target}"
      return 0
    fi
  done
}

stop_probe_only() {
  if [[ -n "${active_child}" ]] && kill -0 "${active_child}" 2>/dev/null; then
    # active_child is the probe's setsid process-group leader. Never target
    # EXPECTED_ROBOTWIN_PID here or anywhere else in this launcher.
    kill -TERM -- "-${active_child}" 2>/dev/null || true
    wait "${active_child}" 2>/dev/null || true
  fi
  active_child=""
}

handle_signal() {
  signal_name="$1"
  stop_probe_only
  transition_status stopped
  exit 143
}

cleanup() {
  local rc=$?
  stop_probe_only
  if [[ -n "${signal_name}" ]]; then
    return
  fi
  if [[ -e "${LAUNCH_DIR}/STATUS.preflight" \
    || -e "${LAUNCH_DIR}/STATUS.running" ]]; then
    transition_status failed
    printf 'exit_code=%s\n' "${rc}" >> "${LAUNCH_DIR}/STATUS.failed"
  fi
}
trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT
trap 'handle_signal HUP' HUP
trap cleanup EXIT

read_status_worker_pid() {
  sed -n 's/^worker_pid=\([0-9][0-9]*\)$/\1/p' "${ROBOTWIN_STATUS}" \
    | tail -n 1
}

verify_robotwin_identity() {
  if [[ ! -f "${ROBOTWIN_STATUS}" ]]; then
    fail "RoboTwin STATUS.running is missing: ${ROBOTWIN_STATUS}"
    return 1
  fi
  if [[ -e "${ROBOTWIN_RUN_DIR}/STATUS.complete" \
    || -e "${ROBOTWIN_RUN_DIR}/STATUS.failed" \
    || -e "${ROBOTWIN_RUN_DIR}/STATUS.stopped" ]]; then
    fail "RoboTwin has a terminal status marker"
    return 1
  fi

  local status_pid process_exe process_cmd
  status_pid="$(read_status_worker_pid)"
  if [[ -z "${status_pid}" || ! "${status_pid}" =~ ^[0-9]+$ ]]; then
    fail "RoboTwin STATUS.running has no valid worker_pid"
    return 1
  fi
  if [[ -n "${EXPECTED_ROBOTWIN_PID:-}" \
    && "${EXPECTED_ROBOTWIN_PID}" != "${status_pid}" ]]; then
    fail "configured EXPECTED_ROBOTWIN_PID does not match STATUS.running"
    return 1
  fi
  EXPECTED_ROBOTWIN_PID="${status_pid}"
  if ! kill -0 "${EXPECTED_ROBOTWIN_PID}" 2>/dev/null; then
    fail "expected RoboTwin worker PID is not alive: ${EXPECTED_ROBOTWIN_PID}"
    return 1
  fi
  if [[ ! -r "/proc/${EXPECTED_ROBOTWIN_PID}/cmdline" ]]; then
    fail "cannot inspect expected RoboTwin process"
    return 1
  fi
  process_exe="$(readlink -f "/proc/${EXPECTED_ROBOTWIN_PID}/exe" 2>/dev/null || true)"
  process_cmd="$(tr '\0' ' ' < "/proc/${EXPECTED_ROBOTWIN_PID}/cmdline")"
  if [[ "$(basename "${process_exe}")" != python* \
    || "${process_cmd}" != *"starVLA/training/train_starvla.py"* \
    || "${process_cmd}" != *"--run_id ${ROBOTWIN_RUN_ID}"* \
    || "${process_cmd}" != *"robotwin_clean_wm"* ]]; then
    fail "expected PID is not the configured RoboTwin Python training worker"
    return 1
  fi
}

verify_gpu4_exclusivity_and_memory() {
  local free_line free_mib queried_pid
  local -a compute_pids=()
  free_line="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  free_mib="${free_line//[[:space:]]/}"
  if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_MIB )); then
    fail "GPU 4 free memory ${free_mib:-unknown} MiB is below ${MIN_FREE_MIB} MiB"
    return 1
  fi
  while IFS= read -r queried_pid; do
    queried_pid="${queried_pid//[[:space:]]/}"
    if [[ "${queried_pid}" =~ ^[0-9]+$ ]]; then
      compute_pids+=("${queried_pid}")
    fi
  done < <(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true)
  if (( ${#compute_pids[@]} != 1 )) \
    || [[ "${compute_pids[0]:-}" != "${EXPECTED_ROBOTWIN_PID}" ]]; then
    fail "GPU 4 must contain only RoboTwin PID ${EXPECTED_ROBOTWIN_PID}; found: ${compute_pids[*]:-none}"
    return 1
  fi
  GPU4_FREE_MIB="${free_mib}"
}

read_metrics_snapshot() {
  if [[ ! -s "${ROBOTWIN_METRICS}" ]]; then
    fail "RoboTwin metrics file is missing or empty: ${ROBOTWIN_METRICS}"
    return 1
  fi
  "${STARVLA_DIR}/.venv/bin/python" -c \
    'import json, os, sys
path = sys.argv[1]
with open(path, "rb") as stream:
    stream.seek(0, os.SEEK_END)
    end = stream.tell()
    pos = end - 1
    while pos >= 0:
        stream.seek(pos)
        if stream.read(1) == b"\n" and pos < end - 1:
            break
        pos -= 1
    stream.seek(max(0, pos + 1))
    record = json.loads(stream.readline())
step = int(record["step"])
print(f"{step} {os.stat(path).st_mtime_ns}")' \
    "${ROBOTWIN_METRICS}"
}

verify_robotwin_identity
verify_gpu4_exclusivity_and_memory
first_snapshot="$(read_metrics_snapshot)" || exit 3
read -r first_step first_mtime_ns <<< "${first_snapshot}"
progress_deadline=$(( $(date +%s) + PROGRESS_TIMEOUT_SECONDS ))
progressed=0
second_step="${first_step}"
second_mtime_ns="${first_mtime_ns}"
while (( $(date +%s) < progress_deadline )); do
  sleep "${PROGRESS_POLL_SECONDS}"
  verify_robotwin_identity
  verify_gpu4_exclusivity_and_memory
  second_snapshot="$(read_metrics_snapshot)" || exit 3
  read -r second_step second_mtime_ns <<< "${second_snapshot}"
  if (( second_step > first_step && second_mtime_ns > first_mtime_ns )); then
    progressed=1
    break
  fi
done
if (( progressed != 1 )); then
  fail "RoboTwin metrics did not advance within ${PROGRESS_TIMEOUT_SECONDS}s"
  exit 3
fi
metrics_age=$(( $(date +%s) - second_mtime_ns / 1000000000 ))
if (( metrics_age < 0 || metrics_age > METRICS_MAX_AGE_SECONDS )); then
  fail "RoboTwin metrics are stale (${metrics_age}s old)"
  exit 3
fi

# Close the preflight race immediately before launching the probe.
verify_robotwin_identity
verify_gpu4_exclusivity_and_memory

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

probe_command=(
  "${STARVLA_DIR}/.venv/bin/python"
  "${PROBE_SCRIPT}"
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --train-anchors "${TRAIN_ANCHORS}"
  --val-anchors "${VAL_ANCHORS}"
  --test-anchors "${TEST_ANCHORS}"
  --seed "${SEED}"
)
# Optional CLI arguments are accepted, but the safety-critical output/device/
# encode-batch values are appended last so argparse cannot override them.
probe_command+=("$@")
probe_command+=(
  --output-dir "${OUTPUT_DIR}"
  --device cuda:0
  --encode-batch-size "${ENCODE_BATCH}"
)

{
  printf 'preflight_time=%s\n' "$(date -Is)"
  printf 'robotwin_pid=%s\n' "${EXPECTED_ROBOTWIN_PID}"
  printf 'robotwin_metrics_step_before=%s\n' "${first_step}"
  printf 'robotwin_metrics_step_after=%s\n' "${second_step}"
  printf 'gpu4_free_mib=%s\n' "${GPU4_FREE_MIB}"
  printf 'command='
  printf ' %q' "${probe_command[@]}"
  printf '\n'
} >> "${LOG_PATH}"

mv "${LAUNCH_DIR}/STATUS.preflight" "${LAUNCH_DIR}/STATUS.running"
status_path="${LAUNCH_DIR}/STATUS.running"
setsid "${probe_command[@]}" >> "${LOG_PATH}" 2>&1 < /dev/null &
active_child=$!
printf '%s\n' "${active_child}" > "${LAUNCH_DIR}/probe.pid"
printf 'probe_pid=%s\nlaunched_at=%s\n' \
  "${active_child}" "$(date -Is)" >> "${status_path}"

set +e
wait "${active_child}"
probe_rc=$?
set -e
active_child=""
if (( probe_rc == 0 )); then
  transition_status complete
  exit 0
fi
transition_status failed
printf 'exit_code=%s\n' "${probe_rc}" >> "${LAUNCH_DIR}/STATUS.failed"
exit "${probe_rc}"
