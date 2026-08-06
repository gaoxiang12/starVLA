#!/usr/bin/env bash
# Promote Stage A to Stage B only after strict artifact and representation gates.
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"

STAGE_A_RUN="${STAGE_A_RUN:-playground/Checkpoints/lewm_oft_libero_dinov3b_wmonly_localbasis_m4r64_fixedmean_2k}"
STAGE_B_RUN="${STAGE_B_RUN:-playground/Checkpoints/lewm_oft_libero_dinov3b_wmonly_localinc_m4r64_ctx3_from220k_10k}"
PIPELINE_DIR="${PIPELINE_DIR:-playground/Checkpoints/lewm_oft_libero_dinov3b_predictable_local_dynamics_pipeline}"
STAGE_B_LAUNCHER="${STAGE_B_LAUNCHER:-examples/LIBERO/train_files/run_lewm_oft_dinov3_libero_wm_only_predictable_innovation_10k.sh}"
BASIS_GATE_SCRIPT="${BASIS_GATE_SCRIPT:-examples/LIBERO/train_files/summarize_local_basis_gate.py}"
PREDICTOR_GATE_SCRIPT="${PREDICTOR_GATE_SCRIPT:-examples/LIBERO/train_files/summarize_predictable_innovation_gate.py}"
PYTHON_BIN="${PYTHON_BIN:-${STARVLA_DIR}/.venv/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-30}"
EXPECTED_BASIS_STEPS="${EXPECTED_BASIS_STEPS:-2000}"
GATE_WINDOW_STEPS="${GATE_WINDOW_STEPS:-500}"
EXPECTED_PREDICTOR_STEPS="${EXPECTED_PREDICTOR_STEPS:-10000}"
PREDICTOR_GATE_WINDOW_STEPS="${PREDICTOR_GATE_WINDOW_STEPS:-2000}"
CUDA_DEVS="${CUDA_DEVS:-0}"
MAIN_PORT="${MAIN_PORT:-29647}"
MIN_FREE_MIB="${MIN_FREE_MIB:-10000}"
RESOURCE_MAX_POLLS="${RESOURCE_MAX_POLLS:-120}"
STARTUP_MAX_POLLS="${STARTUP_MAX_POLLS:-60}"

mkdir -p "${PIPELINE_DIR}"
LOCK_DIR="${PIPELINE_DIR}/watcher.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Watcher lock already exists: ${LOCK_DIR}" >&2
  exit 3
fi

stage_b_pid=""
handle_signal() {
  signal_name="$1"
  trap - HUP INT TERM
  echo "[$(date -Is)] watcher received ${signal_name}" >&2
  touch "${PIPELINE_DIR}/STATUS.watcher_stopped"
  if [[ -n "${stage_b_pid}" ]] && kill -0 "${stage_b_pid}" 2>/dev/null; then
    kill -TERM -- "-${stage_b_pid}" 2>/dev/null || true
    wait "${stage_b_pid}" 2>/dev/null || true
    if [[ -e "${STAGE_B_RUN}/STATUS.running" ]]; then
      mv "${STAGE_B_RUN}/STATUS.running" "${STAGE_B_RUN}/STATUS.stopped"
    fi
  fi
  exit 130
}
trap 'handle_signal HUP' HUP
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

printf '%s\n' "${BASHPID}" > "${LOCK_DIR}/pid"
printf '%s\n' "$(ps -o lstart= -p "${BASHPID}" | sed 's/^ *//')" > "${LOCK_DIR}/start_time"
printf '%s\n' "${STAGE_A_RUN}" > "${PIPELINE_DIR}/stage_a_run_dir.txt"
printf '%s\n' "${STAGE_B_RUN}" > "${PIPELINE_DIR}/stage_b_run_dir.txt"

if [[ ! "${CUDA_DEVS}" =~ ^[0-7]$ ]]; then
  touch "${PIPELINE_DIR}/STATUS.cuda_selection_invalid"
  echo "This single-process pipeline requires exactly one GPU index, got ${CUDA_DEVS}" >&2
  exit 3
fi
if [[ "${CUDA_DEVS}" == "4" ]]; then
  touch "${PIPELINE_DIR}/STATUS.robotwin_gpu_forbidden"
  echo "GPU4 is reserved for the active RoboTwin run" >&2
  exit 3
fi

if [[ -e "${PIPELINE_DIR}/STATUS.training_gate_passed" ]]; then
  echo "Feasibility pipeline has already passed: ${PIPELINE_DIR}"
  exit 0
fi
if [[ -e "${PIPELINE_DIR}/STATUS.waiting_basis" ]]; then
  echo "A pre-existing waiting marker requires manual stale-owner review" >&2
  touch "${PIPELINE_DIR}/STATUS.lock_conflict"
  exit 3
fi

touch "${PIPELINE_DIR}/STATUS.waiting_basis"
echo "[$(date -Is)] waiting for Stage A: ${STAGE_A_RUN}"

dead_polls=0
missing_pid_polls=0
while [[ ! -e "${STAGE_A_RUN}/STATUS.complete" ]]; do
  if [[ -e "${STAGE_A_RUN}/STATUS.stopped" ]]; then
    mv "${PIPELINE_DIR}/STATUS.waiting_basis" "${PIPELINE_DIR}/STATUS.stage_a_stopped"
    echo "[$(date -Is)] Stage A was stopped; automatic promotion is disabled" >&2
    exit 1
  fi
  if [[ -e "${STAGE_A_RUN}/STATUS.failed" ]]; then
    mv "${PIPELINE_DIR}/STATUS.waiting_basis" "${PIPELINE_DIR}/STATUS.stage_a_failed"
    echo "[$(date -Is)] Stage A failed" >&2
    exit 1
  fi

  if [[ -f "${STAGE_A_RUN}/train.pid" ]]; then
    missing_pid_polls=0
    stage_a_pid="$(tr -dc '0-9' < "${STAGE_A_RUN}/train.pid")"
    if [[ -n "${stage_a_pid}" ]] && ! kill -0 "${stage_a_pid}" 2>/dev/null; then
      dead_polls=$((dead_polls + 1))
      if (( dead_polls >= 3 )); then
        mv "${PIPELINE_DIR}/STATUS.waiting_basis" "${PIPELINE_DIR}/STATUS.stage_a_disappeared"
        echo "[$(date -Is)] Stage A PID ${stage_a_pid} disappeared without terminal status" >&2
        exit 1
      fi
    else
      dead_polls=0
    fi
  else
    missing_pid_polls=$((missing_pid_polls + 1))
    if (( missing_pid_polls >= 3 )); then
      mv "${PIPELINE_DIR}/STATUS.waiting_basis" "${PIPELINE_DIR}/STATUS.stage_a_pid_missing"
      echo "[$(date -Is)] Stage A has no train.pid or terminal status" >&2
      exit 1
    fi
  fi
  sleep "${POLL_SECONDS}"
done

echo "[$(date -Is)] Stage A exited successfully; verifying artifacts and quality"
set +e
"${PYTHON_BIN}" "${BASIS_GATE_SCRIPT}" "${STAGE_A_RUN}" \
  --expected-steps "${EXPECTED_BASIS_STEPS}" \
  --window-steps "${GATE_WINDOW_STEPS}" \
  --json-out "${PIPELINE_DIR}/basis_gate.json" \
  > "${PIPELINE_DIR}/basis_gate.txt" 2>&1
basis_gate_rc=$?
set -e

if (( basis_gate_rc != 0 )); then
  if (( basis_gate_rc == 2 )); then
    mv "${PIPELINE_DIR}/STATUS.waiting_basis" "${PIPELINE_DIR}/STATUS.basis_gate_incomplete"
  else
    mv "${PIPELINE_DIR}/STATUS.waiting_basis" "${PIPELINE_DIR}/STATUS.basis_gate_failed"
  fi
  echo "[$(date -Is)] Stage-A gate did not pass (exit ${basis_gate_rc})" >&2
  sed -n '1,220p' "${PIPELINE_DIR}/basis_gate.txt" >&2
  exit 1
fi

mv "${PIPELINE_DIR}/STATUS.waiting_basis" "${PIPELINE_DIR}/STATUS.basis_passed"
echo "[$(date -Is)] Stage-A gate passed"
sed -n '1,220p' "${PIPELINE_DIR}/basis_gate.txt"

if [[ -e "${STAGE_B_RUN}/STATUS.running" ]]; then
  touch "${PIPELINE_DIR}/STATUS.stage_b_already_running"
  echo "[$(date -Is)] refusing to duplicate an existing Stage-B run" >&2
  exit 3
fi
if [[ -e "${STAGE_B_RUN}/STATUS.failed" || -e "${STAGE_B_RUN}/STATUS.stopped" ]]; then
  touch "${PIPELINE_DIR}/STATUS.stage_b_terminal_conflict"
  echo "[$(date -Is)] refusing to append to a failed/stopped Stage-B directory" >&2
  exit 3
fi

stage_b_run_id="$(basename "${STAGE_B_RUN%/}")"
expected_stage_b_run="playground/Checkpoints/${stage_b_run_id}"
if [[ "${STAGE_B_RUN%/}" != "${expected_stage_b_run}" ]]; then
  touch "${PIPELINE_DIR}/STATUS.stage_b_path_invalid"
  echo "STAGE_B_RUN must be under playground/Checkpoints: ${STAGE_B_RUN}" >&2
  exit 3
fi

stage_b_was_complete=0
if [[ -e "${STAGE_B_RUN}/STATUS.complete" ]]; then
  stage_b_was_complete=1
elif [[ -d "${STAGE_B_RUN}" ]] && find "${STAGE_B_RUN}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  touch "${PIPELINE_DIR}/STATUS.stage_b_nonempty_conflict"
  echo "[$(date -Is)] refusing to append to non-empty Stage-B directory ${STAGE_B_RUN}" >&2
  exit 3
fi

if (( stage_b_was_complete == 0 )); then
  resource_polls=0
  while true; do
    gpu_index="${CUDA_DEVS%%,*}"
    if free_mib="$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"; then
      :
    else
      free_mib=""
    fi
    port_busy=0
    if ss -ltn 2>/dev/null | grep -qE ":${MAIN_PORT}[[:space:]]"; then
      port_busy=1
    fi
    if [[ -n "${free_mib}" ]] && (( free_mib >= MIN_FREE_MIB )) && (( port_busy == 0 )); then
      break
    fi
    if [[ ! -e "${PIPELINE_DIR}/STATUS.waiting_resources" ]]; then
      touch "${PIPELINE_DIR}/STATUS.waiting_resources"
    fi
    resource_polls=$((resource_polls + 1))
    if (( resource_polls >= RESOURCE_MAX_POLLS )); then
      touch "${PIPELINE_DIR}/STATUS.stage_b_resources_unavailable"
      echo "[$(date -Is)] GPU${gpu_index}/port ${MAIN_PORT} unavailable after ${resource_polls} polls" >&2
      exit 1
    fi
    echo "[$(date -Is)] waiting for GPU${gpu_index}: free=${free_mib:-unknown} MiB, port_busy=${port_busy}"
    sleep "${POLL_SECONDS}"
  done
  if [[ -e "${PIPELINE_DIR}/STATUS.waiting_resources" ]]; then
    mv "${PIPELINE_DIR}/STATUS.waiting_resources" "${PIPELINE_DIR}/STATUS.resources_ready"
  fi

  mkdir -p "${STAGE_B_RUN}"
  touch "${STAGE_B_RUN}/STATUS.running"
  echo "[$(date -Is)] launching Stage B on learned, frozen coordinates"
  setsid env \
    STARVLA_DIR="${STARVLA_DIR}" \
    RUN_ID="${stage_b_run_id}" \
    CUDA_DEVS="${CUDA_DEVS}" \
    MAIN_PORT="${MAIN_PORT}" \
    /usr/bin/bash "${STAGE_B_LAUNCHER}" \
    > "${STAGE_B_RUN}/train.log" 2>&1 < /dev/null &
  stage_b_pid=$!
  printf '%s\n' "${stage_b_pid}" > "${STAGE_B_RUN}/train.pid"
  printf '%s\n' "${stage_b_pid}" > "${PIPELINE_DIR}/stage_b_supervisor.pid"

  startup_polls=0
  while kill -0 "${stage_b_pid}" 2>/dev/null && [[ ! -s "${STAGE_B_RUN}/metrics.jsonl" ]]; do
    startup_polls=$((startup_polls + 1))
    if (( startup_polls >= STARTUP_MAX_POLLS )); then
      kill -TERM -- "-${stage_b_pid}" 2>/dev/null || true
      wait "${stage_b_pid}" 2>/dev/null || true
      mv "${STAGE_B_RUN}/STATUS.running" "${STAGE_B_RUN}/STATUS.startup_failed"
      touch "${PIPELINE_DIR}/STATUS.stage_b_no_finite_metric"
      echo "[$(date -Is)] Stage B produced no metric during startup window" >&2
      exit 1
    fi
    sleep 10
  done

  set +e
  wait "${stage_b_pid}"
  stage_b_rc=$?
  set -e
  stage_b_pid=""
  if (( stage_b_rc == 0 )); then
    mv "${STAGE_B_RUN}/STATUS.running" "${STAGE_B_RUN}/STATUS.complete"
    echo "[$(date -Is)] Stage B training completed"
  else
    mv "${STAGE_B_RUN}/STATUS.running" "${STAGE_B_RUN}/STATUS.failed"
    touch "${PIPELINE_DIR}/STATUS.stage_b_failed"
    echo "[$(date -Is)] Stage B failed (exit ${stage_b_rc})" >&2
    exit "${stage_b_rc}"
  fi
else
  echo "[$(date -Is)] Stage B was already complete; evaluating its training gate"
fi

set +e
"${PYTHON_BIN}" "${PREDICTOR_GATE_SCRIPT}" "${STAGE_B_RUN}" \
  --expected-steps "${EXPECTED_PREDICTOR_STEPS}" \
  --window-steps "${PREDICTOR_GATE_WINDOW_STEPS}" --json \
  > "${PIPELINE_DIR}/predictor_training_gate.json" 2>&1
predictor_gate_rc=$?
"${PYTHON_BIN}" "${PREDICTOR_GATE_SCRIPT}" "${STAGE_B_RUN}" \
  --expected-steps "${EXPECTED_PREDICTOR_STEPS}" \
  --window-steps "${PREDICTOR_GATE_WINDOW_STEPS}" \
  > "${PIPELINE_DIR}/predictor_training_gate.txt" 2>&1
set -e

if (( predictor_gate_rc == 0 )); then
  touch "${PIPELINE_DIR}/STATUS.training_gate_passed"
  touch "${PIPELINE_DIR}/STATUS.awaiting_strict_episode_holdout"
  echo "[$(date -Is)] Stage-B training gate passed; strict split rerun/held-out remains"
  exit 0
fi

if (( predictor_gate_rc == 2 )); then
  touch "${PIPELINE_DIR}/STATUS.predictor_gate_incomplete"
else
  touch "${PIPELINE_DIR}/STATUS.predictor_gate_failed"
fi
echo "[$(date -Is)] Stage-B training gate did not pass (exit ${predictor_gate_rc})" >&2
sed -n '1,220p' "${PIPELINE_DIR}/predictor_training_gate.txt" >&2
exit 1
