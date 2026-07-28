#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "Usage: $0 CKPT GPU_IDS [TRIALS_PER_TASK=1] [SEED=7] [EXECUTE_HORIZON=8] [PROGRESS_EMA=0.4]"
  echo "GPU_IDS must contain three comma-separated GPUs for learned,disabled,fixed."
  exit 2
fi

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
CKPT="$(realpath "$1")"
GPU_IDS="$2"
TRIALS_PER_TASK="${3:-1}"
SEED="${4:-7}"
EXECUTE_HORIZON="${5:-8}"
PROGRESS_EMA="${6:-0.4}"
FIXED_PROGRESS="${FIXED_PROGRESS:-0.5}"
BASE_PORT="${BASE_PORT:-30300}"
AB_RUN_VARIANT="${AB_RUN_VARIANT:-}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

IFS=',' read -r -a gpus <<<"${GPU_IDS}"
modes=(learned disabled fixed)
suites=(libero_spatial libero_object libero_goal libero_10)
if (( ${#gpus[@]} != ${#modes[@]} )); then
  echo "Expected exactly ${#modes[@]} GPUs, got ${#gpus[@]}: ${GPU_IDS}" >&2
  exit 2
fi

model_root="${CKPT%%/checkpoints/*}"
run_tag="${TRIALS_PER_TASK}trial_seed${SEED}_exec${EXECUTE_HORIZON}_ema${PROGRESS_EMA}"
if [[ -n "${AB_RUN_VARIANT}" ]]; then
  run_tag="${run_tag}_${AB_RUN_VARIANT}"
fi
ab_dir="${model_root}/progress_closed_loop_ab/${run_tag}"
mkdir -p "${ab_dir}"

printf '%s\n' \
  "checkpoint=${CKPT}" \
  "gpus=${GPU_IDS}" \
  "trials_per_task=${TRIALS_PER_TASK}" \
  "seed=${SEED}" \
  "execute_horizon=${EXECUTE_HORIZON}" \
  "progress_ema=${PROGRESS_EMA}" \
  "fixed_progress=${FIXED_PROGRESS}" \
  >"${ab_dir}/run_config.txt"

pids=()
for mode_index in "${!modes[@]}"; do
  mode="${modes[$mode_index]}"
  gpu="${gpus[$mode_index]}"
  mode_log="${ab_dir}/${mode}.log"
  (
    for suite_index in "${!suites[@]}"; do
      suite="${suites[$suite_index]}"
      port=$((BASE_PORT + mode_index * 10 + suite_index))
      TASK_SUITE_NAME="${suite}" \
      PROGRESS_MODE="${mode}" \
      PROGRESS_EMA="${PROGRESS_EMA}" \
      FIXED_PROGRESS="${FIXED_PROGRESS}" \
      RUN_VARIANT="${AB_RUN_VARIANT}" \
        bash "${STARVLA_DIR}/examples/LIBERO/eval_files/run_checkpoint_eval.sh" \
          "${CKPT}" "${gpu}" "${port}" "${TRIALS_PER_TASK}" "${SEED}" \
          "${EXECUTE_HORIZON}"
    done
  ) >"${mode_log}" 2>&1 &
  mode_pid=$!
  pids+=("${mode_pid}")
  printf '%s\n' "${mode_pid}" >"${ab_dir}/${mode}.pid"
  echo "[progress-ab] mode=${mode} gpu=${gpu} pid=${mode_pid} log=${mode_log}"
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[progress-ab] mode=${modes[$index]} failed" >&2
    status=1
  fi
done

summary="${ab_dir}/success_rates.txt"
: >"${summary}"
for mode in "${modes[@]}"; do
  echo "${mode}:" >>"${summary}"
  grep "Total success rate:" "${ab_dir}/${mode}.log" >>"${summary}" || true
done
cat "${summary}"
exit "${status}"
