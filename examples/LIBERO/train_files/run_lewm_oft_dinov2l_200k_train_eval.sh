#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
RUN_ID="${RUN_ID:-lewm_oft_libero_dinov2l_spatial4x4_trainenc1e6_statecond_ema09_200k_fullstate}"
STEPS="${STEPS:-200000}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
MAIN_PORT="${MAIN_PORT:-29770}"
EVAL_BASE_PORT="${EVAL_BASE_PORT:-29780}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
SEED="${SEED:-7}"

cd "${STARVLA_DIR}"
export PATH="${STARVLA_DIR}/.venv/bin:${PATH}"

env \
  RUN_ID="${RUN_ID}" \
  STEPS="${STEPS}" \
  CUDA_DEVS="${TRAIN_GPUS}" \
  MAIN_PORT="${MAIN_PORT}" \
  BASE_WM="facebook/dinov2-large" \
  TRAIN_ENCODER="${TRAIN_ENCODER:-true}" \
  ENCODER_LR="${ENCODER_LR:-1e-6}" \
  USE_STATE_COND="${USE_STATE_COND:-true}" \
  INCLUDE_STATE="${INCLUDE_STATE:-true}" \
  LATENT_STATS_MOMENTUM="${LATENT_STATS_MOMENTUM:-0.99}" \
  bash examples/LIBERO/train_files/run_lewm_oft_dino_visual_token_train.sh

checkpoint="${STARVLA_DIR}/playground/Checkpoints/${RUN_ID}/checkpoints/steps_${STEPS}_pytorch_model.pt"
if [[ ! -f "${checkpoint}" ]]; then
  echo "Expected checkpoint not found: ${checkpoint}" >&2
  exit 1
fi

IFS=',' read -r -a eval_gpus <<<"${EVAL_GPUS}"
suites=(libero_spatial libero_object libero_goal libero_10)
if (( ${#eval_gpus[@]} < ${#suites[@]} )); then
  echo "EVAL_GPUS must provide at least ${#suites[@]} GPUs" >&2
  exit 2
fi

pids=()
for index in "${!suites[@]}"; do
  suite="${suites[$index]}"
  gpu="${eval_gpus[$index]}"
  port=$((EVAL_BASE_PORT + index))
  TASK_SUITE_NAME="${suite}" \
    bash examples/LIBERO/eval_files/run_checkpoint_eval.sh \
      "${checkpoint}" "${gpu}" "${port}" "${NUM_TRIALS_PER_TASK}" "${SEED}" &
  pids+=("$!")
done

eval_status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    eval_status=1
  fi
done
if (( eval_status != 0 )); then
  echo "At least one LIBERO evaluation failed" >&2
  exit "${eval_status}"
fi

summary="${STARVLA_DIR}/playground/Checkpoints/${RUN_ID}/libero_success_rates_steps_${STEPS}.txt"
: >"${summary}"
for suite in "${suites[@]}"; do
  log_dir="${STARVLA_DIR}/playground/Checkpoints/${RUN_ID}/logs/${suite}_10x10"
  log_file="$(find "${log_dir}" -maxdepth 1 -type f -name "steps_${STEPS}_pytorch_model_${suite}_*.log" ! -name '*_server.log' | sort | tail -n 1)"
  success_rate="$(grep 'Total success rate:' "${log_file}" | tail -n 1 | sed 's/^.*Total success rate: //')"
  printf '%s: %s\n' "${suite}" "${success_rate}" | tee -a "${summary}"
done

echo "Success-rate summary: ${summary}"