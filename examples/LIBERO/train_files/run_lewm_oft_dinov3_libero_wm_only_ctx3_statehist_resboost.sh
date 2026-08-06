#!/usr/bin/env bash
# Action-free LIBERO latent residual boosting with longer causal history.
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"
export PATH="${STARVLA_DIR}/.venv/bin:${PATH}"

CONFIG="${CONFIG:-examples/LIBERO/train_files/starvla_lewm_oft_dinov3_libero_wm_only_ctx3_statehist_resboost.yaml}"
RUN_ID="${RUN_ID:-lewm_oft_libero_dinov3b_wmonly_ctx3_statehist3_resboost_from220k_50k}"
CUDA_DEVS="${CUDA_DEVS:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH="${BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
STEPS="${STEPS:-50000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
WARMUP="${WARMUP:-500}"
MAIN_PORT="${MAIN_PORT:-29643}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVS}"
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${GRAD_ACCUM}"

OUTPUT_DIR="playground/Checkpoints/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

"${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --trainer.max_train_steps "${STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.num_warmup_steps "${WARMUP}" \
  --run_id "${RUN_ID}"
