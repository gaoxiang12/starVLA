#!/usr/bin/env bash
# Stage A: select compact local-increment coordinates on frozen LIBERO features.
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"
export PATH="${STARVLA_DIR}/.venv/bin:${PATH}"

CONFIG="${CONFIG:-examples/LIBERO/train_files/starvla_lewm_oft_dinov3_libero_wm_only_local_basis_2k.yaml}"
RUN_ID="${RUN_ID:-lewm_oft_libero_dinov3b_wmonly_localbasis_m4r64_fixedmean_2k}"
CUDA_DEVS="${CUDA_DEVS:-0}"
MAIN_PORT="${MAIN_PORT:-29649}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVS}"
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${GRAD_ACCUM:-8}"

OUTPUT_DIR="playground/Checkpoints/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

"${ACCELERATE_BIN:-accelerate}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  --main_process_port "${MAIN_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --datasets.vla_data.per_device_batch_size "${BATCH:-4}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --trainer.max_train_steps "${STEPS:-2000}" \
  --trainer.save_interval "${SAVE_INTERVAL:-2000}" \
  --trainer.num_warmup_steps "${WARMUP:-100}" \
  --run_id "${RUN_ID}"
