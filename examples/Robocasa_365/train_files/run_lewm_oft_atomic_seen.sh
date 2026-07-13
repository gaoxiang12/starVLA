#!/usr/bin/env bash
# Train DINOv3 LeWM-OFT on the 18 RoboCasa365 Atomic-Seen target/human tasks.
set -euo pipefail

command -v accelerate >/dev/null || {
  echo "accelerate not found; activate the starVLA Python environment first" >&2
  exit 1
}

CONFIG=./examples/Robocasa_365/train_files/starvla_lewm_oft_robocasa365_atomic_seen.yaml
RUN_ROOT=${RUN_ROOT:-./playground/Checkpoints}
RUN_ID=${RUN_ID:-lewm_oft_robocasa365_atomic_seen}
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch; print(torch.cuda.device_count())")}
BATCH=${BATCH:-8}
STEPS=${STEPS:-200000}
SAVE_EVERY=${SAVE_EVERY:-10000}
MAIN_PORT=${MAIN_PORT:-29593}

mkdir -p "${RUN_ROOT}/${RUN_ID}"
cp "$0" "${RUN_ROOT}/${RUN_ID}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MAIN_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --trainer.max_train_steps "${STEPS}" \
  --trainer.save_interval "${SAVE_EVERY}" \
  --run_root_dir "${RUN_ROOT}" \
  --run_id "${RUN_ID}" \
  --wandb_project starVLA_robocasa365