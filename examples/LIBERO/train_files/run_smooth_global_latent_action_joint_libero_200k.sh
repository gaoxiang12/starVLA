#!/usr/bin/env bash
# Joint 200k LIBERO policy: global latent world model + latent-only action head.
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"
export PATH="${STARVLA_DIR}/.venv/bin:${PATH}"

CONFIG="${CONFIG:-examples/LIBERO/train_files/starvla_smooth_global_latent_action_joint_libero_200k.yaml}"
DATA_ROOT="${DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_all_smooth_latent_wm_l10_augmented}"
RUN_ROOT="${RUN_ROOT:-playground/Checkpoints}"
RUN_ID="${RUN_ID:-lewm_oft_libero_dinov3b_smooth_global384_action_joint_200k}"
BASE_WM="${BASE_WM:-dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"
CUDA_DEVS="${CUDA_DEVS:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH="${BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
STEPS="${STEPS:-200000}"
WARMUP="${WARMUP:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
BASE_LR="${BASE_LR:-1e-4}"
MAIN_PORT="${MAIN_PORT:-29656}"
ACCELERATE_BIN="${ACCELERATE_BIN:-.venv/bin/accelerate}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "LIBERO data root not found: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${BASE_WM}" ]]; then
  echo "DINOv3 checkpoint not found: ${BASE_WM}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVS}"
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${GRAD_ACCUM}"

OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"
cp "${CONFIG}" "${OUTPUT_DIR}/launch_config.yaml"

exec "${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --framework.world_model.base_wm "${BASE_WM}" \
  --framework.world_model.train_encoder false \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --trainer.max_train_steps "${STEPS}" \
  --trainer.num_warmup_steps "${WARMUP}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --trainer.learning_rate.base "${BASE_LR}" \
  --run_root_dir "${RUN_ROOT}" \
  --run_id "${RUN_ID}"
