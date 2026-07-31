#!/usr/bin/env bash
# Train spatial-token DINOv2-base LeWM-OFT on RoboTwin clean + randomized data.
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"
export PATH="${STARVLA_DIR}/.venv/bin:${PATH}"

CONFIG="${CONFIG:-examples/Robotwin/train_files/starvla_lewm_oft_dinov2b_robotwin.yaml}"
DATA_ROOT="${DATA_ROOT:-playground/Datasets/RoboTwin}"
DATA_MIX="${DATA_MIX:-robotwin_all_wm}"
RUN_ROOT="${RUN_ROOT:-playground/Checkpoints}"
RUN_ID="${RUN_ID:-lewm_oft_robotwin_dinov2b_spatial4x4_200k}"
BASE_WM="${BASE_WM:-facebook/dinov2-base}"
CUDA_DEVS="${CUDA_DEVS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-$(tr ',' '\n' <<<"${CUDA_DEVS}" | wc -l)}"
BATCH="${BATCH:-8}"
STEPS="${STEPS:-200000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
WARMUP="${WARMUP:-2000}"
BASE_LR="${BASE_LR:-1e-4}"
ACTION_LR="${ACTION_LR:-1e-4}"
ENCODER_LR="${ENCODER_LR:-1e-6}"
MAIN_PORT="${MAIN_PORT:-29630}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "RoboTwin data root not found: ${DATA_ROOT}" >&2
  echo "Run examples/Robotwin/data_preparation.py first." >&2
  exit 1
fi
if ! command -v "${ACCELERATE_BIN}" >/dev/null 2>&1; then
  echo "accelerate not found: ${ACCELERATE_BIN}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVS}"

OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

pretrained_args=()
if [[ -n "${PRETRAINED_CKPT}" ]]; then
  if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
    echo "Pretrained checkpoint not found: ${PRETRAINED_CKPT}" >&2
    exit 1
  fi
  pretrained_args+=(--trainer.pretrained_checkpoint "${PRETRAINED_CKPT}")
fi

"${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --framework.name LeWMOFT \
  --framework.world_model.base_wm "${BASE_WM}" \
  --framework.world_model.train_encoder "${TRAIN_ENCODER:-true}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.future_obs_frames true \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  "${pretrained_args[@]}" \
  --trainer.is_resume "${IS_RESUME:-false}" \
  --trainer.repair_lr_scheduler_on_resume "${REPAIR_LR_SCHEDULER_ON_RESUME:-false}" \
  --trainer.freeze_modules "${FREEZE_MODULES:-}" \
  --trainer.max_train_steps "${STEPS}" \
  --trainer.num_warmup_steps "${WARMUP}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.learning_rate.base "${BASE_LR}" \
  --trainer.learning_rate.action_model "${ACTION_LR}" \
  --trainer.learning_rate.backbone.encoder "${ENCODER_LR}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY:-100}" \
  --trainer.eval_interval "${EVAL_INTERVAL:-2000}" \
  --run_root_dir "${RUN_ROOT}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT:-starVLA_RoboTwin}" \
  --wandb_entity "${WANDB_ENTITY:-your_name}"