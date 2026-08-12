#!/usr/bin/env bash
# Continue the 200k smooth-global state-conditioned train-encoder policy to
# step 300k on four GPUs. The source checkpoint was produced with one-GPU
# ZeRO-2, whose optimizer state cannot be repartitioned to four GPUs by the
# installed DeepSpeed. This launcher therefore resumes model weights and step
# numbering, while rebuilding AdamW and positioning the 300k cosine schedule
# at step 200k.
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"
export PATH="${STARVLA_DIR}/.venv/bin:${PATH}"

CONFIG="${CONFIG:-examples/LIBERO/train_files/starvla_smooth_global_latent_action_joint_libero_statecond_trainenc_200k.yaml}"
DATA_ROOT="${DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_all_smooth_latent_wm_l10_augmented_statecond}"
RUN_ROOT="${RUN_ROOT:-playground/Checkpoints}"
RUN_ID="${RUN_ID:-lewm_oft_libero_dinov3b_smooth_global384_statecond_trainenc_continue200k_to300k}"
SOURCE_CKPT="${SOURCE_CKPT:-playground/Checkpoints/lewm_oft_libero_dinov3b_smooth_global384_statecond_trainenc_200k/checkpoints/steps_200000_pytorch_model.pt}"
BASE_WM="${BASE_WM:-dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"
CUDA_DEVS="${CUDA_DEVS:-0,4,5,6}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH="${BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
STEPS="${STEPS:-300000}"
WARMUP="${WARMUP:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
BASE_LR="${BASE_LR:-1e-4}"
ENCODER_LR="${ENCODER_LR:-1e-6}"
MAIN_PORT="${MAIN_PORT:-29667}"
ACCELERATE_BIN="${ACCELERATE_BIN:-.venv/bin/accelerate}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "LIBERO data root not found: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${BASE_WM}" ]]; then
  echo "DINOv3 checkpoint not found: ${BASE_WM}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_CKPT}" ]]; then
  echo "Source checkpoint not found: ${SOURCE_CKPT}" >&2
  exit 1
fi
GLOBAL_BATCH=$(( NUM_PROCESSES * BATCH * GRAD_ACCUM ))
if (( GLOBAL_BATCH != 32 )); then
  echo "This continuation requires global batch 32 (NUM_PROCESSES x BATCH x GRAD_ACCUM), got ${GLOBAL_BATCH} (${NUM_PROCESSES} x ${BATCH} x ${GRAD_ACCUM})." >&2
  exit 2
fi

OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
RESUME_CKPT="${OUTPUT_DIR}/checkpoints/steps_200000_pytorch_model.pt"
mkdir -p "${OUTPUT_DIR}/checkpoints"
if [[ -e "${RESUME_CKPT}" ]]; then
  if [[ "$(realpath "${RESUME_CKPT}")" != "$(realpath "${SOURCE_CKPT}")" ]]; then
    echo "Refusing to replace an existing, different resume checkpoint: ${RESUME_CKPT}" >&2
    exit 3
  fi
else
  ln -s "$(realpath "${SOURCE_CKPT}")" "${RESUME_CKPT}"
fi
cp "$0" "${OUTPUT_DIR}/"
cp "${CONFIG}" "${OUTPUT_DIR}/launch_config.yaml"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVS}"
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${GRAD_ACCUM}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export STARVLA_DISABLE_TQDM="${STARVLA_DISABLE_TQDM:-1}"

exec "${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --framework.world_model.base_wm "${BASE_WM}" \
  --framework.world_model.train_encoder true \
  --framework.world_model.use_state_cond true \
  --framework.world_model.state_cond_dim 8 \
  --framework.world_model.state_cond_hidden_dim 256 \
  --framework.world_model.state_cond_dropout 0.1 \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --trainer.is_resume true \
  --trainer.repair_lr_scheduler_on_resume false \
  --trainer.max_train_steps "${STEPS}" \
  --trainer.num_warmup_steps "${WARMUP}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --trainer.learning_rate.base "${BASE_LR}" \
  --trainer.learning_rate.backbone.encoder "${ENCODER_LR}" \
  --run_root_dir "${RUN_ROOT}" \
  --run_id "${RUN_ID}"
