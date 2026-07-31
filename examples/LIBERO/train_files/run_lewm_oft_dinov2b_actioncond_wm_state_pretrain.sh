#!/usr/bin/env bash
# Warm-start the DINOv2-base spatial-token checkpoint and train only future
# latent dynamics. The policy/action path and legacy joint flow predictor stay
# frozen; expert macro-actions condition the deterministic delta world model.

set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"
export PATH="${STARVLA_DIR}/.venv/bin:${PATH}"

BASE_RUN="playground/Checkpoints/lewm_oft_libero_dinov2b_spatial4x4_trainenc1e6_statecond_ema09_200k_fullstate_v2"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-${BASE_RUN}/checkpoints/steps_160000_pytorch_model.pt}"
RUN_ID="${RUN_ID:-lewm_oft_libero_dinov2b_actioncond_wm_state_from160k_50k}"
STEPS="${STEPS:-50000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
BATCH="${BATCH:-8}"
LR="${LR:-3e-5}"
WARMUP="${WARMUP:-1000}"
CUDA_DEVS="${CUDA_DEVS:-0,1,2,3}"
MAIN_PORT="${MAIN_PORT:-29790}"
TRAIN_POOLER="${TRAIN_POOLER:-false}"

if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
  echo "Checkpoint not found: ${PRETRAINED_CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVS}"
NUM_PROCESSES="${NUM_PROCESSES:-$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | wc -l)}"

FREEZE_MODULES="backbone.encoder,world_model.predictor,action_model,visual_action_head,action_query_proj,future_action_context_proj,view_fuse"
if [[ "${TRAIN_POOLER}" != "true" ]]; then
  FREEZE_MODULES="${FREEZE_MODULES},visual_token_pooler"
fi

OUTPUT_DIR="playground/Checkpoints/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LIBERO/train_files/starvla_lewm_oft_libero.yaml \
  --framework.name LeWMOFT \
  --framework.world_model.base_wm facebook/dinov2-base \
  --framework.world_model.train_encoder false \
  --framework.world_model.use_visual_token_wm true \
  --framework.world_model.visual_tokens_per_view 16 \
  --framework.world_model.visual_token_dim 384 \
  --framework.world_model.ctx_len 1 \
  --framework.world_model.n_future 2 \
  --framework.world_model.world_model_only true \
  --framework.world_model.action_source oft \
  --framework.world_model.use_delta_head true \
  --framework.world_model.delta_head_type transformer \
  --framework.world_model.delta_head_condition_on_action true \
  --framework.world_model.delta_head_inference true \
  --framework.world_model.oft_future_from_delta true \
  --framework.world_model.loss_delta_weight "${DELTA_WEIGHT:-1.0}" \
  --framework.world_model.delta_head_sigreg_weight "${DELTA_SIGREG_WEIGHT:-0.0}" \
  --framework.world_model.loss_latent_weight 0.0 \
  --framework.world_model.loss_action_weight 0.0 \
  --framework.world_model.use_state_cond false \
  --framework.world_model.use_state_probe true \
  --framework.world_model.state_dim 8 \
  --framework.world_model.loss_state_current_weight "${STATE_CURRENT_WEIGHT:-0.1}" \
  --framework.world_model.loss_state_future_weight "${STATE_FUTURE_WEIGHT:-0.1}" \
  --framework.world_model.loss_state_delta_weight "${STATE_DELTA_WEIGHT:-0.1}" \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_all_wm \
  --datasets.vla_data.future_obs_frames true \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --trainer.pretrained_checkpoint "${PRETRAINED_CKPT}" \
  --trainer.is_resume false \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.learning_rate.base "${LR}" \
  --trainer.learning_rate.action_model "${LR}" \
  --trainer.num_warmup_steps "${WARMUP}" \
  --trainer.max_train_steps "${STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY:-50}" \
  --trainer.eval_interval 1000 \
  --run_root_dir playground/Checkpoints \
  --run_id "${RUN_ID}" \
  --wandb_project starVLA_Libero \
  --wandb_entity your_name