#!/usr/bin/env bash
# Train a zero-initialized temporal/action/state residual on top of the frozen
# DINOv2-base action-free delta head. This is a 5k gate, not a long production run.

set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"
export PATH="${STARVLA_DIR}/.venv/bin:${PATH}"

BASE_RUN="playground/Checkpoints/lewm_oft_libero_dinov2b_spatial4x4_trainenc1e6_statecond_ema09_200k_fullstate_v2"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-${BASE_RUN}/checkpoints/steps_160000_pytorch_model.pt}"
RUN_ID="${RUN_ID:-lewm_oft_libero_dinov2b_temporal_residual_ctx2_h8_from160k_5k}"
STEPS="${STEPS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
BATCH="${BATCH:-8}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-200}"
CUDA_DEVS="${CUDA_DEVS:-0,1,2,3}"
MAIN_PORT="${MAIN_PORT:-29791}"

if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
  echo "Checkpoint not found: ${PRETRAINED_CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVS}"
NUM_PROCESSES="${NUM_PROCESSES:-$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | wc -l)}"

# The original action-free predictor is the immutable baseline. Only the new
# world_model.delta_residual_head remains trainable.
FREEZE_MODULES="backbone.encoder,visual_token_pooler,world_model.predictor,world_model.delta_head,action_model,visual_action_head,action_query_proj,future_action_context_proj,view_fuse,task_embedding"

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
  --framework.world_model.ctx_len 2 \
  --framework.world_model.n_future 2 \
  --framework.world_model.world_model_only true \
  --framework.world_model.action_source oft \
  --framework.world_model.use_delta_head true \
  --framework.world_model.delta_head_type transformer \
  --framework.world_model.delta_head_condition_on_action false \
  --framework.world_model.delta_temporal_residual true \
  --framework.world_model.delta_temporal_depth "${TEMPORAL_DEPTH:-2}" \
  --framework.world_model.delta_head_inference true \
  --framework.world_model.oft_future_from_delta true \
  --framework.world_model.loss_delta_weight 1.0 \
  --framework.world_model.delta_head_sigreg_weight 0.0 \
  --framework.world_model.loss_latent_weight 0.0 \
  --framework.world_model.loss_action_weight 0.0 \
  --framework.world_model.visual_token_diversity_weight 0.0 \
  --framework.world_model.visual_token_variance_weight 0.0 \
  --framework.world_model.use_state_cond true \
  --framework.world_model.state_cond_dim 8 \
  --framework.world_model.state_cond_dropout 0.0 \
  --framework.world_model.use_state_probe false \
  --framework.action_model.action_horizon 8 \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_all_wm_ctx2_h8 \
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
