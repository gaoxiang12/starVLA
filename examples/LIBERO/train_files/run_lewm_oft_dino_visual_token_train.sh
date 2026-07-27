#!/bin/bash
# Train LeWM-OFT with spatially anchored DINO tokens. Each camera's patch grid
# is pooled to a fixed 4x4 grid and an action-free delta head predicts future
# token residuals. This architecture is incompatible with old learned-query
# visual-token checkpoints; use a new RUN_ID.

set -e

Framework_name=LeWMOFT
base_wm=${BASE_WM:-facebook/dinov2-base}
config_yaml=./examples/LIBERO/train_files/starvla_lewm_oft_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=${DATA_MIX:-libero_all_wm}
run_root_dir=./playground/Checkpoints
pretrained_ckpt=${PRETRAINED_CKPT:-}

run_id=${RUN_ID:-lewm_oft_libero_dinov2b_spatial4x4_delta_dim384_100k}
batch=${BATCH:-8}
steps=${STEPS:-100000}
save_interval=${SAVE_INTERVAL:-10000}
main_port=${MAIN_PORT:-29592}
lr=${LR:-1e-4}
base_lr=${BASE_LR:-${lr}}
action_lr=${ACTION_LR:-${lr}}
encoder_lr=${ENCODER_LR:-1e-6}
warmup=${WARMUP:-2000}
visual_tokens_per_view=${VISUAL_TOKENS_PER_VIEW:-16}
visual_token_dim=${VISUAL_TOKEN_DIM:-384}
action_horizon=${ACTION_HORIZON:-8}
ctx_len=${CTX_LEN:-1}
n_future=${N_FUTURE:-2}

export CUDA_VISIBLE_DEVICES=${CUDA_DEVS:-0,1,2,3}
num_processes=${NUM_PROCESSES:-$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

pretrained_args=()
if [[ -n "${pretrained_ckpt}" ]]; then
  pretrained_args+=(--trainer.pretrained_checkpoint ${pretrained_ckpt})
fi

${ACCELERATE_BIN:-accelerate} launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  --main_process_port ${main_port} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.world_model.base_wm ${base_wm} \
  --framework.world_model.train_encoder ${TRAIN_ENCODER:-false} \
  --framework.world_model.visual_tokens_per_view ${visual_tokens_per_view} \
  --framework.world_model.visual_token_dim ${visual_token_dim} \
  --framework.world_model.visual_token_diversity_weight ${TOKEN_DIVERSITY_WEIGHT:-0.02} \
  --framework.world_model.visual_token_variance_weight ${TOKEN_VARIANCE_WEIGHT:-0.02} \
  --framework.world_model.visual_token_min_std ${TOKEN_MIN_STD:-0.1} \
  --framework.world_model.latent_stats_momentum ${LATENT_STATS_MOMENTUM:-0.99} \
  --framework.world_model.visual_diagnostics true \
  --framework.world_model.use_state_cond ${USE_STATE_COND:-false} \
  --framework.world_model.state_cond_dim ${STATE_COND_DIM:-8} \
  --framework.world_model.state_cond_hidden_dim ${STATE_COND_HIDDEN_DIM:-256} \
  --framework.world_model.state_cond_dropout ${STATE_COND_DROPOUT:-0.1} \
  --framework.world_model.state_cond_only ${STATE_COND_ONLY:-false} \
  --framework.world_model.ctx_len ${ctx_len} \
  --framework.world_model.n_future ${n_future} \
  --framework.world_model.loss_latent_weight ${LATENT_WEIGHT:-1.0} \
  --framework.world_model.residual_predictor_dim ${RESIDUAL_DIM:-384} \
  --framework.world_model.residual_predictor_depth ${RESIDUAL_DEPTH:-4} \
  --framework.world_model.residual_predictor_heads ${RESIDUAL_HEADS:-6} \
  --framework.world_model.residual_predictor_ffn ${RESIDUAL_FFN:-1024} \
  --framework.world_model.residual_predictor_sigreg_weight ${RESIDUAL_SIGREG_WEIGHT:-0.0} \
  --framework.world_model.use_state_probe false \
  --framework.action_model.action_horizon ${action_horizon} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.future_obs_frames true \
  --datasets.vla_data.include_state ${INCLUDE_STATE:-true} \
  --datasets.vla_data.per_device_batch_size ${batch} \
  "${pretrained_args[@]}" \
  --trainer.is_resume ${IS_RESUME:-false} \
  --trainer.repair_lr_scheduler_on_resume ${REPAIR_LR_SCHEDULER_ON_RESUME:-false} \
  --trainer.max_train_steps ${steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.freeze_modules "${FREEZE_MODULES:-}" \
  --trainer.num_warmup_steps ${warmup} \
  --trainer.learning_rate.base ${base_lr} \
  --trainer.learning_rate.action_model ${action_lr} \
  --trainer.learning_rate.backbone.encoder ${encoder_lr} \
  --trainer.logging_frequency ${LOGGING_FREQUENCY:-100} \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity your_name
