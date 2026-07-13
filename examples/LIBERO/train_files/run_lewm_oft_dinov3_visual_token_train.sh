#!/bin/bash
# Train LeWM-OFT with a visual-token world model on a DINOv3 backbone.
#
# Unlike the DINOv2 variant, the encoder is loaded from a *raw* facebookresearch
# /dinov3 torchhub checkpoint (converted to HF DINOv3ViTModel on the fly). The
# DINOv3 token layout is [CLS, 4 register, patches]; LeWM.py skips the register
# tokens automatically. ViT-B/16 keeps hidden=768 -> wm_hidden=1536, identical
# to dinov2-base, so all downstream widths are unchanged. The backbone features
# differ entirely from DINOv2, so we do NOT warm-start from a DINOv2 checkpoint.

set -e

Framework_name=LeWMOFT
# Raw DINOv3 ViT-B/16 checkpoint (see dinov3_weights/). LeWM.py detects the .pth
# + "dinov3" name and routes through the dedicated converter.
base_wm=${BASE_WM:-dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}
config_yaml=./examples/LIBERO/train_files/starvla_lewm_oft_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all_wm
run_root_dir=./playground/Checkpoints
pretrained_ckpt=${PRETRAINED_CKPT:-}

run_id=${RUN_ID:-lewm_oft_libero_dinov3b_spatial4x4_trainenc1e6_statecond_ema09_200k_fullstate}
batch=${BATCH:-8}
steps=${STEPS:-200000}
save_interval=${SAVE_INTERVAL:-10000}
main_port=${MAIN_PORT:-29593}
lr=${LR:-1e-4}
base_lr=${BASE_LR:-${lr}}
action_lr=${ACTION_LR:-${lr}}
encoder_lr=${ENCODER_LR:-1e-6}
warmup=${WARMUP:-2000}
visual_tokens_per_view=${VISUAL_TOKENS_PER_VIEW:-16}
visual_token_dim=${VISUAL_TOKEN_DIM:-384}

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
  --framework.world_model.train_encoder ${TRAIN_ENCODER:-true} \
  --framework.world_model.use_visual_token_wm true \
  --framework.world_model.visual_tokens_per_view ${visual_tokens_per_view} \
  --framework.world_model.visual_token_dim ${visual_token_dim} \
  --framework.world_model.visual_token_diversity_weight ${TOKEN_DIVERSITY_WEIGHT:-0.02} \
  --framework.world_model.visual_token_variance_weight ${TOKEN_VARIANCE_WEIGHT:-0.02} \
  --framework.world_model.visual_token_min_std ${TOKEN_MIN_STD:-0.1} \
  --framework.world_model.latent_stats_momentum ${LATENT_STATS_MOMENTUM:-0.99} \
  --framework.world_model.visual_diagnostics true \
  --framework.world_model.use_state_cond ${USE_STATE_COND:-true} \
  --framework.world_model.state_cond_dim ${STATE_COND_DIM:-8} \
  --framework.world_model.state_cond_hidden_dim ${STATE_COND_HIDDEN_DIM:-256} \
  --framework.world_model.state_cond_dropout ${STATE_COND_DROPOUT:-0.1} \
  --framework.world_model.state_cond_only ${STATE_COND_ONLY:-false} \
  --framework.world_model.ctx_len 1 \
  --framework.world_model.n_future 2 \
  --framework.world_model.action_source ${ACTION_SOURCE:-oft} \
  --framework.world_model.loss_latent_weight ${LATENT_WEIGHT:-0.0} \
  --framework.world_model.loss_action_weight ${ACTION_WEIGHT:-0.5} \
  --framework.world_model.use_delta_head true \
  --framework.world_model.loss_delta_weight ${DELTA_WEIGHT:-1.0} \
  --framework.world_model.delta_head_dim ${DELTA_DIM:-384} \
  --framework.world_model.delta_head_depth ${DELTA_DEPTH:-4} \
  --framework.world_model.delta_head_heads ${DELTA_HEADS:-6} \
  --framework.world_model.delta_head_ffn ${DELTA_FFN:-1024} \
  --framework.world_model.delta_head_sigreg_weight ${DELTA_SIGREG_WEIGHT:-0.02} \
  --framework.world_model.delta_head_inference true \
  --framework.world_model.oft_future_from_delta true \
  --framework.world_model.use_state_probe false \
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
