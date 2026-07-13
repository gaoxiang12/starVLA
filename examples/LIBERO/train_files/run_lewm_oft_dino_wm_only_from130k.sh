#!/bin/bash
# Continue the DINOv2-base LeWM-OFT run from steps_130000, but train only the
# latent/action flow world model. The OFT action path is frozen and excluded
# from the loss via action_source=flow, so this is a clean WM-only diagnostic.

set -e

Framework_name=LeWMOFT
base_wm=facebook/dinov2-base
config_yaml=./examples/LIBERO/train_files/starvla_lewm_oft_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all_wm
run_root_dir=./playground/Checkpoints
pretrained_ckpt=${PRETRAINED_CKPT:-playground/Checkpoints/lewm_oft_libero_dinov2b_100k/checkpoints/steps_130000_pytorch_model.pt}

run_id=${RUN_ID:-lewm_oft_libero_dinov2b_wm_only_from130k}
batch=${BATCH:-16}
steps=${STEPS:-30000}
save_interval=${SAVE_INTERVAL:-5000}
main_port=${MAIN_PORT:-29591}
lr=${LR:-3e-5}
warmup=${WARMUP:-500}

export CUDA_VISIBLE_DEVICES=${CUDA_DEVS:-0,1,2,3}
num_processes=${NUM_PROCESSES:-$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  --main_process_port ${main_port} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.world_model.base_wm ${base_wm} \
  --framework.world_model.train_encoder false \
  --framework.world_model.action_source flow \
  --framework.world_model.loss_latent_weight ${LATENT_WEIGHT:-1.0} \
  --framework.world_model.loss_action_weight ${ACTION_WEIGHT:-0.5} \
  --framework.world_model.use_state_probe false \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.future_obs_frames true \
  --datasets.vla_data.per_device_batch_size ${batch} \
  --trainer.pretrained_checkpoint ${pretrained_ckpt} \
  --trainer.is_resume false \
  --trainer.freeze_modules "backbone,action_model,action_query_proj,future_action_context_proj" \
  --trainer.learning_rate.base ${lr} \
  --trainer.learning_rate.action_model ${lr} \
  --trainer.num_warmup_steps ${warmup} \
  --trainer.max_train_steps ${steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity your_name