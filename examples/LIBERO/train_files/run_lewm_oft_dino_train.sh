#!/bin/bash
# Train StarVLA-LeWM-OFT on LIBERO with a DINOv2 vision encoder (LeWM front-end)
# + the standard MLP OFT action head (UNCHANGED). The DINOv2 encoder is frozen;
# only the world model + OFT head train. Trained from scratch (a DINO encoder
# has a different latent width than the vit-tiny baseline, so the vit-tiny
# checkpoint cannot be reused).
#
# The framework auto-derives every downstream width from the encoder hidden
# size (wm_hidden = 2 * encoder.hidden_size). For facebook/dinov2-base
# (hidden=768) that is a 1536-d LeWM latent, and the OFT head / world model /
# view-fusion resize automatically — no code changes needed.
#
# Isolated from any concurrently running job: its own run_id, log, and
# main_process_port, and pinned to a GPU subset via CUDA_VISIBLE_DEVICES.
#
# Env overrides:
#   CUDA_DEVS      GPUs to use            (default 4,5,6,7)
#   NUM_PROCESSES  #GPUs                  (default: count of CUDA_DEVS)
#   BATCH          per_device_batch_size  (default 16)
#   STEPS          max_train_steps        (default 100000)
#   SAVE_INTERVAL  checkpoint interval    (default 10000)
#   MAIN_PORT      torch rendezvous port  (default 29580)
#   RUN_ID         output run id
set -e

Framework_name=LeWMOFT
base_wm=facebook/dinov2-base
config_yaml=./examples/LIBERO/train_files/starvla_lewm_oft_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all_wm
run_root_dir=./playground/Checkpoints

run_id=${RUN_ID:-lewm_oft_libero_dinov2b_100k}
batch=${BATCH:-16}
steps=${STEPS:-100000}
save_interval=${SAVE_INTERVAL:-10000}
main_port=${MAIN_PORT:-29580}

# Pin to a GPU subset so this run does not evict the concurrently running job.
export CUDA_VISIBLE_DEVICES=${CUDA_DEVS:-4,5,6,7}
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
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.future_obs_frames true \
  --datasets.vla_data.per_device_batch_size ${batch} \
  --trainer.is_resume ${IS_RESUME:-false} \
  --trainer.max_train_steps ${steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity your_name
