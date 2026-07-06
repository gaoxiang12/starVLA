#!/bin/bash
# Exp9: proprioceptive state conditioning of the OFT head.
# Warm-start from the 83% baseline (steps_8000), freeze the entire perception
# stack AND the action head, and train ONLY the zero-init state_encoder. The
# encoder learns an additive proprio residual on the action queries; since it
# starts as a no-op, the lower bound is exactly the baseline (preserves the
# fragile wine-on-rack behaviour that collapses under head finetuning).

Framework_name=LeWMOFT
base_wm=WinKawaks/vit-tiny-patch16-224
config_yaml=./examples/LIBERO/train_files/starvla_lewm_oft_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all_wm
run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-lewm_oft_libero_state_cond}

pretrained_ckpt=${PRETRAINED_CKPT:-playground/Checkpoints/lewm_oft_libero_wm_oft_future_aw05_state0_vitft_10k/checkpoints/steps_8000_pytorch_model.pt}
# Freeze everything except the new state_encoder.
freeze="backbone,world_model,view_fuse,task_embedding,state_probe,action_model,action_query_proj,future_action_context_proj"

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  --main_process_port ${MAIN_PORT:-29601} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.world_model.base_wm ${base_wm} \
  --framework.world_model.train_encoder false \
  --framework.world_model.use_state_cond true \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.future_obs_frames true \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.per_device_batch_size ${BATCH:-16} \
  --trainer.pretrained_checkpoint ${pretrained_ckpt} \
  --trainer.freeze_modules "${freeze}" \
  --trainer.max_train_steps ${STEPS:-10000} \
  --trainer.num_warmup_steps ${WARMUP:-500} \
  --trainer.save_interval ${SAVE_INTERVAL:-2000} \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity your_name
