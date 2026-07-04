#!/bin/bash
# Exp11: dense SWA (stochastic weight averaging) continuation of the baseline.
# Warm-start from the baseline steps_8000 and continue the *same* recipe
# (train everything, train_encoder=true) at the baseline LR for a short phase,
# snapshotting densely. Averaging these closely-spaced late checkpoints
# (SWA / model soup) targets a flatter minimum than any single checkpoint.
# This is the principled version of Exp10's 2-point soup (6000+8000, +1.7%).

Framework_name=LeWMOFT
base_wm=WinKawaks/vit-tiny-patch16-224
config_yaml=./examples/LIBERO/train_files/starvla_lewm_oft_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all_wm
run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-lewm_oft_libero_swa}

pretrained_ckpt=${PRETRAINED_CKPT:-playground/Checkpoints/lewm_oft_libero_wm_oft_future_aw05_state0_vitft_10k/checkpoints/steps_8000_pytorch_model.pt}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  --main_process_port ${MAIN_PORT:-29611} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.world_model.base_wm ${base_wm} \
  --framework.world_model.train_encoder true \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.future_obs_frames true \
  --datasets.vla_data.per_device_batch_size ${BATCH:-16} \
  --trainer.pretrained_checkpoint ${pretrained_ckpt} \
  --trainer.freeze_modules "" \
  --trainer.learning_rate.base ${LR:-2e-5} \
  --trainer.learning_rate.action_model ${LR:-2e-5} \
  --trainer.max_train_steps ${STEPS:-4000} \
  --trainer.num_warmup_steps ${WARMUP:-0} \
  --trainer.save_interval ${SAVE_INTERVAL:-1000} \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval ${EVAL_INTERVAL:-999999} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity your_name
