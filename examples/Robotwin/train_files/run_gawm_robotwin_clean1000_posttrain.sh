#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${repo_root}"

config_yaml=${CONFIG_YAML:-examples/Robotwin/train_files/starvla_gawm_robotwin_clean1000_posttrain.yaml}
run_root_dir=playground/Checkpoints
run_id=${RUN_ID:-gawm_s_base_robotwin_clean1000_49tasks_40k}
main_port=${MAIN_PORT:-29649}
batch=${BATCH:-8}
steps=${STEPS:-40000}
grad_accum=${GRADIENT_ACCUMULATION_STEPS:-8}
accelerate_bin=${ACCELERATE_BIN:-../.venvs/starVLA/bin/accelerate}
pretrained_checkpoint=${PRETRAINED_CHECKPOINT:-playground/Checkpoints/GAWM-S-base/checkpoints/steps_160000_pytorch_model.pt}
run_dir=${run_root_dir}/${run_id}

if [[ -f "${run_dir}/config.full.yaml" || -d "${run_dir}/checkpoints" ]]; then
  echo "Refusing to overwrite existing training run: ${run_dir}; choose a new RUN_ID." >&2
  exit 1
fi

if [[ ! -x "${accelerate_bin}" ]]; then
  echo "accelerate launcher not found or not executable: ${accelerate_bin}" >&2
  exit 1
fi
if [[ ! -f "${pretrained_checkpoint}" ]]; then
  echo "pretrained checkpoint not found: ${pretrained_checkpoint}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_DEVS:-4,5,6}
num_processes=${NUM_PROCESSES:-$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | wc -l)}

mkdir -p "${run_dir}"
printf '%s\n' "$$" >"${run_dir}/train.pid"
printf 'steps=%s batch_per_gpu=%s grad_accum=%s gpus=%s processes=%s pretrained=%s\n' \
  "${steps}" "${batch}" "${grad_accum}" "${CUDA_VISIBLE_DEVICES}" "${num_processes}" \
  "${pretrained_checkpoint}" >"${run_dir}/STATUS.running"

exec "${accelerate_bin}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port "${main_port}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --datasets.vla_data.per_device_batch_size "${batch}" \
  --trainer.max_train_steps "${steps}" \
  --trainer.gradient_accumulation_steps "${grad_accum}" \
  --trainer.pretrained_checkpoint "${pretrained_checkpoint}" \
  --trainer.is_resume false \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
