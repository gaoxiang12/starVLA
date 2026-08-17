#!/usr/bin/env bash
set -euo pipefail

config_yaml=examples/UnifiedPretrain/train_files/starvla_lewm_unified_pretrain.yaml
run_root_dir=playground/Checkpoints
run_id=${RUN_ID:-starvla_lewm_unified_pretrain}
main_port=${MAIN_PORT:-29620}
batch=${BATCH:-8}
steps=${STEPS:-200000}
accelerate_bin=${ACCELERATE_BIN:-.venv/bin/accelerate}

if [[ ! -x "${accelerate_bin}" ]]; then
  echo "accelerate launcher not found or not executable: ${accelerate_bin}" >&2
  echo "Set ACCELERATE_BIN or create the repository .venv first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_DEVS:-0,1,2,3,4,5,6,7}
num_processes=${NUM_PROCESSES:-$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | wc -l)}

"${accelerate_bin}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port "${main_port}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --datasets.vla_data.per_device_batch_size "${batch}" \
  --trainer.max_train_steps "${steps}" \
  --trainer.is_resume "${IS_RESUME:-false}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
