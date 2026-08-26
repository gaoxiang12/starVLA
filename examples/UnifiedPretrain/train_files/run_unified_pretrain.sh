#!/usr/bin/env bash
set -euo pipefail

config_yaml=examples/UnifiedPretrain/train_files/starvla_lewm_unified_pretrain.yaml
run_root_dir=playground/Checkpoints
run_id=${RUN_ID:-starvla_lewm_unified_pretrain}
main_port=${MAIN_PORT:-29620}
batch=${BATCH:-8}
steps=${STEPS:-200000}
accelerate_bin=${ACCELERATE_BIN:-.venv/bin/accelerate}
pretrained_checkpoint=${PRETRAINED_CHECKPOINT:-}
is_resume=${IS_RESUME:-false}
run_dir=${run_root_dir}/${run_id}

if [[ ! -x "${accelerate_bin}" ]]; then
  echo "accelerate launcher not found or not executable: ${accelerate_bin}" >&2
  echo "Set ACCELERATE_BIN or create the repository .venv first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_DEVS:-0,1,2,3,4,5,6,7}
num_processes=${NUM_PROCESSES:-$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | wc -l)}

trainer_args=(
  --trainer.is_resume "${is_resume}"
)
if [[ -n "${DATA_MIX:-}" ]]; then
  trainer_args+=(--datasets.vla_data.data_mix "${DATA_MIX}")
fi
if [[ -n "${EMBODIMENT_SAMPLING_WEIGHTS:-}" ]]; then
  trainer_args+=(
    --datasets.vla_data.embodiment_sampling_weights
    "${EMBODIMENT_SAMPLING_WEIGHTS}"
  )
fi
if [[ -n "${NUM_WORKERS:-}" ]]; then
  trainer_args+=(--datasets.vla_data.num_workers "${NUM_WORKERS}")
fi
if [[ -n "${LOGGING_FREQUENCY:-}" ]]; then
  trainer_args+=(--trainer.logging_frequency "${LOGGING_FREQUENCY}")
fi
if [[ -n "${WARMUP_STEPS:-}" ]]; then
  trainer_args+=(--trainer.num_warmup_steps "${WARMUP_STEPS}")
fi
if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  trainer_args+=(
    --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  )
fi
if [[ -n "${pretrained_checkpoint}" ]]; then
  trainer_args+=(--trainer.pretrained_checkpoint "${pretrained_checkpoint}")
fi

mkdir -p "${run_dir}"
printf '%s\n' "$$" >"${run_dir}/train.pid"
printf 'steps=%s batch_per_gpu=%s gpus=%s resume=%s pretrained=%s\n' \
  "${steps}" "${batch}" "${num_processes}" "${is_resume}" "${pretrained_checkpoint:-none}" \
  >"${run_dir}/STATUS.running"

"${accelerate_bin}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port "${main_port}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --datasets.vla_data.per_device_batch_size "${batch}" \
  --trainer.max_train_steps "${steps}" \
  "${trainer_args[@]}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
