#!/usr/bin/env bash
# Default C recipe. Explicit CLI overrides follow the recipe configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train.sh CONFIG.yaml [--trainer.KEY VALUE ...]" >&2
  exit 2
fi
config_yaml=$1
shift
python_bin=${STARVLA_PYTHON:-}
if [[ -z "${python_bin}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    python_bin=.venv/bin/python
  elif [[ -x ../.venvs/starVLA/bin/python ]]; then
    python_bin=../.venvs/starVLA/bin/python
  else
    python_bin=python3
  fi
fi
num_processes=${NUM_PROCESSES:-8}
main_port=${MAIN_PORT:-29621}
export ACCELERATE_USE_DEEPSPEED=false
export ACCELERATE_MIXED_PRECISION=bf16
command=("${python_bin}" -m torch.distributed.run --standalone
  --nproc_per_node "${num_processes}" --master_port "${main_port}"
  --module starVLA.training.train_starvla --config_yaml "${config_yaml}" "$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  exec "${command[@]}"
fi
