#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CONFIG_YAML=${CONFIG_YAML:-${script_dir}/starvla_gawm_robotwin_continuous_next.yaml}
export RUN_ID=${RUN_ID:-gawm_s_robotwin_continuous_next_49tasks_40k_20260905}
export MAIN_PORT=${MAIN_PORT:-29669}
export WANDB_MODE=${WANDB_MODE:-disabled}
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONNOUSERSITE=1
exec bash "${script_dir}/run_gawm_robotwin_clean1000_posttrain.sh"
