#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
CODE_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"
DATA_ROOT="${ROBOTWIN_DATA_ROOT:-/data/gaoxiang}"
GENERATION_GPUS="${ROBOTWIN_GENERATION_GPUS:-3,4,5,6,7}"
REMOTE_DESTINATION="${ROBOTWIN_SYNC_DESTINATION:-36.212.196.90:/data/gaoxiang/RoboTwinGenerated/}"
read -r -a EXCLUDED_TASKS <<< "${ROBOTWIN_EXCLUDED_TASKS:-open_laptop}"

cd "${REPO_ROOT}"
"${CODE_ROOT}/.venvs/starVLA/bin/python" -u \
  examples/Robotwin/generate_local_dataset.py pipeline \
  --tasks all \
  --exclude-tasks "${EXCLUDED_TASKS[@]}" \
  --splits clean \
  --clean-target 500 \
  --data-root "${DATA_ROOT}" \
  --gpus "${GENERATION_GPUS}" \
  --poll-seconds 60 \
  --deep

# AGENTS.md requires the completed local dataset to be synchronized to the
# shared remote host. Keep raw planner/HDF5 intermediates local; synchronize
# only the fully converted and deeply audited LeRobot datasets.
rsync -a --partial --info=stats2 \
  -e "ssh -p 1227" \
  "${DATA_ROOT}/RoboTwinGenerated/" \
  "${REMOTE_DESTINATION}"
