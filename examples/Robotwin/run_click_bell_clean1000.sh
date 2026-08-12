#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOTWIN_ROOT="${REPO_ROOT}/thirdparty/RoboTwin"
RUN_DIR="${REPO_ROOT}/playground/Datasets/RoboTwinGenerated_raw/click_bell/starvla_click_bell_clean1000"
PYTHON_BIN="/home/zskj/data/miniconda3/envs/robotwin/bin/python"
GPU_ID="${GPU_ID:-5}"

mkdir -p "${RUN_DIR}"
if [[ -s "${RUN_DIR}/collector.pid" ]] && kill -0 "$(<"${RUN_DIR}/collector.pid")" 2>/dev/null; then
  echo "collector already running with PID $(<"${RUN_DIR}/collector.pid")"
  exit 1
fi

date +%s > "${RUN_DIR}/job_started_at.txt"
(
  cd "${ROBOTWIN_ROOT}"
  nohup setsid env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" script/collect_data.py click_bell starvla_click_bell_clean1000 \
    > "${RUN_DIR}/collect.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/collector.pid"
)

nohup setsid "${PYTHON_BIN}" "${REPO_ROOT}/examples/Robotwin/data_generation_progress.py" \
  --run-dir "${RUN_DIR}" --target 1000 --start-seed 20000 \
  > "${RUN_DIR}/progress_monitor.log" 2>&1 < /dev/null &
echo $! > "${RUN_DIR}/progress_monitor.pid"

echo "collector PID: $(<"${RUN_DIR}/collector.pid")"
echo "monitor PID: $(<"${RUN_DIR}/progress_monitor.pid")"
echo "progress: ${RUN_DIR}/progress.txt"
