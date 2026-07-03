#!/usr/bin/env bash
set -euo pipefail

# Quick prep for BEHAVIOR_challenge data paths used by StarVLA.
#
# Default behavior:
# 1) Ensure local data roots exist under playground/Datasets.
# 2) Ensure BEHAVIOR-1K repo exists (clone if missing).
# 3) Create/update symlink playground/Datasets/BEHAVIOR_challenge -> BEHAVIOR-1K/datasets.
# 4) Copy tasks.jsonl into playground/Datasets/behavior-1k/tasks.jsonl for convenience.
#
# Optional heavy steps:
# - RUN_SETUP=1: run BEHAVIOR-1K setup.sh to install/download simulator assets.
# - DOWNLOAD_DEMOS=1: download challenge demos from Hugging Face to
#   playground/Datasets/behavior-1k/BEHAVIOR_challenge.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BEHAVIOR_REPO_DIR="${BEHAVIOR_REPO_DIR:-${REPO_ROOT}/BEHAVIOR-1K}"
DATASETS_ROOT="${DATASETS_ROOT:-${REPO_ROOT}/playground/Datasets}"
TRAIN_ROOT="${TRAIN_ROOT:-${DATASETS_ROOT}/behavior-1k}"
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-${TRAIN_ROOT}/BEHAVIOR_challenge}"
TASKS_SRC="${TASKS_SRC:-${SCRIPT_DIR}/tasks.jsonl}"
TASKS_DST="${TASKS_DST:-${TRAIN_ROOT}/tasks.jsonl}"

RUN_SETUP="${RUN_SETUP:-0}"
DOWNLOAD_DEMOS="${DOWNLOAD_DEMOS:-0}"

echo "[prep] repo root: ${REPO_ROOT}"
echo "[prep] behavior repo: ${BEHAVIOR_REPO_DIR}"
echo "[prep] datasets root: ${DATASETS_ROOT}"
echo "[prep] train root: ${TRAIN_ROOT}"

mkdir -p "${DATASETS_ROOT}" "${TRAIN_ROOT}"

if [[ ! -d "${BEHAVIOR_REPO_DIR}" ]]; then
  echo "[prep] cloning BEHAVIOR-1K..."
  git clone https://github.com/StanfordVL/BEHAVIOR-1K.git "${BEHAVIOR_REPO_DIR}"
else
  echo "[prep] BEHAVIOR-1K already exists, skip clone"
fi

if [[ -f "${TASKS_SRC}" ]]; then
  cp -f "${TASKS_SRC}" "${TASKS_DST}"
  echo "[prep] copied tasks list -> ${TASKS_DST}"
else
  echo "[prep] warning: tasks file not found at ${TASKS_SRC}"
fi

if [[ -d "${BEHAVIOR_REPO_DIR}/datasets" ]]; then
  ln -sfn "${BEHAVIOR_REPO_DIR}/datasets" "${DATASETS_ROOT}/BEHAVIOR_challenge"
  echo "[prep] symlink ready: ${DATASETS_ROOT}/BEHAVIOR_challenge -> ${BEHAVIOR_REPO_DIR}/datasets"
else
  echo "[prep] datasets dir not found yet under BEHAVIOR-1K (expected before/after setup depending on stage)"
fi

if [[ "${RUN_SETUP}" == "1" ]]; then
  echo "[prep] running BEHAVIOR-1K setup (this may take a long time)..."
  (
    cd "${BEHAVIOR_REPO_DIR}"
    ./setup.sh --omnigibson --bddl --joylo --dataset
  )
fi

if [[ "${DOWNLOAD_DEMOS}" == "1" ]]; then
  echo "[prep] downloading behavior-1k/2025-challenge-demos from Hugging Face..."
  python -m pip install -U "huggingface-hub>=0.24.0"
  hf download behavior-1k/2025-challenge-demos \
    --repo-type dataset \
    --local-dir "${TRAIN_DATASET_DIR}"
fi

echo
echo "[prep] final check"
[[ -L "${DATASETS_ROOT}/BEHAVIOR_challenge" ]] && ls -l "${DATASETS_ROOT}/BEHAVIOR_challenge" || true
[[ -d "${BEHAVIOR_REPO_DIR}/datasets" ]] && du -sh "${BEHAVIOR_REPO_DIR}/datasets" || true
[[ -d "${TRAIN_DATASET_DIR}" ]] && du -sh "${TRAIN_DATASET_DIR}" || true
[[ -f "${TASKS_DST}" ]] && echo "tasks jsonl: ${TASKS_DST}" || true

echo
echo "[prep] done"
