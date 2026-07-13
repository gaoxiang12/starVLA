#!/usr/bin/env bash
# RoboCasa365 — download the 18 Atomic-Seen `target/human` LeRobot bundles.
# Run from the repo root.
#
# Output goes to ${DATASET_BASE_PATH} configured in
#   playground/Code/robocasa365/robocasa/macros_private.py
# i.e. ./playground/Datasets/robocasa365/v1.0/target/{atomic,composite}/...
#
set -euo pipefail

mkdir -p tmp/logs
LOG=${LOG:-tmp/logs/download_robocasa365_atomic_seen_target_human.log}

echo "[robocasa365] downloading 18 Atomic-Seen target/human LeRobot bundles -> $LOG"
python -m examples.Robocasa_365.train_files.download_target_human_direct \
  --tasks atomic-seen \
  2>&1 | tee "${LOG}"

echo "[robocasa365] done. listing downloaded Atomic-Seen tasks:"
ls playground/Datasets/robocasa365/v1.0/target/atomic    2>/dev/null || true
