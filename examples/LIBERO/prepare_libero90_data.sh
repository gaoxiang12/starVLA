#!/usr/bin/env bash
set -euo pipefail

# Prepare the public LeRobot v2 LIBERO-90 conversion and, optionally, the
# official raw HDF5 expert trajectories used to audit conversion coverage.
#
# Environment overrides:
#   LEROBOT_ROOT   Dataset root used by training.
#   RAW_ROOT       Parent directory for the official raw libero_90 directory.
#   DOWNLOAD_RAW   Set to 0 to skip the 62 GiB raw HDF5 download.
#   MAX_WORKERS    Hugging Face download workers (default: 8).
#   AUDIT_OUTPUT   Path for the raw-vs-converted JSON report.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPO_ROOT}/playground/Datasets/LEROBOT_LIBERO_DATA}"
RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/playground/LIBERO-raw-data}"
DOWNLOAD_RAW="${DOWNLOAD_RAW:-1}"
MAX_WORKERS="${MAX_WORKERS:-8}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:-${REPO_ROOT}/playground/LIBERO-audits/libero90_raw_vs_lerobot.json}"

HF_CLI="${HF_CLI:-${REPO_ROOT}/.venv/bin/hf}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
LEROBOT_DATASET="${LEROBOT_ROOT}/libero_90_no_noops_lerobot"

IPEC_REVISION="70696aef03def70c17917f43c5e79276b7e5fbe7"
RAW_REVISION="f13aa24a3da8c43c7225569f28c562979fa0e35a"

if [[ ! -x "${HF_CLI}" ]]; then
  echo "ERROR: Hugging Face CLI not found or not executable: ${HF_CLI}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

"${HF_CLI}" download \
  IPEC-COMMUNITY/libero_90_no_noops_lerobot \
  --repo-type dataset \
  --revision "${IPEC_REVISION}" \
  --local-dir "${LEROBOT_DATASET}" \
  --max-workers "${MAX_WORKERS}"

install -D -m 0644 \
  "${REPO_ROOT}/examples/LIBERO/train_files/modality.json" \
  "${LEROBOT_DATASET}/meta/modality.json"

if [[ "${DOWNLOAD_RAW}" == "0" ]]; then
  echo "LeRobot LIBERO-90 prepared at ${LEROBOT_DATASET}"
  exit 0
fi

"${HF_CLI}" download \
  yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --revision "${RAW_REVISION}" \
  --include "libero_90/*" \
  --local-dir "${RAW_ROOT}" \
  --max-workers "${MAX_WORKERS}"

"${PYTHON_BIN}" \
  "${REPO_ROOT}/examples/LIBERO/data_tools/audit_libero90_raw_vs_lerobot.py" \
  --raw-dir "${RAW_ROOT}/libero_90" \
  --lerobot-dir "${LEROBOT_DATASET}" \
  --output "${AUDIT_OUTPUT}"

echo "LeRobot LIBERO-90 prepared at ${LEROBOT_DATASET}"
echo "Official raw LIBERO-90 prepared at ${RAW_ROOT}/libero_90"
echo "Coverage audit written to ${AUDIT_OUTPUT}"
