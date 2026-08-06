#!/usr/bin/env bash
# Fixed-anchor Stage-B capacity/optimization diagnostic. This does not modify
# or resume the completed Stage-A/Stage-B experiments.
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"
export PATH="${STARVLA_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVS:-0}"
export WANDB_MODE=disabled

OUTPUT_DIR="${OUTPUT_DIR:-playground/Checkpoints/lewm_oft_libero_dinov3b_localinc_stageb_overfit128_fp32_seed42_2k}"
mkdir -p "${OUTPUT_DIR}"

exec python examples/LIBERO/train_files/overfit_predictable_innovation_anchors.py \
  --config "${CONFIG:-examples/LIBERO/train_files/starvla_lewm_oft_dinov3_libero_wm_only_predictable_innovation_10k.yaml}" \
  --checkpoint "${CHECKPOINT:-playground/Checkpoints/lewm_oft_libero_dinov3b_wmonly_localbasis_m4r64_fixedmean_2k/checkpoints/steps_2000_pytorch_model.pt}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --anchors "${ANCHORS:-128}" \
  --encode-batch-size "${ENCODE_BATCH:-8}" \
  --train-batch-size "${TRAIN_BATCH:-16}" \
  --eval-batch-size "${EVAL_BATCH:-32}" \
  --steps "${STEPS:-2000}" \
  --eval-interval "${EVAL_INTERVAL:-50}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --max-grad-norm "${MAX_GRAD_NORM:-10}" \
  --raw-loss-weight "${RAW_LOSS_WEIGHT:-0}" \
  --seed "${SEED:-42}"
