#!/usr/bin/env bash
# Explicitly launch the standard StarVLA trainer, stage 1 then stage 2.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
LILA_PYTHON="${LILA_PYTHON:-/data/gaoxiang/Code/.venvs/starVLA/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 WANDB_MODE=disabled
export NO_ALBUMENTATIONS_UPDATE=1
export ACCELERATE_MIXED_PRECISION=bf16
unset ACCELERATE_USE_DEEPSPEED
LILA_PROCESSES="${LILA_PROCESSES:-8}"
if (( 64 % LILA_PROCESSES != 0 )); then
    echo 'Process count must divide 64 (global batch 128 / microbatch 2)' >&2
    exit 1
fi
LILA_ACCUMULATION=$((64 / LILA_PROCESSES))
"$LILA_PYTHON" - <<'PY'
import json
from pathlib import Path
path=Path('/data/gaoxiang/Checkpoints/lila_starvla_assets/robotwin_3view_aligned/preparation_audit.json')
audit=json.loads(path.read_text())
assert audit['status']=='prepared' and audit['task_count']==50, 'Run prepare_robotwin_aligned first'
assert audit['frames']==6102603 and audit['episodes']==27500, 'Dataset budget changed'
for stage in (1,2):
    output=Path(f'/data/gaoxiang/ckpts/lila_starvla/lila_robotwin_3view_aligned_stage{stage}')
    if output.exists():
        raise RuntimeError(f'{output} exists; inspect it and explicitly resume via train_starvla rather than overwrite it')
PY
for LILA_STAGE in 1 2; do
    "$LILA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$LILA_PROCESSES" \
        --module starVLA.training.train_starvla \
        --config_yaml "examples/LiLaWAM/train_files/robotwin_3view_aligned_stage${LILA_STAGE}.yaml" \
        "trainer.gradient_accumulation_steps=$LILA_ACCUMULATION"
done
