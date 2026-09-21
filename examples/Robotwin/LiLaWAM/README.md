# LiLa-WAM official-checkpoint reproduction

This inference integration loads the official LiLa-WAM architecture from a
separate checkout and serves it with StarVLA's WebSocket server. It does not
claim to implement StarVLA/LeRobot training for this model.

The agreed benchmark is **all 50 RoboTwin 2.0 tasks, demo_clean, 100 valid
episodes per task**, using the official expert feasibility filter and original
task success predicates. The acceptance threshold is an observed macro average
of at least 90%; missing tasks or incomplete episodes cannot pass. A pooled
Wilson interval is reported separately, and is not a guarantee for each task.
The upstream 90.48% is a published result, not a local measurement.

Sources:

- https://github.com/teee000/LiLa-WAM
- https://www.modelscope.cn/models/yangfan97/LiLa-WAM_RoboTwin2_0
- https://www.modelscope.cn/models/facebook/dinov3-vitl16-pretrain-lvd1689m

## Setup

Edit `examples/Robotwin/eval_files/lila_wam_official.yaml` for your local paths. Reuse shared
weights under `/data/gaoxiang/ckpts`; the downloader verifies publisher SHA-256
hashes and saves source manifests. The source checkout must supply the official
`models/`, `data-500-taskcond/`, and `utils/stat-500-all.json`.

From the StarVLA root:

```bash
../.venvs/starVLA/bin/python examples/Robotwin/eval_files/fetch_lila_assets.py
../.venvs/starVLA/bin/python -m unittest tests.test_lila_wam -v
CUDA_VISIBLE_DEVICES=1 ../.venvs/starVLA/bin/python -m examples.Robotwin.eval_files.check_lila_parity \
  --config examples/Robotwin/eval_files/lila_wam_official.yaml \
  --output examples/Robotwin/audits/lila_wam_parity.json
```

The parity check loads both the official inference implementation and the
StarVLA adapter with the real checkpoint, then requires bitwise equal actions
for identical observations and RNG seeds. This is an integration check, not a
rollout success result.

## Evaluation

Use free GPU IDs after checking current occupancy. Each GPU slot starts a
separate policy server and runs one simulation process. Each task gets a fresh
server. Each inference transfers the simulator CUDA RNG state to the server
and restores the post-inference state back to the simulator. This preserves
upstream's shared RNG stream, including per-scene reseeding and any random
numbers consumed during simulator setup. Output directories must be new.

```bash
# Smoke test, stored separately from the benchmark.
../.venvs/starVLA/bin/python -m examples.Robotwin.eval_files.run_lila_benchmark \
  --output /data/gaoxiang/ckpts/lila_wam_smoke \
  --tasks adjust_bottle blocks_ranking_rgb --episodes 2 --seed 7 --gpus 1 2

# Full agreed protocol: 50 x 100 valid rollouts, seed 0 -> scene seeds >=100000.
../.venvs/starVLA/bin/python -m examples.Robotwin.eval_files.run_lila_benchmark \
  --output /data/gaoxiang/ckpts/lila_wam_clean50_n100 \
  --tasks all --episodes 100 --gpus 1 2 4
```

`protocol.json` records the fixed task list, budget, seeds and source/weight
hashes. Each task directory contains `server.log`, `eval.log`, and `status.json`
with actual episode seeds and outcomes parsed from the official evaluator's
counters. `summary.json` is written when the campaign exits. Infrastructure
errors remain failures of evaluation completeness, not silently discarded model
episodes. The script never selects the best of multiple trials or reports a
partial task set as passing the full protocol.

The adapter preserves upstream head-camera processing (320x240, OpenCV linear
resize, ImageNet normalization, no channel swap), 16-D endpose state, task VTT,
10 flow steps, a 32-action chunk, 16-action execution horizon, official min/max
statistics and B-spline smoothing. Actions stay in RoboTwin's original order:
left joints, left gripper, right joints, right gripper. The existing GAWM joint
state preprocessing and action permutation are not applicable to this model.

## Current reproduction

The 2026-09-11 reproduction uses GPUs 1, 2 and 4 and stores its durable status
and results at `/data/gaoxiang/ckpts/LiLa-WAM_eval_20260911_r3`. Its supervisor
performs real-observation inference parity, a separate six-rollout smoke test
(three tasks, seed 7), and then the full 5,000-rollout protocol (seed 0).
Inspect `status.json` and the `parity.log`, `smoke.log`, and `full.log` files
there. Only `full/summary.json` can establish the agreed 90% result.
Read live progress from the StarVLA root:

```bash
../.venvs/starVLA/bin/python -m examples.Robotwin.eval_files.lila_progress \
  /data/gaoxiang/ckpts/LiLa-WAM_eval_20260911_r3
```

The earlier directories without `_r3` retain intentionally stopped engineering
smoke runs; they are not benchmark data. The finalized run allows 24 hours per
task, preserving the full 100-rollout budget for slow tasks on A100 rendering.

The initial local source revisions are LiLa-WAM
`65a320397faae07f311680fcb32f6c2f27dca1a0` and RoboTwin
`c3ddfa8b97d5519efa828b075999bd0006778e5e`. RoboTwin already has local
collection/language changes and an evaluator option for configurable test_num;
the task success predicates are unchanged. The actual evaluated file contents
are recorded by SHA-256, including the evaluator and every task implementation.
