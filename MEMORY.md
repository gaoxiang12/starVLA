# starVLA Workspace Memory

This file records stable, non-secret facts needed to continue work in this
repository. Re-check volatile state such as GPU occupancy and running processes
instead of trusting an old snapshot.

## Workspace and environments

- Repository root: `/home/zskj/data/gaoxiang/starVLA`.
- Primary environment for starVLA training and tests: `.venv`.
  - Python: 3.10 (`.venv/bin/python`, linked to `/usr/bin/python3`).
  - PyTorch: `2.6.0+cu124`.
  - Accelerate: `1.5.2`; invoke it explicitly as
    `.venv/bin/accelerate` when `.venv` is not activated.
- Do not infer that CUDA is unavailable merely because an ordinary sandboxed
  command cannot initialize NVML. Host-level `nvidia-smi` shows 8 NVIDIA
  A100-PCIE-40GB GPUs; always re-check current availability before launching.
- `.venv-libero` currently has CPU PyTorch `2.6.0+cpu` and does not import the
  `libero` package. It is not the environment used by the current starVLA
  train/eval scripts.
- Relevant unit test command:

  ```bash
  .venv/bin/python -m unittest tests.test_visual_token_world_model
  ```

  On 2026-07-22 this ran 11 tests successfully.

## DINOv3 LIBERO baseline

- Current development branch: `dev.ai`.
- Validated baseline checkpoint:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_spatial4x4_trainenc1e6_statecond_ema09_200k_fullstate/checkpoints/steps_200000_pytorch_model.pt
  ```

- DINOv3 ViT-B/16 weight:

  ```text
  dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
  ```

- LIBERO dataset root:

  ```text
  playground/Datasets/LEROBOT_LIBERO_DATA
  ```

- Recorded success rates for the 200k baseline are spatial `0.94`, object
  `0.99`, goal `0.96`, and libero10 `0.71` (equal-suite average `0.90`).
- The baseline's deployed path is DINOv3 -> 4x4 spatial tokens per view ->
  deterministic future-token residual predictor -> visual-action cross
  attention -> OFT action head.

## Current WALA-style transition work

- The WALA-style mechanism is being added on top of the validated DINOv3
  baseline, not the compact/interleaved experimental model.
- The default workflow is one `combined` run directly from the validated 200k
  baseline. Teacher reconstruction, student alignment/decoding, and deployed
  action losses are optimized concurrently.
- The legacy `teacher -> student -> joint` modes remain available for ablations,
  but are no longer required by the default launcher.
- Auxiliary modules are training-only. `predict_action` does not call them.
- Keep the DINO encoder, spatial token pooler, original latent world model, task
  embedding, and `delta_scale` EMA frozen during auxiliary stages by default.
- Implementation guide:

  ```text
  examples/LIBERO/train_files/WALA_TRANSITION_ON_DINOV3_BASELINE.txt
  ```

- Single-run launch command (choose GPUs after a fresh availability check):

  ```bash
  CUDA_DEVS=<free-gpu-ids> \
  ACCELERATE_BIN=.venv/bin/accelerate \
  WANDB_MODE=disabled \
  bash examples/LIBERO/train_files/run_lewm_oft_dinov3_wala_transition.sh
  ```

- The teacher run completed on 2026-07-22. Its accepted checkpoint is:

  ```text
  playground/Checkpoints/lewm_oft_dinov3b_wala_transition_teacher_from200k/checkpoints/steps_20000_pytorch_model.pt
  ```

- Teacher checkpoint analysis is stored at:

  ```text
  playground/Checkpoints/lewm_oft_dinov3b_wala_transition_teacher_from200k/teacher_checkpoint_analysis.json
  ```

  On 64 fixed samples, 20k reconstruction/L1/cosine losses were
  `0.28548 / 0.26715 / 0.18331`. Shuffling transition tokens raised
  reconstruction to `0.59418`, and zeroing them raised it to `0.57065`, so the
  decoder uses the transition bottleneck. All 322 compatible non-auxiliary
  tensors remained bitwise equal to the validated baseline.
- The accepted student checkpoint is:

  ```text
  playground/Checkpoints/lewm_oft_dinov3b_wala_transition_student/checkpoints/steps_20000_pytorch_model.pt
  ```

  On 64 fixed samples its transition alignment/decode losses were
  `0.20959 / 0.44679`. The teacher and baseline tensors remained frozen while
  all 41 student tensors changed.
- The staged joint run completed on 2026-07-23. Its evaluated checkpoint is:

  ```text
  playground/Checkpoints/lewm_oft_dinov3b_wala_transition_joint/checkpoints/steps_20000_pytorch_model.pt
  ```

  Standard LIBERO evaluation used seed 7, 10 tasks per suite, 10 trials per
  task, and the model's 8-step execution horizon. Success rates were spatial
  `0.95`, object `1.00`, goal `0.92`, and libero10 `0.78`, for `365/400 =
  0.9125` overall. The validated baseline rates are
  `0.94 / 0.99 / 0.96 / 0.71` (`0.90` overall), so the changes were
  `+0.01 / +0.01 / -0.04 / +0.07` and `+0.0125` overall. A paired exact
  McNemar comparison gave `p=0.52`, so the single-seed gain is not yet
  statistically conclusive.
- The new one-run mode is `transition_mode=combined`. It starts from the
  validated 200k baseline and defaults to run id:

  ```text
  lewm_oft_dinov3b_wala_transition_combined_from200k
  ```

  In combined mode, teacher encoder/decoder, student resampler, visual action
  head, and action model train together. Teacher tokens are detached before
  student alignment. DINO, spatial pooler, original latent world model, task
  embedding, and `delta_scale` remain frozen. The default launcher runs 20k
  steps once; it does not require intermediate checkpoints.
- The configured W&B entity is the placeholder `your_name`, so W&B init fails
  harmlessly. New runs persist metrics every 100 steps to `metrics.jsonl`; use
  that file for stage analysis and launch with `WANDB_MODE=disabled` unless a
  real W&B entity is configured.

## Worktree safety

- Preserve unrelated user-owned untracked files and directories, especially
  `thirdparty/`, `BEHAVIOR-1K/`, DINO weights, and compact/interleaved experiment
  files.
- Tracked changes from the earlier `dino-compact-latent-wm` work were saved as:

  ```text
  stash@{0}: pre-wala-dev-ai-switch-20260722
  ```

  Stash indices can change; verify with `git stash list` before using it.
- Do not restore, delete, or overwrite that stash or unrelated untracked work
  unless explicitly requested.
