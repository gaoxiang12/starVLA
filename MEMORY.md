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

## RoboTwin LeWM-OFT baseline

- The converted RoboTwin dataset root is
  `playground/Datasets/RoboTwin`. It contains all 50 tasks: Clean has 2,500
  episodes / 552,287 frames and Randomized has 25,000 episodes / 5,550,316
  frames.
- The existing DINOv2-B LeWM-OFT run completed normally at 200k steps:

  ```text
  playground/Checkpoints/lewm_oft_robotwin_dinov2b_spatial4x4_200k
  playground/Checkpoints/lewm_oft_robotwin_dinov2b_spatial4x4_200k/checkpoints/steps_200000_pytorch_model.pt
  ```

  Its saved run config is a legacy variant with action hidden dimension 1536
  and residual SIGReg weight 0.02. The current YAML requests action hidden
  dimension 384 and SIGReg 0.0, but `LeWM_OFT` currently normalizes the runtime
  action hidden dimension to the DINOv2 `wm_hidden` value 1536. A 2026-08-03
  construction smoke test confirmed the current runtime is 1536, so the
  effective recipe difference here is SIGReg and later source changes, not the
  action-head width.
- The official RoboTwin checkout is `thirdparty/RoboTwin` at commit
  `13c3c47ff4312dd62484bcd51be034af55c062d1`. Its Python 3.10 environment is
  `/home/zskj/data/miniconda3/envs/robotwin` with PyTorch 2.4.1+cu124,
  torchvision 0.19.1+cu124, SAPIEN 3.0.0b1, mplib 0.2.1, and compiled CuRobo
  0.7.8. Always set `PYTHONNOUSERSITE=1`; PyTorch3D is optional and is not
  installed.
- Evaluation consumes RGB views in `[head, left wrist, right wrist]` order,
  includes the reordered 14-D joint state, predicts 16-step absolute-qpos
  chunks, and converts actions back to RoboTwin's
  `[left joints, left gripper, right joints, right gripper]` order. Progress
  conditioning is disabled for this checkpoint.
- On 2026-07-31, the 200k checkpoint passed the policy-server metadata and
  synthetic-action checks, SAPIEN GPU rendering, and full one-rollout
  `click_bell` smoke tests at seed 91 in both `demo_clean` and
  `demo_randomized` (`1/1` success in each). These are interface diagnostics,
  not a final benchmark. The final protocol remains 50 tasks x 2 modes x 100
  valid rollouts, seed 0, as documented in
  `examples/Robotwin/eval_files/LEWM_OFT_EVALUATION.txt`.
- The formal legacy-200k baseline evaluation was launched detached on
  2026-07-31 with setting `legacy200k_baseline_seed0_n100_formal`, seed 0,
  100 valid rollouts, all 50 tasks, and both Clean/Randomized modes. Its run
  directory is:

  ```text
  playground/Checkpoints/lewm_oft_robotwin_dinov2b_spatial4x4_200k/robotwin_baseline_legacy200k_seed0_n100_formal
  ```

  The first attempt was interrupted by a full host reboot at 16:24 before any
  task reached a complete 100-rollout `_result.txt`; its 2-6 rollout partial
  counters must not be used as baseline results. `resume_baseline.sh` skips
  complete task/mode results and restarts only incomplete ones, then runs the
  strict summarizer and writes `report/` plus `STATUS.complete`.
  `wait_and_resume.sh` is detached and records its PID in
  `wait_and_resume.pid`; it automatically invokes the resume script when
  `nvidia-smi` works and at least two suitable GPUs are available.
- After the reboot, all eight A100s were visible on PCIe but kernel
  `5.15.0-185-generic` had no NVIDIA module or `/dev/nvidia*` nodes.
  `/usr/src/nvidia-550.54.14` and matching kernel headers exist, but restoring
  the driver requires administrator action. If recovery itself reboots the
  host, relaunch `wait_and_resume.sh`; GPU/process state is volatile and must
  be re-checked.
- The driver was restored with a second administrator reboot at 17:20/17:28.
  Formal resume attempt `20260731_174953` was then launched detached on GPUs
  0-3. It was stopped at the user's request on 2026-08-03 11:17 after 28/50
  Clean tasks and 0/50 Randomized tasks had complete 100-rollout results.
  `STATUS.stopped` and `resume_after_driver.log` record the state; partial
  tasks are not counted. All processes in evaluation session 23902, including
  the four older persistent policy servers, were terminated.
- RoboTwin evaluation now defaults `ROBOTWIN_EVAL_VIDEO_LOG=0`; this skips
  per-rollout ffmpeg/MP4 generation without changing success semantics. Set it
  to `1` only for targeted diagnostics. The official expert pass is still
  required because it filters valid scenes and produces instruction metadata.
  Repeated checkpoint evaluation can remove its recurring cost only after a
  frozen expert seed/instruction manifest is implemented and validated.
- `run_policy_server.sh` now `exec`s the Python server so task cleanup targets
  the actual server instead of leaving its child bound to the slot port.
- The current LeWM implementation accepts only raw DINOv3 `.pth` encoders, so
  the Clean-only RoboTwin run is DINOv3-B at
  `playground/Checkpoints/lewm_oft_robotwin_dinov3b_clean50_spatial4x4_current_200k`.
  It uses the new `robotwin_clean_wm` mixture (all 50 Clean task datasets,
  2,500 episodes), no LIBERO or RoboTwin policy checkpoint, 3 views, 14-D
  state/action, a 16-step horizon, SIGReg 0, and future video indices
  `[0, 8, 16]`. The raw DINOv3-B vision encoder is pretrained; the policy and
  LeWM/OFT heads start from scratch. Dataset validation confirmed 3 current
  views, 2 future frames with 3 views each, `action [16,14]`, and state
  `[1,14]`.
- This run was launched detached on GPU 4 on 2026-08-03 with supervisor PID
  1239530, one process, micro-batch 8, gradient accumulation 4, and global
  batch 32. It completed normally at 200k steps on 2026-08-05 16:07. The final
  action L1 / latent loss were 0.00631 / 0.47227, and the final checkpoint is
  `checkpoints/steps_200000_pytorch_model.pt` under the run directory.
- The final checkpoint passed a one-rollout Clean `click_bell` smoke test
  (`1/1`, seed 91) with video disabled. A reduced Clean diagnostic was then
  launched detached on 2026-08-05 with setting
  `robotwin_clean200k_dinov3b_seed0_n10`: all 50 tasks, 10 valid rollouts per
  task, seed 0, no video, official expert pass retained, GPUs 0 and 4. Its run
  directory is
  `robotwin_clean200k_seed0_n10_all50` under the training run, and supervisor
  PID was `840342`. This is a 500-rollout diagnostic, not the official
  50 x 100 final benchmark; re-check the volatile PID/log/process state.
- A from-scratch Clean-50 rerun with canonical per-task language was launched
  detached on GPU 0 on 2026-08-06.  Its run directory is
  `playground/Checkpoints/lewm_oft_robotwin_dinov3b_clean50_canonical_tasktext_fromscratch_200k`,
  supervisor PID file is `train.pid`, and the live log is `train.log`.  It uses
  `task_language_mode=dataset_name`, so all episodes of e.g. `click_bell` use
  `click bell` at both training and evaluation.  It starts from the raw DINOv3-B
  encoder (no RoboTwin policy checkpoint), trains the encoder, uses micro-batch
  4 with gradient accumulation 8 (global batch 32), and targets 200k steps.

- The canonical Clean-50 run completed normally at 200k on 2026-08-10. Its
  reduced seed-0 Clean evaluation scored `click_bell` at `9/10`; this small
  sample is only a preliminary baseline. On 2026-08-12 a paired 100-rollout
  Clean baseline was launched on GPU 5 under
  `click_bell_baseline_seed0_n100` in the run directory.
- A locally collected `click_bell` Clean dataset is preserved in raw HDF5 at
  `playground/Datasets/RoboTwinGenerated_raw/click_bell/starvla_click_bell_clean1000`
  (1,000 episodes, seeds 20000--20999). Its LeRobot conversion is
  `playground/Datasets/RoboTwinClickBellClean1000/click_bell`: 1,000 episodes,
  78,055 frames, three views, 14-D state/action, and one effective canonical
  training string (`click bell`) via `task_language_mode=dataset_name`.
- Targeted fine-tuning from the canonical 200k checkpoint was launched
  detached on GPU 0 on 2026-08-12 at
  `playground/Checkpoints/lewm_oft_robotwin_dinov3b_clean50_canonical_click_bell_clean1000_ft20k`.
  It uses only the new 1,000-episode mixture, trains all modules for 20k steps,
  uses micro-batch 4 / accumulation 8 (global batch 32), warmup 500, base and
  action LR `1e-5`, encoder LR `1e-7`, and saves every 5k. The pretrained
  weights loaded successfully and initial optimization was stable. A detached
  watcher evaluates 5k/10k/15k/20k on the same Clean seed-0 100-rollout
  protocol after each checkpoint appears; re-check live processes and logs.


## Long-running training process policy

- Launch long training jobs detached from the interactive terminal with
  `nohup` and `setsid`, redirect stdin from `/dev/null`, and redirect stdout and
  stderr to a persistent `train.log`.
- Save the background process PID in the run directory and verify the PID and
  log after launch. A representative pattern is:

  ```bash
  nohup setsid env <training-environment> bash <launcher> \
    > <run-dir>/train.log 2>&1 < /dev/null &
  echo $! > <run-dir>/train.pid
  ```

- Do not rely on a Codex persistent PTY for future long-running training.
- When reporting or monitoring training progress for the user, point them to
  the live training-curves dashboard first. Do not default to asking them to
  inspect `train.log` or `metrics.jsonl` manually; those files remain backend
  diagnostics for the agent when investigating failures or missing metrics.
## Dataset task-language preflight

- Before launching training or evaluation on any new or changed dataset,
  explicitly audit and report: semantic task-class count, episode count,
  unique language-string count, episodes per string, whether paraphrases or
  instance-specific goal attributes cause the variation, train/eval language
  agreement, and hash-bucket collisions for hash-conditioned models.  Warn the
  user before launch whenever the task-language identity is not LIBERO-like;
  never assume `tasks.jsonl` rows are the benchmark's semantic task classes.
- Current local audit: BEHAVIOR is stable (50 task strings, exactly 200 episodes
  per string).  LIBERO uses one stable canonical string per task, although the
  no-noop conversion has fewer than 50 episodes for some tasks.  Original
  RoboTwin has episode-level paraphrases and is now canonicalized by task name
  for the new Clean-50 rerun.  RoboCasa365 is materially different: some broad
  task folders contain real instance-specific goals (for example 105 used
  object-specific strings across 502 `PickPlaceCounterToCabinet` episodes, and
  13 destination strings across 500 `NavigateKitchen` episodes).  Do not
  collapse those RoboCasa strings to the broad folder name without designing a
  semantic/structured goal representation.  Datasets not installed locally
  remain unaudited and must be checked before use.

## Loss defaults

- Keep SIGReg disabled by default in all visual-token world-model training:
  `residual_predictor_sigreg_weight: 0.0` (and legacy
  `delta_head_sigreg_weight: 0.0`). Enable it only for an explicitly named
  SIGReg ablation.
- Keep direct supervised latent/residual prediction enabled by default with
  weight `1.0`, unless an experiment explicitly studies its removal.

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

## LIBERO smooth-global joint run

- The current smooth-global 384-D latent/action joint run is training at:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_smooth_global384_action_joint_200k
  ```

  Its 160k checkpoint was saved on 2026-08-07 and is
  `checkpoints/steps_160000_pytorch_model.pt`. The training mixture has 1,803
  episodes across the four standard LIBERO suites, 40 semantic tasks, and 40
  unique canonical language strings. Each string has 33--50 episodes after
  combining the base and augmented LIBERO-10 data. All 40 evaluation strings
  exactly match training strings, and the checkpoint's 4,096-bucket MD5 task
  conditioning has no collisions among them.
- A detached closed-loop evaluation of the 160k checkpoint was launched on
  2026-08-07 with seed 7, 10 tasks per suite, 10 trials per task, the model's
  8-step execution horizon, progress conditioning disabled, and GPUs 1/2/3/5.
  The launcher, supervisor PID, live status, and eventual summary are
  `eval_steps_160000.sh`, `eval_steps_160000.pid`,
  `eval_steps_160000.status`, and `libero_success_rates_steps_160000.txt` in
  the run directory. Re-check the volatile process/status before relying on
  it. The user `zskj` currently lacks the host `video`/`render` groups, so EGL
  cannot open `/dev/dri`; this evaluation uses one Xvfb + Mesa software-GL
  display per suite while retaining GPU policy inference. The GLX smoke test
  and the first completed spatial/object/goal rollouts passed.

## LIBERO latent-below-0.5 experiments

- The first isolated world-model-only experiment was launched detached on
  GPU 0 on 2026-08-03 at:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_wmonly_ctx2_state8_cos01_from220k_50k
  ```

  It warm-starts from the augmented-L10 220k checkpoint (whose float32
  `delta_scale` is 1.6803), freezes DINOv3, the spatial-token pooler, and all
  action modules, and trains only the residual world model plus task embedding.
  The opt-in predictor uses a zero-initialized previous-to-current visual-token
  motion adapter, zero-initialized current 8-D proprio conditioning, and a 0.1
  cosine auxiliary loss. The original normalized latent MSE remains separately
  logged, including per-horizon `latent_loss_horizon_1/2`, so crossing 0.5 is
  directly comparable rather than a loss-rescaling artifact.
- The run uses the matching augmented LIBERO mixture with frame indices
  `[-1, 0, 4, 8]`, one process on GPU 0, micro-batch 4, accumulation 8, global
  batch 32, and saves every 5k steps. It was stopped on 2026-08-03 at logged
  step 7,650 after showing no improvement: mean raw normalized MSE was 0.50543
  for steps 0-500, 0.51192 for steps 3k-5k, 0.51084 for steps 5k-7k, and
  0.51030 over the final 100 logged records. Its 5k checkpoint, complete log,
  metrics, and `STATUS.stopped` are preserved.
- The follow-up experiment runs at:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_wmonly_ctx3_statehist3_resboost_from220k_50k
  ```

  It still uses no action conditioning and keeps the original raw normalized
  latent MSE as the sole optimized objective. It freezes the mature predictor
  and task embedding, then trains only an 11.263M-parameter zero-output
  residual correction transformer. The correction consumes three causal
  visual frames and normalized proprio states at `[-8, -4, 0]`, including
  finite-difference state motion, plus the frozen predictor's +4/+8 output.
  This preserves the warm-start prediction exactly at step 0 while giving a
  larger model explicit motion history. Config and launcher are
  `starvla_lewm_oft_dinov3_libero_wm_only_ctx3_statehist_resboost.yaml` and
  `run_lewm_oft_dinov3_libero_wm_only_ctx3_statehist_resboost.sh`.
- The follow-up completed normally at 50k. Over steps 45k-50k, corrected raw
  normalized MSE averaged 0.504204 versus frozen-base 0.506632, an absolute
  improvement of only 0.002427. This confirms that longer causal history can
  help, but the gain is too small to justify another full-resolution residual
  booster. Its final checkpoint, log, metrics, and `STATUS.complete` are kept.

## Removed predictable-innovation branch

- The opt-in predictable-innovation bottleneck and its calibration, staged
  training, gating, overfit, and held-out diagnostic code were removed from
  LeWM-OFT on 2026-08-07. The policy server again derives visual history only
  from `world_model.ctx_len`.
- The archived experiments gave a negative result worth retaining: the learned
  basis captured residual energy, but the action/state-free predictor produced
  effectively no held-out raw latent gain. Historical checkpoints and reports
  remain under `playground/Checkpoints`, but current source no longer supports
  loading or training that architecture.

## LIBERO-10 augmented-data run

- The official LIBERO-10 raw-data audit found 500 source demonstrations versus
  379 in the existing LeRobot conversion. Strict simulator replay recovered 21
  missing demonstrations, and the Qwen3-VL-OFT teacher recovered 89 more from
  the remaining official training initial states. The resulting LIBERO-10
  coverage is 489 episodes / 130,971 frames; fixed benchmark evaluation states
  were not used for collection.
- The registered training mixture is `libero_all_wm_l10_augmented`. Auxiliary
  dataset weights are proportional to their episode counts (21/379, 15/379,
  and 74/379), so the augmentation does not manually oversample individual
  LIBERO-10 trajectories.
- A from-scratch 220k-step DINOv3 LEWM-OFT run was launched detached on
  2026-07-29 using shared GPUs 4-7:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_l10aug489_spatial4x4_trainenc1e6_statecond_220k
  ```

  It uses four processes, per-GPU batch 8, gradient accumulation 1, global
  batch 32, 2k warmup steps, and saves every 10k steps. `train.pid` identifies
  the detached supervisor and `train.log` / `metrics.jsonl` contain progress.
  At step 100, action/L1/latent losses were
  `1.10901 / 0.20724 / 0.89855`; training was stable without OOM despite GPU
  sharing.
- The run completed normally at 220k steps on 2026-07-30. Closed-loop
  evaluation of `steps_220000_pytorch_model.pt` used seed 7, 10 tasks per
  suite, 10 trials per task, the model's 8-step execution horizon, and disabled
  progress conditioning. Success rates were spatial `0.96`, object `0.99`,
  goal `0.95`, and LIBERO-10 `0.90`, for `380/400 = 0.95` overall. Relative to
  the validated 200k baseline (`0.94 / 0.99 / 0.96 / 0.71`, `360/400 = 0.90`),
  the changes were `+0.02 / 0.00 / -0.01 / +0.19` and `+0.05` overall.
  Results are recorded in
  `libero_success_rates_steps_220000.txt` in the run directory.

## LIBERO-90 combined-suite run

- The public LIBERO-90 LeRobot v2 conversion is available at
  `playground/Datasets/LEROBOT_LIBERO_DATA/libero_90_no_noops_lerobot` with
  3,921 episodes / 569,249 frames. The registered five-suite mixture is
  `libero_all_wm_l10_augmented_l90`: spatial, object, goal, the augmented
  LIBERO-10 data, and LIBERO-90.
- A from-scratch 360k-step DINOv3 LeWM-OFT run was launched detached on
  2026-07-29 using shared GPUs 5-7:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_l10aug489_l90_spatial4x4_trainenc1e6_statecond_360k
  ```

  It uses three processes, per-GPU batch 8, gradient accumulation 1, global
  batch 24, 3k warmup steps, and saves every 10k steps. `train.pid` identifies
  the detached supervisor. At step 100, action/L1/latent losses were
  `1.25327 / 0.27086 / 0.97911`; all were finite, no launch errors were found,
  and each GPU still had about 21.8 GiB free while sharing with the augmented
  LIBERO-10 run.
- The run completed normally at 360k steps on 2026-07-31. Closed-loop
  evaluation of `steps_360000_pytorch_model.pt` used seed 7, 10 tasks per
  suite, 10 trials per task, the model's default 8-step execution horizon, and
  disabled progress conditioning. Success rates were spatial `0.97`, object
  `0.99`, goal `0.93`, and LIBERO-10 `0.88`, for `377/400 = 0.9425` overall.
  Relative to the augmented-L10-only 220k run (`0.96 / 0.99 / 0.95 / 0.90`,
  `380/400 = 0.95`), the changes were `+0.01 / 0.00 / -0.02 / -0.02` and
  `-0.0075` overall. Results are recorded in
  `libero_success_rates_steps_360000.txt` in the run directory.

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

- For any from-scratch `joint` or `combined` run with
  `transition_joint_freeze_base=false`, keep the `delta_scale` EMA enabled so
  it follows the trainable encoder/pooler coordinate system. Freeze the EMA only
  in `teacher`/`student` stages, frozen-base `joint`/`combined` stages, or a
  frozen dense-base ablation.
- The formal from-scratch WALA combined 200k run was restarted on 2026-07-27
  after fixing that condition. Run directory and detached supervisor PID file:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_spatial4x4_wala_combined_sigreg0_trainenc1e6_statecond_200k
  playground/Checkpoints/lewm_oft_libero_dinov3b_spatial4x4_wala_combined_sigreg0_trainenc1e6_statecond_200k/train.pid
  ```

  The invalid frozen-scale run was preserved as
  `..._invalid_frozen_delta_scale_step126xx_20260727`. At step 100 of the
  corrected run, `delta_scale=0.33268` and target RMS `=0.30837`, confirming
  that the EMA is active.


## Removed dense current-patch action residual

- The opt-in action-query residual over unpooled current-frame patches was
  removed from LeWM-OFT on 2026-08-07, together with its launcher, metrics, and
  optimizer/config plumbing. Historical checkpoints remain under
  `playground/Checkpoints`, but an enabled legacy branch is no longer loadable.
- This does not remove the dense-only 14x14 model below; that model uses dense
  tokens as its primary world-model representation rather than as an action
  residual alongside the compact path.

## Dense-only 14x14 strict baseline control

- The primary high-resolution experiment is a strict spatial-resolution control,
  not a continuation from the validated 200k checkpoint. It starts from the
  same raw pretrained DINOv3 encoder as the 4x4 baseline; every downstream
  module starts randomly initialized.
- The only intended model variable is spatial resolution: each of two views
  keeps all `14 x 14 = 196` DINO patches (392 tokens/frame). The world model
  predicts two future 392-token frames, and the action head reads the current
  plus predicted future dense tokens. It has no parallel 4x4 path or separate
  dense action adapter.
- Match the baseline training protocol: all 131.131M parameters train for 200k
  optimizer steps, per-GPU batch 8 on four GPUs (global batch 32), base/action
  LR `1e-4`, encoder LR `1e-6`, latent loss weight 1, SIGReg weight 0, seed 42.
- Launcher:

  ```text
  examples/LIBERO/train_files/run_lewm_oft_dinov3_dense14x14_200k_train_eval.sh
  ```

- The launcher automatically evaluates the final 200k checkpoint on
  `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`, using seed
  7, 10 trials per task, and the model execution horizon, then writes the same
  success-rate summary format as the baseline.
- Formal run directory:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_dense14x14_latent1_sigreg0_trainenc1e6_statecond_200k
  ```

- `train.log` contains the detached train/eval console stream and `train.pid`
  contains the `nohup + setsid` supervisor PID. The job was queued on
  2026-07-23 to start on GPUs 4-7 after the dense-residual ablation exits.
  Re-check the PID, log, metrics, and GPU state rather than trusting this
  volatile snapshot.

## Removed joint reconstructive compact-latent branch

- The opt-in `world_model.reconstructive_latent_enabled` branch and its
  dedicated module, tests, configs, launchers, and data registrations were
  removed from LeWM-OFT on 2026-08-07. It was world-model-only and never
  connected to `predict_action`; removing it does not change the default
  model's parameters or inference path.
- Historical checkpoint artifacts may still contain the old configuration and
  state-dict keys, but current source no longer supports loading or training
  that architecture.

## Smooth spatial latent world model

- The new opt-in LIBERO representation experiment is
  `world_model.smooth_latent_enabled=true`. Frozen DINOv3 patch features are
  pooled to a fixed `4 x 4` grid per view. A shared first projection compresses
  each of the 32 two-view spatial cells from 768 to 128 dimensions; the fixed
  camera/grid order is then flattened and a second projection produces one
  384-D frame latent. Spatial averaging is not used. There is no reconstruction,
  robot-state/action input, or action-head training.
- The data schema is `[t,t+1,t+2,t+8]`. The world model predicts `t+1` and
  `t+2` directly from `z_t`; the primary loss is unnormalized future-latent
  MSE, i.e. squared L2 on `pred(z_t) - stopgrad(z_{t+h})`. SIGReg is applied to
  actual content latents only. First-difference slowness, second-difference
  acceleration, and a same-episode `t+8` temporal-order hinge encourage smooth
  but non-static dynamics.
- The dataset optionally emits `future_frame_valid_mask`, computed before
  clamping video offsets. This excludes repeated end-padding frames from every
  temporal loss. A real LIBERO sample verified two views, three future image
  groups, an interior mask `[true,true,true,true]`, and a last-step mask
  `[true,false,false,false]`.
- Config and launcher:

  ```text
  examples/LIBERO/train_files/starvla_smooth_spatial_latent_wm_libero.yaml
  examples/LIBERO/train_files/run_smooth_spatial_latent_wm_libero.sh
  ```

  The pilot run ID is
  `lewm_oft_libero_dinov3b_smooth_spatial_global384_short12_far8_10k`. It is
  configured for 10k optimizer steps, batch 4 with accumulation 8, and a
  from-scratch smooth branch over pretrained frozen DINOv3-B. It was launched
  detached on otherwise-empty GPU 1 on 2026-08-05 11:43 Asia/Shanghai with
  supervisor PID `672062`, micro-batch 4, accumulation 8, and global batch 32.
  The run directory contains `train.log`, `metrics.jsonl`, `train.pid`, and
  `STATUS.running`. The first 40 optimizer steps were finite; GPU usage was
  about 3.6 GiB and throughput was about 1.3 optimizer steps/s. Re-check the
  live process, metrics, and GPU state because these facts are volatile.

## Smooth global-latent action joint training

- The opt-in joint LIBERO policy uses
  `world_model.smooth_action_enabled=true`. Its action path does not consume
  DINO spatial tokens or robot state directly. Frozen DINOv3-B features from
  two views are compressed through the fixed 4x4-per-view two-stage projector
  to one 384-D `z_t`; the world predictor produces two future latents, and the
  action head receives exactly `[z_t, pred(z_t)_{t+1}, pred(z_t)_{t+2}]` as
  three single-token frames. Ground-truth future latents remain training
  targets only and are not fed to the action head.
- The total objective is action L1 plus the full smooth world-model objective,
  both weighted 1.0. The latter contains future-latent MSE plus the established
  SIGReg, slowness, acceleration, and temporal-order terms. DINO and the legacy
  visual/action path are frozen; trainable components are the two-stage latent
  projector/predictor, task embedding, one-token action cross-attention head,
  and action model.
- Formal config and launcher:

  ```text
  examples/LIBERO/train_files/starvla_smooth_global_latent_action_joint_libero_200k.yaml
  examples/LIBERO/train_files/run_smooth_global_latent_action_joint_libero_200k.sh
  ```

- The from-scratch joint branch (with only pretrained frozen DINO) was launched
  detached on GPU 0 on 2026-08-05 14:37 Asia/Shanghai. Run directory:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_smooth_global384_action_joint_200k
  ```

  Supervisor PID was `461343`; micro-batch 16, accumulation 2, global batch 32,
  and target 200k optimizer steps. At step 100, joint/action/world-model losses
  were 0.554449 / 0.221625 / 0.332824, respectively, and GPU usage was about
  5.4 GiB. Re-check the live process, log, metrics, and GPU state because these
  facts are volatile.
- Training reached 200k and saved
  `checkpoints/steps_200000_pytorch_model.pt` at 2026-08-07 18:19
  Asia/Shanghai. The final log reports normal completion and the former
  training process is no longer running. A structural load check found 381
  state-dict entries and confirmed the trained action path remains
  `384 -> 1536` (`visual_action_head.kv_proj`), followed by the checkpoint's
  `1536 -> 3072 -> 7` action MLP.
- The 160k checkpoint evaluation baseline was 0.94 LIBERO-Spatial, 0.94
  LIBERO-Object, 0.86 LIBERO-Goal, and 0.80 LIBERO-10: 354/400 = 0.885 overall.
- The corresponding 200k evaluation was launched detached at 2026-08-07 18:30
  Asia/Shanghai with supervisor PID `3574993`, GPUs `1,2,3,5`, ports
  `29970-29973`, seed 7, 10 trials for each of 10 tasks in all four suites,
  model-default execute horizon 8, and progress disabled. It uses Xvfb plus
  Mesa software GL while policy inference remains on GPU. Track it via
  `eval_steps_200000.status`, `eval_steps_200000.log`, and
  `eval_steps_200000_child_pids.txt` in the run directory. All four policy
  servers connected and entered their first episode before handoff; re-check
  these volatile facts before relying on them.
- That 200k evaluation completed at 2026-08-07 20:11 Asia/Shanghai. Success
  rates were 0.95 LIBERO-Spatial, 0.94 LIBERO-Object, 0.85 LIBERO-Goal, and
  0.78 LIBERO-10: 352/400 = 0.88 overall. The final summary is
  `libero_success_rates_steps_200000.txt` in the run directory.

## Smooth global state-conditioned train-encoder evaluation

- The `smooth_global384` ablation with normalized 8-D proprio conditioning
  and jointly finetuned DINOv3-B completed at 200k on 2026-08-10 06:05
  Asia/Shanghai. Its run directory and final checkpoint are:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_smooth_global384_statecond_trainenc_200k
  playground/Checkpoints/lewm_oft_libero_dinov3b_smooth_global384_statecond_trainenc_200k/checkpoints/steps_200000_pytorch_model.pt
  ```

- A strict current-source construction check loaded all checkpoint weights and
  confirmed `smooth_action_enabled=true`, `use_state_cond=true`,
  `expects_normalized_state=true`, `train_encoder=true`, and action horizon 8.
  All 387 state-dict tensors were finite. The action path is still
  `384 -> 1536 -> 3072 -> 7`; the state encoder is present and the deployment
  wrapper normalizes each LIBERO 8-D state using the saved training statistics.
  The statecond mixture uses the same 1,803 episodes, 40 semantic tasks, and 40
  canonical strings as the prior smooth-global run; only its proprio schema and
  normalization differ.
- Its four-suite closed-loop evaluation was launched detached at 2026-08-10
  10:32 Asia/Shanghai with supervisor PID `1322540`, GPUs `1,2,3,6`, ports
  `29980-29983`, seed 7, 10 trials for each of 10 tasks per suite, model-default
  execute horizon 8, progress disabled, and Xvfb/Mesa software GL. Track it via
  `eval_steps_200000.status`, `eval_steps_200000.log`, and
  `eval_steps_200000_child_pids.txt` in that run directory. All four policy
  servers connected, advertised the expected 8-D state schema, and entered
  their first episode without OOM, traceback, or missing-state errors before
  handoff; re-check these volatile facts before relying on them.
- A further 100k-step continuation is queued in a separate run directory so
  the original 200k artifacts and evaluation are not overwritten:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_smooth_global384_statecond_trainenc_continue200k_to300k
  ```

  Its detached queue/training supervisor PID is `1481610`. It waits for the
  ongoing 200k LIBERO evaluation supervisor `1322540` to exit (GPU 6 is still
  running LIBERO-10), then automatically launches on physical GPUs `0,4,5,6`.
  The target is external step 300k, using four processes, per-GPU batch 8,
  accumulation 1, and global batch 32. Training state cannot be restored
  exactly because the installed DeepSpeed 0.16.9 does not repartition a
  one-GPU ZeRO-2 optimizer checkpoint to a four-GPU DP world size. The run
  therefore preserves the 200k model weights and step numbering but rebuilds
  AdamW, advancing the newly constructed 300k cosine schedule to step 200k
  (base/encoder LR approximately `2.605e-5` / `2.605e-7`). Track it through
  `train.status`, `train.log`, and `train.pid` in the continuation run. The
  launcher is
  `examples/LIBERO/train_files/run_smooth_global_latent_action_joint_libero_statecond_trainenc_continue_100k.sh`.

## RoboTwin click_bell normalization-statistics ablation

- The first 20k click_bell fine-tune recomputed normalization statistics on the
  single-task 1,000-episode dataset. This changed the coordinate system relative
  to the Clean-50 200k warm-start checkpoint and was a major train/eval risk.
- `datasets.vla_data.normalization_statistics_path` is now an opt-in, strict
  training override. It reads a prior run's flattened `dataset_statistics.json`,
  validates embodiment tags, modality key order, and dimensions, splits the
  arrays back into per-key training metadata, installs them on the transforms,
  and saves the same statistics for deployment-time denormalization. Invalid
  files fail instead of silently falling back to statistics from new data.
- The click_bell launcher now defaults to the Clean-50 base run's statistics and
  resolved `config.full.yaml` (to lock the 1536-D action head and legacy language
  head as well as loading the weights), and uses a non-overwriting run ID:

  ```text
  playground/Checkpoints/lewm_oft_robotwin_dinov3b_clean50_canonical_click_bell_clean1000_basestats_ft20k
  ```

- A real-data smoke test on 78,055 frames / 1,000 trajectories verified that all
  saved action/state statistics and the action mask match the Clean-50 base file
  exactly. Unit coverage is in `tests/test_normalization_statistics_override.py`.
- The initial GPU-5 queue was cancelled at the user's request. The
  fixed-statistics 20k ablation was switched to immediate execution on physical
  GPU 1 on 2026-08-13 with supervisor PID `1935919`. Track `train.pid`,
  `train.log`, and `STATUS.running` in the new run directory. Re-check the PID
  and GPU state because they are volatile.

## Worktree safety

- Preserve unrelated user-owned untracked files and directories, especially
  `thirdparty/`, `BEHAVIOR-1K/`, and DINO weights.
- Tracked changes from the earlier `dino-compact-latent-wm` work were saved as:

  ```text
  stash@{0}: pre-wala-dev-ai-switch-20260722
  ```

  Stash indices can change; verify with `git stash list` before using it.
- Do not restore, delete, or overwrite that stash or unrelated untracked work
  unless explicitly requested.
