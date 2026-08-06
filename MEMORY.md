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
- The run `lewm_oft_dinov3b_densepatch_action_zeroout_from200k` is an exception:
  it was launched in a persistent PTY without `nohup` or `setsid`.

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

## LIBERO predictable local-dynamics experiment

- The next experiment redefines the target rather than adding another 384-D
  residual head. Frozen-base cumulative errors at +4/+8 are converted into
  local increments; a shared 4x32 row-orthonormal spatial basis and shared
  64x384 row-orthonormal channel basis encode each increment as four 64-D
  transition modes. Decoding and cumulative summation returns to the original
  normalized DINO delta coordinates, so `latent_loss` remains directly
  comparable. The code has 512 dimensions (`2*4*64`) instead of the rejected
  per-spatial-token candidate's 4,096 dimensions (`2*32*64`).
- The predictor is action- and state-free. Its private visual history is the
  deployment-compatible `[t-2,t-1,t]`; the validated 220k base predictor and
  action path remain strictly ctx1 and see `[t,t+4,t+8]`. Policy-server
  metadata now advertises `innovation_context_len=3` only for such checkpoints.
- Fixed token identity is removed with a frozen raw-coordinate training-set
  mean, never a micro-batch mean. Calibration used 8,192 samples from
  `libero_all_wm_l10_augmented_localctx3_h8` with the frozen DINOv3 encoder,
  spatial pooler, task embedding, and 220k base predictor. Artifact:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_localinc_calibration/train_local_error_mean_8192.pt
  ```

  It is finite fp32 with shape `[1,2,32,384]`, `delta_scale=1.680324673652649`,
  mean RMS 0.0113793, dynamic std mean 0.843115, and first-half/second-half mean
  RMS difference 0.0189181 (2.24% of dynamic std).
- Training is deliberately staged. Stage A trains only the two bases (24,704
  parameters) for 2k steps using dynamic capture; Stage B freezes the learned
  coordinates and trains only the causal predictor (6,048,320 parameters) for
  10k steps. Both preserve all 319 source checkpoint keys by shape, use batch
  4 / accumulation 8 / global batch 32, and do not touch the RoboTwin GPU4 run.
  Stage A run directory and detached supervisor are:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_wmonly_localbasis_m4r64_fixedmean_2k
  supervisor PID 3080835, GPU 0, port 29649
  ```

  At its initial random-basis batch, dynamic explained fraction was about
  0.0259 versus the isotropic separable null
  `(4/32)*(64/384)=0.02083`. By step 1,050, the per-optimizer-step aggregated
  trailing-500-step window explained 0.3173 of dynamic error, reduced raw
  normalized MSE by an oracle 0.1613 (0.1617 / 0.1609 at +4 / +8), and had
  p10 explained fraction 0.3095. The fixed-mean-only gain remained only
  `-0.0000196`, `delta_scale` was unchanged, and the zero/frozen student still
  matched the base exactly. This is strong representation capture, not yet
  causal predictability.
- Automatic promotion is supervised by the detached pipeline at:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_predictable_local_dynamics_pipeline
  strict watcher PID 1575005
  ```

  It waits for the successful Stage-A supervisor exit, then requires a
  loadable step-2k checkpoint and final model, the step-2k summary record,
  exact basis/predictor invariants, and 20 complete eight-microbatch groups in
  steps 1,500--1,975. The representation gate requires explained fraction
  >=0.25, p10>=0.20, raw oracle gain>=0.10, each horizon gain>=0.07, effective
  rank>=6, orthogonality error<1e-4, negligible fixed-mean gain, and no recent
  collapse. Only then does it launch the action/state-free Stage-B predictor
  on GPU0 and run the separate predictor training gate at 10k.
- The first cumulative per-token implementation attempts were stopped before
  warmup and archived because seed/policy compatibility and then mixed
  batch-token statistics invalidated their conclusions. Do not use artifacts
  from directories suffixed `unseeded_attempt_20260804_112633` or
  `tokenbias_attempt_20260804_113320`.
- A training PASS is only a feasibility gate. The current calibration and both
  stages sample the full training mixture, so an episode split made afterward
  must not be called leakage-free held-out evaluation. A strict result requires
  a frozen per-dataset episode manifest made *before* calibration, with the
  mean, basis, and predictor all fit on train episodes only, followed by paired
  held-out raw gain/confidence intervals and zero-code/history-null/
  matched-history-shuffle ablations. The separate RoboTwin training on GPU 4
  remained alive and unchanged.
- Stage B completed all 10k steps but failed its predictor gate over steps
  8k--10k: code NMSE was `0.99994`, cosine `0.00748`, predicted effective rank
  `3.01`, raw normalized-latent gain `0.0000098`, and realized oracle headroom
  `0.000062`. The learned Stage-A coordinates therefore captured error energy
  but were not learned by the action/state-free predictor on the normal stream.
- A separate fixed-anchor capacity probe was then run without changing either
  Stage A/B. It strict-loaded the Stage-A 2k checkpoint, cached one unpadded
  sample from each of 128 distinct episodes, and trained only the 6.048M
  predictor parameters in fp32 with a code-only objective. It passed three
  consecutive full-set gates and stopped at step 550: code NMSE `0.03735`,
  cosine `0.98154`, and realized headroom `0.93424`. Artifacts are at:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_localinc_stageb_overfit128_fp32_seed42_2k
  playground/Checkpoints/lewm_oft_libero_dinov3b_localinc_stageb_anchor_overfit_pipeline
  ```

  This rules out a basic gradient disconnection or fixed-sample memorization
  capacity failure. It does **not** establish held-out predictability: the
  optimizer is an easier capacity recipe (`1e-3`, fixed-energy code NMSE,
  no raw loss), and the gate is on its training anchors only. The next probe
  must fit a predictive subspace on train episodes, select on validation, and
  open disjoint test episodes once. GPU 4 sharing for this short diagnostic was
  explicitly authorized; its extra approximately 1.5 GiB was released, and
  the RoboTwin run remained alive.
- Stage C implemented that conditional held-out probe in
  `diagnose_raw_predictive_subspace.py`. It ignores the Stage-A code/basis as a
  target, freezes the full-mixture 220k encoder/pooler/base/task path, and fits
  a new reduced-rank regression target on the raw local residual left after the
  base prediction at +4/+8. Inputs are current latent, two causal history-motion
  differences, the frozen base prediction, and goal; no action, state, or future
  input is used. One unpadded anchor per disjoint episode is split
  train/validation/test=`640/192/256`, covering all 40 tasks. Normalisation,
  response mean, kernels, and subspaces are train-only; rank/ridge are selected
  on validation before the sealed test cache is opened for metrics.
- The final audited Stage-C run completed normally in:

  ```text
  playground/Checkpoints/lewm_oft_libero_dinov3b_raw_predictive_subspace_probe_train640_val192_test256_seed42_bijective_v3
  ```

  The history-shuffle null is a true bijection with no fixed points and exactly
  preserves the empirical history marginal. Train/test keep all 640/256 donors
  within task; validation keeps 189/192 within task and cycles its three
  singleton tasks as an explicit fallback. All four validation variants
  (full, no-history, task-only, fit-time shuffled history) selected rank 0. The
  best nonzero full candidate was rank 4 / ridge 10, but was worse than the
  train-mean correction: raw MSE `0.508957` versus `0.508733`, gain
  `-0.000224`, code NMSE `1.03413`, cosine `0.08385`, and realized oracle
  headroom `-0.1090`. On the sealed test set, the frozen base raw MSE was
  `0.508689`; the selected rank-0/train-mean result was `0.509461`, a gain of
  `-0.000772` with 2,000-repeat paired-task bootstrap 95% CI
  `[-0.000941,-0.000612]`. Only 2/40 tasks improved, and all eight scientific
  gates failed. Fifteen declared artifact hashes, tensor finiteness, split
  isolation, selection freeze, and status files were audited successfully.
- Stage C therefore gives a clear negative result for this exact recipe: the
  +4/+8 frozen-base residual contains compressible/oracle energy, and the
  128-anchor probe proves memorisation capacity, but no episode-held-out linear
  action/state-free predictable subspace was found from current/history/base/
  goal inputs. Do not scale rank, fitting time, or another neural head on this
  same target. If continuing without action conditioning, change the prediction
  problem first (for example +1/+2 local horizons and an explicitly
  predictability-trained representation). This remains a conditional probe,
  not an end-to-end leakage-free benchmark, because its frozen representation
  was trained on the full mixture.

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


## Patch Policy-style dense current-patch action residual

- The validated DINOv3-B/16 encoder yields a `14 x 14 x 768` patch grid per
  view. The baseline compresses this to a `4 x 4 x 384` grid per view before
  its compact world model.
- The low-risk Patch Policy experiment keeps that compact path frozen and adds
  a training/deployment action residual that cross-attends the action queries
  to all current-frame raw patches (`2 x 14 x 14`). It never consumes true
  future patches.
- The residual output projection is zero-initialized, so step-0 action outputs
  are exactly equal to the validated 200k baseline while gradients can open the
  dense branch immediately.
- Launcher:

  ```bash
  CUDA_DEVS=<free-gpu-ids> \
  ACCELERATE_BIN=.venv/bin/accelerate \
  WANDB_MODE=disabled \
  bash examples/LIBERO/train_files/run_lewm_oft_dinov3_dense_patch_action.sh
  ```

- Formal 20k-step run started on 2026-07-23 from the validated 200k baseline:

  ```text
  playground/Checkpoints/lewm_oft_dinov3b_densepatch_action_zeroout_from200k
  ```

  At step 100/500, action L1 was `0.05023 / 0.04813` and
  `dense_patch_query_update_ratio` was `0.00790 / 0.01631`, confirming that the
  branch is active. Monitor `action_dit_loss` for policy fitting and
  `dense_patch_query_update_ratio` plus `dense_patch_residual_rms` for branch
  usage. Checkpoints are written every 2k steps.

## Dense-only 14x14 strict baseline control

- The primary high-resolution experiment is a strict spatial-resolution control,
  not a continuation from the validated 200k checkpoint. It starts from the
  same raw pretrained DINOv3 encoder as the 4x4 baseline; every downstream
  module starts randomly initialized.
- The only intended model variable is spatial resolution: each of two views
  keeps all `14 x 14 = 196` DINO patches (392 tokens/frame). The world model
  predicts two future 392-token frames, and the action head reads the current
  plus predicted future dense tokens. The 4x4 path and dense residual adapter
  are disabled.
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

## Joint reconstructive compact-latent world model

- The opt-in RoboTwin Clean implementation is
  `world_model.reconstructive_latent_enabled=true`. DINOv3-B/16 remains
  frozen; the trainable path jointly optimizes only a spatial codec, compact
  latent predictor, state decoder, and task embedding.
- It uses one training flow and exactly three weighted objectives: normalized
  pooled-DINO reconstruction, future compact-latent prediction, and aligned
  14-D robot-state decoding. There are no variance/covariance losses and no
  state/action input to the dynamics predictor.
- The dedicated data type `robotwin_reconstructive_wm` loads image and state
  indices `[0,1,2]`. Keep it separate from the established RoboTwin world-model
  recipe, whose image indices are `[0,8,16]` and state is current-only.
- Config and launcher:

  ```text
  examples/Robotwin/train_files/starvla_reconstructive_latent_wm_robotwin_clean.yaml
  examples/Robotwin/train_files/run_reconstructive_latent_wm_clean.sh
  ```

  This is a world-model-only representation experiment; `predict_action`
  intentionally rejects it because the compact latent is not yet connected to
  the action head. No formal run was launched when the implementation landed.

- The corresponding LIBERO experiment uses the same three-loss architecture
  with two views, 8-D normalized state, and aligned consecutive frame indices
  `[0,1,2]`. It samples the same four-suite plus recovered/teacher LIBERO-10
  mixture as the previous augmented-L10 experiments. It starts the compact
  branch from scratch (only DINOv3-B is pretrained), with micro-batch 4,
  accumulation 8, global batch 32, and 50k optimizer steps. Config, launcher,
  and run directory are:

  ```text
  examples/LIBERO/train_files/starvla_reconstructive_latent_wm_libero.yaml
  examples/LIBERO/train_files/run_reconstructive_latent_wm_libero.sh
  playground/Checkpoints/lewm_oft_libero_dinov3b_reconstructive_latent_joint_short12_50k
  ```

  It was launched detached on shared GPU 4 on 2026-08-04 with supervisor PID
  `1818532`. At step 20 it used about 4.55 GiB in addition to the existing
  RoboTwin process, leaving about 22.5 GiB free. The logged three losses were
  finite and optimization had started. The first launch attempt is preserved
  as `train.attempt1_world_model_eval_guard.log`; it stopped before step 1
  because the trainer tried to run an irrelevant action diagnostic at step 0.
  `train_starvla.py` now skips action evaluation for every explicit
  `world_model_only` run.

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
