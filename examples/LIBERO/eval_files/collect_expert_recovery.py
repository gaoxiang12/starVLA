# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Hand the failure cases over to a strong expert (e.g. Qwen-OFT) and export the
successful *recoveries* as a training-ready LeRobot v2.0 dataset.

This is step two of the DAgger / intervention-style loop:

    1. ``collect_failures.py``  -> records every failed rollout of YOUR model,
       including the flattened mujoco ``sim_states`` captured *before* each
       policy step (so any decision point can be reproduced exactly).

    2. ``collect_expert_recovery.py`` (this file) -> for every failed episode it
       resets the environment to a chosen *handover* timestep, then lets the
       EXPERT policy (served on a separate port) take over. If the expert
       succeeds, the expert-driven segment is written out as a LeRobot episode.

The produced dataset lands under ``--args.out-dataset-dir`` (auto-placed next to
the failure cases by default) and follows the exact on-disk schema consumed by
``starVLA.dataloader.gr00t_lerobot.datasets.LeRobotSingleDataset`` (LeRobot
v2.0): ``data/chunk-000/episode_XXXXXX.parquet`` +
``videos/chunk-000/{image,wrist_image}/episode_XXXXXX.mp4`` + ``meta/*``. The
loader auto-computes ``meta/stats_gr00t.json`` and ``meta/steps_data_index.pkl``
on first use, so those are intentionally not written here.

Handover strategies (``--args.handover-mode``):
    * ``start``    -> always hand over at the first decision point (k=0). The
                      expert effectively re-solves the whole task from the
                      episode's init state. NOTE: these init states are the
                      standard suite init states the expert already trained on,
                      so this data is ~redundant with the original dataset and
                      of little value for correcting YOUR model.
    * ``fraction`` -> hand over once, at ``handover_fractions[0]`` of the way
                      through the failed rollout.
    * ``scan``     -> try each fraction in ascending order and keep the FIRST
                      one that recovers (earliest / smallest-k takeover).
    * ``deepest``  -> (default) try fractions in DESCENDING order and keep the
                      LARGEST k that still recovers -- i.e. the most drifted,
                      off-distribution state your model actually visited that
                      the expert can still rescue. This is the genuinely novel
                      "recovery" data missing from the original dataset.

Set ``--args.keep-all-recoveries`` to write EVERY recoverable handover point of
a failure as its own episode (multiple drift levels per failure), instead of
stopping at the first success.

Usage (THREE things run at once):

    # 1) expert policy server (Qwen-OFT), on its own GPU/port
    CKPT=/path/to/qwen_oft/checkpoints/steps_XXXXX_pytorch_model.pt \
        GPU_ID=1 PORT=6700 bash examples/LIBERO/eval_files/run_policy_server.sh

    # 2) point this collector at that server + the failure cases produced earlier
    EXPERT_CKPT=/path/to/qwen_oft/.../steps_XXXXX_pytorch_model.pt \
        LIBERO_HOME=$PWD/playground/LIBERO PORT=6700 \
        FAILURES_DIR=$PWD/playground/Checkpoints/lewm_oft_libero_wm_vfuse/failure_cases/libero_goal/steps_80000_pytorch_model \
        bash examples/LIBERO/eval_files/collect_expert_recovery.sh
"""

import dataclasses
import json
import logging
import os
import pathlib
import shutil
import time

import imageio
import numpy as np
import pandas as pd
import torch
import tqdm
import tyro

# Reuse the exact preprocessing / env helpers from the evaluator so recovery
# trajectories stay byte-compatible with normal evaluation and with the failure
# cases. Importing this module also installs the ``torch.load(weights_only=False)``
# shim needed by the LIBERO init-state files.
from examples.LIBERO.eval_files.eval_libero import (
    LIBERO_ENV_RESOLUTION,
    _binarize_gripper_open,
    _get_libero_env,
    _quat2axisangle,
)
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

from libero.libero import benchmark

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Keep a reference so the import above is not flagged as unused; the shim is a
# side effect of importing eval_libero.
_ = torch.load

_MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DEFAULT_TEMPLATE_INFO = (
    _REPO_ROOT
    / "playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/meta/info.json"
)
_DEFAULT_MODALITY = _REPO_ROOT / "examples/LIBERO/train_files/modality.json"

# The two LeRobot video keys expected by the LIBERO modality.json.
_VIDEO_KEYS = ["observation.images.image", "observation.images.wrist_image"]


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 6700  # port of the EXPERT policy server (Qwen-OFT)

    #################################################################################################################
    # Inputs
    #################################################################################################################
    # Root produced by collect_failures.py (contains manifest.jsonl + per-episode dirs).
    failures_dir: str = ""
    task_suite_name: str = "libero_goal"
    max_steps: int = -1  # -1 -> use the per-suite table
    limit_episodes: int = -1  # cap number of failure cases processed (-1 = all)

    #################################################################################################################
    # Handover strategy
    #################################################################################################################
    handover_mode: str = "deepest"  # one of: deepest | scan | fraction | start
    # Comma-separated fractions of the failed rollout at which to try handing
    # over. k = round(fraction * (N - 1)) where N is the failed episode length.
    handover_fractions: str = "0.1,0.2,0.3,0.4,0.5,0.6,0.7"
    # If True, keep EVERY recoverable handover point (multiple episodes per
    # failure). If False, stop at the first success in the scan order
    # (deepest -> largest k; scan -> smallest k).
    keep_all_recoveries: bool = False

    #################################################################################################################
    # Output (LeRobot v2.0 dataset)
    #################################################################################################################
    # Empty == auto: <failures_dir>/../recovery_lerobot/<suite>_recovery_lerobot
    out_dataset_dir: str = ""
    template_info_json: str = str(_DEFAULT_TEMPLATE_INFO)
    modality_json: str = str(_DEFAULT_MODALITY)
    fps: int = 20
    save_debug_video: bool = False  # also dump a plain .mp4 per recovery for eyeballing

    seed: int = 7
    expert_ckpt: str = ""  # provenance only, recorded in meta
    unnorm_key: str | None = None

    job_name: str = "collect_expert_recovery"


# ---------------------------------------------------------------------------
# LeRobot v2.0 incremental writer
# ---------------------------------------------------------------------------
class LeRobotV2Writer:
    """Write successful recovery segments into a LeRobot v2.0 dataset directory.

    Layout matches what ``LeRobotSingleDataset`` (v2.0 path) expects:
        data/chunk-000/episode_{idx:06d}.parquet
        videos/chunk-000/{video_key}/episode_{idx:06d}.mp4
        meta/{info.json, tasks.jsonl, episodes.jsonl, modality.json}
    ``stats_gr00t.json`` and ``steps_data_index.pkl`` are auto-generated by the
    loader on first use, so they are not written here.
    """

    def __init__(
        self,
        out_dir: str | pathlib.Path,
        template_info_json: str,
        modality_json: str,
        fps: int = 20,
        chunks_size: int = 1000,
    ) -> None:
        self.root = pathlib.Path(out_dir)
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for vk in _VIDEO_KEYS:
            (self.root / "videos" / "chunk-000" / vk).mkdir(parents=True, exist_ok=True)

        shutil.copyfile(modality_json, self.root / "meta" / "modality.json")
        with open(template_info_json, "r") as f:
            self.info_template = json.load(f)

        self.fps = fps
        self.chunks_size = chunks_size
        self.tasks: dict[str, int] = {}
        self.episodes: list[dict] = []
        self.global_index = 0
        self.total_frames = 0

    def _task_index(self, task: str) -> int:
        if task not in self.tasks:
            self.tasks[task] = len(self.tasks)
        return self.tasks[task]

    def add_episode(
        self,
        images_main: list,
        images_wrist: list,
        states: list,
        actions: list,
        task: str,
    ) -> int:
        ep = len(self.episodes)
        n = len(actions)
        assert n > 0, "cannot write an empty episode"
        assert len(images_main) == len(images_wrist) == len(states) == n, (
            f"length mismatch: imgs={len(images_main)}/{len(images_wrist)} "
            f"states={len(states)} actions={n}"
        )
        task_index = self._task_index(task)

        frame_index = np.arange(n, dtype=np.int64)
        df = pd.DataFrame(
            {
                "observation.state": [np.asarray(s, dtype=np.float32) for s in states],
                "action": [np.asarray(a, dtype=np.float32) for a in actions],
                "timestamp": (frame_index / float(self.fps)).astype(np.float32),
                "frame_index": frame_index,
                "episode_index": np.full(n, ep, dtype=np.int64),
                "index": np.arange(self.global_index, self.global_index + n, dtype=np.int64),
                "task_index": np.full(n, task_index, dtype=np.int64),
            }
        )
        df.to_parquet(self.root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet", index=False)

        self._write_video(
            self.root / "videos" / "chunk-000" / "observation.images.image" / f"episode_{ep:06d}.mp4",
            images_main,
        )
        self._write_video(
            self.root
            / "videos"
            / "chunk-000"
            / "observation.images.wrist_image"
            / f"episode_{ep:06d}.mp4",
            images_wrist,
        )

        self.episodes.append({"episode_index": ep, "tasks": [task], "length": n})
        self.global_index += n
        self.total_frames += n
        return ep

    def _write_video(self, path: pathlib.Path, frames: list) -> None:
        imageio.mimwrite(
            path,
            [np.ascontiguousarray(np.asarray(f, dtype=np.uint8)) for f in frames],
            fps=self.fps,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
            macro_block_size=1,
            output_params=["-crf", "18"],
        )

    def finalize(self) -> dict:
        with open(self.root / "meta" / "tasks.jsonl", "w") as f:
            for task, ti in sorted(self.tasks.items(), key=lambda kv: kv[1]):
                f.write(json.dumps({"task_index": ti, "task": task}) + "\n")

        with open(self.root / "meta" / "episodes.jsonl", "w") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep) + "\n")

        info = dict(self.info_template)
        info["total_episodes"] = len(self.episodes)
        info["total_frames"] = self.total_frames
        info["total_tasks"] = len(self.tasks)
        info["total_videos"] = len(self.episodes) * len(_VIDEO_KEYS)
        info["total_chunks"] = 1
        info["chunks_size"] = self.chunks_size
        info["fps"] = self.fps
        info["splits"] = {"train": f"0:{len(self.episodes)}"}
        # Reflect the codec we actually wrote (template says av1).
        for vk in _VIDEO_KEYS:
            feat = info.get("features", {}).get(vk)
            if feat and "info" in feat:
                feat["info"]["video.codec"] = "h264"
                feat["info"]["video.pix_fmt"] = "yuv420p"
                feat["info"]["video.fps"] = self.fps
        with open(self.root / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=4)

        return {
            "total_episodes": len(self.episodes),
            "total_frames": self.total_frames,
            "total_tasks": len(self.tasks),
        }


# ---------------------------------------------------------------------------
# Failure-case discovery
# ---------------------------------------------------------------------------
def _iter_failure_cases(failures_dir: str):
    """Yield ``(episode_dir, meta_dict)`` for every FAILED episode in the dir.

    Prefers ``manifest.jsonl`` (written by collect_failures) but falls back to
    globbing ``**/episode_*/meta.json`` if the manifest is missing.
    """
    root = pathlib.Path(failures_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"failures_dir does not exist: {root}")

    manifest = root / "manifest.jsonl"
    seen = set()
    if manifest.is_file():
        with open(manifest, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("success", False):
                    continue  # only recover from failures
                ep_dir = root / entry["dir"]
                meta_path = ep_dir / "meta.json"
                if not meta_path.is_file():
                    logging.warning(f"manifest points to missing meta: {meta_path}")
                    continue
                seen.add(str(ep_dir))
                with open(meta_path, "r") as mf:
                    yield ep_dir, json.load(mf)
        return

    logging.warning("manifest.jsonl not found; globbing episode dirs instead.")
    for meta_path in sorted(root.glob("**/episode_*/meta.json")):
        ep_dir = meta_path.parent
        if str(ep_dir) in seen:
            continue
        with open(meta_path, "r") as mf:
            meta = json.load(mf)
        if meta.get("success", False):
            continue
        yield ep_dir, meta


# ---------------------------------------------------------------------------
# Expert rollout from a handover state
# ---------------------------------------------------------------------------
def _rollout_expert(
    env,
    client_model: ModelClient,
    task_description: str,
    init_state: np.ndarray,
    handover_state: np.ndarray,
    max_steps: int,
):
    """Reset ``env`` to ``handover_state`` and let the expert policy run.

    Returns ``(success, images_main, images_wrist, states, actions)`` where the
    lists are aligned (frame[t] / state[t] paired with the action[t] applied at
    step t), matching the LIBERO training-demo convention.
    """
    client_model.reset(task_description=task_description)
    env.reset()
    # Establish the canonical initial scene, then jump to the failure point.
    env.set_init_state(np.asarray(init_state))
    obs = env.regenerate_obs_from_state(np.asarray(handover_state, dtype=np.float64))

    images_main: list = []
    images_wrist: list = []
    states: list = []
    actions: list = []

    success = False
    for step in range(max_steps):
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ).astype(np.float32)

        example_dict = {"image": [img, wrist_img], "lang": str(task_description)}
        response = client_model.step(example=example_dict, step=step)
        raw_action = response["raw_action"]

        world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
        rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
        open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
        gripper = _binarize_gripper_open(open_gripper)

        if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
            raise ValueError(
                f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                f"rotation_delta={rotation_delta.shape}, gripper={open_gripper.shape}"
            )
        delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

        images_main.append(img)
        images_wrist.append(wrist_img)
        states.append(state)
        actions.append(delta_action.astype(np.float32))

        obs, reward, done, info = env.step(delta_action.tolist())
        if done:
            success = True
            break

    return success, images_main, images_wrist, states, actions


def _resolve_handover_ks(n: int, mode: str, fractions: list[float]) -> list[int]:
    """Map handover fractions to sorted, de-duplicated step indices in [0, n-1]."""
    if n <= 0:
        return []
    if mode == "start":
        return [0]
    if mode == "fraction":
        fractions = fractions[:1]
    ks = sorted(
        {max(0, min(n - 1, int(round(f * (n - 1))))) for f in fractions if 0.0 <= f <= 1.0},
        reverse=(mode == "deepest"),
    )
    return ks


def collect_expert_recovery(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    np.random.seed(args.seed)

    if not args.failures_dir:
        raise ValueError("--args.failures-dir is required (output of collect_failures.py).")
    if args.task_suite_name not in _MAX_STEPS_BY_SUITE:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = args.max_steps if args.max_steps > 0 else _MAX_STEPS_BY_SUITE[args.task_suite_name]

    fractions = [float(x) for x in args.handover_fractions.split(",") if x.strip()]
    if not fractions:
        fractions = [0.0]

    # Output location.
    if args.out_dataset_dir:
        out_dir = args.out_dataset_dir
    else:
        out_dir = str(
            pathlib.Path(args.failures_dir).parent
            / "recovery_lerobot"
            / f"{args.task_suite_name}_recovery_lerobot"
        )
    logging.info(f"Recovery dataset will be written to: {out_dir}")

    debug_dir = pathlib.Path(out_dir) / "_debug_videos"
    if args.save_debug_video:
        debug_dir.mkdir(parents=True, exist_ok=True)

    writer = LeRobotV2Writer(
        out_dir=out_dir,
        template_info_json=args.template_info_json,
        modality_json=args.modality_json,
        fps=args.fps,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    client_model = ModelClient(host=args.host, port=args.port, unnorm_key=args.unnorm_key)

    # Lazily created env per task_id (reused across episodes of the same task).
    env_cache: dict[int, tuple] = {}

    def _get_env(task_id: int):
        if task_id not in env_cache:
            task = task_suite.get_task(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            env_cache[task_id] = (env, task_description)
        return env_cache[task_id]

    manifest_path = pathlib.Path(out_dir) / "recovery_manifest.jsonl"
    n_processed = 0
    n_recovered = 0
    per_frac_hits: dict[str, int] = {}

    cases = list(_iter_failure_cases(args.failures_dir))
    if args.limit_episodes > 0:
        cases = cases[: args.limit_episodes]
    logging.info(f"Found {len(cases)} failure case(s) to attempt recovery on.")

    for ep_dir, meta in tqdm.tqdm(cases, desc="failures"):
        task_id = int(meta["task_id"])
        task_description = meta["task_description"]
        if meta.get("task_suite_name") and meta["task_suite_name"] != args.task_suite_name:
            logging.warning(
                f"case {ep_dir} was from suite {meta['task_suite_name']} but "
                f"task_suite_name={args.task_suite_name}; skipping."
            )
            continue

        sim_states = np.load(ep_dir / "sim_states.npy")
        init_state = np.load(ep_dir / "init_state.npy")
        n = int(sim_states.shape[0])
        if n == 0:
            logging.warning(f"{ep_dir}: empty sim_states, skipping.")
            continue

        env, env_task_desc = _get_env(task_id)
        candidate_ks = _resolve_handover_ks(n, args.handover_mode, fractions)

        n_processed += 1
        recovered = False
        for k in candidate_ks:
            success, imgs, wrists, states, actions = _rollout_expert(
                env=env,
                client_model=client_model,
                task_description=env_task_desc,
                init_state=init_state,
                handover_state=sim_states[k],
                max_steps=max_steps,
            )
            if success and len(actions) > 0:
                ep_index = writer.add_episode(
                    images_main=imgs,
                    images_wrist=wrists,
                    states=states,
                    actions=actions,
                    task=env_task_desc,
                )
                if args.save_debug_video:
                    imageio.mimwrite(
                        debug_dir / f"recovery_ep{ep_index:06d}_task{task_id}_k{k}.mp4",
                        [np.asarray(x) for x in imgs],
                        fps=10,
                        format="FFMPEG",
                        macro_block_size=1,
                    )
                frac = round(k / max(1, n - 1), 4)
                per_frac_hits[str(frac)] = per_frac_hits.get(str(frac), 0) + 1
                with open(manifest_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "recovery_episode_index": ep_index,
                                "source_dir": str(pathlib.Path(ep_dir).relative_to(args.failures_dir)),
                                "task_id": task_id,
                                "task_description": env_task_desc,
                                "handover_step": k,
                                "handover_fraction": frac,
                                "failed_length": n,
                                "recovery_length": len(actions),
                            }
                        )
                        + "\n"
                    )
                recovered = True
                if not args.keep_all_recoveries:
                    break

        if recovered:
            n_recovered += 1
        logging.info(
            f"{pathlib.Path(ep_dir).name}: {'RECOVERED' if recovered else 'still failed'} "
            f"| recovered={n_recovered}/{n_processed}"
        )

    for _, (env, _desc) in env_cache.items():
        env.close()

    stats = writer.finalize()

    summary = {
        "failures_dir": args.failures_dir,
        "out_dataset_dir": out_dir,
        "task_suite_name": args.task_suite_name,
        "handover_mode": args.handover_mode,
        "handover_fractions": fractions,
        "keep_all_recoveries": args.keep_all_recoveries,
        "processed_failures": n_processed,
        "recovered_failures": n_recovered,
        "recovery_rate": (n_recovered / n_processed) if n_processed else 0.0,
        "recoveries_by_handover_fraction": per_frac_hits,
        "dataset_stats": stats,
        "expert_ckpt": args.expert_ckpt,
        "server_meta": client_model._server_metadata,
    }
    with open(pathlib.Path(out_dir) / "recovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Done. {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    tyro.cli(collect_expert_recovery)
