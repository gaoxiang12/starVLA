#!/usr/bin/env python3
"""Collect successful teacher rollouts from missing LIBERO-10 train states.

The audit produced by ``audit_libero10_raw_vs_lerobot.py`` identifies official
training demonstrations that are absent from the current LeRobot conversion.
This collector resets LIBERO to those demonstrations' initial simulator states,
runs a policy served by ``run_policy_server.sh``, and writes only successful
rollouts to a separate LeRobot v2 dataset.

This intentionally does not use ``task_suite.get_task_init_states``: those are
the benchmark's fixed evaluation states and must not become training data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import h5py
import imageio
import numpy as np

from examples.LIBERO.eval_files.collect_expert_recovery import LeRobotV2Writer
from examples.LIBERO.eval_files.eval_libero import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _get_libero_env,
    _quat2axisangle,
)
from examples.LIBERO.eval_files.model2libero_interface import ModelClient
from libero.libero import benchmark


def task_from_source_file(source_file: str) -> str:
    return source_file.removesuffix("_demo.hdf5")


def successful_sources(manifest_paths: list[Path]) -> set[tuple[str, str]]:
    """Return raw (file, demo) pairs already recovered by earlier collectors."""
    recovered: set[tuple[str, str]] = set()
    for path in manifest_paths:
        if not path.is_file():
            raise FileNotFoundError(f"exclude manifest does not exist: {path}")
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("success", False):
                    recovered.add(
                        (record["source_file"], record["source_demo"])
                    )
    return recovered


def rollout_teacher(
    *,
    env,
    client: ModelClient,
    task_description: str,
    init_state: np.ndarray,
    settle_steps: int,
    max_steps: int,
) -> tuple[bool, list, list, list, list]:
    """Run one teacher rollout and return aligned LeRobot frame data."""
    client.reset(task_description=task_description)
    env.reset()
    obs = env.set_init_state(np.asarray(init_state, dtype=np.float64))
    for _ in range(settle_steps):
        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        if done:
            # A task that is already solved is not a useful policy trajectory.
            return False, [], [], [], []

    images_main: list[np.ndarray] = []
    images_wrist: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for step in range(max_steps):
        image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(
            obs["robot0_eye_in_hand_image"][::-1, ::-1]
        )
        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ).astype(np.float32)

        response = client.step(
            example={
                "image": [image, wrist],
                "lang": str(task_description),
                # QwenOFT consumes a short state history and indexes state[0]
                # when constructing its discretized-state prompt.
                "state": np.expand_dims(state, axis=0),
            },
            step=step,
        )
        raw_action = response["raw_action"]
        translation = np.asarray(
            raw_action.get("world_vector"), dtype=np.float32
        ).reshape(-1)
        rotation = np.asarray(
            raw_action.get("rotation_delta"), dtype=np.float32
        ).reshape(-1)
        open_gripper = np.asarray(
            raw_action.get("open_gripper"), dtype=np.float32
        ).reshape(-1)
        if not (
            translation.size == 3
            and rotation.size == 3
            and open_gripper.size == 1
        ):
            raise ValueError(
                "invalid teacher action sizes: "
                f"translation={translation.shape}, rotation={rotation.shape}, "
                f"open_gripper={open_gripper.shape}"
            )

        # LeRobot stores open_gripper as {0=closed, 1=open}. LIBERO's simulator
        # instead consumes {+1=close, -1=open}.
        stored_open = np.asarray(
            [float(open_gripper[0] > 0.5)], dtype=np.float32
        )
        stored_action = np.concatenate(
            [translation, rotation, stored_open], axis=0
        )
        simulator_action = stored_action.copy()
        simulator_action[-1] = 1.0 - 2.0 * stored_action[-1]

        images_main.append(image)
        images_wrist.append(wrist)
        states.append(state)
        actions.append(stored_action)

        obs, _, done, _ = env.step(simulator_action.tolist())
        if done:
            return True, images_main, images_wrist, states, actions

    return False, images_main, images_wrist, states, actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6700)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=0,
        help="Actions executed per prediction; 0 uses the model chunk size.",
    )
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="JSONL replay/teacher manifest whose successful sources are skipped.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--template-info-json",
        type=Path,
        default=Path(
            "playground/Datasets/LEROBOT_LIBERO_DATA/"
            "libero_10_no_noops_1.0.0_lerobot/meta/info.json"
        ),
    )
    parser.add_argument(
        "--modality-json",
        type=Path,
        default=Path("examples/LIBERO/train_files/modality.json"),
    )
    parser.add_argument("--task-filter", default="")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--save-debug-video", action="store_true")
    parser.add_argument("--teacher-ckpt", default="", help="Provenance only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s | %(message)s",
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}. "
            "Use a new directory so collection provenance is not mixed."
        )

    audit = json.loads(args.audit_manifest.read_text())
    excluded = successful_sources(args.exclude_manifest)
    needle = args.task_filter.strip().lower()
    candidates: list[tuple[str, dict]] = []
    skipped_existing = 0
    for task_description, task_report in audit["tasks"].items():
        if needle and needle not in task_description.lower():
            continue
        for demo in task_report["missing_demos"]:
            source = (demo["source_file"], demo["demo"])
            if source in excluded:
                skipped_existing += 1
                continue
            candidates.append((task_description, demo))
    if args.limit > 0:
        candidates = candidates[: args.limit]
    if not candidates:
        raise ValueError("no missing, non-excluded demos match this selection")

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    tasks_by_name = {
        suite.get_task(task_id).name: (task_id, suite.get_task(task_id))
        for task_id in range(suite.n_tasks)
    }
    client = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
        execute_horizon=(
            None if args.execute_horizon == 0 else args.execute_horizon
        ),
    )
    writer = LeRobotV2Writer(
        out_dir=args.output_dir,
        template_info_json=str(args.template_info_json),
        modality_json=str(args.modality_json),
        fps=20,
    )
    manifest_path = args.output_dir / "teacher_manifest.jsonl"
    debug_dir = args.output_dir / "_debug_videos"
    if args.save_debug_video:
        debug_dir.mkdir(parents=True, exist_ok=True)

    env_cache: dict[str, tuple] = {}
    hdf5_cache: dict[str, h5py.File] = {}
    attempted = succeeded = 0
    per_task: dict[str, dict[str, int]] = {}

    try:
        for task_description, demo in candidates:
            source_file = demo["source_file"]
            source_demo = demo["demo"]
            task_name = task_from_source_file(source_file)
            if task_name not in tasks_by_name:
                raise KeyError(
                    f"raw task is absent from LIBERO-10 benchmark: {task_name}"
                )
            if task_name not in env_cache:
                task_id, task = tasks_by_name[task_name]
                env, canonical_description = _get_libero_env(
                    task, LIBERO_ENV_RESOLUTION, args.seed
                )
                if canonical_description.strip().lower() != task_description:
                    raise ValueError(
                        f"task mismatch for {source_file}: "
                        f"audit={task_description!r}, "
                        f"benchmark={canonical_description!r}"
                    )
                env_cache[task_name] = (
                    task_id,
                    env,
                    canonical_description,
                )
            if source_file not in hdf5_cache:
                hdf5_cache[source_file] = h5py.File(
                    args.raw_dir / source_file, "r"
                )

            raw_demo = hdf5_cache[source_file]["data"][source_demo]
            task_id, env, canonical_description = env_cache[task_name]
            success, images, wrists, states, actions = rollout_teacher(
                env=env,
                client=client,
                task_description=canonical_description,
                init_state=np.asarray(raw_demo["states"])[0],
                settle_steps=args.settle_steps,
                max_steps=args.max_steps,
            )
            attempted += 1
            task_stats = per_task.setdefault(
                canonical_description, {"attempted": 0, "succeeded": 0}
            )
            task_stats["attempted"] += 1

            output_episode = None
            if success:
                output_episode = writer.add_episode(
                    images_main=images,
                    images_wrist=wrists,
                    states=states,
                    actions=actions,
                    task=canonical_description,
                )
                succeeded += 1
                task_stats["succeeded"] += 1
                if args.save_debug_video:
                    imageio.mimwrite(
                        debug_dir / f"episode_{output_episode:06d}.mp4",
                        images,
                        fps=10,
                        format="FFMPEG",
                        macro_block_size=1,
                    )

            record = {
                "source_file": source_file,
                "source_demo": source_demo,
                "task_id": task_id,
                "task_description": canonical_description,
                "success": success,
                "output_episode_index": output_episode,
                "trajectory_length": len(actions),
            }
            with manifest_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            logging.info(
                "%s:%s %s | %d/%d succeeded",
                source_file,
                source_demo,
                "SUCCESS" if success else "failed",
                succeeded,
                attempted,
            )
    finally:
        for handle in hdf5_cache.values():
            handle.close()
        for _, env, _ in env_cache.values():
            env.close()

    dataset_stats = writer.finalize()
    summary = {
        "audit_manifest": str(args.audit_manifest.resolve()),
        "raw_dir": str(args.raw_dir.resolve()),
        "exclude_manifests": [
            str(path.resolve()) for path in args.exclude_manifest
        ],
        "output_dir": str(args.output_dir.resolve()),
        "task_filter": args.task_filter,
        "candidate_count": len(candidates),
        "skipped_existing_successes": skipped_existing,
        "attempted": attempted,
        "succeeded": succeeded,
        "success_rate": succeeded / attempted if attempted else 0.0,
        "per_task": per_task,
        "dataset_stats": dataset_stats,
        "seed": args.seed,
        "settle_steps": args.settle_steps,
        "max_steps": args.max_steps,
        "teacher_ckpt": args.teacher_ckpt,
        "server_meta": client._server_metadata,
        "uses_fixed_evaluation_init_states": False,
    }
    (args.output_dir / "teacher_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    logging.info("Teacher collection complete: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
