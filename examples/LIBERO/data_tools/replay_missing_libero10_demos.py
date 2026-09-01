#!/usr/bin/env python3
"""Replay missing official LIBERO-10 demos into a separate LeRobot dataset.

Input is the manifest produced by ``audit_libero10_raw_vs_lerobot.py``.  Only
raw demos absent from the existing converted dataset are attempted.  A demo is
written when the LIBERO success predicate is true after replaying its complete
no-op-filtered action sequence.

The output deliberately remains a separate dataset.  This preserves provenance
and lets training assign recovered demos an explicit mixture weight.
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
from libero.libero import benchmark


def is_noop(action: np.ndarray, previous: np.ndarray | None, threshold: float) -> bool:
    if np.linalg.norm(action[:-1]) >= threshold:
        return False
    return previous is None or action[-1] == previous[-1]


def task_from_source_file(source_file: str) -> str:
    return source_file.removesuffix("_demo.hdf5")


def replay_demo(
    env,
    raw_states: np.ndarray,
    raw_actions: np.ndarray,
    noop_threshold: float,
    settle_steps: int,
) -> tuple[bool, list, list, list, list, dict]:
    env.reset()
    obs = env.set_init_state(np.asarray(raw_states[0]))
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    images_main: list[np.ndarray] = []
    images_wrist: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    previous_kept: np.ndarray | None = None
    ever_succeeded = False
    final_success = False

    for raw_action in np.asarray(raw_actions):
        raw_action = np.asarray(raw_action, dtype=np.float32)
        if is_noop(raw_action, previous_kept, noop_threshold):
            continue
        previous_kept = raw_action

        images_main.append(
            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        )
        images_wrist.append(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        )
        states.append(
            np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ).astype(np.float32)
        )
        output_action = raw_action.copy()
        output_action[-1] = (1.0 - output_action[-1]) / 2.0
        actions.append(output_action)

        obs, _, done, _ = env.step(raw_action.tolist())
        final_success = bool(done)
        ever_succeeded = ever_succeeded or final_success

    diagnostics = {
        "raw_length": int(len(raw_actions)),
        "filtered_length": int(len(actions)),
        "ever_succeeded": ever_succeeded,
        "final_success": final_success,
    }
    return (
        final_success,
        images_main,
        images_wrist,
        states,
        actions,
        diagnostics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--noop-threshold", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--task-filter", type=str, default="")
    parser.add_argument(
        "--selection",
        choices=("missing", "matched"),
        default="missing",
        help="Replay missing demos for recovery, or matched demos as a control.",
    )
    parser.add_argument("--save-debug-video", action="store_true")
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
            "Use a new directory so provenance is not mixed."
        )

    audit = json.loads(args.audit_manifest.read_text())
    missing: list[tuple[str, dict]] = []
    needle = args.task_filter.strip().lower()
    for task_description, task_report in audit["tasks"].items():
        if needle and needle not in task_description:
            continue
        demos_key = (
            "missing_demos" if args.selection == "missing" else "matched_demos"
        )
        for demo in task_report[demos_key]:
            missing.append((task_description, demo))
    if args.limit > 0:
        missing = missing[: args.limit]
    if not missing:
        raise ValueError(
            f"audit manifest contains no {args.selection} demos for this selection"
        )

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    tasks_by_name = {
        suite.get_task(task_id).name: (task_id, suite.get_task(task_id))
        for task_id in range(suite.n_tasks)
    }

    writer = LeRobotV2Writer(
        out_dir=args.output_dir,
        template_info_json=str(args.template_info_json),
        modality_json=str(args.modality_json),
        fps=20,
    )
    manifest_path = args.output_dir / "replay_manifest.jsonl"
    debug_dir = args.output_dir / "_debug_videos"
    if args.save_debug_video:
        debug_dir.mkdir(parents=True, exist_ok=True)

    env_cache: dict[str, tuple] = {}
    hdf5_cache: dict[str, h5py.File] = {}
    attempted = recovered = 0
    per_task: dict[str, dict[str, int]] = {}

    try:
        for task_description, demo in missing:
            source_file = demo["source_file"]
            task_name = task_from_source_file(source_file)
            if task_name not in tasks_by_name:
                raise KeyError(f"raw task is absent from LIBERO-10 benchmark: {task_name}")

            if task_name not in env_cache:
                task_id, task = tasks_by_name[task_name]
                env, canonical_description = _get_libero_env(
                    task, LIBERO_ENV_RESOLUTION, args.seed
                )
                if canonical_description.strip().lower() != task_description:
                    raise ValueError(
                        f"task mismatch for {source_file}: audit={task_description!r}, "
                        f"benchmark={canonical_description!r}"
                    )
                env_cache[task_name] = (task_id, env, canonical_description)

            if source_file not in hdf5_cache:
                hdf5_cache[source_file] = h5py.File(
                    args.raw_dir / source_file, "r"
                )
            raw_demo = hdf5_cache[source_file]["data"][demo["demo"]]
            task_id, env, canonical_description = env_cache[task_name]
            success, imgs, wrists, states, actions, diagnostics = replay_demo(
                env=env,
                raw_states=np.asarray(raw_demo["states"]),
                raw_actions=np.asarray(raw_demo["actions"]),
                noop_threshold=args.noop_threshold,
                settle_steps=args.settle_steps,
            )
            attempted += 1
            task_stats = per_task.setdefault(
                canonical_description, {"attempted": 0, "recovered": 0}
            )
            task_stats["attempted"] += 1

            output_episode = None
            if success:
                output_episode = writer.add_episode(
                    images_main=imgs,
                    images_wrist=wrists,
                    states=states,
                    actions=actions,
                    task=canonical_description,
                )
                recovered += 1
                task_stats["recovered"] += 1
                if args.save_debug_video:
                    imageio.mimwrite(
                        debug_dir / f"episode_{output_episode:06d}.mp4",
                        imgs,
                        fps=10,
                        format="FFMPEG",
                        macro_block_size=1,
                    )

            record = {
                "source_file": source_file,
                "source_demo": demo["demo"],
                "task_id": task_id,
                "task_description": canonical_description,
                "success": success,
                "output_episode_index": output_episode,
                **diagnostics,
            }
            with manifest_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            logging.info(
                "%s:%s %s (%d/%d recovered)",
                source_file,
                demo["demo"],
                "RECOVERED" if success else "failed",
                recovered,
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
        "output_dir": str(args.output_dir.resolve()),
        "attempted": attempted,
        "recovered": recovered,
        "recovery_rate": recovered / attempted if attempted else 0.0,
        "per_task": per_task,
        "dataset_stats": dataset_stats,
        "seed": args.seed,
        "settle_steps": args.settle_steps,
        "noop_threshold": args.noop_threshold,
        "selection": args.selection,
    }
    (args.output_dir / "replay_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    logging.info("Replay complete: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
