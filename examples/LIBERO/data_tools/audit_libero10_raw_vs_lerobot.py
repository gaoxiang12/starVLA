#!/usr/bin/env python3
"""Audit official LIBERO-10 HDF5 demos against a converted LeRobot dataset.

The official raw files retain ``demo_i`` identifiers, while the LeRobot
conversion renumbers successful episodes globally.  This tool matches episodes
using a hash of the complete action sequence after applying OpenVLA's no-op
filter.  The resulting JSON manifest is suitable as input to a replay/recovery
step because it records the exact missing ``demo_i`` values for every task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


SCENE_PREFIX = re.compile(r"^(?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_")


def is_noop(action: np.ndarray, previous: np.ndarray | None, threshold: float) -> bool:
    """Match the no-op predicate used by OpenVLA's LIBERO regeneration."""
    if np.linalg.norm(action[:-1]) >= threshold:
        return False
    return previous is None or action[-1] == previous[-1]


def remove_noops(actions: np.ndarray, threshold: float) -> np.ndarray:
    kept: list[np.ndarray] = []
    for action in np.asarray(actions):
        previous = kept[-1] if kept else None
        if not is_noop(action, previous, threshold):
            kept.append(np.asarray(action, dtype=np.float32))
    if not kept:
        return np.empty((0, actions.shape[-1]), dtype=np.float32)
    return np.stack(kept)


def action_signature(actions: np.ndarray, decimals: int) -> str:
    canonical = np.round(np.asarray(actions, dtype=np.float32), decimals=decimals)
    payload = canonical.astype("<f4", copy=False).tobytes()
    header = f"{canonical.shape[0]}x{canonical.shape[1]}:".encode()
    return hashlib.blake2b(header + payload, digest_size=20).hexdigest()


def task_from_raw_filename(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    stem = SCENE_PREFIX.sub("", stem)
    return stem.replace("_", " ").lower()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_raw(
    raw_dir: Path,
    threshold: float,
    decimals: int,
    raw_gripper_to_open: bool,
) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    by_task: dict[str, list[dict]] = {}
    signature_collisions: dict[str, list[str]] = defaultdict(list)
    for hdf5_path in sorted(raw_dir.glob("*.hdf5")):
        task = task_from_raw_filename(hdf5_path)
        demos: list[dict] = []
        with h5py.File(hdf5_path, "r") as handle:
            group = handle["data"]
            demo_names = sorted(
                group.keys(), key=lambda value: int(value.rsplit("_", 1)[-1])
            )
            for demo_name in demo_names:
                raw_actions = np.asarray(group[demo_name]["actions"])
                filtered_actions = remove_noops(raw_actions, threshold)
                # The official HDF5 stores the robosuite command convention
                # (-1=open, +1=closed).  The IPEC LeRobot conversion stores
                # ``open_gripper`` (1=open, 0=closed).
                if raw_gripper_to_open and len(filtered_actions):
                    filtered_actions[:, -1] = (
                        1.0 - filtered_actions[:, -1]
                    ) / 2.0
                signature = action_signature(filtered_actions, decimals)
                source = f"{hdf5_path.name}:{demo_name}"
                signature_collisions[signature].append(source)
                demos.append(
                    {
                        "demo": demo_name,
                        "source_file": hdf5_path.name,
                        "raw_length": int(len(raw_actions)),
                        "filtered_length": int(len(filtered_actions)),
                        "action_signature": signature,
                    }
                )
        by_task[task] = demos
    return by_task, {
        sig: sources for sig, sources in signature_collisions.items() if len(sources) > 1
    }


def episode_parquet_path(dataset_dir: Path, episode_index: int, chunk_size: int) -> Path:
    return (
        dataset_dir
        / "data"
        / f"chunk-{episode_index // chunk_size:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def load_lerobot(dataset_dir: Path, decimals: int) -> dict[str, list[dict]]:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunk_size = int(info.get("chunks_size", 1000))
    by_task: dict[str, list[dict]] = defaultdict(list)
    for episode in read_jsonl(dataset_dir / "meta" / "episodes.jsonl"):
        episode_index = int(episode["episode_index"])
        parquet_path = episode_parquet_path(dataset_dir, episode_index, chunk_size)
        frame = pd.read_parquet(parquet_path, columns=["action"])
        actions = np.stack(
            [np.asarray(value, dtype=np.float32) for value in frame["action"]]
        )
        task = episode["tasks"][0].strip().lower()
        by_task[task].append(
            {
                "episode_index": episode_index,
                "length": int(len(actions)),
                "action_signature": action_signature(actions, decimals),
            }
        )
    return dict(by_task)


def audit(raw: dict[str, list[dict]], converted: dict[str, list[dict]]) -> dict:
    report: dict = {"tasks": {}, "summary": {}}
    all_tasks = sorted(set(raw) | set(converted))
    total_raw = total_converted = total_matched = 0

    for task in all_tasks:
        raw_demos = raw.get(task, [])
        converted_episodes = converted.get(task, [])
        raw_by_signature = {
            item["action_signature"]: item for item in raw_demos
        }
        converted_by_signature: dict[str, list[dict]] = defaultdict(list)
        for item in converted_episodes:
            converted_by_signature[item["action_signature"]].append(item)

        matched: list[dict] = []
        missing: list[dict] = []
        for demo in raw_demos:
            candidates = converted_by_signature.get(demo["action_signature"], [])
            if candidates:
                matched.append(
                    {
                        **demo,
                        "episode_indices": [
                            candidate["episode_index"] for candidate in candidates
                        ],
                    }
                )
            else:
                missing.append(demo)

        unmatched_converted = [
            episode
            for episode in converted_episodes
            if episode["action_signature"] not in raw_by_signature
        ]
        report["tasks"][task] = {
            "raw_count": len(raw_demos),
            "converted_count": len(converted_episodes),
            "matched_count": len(matched),
            "matched_demos": matched,
            "missing_count": len(missing),
            "missing_demos": missing,
            "unmatched_converted_count": len(unmatched_converted),
            "unmatched_converted": unmatched_converted,
        }
        total_raw += len(raw_demos)
        total_converted += len(converted_episodes)
        total_matched += len(matched)

    report["summary"] = {
        "raw_count": total_raw,
        "converted_count": total_converted,
        "matched_count": total_matched,
        "missing_raw_count": total_raw - total_matched,
        "unmatched_converted_count": total_converted - total_matched,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--lerobot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--noop-threshold", type=float, default=1e-4)
    parser.add_argument("--signature-decimals", type=int, default=6)
    parser.add_argument(
        "--raw-gripper-to-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map official raw gripper commands from {-1,+1} to open_gripper {1,0}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw, collisions = load_raw(
        args.raw_dir,
        args.noop_threshold,
        args.signature_decimals,
        args.raw_gripper_to_open,
    )
    converted = load_lerobot(args.lerobot_dir, args.signature_decimals)
    report = audit(raw, converted)
    report["raw_signature_collisions"] = collisions
    report["parameters"] = {
        "raw_dir": str(args.raw_dir.resolve()),
        "lerobot_dir": str(args.lerobot_dir.resolve()),
        "noop_threshold": args.noop_threshold,
        "signature_decimals": args.signature_decimals,
        "raw_gripper_to_open": args.raw_gripper_to_open,
    }

    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"Wrote audit manifest: {args.output}")
    print(json.dumps(report["summary"], indent=2))
    for task, task_report in report["tasks"].items():
        print(
            f"{task}: raw={task_report['raw_count']} "
            f"converted={task_report['converted_count']} "
            f"matched={task_report['matched_count']} "
            f"missing={task_report['missing_count']} "
            f"unmatched_converted={task_report['unmatched_converted_count']}"
        )


if __name__ == "__main__":
    main()
