#!/usr/bin/env python3
"""Audit all official LIBERO-90 demos against the IPEC LeRobot conversion.

LIBERO-90 has 90 scene-specific task definitions but fewer unique language
instructions. The LeRobot metadata uses language strings as task IDs, so task
counts alone cannot tell whether all scene tasks are represented. This tool
instead hashes each complete no-op-filtered action sequence and matches it
globally back to ``source_file:demo_i`` in the official HDF5 files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def is_noop(
    action: np.ndarray, previous: np.ndarray | None, threshold: float
) -> bool:
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
    canonical = np.round(
        np.asarray(actions, dtype=np.float32), decimals=decimals
    )
    payload = canonical.astype("<f4", copy=False).tobytes()
    header = f"{canonical.shape[0]}x{canonical.shape[1]}:".encode()
    return hashlib.blake2b(header + payload, digest_size=20).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_language(data_group: h5py.Group, source_file: str) -> str:
    value = data_group.attrs.get("problem_info")
    if isinstance(value, bytes):
        value = value.decode()
    if value:
        try:
            parsed = json.loads(value)
            language = parsed.get("language_instruction")
            if language:
                return str(language).strip().lower()
        except (json.JSONDecodeError, TypeError):
            pass
    stem = source_file.removesuffix("_demo.hdf5")
    parts = stem.split("_", 2)
    return (parts[2] if len(parts) == 3 else stem).replace("_", " ").lower()


def load_raw(
    raw_dir: Path,
    threshold: float,
    decimals: int,
    raw_gripper_to_open: bool,
) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    records: list[dict] = []
    invalid_files: list[dict] = []
    signatures: dict[str, list[str]] = defaultdict(list)
    for hdf5_path in sorted(raw_dir.glob("*.hdf5")):
        try:
            with h5py.File(hdf5_path, "r") as handle:
                group = handle["data"]
                language = task_language(group, hdf5_path.name)
                demo_names = sorted(
                    group.keys(),
                    key=lambda value: int(value.rsplit("_", 1)[-1]),
                )
                for demo_name in demo_names:
                    raw_actions = np.asarray(group[demo_name]["actions"])
                    filtered = remove_noops(raw_actions, threshold)
                    if raw_gripper_to_open and len(filtered):
                        filtered[:, -1] = (1.0 - filtered[:, -1]) / 2.0
                    signature = action_signature(filtered, decimals)
                    source = f"{hdf5_path.name}:{demo_name}"
                    signatures[signature].append(source)
                    records.append(
                        {
                            "source_file": hdf5_path.name,
                            "source_demo": demo_name,
                            "task_language": language,
                            "raw_length": len(raw_actions),
                            "filtered_length": len(filtered),
                            "action_signature": signature,
                        }
                    )
        except (OSError, KeyError, ValueError) as exc:
            invalid_files.append(
                {"source_file": hdf5_path.name, "error": repr(exc)}
            )
    collisions = {
        signature: sources
        for signature, sources in signatures.items()
        if len(sources) > 1
    }
    return records, invalid_files, collisions


def episode_parquet_path(
    dataset_dir: Path, episode_index: int, chunk_size: int
) -> Path:
    return (
        dataset_dir
        / "data"
        / f"chunk-{episode_index // chunk_size:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def load_lerobot(dataset_dir: Path, decimals: int) -> list[dict]:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunk_size = int(info.get("chunks_size", 1000))
    records: list[dict] = []
    for episode in read_jsonl(dataset_dir / "meta" / "episodes.jsonl"):
        episode_index = int(episode["episode_index"])
        parquet_path = episode_parquet_path(
            dataset_dir, episode_index, chunk_size
        )
        frame = pd.read_parquet(parquet_path, columns=["action"])
        actions = np.stack(
            [np.asarray(value, dtype=np.float32) for value in frame["action"]]
        )
        records.append(
            {
                "episode_index": episode_index,
                "task_language": episode["tasks"][0].strip().lower(),
                "length": len(actions),
                "action_signature": action_signature(actions, decimals),
            }
        )
    return records


def build_report(
    raw_records: list[dict],
    converted_records: list[dict],
    invalid_raw_files: list[dict],
    collisions: dict[str, list[str]],
) -> dict:
    converted_by_signature: dict[str, list[dict]] = defaultdict(list)
    for record in converted_records:
        converted_by_signature[record["action_signature"]].append(record)
    raw_signatures = {record["action_signature"] for record in raw_records}

    per_source: dict[str, dict] = {}
    missing_records: list[dict] = []
    matched_count = 0
    for record in raw_records:
        source = record["source_file"]
        source_report = per_source.setdefault(
            source,
            {
                "task_language": record["task_language"],
                "raw_count": 0,
                "matched_count": 0,
                "missing_count": 0,
                "missing_demos": [],
            },
        )
        source_report["raw_count"] += 1
        candidates = converted_by_signature.get(
            record["action_signature"], []
        )
        if candidates:
            matched_count += 1
            source_report["matched_count"] += 1
        else:
            source_report["missing_count"] += 1
            missing = {
                **record,
                "converted_episode_indices": [],
            }
            source_report["missing_demos"].append(missing)
            missing_records.append(missing)

    unmatched_converted = [
        record
        for record in converted_records
        if record["action_signature"] not in raw_signatures
    ]
    represented_sources = sum(
        report["matched_count"] > 0 for report in per_source.values()
    )
    return {
        "summary": {
            "raw_file_count": len(per_source) + len(invalid_raw_files),
            "valid_raw_file_count": len(per_source),
            "invalid_raw_file_count": len(invalid_raw_files),
            "raw_demo_count": len(raw_records),
            "converted_episode_count": len(converted_records),
            "matched_raw_demo_count": matched_count,
            "missing_raw_demo_count": len(raw_records) - matched_count,
            "unmatched_converted_episode_count": len(unmatched_converted),
            "represented_source_task_count": represented_sources,
            "unrepresented_source_task_count": len(per_source)
            - represented_sources,
            "unique_raw_language_count": len(
                {record["task_language"] for record in raw_records}
            ),
            "unique_converted_language_count": len(
                {
                    record["task_language"]
                    for record in converted_records
                }
            ),
        },
        "per_source_file": dict(sorted(per_source.items())),
        "missing_demos": missing_records,
        "unmatched_converted_episodes": unmatched_converted,
        "invalid_raw_files": invalid_raw_files,
        "raw_signature_collisions": collisions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--lerobot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noop-threshold", type=float, default=1e-4)
    parser.add_argument("--signature-decimals", type=int, default=6)
    parser.add_argument(
        "--raw-gripper-to-open",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw, invalid, collisions = load_raw(
        args.raw_dir,
        args.noop_threshold,
        args.signature_decimals,
        args.raw_gripper_to_open,
    )
    converted = load_lerobot(args.lerobot_dir, args.signature_decimals)
    report = build_report(raw, converted, invalid, collisions)
    report["parameters"] = {
        "raw_dir": str(args.raw_dir.resolve()),
        "lerobot_dir": str(args.lerobot_dir.resolve()),
        "noop_threshold": args.noop_threshold,
        "signature_decimals": args.signature_decimals,
        "raw_gripper_to_open": args.raw_gripper_to_open,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    for source, source_report in report["per_source_file"].items():
        if source_report["missing_count"]:
            print(
                f"{source}: matched={source_report['matched_count']}/"
                f"{source_report['raw_count']}, "
                f"missing={source_report['missing_count']}"
            )
    if invalid:
        print("Invalid raw files:")
        for item in invalid:
            print(f"  {item['source_file']}: {item['error']}")


if __name__ == "__main__":
    main()
