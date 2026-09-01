#!/usr/bin/env python3
"""Build a complete, deduplicated SO-family manifest from community data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from examples.UnifiedPretrain.data_tools.community_so100 import (
    SO100_EXCLUDED_NAME_TOKENS,
    classify_so_family_metadata,
    select_so100_video_keys,
    validate_so_family_dataset_metadata,
)


def _tasks(dataset_path: Path) -> list[str]:
    table = pq.read_table(dataset_path / "meta/tasks.parquet")
    return [str(task) for task in table.column("task").to_pylist()]


def _episode_records(dataset_path: Path, info: dict) -> list[dict] | None:
    episodes = []
    for path in sorted((dataset_path / "meta/episodes").glob("*/*.parquet")):
        episodes.extend(pq.read_table(path).to_pylist())
    if len(episodes) != int(info.get("total_episodes", -1)):
        return None
    return episodes


def _episode_file_path(dataset_path: Path, info: dict, episode: dict) -> Path:
    data_chunk = int(episode["data/chunk_index"])
    data_file = int(episode["data/file_index"])
    return dataset_path / info["data_path"].format(
        chunk_index=data_chunk,
        file_index=data_file,
        episode_chunk=data_chunk,
        episode_index=int(episode["episode_index"]),
    )


def _complete_episode_records(
    dataset_path: Path,
    info: dict,
    video_keys: tuple[str, ...],
) -> list[dict] | None:
    episodes = _episode_records(dataset_path, info)
    if episodes is None:
        return None

    for episode in episodes:
        data_path = _episode_file_path(dataset_path, info, episode)
        if not data_path.is_file() or data_path.stat().st_size == 0:
            return None

        data_chunk = int(episode["data/chunk_index"])
        data_file = int(episode["data/file_index"])
        episode_index = int(episode["episode_index"])
        for video_key in video_keys:
            video_chunk = int(
                episode.get(f"videos/{video_key}/chunk_index", data_chunk)
            )
            video_file = int(
                episode.get(f"videos/{video_key}/file_index", data_file)
            )
            video_path = dataset_path / info["video_path"].format(
                video_key=video_key,
                chunk_index=video_chunk,
                file_index=video_file,
                episode_chunk=video_chunk,
                episode_index=episode_index,
            )
            if not video_path.is_file() or video_path.stat().st_size == 0:
                return None
    return episodes


def _trajectory_fingerprints(
    dataset_path: Path, info: dict, episodes: list[dict]
) -> tuple[dict[int, str], set[int]]:
    """Hash rounded action/state trajectories, reading each parquet file once."""

    episodes_by_file: dict[Path, list[dict]] = {}
    for episode in episodes:
        episodes_by_file.setdefault(
            _episode_file_path(dataset_path, info, episode), []
        ).append(episode)

    fingerprints = {}
    invalid_episode_indices = set()
    for data_path, file_episodes in episodes_by_file.items():
        table = pq.read_table(
            data_path,
            columns=["episode_index", "action", "observation.state"],
        )
        episode_indices = np.asarray(table.column("episode_index").to_numpy())
        action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        state = np.asarray(
            table.column("observation.state").to_pylist(), dtype=np.float32
        )
        for episode in file_episodes:
            episode_index = int(episode["episode_index"])
            mask = episode_indices == episode_index
            if int(mask.sum()) != int(episode["length"]):
                invalid_episode_indices.add(episode_index)
                continue
            digest = hashlib.blake2b(digest_size=20)
            digest.update(np.asarray([int(mask.sum())], dtype="<i8").tobytes())
            for values in (action[mask], state[mask]):
                canonical = np.nan_to_num(
                    np.round(values, decimals=4),
                    nan=0.0,
                    posinf=np.finfo(np.float32).max,
                    neginf=np.finfo(np.float32).min,
                ).astype("<f4", copy=False)
                digest.update(canonical.tobytes())
            fingerprints[episode_index] = digest.hexdigest()
    return fingerprints, invalid_episode_indices


def _name_suspicion(path: str) -> tuple[int, str]:
    lower = path.lower()
    return sum(token in lower for token in SO100_EXCLUDED_NAME_TOKENS), path


def build_manifest(
    community_root: Path, *, source_revision: str | None = None
) -> tuple[dict, Counter]:
    community_root = community_root.resolve()
    candidates = []
    excluded = Counter()
    for info_path in sorted(community_root.glob("*/*/meta/info.json")):
        dataset_path = info_path.parent.parent
        try:
            info = json.loads(info_path.read_text())
            tasks = _tasks(dataset_path)
        except Exception:
            excluded["unreadable_metadata"] += 1
            continue

        reason = validate_so_family_dataset_metadata(info, tasks)
        if reason is not None:
            excluded[reason] += 1
            continue
        video_keys = select_so100_video_keys(info)
        assert video_keys is not None
        episodes = _complete_episode_records(dataset_path, info, video_keys)
        if episodes is None:
            excluded["incomplete_local_files"] += 1
            continue
        data_config = classify_so_family_metadata(info)
        assert data_config is not None
        relative_path = dataset_path.relative_to(community_root).as_posix()
        candidates.append(
            {
                "dataset_path": dataset_path,
                "relative_path": relative_path,
                "info": info,
                "episodes_meta": episodes,
                "data_config": data_config,
                "video_keys": video_keys,
            }
        )

    # Prefer cleanly named roots when exact trajectory copies also appear under
    # test/debug/trial version directories. Fingerprints are scoped by action
    # calibration family, preventing cross-calibration false matches.
    candidates.sort(key=lambda item: _name_suspicion(item["relative_path"]))
    seen_fingerprints: dict[tuple[str, str], tuple[str, int]] = {}
    datasets = []
    duplicate_episodes = 0
    duplicate_frames = 0
    corrupt_episodes = 0
    corrupt_frames = 0
    for candidate in candidates:
        fingerprints, invalid_episode_indices = _trajectory_fingerprints(
            candidate["dataset_path"], candidate["info"], candidate["episodes_meta"]
        )
        excluded_episode_indices = []
        included_episodes = 0
        included_frames = 0
        for episode in candidate["episodes_meta"]:
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            if episode_index in invalid_episode_indices:
                excluded_episode_indices.append(episode_index)
                corrupt_episodes += 1
                corrupt_frames += length
                continue
            key = (candidate["data_config"], fingerprints[episode_index])
            if key in seen_fingerprints:
                excluded_episode_indices.append(episode_index)
                duplicate_episodes += 1
                duplicate_frames += length
                continue
            seen_fingerprints[key] = (candidate["relative_path"], episode_index)
            included_episodes += 1
            included_frames += length

        if included_episodes == 0:
            excluded["fully_duplicate_dataset"] += 1
            continue
        entry = {
            "path": candidate["relative_path"],
            "data_config": candidate["data_config"],
            "robot_type": candidate["info"]["robot_type"],
            "episodes": included_episodes,
            "frames": included_frames,
            "video_keys": list(candidate["video_keys"]),
        }
        if excluded_episode_indices:
            entry["excluded_episode_indices"] = excluded_episode_indices
        datasets.append(entry)

    episodes_by_data_config = Counter()
    datasets_by_data_config = Counter()
    for item in datasets:
        episodes_by_data_config[item["data_config"]] += item["episodes"]
        datasets_by_data_config[item["data_config"]] += 1

    manifest = {
        "format_version": 1,
        "source_root": community_root.name,
        "source_revision": source_revision,
        "selection": {
            "robot_types": sorted(
                {
                    "so100",
                    "so100-blue",
                    "so100-red",
                    "so101",
                    "so100_follower",
                    "so101_follower",
                }
            ),
            "codebase_version": "v3.0",
            "fps": 30,
            "action_shape": [6],
            "state_shape": [6],
            "minimum_usable_views": 1,
            "require_complete_local_files": True,
            "trajectory_deduplication": "rounded_action_state_blake2b",
        },
        "datasets": sorted(datasets, key=lambda item: item["path"]),
        "summary": {
            "datasets": len(datasets),
            "episodes": sum(item["episodes"] for item in datasets),
            "frames": sum(item["frames"] for item in datasets),
            "datasets_by_data_config": dict(sorted(datasets_by_data_config.items())),
            "episodes_by_data_config": dict(sorted(episodes_by_data_config.items())),
            "duplicate_episodes_excluded": duplicate_episodes,
            "duplicate_frames_excluded": duplicate_frames,
            "corrupt_episodes_excluded": corrupt_episodes,
            "corrupt_frames_excluded": corrupt_frames,
            "excluded_datasets_by_reason": dict(sorted(excluded.items())),
        },
    }
    return manifest, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("community_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to <community_root>/community_so_family_manifest.json",
    )
    parser.add_argument(
        "--source-revision",
        help="Optional source snapshot SHA recorded in the manifest.",
    )
    args = parser.parse_args()
    output = args.output or args.community_root / "community_so_family_manifest.json"
    manifest, _ = build_manifest(
        args.community_root, source_revision=args.source_revision
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(manifest["summary"], indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
