#!/usr/bin/env python3
"""Build a deterministic Bridge task taxonomy and a reviewable CSV catalog."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

from starVLA.task_language import classify_bridge_task


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def task_id(family: str, canonical_text: str, status: str) -> str:
    payload = f"{status}\0{canonical_text}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return f"bridge.{family}.{digest}"


def video_key_counts(info: dict, training_video_keys: list[str]) -> tuple[int, int]:
    available = {
        key.removeprefix("observation.images.")
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    }
    requested = {key.removeprefix("video.") for key in training_video_keys}
    missing = requested - available
    if missing:
        raise ValueError(f"Training video keys absent from metadata: {sorted(missing)}")
    return len(available), len(requested)


def build_catalog(
    dataset_dir: Path,
    training_video_keys: list[str],
    min_episodes: int = 20,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    if min_episodes < 1:
        raise ValueError("min_episodes must be at least 1")

    meta_dir = dataset_dir / "meta"
    task_rows = list(read_jsonl(meta_dir / "tasks.jsonl"))
    episodes = list(read_jsonl(meta_dir / "episodes.jsonl"))
    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    all_camera_count, training_camera_count = video_key_counts(
        info, training_video_keys
    )

    episode_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    for episode in episodes:
        for raw_task in episode.get("tasks", []):
            raw = str(raw_task or "")
            episode_counts[raw] += 1
            frame_counts[raw] += int(episode.get("length", 0))

    mappings = []
    grouped: dict[str, dict] = defaultdict(
        lambda: {
            "task_indices": [],
            "raw_descriptions": [],
            "episode_count": 0,
            "frame_count": 0,
        }
    )
    for row in task_rows:
        raw = str(row.get("task") or "")
        label = classify_bridge_task(raw)
        identifier = task_id(label.family, label.canonical_text, label.status)
        mapping = {
            "task_index": int(row["task_index"]),
            "task_id": identifier,
            "raw_description": raw,
            "canonical_text": label.canonical_text,
            "family": label.family,
            "status": label.status,
            "confidence": label.confidence,
            "episode_count": episode_counts[raw],
            "frame_count": frame_counts[raw],
            "all_camera_video_count": episode_counts[raw] * all_camera_count,
            "training_camera_video_count": episode_counts[raw]
            * training_camera_count,
        }
        mappings.append(mapping)
        group = grouped[identifier]
        group.update(
            {
                "task_id": identifier,
                "canonical_text": label.canonical_text,
                "family": label.family,
                "status": label.status,
                "confidence": label.confidence,
            }
        )
        group["task_indices"].append(int(row["task_index"]))
        group["raw_descriptions"].append(raw)
        group["episode_count"] += episode_counts[raw]
        group["frame_count"] += frame_counts[raw]

    catalog = []
    for group in grouped.values():
        include_for_training = bool(group["canonical_text"]) and (
            group["episode_count"] >= min_episodes
        )
        if not group["canonical_text"]:
            exclusion_reason = "empty_task_description"
        elif group["episode_count"] < min_episodes:
            exclusion_reason = "below_min_episodes"
        else:
            exclusion_reason = ""
        group["include_for_training"] = include_for_training
        group["exclusion_reason"] = exclusion_reason
        group["raw_description_count"] = len(group["raw_descriptions"])
        group["all_camera_video_count"] = group["episode_count"] * all_camera_count
        group["training_camera_video_count"] = (
            group["episode_count"] * training_camera_count
        )
        catalog.append(group)
    catalog.sort(
        key=lambda row: (
            row["status"] != "unlabeled",
            -row["episode_count"],
            row["canonical_text"],
        )
    )

    groups_by_id = {row["task_id"]: row for row in catalog}
    task_ids_by_raw_description = {}
    for row in mappings:
        group = groups_by_id[row["task_id"]]
        row["taxonomy_episode_count"] = group["episode_count"]
        row["include_for_training"] = group["include_for_training"]
        row["exclusion_reason"] = group["exclusion_reason"]
        task_ids_by_raw_description[row["raw_description"]] = row["task_id"]

    video_health_blacklist = {}
    video_health_path = meta_dir / "video_health" / "bad_episodes.jsonl"
    if video_health_path.exists():
        for row in read_jsonl(video_health_path):
            video_health_blacklist[int(row["episode_index"])] = row

    excluded_episodes = []
    training_episodes = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        raw_tasks = [str(task or "") for task in episode.get("tasks", [])]
        task_ids = [
            task_ids_by_raw_description.get(raw_task) for raw_task in raw_tasks
        ]
        task_groups = [
            groups_by_id[identifier]
            for identifier in task_ids
            if identifier is not None
        ]
        reasons = {
            group["exclusion_reason"]
            for group in task_groups
            if not group["include_for_training"]
        }
        if not raw_tasks or len(task_groups) != len(raw_tasks):
            reasons.add("missing_task_metadata")
        if episode_index in video_health_blacklist:
            reasons.add("video_health")
        if reasons:
            excluded_episodes.append(
                {
                    "episode_index": episode_index,
                    "reasons": sorted(reasons),
                    "tasks": raw_tasks,
                    "task_ids": [identifier for identifier in task_ids if identifier],
                }
            )
        else:
            training_episodes.append(episode)

    status_task_counts = Counter(row["status"] for row in catalog)
    status_episode_counts = {
        status: sum(
            row["episode_count"] for row in catalog if row["status"] == status
        )
        for status in status_task_counts
    }
    training_catalog = [row for row in catalog if row["include_for_training"]]
    excluded_empty_catalog = [
        row for row in catalog if row["exclusion_reason"] == "empty_task_description"
    ]
    excluded_small_catalog = [
        row for row in catalog if row["exclusion_reason"] == "below_min_episodes"
    ]
    summary = {
        "min_episodes_per_task": min_episodes,
        "source_task_descriptions": len(task_rows),
        "taxonomy_tasks": len(catalog),
        "collapsed_descriptions": len(task_rows) - len(catalog),
        "episodes": len(episodes),
        "episode_references": sum(row["episode_count"] for row in catalog),
        "frames": sum(row["frame_count"] for row in catalog),
        "available_cameras": all_camera_count,
        "training_cameras": training_camera_count,
        "all_camera_video_count": sum(
            row["all_camera_video_count"] for row in catalog
        ),
        "training_camera_video_count": sum(
            row["training_camera_video_count"] for row in catalog
        ),
        "status_task_counts": dict(status_task_counts),
        "status_episode_counts": status_episode_counts,
        "training_task_count": len(training_catalog),
        "training_episode_count": len(training_episodes),
        "training_frame_count": sum(
            int(episode.get("length", 0)) for episode in training_episodes
        ),
        "training_all_camera_video_count": len(training_episodes)
        * all_camera_count,
        "training_selected_camera_video_count": len(training_episodes)
        * training_camera_count,
        "excluded_empty_task_count": len(excluded_empty_catalog),
        "excluded_empty_episode_references": sum(
            row["episode_count"] for row in excluded_empty_catalog
        ),
        "excluded_below_min_task_count": len(excluded_small_catalog),
        "excluded_below_min_episode_references": sum(
            row["episode_count"] for row in excluded_small_catalog
        ),
        "excluded_episode_count": len(excluded_episodes),
        "video_health_blacklist_episode_count": len(video_health_blacklist),
    }
    return mappings, catalog, excluded_episodes, summary


def write_outputs(
    dataset_dir: Path,
    output_dir: Path,
    training_video_keys: list[str],
    min_episodes: int = 20,
) -> dict:
    mappings, catalog, excluded_episodes, summary = build_catalog(
        dataset_dir, training_video_keys, min_episodes
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = output_dir / "bridge_task_taxonomy.jsonl"
    with mapping_path.open("w", encoding="utf-8") as handle:
        for row in mappings:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    catalog_path = output_dir / "bridge_task_catalog.csv"
    columns = [
        "task_id",
        "canonical_text",
        "family",
        "status",
        "confidence",
        "include_for_training",
        "exclusion_reason",
        "raw_description_count",
        "episode_count",
        "frame_count",
        "all_camera_video_count",
        "training_camera_video_count",
        "task_indices",
        "raw_descriptions",
    ]
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in catalog:
            serialized = dict(row)
            serialized["task_indices"] = json.dumps(row["task_indices"])
            serialized["raw_descriptions"] = json.dumps(
                row["raw_descriptions"], ensure_ascii=False
            )
            writer.writerow({key: serialized[key] for key in columns})

    blacklist_path = output_dir / "bridge_pretrain_excluded_episodes.jsonl"
    with blacklist_path.open("w", encoding="utf-8") as handle:
        for row in excluded_episodes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_dir / "bridge_task_taxonomy_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--training-video-keys",
        nargs="+",
        default=["image_0", "image_1", "image_2"],
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=20,
        help="Exclude canonical tasks with fewer episodes (default: 20)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.dataset_dir / "meta" / "task_language"
    summary = write_outputs(
        args.dataset_dir,
        output_dir,
        args.training_video_keys,
        args.min_episodes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote Bridge taxonomy and catalog to {output_dir}")


if __name__ == "__main__":
    main()
