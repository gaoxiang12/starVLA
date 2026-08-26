#!/usr/bin/env python3
"""Audit and prepare the success-filtered KUKA LeRobot export for pretraining."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Iterable

import av
import numpy as np
import pyarrow.parquet as pq


LOW_DIMENSIONAL_KEYS = ("observation.state", "action")
EXPECTED_STATE_DIM = 8
EXPECTED_ACTION_DIM = 7
EXPECTED_FPS = 10
EXPECTED_VIDEO_SHAPE = (640, 512)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def _data_path(dataset_dir: Path, info: dict, episode_index: int) -> Path:
    return dataset_dir / str(info["data_path"]).format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
    )


def _video_path(dataset_dir: Path, info: dict, episode_index: int) -> Path:
    return dataset_dir / str(info["video_path"]).format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
        video_key="observation.images.image",
    )


def _fixed_list_numpy(table, key: str, width: int) -> np.ndarray:
    column = table.column(key).combine_chunks()
    values = np.asarray(column.values.to_numpy(zero_copy_only=False))
    return values.astype(np.float32, copy=False).reshape(table.num_rows, width)


def _scan_parquet(job):
    dataset_dir, info, episode, expected_start = job
    episode_index = int(episode["episode_index"])
    expected_length = int(episode["length"])
    path = _data_path(dataset_dir, info, episode_index)
    errors = []
    try:
        table = pq.read_table(
            path,
            columns=[
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ],
        )
        length = table.num_rows
        if length != expected_length:
            errors.append(f"length:{length}!={expected_length}")
        state = _fixed_list_numpy(table, "observation.state", EXPECTED_STATE_DIM)
        action = _fixed_list_numpy(table, "action", EXPECTED_ACTION_DIM)
        timestamp = np.asarray(table.column("timestamp").to_numpy())
        frame_index = np.asarray(table.column("frame_index").to_numpy())
        episode_ids = np.asarray(table.column("episode_index").to_numpy())
        global_index = np.asarray(table.column("index").to_numpy())
        task_index = np.asarray(table.column("task_index").to_numpy())
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            errors.append("nonfinite_state_or_action")
        if not np.isfinite(timestamp).all():
            errors.append("nonfinite_timestamp")
        if not np.array_equal(frame_index, np.arange(length)):
            errors.append("invalid_frame_index")
        if not np.all(episode_ids == episode_index):
            errors.append("invalid_episode_index")
        if not np.array_equal(
            global_index, np.arange(expected_start, expected_start + length)
        ):
            errors.append("invalid_global_index")
        if not np.all(task_index == 0):
            errors.append("invalid_task_index")
        if not np.allclose(timestamp, np.arange(length) / EXPECTED_FPS, atol=2e-6):
            errors.append("invalid_timestamp")
        digest = hashlib.blake2b(digest_size=20)
        digest.update(np.ascontiguousarray(state).tobytes())
        digest.update(np.ascontiguousarray(action).tobytes())
        return episode_index, errors, digest.hexdigest(), state, action
    except Exception as error:
        return (
            episode_index,
            [f"unreadable_parquet:{type(error).__name__}:{error}"],
            None,
            None,
            None,
        )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_video(job):
    dataset_dir, info, episode, decode_full = job
    episode_index = int(episode["episode_index"])
    expected_length = int(episode["length"])
    path = _video_path(dataset_dir, info, episode_index)
    errors = []
    try:
        with av.open(str(path), mode="r") as container:
            streams = container.streams.video
            if len(streams) != 1:
                return episode_index, [f"video_stream_count:{len(streams)}"]
            stream = streams[0]
            if (int(stream.width), int(stream.height)) != EXPECTED_VIDEO_SHAPE:
                errors.append(
                    f"video_shape:{stream.width}x{stream.height}"
                )
            if stream.average_rate is None or not np.isclose(
                float(stream.average_rate), EXPECTED_FPS
            ):
                errors.append(f"video_fps:{stream.average_rate}")
            if int(stream.frames) != expected_length:
                errors.append(f"container_frames:{stream.frames}!={expected_length}")
            if decode_full:
                decoded = 0
                for frame in container.decode(stream):
                    decoded += 1
                    if (frame.width, frame.height) != EXPECTED_VIDEO_SHAPE:
                        errors.append(
                            f"decoded_shape:{frame.width}x{frame.height}"
                        )
                        break
                if decoded != expected_length:
                    errors.append(f"decoded_frames:{decoded}!={expected_length}")
    except Exception as error:
        errors.append(f"unreadable_video:{type(error).__name__}:{error}")
    return episode_index, errors


def _statistics(values: np.ndarray) -> dict:
    return {
        "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
        "std": np.std(values, axis=0, dtype=np.float64).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def prepare(
    dataset_dir: Path,
    *,
    workers: int,
    decode_sample_episodes: int,
    source_revision: str | None,
) -> tuple[list[dict], dict, dict]:
    info = json.loads((dataset_dir / "meta/info.json").read_text())
    episodes = list(read_jsonl(dataset_dir / "meta/episodes.jsonl"))
    tasks = list(read_jsonl(dataset_dir / "meta/tasks.jsonl"))
    expected_episode_ids = list(range(len(episodes)))
    actual_episode_ids = [int(row["episode_index"]) for row in episodes]
    if actual_episode_ids != expected_episode_ids:
        raise ValueError("KUKA episode indices are not contiguous from zero")
    lengths = np.asarray([int(row["length"]) for row in episodes], dtype=np.int64)
    if int(lengths.sum()) != int(info["total_frames"]):
        raise ValueError("KUKA episode lengths do not sum to info.total_frames")
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError("KUKA episode count does not match info.total_episodes")
    if tasks != [{"task_index": 0, "task": "pick anything"}]:
        raise ValueError(f"Unexpected KUKA task table: {tasks!r}")

    starts = np.cumsum(np.concatenate(([0], lengths[:-1])))
    jobs = (
        (dataset_dir, info, episode, int(starts[index]))
        for index, episode in enumerate(episodes)
    )
    fingerprints = defaultdict(list)
    arrays = [None] * len(episodes)
    parquet_failures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, result in enumerate(
            executor.map(_scan_parquet, jobs), start=1
        ):
            episode_index, errors, fingerprint, state, action = result
            if errors:
                parquet_failures[episode_index] = errors
            if fingerprint is not None:
                fingerprints[fingerprint].append(episode_index)
                arrays[episode_index] = (state, action)
            if completed % 25000 == 0:
                print(f"parquet_progress={completed}/{len(episodes)}", flush=True)

    duplicate_of = {}
    low_dimensional_collisions = []
    for episode_ids in fingerprints.values():
        if len(episode_ids) < 2:
            continue
        videos_by_digest = defaultdict(list)
        for episode_index in episode_ids:
            digest = _file_digest(_video_path(dataset_dir, info, episode_index))
            videos_by_digest[digest].append(episode_index)
        for video_ids in videos_by_digest.values():
            if len(video_ids) < 2:
                continue
            keep = min(video_ids)
            for episode_index in sorted(video_ids):
                if episode_index != keep:
                    duplicate_of[episode_index] = keep
        if len(videos_by_digest) > 1:
            low_dimensional_collisions.append(sorted(episode_ids))

    sample_count = min(max(0, decode_sample_episodes), len(episodes))
    decoded_ids = set(
        np.linspace(0, len(episodes) - 1, num=sample_count, dtype=np.int64).tolist()
    )
    video_jobs = (
        (dataset_dir, info, episode, index in decoded_ids)
        for index, episode in enumerate(episodes)
    )
    video_failures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, (episode_index, errors) in enumerate(
            executor.map(_probe_video, video_jobs), start=1
        ):
            if errors:
                video_failures[episode_index] = errors
            if completed % 25000 == 0:
                print(f"video_progress={completed}/{len(episodes)}", flush=True)

    task_episode_counts = Counter(
        str(task).strip() for row in episodes for task in row.get("tasks", [])
    )
    language_failures = {}
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        descriptions = [str(task).strip() for task in episode.get("tasks", [])]
        if not descriptions or not any(descriptions):
            language_failures[episode_index] = ["empty_task_language"]
        elif any(task_episode_counts[text] < 20 for text in descriptions):
            language_failures[episode_index] = ["rare_task_language"]

    reasons = defaultdict(list)
    for episode_index, errors in parquet_failures.items():
        reasons[episode_index].extend(errors)
    for episode_index, keep in duplicate_of.items():
        reasons[episode_index].append(f"exact_duplicate_of:{keep}")
    for episode_index, errors in video_failures.items():
        reasons[episode_index].extend(errors)
    for episode_index, errors in language_failures.items():
        reasons[episode_index].extend(errors)

    excluded_ids = set(reasons)
    retained_frames = int(
        sum(
            int(episode["length"])
            for episode in episodes
            if int(episode["episode_index"]) not in excluded_ids
        )
    )
    state_values = np.empty((retained_frames, EXPECTED_STATE_DIM), dtype=np.float32)
    action_values = np.empty(
        (retained_frames, EXPECTED_ACTION_DIM), dtype=np.float32
    )
    cursor = 0
    for episode_index, episode in enumerate(episodes):
        if episode_index in excluded_ids:
            continue
        pair = arrays[episode_index]
        if pair is None:
            raise RuntimeError(f"Missing retained arrays for episode {episode_index}")
        state, action = pair
        length = len(state)
        state_values[cursor : cursor + length] = state
        action_values[cursor : cursor + length] = action
        cursor += length
        arrays[episode_index] = None
    if cursor != retained_frames:
        raise RuntimeError(f"Stats fill mismatch: {cursor} != {retained_frames}")

    blacklist = []
    for episode_index in sorted(excluded_ids):
        episode = episodes[episode_index]
        blacklist.append(
            {
                "episode_index": episode_index,
                "length": int(episode["length"]),
                "tasks": episode.get("tasks", []),
                "reasons": reasons[episode_index],
            }
        )
    blacklist_jsonl = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in blacklist
    )
    blacklist_sha256 = hashlib.sha256(blacklist_jsonl.encode()).hexdigest()
    stats = {
        "__format_version": 2,
        "__cache_config": {"mode": "abs"},
        "statistics": {
            "observation.state": _statistics(state_values),
            "action": _statistics(action_values),
        },
        "__provenance": {
            "source_revision": source_revision,
            "method": "exact_all_retained_frames",
            "retained_episodes": len(episodes) - len(excluded_ids),
            "retained_frames": retained_frames,
            "episode_blacklist": "meta/pretrain_audit/excluded_episodes.jsonl",
            "episode_blacklist_sha256": blacklist_sha256,
        },
    }
    report = {
        "format_version": 1,
        "dataset": str(dataset_dir.resolve()),
        "source_revision": source_revision,
        "source_selection": "upstream KUKA conversion filters success=true",
        "total_episodes": len(episodes),
        "total_frames": int(lengths.sum()),
        "retained_episodes": len(episodes) - len(excluded_ids),
        "retained_frames": retained_frames,
        "excluded_episodes": len(excluded_ids),
        "excluded_frames": int(lengths.sum()) - retained_frames,
        "parquet_failures": parquet_failures,
        "exact_duplicate_of": duplicate_of,
        "low_dimensional_nonvideo_collisions": low_dimensional_collisions,
        "video_scan": {
            "containers_checked": len(episodes),
            "fully_decoded_sample_episodes": len(decoded_ids),
            "failures": video_failures,
        },
        "task_episode_counts": dict(sorted(task_episode_counts.items())),
        "language_failures": language_failures,
        "action_audit": {
            "gripper_open_fraction": float(np.mean(action_values[:, 6] > 0.5)),
            "roll_nonzero_fraction": float(np.mean(action_values[:, 3] != 0)),
            "pitch_nonzero_fraction": float(np.mean(action_values[:, 4] != 0)),
        },
        "timing_and_masks": {
            "control_hz": EXPECTED_FPS,
            "future_time_offsets_s": [0.0, 0.2, 0.4],
            "future_frame_indices": [0, 2, 4],
            "action_horizon": 8,
            "action_valid_fraction": float(
                sum(
                    sum(min(8, int(length) - base) for base in range(int(length)))
                    for length in lengths
                )
                / (8 * int(lengths.sum()))
            ),
            "future_0_2s_valid_fraction": float(
                sum(max(int(length) - 2, 0) for length in lengths)
                / int(lengths.sum())
            ),
            "future_0_4s_valid_fraction": float(
                sum(max(int(length) - 4, 0) for length in lengths)
                / int(lengths.sum())
            ),
        },
        "blacklist_sha256": blacklist_sha256,
    }
    return blacklist, report, stats


def write_outputs(
    output_dir: Path,
    stats_output: Path,
    blacklist: list[dict],
    report: dict,
    stats: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "excluded_episodes.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in blacklist:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "pretrain_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--decode-sample-episodes", type=int, default=1000)
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output_dir = args.output_dir or args.dataset_dir / "meta/pretrain_audit"
    stats_output = args.stats_output or args.dataset_dir / "meta/stats_gr00t.json"
    blacklist, report, stats = prepare(
        args.dataset_dir,
        workers=args.workers,
        decode_sample_episodes=args.decode_sample_episodes,
        source_revision=args.source_revision,
    )
    write_outputs(output_dir, stats_output, blacklist, report, stats)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "stats_output": str(stats_output),
                "retained_episodes": report["retained_episodes"],
                "retained_frames": report["retained_frames"],
                "excluded_episodes": report["excluded_episodes"],
                "video_failures": len(report["video_scan"]["failures"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
