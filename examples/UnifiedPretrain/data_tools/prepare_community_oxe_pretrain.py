#!/usr/bin/env python3
"""Audit and prepare TACO Play, BC-Z, Fractal, and FMB for unified pretraining.

The source Parquet/video payload is never rewritten.  Invalid, duplicate,
ambiguous, and rare-task episodes are recorded in a deterministic blacklist;
all normalization statistics are computed from the retained training split.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable

import av
import numpy as np
import pyarrow.parquet as pq

from starVLA.task_language import classify_oxe_task, resolve_task_language


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_ROOT = REPO_ROOT / "examples/UnifiedPretrain/train_files"
LOW_DIMENSIONAL_KEYS = ("observation.state", "action")
MIN_TASK_EPISODES = 20


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    repo_id: str
    modality_template: str
    control_hz: int
    action_horizon: int
    future_frame_indices: tuple[int, int, int]
    video_keys: tuple[str, ...]
    robot_tag: str
    action_spec_id: str
    state_spec_id: str
    action_semantics: str
    state_semantics: str


SPECS = {
    "taco_play": DatasetSpec(
        name="taco_play",
        root=Path("/data/gaoxiang/taco_play_lerobot"),
        repo_id="IPEC-COMMUNITY/taco_play_lerobot",
        modality_template="taco_play_modality.json",
        control_hz=15,
        action_horizon=8,
        future_frame_indices=(0, 3, 6),
        video_keys=(
            "observation.images.rgb_static",
            "observation.images.rgb_gripper",
        ),
        robot_tag="taco_franka",
        action_spec_id=(
            "taco_franka_world_delta_scaled_xyz50_rpy20_gripper_open_abs_7_15hz"
        ),
        state_spec_id="taco_franka_eef_xyz_rpy_pad_gripper_state_8",
        action_semantics=(
            "TACO rel_actions_world: scaled world-frame XYZ/RPY relative command; "
            "absolute gripper-open bit"
        ),
        state_semantics="absolute world-frame EEF XYZ/RPY, pad, gripper position",
    ),
    "bc_z": DatasetSpec(
        name="bc_z",
        root=Path("/data/gaoxiang/bc_z_lerobot"),
        repo_id="IPEC-COMMUNITY/bc_z_lerobot",
        modality_template="bc_z_modality.json",
        control_hz=10,
        action_horizon=8,
        future_frame_indices=(0, 2, 4),
        video_keys=("observation.images.image",),
        robot_tag="google_bcz",
        action_spec_id=(
            "google_bcz_eef_delta_xyz_axis_angle_gripper_open_abs_7_10hz"
        ),
        state_spec_id="google_bcz_eef_xyz_rpy_pad_gripper_position_8",
        action_semantics=(
            "future/xyz_residual and future/axis_angle_residual; absolute "
            "inverted target_close gripper-open bit"
        ),
        state_semantics="absolute EEF XYZ/RPY, pad, continuous gripper position",
    ),
    "fractal": DatasetSpec(
        name="fractal",
        root=Path("/data/gaoxiang/fractal20220817_data_lerobot"),
        repo_id="IPEC-COMMUNITY/fractal20220817_data_lerobot",
        modality_template="fractal_modality.json",
        control_hz=3,
        action_horizon=8,
        future_frame_indices=(0, 1, 1),
        video_keys=("observation.images.image",),
        robot_tag="google_rt1",
        action_spec_id="google_rt1_eef_delta_xyz_rpy_gripper_open_abs_7_3hz",
        state_spec_id="google_rt1_eef_xyz_quaternion_xyzw_gripper_closed_8",
        action_semantics=(
            "base-relative world_vector and RPY rotation_delta; absolute "
            "gripper-open bit reconstructed from relative commands"
        ),
        state_semantics="base-relative EEF XYZ/quaternion XYZW and gripper-closed bit",
    ),
    "fmb": DatasetSpec(
        name="fmb",
        root=Path("/data/gaoxiang/fmb_dataset_lerobot"),
        repo_id="IPEC-COMMUNITY/fmb_dataset_lerobot",
        modality_template="fmb_modality.json",
        control_hz=10,
        action_horizon=8,
        future_frame_indices=(0, 2, 4),
        video_keys=(
            "observation.images.image_side_1",
            "observation.images.image_side_2",
            "observation.images.image_wrist_1",
            "observation.images.image_wrist_2",
        ),
        robot_tag="fmb_franka",
        action_spec_id="fmb_franka_eef_twist_normalized_gripper_open_abs_7_10hz",
        state_spec_id="fmb_franka_eef_xyz_quaternion_xyzw_gripper_position_8",
        action_semantics=(
            "normalized 6D EEF linear/angular twist and absolute gripper-open bit"
        ),
        state_semantics="absolute EEF XYZ/quaternion XYZW and gripper position",
    ),
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_existing_duplicate_audit(spec: DatasetSpec) -> dict:
    """Confirm low-dimensional duplicate candidates also share every video."""

    info = json.loads((spec.root / "meta/info.json").read_text(encoding="utf-8"))
    audit_dir = spec.root / "meta/pretrain_audit"
    duplicates = read_jsonl(audit_dir / "duplicate_episodes.jsonl")
    audit_report_path = audit_dir / "pretrain_audit_report.json"
    audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    digest_cache: dict[tuple[int, str], str] = {}

    def video_digest(episode_index: int, key: str) -> str:
        cache_key = (episode_index, key)
        if cache_key not in digest_cache:
            digest_cache[cache_key] = file_sha256(
                video_path(spec.root, info, episode_index, key)
            )
        return digest_cache[cache_key]

    mismatches = []
    for row in duplicates:
        episode_index = int(row["episode_index"])
        owner = int(row["duplicate_of"])
        different_keys = [
            key
            for key in spec.video_keys
            if video_digest(episode_index, key) != video_digest(owner, key)
        ]
        if different_keys:
            mismatches.append(
                {
                    "episode_index": episode_index,
                    "duplicate_of": owner,
                    "different_video_keys": different_keys,
                }
            )

    report = {
        "algorithm": "sha256_of_encoded_video_file_for_every_configured_view",
        "low_dimensional_duplicate_candidates": audit_report.get(
            "low_dimensional_duplicate_candidates", len(duplicates)
        ),
        "rejected_low_dimensional_candidates": audit_report.get(
            "rejected_duplicate_candidates", 0
        ),
        "duplicate_records_checked": len(duplicates),
        "verified_exact_duplicate_episodes": len(duplicates) - len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    (audit_dir / "duplicate_video_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if mismatches:
        raise RuntimeError(
            f"{spec.name}: {len(mismatches)} low-dimensional duplicate "
            "candidates have different videos; rerun with corrected filtering"
        )
    return report


def data_path(root: Path, info: dict, episode_index: int) -> Path:
    return root / str(info["data_path"]).format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
    )


def video_path(root: Path, info: dict, episode_index: int, key: str) -> Path:
    return root / str(info["video_path"]).format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
        video_key=key,
    )


def fixed_list_numpy(table, key: str, width: int) -> np.ndarray:
    column = table.column(key).combine_chunks()
    values = np.asarray(column.values.to_numpy(zero_copy_only=False))
    return values.astype(np.float32, copy=False).reshape(table.num_rows, width)


def scan_parquet(job: tuple) -> dict:
    spec, info, episode, expected_start, valid_task_indices = job
    episode_index = int(episode["episode_index"])
    expected_length = int(episode["length"])
    path = data_path(spec.root, info, episode_index)
    errors: list[str] = []
    result = {
        "episode_index": episode_index,
        "path": str(path.relative_to(spec.root)),
        "length": expected_length,
        "errors": errors,
    }
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
        if length == 0:
            errors.append("empty_episode")
        if length != expected_length:
            errors.append(f"length:{length}!={expected_length}")
        state = fixed_list_numpy(table, "observation.state", 8)
        action = fixed_list_numpy(table, "action", 7)
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
        unique_tasks = sorted({int(value) for value in task_index.tolist()})
        if len(unique_tasks) != 1:
            errors.append(f"multiple_task_indices:{unique_tasks}")
        if any(value not in valid_task_indices for value in unique_tasks):
            errors.append(f"task_index_out_of_range:{unique_tasks}")
        expected_timestamp = np.arange(length) / float(spec.control_hz)
        if not np.allclose(timestamp, expected_timestamp, atol=2e-5):
            errors.append("invalid_timestamp")
        digest = hashlib.blake2b(digest_size=20)
        digest.update(np.asarray([length], dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(state).tobytes())
        digest.update(np.ascontiguousarray(action).tobytes())
        result.update(
            {
                "digest": digest.hexdigest(),
                "task_indices": unique_tasks,
                "gripper_open_count": int(np.count_nonzero(action[:, 6] > 0.5)),
                "gripper_total": int(length),
            }
        )
    except Exception as error:
        errors.append(f"unreadable_parquet:{type(error).__name__}:{error}")
    return result


def scan_video(job: tuple) -> dict:
    spec, info, episode_index, expected_frames, key = job
    path = video_path(spec.root, info, episode_index, key)
    errors: list[str] = []
    row = {
        "episode_index": episode_index,
        "video_key": key,
        "path": str(path.relative_to(spec.root)),
        "errors": errors,
    }
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            # Keep full-dataset audits memory bounded.  AV1 decoders may
            # otherwise create many codec threads per Python worker.
            stream.codec_context.thread_count = 1
            frames = sum(1 for _ in container.decode(stream))
            fps = float(stream.average_rate) if stream.average_rate else None
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
        feature = info["features"][key]
        video_info = feature.get("info", {})
        expected_width = int(video_info.get("video.width", feature["shape"][1]))
        expected_height = int(video_info.get("video.height", feature["shape"][0]))
        if frames != expected_frames:
            errors.append(f"decoded_frames:{frames}!={expected_frames}")
        if fps is None or abs(fps - float(spec.control_hz)) > 0.05:
            errors.append(f"fps:{fps}!={spec.control_hz}")
        if (width, height) != (expected_width, expected_height):
            errors.append(
                f"resolution:{width}x{height}!={expected_width}x{expected_height}"
            )
        row.update(
            {
                "decoded_frames": frames,
                "fps": fps,
                "resolution": [width, height],
                "size_bytes": path.stat().st_size,
            }
        )
    except Exception as error:
        errors.append(f"unreadable_video:{type(error).__name__}:{error}")
    return row


class StreamingStats:
    def __init__(self, width: int):
        self.count = 0
        self.total = np.zeros(width, dtype=np.float64)
        self.total_sq = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf, dtype=np.float64)
        self.maximum = np.full(width, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values64 = np.asarray(values, dtype=np.float64)
        self.count += len(values64)
        self.total += values64.sum(axis=0)
        self.total_sq += np.square(values64).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values64.min(axis=0))
        self.maximum = np.maximum(self.maximum, values64.max(axis=0))

    def finish(self, samples: np.ndarray) -> dict:
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        return {
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "q01": np.quantile(samples, 0.01, axis=0).tolist(),
            "q99": np.quantile(samples, 0.99, axis=0).tolist(),
        }


def build_statistics(
    spec: DatasetSpec,
    info: dict,
    retained_ids: list[int],
    *,
    quantile_sample_episodes: int,
) -> tuple[dict, dict]:
    state_stats = StreamingStats(8)
    action_stats = StreamingStats(7)
    sample_count = min(quantile_sample_episodes, len(retained_ids))
    sample_positions = set(
        np.linspace(0, len(retained_ids) - 1, num=sample_count, dtype=np.int64).tolist()
    )
    state_samples: list[np.ndarray] = []
    action_samples: list[np.ndarray] = []
    for position, episode_index in enumerate(retained_ids):
        table = pq.read_table(
            data_path(spec.root, info, episode_index),
            columns=list(LOW_DIMENSIONAL_KEYS),
        )
        state = fixed_list_numpy(table, "observation.state", 8)
        action = fixed_list_numpy(table, "action", 7)
        state_stats.update(state)
        action_stats.update(action)
        if position in sample_positions:
            state_samples.append(state)
            action_samples.append(action)
    state_sample = np.concatenate(state_samples, axis=0)
    action_sample = np.concatenate(action_samples, axis=0)
    statistics = {
        "observation.state": state_stats.finish(state_sample),
        "action": action_stats.finish(action_sample),
    }
    provenance = {
        "method": "exact_streaming_mean_std_min_max_with_deterministic_quantiles",
        "retained_episodes": len(retained_ids),
        "retained_frames": state_stats.count,
        "quantile_sample_episodes": sample_count,
        "quantile_sample_frames": len(state_sample),
    }
    return statistics, provenance


def timing_valid_fraction(lengths: list[int], offsets: tuple[int, ...]) -> list[float]:
    total = sum(lengths)
    return [sum(max(length - offset, 0) for length in lengths) / total for offset in offsets]


def prepare(spec: DatasetSpec, *, workers: int, video_workers: int, quantile_sample_episodes: int) -> dict:
    root = spec.root
    info_path = root / "meta/info.json"
    episodes_path = root / "meta/episodes.jsonl"
    tasks_path = root / "meta/tasks.jsonl"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = read_jsonl(episodes_path)
    tasks = read_jsonl(tasks_path)
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(f"episode metadata count mismatch for {spec.name}")
    if len(tasks) != int(info["total_tasks"]):
        raise ValueError(f"task metadata count mismatch for {spec.name}")
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks}
    valid_task_indices = set(task_by_index)

    starts = np.cumsum([0] + [int(row["length"]) for row in episodes[:-1]])
    jobs = [
        (spec, info, episode, int(starts[position]), valid_task_indices)
        for position, episode in enumerate(episodes)
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parquet_rows = []
        for completed, row in enumerate(executor.map(scan_parquet, jobs), start=1):
            parquet_rows.append(row)
            if completed % 10_000 == 0:
                print(
                    f"[{spec.name}] parquet audit: {completed}/{len(jobs)}",
                    flush=True,
                )
    parquet_failures = [row for row in parquet_rows if row["errors"]]
    excluded_reasons: dict[int, list[str]] = defaultdict(list)
    for row in parquet_failures:
        excluded_reasons[int(row["episode_index"])].extend(row["errors"])

    digest_owner: dict[str, int] = {}
    duplicate_candidates = []
    for row in parquet_rows:
        digest = row.get("digest")
        if not digest or row["errors"]:
            continue
        episode_index = int(row["episode_index"])
        if digest in digest_owner:
            owner = digest_owner[digest]
            duplicate_candidates.append(
                {"episode_index": episode_index, "duplicate_of": owner, "digest": digest}
            )
        else:
            digest_owner[digest] = episode_index

    video_jobs = [
        (spec, info, int(episode["episode_index"]), int(episode["length"]), key)
        for episode in episodes
        for key in spec.video_keys
    ]
    with ThreadPoolExecutor(max_workers=video_workers) as executor:
        video_rows = []
        for completed, row in enumerate(executor.map(scan_video, video_jobs), start=1):
            video_rows.append(row)
            if completed % 10_000 == 0:
                print(
                    f"[{spec.name}] video audit: {completed}/{len(video_jobs)}",
                    flush=True,
                )
    video_failures = [row for row in video_rows if row["errors"]]
    infrastructure_failures = [
        row
        for row in video_failures
        if any(
            marker in error
            for error in row["errors"]
            for marker in ("MemoryError", "Cannot allocate memory", "Too many open files")
        )
    ]
    if infrastructure_failures:
        examples = infrastructure_failures[:3]
        raise RuntimeError(
            f"video audit infrastructure failure for {spec.name}; "
            f"reduce --video-workers (count={len(infrastructure_failures)}, "
            f"examples={examples})"
        )
    failed_video_episodes = {
        int(row["episode_index"]) for row in video_failures
    }
    for row in video_failures:
        excluded_reasons[int(row["episode_index"])].extend(
            f"{row['video_key']}:{error}" for error in row["errors"]
        )

    # Equal state/action arrays alone are not sufficient: stationary motions
    # can occur in different visual scenes.  Confirm duplicates using every
    # configured encoded video before blacklisting an episode.
    video_digest_cache: dict[tuple[int, str], str] = {}

    def video_digest(episode_index: int, key: str) -> str:
        cache_key = (episode_index, key)
        if cache_key not in video_digest_cache:
            video_digest_cache[cache_key] = file_sha256(
                video_path(root, info, episode_index, key)
            )
        return video_digest_cache[cache_key]

    duplicates = []
    rejected_duplicate_candidates = []
    for candidate in duplicate_candidates:
        episode_index = int(candidate["episode_index"])
        owner = int(candidate["duplicate_of"])
        if episode_index in failed_video_episodes or owner in failed_video_episodes:
            rejected_duplicate_candidates.append(
                {**candidate, "verification": "video_unavailable"}
            )
            continue
        different_keys = [
            key
            for key in spec.video_keys
            if video_digest(episode_index, key) != video_digest(owner, key)
        ]
        if different_keys:
            rejected_duplicate_candidates.append(
                {**candidate, "different_video_keys": different_keys}
            )
            continue
        excluded_reasons[episode_index].append(f"exact_duplicate_of:{owner}")
        duplicates.append(
            {
                **candidate,
                "video_sha256": {
                    key: video_digest(episode_index, key) for key in spec.video_keys
                },
            }
        )
    row_by_episode = {int(row["episode_index"]): row for row in parquet_rows}
    taxonomy_by_task = {}
    for task_index, raw_text in task_by_index.items():
        label = classify_oxe_task(raw_text)
        canonical = resolve_task_language(raw_text, spec.name, "oxe_taxonomy")
        taxonomy_by_task[task_index] = (raw_text, canonical, label)

    def valid_episode_task(episode_index: int) -> int | None:
        row = row_by_episode.get(episode_index)
        if row is None or row["errors"] or len(row.get("task_indices", [])) != 1:
            return None
        return int(row["task_indices"][0])

    # Recompute counts until the rare-task filter stabilizes after structural
    # and video exclusions.
    for _ in range(2):
        canonical_counts = Counter()
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            if excluded_reasons.get(episode_index):
                continue
            task_index = valid_episode_task(episode_index)
            if task_index is not None:
                canonical_counts[taxonomy_by_task[task_index][1]] += 1
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            if excluded_reasons.get(episode_index):
                continue
            task_index = valid_episode_task(episode_index)
            if task_index is None:
                continue
            _, canonical, label = taxonomy_by_task[task_index]
            if label.status != "classified":
                excluded_reasons[episode_index].append(
                    f"task_language_{label.status}:{canonical or '<empty>'}"
                )
            elif canonical_counts[canonical] < MIN_TASK_EPISODES:
                excluded_reasons[episode_index].append(
                    f"task_class_below_{MIN_TASK_EPISODES}_episodes:{canonical}"
                )

    retained_ids = [
        int(episode["episode_index"])
        for episode in episodes
        if not excluded_reasons.get(int(episode["episode_index"]))
    ]
    if not retained_ids:
        raise RuntimeError(f"{spec.name}: no episodes remain after filtering")
    retained_set = set(retained_ids)
    statistics, stats_provenance = build_statistics(
        spec,
        info,
        retained_ids,
        quantile_sample_episodes=quantile_sample_episodes,
    )

    output_dir = root / "meta/pretrain_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE_ROOT / spec.modality_template
    shutil.copyfile(template, root / "meta/modality.json")

    blacklist = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        reasons = sorted(set(excluded_reasons.get(episode_index, [])))
        if reasons:
            blacklist.append(
                {
                    "episode_index": episode_index,
                    "length": int(episode["length"]),
                    "tasks": episode.get("tasks", []),
                    "reasons": reasons,
                }
            )
    write_jsonl(output_dir / "excluded_episodes.jsonl", blacklist)
    write_jsonl(output_dir / "parquet_failures.jsonl", parquet_failures)
    write_jsonl(output_dir / "corrupt_videos.jsonl", video_failures)
    write_jsonl(output_dir / "duplicate_episodes.jsonl", duplicates)
    write_jsonl(
        output_dir / "rejected_duplicate_candidates.jsonl",
        rejected_duplicate_candidates,
    )

    final_canonical_counts = Counter()
    raw_retained_counts = Counter()
    for episode_index in retained_ids:
        task_index = valid_episode_task(episode_index)
        if task_index is None:
            continue
        raw_text, canonical, _ = taxonomy_by_task[task_index]
        final_canonical_counts[canonical] += 1
        raw_retained_counts[task_index] += 1
    task_audit = []
    canonical_variants: dict[str, list[dict]] = defaultdict(list)
    for task_index in sorted(task_by_index):
        raw_text, canonical, label = taxonomy_by_task[task_index]
        row = {
            "task_index": task_index,
            "raw_text": raw_text,
            "canonical_text": canonical,
            "family": label.family,
            "status": label.status,
            "confidence": label.confidence,
            "retained_episode_count": raw_retained_counts[task_index],
            "canonical_retained_episode_count": final_canonical_counts[canonical],
        }
        task_audit.append(row)
        canonical_variants[canonical].append(
            {"task_index": task_index, "raw_text": raw_text}
        )
    write_jsonl(output_dir / "task_language_audit.jsonl", task_audit)
    write_jsonl(
        output_dir / "task_language_catalog.jsonl",
        (
            {
                "canonical_text": canonical,
                "retained_episode_count": final_canonical_counts[canonical],
                "raw_variants": variants,
            }
            for canonical, variants in sorted(canonical_variants.items())
        ),
    )

    blacklist_sha256 = file_sha256(output_dir / "excluded_episodes.jsonl")
    stats_payload = {
        "__format_version": 2,
        "__cache_config": {"mode": "abs"},
        "statistics": statistics,
        "__provenance": {
            **stats_provenance,
            "dataset": spec.repo_id,
            "filter": "meta/pretrain_audit/excluded_episodes.jsonl",
            "filter_sha256": blacklist_sha256,
        },
    }
    (root / "meta/stats_gr00t.json").write_text(
        json.dumps(stats_payload, indent=2) + "\n", encoding="utf-8"
    )
    stats_json_path = root / "meta/stats.json"
    stats_json = (
        json.loads(stats_json_path.read_text(encoding="utf-8"))
        if stats_json_path.is_file()
        else {}
    )
    stats_json.update(statistics)
    stats_json_path.write_text(
        json.dumps(stats_json, indent=2) + "\n", encoding="utf-8"
    )

    retained_lengths = [
        int(episodes[episode_index]["length"]) for episode_index in retained_ids
    ]
    gripper_open = sum(
        int(row_by_episode[index].get("gripper_open_count", 0))
        for index in retained_ids
    )
    gripper_total = sum(
        int(row_by_episode[index].get("gripper_total", 0)) for index in retained_ids
    )
    canonical_merges = sum(len(variants) > 1 for variants in canonical_variants.values())
    retained_ratio = len(retained_ids) / len(episodes)
    score_components = {
        "schema_and_numeric": 25 if not parquet_failures else 20,
        "video_integrity": 20 if not video_failures else 15,
        "language_audit": round(15 * retained_ratio, 2),
        "metadata_and_stats": 20,
        "scale_and_task_diversity": 20 if len(final_canonical_counts) >= 10 else 15,
    }
    value_score = round(sum(score_components.values()), 2)
    report = {
        "format_version": 1,
        "dataset": spec.repo_id,
        "dataset_root": str(root),
        "source_metadata_sha256": {
            "info.json": file_sha256(info_path),
            "episodes.jsonl": file_sha256(episodes_path),
            "tasks.jsonl": file_sha256(tasks_path),
        },
        "robot_tag": spec.robot_tag,
        "action_spec_id": spec.action_spec_id,
        "state_spec_id": spec.state_spec_id,
        "action_semantics": spec.action_semantics,
        "state_semantics": spec.state_semantics,
        "total_episodes": len(episodes),
        "total_frames": int(sum(int(row["length"]) for row in episodes)),
        "retained_episodes": len(retained_ids),
        "retained_frames": int(sum(retained_lengths)),
        "excluded_episodes": len(blacklist),
        "parquet_failures": len(parquet_failures),
        "video_files_fully_decoded": len(video_rows),
        "video_failures": len(video_failures),
        "exact_duplicate_episodes": len(duplicates),
        "low_dimensional_duplicate_candidates": len(duplicate_candidates),
        "rejected_duplicate_candidates": len(rejected_duplicate_candidates),
        "task_language": {
            "mode": "oxe_taxonomy",
            "minimum_episodes": MIN_TASK_EPISODES,
            "raw_task_count": len(tasks),
            "retained_canonical_task_count": len(final_canonical_counts),
            "canonical_groups_with_multiple_raw_variants": canonical_merges,
        },
        "normalization": stats_provenance,
        "gripper_open_fraction": gripper_open / gripper_total,
        "timing_and_masks": {
            "control_hz": spec.control_hz,
            "future_time_offsets_s": [0.0, 0.2, 0.4],
            "future_frame_indices": list(spec.future_frame_indices),
            "repeated_future_offsets_masked": len(set(spec.future_frame_indices)) != 3,
            "future_valid_fraction": timing_valid_fraction(
                retained_lengths, spec.future_frame_indices
            ),
            "action_horizon": spec.action_horizon,
            "action_valid_fraction": sum(
                sum(min(spec.action_horizon, length - base) for base in range(length))
                for length in retained_lengths
            )
            / (spec.action_horizon * sum(retained_lengths)),
        },
        "blacklist_sha256": blacklist_sha256,
        "pretraining_value": {
            "score": value_score,
            "components": score_components,
            "verdict": "accept_candidate" if value_score >= 75 else "hold",
            "formal_mixture_enabled": False,
        },
    }
    (output_dir / "pretrain_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=tuple(SPECS) + ("all",))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--video-workers", type=int, default=8)
    parser.add_argument("--quantile-sample-episodes", type=int, default=8192)
    parser.add_argument("--verify-duplicates-only", action="store_true")
    args = parser.parse_args()
    names = tuple(SPECS) if args.dataset == "all" else (args.dataset,)
    if args.verify_duplicates_only:
        for name in names:
            report = verify_existing_duplicate_audit(SPECS[name])
            print(json.dumps({"dataset": name, **report}, indent=2), flush=True)
        return 0
    for name in names:
        report = prepare(
            SPECS[name],
            workers=args.workers,
            video_workers=args.video_workers,
            quantile_sample_episodes=args.quantile_sample_episodes,
        )
        print(
            json.dumps(
                {
                    "dataset": name,
                    "retained_episodes": report["retained_episodes"],
                    "excluded_episodes": report["excluded_episodes"],
                    "video_failures": report["video_failures"],
                    "value_score": report["pretraining_value"]["score"],
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
