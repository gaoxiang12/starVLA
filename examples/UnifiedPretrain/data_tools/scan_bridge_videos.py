#!/usr/bin/env python3
"""Scan required Bridge camera videos and write a non-destructive blacklist."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Iterable

import av


LFS_HEADER = b"version https://git-lfs.github.com/spec/"
DEFAULT_VIEWS = tuple(f"observation.images.image_{index}" for index in range(3))


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def probe_video(path: Path, decode_first_frame: bool = False) -> str | None:
    """Return a stable failure reason, or ``None`` for a usable video."""
    if not path.exists():
        return "missing"
    try:
        with path.open("rb") as handle:
            header = handle.read(len(LFS_HEADER))
    except OSError:
        return "unreadable"
    if header == LFS_HEADER:
        return "git_lfs_pointer"
    try:
        with av.open(str(path), mode="r") as container:
            streams = container.streams.video
            if not streams:
                return "no_video_stream"
            if decode_first_frame:
                try:
                    next(container.decode(streams[0]))
                except StopIteration:
                    return "no_decodable_frame"
    except Exception:
        return "invalid_container"
    return None


def read_lfs_pointer(path: Path) -> tuple[str, int] | None:
    """Return the object id and declared byte size for a Git LFS pointer."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or not lines[0].startswith(LFS_HEADER.decode("ascii")):
            return None
        object_id = next(line.split(":", 1)[1] for line in lines if line.startswith("oid sha256:"))
        size = int(next(line.split()[1] for line in lines if line.startswith("size ")))
        return object_id, size
    except (OSError, StopIteration, UnicodeDecodeError, ValueError):
        return None


def _video_path(dataset_dir: Path, pattern: str, chunks_size: int, episode: int, view: str) -> Path:
    return dataset_dir / pattern.format(
        episode_chunk=episode // chunks_size,
        episode_index=episode,
        video_key=view,
    )


def scan(
    dataset_dir: Path,
    views: tuple[str, ...],
    *,
    workers: int,
    decode_first_frame: bool,
) -> tuple[list[dict], dict]:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    episodes = list(read_jsonl(dataset_dir / "meta" / "episodes.jsonl"))
    pattern = str(info["video_path"])
    chunks_size = int(info["chunks_size"])

    jobs = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        for view in views:
            jobs.append(
                (
                    episode_index,
                    view,
                    _video_path(dataset_dir, pattern, chunks_size, episode_index, view),
                )
            )

    def run(job: tuple[int, str, Path]) -> tuple[int, str, Path, str | None]:
        episode_index, view, path = job
        return episode_index, view, path, probe_video(path, decode_first_frame)

    failures: dict[int, list[dict]] = {}
    reason_counts: Counter[str] = Counter()
    view_counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for episode_index, view, path, reason in executor.map(run, jobs):
            if reason is None:
                continue
            reason_counts[reason] += 1
            view_counts[view] += 1
            failures.setdefault(episode_index, []).append(
                {
                    "view": view,
                    "reason": reason,
                    "path": str(path.relative_to(dataset_dir)),
                }
            )

    episode_by_id = {int(row["episode_index"]): row for row in episodes}
    blacklist = []
    for episode_index in sorted(failures):
        episode = episode_by_id[episode_index]
        blacklist.append(
            {
                "episode_index": episode_index,
                "length": int(episode.get("length", 0)),
                "tasks": episode.get("tasks", []),
                "failures": sorted(failures[episode_index], key=lambda item: item["view"]),
            }
        )

    bad_ids = set(failures)
    lfs_objects: dict[str, int] = {}
    for rows in failures.values():
        for failure in rows:
            if failure["reason"] != "git_lfs_pointer":
                continue
            pointer = read_lfs_pointer(dataset_dir / failure["path"])
            if pointer is not None:
                object_id, size = pointer
                lfs_objects[object_id] = size
    valid_episodes = [row for row in episodes if int(row["episode_index"]) not in bad_ids]
    nonempty_valid = [row for row in valid_episodes if any(str(task).strip() for task in row.get("tasks", []))]
    summary = {
        "required_views": list(views),
        "probe_mode": "decode_first_frame" if decode_first_frame else "container",
        "total_episodes": len(episodes),
        "total_video_references": len(jobs),
        "blacklisted_episodes": len(blacklist),
        "blacklisted_frames": sum(row["length"] for row in blacklist),
        "valid_video_episodes": len(valid_episodes),
        "valid_video_frames": sum(int(row.get("length", 0)) for row in valid_episodes),
        "valid_nonempty_language_episodes": len(nonempty_valid),
        "valid_nonempty_language_frames": sum(int(row.get("length", 0)) for row in nonempty_valid),
        "failure_references_by_reason": dict(sorted(reason_counts.items())),
        "failure_references_by_view": dict(sorted(view_counts.items())),
        "missing_lfs_unique_objects": len(lfs_objects),
        "missing_lfs_declared_bytes": sum(lfs_objects.values()),
    }
    return blacklist, summary


def write_outputs(dataset_dir: Path, output_dir: Path, blacklist: list[dict], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    blacklist_path = output_dir / "bad_episodes.jsonl"
    report_path = output_dir / "video_health_report.json"
    with blacklist_path.open("w", encoding="utf-8") as handle:
        for row in blacklist:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--decode-first-frame", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output_dir = args.output_dir or args.dataset_dir / "meta" / "video_health"
    blacklist, summary = scan(
        args.dataset_dir,
        tuple(args.views),
        workers=args.workers,
        decode_first_frame=args.decode_first_frame,
    )
    write_outputs(args.dataset_dir, output_dir, blacklist, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote video-health sidecars to {output_dir}")


if __name__ == "__main__":
    main()
