#!/usr/bin/env python3
"""Audit Bridge task aliases without modifying the source metadata.

The generated JSONL is a reviewable sidecar. Runtime canonicalization lives in
``starVLA.task_language`` so serving does not depend on this data-local file.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable

from starVLA.task_language import canonical_bridge_task, canonical_metadata_text


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def transformation_kind(raw: str, canonical: str) -> str:
    normalized = canonical_metadata_text(raw)
    if raw == canonical:
        return "exact"
    if normalized == canonical:
        return "unicode_case_whitespace"
    return "lexical_alias"


def audit(dataset_dir: Path) -> tuple[list[dict], dict]:
    meta_dir = dataset_dir / "meta"
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"

    task_rows = list(read_jsonl(tasks_path))
    episode_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    for episode in read_jsonl(episodes_path):
        for raw_task in episode.get("tasks", []):
            raw_task = str(raw_task or "")
            episode_counts[raw_task] += 1
            frame_counts[raw_task] += int(episode.get("length", 0))

    aliases = []
    groups: dict[str, list[str]] = defaultdict(list)
    for row in task_rows:
        raw = str(row.get("task") or "")
        canonical = canonical_bridge_task(raw)
        groups[canonical].append(raw)
        aliases.append(
            {
                "task_index": int(row["task_index"]),
                "raw_task": raw,
                "canonical_task": canonical,
                "transformation": transformation_kind(raw, canonical),
                "episode_count": episode_counts[raw],
                "frame_count": frame_counts[raw],
            }
        )

    merged = {key: values for key, values in groups.items() if len(values) > 1}
    examples = sorted(
        (
            {
                "canonical_task": canonical,
                "alias_count": len(raw_values),
                "raw_tasks": raw_values[:20],
            }
            for canonical, raw_values in merged.items()
        ),
        key=lambda item: (-item["alias_count"], item["canonical_task"]),
    )[:100]
    summary = {
        "source_tasks": len(task_rows),
        "source_unique_raw_tasks": len({str(row.get('task') or '') for row in task_rows}),
        "canonical_tasks": len(groups),
        "collapsed_aliases": len(task_rows) - len(groups),
        "merged_groups": len(merged),
        "empty_source_tasks": sum(not str(row.get("task") or "").strip() for row in task_rows),
        "episode_references": sum(episode_counts.values()),
        "frame_references": sum(frame_counts.values()),
        "transformations": dict(Counter(row["transformation"] for row in aliases)),
        "largest_alias_groups": examples,
    }
    return aliases, summary


def write_outputs(dataset_dir: Path, output_dir: Path) -> dict:
    aliases, summary = audit(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aliases_path = output_dir / "bridge_task_aliases.jsonl"
    report_path = output_dir / "bridge_task_language_audit.json"
    with aliases_path.open("w", encoding="utf-8") as handle:
        for row in aliases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Bridge LeRobot dataset directory containing meta/tasks.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Sidecar output directory (default: DATASET_DIR/meta/task_language)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.dataset_dir / "meta" / "task_language"
    summary = write_outputs(args.dataset_dir, output_dir)
    concise = {key: value for key, value in summary.items() if key != "largest_alias_groups"}
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    print(f"wrote audit sidecars to {output_dir}")


if __name__ == "__main__":
    main()
