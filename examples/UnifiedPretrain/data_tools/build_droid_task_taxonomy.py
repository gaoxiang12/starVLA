"""Build a deterministic task taxonomy for the DROID LeRobot export.

The policy uses language as a task identifier, not as an open-vocabulary VLM
prompt.  This tool maps free-form DROID annotations onto stable, reusable task
descriptions and writes reviewable sidecars without modifying source metadata.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable


from starVLA.droid_task_taxonomy import (
    EMPTY_LABEL,
    NO_ACTION_LABEL,
    OTHER_LABEL,
)
from starVLA.task_language import canonical_droid_task


def classify_task(text: str) -> str:
    """Backward-compatible alias for existing audit scripts and tests."""
    return canonical_droid_task(text)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_taxonomy(dataset_dir: Path, output_dir: Path, min_episodes: int) -> dict:
    meta_dir = dataset_dir / "meta"
    tasks = read_jsonl(meta_dir / "tasks.jsonl")
    episodes = read_jsonl(meta_dir / "episodes.jsonl")

    raw_to_category = {
        row["task"]: classify_task(row.get("task", "")) for row in tasks
    }
    episode_categories = []
    for episode in episodes:
        raw = (episode.get("tasks") or [""])[0]
        episode_categories.append(raw_to_category.get(raw, classify_task(raw)))

    counts = Counter(episode_categories)
    frame_counts: Counter[str] = Counter()
    examples: dict[str, Counter[str]] = defaultdict(Counter)
    for episode, category in zip(episodes, episode_categories):
        frame_counts[category] += int(episode["length"])
        raw = (episode.get("tasks") or [""])[0]
        examples[category][raw] += 1

    excluded_categories = {
        category
        for category, count in counts.items()
        if not category
        or category in {NO_ACTION_LABEL, OTHER_LABEL}
        or count < min_episodes
    }
    excluded_episodes = []
    episode_rows = []
    for episode, category in zip(episodes, episode_categories):
        eligible = category not in excluded_categories
        row = {
            "episode_index": int(episode["episode_index"]),
            "raw_task": (episode.get("tasks") or [""])[0],
            "task_category": category,
            "eligible": eligible,
        }
        episode_rows.append(row)
        if not eligible:
            reason = (
                "empty_task"
                if not category
                else "non_action_or_ambiguous"
                if category in {NO_ACTION_LABEL, OTHER_LABEL}
                else "category_below_min_episodes"
            )
            excluded_episodes.append(
                {"episode_index": row["episode_index"], "reason": reason}
            )

    category_rows = []
    for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        category_rows.append(
            {
                "task_category": category,
                "episode_count": count,
                "frame_count": frame_counts[category],
                "eligible": category not in excluded_categories,
                "example_descriptions": [
                    text for text, _ in examples[category].most_common(10)
                ],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "droid_task_categories.jsonl", category_rows)
    write_jsonl(output_dir / "droid_episode_task_categories.jsonl", episode_rows)
    write_jsonl(output_dir / "droid_pretrain_excluded_episodes.jsonl", excluded_episodes)

    eligible_categories = [row for row in category_rows if row["eligible"]]
    report = {
        "dataset_dir": str(dataset_dir),
        "min_episodes": min_episodes,
        "total_episodes": len(episodes),
        "total_frames": sum(int(row["length"]) for row in episodes),
        "raw_task_count": len(tasks),
        "category_count": len(category_rows),
        "eligible_category_count": len(eligible_categories),
        "eligible_episode_count": sum(row["episode_count"] for row in eligible_categories),
        "eligible_frame_count": sum(row["frame_count"] for row in eligible_categories),
        "excluded_episode_count": len(excluded_episodes),
        "empty_episode_count": counts.get(EMPTY_LABEL, 0),
        "no_action_episode_count": counts.get(NO_ACTION_LABEL, 0),
    }
    with (output_dir / "droid_task_language_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-episodes", type=int, default=20)
    args = parser.parse_args()
    output_dir = args.output_dir or args.dataset_dir / "meta" / "task_language"
    report = build_taxonomy(args.dataset_dir, output_dir, args.min_episodes)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
