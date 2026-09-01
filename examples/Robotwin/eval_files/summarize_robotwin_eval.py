"""Summarize RoboTwin 2.0 success-rate files for one evaluated checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ALL_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle_horizontally",
    "shake_bottle",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)


@dataclass(frozen=True)
class Result:
    task: str
    mode: str
    successes: int
    trials: int
    success_rate: float
    ci95_low: float
    ci95_high: float
    source: str


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def parse_rate(path: Path) -> float:
    numeric_lines: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if re.fullmatch(r"(?:0(?:\.\d*)?|1(?:\.0*)?)", stripped):
            numeric_lines.append(float(stripped))
    if not numeric_lines:
        raise ValueError(f"No success rate in [0, 1] found in {path}")
    return numeric_lines[-1]


def read_task_file(path: Path) -> tuple[str, ...]:
    tasks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        task = line.split("#", 1)[0].strip()
        if task:
            tasks.append(task)
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    return tuple(tasks)


def resolve_tasks(value: str) -> tuple[str, ...]:
    if value == "all":
        return ALL_TASKS
    path = Path(value)
    if path.is_file():
        return read_task_file(path)
    tasks = tuple(task.strip() for task in value.split(",") if task.strip())
    if not tasks:
        raise ValueError("--tasks must be 'all', a task-list file, or comma-separated names")
    return tasks


def latest_result_file(
    result_root: Path,
    task: str,
    policy: str,
    mode: str,
    setting: str,
) -> Path | None:
    setting_root = result_root / task / policy / mode / setting
    candidates = list(setting_root.glob("*/_result.txt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def collect_results(
    result_root: Path,
    tasks: Iterable[str],
    policy: str,
    modes: Iterable[str],
    setting: str,
    episodes: int,
) -> tuple[list[Result], list[dict[str, str]]]:
    rows: list[Result] = []
    missing: list[dict[str, str]] = []
    for task in tasks:
        for mode in modes:
            path = latest_result_file(result_root, task, policy, mode, setting)
            if path is None:
                missing.append({"task": task, "mode": mode})
                continue
            rate = parse_rate(path)
            successes = min(episodes, max(0, int(round(rate * episodes))))
            low, high = wilson_interval(successes, episodes)
            rows.append(
                Result(
                    task=task,
                    mode=mode,
                    successes=successes,
                    trials=episodes,
                    success_rate=successes / episodes,
                    ci95_low=low,
                    ci95_high=high,
                    source=str(path),
                )
            )
    return rows, missing


def mode_summaries(
    rows: Iterable[Result],
    modes: Iterable[str],
) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    all_rows = list(rows)
    for mode in modes:
        mode_rows = [row for row in all_rows if row.mode == mode]
        if not mode_rows:
            continue
        successes = sum(row.successes for row in mode_rows)
        trials = sum(row.trials for row in mode_rows)
        low, high = wilson_interval(successes, trials)
        summaries[mode] = {
            "tasks": len(mode_rows),
            "successes": successes,
            "trials": trials,
            "macro_success_rate": sum(row.success_rate for row in mode_rows)
            / len(mode_rows),
            "micro_success_rate": successes / trials,
            "micro_ci95_low": low,
            "micro_ci95_high": high,
        }
    return summaries


def paired_gaps(rows: Iterable[Result]) -> dict[str, float | int] | None:
    by_task: dict[str, dict[str, float]] = {}
    for row in rows:
        by_task.setdefault(row.task, {})[row.mode] = row.success_rate
    gaps = [
        rates["demo_clean"] - rates["demo_randomized"]
        for rates in by_task.values()
        if "demo_clean" in rates and "demo_randomized" in rates
    ]
    if not gaps:
        return None
    return {
        "paired_tasks": len(gaps),
        "mean_easy_minus_hard": sum(gaps) / len(gaps),
        "max_easy_minus_hard": max(gaps),
        "min_easy_minus_hard": min(gaps),
    }


def write_outputs(
    output_dir: Path,
    rows: list[Result],
    metadata: dict[str, object],
    summaries: dict[str, dict[str, float | int]],
    gaps: dict[str, float | int] | None,
    missing: list[dict[str, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "robotwin_eval_summary.csv"
    json_path = output_dir / "robotwin_eval_summary.json"

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    payload = {
        "metadata": metadata,
        "summary": summaries,
        "easy_hard_gap": gaps,
        "missing": missing,
        "results": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(
    summaries: dict[str, dict[str, float | int]],
    gaps: dict[str, float | int] | None,
    missing: list[dict[str, str]],
) -> None:
    print("mode             tasks   macro SR   micro SR       micro 95% CI")
    print("----------------------------------------------------------------")
    for mode, values in summaries.items():
        print(
            f"{mode:17s} {int(values['tasks']):5d}   "
            f"{100.0 * float(values['macro_success_rate']):7.2f}%   "
            f"{100.0 * float(values['micro_success_rate']):7.2f}%   "
            f"[{100.0 * float(values['micro_ci95_low']):6.2f}%, "
            f"{100.0 * float(values['micro_ci95_high']):6.2f}%]"
        )
    if gaps is not None:
        print(
            "paired Easy-Hard gap: "
            f"{100.0 * float(gaps['mean_easy_minus_hard']):.2f} pp "
            f"over {int(gaps['paired_tasks'])} tasks"
        )
    if missing:
        print(f"missing task/mode results: {len(missing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        required=True,
        help="RoboTwin eval_result directory",
    )
    parser.add_argument("--setting", required=True, help="ckpt_setting used by the launcher")
    parser.add_argument("--policy", default="model2robotwin_interface")
    parser.add_argument(
        "--modes",
        default="demo_clean,demo_randomized",
        help="Comma-separated evaluation modes",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="'all', a task-list file, or comma-separated task names",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", help="Fail if any result is missing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    modes = tuple(mode.strip() for mode in args.modes.split(",") if mode.strip())
    if not modes:
        raise SystemExit("--modes resolved to an empty list")
    tasks = resolve_tasks(args.tasks)
    rows, missing = collect_results(
        args.result_root,
        tasks,
        args.policy,
        modes,
        args.setting,
        args.episodes,
    )
    if not rows:
        raise SystemExit("No RoboTwin result files matched the requested evaluation")

    summaries = mode_summaries(rows, modes)
    gaps = paired_gaps(rows)
    metadata: dict[str, object] = {
        "result_root": str(args.result_root),
        "policy": args.policy,
        "setting": args.setting,
        "episodes_per_task": args.episodes,
        "requested_tasks": len(tasks),
        "modes": list(modes),
    }
    write_outputs(args.output_dir, rows, metadata, summaries, gaps, missing)
    print_summary(summaries, gaps, missing)
    if args.strict and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
