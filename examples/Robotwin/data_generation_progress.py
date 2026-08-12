#!/usr/bin/env python3
"""Continuously render RoboTwin collection progress to a small text file."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=20000)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--fallback-replay-seconds", type=float, default=65.0)
    return parser.parse_args()


def count_seeds(path: Path) -> tuple[int, list[int]]:
    try:
        seeds = [int(value) for value in path.read_text().split()]
    except (FileNotFoundError, ValueError):
        seeds = []
    return len(seeds), seeds


def episode_files(run_dir: Path) -> list[Path]:
    def episode_index(path: Path) -> int:
        return int(path.stem.removeprefix("episode"))

    return sorted((run_dir / "data").glob("episode*.hdf5"), key=episode_index)


def process_alive(pid_file: Path) -> tuple[bool, int | None]:
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True, pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False, None


def bar(done: int, total: int, width: int = 32) -> str:
    ratio = min(1.0, done / total) if total else 1.0
    filled = int(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done:4d}/{total} {ratio:6.2%}"


def duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "calibrating"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s"


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    started_file = run_dir / "job_started_at.txt"
    try:
        started_at = float(started_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        started_at = time.time()
    output = run_dir / "progress.txt"

    while True:
        now = time.time()
        planned, seeds = count_seeds(run_dir / "seed.txt")
        files = episode_files(run_dir)
        materialized = len(files)
        alive, pid = process_alive(run_dir / "collector.pid")
        attempts = max(0, (max(seeds) - args.start_seed + 1) if seeds else 0)
        failures = max(0, attempts - planned)
        success_rate = planned / attempts if attempts else None

        plan_rate = planned / max(1.0, now - started_at)
        plan_eta = (args.target - planned) / plan_rate if plan_rate > 0 else None

        replay_rate = None
        if materialized >= 2:
            span = files[-1].stat().st_mtime - files[0].stat().st_mtime
            if span > 0:
                replay_rate = (materialized - 1) / span
        elif materialized == 1:
            plan_finished = (run_dir / "seed.txt").stat().st_mtime
            replay_rate = 1.0 / max(1.0, files[0].stat().st_mtime - plan_finished)

        if planned < args.target:
            replay_remaining = args.target * args.fallback_replay_seconds
            eta_seconds = None if plan_eta is None else plan_eta + replay_remaining
            stage = "expert planning"
        elif materialized < args.target:
            eta_seconds = ((args.target - materialized) / replay_rate) if replay_rate else None
            stage = "HDF5 replay"
        else:
            eta_seconds = 0.0
            stage = "complete" if not alive else "finalizing instructions"

        free = shutil.disk_usage(run_dir).free / (1024**3)
        rate_text = f"{plan_rate * 60:.2f} accepted/min" if planned else "calibrating"
        replay_text = f"{replay_rate * 60:.2f} episodes/min" if replay_rate else "calibrating"
        status = "RUNNING" if alive else ("DONE" if materialized >= args.target else "NOT RUNNING")
        lines = [
            "RoboTwin click_bell Clean-1000 live progress",
            f"updated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"status: {status} | stage: {stage} | collector_pid: {pid or '-'} | GPU: 5",
            "",
            f"plans  {bar(planned, args.target)} | {rate_text}",
            f"replay {bar(materialized, args.target)} | {replay_text}",
            "",
            f"candidate attempts: {attempts} | failures: {failures} | "
            + (f"plan success: {success_rate:.2%}" if success_rate is not None else "plan success: calibrating"),
            f"dynamic ETA: {duration(eta_seconds)} | elapsed: {duration(now - started_at)} | free disk: {free:.1f} GiB",
            f"seed range: {args.start_seed}..{(max(seeds) if seeds else '-')} (evaluation starts at 100000)",
        ]
        atomic_write(output, "\n".join(lines) + "\n")

        if not alive and materialized >= args.target:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
