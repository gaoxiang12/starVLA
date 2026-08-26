#!/usr/bin/env python3
"""Generate, convert, and validate a large local RoboTwin dataset.

The default target is 2,000 Clean and 1,000 Randomized episodes for every
official RoboTwin task. Collection is resumable at both the expert-planning
and HDF5 replay stages. Use ``--smoke`` before starting the full collection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import io
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from data_preparation import CAMERA_MAP, TASKS, convert_extracted, read_instruction
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = REPO_ROOT / "thirdparty" / "RoboTwin"
LOCAL_ROBOTWIN_PYTHON = REPO_ROOT / ".venv-robotwin" / "bin" / "python"
LEGACY_ROBOTWIN_PYTHON = Path("/home/zskj/data/miniconda3/envs/robotwin/bin/python")
ROBOTWIN_PYTHON = LOCAL_ROBOTWIN_PYTHON if LOCAL_ROBOTWIN_PYTHON.exists() else LEGACY_ROBOTWIN_PYTHON
MODALITY_FILE = REPO_ROOT / "examples" / "Robotwin" / "train_files" / "modality.json"
SPLIT_CONFIG = {"clean": "demo_clean", "randomized": "demo_randomized"}
SPLIT_DIR = {"clean": "Clean", "randomized": "Randomized"}
DEFAULT_TARGET = {"clean": 2000, "randomized": 1000}
START_SEED = {"clean": 20_000_000, "randomized": 40_000_000}


@dataclass(frozen=True)
class Job:
    task: str
    split: str
    target: int
    task_index: int

    @property
    def config_name(self) -> str:
        return SPLIT_CONFIG[self.split]

    @property
    def split_dir(self) -> str:
        return SPLIT_DIR[self.split]


def default_data_root() -> Path:
    configured = os.environ.get("ROBOTWIN_GENERATED_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    stable_local_root = Path("/home/gaoxiang/data/gaoxiang")
    if stable_local_root.exists():
        return stable_local_root.resolve()
    return Path("/data/gaoxiang").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "collect", "convert", "validate", "pipeline"])
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_CONFIG),
        default=["clean", "randomized"],
    )
    parser.add_argument("--clean-target", type=int, default=DEFAULT_TARGET["clean"])
    parser.add_argument("--randomized-target", type=int, default=DEFAULT_TARGET["randomized"])
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--collector-python", type=Path, default=ROBOTWIN_PYTHON)
    parser.add_argument("--language-num", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--replay-retries",
        type=int,
        default=20,
        help="Attempts for one planned episode before restarting its collector",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="Polling interval while pipeline waits for an existing collector",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Open and validate every raw HDF5 episode instead of representative samples",
    )
    return parser.parse_args()


def selected_tasks(values: list[str]) -> list[str]:
    if "all" in values:
        return list(TASKS)
    unknown = sorted(set(values) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown RoboTwin tasks: {', '.join(unknown)}")
    return values


def roots(args: argparse.Namespace) -> tuple[Path, Path]:
    suffix = "_smoke" if args.smoke else ""
    raw = args.raw_root or args.data_root / f"RoboTwinGenerated{suffix}_raw"
    output = args.output_root or args.data_root / f"RoboTwinGenerated{suffix}"
    return raw.expanduser().resolve(), output.expanduser().resolve()


def jobs(args: argparse.Namespace) -> list[Job]:
    targets = {
        "clean": 1 if args.smoke else args.clean_target,
        "randomized": 1 if args.smoke else args.randomized_target,
    }
    result = []
    for task in selected_tasks(args.tasks):
        task_index = TASKS.index(task)
        for split in args.splits:
            if targets[split] <= 0:
                raise ValueError(f"Target for {split} must be positive")
            result.append(Job(task, split, targets[split], task_index))
    return result


def raw_run_dir(raw_root: Path, job: Job) -> Path:
    return raw_root / job.split_dir / job.task / job.config_name


def output_dir(output_root: Path, job: Job) -> Path:
    return output_root / job.split_dir / job.task


def indexed_files(path: Path, suffix: str) -> set[int]:
    result = set()
    for item in path.glob(f"episode*{suffix}"):
        value = item.name.removeprefix("episode").removesuffix(suffix)
        if value.isdigit():
            result.add(int(value))
    return result


def raw_counts(run_dir: Path) -> tuple[int, int, int]:
    try:
        seeds = [int(value) for value in (run_dir / "seed.txt").read_text().split()]
    except (FileNotFoundError, ValueError):
        seeds = []
    hdf5_count = len(indexed_files(run_dir / "data", ".hdf5"))
    instruction_count = len(indexed_files(run_dir / "instructions", ".json"))
    return len(seeds), hdf5_count, instruction_count


def raw_complete(run_dir: Path, target: int) -> bool:
    planned, hdf5_count, instruction_count = raw_counts(run_dir)
    expected = set(range(target))
    return (
        planned == target
        and indexed_files(run_dir / "_traj_data", ".pkl") == expected
        and indexed_files(run_dir / "data", ".hdf5") == expected
        and indexed_files(run_dir / "instructions", ".json") == expected
        and hdf5_count == target
        and instruction_count == target
    )


def collection_command(args: argparse.Namespace, raw_root: Path, job: Job) -> list[str]:
    start_seed = START_SEED[job.split] + job.task_index * 100_000
    return [
        str(args.collector_python),
        "script/collect_data.py",
        job.task,
        job.config_name,
        "--episode-num",
        str(job.target),
        "--save-root",
        str(raw_root / job.split_dir),
        "--start-seed",
        str(start_seed),
        "--language-num",
        str(args.language_num),
        "--replay-retries",
        str(args.replay_retries),
    ]


def collect_one(args: argparse.Namespace, raw_root: Path, job: Job, gpu: str) -> tuple[Job, bool, str]:
    run_dir = raw_run_dir(raw_root, job)
    run_dir.mkdir(parents=True, exist_ok=True)
    if raw_complete(run_dir, job.target):
        return job, True, "already complete"

    env = dict(os.environ)
    runtime_dir = raw_root / "_runtime" / f"gpu{gpu}"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "MPLCONFIGDIR": str(runtime_dir / "matplotlib"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": str(runtime_dir),
            "XDG_CACHE_HOME": str(runtime_dir / "cache"),
        }
    )
    command = collection_command(args, raw_root, job)
    for attempt in range(1, args.retries + 1):
        log_path = run_dir / "collect.log"
        with log_path.open("a", buffering=1) as log:
            log.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"attempt={attempt}/{args.retries} gpu={gpu} command={command!r}\n"
            )
            result = subprocess.run(
                command,
                cwd=ROBOTWIN_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0 and raw_complete(run_dir, job.target):
            return job, True, f"complete on GPU {gpu}"
        planned, hdf5_count, instruction_count = raw_counts(run_dir)
        message = f"exit={result.returncode}, planned={planned}, hdf5={hdf5_count}, " f"instructions={instruction_count}"
        with (run_dir / "collect.log").open("a") as log:
            log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    return job, False, message


def collect(args: argparse.Namespace, raw_root: Path, selected_jobs: list[Job]) -> None:
    if not args.collector_python.exists():
        raise FileNotFoundError(f"RoboTwin Python not found: {args.collector_python}")
    gpu_check = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if gpu_check.returncode != 0:
        raise RuntimeError(
            "NVIDIA devices are unavailable; restore the driver before collection: "
            + (gpu_check.stderr.strip() or gpu_check.stdout.strip())
        )
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "supervisor.pid").write_text(f"{os.getpid()}\n")
    partitions = [[] for _ in gpus]
    for index, job in enumerate(selected_jobs):
        partitions[index % len(gpus)].append(job)

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []
        for gpu, partition in zip(gpus, partitions, strict=True):
            futures.append(executor.submit(collect_partition, args, raw_root, partition, gpu))
        for future in concurrent.futures.as_completed(futures):
            failures.extend(future.result())
    if failures:
        details = "; ".join(f"{job.split}/{job.task}: {message}" for job, message in failures)
        raise RuntimeError(f"Collection incomplete: {details}")


def collect_partition(args: argparse.Namespace, raw_root: Path, partition: list[Job], gpu: str) -> list[tuple[Job, str]]:
    failures = []
    for job in partition:
        finished_job, ok, message = collect_one(args, raw_root, job, gpu)
        print(f"[{finished_job.split}/{finished_job.task}] {message}", flush=True)
        if not ok:
            failures.append((finished_job, message))
    return failures


def converted_episode_layout(path: Path) -> tuple[set[int], int, bool]:
    indices = set()
    file_count = 0
    layout_valid = True
    for parquet_path in (path / "data").glob("chunk-*/episode_*.parquet"):
        file_count += 1
        value = parquet_path.stem.removeprefix("episode_")
        if not value.isdigit():
            layout_valid = False
            continue
        index = int(value)
        indices.add(index)
        expected_chunk = f"chunk-{index // 1000:03d}"
        if parquet_path.parent.name != expected_chunk:
            layout_valid = False
    return indices, file_count, layout_valid


def converted_complete(path: Path, target: int) -> bool:
    try:
        info = json.loads((path / "meta" / "info.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    indices, file_count, layout_valid = converted_episode_layout(path)
    return (
        int(info.get("total_episodes", -1)) == target
        and file_count == target
        and indices == set(range(target))
        and layout_valid
    )


def convert(args: argparse.Namespace, raw_root: Path, output_root: Path, selected_jobs: list[Job]) -> None:
    for job in selected_jobs:
        source = raw_run_dir(raw_root, job)
        destination = output_dir(output_root, job)
        if not raw_complete(source, job.target):
            raise RuntimeError(f"Raw collection is incomplete: {source}")
        if converted_complete(destination, job.target):
            try:
                validate_converted(destination, job.target)
                validate_language_alignment(source, destination, job.target)
            except Exception as exc:
                print(f"[reconvert invalid] {job.split}/{job.task}: {exc}")
            else:
                print(f"[skip converted] {job.split}/{job.task}")
                continue
        print(f"[convert] {source} -> {destination}")
        convert_extracted(source, destination, MODALITY_FILE)


def validate_hdf5(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        action = np.asarray(handle["joint_action/vector"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 14 or action.shape[0] <= 0:
            raise ValueError(f"Invalid action shape {action.shape} in {path}")
        if not np.isfinite(action).all():
            raise ValueError(f"Non-finite action values in {path}")
        length = action.shape[0]
        for camera in CAMERA_MAP.values():
            frames = handle[f"observation/{camera}/rgb"]
            if len(frames) != length:
                raise ValueError(f"Camera/action length mismatch in {path}: {camera}={len(frames)}, action={length}")
            for frame_index in sorted({0, length // 2, length - 1}):
                with Image.open(io.BytesIO(bytes(frames[frame_index]))) as image:
                    image.verify()


def validate_raw(args: argparse.Namespace, raw_root: Path, selected_jobs: list[Job]) -> None:
    errors = []
    for job in selected_jobs:
        source = raw_run_dir(raw_root, job)
        planned, hdf5_count, instruction_count = raw_counts(source)
        if not raw_complete(source, job.target):
            errors.append(
                f"{job.split}/{job.task}: raw planned={planned}, hdf5={hdf5_count}, "
                f"instructions={instruction_count}, target={job.target}"
            )
            continue
        raw_files = sorted(
            (source / "data").glob("episode*.hdf5"),
            key=lambda path: int(path.stem.removeprefix("episode")),
        )
        check_files = raw_files if args.deep else sorted({raw_files[0], raw_files[len(raw_files) // 2], raw_files[-1]})
        try:
            for path in check_files:
                validate_hdf5(path)
            instruction_indices = range(job.target) if args.deep else sorted({0, job.target // 2, job.target - 1})
            for index in instruction_indices:
                payload = json.loads((source / "instructions" / f"episode{index}.json").read_text())
                if not payload.get("seen") and not payload.get("unseen"):
                    raise ValueError(f"Empty instruction for episode {index}")
        except Exception as exc:
            errors.append(f"{job.split}/{job.task}: {exc}")
            continue
        print(f"[raw-valid] {job.split}/{job.task}: {job.target} episodes")
    if errors:
        raise RuntimeError("Raw validation failed:\n" + "\n".join(errors))


def validate_converted(path: Path, target: int) -> None:
    if not converted_complete(path, target):
        raise ValueError(f"Converted dataset incomplete: {path}")
    info = json.loads((path / "meta" / "info.json").read_text())
    if int(info["total_chunks"]) != (target + 999) // 1000:
        raise ValueError(f"Incorrect chunk count in {path}")
    episodes = [json.loads(line) for line in (path / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    if (
        len(episodes) != target
        or [row.get("episode_index") for row in episodes] != list(range(target))
        or any(int(row.get("length", 0)) <= 0 for row in episodes)
    ):
        raise ValueError(f"Invalid converted episode metadata in {path}")
    if int(info.get("total_frames", -1)) != sum(int(row["length"]) for row in episodes):
        raise ValueError(f"Converted total frame count mismatch in {path}")
    tasks = [json.loads(line) for line in (path / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip()]
    if not tasks or any(not str(row.get("task", "")).strip() for row in tasks):
        raise ValueError(f"Missing converted task language in {path}")
    stats = json.loads((path / "meta" / "stats_gr00t.json").read_text())
    if stats.get("__format_version") != 2 or stats.get("__cache_config") != {"mode": "abs"}:
        raise ValueError(f"Invalid statistics cache header in {path}")
    statistics = stats.get("statistics", {})
    for key in ("observation.state", "action"):
        key_stats = statistics.get(key, {})
        for statistic in ("mean", "std", "min", "max", "q01", "q99"):
            values = np.asarray(key_stats.get(statistic, []), dtype=np.float32)
            if values.shape != (14,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key}/{statistic} statistics in {path}: {values.shape}")
    for index in sorted({0, target // 2, target - 1}):
        parquet_path = path / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"
        frame = pd.read_parquet(
            parquet_path,
            columns=["observation.state", "action", "episode_index"],
        )
        if frame.empty or int(frame["episode_index"].iloc[0]) != index:
            raise ValueError(f"Invalid converted episode in {parquet_path}")
        state = np.stack(frame["observation.state"].to_numpy())
        action = np.stack(frame["action"].to_numpy())
        if state.ndim != 2 or state.shape[1] != 14 or action.shape != state.shape:
            raise ValueError(f"Invalid state/action shapes in {parquet_path}: {state.shape}, {action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"Non-finite converted values in {parquet_path}")


def validate_language_alignment(
    source: Path,
    destination: Path,
    target: int,
    indices: range | list[int] | None = None,
) -> None:
    episodes = [
        json.loads(line) for line in (destination / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()
    ]
    if len(episodes) != target:
        raise ValueError(f"Converted episode metadata count mismatch in {destination}")
    check_indices = range(target) if indices is None else indices
    for index in check_indices:
        instruction = read_instruction(source / "instructions" / f"episode{index}.json")
        if not instruction:
            raise ValueError(f"Empty raw instruction for episode {index} in {source}")
        if episodes[index].get("tasks") != [instruction]:
            raise ValueError(
                f"Instruction mismatch for episode {index}: "
                f"raw={instruction!r}, converted={episodes[index].get('tasks')!r}"
            )


def validate(args: argparse.Namespace, raw_root: Path, output_root: Path, selected_jobs: list[Job]) -> None:
    errors = []
    for job in selected_jobs:
        source = raw_run_dir(raw_root, job)
        destination = output_dir(output_root, job)
        planned, hdf5_count, instruction_count = raw_counts(source)
        if not raw_complete(source, job.target):
            errors.append(
                f"{job.split}/{job.task}: raw planned={planned}, hdf5={hdf5_count}, "
                f"instructions={instruction_count}, target={job.target}"
            )
            continue
        raw_files = sorted((source / "data").glob("episode*.hdf5"))
        check_files = raw_files if args.deep else sorted({raw_files[0], raw_files[len(raw_files) // 2], raw_files[-1]})
        try:
            for path in check_files:
                validate_hdf5(path)
            instruction_indices = range(job.target) if args.deep else sorted({0, job.target // 2, job.target - 1})
            for index in instruction_indices:
                payload = json.loads((source / "instructions" / f"episode{index}.json").read_text())
                if not payload.get("seen") and not payload.get("unseen"):
                    raise ValueError(f"Empty instruction for episode {index}")
            validate_converted(destination, job.target)
            validate_language_alignment(source, destination, job.target, instruction_indices)
        except Exception as exc:
            errors.append(f"{job.split}/{job.task}: {exc}")
            continue
        print(f"[valid] {job.split}/{job.task}: {job.target} episodes")
    if errors:
        raise RuntimeError("Validation failed:\n" + "\n".join(errors))


def status(raw_root: Path, output_root: Path, selected_jobs: list[Job]) -> None:
    total_target = sum(job.target for job in selected_jobs)
    total_planned = 0
    total_hdf5 = 0
    total_instructions = 0
    total_converted = 0
    for job in selected_jobs:
        planned, hdf5_count, instruction_count = raw_counts(raw_run_dir(raw_root, job))
        converted = converted_complete(output_dir(output_root, job), job.target)
        total_planned += min(planned, job.target)
        total_hdf5 += min(hdf5_count, job.target)
        total_instructions += min(instruction_count, job.target)
        total_converted += job.target if converted else 0
        print(
            f"{job.split:10s} {job.task:32s} "
            f"plan={planned:4d}/{job.target} hdf5={hdf5_count:4d}/{job.target} "
            f"lang={instruction_count:4d}/{job.target} converted={'yes' if converted else 'no'}"
        )
    usage = shutil.disk_usage(raw_root.parent if raw_root.parent.exists() else Path("/"))
    print(
        f"TOTAL plan={total_planned}/{total_target} "
        f"hdf5={total_hdf5}/{total_target} "
        f"lang={total_instructions}/{total_target} "
        f"converted={total_converted}/{total_target} "
        f"free={usage.free / 1024**3:.1f} GiB"
    )


def collection_processes_running(raw_root: Path) -> list[int]:
    """Return live collector PIDs writing beneath ``raw_root``.

    Looking at child collectors as well as the supervisor avoids launching a
    duplicate collection if the supervisor dies while a replay subprocess is
    still shutting down.
    """

    result = []
    raw_root_text = str(raw_root)
    for command_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = command_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "script/collect_data.py" in command and raw_root_text in command:
            result.append(int(command_path.parent.name))
    return sorted(result)


def raw_progress(raw_root: Path, selected_jobs: list[Job]) -> tuple[int, int, int]:
    planned = hdf5_count = instruction_count = 0
    for job in selected_jobs:
        job_counts = raw_counts(raw_run_dir(raw_root, job))
        planned += min(job_counts[0], job.target)
        hdf5_count += min(job_counts[1], job.target)
        instruction_count += min(job_counts[2], job.target)
    return planned, hdf5_count, instruction_count


def pipeline(
    args: argparse.Namespace,
    raw_root: Path,
    output_root: Path,
    selected_jobs: list[Job],
) -> None:
    """Keep collection alive, then convert and validate the complete dataset."""

    if args.poll_seconds < 10:
        raise ValueError("--poll-seconds must be at least 10")
    raw_root.mkdir(parents=True, exist_ok=True)
    lock_path = raw_root / "pipeline.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another pipeline already holds {lock_path}") from exc

        (raw_root / "pipeline.pid").write_text(f"{os.getpid()}\n")
        total_target = sum(job.target for job in selected_jobs)
        while not all(raw_complete(raw_run_dir(raw_root, job), job.target) for job in selected_jobs):
            collector_pids = collection_processes_running(raw_root)
            planned, hdf5_count, instruction_count = raw_progress(raw_root, selected_jobs)
            print(
                f"[pipeline] raw plan={planned}/{total_target} "
                f"hdf5={hdf5_count}/{total_target} "
                f"lang={instruction_count}/{total_target} "
                f"collector_pids={collector_pids}",
                flush=True,
            )
            if collector_pids:
                time.sleep(args.poll_seconds)
                continue

            print("[pipeline] collector stopped before completion; resuming", flush=True)
            try:
                collect(args, raw_root, selected_jobs)
            except Exception as exc:
                print(
                    f"[pipeline] collection attempt failed ({type(exc).__name__}: {exc}); "
                    f"retrying in {args.poll_seconds}s",
                    flush=True,
                )
                time.sleep(args.poll_seconds)

        while True:
            try:
                print("[pipeline] raw collection complete; auditing", flush=True)
                validate_raw(args, raw_root, selected_jobs)
                print("[pipeline] raw collection complete; converting", flush=True)
                convert(args, raw_root, output_root, selected_jobs)
                print("[pipeline] conversion complete; validating", flush=True)
                validate(args, raw_root, output_root, selected_jobs)
            except Exception as exc:
                print(
                    f"[pipeline] finalization failed ({type(exc).__name__}: {exc}); "
                    f"retrying in {args.poll_seconds}s",
                    flush=True,
                )
                time.sleep(args.poll_seconds)
                continue
            break
        completion = {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "raw_root": str(raw_root),
            "output_root": str(output_root),
            "jobs": len(selected_jobs),
            "episodes": total_target,
            "deep_validation": bool(args.deep),
        }
        (raw_root / "pipeline.complete.json").write_text(json.dumps(completion, indent=2) + "\n")
        print(f"[pipeline] complete: {completion}", flush=True)


def main() -> None:
    args = parse_args()
    raw_root, output_root = roots(args)
    selected_jobs = jobs(args)
    print(f"raw_root={raw_root}")
    print(f"output_root={output_root}")
    print(f"jobs={len(selected_jobs)} target_episodes={sum(job.target for job in selected_jobs)}")
    if args.command == "status":
        status(raw_root, output_root, selected_jobs)
    elif args.command == "collect":
        collect(args, raw_root, selected_jobs)
    elif args.command == "convert":
        convert(args, raw_root, output_root, selected_jobs)
    elif args.command == "validate":
        validate(args, raw_root, output_root, selected_jobs)
    else:
        pipeline(args, raw_root, output_root, selected_jobs)


if __name__ == "__main__":
    main()
