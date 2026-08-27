#!/usr/bin/env python3
"""Generate, convert, and validate a local RoboTwin dataset.

The default target is 500 successful Clean episodes for every official
RoboTwin task. Collection is resumable at both the expert-planning and HDF5
replay stages. Use ``--smoke`` before starting the full collection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import av
import h5py
import numpy as np
import pandas as pd
from PIL import Image

from starVLA.task_language import canonical_task_text

try:
    from .data_preparation import (
        CAMERA_MAP,
        TASKS,
        convert_extracted,
        read_instruction,
        summarize_numeric_chunks,
    )
except ImportError:  # Direct execution: python examples/Robotwin/generate_local_dataset.py
    from data_preparation import (
        CAMERA_MAP,
        TASKS,
        convert_extracted,
        read_instruction,
        summarize_numeric_chunks,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
MODALITY_FILE = REPO_ROOT / "examples" / "Robotwin" / "train_files" / "modality.json"
SPLIT_CONFIG = {"clean": "demo_clean", "randomized": "demo_randomized"}
SPLIT_DIR = {"clean": "Clean", "randomized": "Randomized"}
DEFAULT_TARGET = {"clean": 500, "randomized": 500}
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


def default_robotwin_root() -> Path:
    configured = os.environ.get("ROBOTWIN_PATH")
    candidates = [
        Path(configured).expanduser() if configured else None,
        REPO_ROOT.parent / "RoboTwin",
        REPO_ROOT / "thirdparty" / "RoboTwin",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "script" / "collect_data.py").is_file():
            return candidate.resolve()
    return (REPO_ROOT.parent / "RoboTwin").resolve()


def default_robotwin_python() -> Path:
    configured = os.environ.get("ROBOTWIN_PYTHON")
    candidates = [
        Path(configured).expanduser() if configured else None,
        REPO_ROOT.parent / ".venvs" / "RoboTwin" / "bin" / "python",
        REPO_ROOT / ".venv-robotwin" / "bin" / "python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            # Do not resolve the venv's ``bin/python`` symlink: invoking the
            # base interpreter target directly drops the virtual environment's
            # site-packages (including SAPIEN and cuRobo).
            return candidate.expanduser().absolute()
    return (REPO_ROOT.parent / ".venvs" / "RoboTwin" / "bin" / "python").absolute()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["status", "collect", "convert", "validate", "pipeline", "prefinalize"],
    )
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_CONFIG),
        default=["clean"],
    )
    parser.add_argument("--clean-target", type=int, default=DEFAULT_TARGET["clean"])
    parser.add_argument("--randomized-target", type=int, default=DEFAULT_TARGET["randomized"])
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--robotwin-root", type=Path, default=default_robotwin_root())
    parser.add_argument("--collector-python", type=Path, default=default_robotwin_python())
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
    parser.add_argument(
        "--finalize-workers",
        type=int,
        default=3,
        help="CPU worker processes used by prefinalize",
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
        "clean": 2 if args.smoke else args.clean_target,
        "randomized": 2 if args.smoke else args.randomized_target,
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


def collection_lock_available(run_dir: Path) -> bool:
    """Return whether the collector has completely released one raw job."""

    lock_path = run_dir / "collect.lock"
    if not lock_path.exists():
        return True
    with lock_path.open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(lock, fcntl.LOCK_UN)
    return True


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
    with (run_dir / "collect.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return job, False, "another collector holds this job lock"
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
        message = "collector did not run"
        for attempt in range(1, args.retries + 1):
            log_path = run_dir / "collect.log"
            with log_path.open("a", buffering=1) as log:
                log.write(
                    f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"attempt={attempt}/{args.retries} gpu={gpu} command={command!r}\n"
                )
                result = subprocess.run(
                    command,
                    cwd=args.robotwin_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode == 0 and raw_complete(run_dir, job.target):
                return job, True, f"complete on GPU {gpu}"
            planned, hdf5_count, instruction_count = raw_counts(run_dir)
            message = (
                f"exit={result.returncode}, planned={planned}, hdf5={hdf5_count}, "
                f"instructions={instruction_count}"
            )
            with (run_dir / "collect.log").open("a") as log:
                log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        return job, False, message


def collect(args: argparse.Namespace, raw_root: Path, selected_jobs: list[Job]) -> None:
    if not args.collector_python.exists():
        raise FileNotFoundError(f"RoboTwin Python not found: {args.collector_python}")
    if not (args.robotwin_root / "script" / "collect_data.py").is_file():
        raise FileNotFoundError(f"RoboTwin checkout not found: {args.robotwin_root}")
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
                validate_converted(destination, job.target, deep=False)
                validate_language_alignment(source, destination, job.target)
            except Exception as exc:
                print(f"[reconvert invalid] {job.split}/{job.task}: {exc}")
            else:
                print(f"[skip converted] {job.split}/{job.task}")
                continue
        print(f"[convert] {source} -> {destination}")
        convert_extracted(source, destination, MODALITY_FILE, task_name=job.task)


def prefinalize_one(raw_root: Path, output_root: Path, job: Job) -> str:
    """Convert one collector-complete job and run representative validation."""

    source = raw_run_dir(raw_root, job)
    destination = output_dir(output_root, job)
    if not raw_complete(source, job.target):
        raise RuntimeError(f"Raw collection became incomplete: {source}")
    if not collection_lock_available(source):
        raise RuntimeError(f"Collector still holds {source / 'collect.lock'}")

    if converted_complete(destination, job.target):
        try:
            validate_converted(destination, job.target, deep=False)
            validate_language_alignment(source, destination, job.target)
        except Exception:
            pass
        else:
            return "already converted and valid"

    convert_extracted(source, destination, MODALITY_FILE, task_name=job.task)
    validate_converted(destination, job.target, deep=False)
    validate_language_alignment(source, destination, job.target)
    return "converted and representative validation passed"


def prefinalize(
    args: argparse.Namespace,
    raw_root: Path,
    output_root: Path,
    selected_jobs: list[Job],
) -> None:
    """Convert completed jobs while GPU collectors continue on other tasks.

    This intentionally performs representative validation only. The main
    pipeline remains responsible for the final all-episode raw and converted
    deep audits before it writes ``pipeline.complete.json`` and synchronizes
    the dataset.
    """

    if args.poll_seconds < 10:
        raise ValueError("--poll-seconds must be at least 10")
    if args.finalize_workers < 1:
        raise ValueError("--finalize-workers must be positive")
    raw_root.mkdir(parents=True, exist_ok=True)
    lock_path = raw_root / "prefinalize.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another prefinalize process already holds {lock_path}") from exc

        retry_after: dict[Job, float] = {}
        active: dict[concurrent.futures.Future, Job] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.finalize_workers) as executor:
            while True:
                for future in list(active):
                    if not future.done():
                        continue
                    job = active.pop(future)
                    try:
                        message = future.result()
                    except Exception as exc:
                        retry_after[job] = time.monotonic() + args.poll_seconds
                        print(
                            f"[prefinalize-retry] {job.split}/{job.task}: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    else:
                        retry_after.pop(job, None)
                        print(f"[prefinalized] {job.split}/{job.task}: {message}", flush=True)

                converted_jobs = {
                    job for job in selected_jobs if converted_complete(output_dir(output_root, job), job.target)
                }
                if len(converted_jobs) == len(selected_jobs) and not active:
                    break

                active_jobs = set(active.values())
                available_slots = args.finalize_workers - len(active)
                now = time.monotonic()
                if available_slots > 0:
                    ready_jobs = [
                        job
                        for job in selected_jobs
                        if job not in converted_jobs
                        and job not in active_jobs
                        and retry_after.get(job, 0.0) <= now
                        and raw_complete(raw_run_dir(raw_root, job), job.target)
                        and collection_lock_available(raw_run_dir(raw_root, job))
                    ]
                    for job in ready_jobs[:available_slots]:
                        print(f"[prefinalize] submitting {job.split}/{job.task}", flush=True)
                        active[executor.submit(prefinalize_one, raw_root, output_root, job)] = job

                print(
                    f"[prefinalize] converted={len(converted_jobs)}/{len(selected_jobs)} "
                    f"active={[f'{job.split}/{job.task}' for job in active.values()]}",
                    flush=True,
                )
                time.sleep(args.poll_seconds)

        completion = {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output_root": str(output_root),
            "jobs": len(selected_jobs),
            "episodes": sum(job.target for job in selected_jobs),
            "validation": "representative; final deep validation remains mandatory",
        }
        (raw_root / "prefinalize.complete.json").write_text(json.dumps(completion, indent=2) + "\n")
        print(f"[prefinalize] complete: {completion}", flush=True)


def validate_hdf5(path: Path, *, deep: bool = False) -> str:
    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        action = np.asarray(handle["joint_action/vector"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 14 or action.shape[0] <= 0:
            raise ValueError(f"Invalid action shape {action.shape} in {path}")
        if not np.isfinite(action).all():
            raise ValueError(f"Non-finite action values in {path}")
        digest.update(action.tobytes(order="C"))
        length = action.shape[0]
        for camera in CAMERA_MAP.values():
            frames = handle[f"observation/{camera}/rgb"]
            if len(frames) != length:
                raise ValueError(f"Camera/action length mismatch in {path}: {camera}={len(frames)}, action={length}")
            frame_indices = range(length) if deep else sorted({0, length // 2, length - 1})
            for frame_index in frame_indices:
                encoded = bytes(frames[frame_index])
                if deep:
                    digest.update(encoded)
                with Image.open(io.BytesIO(encoded)) as image:
                    image.verify()
    return digest.hexdigest()


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
        duplicates = []
        try:
            fingerprints = {}
            for path in check_files:
                fingerprint = validate_hdf5(path, deep=args.deep)
                if args.deep and fingerprint in fingerprints:
                    duplicates.append(
                        {
                            "episode_index": int(path.stem.removeprefix("episode")),
                            "duplicate_of": fingerprints[fingerprint],
                            "fingerprint": fingerprint,
                            "reason": "exact action-and-image duplicate",
                        }
                    )
                elif args.deep:
                    fingerprints[fingerprint] = int(path.stem.removeprefix("episode"))
            instruction_indices = range(job.target) if args.deep else sorted({0, job.target // 2, job.target - 1})
            for index in instruction_indices:
                payload = json.loads((source / "instructions" / f"episode{index}.json").read_text())
                if not payload.get("seen") and not payload.get("unseen"):
                    raise ValueError(f"Empty instruction for episode {index}")
            audit_dir = source / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            with (audit_dir / "duplicate_episodes.jsonl").open("w", encoding="utf-8") as handle:
                for row in duplicates:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            with (audit_dir / "episode_blacklist.jsonl").open("w", encoding="utf-8") as handle:
                for row in duplicates:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            (audit_dir / "raw_validation.json").write_text(
                json.dumps(
                    {
                        "task": job.task,
                        "split": job.split,
                        "target_episodes": job.target,
                        "validated_episodes": len(check_files),
                        "deep": bool(args.deep),
                        "duplicate_episodes": len(duplicates),
                    },
                    indent=2,
                )
                + "\n"
            )
            if duplicates:
                raise ValueError(
                    f"Detected {len(duplicates)} exact duplicate episodes; "
                    f"see {audit_dir / 'episode_blacklist.jsonl'}"
                )
        except Exception as exc:
            errors.append(f"{job.split}/{job.task}: {exc}")
            continue
        print(f"[raw-valid] {job.split}/{job.task}: {job.target} episodes")
    if errors:
        raise RuntimeError("Raw validation failed:\n" + "\n".join(errors))


def validate_video(path: Path, expected_frames: int, expected_size: tuple[int, int]) -> None:
    decoded = 0
    with av.open(str(path), mode="r") as container:
        for frame in container.decode(video=0):
            if (frame.width, frame.height) != expected_size:
                raise ValueError(
                    f"Unexpected video frame size in {path}: "
                    f"{(frame.width, frame.height)} != {expected_size}"
                )
            decoded += 1
    if decoded != expected_frames:
        raise ValueError(f"Video frame count mismatch in {path}: {decoded} != {expected_frames}")


def validate_converted(path: Path, target: int, *, deep: bool = False) -> None:
    if not converted_complete(path, target):
        raise ValueError(f"Converted dataset incomplete: {path}")
    required_meta = (
        "info.json",
        "episodes.jsonl",
        "tasks.jsonl",
        "modality.json",
        "stats.json",
        "stats_gr00t.json",
        "task_language/robotwin_task_language_audit.jsonl",
        "audit/episode_blacklist.jsonl",
        "audit/duplicate_episodes.jsonl",
        "audit/conversion_audit.json",
    )
    for relative in required_meta:
        if not (path / "meta" / relative).is_file():
            raise ValueError(f"Missing required metadata: {path / 'meta' / relative}")
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
    expected_task = canonical_task_text(path.name)
    if tasks != [{"task_index": 0, "task": expected_task}] or int(info.get("total_tasks", -1)) != 1:
        raise ValueError(f"RoboTwin dataset must contain one canonical task ID in {path}: {tasks}")
    language_audit = [
        json.loads(line)
        for line in (path / "meta" / "task_language" / "robotwin_task_language_audit.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(language_audit) != target or any(
        row.get("canonical_description") != expected_task or not row.get("included")
        for row in language_audit
    ):
        raise ValueError(f"Invalid task-language audit in {path}")
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
    plain_stats = json.loads((path / "meta" / "stats.json").read_text())
    if plain_stats != statistics:
        raise ValueError(f"stats.json and stats_gr00t.json disagree in {path}")
    check_indices = range(target) if deep else sorted({0, target // 2, target - 1})
    numeric_chunks = {"observation.state": [], "action": []}
    expected_width = int(info["features"][next(iter(CAMERA_MAP))]["shape"][1])
    expected_height = int(info["features"][next(iter(CAMERA_MAP))]["shape"][0])
    for index in check_indices:
        parquet_path = path / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"
        frame = pd.read_parquet(parquet_path)
        expected_length = int(episodes[index]["length"])
        if len(frame) != expected_length or frame.empty or not np.all(frame["episode_index"].to_numpy() == index):
            raise ValueError(f"Invalid converted episode in {parquet_path}")
        if not np.array_equal(frame["frame_index"].to_numpy(), np.arange(expected_length)):
            raise ValueError(f"Invalid frame_index in {parquet_path}")
        if not np.all(frame["task_index"].to_numpy() == 0):
            raise ValueError(f"Out-of-range task_index in {parquet_path}")
        expected_timestamp = np.arange(expected_length, dtype=np.float32) / 30.0
        if not np.allclose(frame["timestamp"].to_numpy(), expected_timestamp, atol=1e-6):
            raise ValueError(f"Invalid timestamps in {parquet_path}")
        state = np.stack(frame["observation.state"].to_numpy())
        action = np.stack(frame["action"].to_numpy())
        if state.ndim != 2 or state.shape[1] != 14 or action.shape != state.shape:
            raise ValueError(f"Invalid state/action shapes in {parquet_path}: {state.shape}, {action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"Non-finite converted values in {parquet_path}")
        numeric_chunks["observation.state"].append(state)
        numeric_chunks["action"].append(action)
        for video_key in CAMERA_MAP:
            video_path = (
                path
                / "videos"
                / f"chunk-{index // 1000:03d}"
                / video_key
                / f"episode_{index:06d}.mp4"
            )
            if not video_path.is_file():
                raise ValueError(f"Missing converted video: {video_path}")
            validate_video(video_path, expected_length, (expected_width, expected_height))
    if deep:
        for key, chunks in numeric_chunks.items():
            recomputed = summarize_numeric_chunks(chunks)
            for statistic, expected_values in recomputed.items():
                if not np.allclose(
                    np.asarray(statistics[key][statistic], dtype=np.float64),
                    np.asarray(expected_values, dtype=np.float64),
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise ValueError(f"Statistics mismatch for {key}/{statistic} in {path}")
        validation_audit = {
            "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "episodes": target,
            "parquets": target,
            "videos": target * len(CAMERA_MAP),
            "deep": True,
            "duplicates": 0,
            "blacklisted": 0,
            "canonical_task": expected_task,
        }
        (path / "meta" / "audit" / "validation_audit.json").write_text(
            json.dumps(validation_audit, indent=2) + "\n"
        )


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
    canonical = canonical_task_text(destination.name)
    audit_rows = [
        json.loads(line)
        for line in (
            destination / "meta" / "task_language" / "robotwin_task_language_audit.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    if len(audit_rows) != target:
        raise ValueError(f"Task-language audit count mismatch in {destination}")
    check_indices = range(target) if indices is None else indices
    for index in check_indices:
        instruction = read_instruction(source / "instructions" / f"episode{index}.json")
        if not instruction:
            raise ValueError(f"Empty raw instruction for episode {index} in {source}")
        if episodes[index].get("tasks") != [canonical]:
            raise ValueError(
                f"Canonical instruction mismatch for episode {index}: "
                f"expected={canonical!r}, converted={episodes[index].get('tasks')!r}"
            )
        audit = audit_rows[index]
        if (
            audit.get("episode_index") != index
            or audit.get("raw_description") != instruction
            or audit.get("canonical_description") != canonical
        ):
            raise ValueError(f"Task-language audit mismatch for episode {index}")


def validate(
    args: argparse.Namespace,
    raw_root: Path,
    output_root: Path,
    selected_jobs: list[Job],
    *,
    include_raw: bool = True,
) -> None:
    if include_raw:
        validate_raw(args, raw_root, selected_jobs)
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
        try:
            instruction_indices = range(job.target) if args.deep else sorted({0, job.target // 2, job.target - 1})
            validate_converted(destination, job.target, deep=args.deep)
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
                validate(args, raw_root, output_root, selected_jobs, include_raw=False)
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
    elif args.command == "prefinalize":
        prefinalize(args, raw_root, output_root, selected_jobs)
    else:
        pipeline(args, raw_root, output_root, selected_jobs)


if __name__ == "__main__":
    main()
