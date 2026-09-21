"""Retry a terminal Vulkan failure without mixing partial benchmark results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run_with_retry(command, env, metrics_path, log_root, attempts=2, runner=subprocess.run):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    metrics_path, log_root = Path(metrics_path), Path(log_root)
    if metrics_path.exists() or metrics_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing metrics: {metrics_path}")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    history = []
    for attempt in range(1, attempts + 1):
        attempt_logs = log_root / f"attempt_{attempt}"
        attempt_logs.mkdir(exist_ok=False)
        attempt_metrics = metrics_path.with_name(f"{metrics_path.stem}.attempt_{attempt}{metrics_path.suffix}")
        # Point progress readers at the current attempt, retaining previous data.
        if metrics_path.is_symlink():
            metrics_path.unlink()
        metrics_path.symlink_to(attempt_metrics.name)
        child_env = dict(env, ROBOTWIN_LOG_ROOT=str(attempt_logs),
                         ROBOTWIN_RANKING_METRICS_PATH=str(attempt_metrics))
        result = runner(command, env=child_env)
        vulkan_failure = False
        if result.returncode:
            vulkan_failure = any(
                "ErrorDeviceLost" in path.read_text(errors="replace")
                or "vk::DeviceLostError" in path.read_text(errors="replace")
                for path in attempt_logs.rglob("*_eval.log")
            )
        history.append(dict(attempt=attempt, returncode=result.returncode, vulkan_failure=vulkan_failure,
                            logs=str(attempt_logs), metrics=str(attempt_metrics)))
        temporary = log_root / "attempts.tmp"
        temporary.write_text(json.dumps(history, indent=2) + "\n")
        temporary.replace(log_root / "attempts.json")
        if result.returncode == 0 or not vulkan_failure or attempt == attempts:
            return result.returncode
        print(f"[RETRY] Vulkan device lost in attempt {attempt}; preserving partial results and repeating the same seed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("command required after --")
    return run_with_retry(command, os.environ, args.metrics, args.log_root, args.attempts)


if __name__ == "__main__":
    sys.exit(main())
