"""Read-only, standard-library probe, streamed to nodes over SSH."""

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def nvidia_query(kind, fields):
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=8, check=True,
    )
    return [dict(zip(fields, row)) for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True)]


def read_records(path, limit=600):
    # Bound IO even for multi-gigabyte runs; never consume an unfinished write.
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - 2 * 1024 * 1024))
        if stream.tell():
            stream.readline()
        data = stream.read()
    records = []
    errors = 0
    for line in data[:data.rfind(b"\n") + 1].splitlines():
        try:
            row = json.loads(line)
            if (not isinstance(row, dict) or not isinstance(row.get("step"), (int, float))
                    or isinstance(row.get("step"), bool) or number(row.get("step")) is None):
                raise ValueError("missing step")
            records.append({key: value for key, value in row.items()
                            if isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value)})
        except (ValueError, UnicodeError):
            errors += 1
    return records[-limit:], errors


def collect_history(directory, offset=0, identity=None, chunk_size=2 * 1024 * 1024):
    """Read full-history pages, preserving incomplete lines for the next poll."""
    path = Path(directory) / "metrics.jsonl"
    with path.open("rb") as stream:
        stat = os.fstat(stream.fileno())
        prefix = stream.readline(65536)
        current_identity = f"{stat.st_dev}:{stat.st_ino}:{hashlib.sha256(prefix).hexdigest()}"
        reset = identity != current_identity or offset < 0 or offset > stat.st_size
        start = 0 if reset else offset
        stream.seek(start)
        data = stream.read(chunk_size)
    end = data.rfind(b"\n") + 1
    if not end and len(data) == chunk_size:
        raise ValueError("A metrics line exceeds the history page size")
    records, errors = [], 0
    for line in data[:end].splitlines():
        try:
            row = json.loads(line)
            step = row.get("step") if isinstance(row, dict) else None
            if not isinstance(step, (int, float)) or isinstance(step, bool) or number(step) is None:
                raise ValueError("missing step")
            records.append({key: value for key, value in row.items()
                            if (key == "step" or re.search("loss|mse|error", key, re.I))
                            and isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value)})
        except (ValueError, UnicodeError):
            errors += 1
    return {"records": records, "offset": start + end, "identity": current_identity,
            "reset": reset, "done": start + len(data) >= stat.st_size,
            "parse_errors": errors}


def collect_run(path):
    records, errors = read_records(path)
    directory = path.parent
    target = None
    for filename in ("config.yaml", "config.full.yaml", "launch_config.yaml"):
        try:
            match = re.search(r"^\s*max_train_steps\s*:\s*(\d+)", (directory / filename).read_text(), re.M)
            if match:
                target = int(match[1])
                break
        except OSError:
            pass
    pid = None
    alive = False
    try:
        pid = int((directory / "train.pid").read_text().strip())
        if pid > 0:
            os.kill(pid, 0)
            alive = True
    except PermissionError:
        alive = True
    except (OSError, ValueError):
        pass
    state = "unknown"
    for marker in ("complete", "failed", "stopped", "running"):
        if (directory / f"STATUS.{marker}").exists():
            state = "stale" if marker == "running" else marker
            break
    if alive:
        state = "running"
    step = records[-1]["step"] if records else None
    # Reaching a target alone does not prove that final saving completed.
    if state == "unknown" and target and step is not None and step >= target:
        state = "target_reached"
    return {"id": str(directory), "name": directory.name, "state": state, "pid": pid,
            "step": step, "target": target, "updated_at": path.stat().st_mtime,
            "records": records, "parse_errors": errors}


def cpu_ticks():
    ticks = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
    return sum(ticks), ticks[3] + ticks[4]


def collect(paths):
    warnings = []
    gpus = []
    try:
        fields = ["index", "uuid", "name", "utilization.gpu", "memory.used", "memory.total",
                  "temperature.gpu", "power.draw", "power.limit"]
        for row in nvidia_query("gpu", fields):
            gpus.append({"index": row["index"], "uuid": row["uuid"], "name": row["name"],
                         **{key: number(row[key]) for key in fields[3:]}})
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.append(f"GPU 采集失败: {exc}")
    processes = []
    try:
        processes = nvidia_query("compute-apps", ["gpu_uuid", "pid", "process_name", "used_gpu_memory"])
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.append(f"GPU 进程采集失败: {exc}")
    resources = {}
    try:
        before, idle_before = cpu_ticks()
        time.sleep(0.15)
        after, idle_after = cpu_ticks()
        resources["cpu_percent"] = 100 * (1 - (idle_after - idle_before) / max(1, after - before))
        memory = {line.split(':')[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
        resources.update(memory_total=memory["MemTotal"] * 1024,
                         memory_used=(memory["MemTotal"] - memory["MemAvailable"]) * 1024,
                         load=os.getloadavg()[0], cpu_count=os.cpu_count())
    except (OSError, ValueError, KeyError) as exc:
        warnings.append(f"系统资源采集失败: {exc}")
    files = set()
    for source in paths:
        root = Path(source).expanduser()
        if not root.exists():
            warnings.append(f"训练目录不存在: {root}")
            continue
        if root.is_file():
            files.add(root)
            continue
        # Do not walk model weights/datasets or follow directory symlinks.
        for base, directories, names in os.walk(root):
            directories[:] = [d for d in directories if d not in {"checkpoints", "final_model", ".git", ".venv"}]
            if "metrics.jsonl" in names:
                files.add(Path(base) / "metrics.jsonl")
    runs = []
    available = []
    for path in files:
        try:
            available.append((path.stat().st_mtime, path))
        except OSError:
            pass
    for _, path in sorted(available, reverse=True)[:30]:
        try:
            runs.append(collect_run(path))
        except (OSError, ValueError) as exc:
            warnings.append(f"读取 {path} 失败: {exc}")
    return {"hostname": socket.gethostname(), "sampled_at": time.time(), "gpus": gpus,
            "processes": processes, "resources": resources, "runs": runs, "warnings": warnings,
            "run_count": len(available)}
