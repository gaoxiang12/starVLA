#!/usr/bin/env python3
"""Serve a live, dependency-free dashboard for starVLA ``metrics.jsonl`` files.

The input may be one metrics file, one run directory, or a directory containing
many runs. In the last case the web UI provides a run selector.

Examples:
    python scripts/training_dashboard.py playground/Checkpoints
    python scripts/training_dashboard.py playground/Checkpoints/my_run --port 6007
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import re
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

MAX_TRAIN_STEPS_RE = re.compile(r"^\s*max_train_steps\s*:\s*(\d+)\s*(?:#.*)?$", re.MULTILINE)
STATUS_PRIORITY = ("complete", "failed", "stopped", "running")


def is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def pid_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def read_pid(run_dir: Path) -> int | None:
    try:
        return int((run_dir / "train.pid").read_text(encoding="utf-8").strip().splitlines()[0])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def read_max_train_steps(run_dir: Path) -> int | None:
    for name in ("config.yaml", "config.full.yaml", "launch_config.yaml"):
        try:
            match = MAX_TRAIN_STEPS_RE.search((run_dir / name).read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError):
            continue
        if match:
            return int(match.group(1))
    return None


def run_status(run_dir: Path, metrics_mtime: float) -> dict[str, Any]:
    pid = read_pid(run_dir)
    alive = pid_is_alive(pid)
    markers: dict[str, Path] = {}
    for state in STATUS_PRIORITY:
        marker = run_dir / f"STATUS.{state}"
        if marker.is_file():
            markers[state] = marker

    if alive:
        state = "running"
    elif "complete" in markers:
        state = "complete"
    elif "failed" in markers:
        state = "failed"
    elif "stopped" in markers:
        state = "stopped"
    elif "running" in markers:
        state = "stale"
    else:
        state = "idle"

    return {
        "state": state,
        "pid": pid,
        "pid_alive": alive,
        "last_update": datetime.fromtimestamp(metrics_mtime).astimezone().isoformat(timespec="seconds"),
        "last_update_unix": metrics_mtime,
        "age_seconds": max(0.0, time.time() - metrics_mtime),
    }


class JsonlMetricStore:
    """Incrementally parse a JSONL file while another process appends to it."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._identity: tuple[int, int] | None = None
        self._offset = 0
        self._generation = 0
        self._records: list[dict[str, float | int]] = []
        self._metric_names: set[str] = set()
        self._parse_errors = 0

    def _reset(self, identity: tuple[int, int]) -> None:
        self._identity = identity
        self._offset = 0
        self._generation += 1
        self._records = []
        self._metric_names = set()
        self._parse_errors = 0

    def _refresh(self) -> os.stat_result:
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self._identity != identity or stat.st_size < self._offset:
            self._reset(identity)

        with self.path.open("rb") as stream:
            stream.seek(self._offset)
            chunk = stream.read()
        final_newline = chunk.rfind(b"\n")
        if final_newline < 0:
            return stat

        complete = chunk[: final_newline + 1]
        self._offset += len(complete)
        for raw_line in complete.splitlines():
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._parse_errors += 1
                continue
            if not isinstance(value, dict) or not is_finite_number(value.get("step")):
                self._parse_errors += 1
                continue
            record = {key: number for key, number in value.items() if is_finite_number(number)}
            if "step" not in record:
                self._parse_errors += 1
                continue
            self._records.append(record)
            self._metric_names.update(key for key in record if key != "step")
        return stat

    def payload(self, client_generation: int | None, after: int) -> dict[str, Any]:
        with self._lock:
            stat = self._refresh()
            reset = client_generation != self._generation or after < 0 or after > len(self._records)
            start = 0 if reset else after
            records = self._records[start:]
            latest_step = self._records[-1]["step"] if self._records else None
            return {
                "generation": self._generation,
                "reset": reset,
                "cursor": len(self._records),
                "records": records,
                "metric_names": sorted(self._metric_names),
                "record_count": len(self._records),
                "latest_step": latest_step,
                "parse_errors": self._parse_errors,
                "file_size": stat.st_size,
                "file_mtime": stat.st_mtime,
            }


@dataclass(frozen=True)
class RunEntry:
    run_id: str
    name: str
    run_dir: Path
    metrics_path: Path
    mtime: float


class RunCatalog:
    def __init__(self, source: Path, scan_interval: float = 3.0):
        self.source = source.expanduser().resolve()
        self.scan_interval = scan_interval
        self._lock = threading.RLock()
        self._last_scan = 0.0
        self._runs: dict[str, RunEntry] = {}
        self._stores: dict[Path, JsonlMetricStore] = {}
        self._single_file = self.source.is_file()

        if self._single_file and self.source.name != "metrics.jsonl":
            raise ValueError(f"Expected a metrics.jsonl file, got: {self.source}")
        if not self.source.exists():
            raise ValueError(f"Input path does not exist: {self.source}")

    def _discover(self) -> dict[str, RunEntry]:
        if self._single_file:
            paths = [self.source]
            root = self.source.parent
        elif (self.source / "metrics.jsonl").is_file():
            paths = [self.source / "metrics.jsonl"]
            root = self.source
        else:
            paths = list(self.source.rglob("metrics.jsonl"))
            root = self.source

        runs: dict[str, RunEntry] = {}
        for path in paths:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            run_dir = path.parent
            if len(paths) == 1 and (self._single_file or run_dir == root):
                run_id = "."
            else:
                run_id = run_dir.relative_to(root).as_posix()
            runs[run_id] = RunEntry(run_id, run_dir.name, run_dir, path, mtime)
        return runs

    def scan(self, force: bool = False) -> list[RunEntry]:
        with self._lock:
            now = time.monotonic()
            if force or now - self._last_scan >= self.scan_interval:
                self._runs = self._discover()
                self._last_scan = now
            return sorted(self._runs.values(), key=lambda entry: entry.mtime, reverse=True)

    def get(self, run_id: str) -> RunEntry | None:
        self.scan()
        with self._lock:
            return self._runs.get(run_id)

    def store(self, entry: RunEntry) -> JsonlMetricStore:
        with self._lock:
            return self._stores.setdefault(entry.metrics_path, JsonlMetricStore(entry.metrics_path))

    def runs_payload(self) -> dict[str, Any]:
        items = []
        for entry in self.scan():
            status = run_status(entry.run_dir, entry.mtime)
            items.append(
                {
                    "id": entry.run_id,
                    "name": entry.name,
                    "mtime": entry.mtime,
                    "max_train_steps": read_max_train_steps(entry.run_dir),
                    **status,
                }
            )
        return {"runs": items, "source": str(self.source)}


def make_handler(catalog: RunCatalog, html: bytes, poll_interval_ms: int) -> type[BaseHTTPRequestHandler]:
    page = html.replace(b"__POLL_INTERVAL_MS__", str(poll_interval_ms).encode("ascii"))

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "starVLA-dashboard/1"

        def log_message(self, format_string: str, *args: object) -> None:
            print(f"[{self.log_date_time_string()}] {format_string % args}", flush=True)

        def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(page, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/runs":
                self._send_json(catalog.runs_payload())
                return
            if parsed.path == "/api/metrics":
                query = parse_qs(parsed.query)
                run_id = query.get("run", [""])[0]
                entry = catalog.get(run_id)
                if entry is None:
                    self._send_json({"error": "unknown run"}, HTTPStatus.NOT_FOUND)
                    return
                try:
                    generation = int(query["generation"][0]) if "generation" in query else None
                    after = int(query.get("after", ["0"])[0])
                    payload = catalog.store(entry).payload(generation, after)
                except (OSError, ValueError) as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                payload["run"] = {
                    "id": entry.run_id,
                    "name": entry.name,
                    "max_train_steps": read_max_train_steps(entry.run_dir),
                    **run_status(entry.run_dir, payload["file_mtime"]),
                }
                self._send_json(payload)
                return
            if parsed.path == "/api/health":
                self._send_json({"ok": True})
                return
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return DashboardHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("playground/Checkpoints"),
        help="metrics.jsonl, run directory, or directory containing runs",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=6006, help="Listen port (default: 6006)")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Browser refresh interval in seconds (default: 2)",
    )
    parser.add_argument("--pid-file", type=Path, help="Write the dashboard server PID here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_interval < 0.25:
        raise SystemExit("--poll-interval must be at least 0.25 seconds")
    html_path = Path(__file__).with_name("training_dashboard.html")
    try:
        catalog = RunCatalog(args.path)
        runs = catalog.scan(force=True)
        html = html_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not runs:
        raise SystemExit(f"No metrics.jsonl files found under {catalog.source}")

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(catalog, html, round(args.poll_interval * 1000)),
    )
    if args.pid_file:
        args.pid_file.parent.mkdir(parents=True, exist_ok=True)
        args.pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")

    display_host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0", "::"} else args.host
    print(f"starVLA training dashboard: http://{display_host}:{server.server_port}", flush=True)
    print(f"watching {len(runs)} run(s) under {catalog.source}", flush=True)

    def stop_server(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        if args.pid_file:
            try:
                if args.pid_file.read_text(encoding="utf-8").strip() == str(os.getpid()):
                    args.pid_file.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
