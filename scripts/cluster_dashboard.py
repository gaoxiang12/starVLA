#!/usr/bin/env python3
"""Serve a read-only multi-machine training dashboard. No third-party dependencies."""

import argparse
import copy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starVLA.local_settings import dashboard_config

ROOT = Path(__file__).resolve().parent.parent


def load_nodes(path):
    config = json.loads(path.read_text())
    nodes = config["nodes"]
    ids = set()
    for node in nodes:
        if not node.get("id") or node["id"] in ids:
            raise ValueError("Each node requires a unique id")
        ids.add(node["id"])
        host = node.get("host")
        if host and (host.startswith("-") or any(c.isspace() for c in host)):
            raise ValueError("Invalid SSH host")
        node.setdefault("name", node["id"])
        node.setdefault("paths", config.get("paths", []))
        if not isinstance(node["paths"], list) or not all(isinstance(p, str) for p in node["paths"]):
            raise ValueError("paths must be a list of strings")
        node["paths"] = [p.replace("{repo}", str(ROOT)) for p in node["paths"]]
    if not nodes:
        raise ValueError("At least one node is required")
    return nodes


def probe_node(node, timeout=20, history=None):
    source = Path(__file__).with_name("cluster_probe.py").read_text()
    expression = "collect(" + repr(node["paths"]) + ")" if history is None else "collect_history(**" + repr(history) + ")"
    source += "\nprint(json.dumps(" + expression + ", allow_nan=False))\n"
    host = node.get("host")
    if host:
        command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                   "-o", "StrictHostKeyChecking=yes", "-o", "ServerAliveInterval=5",
                   "-o", "ServerAliveCountMax=2"]
        known_hosts = node.get("known_hosts")
        if known_hosts:
            known_path = Path(known_hosts.replace("{repo}", str(ROOT))).expanduser()
            command += ["-o", f"UserKnownHostsFile={known_path}"]
        command += [host, shlex.join([node.get("python", "python3"), "-"])]
    else:
        command = [sys.executable, "-"]
    result = subprocess.run(command, input=source, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr.strip() or f"Probe exited {result.returncode}")[-800:])
    return json.loads(result.stdout)


class ClusterMonitor:
    def __init__(self, nodes, interval=5, timeout=20):
        self.nodes, self.interval, self.timeout = nodes, interval, timeout
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.snapshots = {n["id"]: self.identity(n) | {"connection": "connecting"} for n in nodes}

    @staticmethod
    def identity(node):
        return {"id": node["id"], "name": node["name"], "host": node.get("host") or "本机",
                "expected_gpus": node.get("expected_gpus", 8)}

    def sample(self, node):
        try:
            data = probe_node(node, self.timeout)
            snapshot = data | self.identity(node) | {"connection": "online", "received_at": time.time()}
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            with self.lock:
                snapshot = copy.deepcopy(self.snapshots[node["id"]])
            snapshot.update(connection="offline", error=str(exc), attempted_at=time.time())
        with self.lock:
            self.snapshots[node["id"]] = snapshot

    def worker(self, node):
        while not self.stop.is_set():
            self.sample(node)
            self.stop.wait(self.interval)

    def start(self):
        for node in self.nodes:
            threading.Thread(target=self.worker, args=(node,), daemon=True).start()

    def payload(self):
        with self.lock:
            return {"nodes": copy.deepcopy(list(self.snapshots.values())), "generated_at": time.time(),
                    "interval": self.interval, "record_limit": 600}

    def history(self, node_id, run_id, offset=0, identity=None):
        # Only discovered runs can be read; URL parameters cannot select arbitrary files.
        with self.lock:
            snapshot = self.snapshots.get(node_id, {})
            if not any(run["id"] == run_id for run in snapshot.get("runs", [])):
                raise KeyError("Unknown node or run")
        node = next(n for n in self.nodes if n["id"] == node_id)
        return probe_node(node, self.timeout, history={"directory": run_id, "offset": offset, "identity": identity})


def make_handler(monitor):
    page = Path(__file__).with_name("cluster_dashboard.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            status = HTTPStatus.OK
            if path == "/":
                body, content_type = page, "text/html; charset=utf-8"
            else:
                content_type = "application/json; charset=utf-8"
                if path == "/api/cluster":
                    data = monitor.payload()
                elif path == "/api/history":
                    query = parse_qs(parsed.query)
                    try:
                        data = monitor.history(query.get("node", [""])[0], query.get("run", [""])[0],
                                               int(query.get("offset", ["0"])[0]), query.get("identity", [None])[0])
                    except KeyError:
                        status, data = HTTPStatus.NOT_FOUND, {"error": "Unknown node or run"}
                    except ValueError as exc:
                        status, data = HTTPStatus.BAD_REQUEST, {"error": str(exc)}
                    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                        status, data = HTTPStatus.BAD_GATEWAY, {"error": str(exc)}
                elif path == "/api/health":
                    data = {"ok": True}
                else:
                    status, data = HTTPStatus.NOT_FOUND, {"error": "not found"}
                body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=dashboard_config())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    if args.interval < 1 or args.timeout < 1:
        parser.error("interval and timeout must be at least 1 second")
    monitor = ClusterMonitor(load_nodes(args.config), args.interval, args.timeout)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(monitor))
    monitor.start()

    def stop(*_args):
        monitor.stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"starVLA cluster dashboard: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        monitor.stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
