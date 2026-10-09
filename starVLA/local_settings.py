"""Machine-specific cluster settings kept outside Git."""

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_settings():
    path = Path(os.environ.get("STARVLA_LOCAL_SETTINGS", ROOT / ".local/settings.json")).expanduser()
    if path.exists():
        return json.loads(path.read_text())
    return {}


def cluster_host(name):
    """Resolve a logical node, or return an explicitly non-routable example."""
    host = load_settings().get("hosts", {}).get(name, f"{name}.example.invalid")
    if not isinstance(host, str) or not host or host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError(f"Invalid cluster host for {name}")
    return host


def cluster_hosts(*names):
    names = names or ("controller", "worker-1", "worker-2", "worker-3", "worker-4")
    return ",".join(cluster_host(name) for name in names)


def ssh_target(host):
    user = load_settings().get("ssh_user", "")
    if user and (not isinstance(user, str) or user.startswith("-") or any(c.isspace() for c in user)):
        raise ValueError("Invalid SSH user")
    return f"{user}@{host}" if user and "@" not in host else host


def dashboard_config():
    local = ROOT / ".local/cluster_nodes.json"
    return local if local.exists() else ROOT / "scripts/cluster_nodes.json"
