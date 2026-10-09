import json

import pytest

from starVLA import local_settings


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(local_settings, "ROOT", tmp_path)
    monkeypatch.delenv("STARVLA_LOCAL_SETTINGS", raising=False)


def test_unconfigured_hosts_use_reserved_example_names():
    assert local_settings.cluster_hosts("controller", "worker-3") == (
        "controller.example.invalid,worker-3.example.invalid"
    )
    assert local_settings.ssh_target("worker-1.example.invalid") == "worker-1.example.invalid"


def test_private_config_overrides_nodes_and_ssh_user(tmp_path, monkeypatch):
    config = tmp_path / "private.json"
    config.write_text(json.dumps({
        "hosts": {"controller": "192.0.2.10", "worker-3": "192.0.2.11"},
        "ssh_user": "test-user",
    }))
    monkeypatch.setenv("STARVLA_LOCAL_SETTINGS", str(config))
    assert local_settings.cluster_hosts("worker-3", "controller") == "192.0.2.11,192.0.2.10"
    assert local_settings.ssh_target("192.0.2.11") == "test-user@192.0.2.11"
    assert local_settings.ssh_target("other-user@192.0.2.11") == "other-user@192.0.2.11"


@pytest.mark.parametrize("host", ["-oProxyCommand=command", "worker one", "", None])
def test_invalid_private_hosts_are_rejected(tmp_path, host):
    config = tmp_path / ".local/settings.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"hosts": {"controller": host}}))
    with pytest.raises(ValueError, match="Invalid cluster host"):
        local_settings.cluster_host("controller")


def test_dashboard_prefers_ignored_local_config(tmp_path):
    assert local_settings.dashboard_config() == tmp_path / "scripts/cluster_nodes.json"
    config = tmp_path / ".local/cluster_nodes.json"
    config.parent.mkdir()
    config.write_text('{"nodes": [{"id": "local"}]}')
    assert local_settings.dashboard_config() == config
