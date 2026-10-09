"""Coordinate pre-provisioned SSH nodes; collect checkpoints and test resume.

Run from the project root on the first host in --hosts. All nodes must have the
same checkout, .venv, and LIBERO Goal dataset under .cache/multinode/data.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starVLA.local_settings import cluster_hosts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", default=cluster_hosts("controller", "worker-3"))
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--run-id", default=time.strftime("smoke-%Y%m%d-%H%M%S"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--model", choices=["smoke", "gawm"], default="smoke")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error("run-id must be a directory name")
    hosts = args.hosts.split(",")
    root = Path.cwd().resolve()
    run = root / ".cache/multinode/runs" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    logs = root / ".cache/multinode/logs" / args.run_id
    logs.mkdir(parents=True, exist_ok=False)

    def ssh(host):
        result = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
        known = root / ".cache/multinode" / f"known_hosts_{host}"
        if known.exists():
            result += ["-o", f"UserKnownHostsFile={known}"]
        return result

    def sync(host, source, destination, *, collect_only=False):
        subprocess.run(
            ["rsync", "-a", "--no-owner", "--no-group", "-e", shlex.join(ssh(host)),
             *(["--ignore-existing"] if collect_only else []), source, destination],
            check=True, timeout=120,
        )

    # Copy only this harness, never overwrite unrelated remote source changes.
    for host in hosts[1:]:
        for filename in ["multinode_smoke.py", "multinode_accelerate.yaml", "multinode_deepspeed.json"]:
            sync(host, str(root / "scripts" / filename), f"{host}:{root}/scripts/")
        if args.model == "gawm":
            for filename in [
                "scripts/gawm_accelerate.yaml", "scripts/gawm_deepspeed.json",
                "examples/LIBERO/train_files/starvla_gawm_multinode.yaml",
                "examples/UnifiedPretrain/train_files/data_registry/data_config.py",
                "starVLA/model/modules/world_model/GAWM.py",
            ]:
                sync(host, str(root / filename), f"{host}:{root / filename}")

    def phase(resume):
        label = "resume" if resume else "initial"
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        jobs = []
        try:
            for machine_rank, host in enumerate(hosts):
                command = [
                    str(root / ".venv/bin/python"), "-m", "accelerate.commands.launch",
                    "--config_file", ("scripts/gawm_accelerate.yaml" if args.model == "gawm"
                                      else "scripts/multinode_accelerate.yaml"),
                    "--num_machines", str(len(hosts)), "--num_processes",
                    str(len(hosts) * len(args.gpus.split(","))),
                    "--machine_rank", str(machine_rank), "--main_process_ip", hosts[0],
                    "--main_process_port", str(port), "--same_network",
                    "scripts/multinode_smoke.py", "--run-id", args.run_id,
                    "--steps", "6" if resume else "4",
                    "--model", args.model,
                ]
                if resume:
                    command += ["--resume", "--expected-start-step", "4"]
                env = {
                    "CUDA_VISIBLE_DEVICES": args.gpus,
                    "NCCL_SOCKET_IFNAME": "eth0", "GLOO_SOCKET_IFNAME": "eth0",
                    "NCCL_IB_DISABLE": "1", "NCCL_DEBUG": "WARN",
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "OMP_NUM_THREADS": "2",
                    "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1", "NO_ALBUMENTATIONS_UPDATE": "1",
                    "PYTHONUNBUFFERED": "1",
                }
                body = (
                    f"cd {shlex.quote(str(root))} && source scripts/activate_env.sh && "
                    "exec timeout --signal=TERM --kill-after=15s " + str(args.timeout) + "s env "
                    + " ".join(shlex.quote(f"{k}={v}") for k, v in env.items())
                    + " " + shlex.join(command)
                )
                cmd = ["bash", "-c", body] if machine_rank == 0 else ssh(host) + [host, "bash -c " + shlex.quote(body)]
                log = (logs / f"{label}_{host}.log").open("w")
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                jobs.append((host, proc, log))
            print(f"START {label}: {len(hosts)} nodes x {len(args.gpus.split(','))} GPUs, port {port}", flush=True)
            # Each node is bounded by timeout, including when the SSH client exits.
            results = [(host, proc.wait(timeout=args.timeout + 30)) for host, proc, _ in jobs]
            print(f"EXIT {label}: {results}", flush=True)
            if any(code for _, code in results):
                raise RuntimeError(f"{label} failed; see {logs}")
        finally:
            for _, proc, log in jobs:
                if proc.poll() is None:
                    proc.terminate()
                log.close()

    def collect():
        # Global-rank optimizer/RNG filenames are unique. Each node saves its
        # local shards, so gather them before distributing a complete checkpoint.
        for host in hosts[1:]:
            # Remotes also hold copies of the previous phase's master-only
            # metrics/configs. Never replace fresh master files with those.
            sync(host, f"{host}:{run}/", str(run) + "/", collect_only=True)

    phase(False)
    collect()
    with ThreadPoolExecutor(max_workers=max(1, len(hosts) - 1)) as pool:
        list(pool.map(lambda host: sync(host, str(run) + "/", f"{host}:{run}/"), hosts[1:]))
    phase(True)
    collect()
    initial = [json.loads(p.read_text()) for p in run.glob("report_initial_rank*.json")]
    resumed = [json.loads(p.read_text()) for p in run.glob("report_resume_rank*.json")]
    world = len(hosts) * len(args.gpus.split(","))
    assert len(initial) == len(resumed) == world
    expected_ranks = set(range(world))
    assert {item["rank"] for item in initial} == {item["rank"] for item in resumed} == expected_ranks
    initial_hashes = {item["final_hash"] for item in initial}
    assert len(initial_hashes) == 1
    assert {item["initial_hash"] for item in resumed} == initial_hashes
    assert len({item["final_hash"] for item in resumed}) == 1
    result = {"status": "PASS", "hosts": hosts, "gpus_per_node": args.gpus,
              "world_size": world, "initial_steps": 4, "resumed_steps": 6,
              "initial": initial, "resumed": resumed}
    (run / "verification.json").write_text(json.dumps(result, indent=2))
    print(f"MULTINODE TRAIN + FULL RESUME PASS: {run / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
