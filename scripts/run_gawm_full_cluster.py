"""Supervise a persistent, pre-provisioned five-node GAWM training job.

Run on the configured controller. Use nohup/start_new_session to keep this supervisor alive
after disconnecting. It collects complete checkpoint shards during training.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starVLA.local_settings import cluster_hosts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--hosts", default=cluster_hosts())
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-step", type=int, default=1000)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("run-id must be a new directory name")
    root = Path.cwd().resolve()
    hosts = args.hosts.split(",")
    run = root / "playground/Checkpoints" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / "logs").mkdir()
    (run / "supervisor.pid").write_text(str(os.getpid()))
    config = "examples/LIBERO/train_files/starvla_gawm_full_12epochs.yaml"
    if args.resume_from:
        source = args.resume_from.resolve()
        source_plan = json.loads((source / "training_plan.json").read_text())
        if source_plan["world_size"] != len(hosts) * len(args.gpus.split(",")):
            raise ValueError("Resume must use the same world size")
        marker = source / f"checkpoint_collected_{args.resume_step}.json"
        if not marker.exists():
            raise ValueError(f"Complete collected checkpoint required: {marker}")
        target = run / "checkpoints"
        target.mkdir()
        state_name = f"steps_{args.resume_step}_training_state"
        shutil.copytree(source / "checkpoints" / state_name, target / state_name)
        weights_name = f"steps_{args.resume_step}_pytorch_model.pt"
        shutil.copy2(source / "checkpoints" / weights_name, target / weights_name)
        (run / "resume_source.json").write_text(json.dumps({"run": str(source), "step": args.resume_step}, indent=2))

    def ssh(host):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
                "-o", f"UserKnownHostsFile={root}/.cache/multinode/known_hosts_{host}"]

    def remote(host, command, **kwargs):
        cmd = ["bash", "-c", command] if host == hosts[0] else ssh(host) + [host, command]
        return subprocess.run(cmd, check=True, timeout=60, **kwargs)

    def sync(host, source, destination, collect=False):
        subprocess.run(["rsync", "-a", "--no-owner", "--no-group",
                        *(["--ignore-existing"] if collect else []),
                        "-e", shlex.join(ssh(host)), source, destination],
                       check=True, timeout=300)

    files = [config, "scripts/train_gawm_epochs.py", "scripts/activate_env.sh",
             "scripts/gawm_accelerate.yaml", "scripts/gawm_deepspeed.json",
             "examples/UnifiedPretrain/train_files/data_registry/data_config.py",
             "starVLA/model/modules/world_model/GAWM.py",
             "starVLA/dataloader/lerobot_datasets.py", "starVLA/dataloader/gr00t_lerobot/video.py"]
    exclusions = "examples/LIBERO/train_files/libero_video_exclusions.json"
    if (root / exclusions).exists():
        files.append(exclusions)

    def provision(host):
        remote(host, f"mkdir -p {shlex.quote(str(run))}")
        for filename in files:
            sync(host, str(root / filename), f"{host}:{root / filename}")
        if args.resume_from:
            sync(host, str(run / "checkpoints") + "/", f"{host}:{run}/checkpoints/")

    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        list(pool.map(provision, hosts[1:]))
    # Preserve the exact initial files independently of subsequent IDE edits.
    snapshot = run / "source_snapshot"
    for filename in files:
        dest = snapshot / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((root / filename).read_bytes())
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    jobs = []
    stopping = False

    def interrupt(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    state = {"run_id": args.run_id, "hosts": hosts, "gpus": args.gpus,
             "port": port, "supervisor_pid": os.getpid(), "started_at": time.time(),
             "validation_only": bool(args.smoke_steps), "status": "starting"}
    if args.resume_from:
        state.update(resume_from=str(args.resume_from.resolve()), resume_step=args.resume_step)

    def record():
        temporary = run / "cluster_status.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(run / "cluster_status.json")

    def stop_node(host):
        # PID is the setsid launcher. Validate its command before killing its
        # process group; never match or terminate other training jobs.
        code = (
            "import os,signal,pathlib; "
            f"p=int(pathlib.Path({str(run / 'train.pid')!r}).read_text()); "
            "c=pathlib.Path('/proc/'+str(p)+'/cmdline').read_bytes(); "
            f"assert {args.run_id.encode()!r} in c; "
            "os.killpg(p,signal.SIGTERM)"
        )
        try:
            remote(host, "python3 -c " + shlex.quote(code), capture_output=True)
        except (subprocess.SubprocessError, OSError):
            pass

    collected = set()

    def collect_ready():
        for marker in sorted(run.glob("checkpoint_ready_*.json")):
            step = int(json.loads(marker.read_text())["step"])
            if step in collected:
                continue
            relative = f"checkpoints/steps_{step}_training_state"
            for host in hosts[1:]:
                sync(host, f"{host}:{run}/{relative}/", str(run / relative) + "/", collect=True)
            shards = list((run / relative / "pytorch_model").glob("*optim_states.pt"))
            expected = len(hosts) * len(args.gpus.split(","))
            if len(shards) != expected:
                raise RuntimeError(f"Checkpoint {step}: {len(shards)} optimizer shards, expected {expected}")
            collected.add(step)
            (run / f"checkpoint_collected_{step}.json").write_text(json.dumps({
                "step": step, "optimizer_shards": len(shards), "collected_at": time.time(),
            }))
            print(f"CHECKPOINT COLLECTED step={step} shards={len(shards)}", flush=True)

    try:
        for rank, host in enumerate(hosts):
            command = [
                str(root / ".venv/bin/python"), "-m", "accelerate.commands.launch",
                "--config_file", "scripts/gawm_accelerate.yaml",
                "--num_machines", str(len(hosts)),
                "--num_processes", str(len(hosts) * len(args.gpus.split(","))),
                "--machine_rank", str(rank), "--main_process_ip", hosts[0],
                "--main_process_port", str(port), "--same_network",
                "scripts/train_gawm_epochs.py", "--config_yaml", config,
                "--run-id", args.run_id,
            ]
            if args.smoke_steps:
                command += ["--smoke-steps", str(args.smoke_steps)]
            if args.resume_from:
                command += ["--resume-step", str(args.resume_step)]
            env = {
                "CUDA_VISIBLE_DEVICES": args.gpus, "NCCL_SOCKET_IFNAME": "eth0",
                "GLOO_SOCKET_IFNAME": "eth0", "NCCL_IB_DISABLE": "1", "NCCL_DEBUG": "WARN",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "OMP_NUM_THREADS": "2",
                "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "NO_ALBUMENTATIONS_UPDATE": "1", "PYTHONUNBUFFERED": "1", "STARVLA_DISABLE_TQDM": "1",
            }
            body = (
                f"cd {shlex.quote(str(root))} && source scripts/activate_env.sh && "
                f"echo $$ > {shlex.quote(str(run / 'train.pid'))} && "
                f"touch {shlex.quote(str(run / 'STATUS.running'))} && exec env "
                + " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
                + " " + shlex.join(command)
            )
            # Each node's launcher has an independent process group and no
            # terminal. The detached master supervisor owns the SSH sessions.
            launch = ["setsid", "--wait", "bash", "-c", body]
            cmd = launch if rank == 0 else ssh(host) + [host, shlex.join(launch)]
            log = (run / "logs" / f"{host}.log").open("w")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((host, proc, log))
        state["status"] = "running"
        record()
        print("CLUSTER STARTED " + json.dumps(state), flush=True)
        while True:
            codes = {host: proc.poll() for host, proc, _ in jobs}
            if stopping:
                state["status"] = "stopped"
                break
            if any(code is not None and code != 0 for code in codes.values()):
                state["status"] = "failed"
                state["exit_codes"] = codes
                break
            # Collection errors leave training alive and are retried on the
            # next poll; the node-local checkpoints remain intact.
            try:
                collect_ready()
                state.pop("checkpoint_collection_error", None)
            except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
                state["checkpoint_collection_error"] = str(exc)
                print("CHECKPOINT COLLECTION RETRY " + str(exc), flush=True)
            state["collected_checkpoints"] = sorted(collected)
            if all(code == 0 for code in codes.values()):
                state["status"] = "complete" if not state.get("checkpoint_collection_error") else "collection_failed"
                break
            record()
            time.sleep(10)
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        raise
    finally:
        if state["status"] not in {"complete", "collection_failed"}:
            with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
                list(pool.map(stop_node, hosts))
        for _, proc, log in jobs:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
            log.close()
        state["finished_at"] = time.time()
        record()
        marker = "complete" if state["status"] == "complete" else "stopped" if stopping else "failed"
        for host in hosts:
            try:
                remote(host, f"rm -f {shlex.quote(str(run / 'STATUS.running'))} && touch {shlex.quote(str(run / ('STATUS.' + marker)))}")
            except subprocess.SubprocessError:
                pass
        print("CLUSTER FINISHED " + json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
