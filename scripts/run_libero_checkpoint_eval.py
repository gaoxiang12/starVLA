#!/usr/bin/env python3
"""Evaluate the four standard LIBERO suites with one policy server per GPU.

Run with scripts/activate_env.sh sourced. A fresh output directory is required.
status.json is updated during evaluation; summary.json contains the final result.
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def progress(path, limit=None):
    content = path.read_text(errors="replace") if path.exists() else ""
    counts = re.findall(r"# episodes completed so far: (\d+).*?# successes: (\d+)", content, re.S)
    if limit is not None:
        counts = counts[:limit]
    episodes, successes = map(int, counts[-1]) if counts else (0, 0)
    final = episodes == limit if limit is not None else "Total episodes:" in content
    return episodes, successes, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port-base", type=int, default=19200)
    parser.add_argument("--reuse-from", type=Path, help="Reuse the first N completed episodes of matching tasks")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    assert checkpoint.is_file(), checkpoint
    assert 1 <= args.trials <= 50
    source = args.reuse_from.resolve() if args.reuse_from else None
    if source:
        source_plan = json.loads((source / "plan.json").read_text())
        assert Path(source_plan["checkpoint"]).resolve() == checkpoint, "Reuse checkpoint mismatch"
        assert source_plan["seed"] == args.seed, "Reuse seed mismatch"
        assert source_plan["execute_horizon"] == 8 and source_plan["unnorm_key"] == "franka"
        assert source_plan["trials_per_task"] >= args.trials
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    assert len(gpus) == len(set(gpus)) and gpus
    usage = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"
    ], text=True)
    memory = {int(row.split(",")[0]): int(row.split(",")[1]) for row in usage.splitlines()}
    for gpu in gpus:
        if memory[gpu] > 512:
            raise RuntimeError(f"GPU {gpu} already uses {memory[gpu]} MiB")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "supervisor.pid").write_text(str(os.getpid()) + "\n")
    env = os.environ.copy()
    egl_libs = ROOT / ".cache/libero_eval/egl/usr/lib/x86_64-linux-gnu"
    if egl_libs.is_dir():
        env["LD_LIBRARY_PATH"] = f"{egl_libs}:{env.get('LD_LIBRARY_PATH', '')}"
    env.update({
        "PYTHONPATH": f"{ROOT / 'playground/LIBERO'}:{ROOT}",
        "LIBERO_CONFIG_PATH": str(ROOT / "playground/LIBERO/libero"),
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "NUMBA_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "NO_ALBUMENTATIONS_UPDATE": "1",
    })
    env.pop("DEBUG", None)
    plan = {
        "checkpoint": str(checkpoint), "gpus": gpus, "trials_per_task": args.trials,
        "seed": args.seed, "execute_horizon": 8, "unnorm_key": "franka",
        "suites": SUITES, "total_tasks": 40, "total_episodes": 40 * args.trials,
        "precision": "FP32 parameters; GAWM visual encoder BF16 autocast",
        "libero_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "playground/LIBERO"), "rev-parse", "HEAD"], text=True
        ).strip(), "started_at": time.time(),
        "reuse_from": str(source) if source else None,
    }
    write_json(output / "plan.json", plan)
    with (output / "packages.txt").open("w") as packages:
        subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"],
                       stdout=packages, check=True)
    pending = deque((suite, task) for task in range(10) for suite in SUITES)
    workers, tasks = [], []
    if source:
        remaining = deque()
        for suite, task_id in pending:
            name = f"{suite}_task{task_id:02d}.log"
            episodes, _, final = progress(source / name, args.trials)
            if not final:
                remaining.append((suite, task_id))
                continue
            shutil.copy2(source / name, output / name)
            tasks.append({"suite": suite, "task_id": task_id, "gpu": None,
                          "log": output / name, "exit_code": 0, "reused": True,
                          "source_log": str(source / name), "counted_first_episodes": episodes})
        pending = remaining
        print(f"Reused {len(tasks)} tasks; {len(pending)} tasks remain", flush=True)

    def snapshot(status):
        suites = {suite: {"episodes": 0, "successes": 0, "completed_tasks": 0} for suite in SUITES}
        rows = []
        for task in tasks:
            episodes, successes, final = progress(task["log"], args.trials)
            row = {key: value for key, value in task.items() if key != "log"}
            row.update(episodes=episodes, successes=successes, log=str(task["log"]))
            rows.append(row)
            suite = suites[task["suite"]]
            suite["episodes"] += episodes
            suite["successes"] += successes
            suite["completed_tasks"] += int(task.get("exit_code") == 0 and final)
        for suite in suites.values():
            suite["success_rate"] = suite["successes"] / suite["episodes"] if suite["episodes"] else None
        result = {"status": status, "updated_at": time.time(), "plan": plan, "suites": suites, "tasks": rows}
        result["episodes"] = sum(suite["episodes"] for suite in suites.values())
        result["successes"] = sum(suite["successes"] for suite in suites.values())
        result["success_rate"] = result["successes"] / result["episodes"] if result["episodes"] else None
        write_json(output / "status.json", result)
        return result

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    status = "failed"
    try:
        for gpu in gpus:
            worker_env = {**env, "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_EGL_DEVICE_ID": str(gpu)}
            port = args.port_base + gpu
            server_log = (output / f"server_gpu{gpu}.log").open("w")
            server = subprocess.Popen([
                sys.executable, "deployment/model_server/server_policy.py",
                "--ckpt_path", str(checkpoint), "--port", str(port), "--idle_timeout", "-1",
            ], cwd=ROOT, env=worker_env, stdout=server_log, stderr=subprocess.STDOUT)
            server_log.close()
            workers.append({"server": server, "client": None, "gpu": gpu, "port": port, "env": worker_env})
        last_update = 0
        while pending or any(worker["client"] is not None for worker in workers):
            for worker in workers:
                if worker["server"].poll() is not None:
                    raise RuntimeError(f"Policy server on GPU {worker['gpu']} exited")
                client = worker["client"]
                if client is not None:
                    if client.poll() is None:
                        if time.time() - worker["task"]["started_at"] > 7200:
                            raise TimeoutError(f"Evaluation task timed out: {worker['task']}")
                        continue
                    task = worker["task"]
                    task.update(exit_code=client.returncode, finished_at=time.time())
                    episodes, _, final = progress(task["log"])
                    if client.returncode or episodes != args.trials or not final:
                        raise RuntimeError(f"Incomplete evaluation: {task}")
                    worker["client"] = None
                if pending:
                    suite, task_id = pending.popleft()
                    log_path = output / f"{suite}_task{task_id:02d}.log"
                    task = {"suite": suite, "task_id": task_id, "gpu": worker["gpu"],
                            "started_at": time.time(), "log": log_path}
                    tasks.append(task)
                    with log_path.open("w") as log:
                        worker["client"] = subprocess.Popen([
                            sys.executable, "examples/LIBERO/eval_files/eval_libero.py",
                            "--args.pretrained-path", str(checkpoint),
                            "--args.port", str(worker["port"]), "--args.task-suite-name", suite,
                            "--args.start-task", str(task_id), "--args.max-tasks", "1",
                            "--args.num-trials-per-task", str(args.trials), "--args.seed", str(args.seed),
                            "--args.unnorm-key", "franka", "--args.execute-horizon", "8",
                            "--args.video-out-path", str(output / "rollouts" / f"{suite}_{task_id}"),
                        ], cwd=ROOT, env=worker["env"], stdout=log, stderr=subprocess.STDOUT)
                    worker["task"] = task
            if time.time() - last_update >= 15:
                result = snapshot("running")
                print(f"Episodes {result['episodes']}/{40 * args.trials}; successes {result['successes']}", flush=True)
                last_update = time.time()
            time.sleep(2)
        status = "complete"
    except KeyboardInterrupt as exc:
        status = "stopped"
        (output / "stop_reason.txt").write_text(str(exc) + "\n")
    except BaseException as exc:
        (output / "error.txt").write_text(repr(exc) + "\n")
        raise
    finally:
        for worker in workers:
            stop(worker["client"])
            stop(worker["server"])
        result = snapshot(status)
        write_json(output / "summary.json", result)
        print(json.dumps({"status": status, "suites": result["suites"]}), flush=True)


if __name__ == "__main__":
    main()
