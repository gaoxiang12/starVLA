"""Run a fixed LiLa-WAM RoboTwin protocol and accept only complete counters."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from importlib.metadata import version
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time

from omegaconf import OmegaConf
from examples.Robotwin.eval_files.fetch_lila_assets import sha256
from examples.Robotwin.eval_files.summarize_robotwin_eval import ALL_TASKS, wilson_interval

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ANSI = re.compile(r"\x1b\[[0-9;]*m")
COUNTER = re.compile(r"Success rate:\s*(\d+)/(\d+)\s*=>.*?current seed:\s*(\d+)")
CANCELLED = threading.Event()


def save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def parse_counters(text, episodes):
    rows = [tuple(map(int, match)) for match in COUNTER.findall(ANSI.sub("", text))]
    if len(rows) != episodes:
        raise ValueError(f"Expected {episodes} episode counters, found {len(rows)}")
    previous_success, previous_seed = 0, -1
    for index, (success, trials, seed) in enumerate(rows, 1):
        if trials != index or success - previous_success not in (0, 1) or seed <= previous_seed:
            raise ValueError("Nonmonotonic or invalid episode counters")
        previous_success, previous_seed = success, seed
    return [{"seed": seed, "success": bool(success - (rows[i-1][0] if i else 0))}
            for i, (success, _, seed) in enumerate(rows)]


def stop(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_task(task, gpu, port, args, protocol):
    output = args.output / task
    output.mkdir()
    server = simulator = None
    row = dict(task=task, gpu=gpu, state="starting", started=time.time())
    save(output / "status.json", row)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1",
               PYTHONPATH=str(ROOT), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               NO_ALBUMENTATIONS_UPDATE="1", HF_HUB_OFFLINE="1",
               ROBOTWIN_PATH=str(args.robotwin), ROBOTWIN_PYTHON=str(args.sim_python),
               ROBOTWIN_POLICY_NAME="lila_wam_interface", ROBOTWIN_TEST_NUM=str(args.episodes),
               ROBOTWIN_EVAL_VIDEO_LOG="1" if getattr(args, 'video', False) else "0",
               DEPLOY_POLICY_TEMPLATE_PATH=str(HERE / "deploy_policy_lila_wam.yml"))
    try:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", port))
        with (output / "server.log").open("w") as log:
            server = subprocess.Popen([str(args.python), "-m", "examples.Robotwin.eval_files.lila_wam_server",
                "--config", str(args.config), "--port", str(port), "--seed", str(args.policy_seed)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 180
        while True:
            if CANCELLED.is_set():
                raise RuntimeError("Campaign cancelled")
            if server.poll() is not None:
                raise RuntimeError(f"Policy server exited {server.returncode}; inspect server.log")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Policy server startup exceeded 180 seconds")
                time.sleep(1)
        row.update(state="evaluating", server_pid=server.pid)
        save(output / "status.json", row)
        command = ["bash", str(HERE / "eval.sh"), task, "demo_clean", args.output.name,
                   str(args.seed), str(gpu), protocol["checkpoint"], str(port)]
        with (output / "eval.log").open("w") as log:
            simulator = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + args.task_timeout
            while simulator.poll() is None:
                if CANCELLED.wait(1):
                    raise RuntimeError("Campaign cancelled")
                if time.monotonic() > deadline:
                    raise TimeoutError("RoboTwin task exceeded its wall-clock limit")
            code = simulator.returncode
        if code:
            raise RuntimeError(f"RoboTwin exited {code}; inspect eval.log")
        episodes = parse_counters((output / "eval.log").read_text(errors="replace"), args.episodes)
        successes = sum(ep["success"] for ep in episodes)
        row.update(state="complete", successes=successes, trials=len(episodes),
                   success_rate=successes / len(episodes), episodes=episodes)
    except Exception as error:
        row.update(state="failed", error=repr(error))
    finally:
        stop(simulator)
        stop(server)
        row["elapsed_seconds"] = time.time() - row["started"]
        save(output / "status.json", row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=HERE / "lila_wam_official.yaml")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["all"])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--gpus", nargs="+", default=["1"])
    p.add_argument("--base-port", type=int, default=5794)
    p.add_argument("--python", type=Path, default=ROOT.parent / ".venvs/starVLA/bin/python")
    p.add_argument("--sim-python", type=Path, default=ROOT.parent / ".venvs/RoboTwin/bin/python")
    p.add_argument("--robotwin", type=Path, default=ROOT.parent / "RoboTwin")
    p.add_argument("--task-timeout", type=int, default=86400)
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    tasks = list(ALL_TASKS) if args.tasks == ["all"] else args.tasks
    if args.episodes <= 0 or len(set(tasks)) != len(tasks) or set(tasks) - set(ALL_TASKS):
        p.error("Require positive episodes and unique official task names")
    if len(set(args.gpus)) != len(args.gpus):
        p.error("GPU slots must be unique")
    args.output = args.output.resolve()
    args.config = args.config.resolve()
    args.robotwin = args.robotwin.resolve()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: CANCELLED.set())
    args.output.mkdir(parents=True, exist_ok=False)
    framework = OmegaConf.load(args.config).framework
    settings = framework.official_robotwin if framework.get('name') == 'GAWMOfficial' else framework.lila_wam
    files = [args.config, Path(settings.config_path), Path(settings.checkpoint_path),
             Path(settings.norm_stats_path), Path(settings.vision_encoder_path) / "model.safetensors",
             Path(settings.source_root) / "models/model_runner.py", Path(settings.source_root) / "models/vla_model_fm.py",
             ROOT / "starVLA/model/framework/WM4A/LiLaWAM.py", HERE / "lila_wam_interface.py",
             HERE / "lila_wam_server.py", HERE / "robotwin_eval_runner.py", Path(__file__),
             args.robotwin / "script/eval_policy.py", args.robotwin / "task_config/demo_clean.yml",
             args.robotwin / "task_config/_eval_step_limit.yml", args.robotwin / "envs/_base_task.py",
             args.robotwin / "description/utils/generate_episode_instructions.py",
             Path(settings.vision_encoder_path) / "config.json"]
    if framework.get('name') == 'GAWMOfficial':
        files += [ROOT / 'examples/LiLaWAM/gawm_official.py', ROOT / 'starVLA/model/framework/WM4A/GAWM.py',
                  ROOT / 'starVLA/model/modules/world_model/GAWM.py',
                  ROOT / 'starVLA/model/modules/world_model/visual_token_delta_world_model.py',
                  ROOT / 'starVLA/model/modules/action_model/ACT_ActionHeader.py', ROOT / 'starVLA/task_language.py']
    else:
        files += [Path(settings.task_cond_dir) / task / "task_cond.npy" for task in tasks]
    files += [args.robotwin / "envs" / f"{task}.py" for task in tasks]
    protocol = dict(tasks=tasks, task_config="demo_clean", episodes_per_task=args.episodes, seed=args.seed,
                    framework=str(framework.name), video=args.video,
                    policy_packages={name: version(name) for name in ("torch", "transformers", "numpy", "scipy")},
                    policy_seed=args.policy_seed, checkpoint=str(Path(settings.checkpoint_path).resolve()),
                    policy_rng="simulator CUDA state synchronized at every inference request",
                    task_success="unmodified official RoboTwin eval_success", expert_filter=True,
                    threshold=.90, aggregation="macro average over all 50 tasks",
                    source_sha256={str(path): sha256(path) for path in files})
    save(args.output / "protocol.json", protocol)
    rows = []

    def worker(slot, gpu):
        results = []
        for task in tasks[slot::len(args.gpus)]:
            if CANCELLED.is_set():
                break
            result = run_task(task, gpu, args.base_port + slot, args, protocol)
            print(task, result["state"], result.get("success_rate", result.get("error")), flush=True)
            results.append(result)
        return results

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, slot, gpu) for slot, gpu in enumerate(args.gpus)]
        for future in as_completed(futures):
            rows.extend(future.result())
    complete = [row for row in rows if row["state"] == "complete"]
    successes = sum(row["successes"] for row in complete)
    trials = sum(row["trials"] for row in complete)
    full = len(complete) == 50 and set(tasks) == set(ALL_TASKS) and args.episodes == 100
    rate = successes / trials if trials else None
    unchanged = all(sha256(Path(path)) == digest for path, digest in protocol["source_sha256"].items())
    passed = full and unchanged and rate >= .90
    summary = dict(state="complete" if len(complete) == len(tasks) else "incomplete",
                   missing_tasks=sorted(set(tasks) - {row["task"] for row in complete}),
                   protocol_complete=full, sources_unchanged=unchanged, passed_90_percent=passed,
                   successes=successes, trials=trials, completed_tasks=len(complete),
                   observed_success_rate=rate, pooled_wilson_ci95=wilson_interval(successes, trials) if trials else None,
                   results=sorted(rows, key=lambda row: row["task"]))
    save(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2), flush=True)
    if len(complete) != len(tasks) or not unchanged or (set(tasks) == set(ALL_TASKS) and not passed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
