"""Paired fixed-checkpoint RGB/BGR rollouts, independent of ongoing training."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "examples/Robotwin/eval_files"
CHECKPOINT = ROOT / "playground/Checkpoints/gawm_s_robotwin_continuous_next_49tasks_40k_20260905/final_model/pytorch_model.pt"


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--port", type=int, default=6206)
    parser.add_argument("--orders", nargs="+", choices=("rgb", "bgr"), default=["rgb", "bgr"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--execute-horizon", type=int)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    campaign = args.output.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    if (campaign / "manifest.json").exists():
        raise RuntimeError("Refusing to overwrite an existing diagnostic")
    checkpoint = args.checkpoint.resolve()
    assert checkpoint.is_file()
    manifest = dict(checkpoint=str(checkpoint), checkpoint_size=checkpoint.stat().st_size,
                    checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
                    seed=0, episodes=args.episodes, task="blocks_ranking_rgb", mode="demo_clean",
                    gpu=args.gpu, orders=args.orders, supervisor_pid=os.getpid(),
                    execute_horizon=args.execute_horizon,
                    note="Screening, not final benchmark. Checkpoint and seed fixed across color orders. Videos remain RGB.")
    save(campaign / "manifest.json", manifest)
    active = None

    def stop(signum, frame):
        raise RuntimeError(f"Signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        for order in manifest["orders"]:
            run = campaign / order
            run.mkdir()
            setting = f"{campaign.name}_{order}"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
                       ROBOTWIN_PATH=str(ROOT.parent / "RoboTwin"),
                       STARVLA_PYTHON=str(ROOT.parent / ".venvs/starVLA/bin/python"),
                       ROBOTWIN_PYTHON=str(ROOT.parent / ".venvs/RoboTwin/bin/python"),
                       DEPLOY_POLICY_TEMPLATE_PATH=str(SCRIPTS / "deploy_policy_gawm_aloha.yml"),
                       ROBOTWIN_EVAL_RUNNER_PATH=str(SCRIPTS / "robotwin_color_eval_runner.py"),
                       ROBOTWIN_RANKING_METRICS_PATH=str(run / "ranking_episode_metrics.jsonl"),
                       ROBOTWIN_COLOR_SNAPSHOT_DIR=str(run / "snapshots"),
                       ROBOTWIN_POLICY_TRACE_DIR=str(run / "policy_traces"),
                       ROBOTWIN_POLICY_CHANNEL_ORDER=order, ROBOTWIN_LOG_ROOT=str(run / "logs"),
                       ROBOTWIN_EVAL_VIDEO_LOG="1", ROBOTWIN_USE_BF16="0",
                       PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
            if args.execute_horizon is None:
                env.pop("ROBOTWIN_EXECUTE_HORIZON", None)
            else:
                env["ROBOTWIN_EXECUTE_HORIZON"] = str(args.execute_horizon)
            command = ["bash", str(SCRIPTS / "start_eval.sh"), "--mode", "demo_clean", "--name", setting,
                       "--ckpt", str(checkpoint), "--seed", "0", "--episodes", str(args.episodes),
                       "--jobs-per-gpu", "1", "--base-port", str(args.port), "blocks_ranking_rgb"]
            command = [env["STARVLA_PYTHON"], str(SCRIPTS / "run_robotwin_eval_retry.py"),
                       "--metrics", env["ROBOTWIN_RANKING_METRICS_PATH"], "--log-root", env["ROBOTWIN_LOG_ROOT"],
                       "--attempts", "2", "--", *command]
            save(run / "command.json", dict(command=command, policy_channel_order=order))
            with (run / "eval.log").open("x") as log:
                active = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while active.poll() is None:
                save(campaign / "status.json", dict(state="running", order=order, pid=active.pid,
                     supervisor_pid=os.getpid(), time=time.strftime("%Y-%m-%d %H:%M:%S")))
                time.sleep(15)
            if active.returncode:
                raise RuntimeError(f"{order} evaluation exited {active.returncode}")
            subprocess.run([env["STARVLA_PYTHON"], str(SCRIPTS / "summarize_robotwin_eval.py"),
                            "--result-root", str(ROOT.parent / "RoboTwin/eval_result"),
                            "--setting", setting, "--modes", "demo_clean", "--tasks", "blocks_ranking_rgb",
                            "--episodes", str(args.episodes), "--output-dir", str(run / "summary"), "--strict"],
                           cwd=ROOT, env=env, check=True)
        save(campaign / "status.json", dict(state="complete", time=time.strftime("%Y-%m-%d %H:%M:%S")))
    except BaseException as exc:
        if active is not None and active.poll() is None:
            # start_eval owns descendant cleanup; allow its TERM trap to run.
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
        save(campaign / "status.json", dict(state="failed", error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
