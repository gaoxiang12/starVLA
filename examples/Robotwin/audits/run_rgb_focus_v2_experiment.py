"""Train and evaluate the camera-conditioned, zero-projection RGB model."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
CAMP = ROOT / "playground/Checkpoints/gawm_rgb_focus_20260907"
RUN_ID = "gawm_rgb_focus_local_v2_5k_20260907"
RUN = ROOT / "playground/Checkpoints" / RUN_ID
PYTHON = ROOT.parent / ".venvs/starVLA/bin/python"


def save(data):
    path = RUN / "supervisor_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"),
                                        supervisor_pid=os.getpid(), **data), indent=2) + "\n")
    temporary.replace(path)


def main():
    smoke = ROOT / "playground/Checkpoints/gawm_rgb_focus_local_v2_smoke20_20260907"
    metrics = [json.loads(s) for s in (smoke / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["step"] == 20
    assert all(math.isfinite(v) for row in metrics for v in row.values() if isinstance(v, float))
    branch = json.loads((CAMP / "branch_usage_v2_smoke20.json").read_text())
    assert branch["samples"] == 128 and branch["residual_mode"] == "zero_projection"
    assert branch["scores"]["no_focus"]["mean_change_from_full"] > 1e-5
    assert not (RUN / "config.full.yaml").exists(), "Run already started"
    RUN.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="6", PYTHONPATH=str(ROOT), OMP_NUM_THREADS="4",
               NO_ALBUMENTATIONS_UPDATE="1", PYTHONNOUSERSITE="1", WANDB_MODE="disabled", PYTHONUNBUFFERED="1")
    train = [str(PYTHON.parent / "accelerate"), "launch", "--config_file",
             str(ROOT / "starVLA/config/deepseeds/deepspeed_zero2.yaml"), "--num_processes", "1",
             "--main_process_port", "29816", str(ROOT / "starVLA/training/train_starvla.py"),
             "--config_yaml", str(ROOT / "examples/Robotwin/train_files/starvla_gawm_rgb_focus_local_v2.yaml")]
    evaluate = ["bash", str(ROOT / "examples/Robotwin/eval_files/eval_ranking_single.sh"),
                RUN_ID, "blocks_ranking_rgb", "6", "6306"]
    active = None

    def stop(signum, frame):
        raise RuntimeError(f"Supervisor received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        for stage, command in (("train", train), ("eval", evaluate)):
            with (RUN / f"{stage}.log").open("x") as log:
                active = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while active.poll() is None:
                save(dict(state="running", stage=stage, pid=active.pid, gpu="6"))
                time.sleep(15)
            if active.returncode:
                raise RuntimeError(f"{stage} exited with code {active.returncode}")
            if stage == "train":
                rows = [json.loads(s) for s in (RUN / "metrics.jsonl").read_text().splitlines()]
                assert rows[-1]["step"] == 5000
                assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v, float))
                assert (RUN / "final_model/pytorch_model.pt").stat().st_size > 1_000_000
            else:
                assert (RUN / "eval_summary/robotwin_eval_summary.json").is_file()
        save(dict(state="complete"))
    except BaseException as exc:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()
        save(dict(state="failed", error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
