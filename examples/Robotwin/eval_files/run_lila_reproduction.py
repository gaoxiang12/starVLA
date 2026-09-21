"""Persist the parity -> smoke -> full 50-task LiLa-WAM reproduction."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from omegaconf import OmegaConf
from examples.Robotwin.eval_files.fetch_lila_assets import sha256

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "examples/Robotwin/eval_files/lila_wam_official.yaml")
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["1", "2", "4"])
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    python = str(ROOT.parent / ".venvs/starVLA/bin/python")
    state = dict(state="running", stage="waiting_for_assets", pid=os.getpid(), started=time.time())
    child = None

    def save():
        temporary = out / "status.json.tmp"
        temporary.write_text(json.dumps(dict(state, updated=time.time()), indent=2) + "\n")
        temporary.replace(out / "status.json")

    def terminate(*_):
        raise KeyboardInterrupt("Reproduction stopped")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, terminate)

    def run(stage, command, env):
        nonlocal child
        state.update(stage=stage)
        save()
        print(stage, flush=True)
        with (out / f"{stage}.log").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            state["child_pid"] = child.pid
            save()
            code = child.wait()
        child = None
        state.pop("child_pid", None)
        if code:
            raise RuntimeError(f"{stage} exited {code}; see {out / (stage + '.log')}")

    try:
        save()
        cfg = OmegaConf.load(args.config).framework.lila_wam
        encoder = Path(cfg.vision_encoder_path) / "model.safetensors"
        weights = Path(cfg.checkpoint_path)
        deadline = time.monotonic() + 7200
        while not (encoder.is_file() and weights.is_file()):
            if time.monotonic() > deadline:
                raise TimeoutError("Required asset download did not finish within two hours")
            time.sleep(10)
        for path in (encoder, weights):
            manifest = json.loads((path.parent / "source_manifest.json").read_text())
            expected = next(item["Sha256"] for item in manifest["Data"]["Files"] if item["Path"] == path.name)
            if sha256(path) != expected:
                raise ValueError(f"Publisher checksum mismatch: {path}")
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
                   OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1")
        run("parity", [python, "-m", "examples.Robotwin.eval_files.check_lila_parity",
            "--config", str(args.config), "--observation", str(args.observation),
            "--output", str(out / "parity.json")], dict(env, CUDA_VISIBLE_DEVICES=args.gpus[0]))
        benchmark = [python, "-m", "examples.Robotwin.eval_files.run_lila_benchmark", "--config", str(args.config)]
        run("smoke", benchmark + ["--output", str(out / "smoke"), "--tasks", "adjust_bottle",
            "blocks_ranking_rgb", "open_laptop", "--episodes", "2", "--seed", "7", "--base-port", "5894",
            "--gpus", *args.gpus], env)
        run("full", benchmark + ["--output", str(out / "full"), "--tasks", "all",
            "--episodes", "100", "--seed", "0", "--base-port", "5894", "--gpus", *args.gpus], env)
        state.update(state="complete", stage="finished")
    except BaseException as error:
        state.update(state="failed", error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        save()


if __name__ == "__main__":
    main()
