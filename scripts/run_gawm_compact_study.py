"""Durable RoboTwin small-backbone and GAWM improvement experiment queue.

Each arm starts from pretrained vision weights, uses the full reference C
budget, and gets a separate 32-step training/deployment preflight. Artifacts
and complete training states are retained on stop. No other job is killed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from omegaconf import OmegaConf

from starVLA.local_settings import cluster_host
from starVLA.model.gawm_config import config_for_gawm_source, migrate_gawm_config

ROOT = Path(os.environ.get("STARVLA_STUDY_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_STUDY = ROOT / "playground/Queues/gawm_compact_improvements_20261007"
REFERENCE = ROOT / "playground/Checkpoints/gawm_robotwin_fixed_dino_temporal_c_20261001"
REFERENCE_SOURCE = ROOT / "playground/Queues/feature_ablation_20261005/sources/robotwin/source_snapshot"
GPU_MAP = {cluster_host("controller"): "0,1,2,3,4,5,6,7", cluster_host("worker-2"): "0,1,2,3,4,5,6,7",
           cluster_host("worker-3"): "0,1,2,3,4,5,6,7", cluster_host("worker-4"): "0,1,2,3", cluster_host("worker-1"): "0,1,2,3"}
CONTROLLER = cluster_host("controller")
STEPS = 765120
ARMS = [
    ("dino_s", "vits16", {}),
    ("dino_splus", "vits16plus", {}),
    ("attention_profile", "vits16plus", {}),
    ("wm_state", "vits16plus", {"wm_state": True}),
    ("act_task", "vits16plus", {"act_task": True}),
    ("temporal_rope", "vits16plus", {"temporal_rope": True}),
    ("selected_combination", "vits16plus", {}),
    ("history2", "vits16plus", {"history_frames": 2}),
    ("adapter384x4", "vits16plus", {"adapter_dim": 384, "adapter_depth": 4}),
    ("adapter384x2", "vits16plus", {"adapter_dim": 384, "adapter_depth": 2}),
    ("motion_weighted", "vits16plus", {"motion_weight": 1.0}),
]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def manifest(source):
    return {str(p.relative_to(source)): digest(p) for p in sorted(Path(source).rglob("*"))
            if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts
            and p.suffix != ".pyc"}


def idle_devices(output):
    # These Blackwell nodes retain a stale 100% utilization reading with
    # 0--1 MiB allocated after their last CUDA process exits. The caller also
    # checks the device UUID against the live compute-process inventory.
    rows = []
    for line in output.splitlines():
        index, uuid, memory, util = [v.strip() for v in line.split(",")]
        if int(memory) <= 200 and (int(util) <= 5 or int(memory) <= 1):
            rows.append((int(index), uuid))
    return rows


def choose_port_base(gpus):
    for base in range(27000, 32000, 16):
        sockets = []
        try:
            for gpu in gpus:
                connection = socket.socket()
                sockets.append(connection)
                connection.bind(("127.0.0.1", base + gpu))
            return base
        except OSError:
            pass
        finally:
            for connection in sockets:
                connection.close()
    raise RuntimeError("No available evaluation port range")


def make_config(study, name, spec, changes):
    cfg = OmegaConf.load(REFERENCE / "input_config.yaml")
    cfg.run_id = f"gawm_compact_{name}_robotwin_c_20261007"
    cfg.run_root_dir = str(ROOT / "playground/Checkpoints")
    wm = cfg.framework.world_model
    wm.encoder_spec = spec
    migrate_gawm_config(cfg)
    wm.vision_encoder_path = str(ROOT / f"playground/Pretrained/dinov3-{spec}-pretrain-lvd1689m")
    wm.feat_layers = [-6, -4, -2]
    cfg.framework.lang_cond.task_vectors_path = str(study / "assets" / spec / "robotwin_official_train_vtt.json")
    flags = {k: v for k, v in changes.items() if not k.startswith("adapter_")}
    if flags:
        cfg.framework.compact_study = flags
    if "adapter_dim" in changes:
        wm.gawm_l_adapter_dim = changes["adapter_dim"]
        wm.gawm_l_adapter_depth = changes["adapter_depth"]
        # Keep head count fixed to isolate adapter width/depth.
        wm.gawm_l_adapter_heads = 8
    if changes.get("history_frames", 1) > 1:
        cfg.datasets.vla_data.history_recorded_offset = 16
    if changes.get("wm_state"):
        cfg.datasets.vla_data.temporal_state_metadata = True
    cfg.training_overrides.trainer.max_train_steps = STEPS
    cfg.training_overrides.trainer.milestone_steps = [STEPS]
    cfg.training_overrides.trainer.is_resume = False
    cfg.training_overrides.trainer.pretrained_checkpoint = None
    return cfg


def prepare(study):
    study = Path(study).resolve()
    study.mkdir(parents=True, exist_ok=False)
    baseline = study / "baseline/source_snapshot"
    shutil.copytree(REFERENCE_SOURCE, baseline, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("git_commit.txt", "git_diff.patch"):
        shutil.copy2(REFERENCE_SOURCE.parent / name, baseline.parent / name)
    support = study / "support"
    support.mkdir()
    for name in ("run_gawm_compact_study.py", "prepare_robotwin_official_vtt.py", "compare_feature_ablation.py", "run_robotwin_hdf5_clean_eval.py"):
        shutil.copy2(ROOT / "scripts" / name, support / name)
    # Preserve the protected original launcher used by the paused study.
    # This study's copy changes resource detection only; model/training
    # configuration resolution and full-state resume stay identical.
    launcher = (ROOT / "scripts/run_gawm_c_cluster.py").read_text()
    launcher = launcher.replace("memory > 200 or utilization > 5", "memory > 200 or (utilization > 5 and memory > 1)")
    (support / "run_gawm_c_cluster.py").write_text(launcher)
    jobs = []
    for name, spec, changes in ARMS:
        directory = study / "jobs" / name
        directory.mkdir(parents=True)
        cfg = make_config(study, name, spec, changes)
        OmegaConf.save(cfg, directory / "launch_config.yaml")
        jobs.append(dict(name=name, encoder=spec, changes=changes, directory=str(directory),
                         config=str(directory / "launch_config.yaml"),
                         run=str(Path(cfg.run_root_dir) / cfg.run_id),
                         source=str(baseline if name in ("dino_s", "dino_splus") else study / "improvements/source_snapshot")))
        save(directory / "status.json", {"status": "queued"})
    plan = dict(root=str(ROOT), created_at=time.time(), scope="robotwin", training_seed=42,
                reference_run=str(REFERENCE), small_baseline="dino_splus", steps=STEPS,
                stage_epochs=[12, 4], global_batch=128, per_device_batch=4, world_size=32,
                gpu_map=GPU_MAP, controller=CONTROLLER,
                gpu_lock=str(ROOT / "playground/Queues/gawm_size_study_20261001/step1/gpu.lock"),
                evaluation=dict(task_configs=["demo_clean", "demo_randomized"], tasks=50, episodes_per_task=10, seed=0),
                constraints=["No VLM or language model", "Frozen DINO patch teacher", "No weights-only resume",
                             "Full C schedule and data budget", "Do not stop unrelated GPU jobs"],
                backbone_changes=dict(layers=[-6, -4, -2], vtt="Recompute train-only VTT with each encoder",
                                      note="Includes unavoidable feature/teacher/VTT dimension changes; not a fixed-teacher encoder-only comparison"),
                combination_rule="Include wm_state/act_task/temporal_rope only when both 500-episode scores are no worse than S+ and either improves by >=5 successes; exploratory selection, not independent confirmation",
                paused_feature_study=str(ROOT / "playground/Queues/feature_ablation_20261005/pause_request_20261007.json"),
                jobs=jobs)
    save(study / "plan.json", plan)
    save(study / "baseline_manifest.json", manifest(baseline))
    save(study / "status.json", {"status": "prepared", "jobs": len(jobs)})
    return plan


class Controller:
    def __init__(self, study):
        self.study = Path(study).resolve()
        self.plan = json.loads((self.study / "plan.json").read_text())
        self.root = Path(self.plan["root"])
        self.job = None
        self.children = []
        self.stopped = False

    def record(self, status, **extra):
        row = dict(status=status, pid=os.getpid(), updated_at=time.time(), **extra)
        if self.job:
            row.update(job=self.job["name"], run=self.job["run"])
            save(Path(self.job["directory"]) / "status.json", row)
        save(self.study / "status.json", row)

    def pause(self, seconds=15):
        for _ in range(seconds):
            if self.stopped:
                raise InterruptedError("Study paused")
            time.sleep(1)

    def stop(self, *_):
        self.stopped = True

    def env(self, source):
        return dict(os.environ, PYTHONPATH=str(source), PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1",
                    STARVLA_STUDY_ROOT=str(self.root),
                    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_ALBUMENTATIONS_UPDATE="1",
                    OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")

    def execute(self, commands, phase, cwd, source):
        directory = Path(self.job["directory"])
        logs = []
        try:
            for i, command in enumerate(commands):
                log = (directory / f"{phase}_{i}.log").open("a")
                logs.append(log)
                self.children.append(subprocess.Popen(command, cwd=cwd, env=self.env(source),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            self.record(phase, child_pids=[p.pid for p in self.children], commands=commands)
            while any(p.poll() is None for p in self.children):
                if any(p.poll() not in (None, 0) for p in self.children):
                    raise RuntimeError(f"{phase} failed; inspect {directory}")
                self.pause(2)
            if any(p.returncode for p in self.children):
                raise RuntimeError(f"{phase} failed; inspect {directory}")
        finally:
            for child in self.children:
                if child.poll() is None:
                    child.terminate()
            for child in self.children:
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
            self.children = []
            for log in logs:
                log.close()

    def probe(self, host):
        prefix = [] if host == self.plan["controller"] else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host]
        try:
            output = subprocess.check_output(prefix + ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=20)
            processes = subprocess.check_output(prefix + ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True, timeout=20)
            active = {s.strip() for s in processes.splitlines()}
            return dict(available=[i for i, uuid in idle_devices(output) if uuid not in active], raw=output, active=list(active))
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            return dict(available=[], error=repr(error))

    def wait_gpus(self, layout=None):
        layout = layout or self.plan["gpu_map"]
        while True:
            with ThreadPoolExecutor(max_workers=5) as pool:
                observations = dict(zip(layout, pool.map(self.probe, layout)))
            if all(set(map(int, devices.split(","))) <= set(observations[host]["available"]) for host, devices in layout.items()):
                save(Path(self.job["directory"]) / "resource_check.json", observations)
                return
            self.record("waiting_free_gpus", observations=observations)
            self.pause()

    def prepare_vtt(self):
        spec = self.job["encoder"]
        target = self.study / "assets" / spec
        if (target / "data_readiness.json").exists():
            if json.loads((target / "data_readiness.json").read_text())["status"] != "ready_for_head_front_training":
                raise ValueError("Invalid VTT readiness marker")
            return
        self.wait_gpus({CONTROLLER: "0,1,2,3,4,5,6,7"})
        cfg = OmegaConf.load(self.job["config"])
        common = [sys.executable, str(self.study / "support/prepare_robotwin_official_vtt.py"),
                  "--root", cfg.datasets.vla_data.data_root_dir, "--encoder", cfg.framework.world_model.vision_encoder_path,
                  "--output", str(target), "--batch-images", "64"]
        commands = [common + ["--shard", str(i), "--shards", "8", "--device", f"cuda:{i}"] for i in range(8)]
        self.execute(commands, "preparing_vtt", self.root, self.job["source"])
        self.execute([common + ["--merge"]], "merging_vtt", self.root, self.job["source"])
        save(target / "asset_hashes.json", {p.name: digest(p) for p in target.glob("*.json") if p.name != "asset_hashes.json"})

    def verify_source(self):
        enhanced = self.job["name"] not in ("dino_s", "dino_splus")
        marker = self.study / ("improvements_ready.json" if enhanced else "baseline_manifest.json")
        while not marker.exists():
            self.record("waiting_validated_implementation", required=str(marker))
            self.pause()
        expected = json.loads(marker.read_text())
        if enhanced:
            expected = expected["source_manifest"]
        if manifest(Path(self.job["source"])) != expected:
            raise ValueError("Study source changed after validation")

    def train(self, run, smoke=False):
        run = Path(run)
        if (run / "training_complete.json").exists():
            self.validate_checkpoint(run, 32 if smoke else STEPS)
            return
        self.wait_gpus()
        source = Path(self.job["directory"]) / "frozen/source_snapshot"
        if not source.exists():
            shutil.copytree(self.job["source"], source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name in ("git_commit.txt", "git_diff.patch"):
                shutil.copy2(Path(self.job["source"]).parent / name, source.parent / name)
            asset = self.study / "assets" / self.job["encoder"] / "robotwin_official_train_vtt.json"
            (source / "initial_assets").mkdir(exist_ok=True)
            shutil.copy2(asset, source / "initial_assets/compact_robotwin_vtt.json")
            save(source.parent / "manifest.json", manifest(source))
        cfg = config_for_gawm_source(OmegaConf.load(self.job["config"]), source)
        cfg.framework.lang_cond.task_vectors_path = str(run / "source_snapshot/initial_assets/compact_robotwin_vtt.json")
        config = str(Path(self.job["directory"]) / ("preflight_config.yaml" if smoke else "runtime_config.yaml"))
        OmegaConf.save(cfg, config)
        command = [sys.executable, str(self.study / "support/run_gawm_c_cluster.py"), "--config", config,
                   "--run-id", run.name, "--hosts", ",".join(self.plan["gpu_map"]),
                   "--gpu-map", json.dumps(self.plan["gpu_map"]), "--controller-host", CONTROLLER]
        if run.exists():
            states = list((run / "checkpoints").glob("steps_*_training_state/complete.json"))
            if not states:
                # Preserve failed preflights/launch artifacts before a fresh attempt.
                if smoke:
                    run.rename(run.with_name(run.name + f"_interrupted_{int(time.time())}"))
                else:
                    raise RuntimeError(f"No complete state for interrupted full run: {run}")
            else:
                command += ["--resume"]
        if "--resume" not in command:
            command += ["--source-snapshot", str(source)]
        if smoke:
            command += ["--smoke-steps", "32"]
        self.execute([command], "preflight_training" if smoke else "training", self.root, self.job["source"])
        self.validate_checkpoint(run, 32 if smoke else STEPS)

    @staticmethod
    def validate_checkpoint(run, step):
        status = json.loads((run / "training_complete.json").read_text())
        state = run / "checkpoints" / f"steps_{step}_training_state"
        meta = json.loads((state / "complete.json").read_text())
        if status.get("completed_steps") != step or meta["step"] != step or len(list(state.glob("random_states_*.pkl"))) != 32:
            raise ValueError("Incomplete training/RNG checkpoint")
        if not (run / "checkpoints" / f"steps_{step}_pytorch_model.pt").is_file():
            raise ValueError("Missing model checkpoint")

    def evaluate(self, run, task_config, smoke=False):
        run = Path(run)
        snapshot = run / "source_snapshot"
        step = 32 if smoke else STEPS
        name = f"{task_config}_{'preflight' if smoke else '10ep_seed0'}_step{step}"
        output = run / "evaluations" / name
        summary = output / "summary.json"
        if summary.exists() and json.loads(summary.read_text()).get("state") == "complete":
            result = json.loads(summary.read_text())
        else:
            if output.exists():
                output.rename(output.with_name(output.name + f"_interrupted_{int(time.time())}"))
            for relative in (".venv", ".venv-robotwin", "thirdparty/RoboTwin", ".cache/robotwin_evaluation"):
                link = snapshot / relative
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(self.root / relative, target_is_directory=True)
            # The archived evaluator has the same stale-zero-memory check as
            # the launcher. Preserve its original and record this sole patch.
            script = snapshot / "scripts/run_robotwin_hdf5_clean_eval.py"
            original = script.read_text()
            evaluator = (self.study / "support/run_robotwin_hdf5_clean_eval.py").read_text()
            patched = evaluator.replace("memory>200 or util>5", "memory>200 or (util>5 and memory>1)")
            if patched != original:
                script.write_text(patched)
                save(run / "evaluation_resource_patch.json", {"before": hashlib.sha256(original.encode()).hexdigest(), "after": digest(script),
                     "changes": "Use frozen Clean/Randomized evaluator and account for stale utilization at 0--1 MiB"})
            layout = {CONTROLLER: "0" if smoke else "0,1,2,3,4,5,6,7"}
            self.wait_gpus(layout)
            gpus = list(map(int, layout[CONTROLLER].split(",")))
            command = [sys.executable, str(script), "--checkpoint", str(run / "checkpoints" / f"steps_{step}_pytorch_model.pt"),
                       "--output", str(output), "--episodes", "1" if smoke else "10", "--seed", "0",
                       "--image-channel-order", "rgb", "--task-config", task_config,
                       "--base-port", str(choose_port_base(gpus)), "--gpus", *map(str, gpus)]
            if smoke:
                command += ["--tasks", "adjust_bottle"]
            self.execute([command], name, snapshot, snapshot)
            result = json.loads(summary.read_text())
        if result.get("state") != "complete" or result.get("trials") != (1 if smoke else 500) or not result.get("sources_unchanged"):
            raise ValueError("Incomplete or changed RoboTwin evaluation")
        return {"summary": str(summary), "successes": result["successes"], "trials": result["trials"]}

    def combination(self):
        baseline = json.loads((self.study / "jobs/dino_splus/results.json").read_text())
        selected = {}
        for name in ("wm_state", "act_task", "temporal_rope"):
            results = json.loads((self.study / "jobs" / name / "results.json").read_text())
            gains = [results[k]["successes"] - baseline[k]["successes"] for k in ("demo_clean", "demo_randomized")]
            if min(gains) >= 0 and max(gains) >= 5:
                selected[name] = True
        save(Path(self.job["directory"]) / "selection.json", dict(changes=selected, rule=self.plan["combination_rule"]))
        if len(selected) < 2:
            self.record("skipped", reason="Fewer than two individually promising changes; no combination to test")
            return False
        cfg = make_config(self.study, self.job["name"], self.job["encoder"], selected)
        OmegaConf.save(cfg, self.job["config"])
        return True

    def run_job(self, job):
        self.job = job
        if json.loads((Path(job["directory"]) / "status.json").read_text())["status"] in ("complete", "skipped"):
            return
        self.verify_source()
        if job["name"] == "selected_combination" and not self.combination():
            return
        self.prepare_vtt()
        if job["name"] == "attention_profile":
            self.wait_gpus({CONTROLLER: "0"})
            reference = next(j for j in self.plan["jobs"] if j["name"] == "dino_splus")
            self.execute([[sys.executable, str(Path(job["source"]) / "scripts/profile_gawm_attention.py"),
                           "--run", reference["run"], "--output", str(Path(job["directory"]) / "profile.json")]],
                         "profiling_attention", self.root, job["source"])
            self.record("complete")
            return
        run = Path(job["run"])
        preflight = run.with_name(run.name + "_preflight")
        marker = Path(job["directory"]) / "preflight_complete.json"
        if not marker.exists():
            self.train(preflight, smoke=True)
            rows = [json.loads(s) for s in (preflight / "metrics.jsonl").read_text().splitlines() if s.strip()]
            if not rows or rows[-1]["step"] != 32 or any(not math.isfinite(v) for r in rows for v in r.values() if isinstance(v, (int, float))):
                raise ValueError("Preflight metrics incomplete/nonfinite")
            result = self.evaluate(preflight, "demo_clean", smoke=True)
            save(marker, dict(training_steps=32, simulation=result, last_metrics=rows[-1]))
        self.train(run)
        results = {mode: self.evaluate(run, mode) for mode in self.plan["evaluation"]["task_configs"]}
        save(Path(job["directory"]) / "results.json", results)
        self.record("complete", results=results)

    def run(self):
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        with (self.study / "supervisor.lock").open("a") as single, Path(self.plan["gpu_lock"]).open("a") as gpu:
            fcntl.flock(single, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                while True:
                    try:
                        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        self.record("waiting_shared_gpu_lock")
                        self.pause()
                for job in self.plan["jobs"]:
                    self.run_job(job)
                self.job = None
                self.record("complete")
            except InterruptedError:
                self.record("paused")
            except BaseException as error:
                self.record("failed", error=repr(error))
                raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "start", "run", "pause", "status"])
    p.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    args = p.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.study), ensure_ascii=False, indent=2))
    elif args.action == "status":
        print((args.study / "status.json").read_text())
    elif args.action == "run":
        Controller(args.study).run()
    elif args.action == "pause":
        study=args.study.resolve()
        pid=int((study / "supervisor.pid").read_text())
        command=Path(f"/proc/{pid}/cmdline")
        if not command.exists() or str(study).encode() not in command.read_bytes() or b"run_gawm_compact_study.py" not in command.read_bytes():
            raise RuntimeError("Recorded supervisor PID does not identify this study")
        save(study / "pause_request.json", dict(pid=pid,requested_at=time.time(),reason="Explicit queue pause"))
        os.kill(pid,signal.SIGTERM)
        print(json.dumps(dict(status="pause_requested",pid=pid)))
    else:
        controller = Controller(args.study)
        pid_file = controller.study / "supervisor.pid"
        if pid_file.exists():
            cmd = Path(f"/proc/{int(pid_file.read_text())}/cmdline")
            if cmd.exists() and str(controller.study).encode() in cmd.read_bytes():
                raise RuntimeError("Study supervisor is already running")
        source = controller.plan["jobs"][0]["source"]
        with (controller.study / "supervisor.log").open("a") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run", "--study", str(controller.study)],
                cwd=controller.root, env=controller.env(source), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pid_file.write_text(str(child.pid) + "\n")
        print(json.dumps(dict(supervisor_pid=child.pid, study=str(controller.study))))


if __name__ == "__main__":
    main()
