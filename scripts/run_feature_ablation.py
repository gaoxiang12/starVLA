"""Prepare and supervise feature-only ablations using each baseline's frozen code.

No training or world-model implementation is changed. The only experimental
field is feat_layers; all three input slots and the frozen teacher are retained.
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

from feature_ablation_spec import VARIANTS, asset_contract, make_config, validate_config

ROOT = Path(__file__).resolve().parents[1]
SIZE_STUDY = ROOT / "playground/Queues/gawm_size_study_20261001/step1"
GPU_LOCK = SIZE_STUDY / "gpu.lock"
DEFAULT_STUDY = ROOT / "playground/Queues/feature_ablation_20261005"
DOMAINS = {
    "libero": {
        "run": ROOT / "playground/Checkpoints/gawm_libero_fixed_dino_c_160k_b160_20260929",
        "source": ROOT / "playground/Queues/libero_fixed_dino_160k_20260929/source_snapshot",
        "steps": [40000, 80000],
        "global_batch": 160,
    },
    "robotwin": {
        "run": ROOT / "playground/Checkpoints/gawm_robotwin_fixed_dino_temporal_c_20261001",
        "source": ROOT / "playground/Queues/robotwin_fixed_dino_temporal_20261001/source_snapshot",
        "steps": [765120],
        "global_batch": 128,
    },
}


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_manifest(source):
    source = Path(source)
    files = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Use a clean queue snapshot without environment symlinks: {path}")
        if path.is_file() and "__pycache__" not in path.parts:
            files[str(path.relative_to(source))] = digest(path)
    if not files:
        raise ValueError(f"Empty source snapshot: {source}")
    return files


def check_hashes(files):
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError(f"Frozen experiment input changed: {path}")


def validate_smoke(run):
    run = Path(run)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if not rows or rows[-1].get("step") != 32:
        raise ValueError("Preflight did not complete 32 updates")
    for row in rows:
        if any(not math.isfinite(value) for value in row.values() if isinstance(value, (int, float))):
            raise ValueError("Non-finite preflight metric")
        if "dino_future_loss" not in row or abs(row.get("predicted_latent_rms", 0) - 1) >= .01:
            raise ValueError("Preflight does not have bounded fixed-DINO predictions")
    full = OmegaConf.load(run / "config.full.yaml")
    temporal = float(full.framework.world_model.get("dense_temporal_smoothness_weight", 0))
    if temporal and not all(row.get("temporal_dense_loss", 0) > 0 for row in rows):
        raise ValueError("Existing dense temporal supervision is missing from preflight")
    validate_checkpoint(run, 32)
    return {"steps": 32, "finite": True, "bounded_predictions": True, "last": rows[-1]}


def validate_checkpoint(run, step):
    run = Path(run)
    status = json.loads((run / "cluster_status.json").read_text())
    complete = json.loads((run / "training_complete.json").read_text())
    checkpoint = run / "checkpoints" / f"steps_{step}_pytorch_model.pt"
    state = run / "checkpoints" / f"steps_{step}_training_state/complete.json"
    if status.get("status") != "complete" or status.get("completed_steps") != step:
        raise ValueError(f"Training is incomplete at step {step}: {status.get('status')}")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not state.is_file():
        raise ValueError(f"Incomplete model/training-state checkpoint: {run}, step {step}")
    metadata = json.loads(state.read_text())
    if metadata.get("step") != step or metadata.get("world_size") != status.get("world_size"):
        raise ValueError("Checkpoint step/world-size does not match completed training")
    if len(list(state.parent.glob("random_states_*.pkl"))) != status["world_size"]:
        raise ValueError("Checkpoint lacks the original rank RNG states")
    if complete.get("completed_steps") != step:
        raise ValueError("Completion marker step mismatch")
    return checkpoint


def dependencies_ready(paths):
    rows = {}
    for path in paths:
        path = Path(path)
        rows[str(path)] = json.loads(path.read_text()).get("status") if path.exists() else "missing"
    return all(value == "complete" for value in rows.values()), rows


def available_gpus(output):
    available = []
    for line in output.splitlines():
        if line.strip():
            index, memory, utilization = map(int, line.split(","))
            if memory <= 200 and utilization <= 5:
                available.append(index)
    return available


def layout_ready(layout, observations):
    return all(set(map(int, devices.split(","))) <= set(observations[host].get("available", []))
               for host, devices in layout.items())


def choose_port_base(gpus):
    for base in range(27000, 32000, 16):
        sockets = []
        try:
            for gpu in gpus:
                connection = socket.socket()
                sockets.append(connection)
                connection.bind(("0.0.0.0", base + gpu))
            return base
        except OSError:
            pass
        finally:
            for connection in sockets:
                connection.close()
    raise RuntimeError("No free policy-server port range")


def prepare(study, robotwin_selection="best_libero"):
    study = Path(study).resolve()
    study.mkdir(parents=True, exist_ok=False)
    protected = {}
    plan = dict(root=str(ROOT), created_at=time.time(), training_seed=42,
                variants={name: list(layers) for name, layers in VARIANTS.items()},
                gpu_lock=str(GPU_LOCK), dependencies=[str(SIZE_STUDY / name / "status.json")
                                                    for name in ("act_l", "enc_b", "enc_s")],
                jobs=[], baselines={}, support={}, robotwin_selection=robotwin_selection)
    support = study / "support"
    support.mkdir()
    for name in ("run_feature_ablation.py", "feature_ablation_spec.py", "compare_feature_ablation.py"):
        target = support / name
        shutil.copy2(ROOT / "scripts" / name, target)
        protected[str(target)] = digest(target)
        plan["support"][name] = str(target)
    # The unchanged cluster launcher locates the installed venv through cwd
    # and copies these two orchestration files from there. Protect the exact
    # baseline versions rather than changing that launcher's behavior.
    for name in ("scripts/run_gawm_c_cluster.py", "scripts/activate_env.sh"):
        protected[str(ROOT / name)] = digest(ROOT / name)
    # Reuse the pretrained encoder, VTT, data registry, losses, trainer and
    # world predictor from each domain's completed baseline independently.
    for domain, settings in DOMAINS.items():
        baseline = settings["run"]
        reference = study / "references" / domain
        reference.mkdir(parents=True)
        base_config = reference / "input_config.yaml"
        shutil.copy2(baseline / "input_config.yaml", base_config)
        protected[str(base_config)] = digest(base_config)
        source = study / "sources" / domain / "source_snapshot"
        files = source_manifest(settings["source"])
        shutil.copytree(settings["source"], source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("git_commit.txt", "git_diff.patch"):
            shutil.copy2(settings["source"].parent / name, source.parent / name)
            protected[str(source.parent / name)] = digest(source.parent / name)
        for relative, expected in files.items():
            protected[str(source / relative)] = expected
            # Prove this is the baseline implementation, allowing only the
            # orchestration additions already in its queue snapshot.
            if relative.startswith(("starVLA/", "examples/", "scripts/")):
                original = baseline / "source_snapshot" / relative
                if original.is_file() and digest(original) != expected:
                    raise ValueError(f"Baseline source parity failed: {relative}")
        for directory in ("starVLA", "examples", "scripts", "deployment"):
            for original in (baseline / "source_snapshot" / directory).rglob("*"):
                if original.is_file() and not original.is_symlink() and "__pycache__" not in original.parts:
                    relative = str(original.relative_to(baseline / "source_snapshot"))
                    if files.get(relative) != digest(original):
                        raise ValueError(f"Baseline source missing or changed in frozen copy: {relative}")
        config = OmegaConf.load(base_config)
        asset = Path(config.framework.lang_cond.task_vectors_path)
        if digest(asset) != digest(source / "initial_assets" / asset.name):
            raise ValueError(f"VTT differs from completed baseline: {domain}")
        protected[str(asset)] = digest(asset)
        contract = asset_contract(config, baseline_dir=baseline)
        save(reference / "asset_contract.json", contract)
        protected[str(reference / "asset_contract.json")] = digest(reference / "asset_contract.json")
        for row in contract["assets"].values():
            protected[row["path"]] = row["sha256"]
        for relative in ("starVLA/training/recipe.py", "starVLA/config/training/c_recipe.yaml",
                         "scripts/run_gawm_c_cluster.py", "scripts/activate_env.sh"):
            if digest(ROOT / relative) != digest(source / relative):
                raise ValueError(f"Configuration resolver differs from baseline: {relative}")
        encoder = Path(config.framework.world_model.vision_encoder_path)
        for path in [encoder / "config.json", *sorted(encoder.glob("*.safetensors"))]:
            protected[str(path)] = digest(path)
        exclusions = config.datasets.vla_data.get("episode_exclusions_file")
        if exclusions:
            protected[str(Path(exclusions))] = digest(exclusions)
        status = json.loads((baseline / "cluster_status.json").read_text())
        if status["status"] != "complete" or status["world_size"] != 32:
            raise ValueError("Expected a completed 32-rank baseline")
        layout = status["gpu_map"]
        steps = settings["steps"]
        summaries = {str(step): str(baseline / "evaluations" /
                     (f"libero4_10ep_seed7_step{step}" if domain == "libero" else f"clean10_seed0_step{step}") /
                     "summary.json") for step in steps}
        plan["baselines"][domain] = dict(run=str(baseline), config=str(base_config), source=str(source),
                                        gpu_map=layout, controller_host=status["controller_host"],
                                        eval_steps=steps, global_batch=settings["global_batch"], summaries=summaries,
                                        vtt_sha256=digest(asset))
        for variant in ("last_layer", "late_layers", "wide_layers"):
            if variant not in VARIANTS:
                raise ValueError(f"Missing variant: {variant}")
            name = f"{domain}_{variant}"
            job = study / "jobs" / name
            job.mkdir(parents=True)
            run = ROOT / f"playground/Checkpoints/gawm_feature_{name}_c_{max(steps)}_20261005"
            candidate = make_config(base_config, run, variant, world_size=32,
                                    max_steps=max(steps), eval_steps=steps)
            validate_config(OmegaConf.load(base_config), candidate)
            config_path = job / "launch_config.yaml"
            OmegaConf.save(candidate, config_path)
            protected[str(config_path)] = digest(config_path)
            plan["jobs"].append(dict(name=name, benchmark=domain, variant=variant, run=str(run),
                                     directory=str(job), config=str(config_path), source=str(source),
                                     max_steps=max(steps), eval_steps=steps))
            save(job / "status.json", dict(status="queued", variant=variant, benchmark=domain))
        if domain == "robotwin":
            protocol = json.loads((Path(summaries[str(max(steps))]).parent / "protocol.json").read_text())
            for filename, expected in protocol["source_sha256"].items():
                path = Path(filename)
                if path.name in ("eval_policy.py", "demo_clean.yml", "environment.json"):
                    if digest(path) != expected:
                        raise ValueError(f"RoboTwin simulator/environment differs from baseline: {path}")
                    protected[str(path)] = expected
    # All LIBERO screen runs first; RoboTwin keeps its full 12+4 budget, even
    # when a feature failed on LIBERO. Cross-benchmark transfer is not assumed.
    save(study / "plan.json", plan)
    protected[str(study / "plan.json")] = digest(study / "plan.json")
    save(study / "frozen_inputs.json", protected)
    save(study / "status.json", dict(status="prepared", jobs=len(plan["jobs"]),
                                    note="No candidate training or evaluation has started"))
    return plan


class Controller:
    def __init__(self, study):
        self.study = Path(study).resolve()
        self.plan = json.loads((self.study / "plan.json").read_text())
        self.root = Path(self.plan.get("root", ROOT))
        self.child = None
        self.stopped = False
        self.job = None

    def record(self, status, **extra):
        value = dict(status=status, pid=os.getpid(), updated_at=time.time(), **extra)
        if self.job:
            value.update(job=self.job["name"], run=self.job["run"])
            save(Path(self.job["directory"]) / "status.json", value)
        save(self.study / "status.json", value)

    def pause(self, seconds=30):
        for _ in range(seconds):
            if self.stopped:
                raise InterruptedError("Feature experiment stopped")
            time.sleep(1)

    def stop(self, *_):
        self.stopped = True

    def validate_inputs(self):
        check_hashes(json.loads((self.study / "frozen_inputs.json").read_text()))
        for job in self.plan["jobs"]:
            baseline = self.plan["baselines"][job["benchmark"]]
            validate_config(OmegaConf.load(baseline["config"]), OmegaConf.load(job["config"]))

    def validate_run_source(self, run, baseline):
        source = Path(baseline["source"])
        for relative, expected in source_manifest(source).items():
            actual = Path(run) / "source_snapshot" / relative
            if not actual.is_file() or digest(actual) != expected:
                raise ValueError(f"Candidate source differs from frozen baseline: {actual}")

    def validate_statistics(self, run, baseline):
        actual = json.loads((Path(run) / "dataset_statistics.json").read_text())
        expected = json.loads((Path(baseline["run"]) / "dataset_statistics.json").read_text())
        if actual != expected:
            raise ValueError("Candidate dataset/action normalization statistics differ from baseline")

    def wait_dependencies(self):
        while True:
            ready, rows = dependencies_ready(self.plan["dependencies"])
            if ready:
                return
            self.record("waiting_existing_experiments", dependencies=rows)
            self.pause()

    def probe(self, host, controller):
        command = ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
        if host != controller:
            command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, *command]
        try:
            output = subprocess.check_output(command, text=True, timeout=25)
            return dict(available=available_gpus(output), raw=output)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            return dict(available=[], error=repr(error))

    def wait_layout(self, baseline):
        layout = baseline["gpu_map"]
        while True:
            with ThreadPoolExecutor(max_workers=5) as pool:
                rows = list(pool.map(lambda host: self.probe(host, baseline["controller_host"]), layout))
            observations = dict(zip(layout, rows))
            if layout_ready(layout, observations):
                save(Path(self.job["directory"]) / "resource_check.json", observations)
                return
            self.record("waiting_original_gpu_layout", observations=observations)
            self.pause()

    def execute(self, command, phase, cwd, env=None):
        if self.stopped:
            raise InterruptedError("Feature experiment stopped")
        with (Path(self.job["directory"]) / f"{phase}.log").open("a") as log:
            self.child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.record(phase, child_pid=self.child.pid, command=command)
        while self.child.poll() is None:
            self.pause(2)
        code = self.child.returncode
        self.child = None
        if code:
            raise RuntimeError(f"{phase} exited {code}; see {self.job['directory']}/{phase}.log")

    def launcher(self, config_path, run, baseline, smoke=False):
        source = Path(baseline["source"])
        layout = baseline["gpu_map"]
        command = [sys.executable, str(source / "scripts/run_gawm_c_cluster.py"),
                   "--config", str(config_path), "--run-id", run.name,
                   "--hosts", ",".join(layout), "--gpu-map", json.dumps(layout),
                   "--controller-host", baseline["controller_host"], "--source-snapshot", str(source)]
        if smoke:
            command += ["--smoke-steps", "32"]
        env = {**os.environ, "PYTHONPATH": str(source), "PYTHONUNBUFFERED": "1",
               "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "NO_ALBUMENTATIONS_UPDATE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
        # The installed venv is found through cwd. The only cwd-based source
        # copies are the two hash-protected orchestration files above; model,
        # trainer and C recipe imports come from the frozen PYTHONPATH.
        return command, env

    def evaluate(self, run, step, baseline, smoke=False):
        snapshot = run / "source_snapshot"
        domain = self.job["benchmark"]
        links = [".venv", "playground/LIBERO", ".cache/libero_eval/egl"] if domain == "libero" else [
            ".venv", ".venv-robotwin", "thirdparty/RoboTwin", ".cache/robotwin_evaluation"]
        for relative in links:
            link = snapshot / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(self.root / relative, target_is_directory=True)
        while True:
            free = self.probe(baseline["controller_host"], baseline["controller_host"])["available"]
            if free:
                break
            self.record("waiting_evaluation_gpus", step=step)
            self.pause()
        gpus = free[:1] if smoke else free
        base_port = choose_port_base(gpus)
        job_path = Path(self.job["directory"])
        checkpoint = run / "checkpoints" / f"steps_{step}_pytorch_model.pt"
        if domain == "libero":
            name = "libero4_preflight_1ep" if smoke else f"libero4_10ep_seed7_step{step}"
            output = run / "evaluations" / name
            bundle = job_path / "eval_models" / run.name / name
            (bundle / "checkpoints").mkdir(parents=True, exist_ok=True)
            link = bundle / "checkpoints" / checkpoint.name
            if not link.exists():
                os.link(checkpoint, link)
            shutil.copy2(run / "config.full.yaml", bundle / "config.yaml")
            shutil.copy2(run / "dataset_statistics.json", bundle / "dataset_statistics.json")
            command = [sys.executable, str(snapshot / "scripts/run_libero_checkpoint_eval.py"),
                       "--checkpoint", str(link), "--output", str(output), "--gpus", ",".join(map(str, gpus)),
                       "--trials", "1" if smoke else "10", "--seed", "7", "--port-base", str(base_port)]
        else:
            name = "clean_preflight_1ep" if smoke else f"clean10_seed0_step{step}"
            output = run / "evaluations" / name
            command = [sys.executable, str(snapshot / "scripts/run_robotwin_hdf5_clean_eval.py"),
                       "--checkpoint", str(checkpoint), "--output", str(output), "--episodes", "1" if smoke else "10",
                       "--seed", "0", "--image-channel-order", "rgb", "--base-port", str(base_port),
                       "--gpus", *map(str, gpus)]
            if smoke:
                command += ["--tasks", "adjust_bottle"]
        summary = output / "summary.json"
        if not summary.exists():
            env = {**os.environ, "PYTHONPATH": str(snapshot), "PYTHONNOUSERSITE": "1",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "NO_ALBUMENTATIONS_UPDATE": "1", "PYTHONUNBUFFERED": "1",
                   "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
            self.execute(command, name, snapshot, env)
        result = json.loads(summary.read_text())
        if domain == "libero":
            if result.get("status") != "complete" or result.get("episodes") != (40 if smoke else 400):
                raise ValueError("Incomplete LIBERO closed-loop evaluation")
        elif result.get("state") != "complete" or not result.get("sources_unchanged") or result.get("trials") != (1 if smoke else 500):
            raise ValueError("Incomplete or changed RoboTwin closed-loop evaluation")
        return summary

    def run_job(self, job):
        self.job = job
        baseline = self.plan["baselines"][job["benchmark"]]
        run = Path(job["run"])
        directory = Path(job["directory"])
        if json.loads((directory / "status.json").read_text()).get("status") == "complete":
            return
        self.record("validating_frozen_inputs")
        self.validate_inputs()
        lock_path = Path(self.plan["gpu_lock"])
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self.record("waiting_shared_gpu_lock")
                    self.pause()
            # Keep the shared lock through training and evaluation. Current
            # work is already complete; this guarantees serial study jobs.
            self.wait_layout(baseline)
            smoke = run.with_name(run.name + "_preflight")
            if not (directory / "preflight_complete.json").exists():
                smoke_config = make_config(baseline["config"], smoke, job["variant"], world_size=32,
                                           max_steps=job["max_steps"], eval_steps=job["eval_steps"])
                smoke_path = directory / "smoke_config.yaml"
                OmegaConf.save(smoke_config, smoke_path)
                command, env = self.launcher(smoke_path, smoke, baseline, smoke=True)
                if not (smoke / "training_complete.json").exists():
                    if smoke.exists():
                        raise FileExistsError(f"Incomplete preflight requires explicit recovery: {smoke}")
                    self.execute(command, "smoke_training", self.root, env)
                save(directory / "smoke_validation.json", validate_smoke(smoke))
                self.validate_run_source(smoke, baseline)
                self.validate_statistics(smoke, baseline)
                simulation = self.evaluate(smoke, 32, baseline, smoke=True)
                save(directory / "preflight_complete.json", dict(simulation=str(simulation), training_steps=32))
            if not (run / "training_complete.json").exists():
                if run.exists():
                    raise FileExistsError(f"Incomplete run requires full-state recovery: {run}")
                self.wait_layout(baseline)
                command, env = self.launcher(job["config"], run, baseline)
                self.execute(command, "training", self.root, env)
            validate_checkpoint(run, job["max_steps"])
            self.validate_inputs()
            self.validate_run_source(run, baseline)
            self.validate_statistics(run, baseline)
            reports = []
            for step in job["eval_steps"]:
                summary = self.evaluate(run, step, baseline)
                comparison = directory / f"comparison_step{step}.json"
                command = [sys.executable, self.plan["support"]["compare_feature_ablation.py"],
                           "--benchmark", job["benchmark"], "--baseline", baseline["summaries"][str(step)],
                           "--candidate", str(summary), "--output", str(comparison)]
                self.execute(command, f"compare_step{step}", self.root)
                reports.append(str(comparison))
            self.validate_inputs()
            self.validate_run_source(run, baseline)
            self.record("complete", comparisons=reports)
        self.job = None

    def run(self):
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        with (self.study / "experiment.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self.validate_inputs()
                self.wait_dependencies()
                selected = None
                for job in self.plan["jobs"]:
                    if job["benchmark"] == "robotwin" and self.plan.get("robotwin_selection") == "best_libero":
                        if selected is None:
                            selected = self.select_robotwin_variant()
                        if job["variant"] != selected:
                            save(Path(job["directory"]) / "status.json",
                                 dict(status="not_selected", reason="LIBERO 80k closed-loop screening",
                                      selected_variant=selected))
                            continue
                    self.run_job(job)
                self.record("complete", report=self.write_report())
            except InterruptedError:
                self.record("stopped")
            except BaseException as error:
                self.record("failed", error=repr(error))
                raise
            finally:
                if self.child is not None and self.child.poll() is None:
                    self.child.terminate()
                    self.child.wait(timeout=180)

    def select_robotwin_variant(self):
        scores = []
        for job in self.plan["jobs"]:
            if job["benchmark"] == "libero":
                report = json.loads((Path(job["directory"]) / f"comparison_step{job['max_steps']}.json").read_text())
                if not report["complete"] or not report["fully_paired"]:
                    raise ValueError("LIBERO screening must be complete and use identical initial states before selection")
                # Predeclared primary budget, then paired net gains; ties use
                # the queue's last_layer/late_layers/wide_layers order.
                scores.append((report["candidate"]["successes"],
                               report["paired"]["gains"] - report["paired"]["losses"], job["variant"]))
        best = max(range(len(scores)), key=lambda index: scores[index][:2])
        selected = scores[best][2]
        save(self.study / "robotwin_selection.json", dict(variant=selected, scores=scores,
                                                         rule="highest LIBERO 80k success count; paired net gains break ties",
                                                         note="Selection is not evidence of RoboTwin improvement"))
        return selected

    def write_report(self):
        reports = {}
        for job in self.plan["jobs"]:
            state = json.loads((Path(job["directory"]) / "status.json").read_text())["status"]
            if state == "not_selected":
                continue
            if state != "complete":
                raise ValueError(f"Cannot summarize unfinished job: {job['name']}")
            primary = Path(job["directory"]) / f"comparison_step{job['max_steps']}.json"
            reports[job["name"]] = json.loads(primary.read_text())
        screening = {}
        for domain in self.plan["baselines"]:
            rows = []
            for job in self.plan["jobs"]:
                if job["benchmark"] != domain:
                    continue
                if job["name"] not in reports:
                    continue
                report = reports[job["name"]]
                paired = report["paired"]
                positive = (report["complete"] and report["fully_paired"]
                            and report["delta_percentage_points"] > 0
                            and paired["gains"] > paired["losses"]
                            and paired["mcnemar_exact_two_sided_p"] < .05 / 3)
                rows.append(dict(variant=job["variant"], feature_attribution_verified=True,
                                 candidate=report["candidate"], baseline=report["baseline"],
                                 delta_percentage_points=report["delta_percentage_points"],
                                 adjusted_p=min(1, 3 * (paired["mcnemar_exact_two_sided_p"] or 0))
                                 if paired["mcnemar_exact_two_sided_p"] is not None else None,
                                 screening_improvement=positive, effective=False,
                                 fully_paired=report["fully_paired"]))
            screening[domain] = sorted(rows, key=lambda row: row["delta_percentage_points"], reverse=True)
        shared_signals = [variant for variant in ("last_layer", "late_layers", "wide_layers")
                          if all(any(row["variant"] == variant and row["screening_improvement"]
                                     for row in rows) for rows in screening.values())]
        output = self.study / "results.json"
        save(output, dict(results=reports, ranking=screening, positive_on_both_benchmarks=shared_signals,
                          primary_steps={k: max(v["eval_steps"]) for k, v in self.plan["baselines"].items()},
                          familywise_alpha=.05, bonferroni_tests_per_benchmark=3,
                          significance_threshold=.05 / 3,
                          interpretation="Compare gains on identical episode states at each domain's primary budget. "
                          "Three methods are screened; require p<0.05/3 for evidence of improvement. "
                          "One training seed is preliminary; positive results need independent training seeds."))
        lines = ["# 特征提取消融闭环结果", "", "仅改变 DINO 特征层选择，训练方法和世界模型沿用各域冻结基线。", "",
                 "| Benchmark | 特征方法 | 成功率 | 基线 | 差值（百分点） | 完整配对 | 多重检验后的初筛信号 |",
                 "|---|---|---:|---:|---:|---|---|"]
        for domain, rows in screening.items():
            for row in rows:
                candidate, baseline = row["candidate"], row["baseline"]
                lines.append(f"| {domain} | {row['variant']} | {candidate['successes']}/{candidate['episodes']} | "
                             f"{baseline['successes']}/{baseline['episodes']} | {row['delta_percentage_points']:+.2f} | "
                             f"{row['fully_paired']} | {row['screening_improvement']} |")
        lines += ["", "初筛采用单个训练 seed，不能直接宣称稳定有效。" ,
                  "3 种方法的主检查点使用 Bonferroni 阈值 0.05/3；40k 和逐任务指标仅用于诊断。", "",
                  "两域都出现初筛收益的方法：" + (", ".join(shared_signals) if shared_signals else "尚无。")]
        (self.study / "results.md").write_text("\n".join(lines) + "\n")
        return str(output)


def start_background(study):
    controller = Controller(study)
    controller.validate_inputs()
    pid_path = controller.study / "supervisor.pid"
    if pid_path.exists():
        pid = int(pid_path.read_text())
        command = Path(f"/proc/{pid}/cmdline")
        if command.exists() and str(controller.study).encode() in command.read_bytes():
            raise FileExistsError(f"Feature-ablation supervisor is already running: {pid}")
    source = controller.plan["baselines"]["libero"]["source"]
    env = {**os.environ, "PYTHONPATH": source, "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "NO_ALBUMENTATIONS_UPDATE": "1",
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
    command = [sys.executable, controller.plan["support"]["run_feature_ablation.py"],
               "run", "--study", str(controller.study)]
    with (controller.study / "supervisor.log").open("a") as log:
        child = subprocess.Popen(command, cwd=controller.root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(str(child.pid) + "\n")
    return child.pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "validate", "start"])
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--robotwin-selection", choices=["best_libero", "all"], default="best_libero",
                        help="prepare three LIBERO arms, then validate its best arm or all arms on RoboTwin")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.study, args.robotwin_selection)
        print(json.dumps(dict(study=str(args.study), jobs=len(result["jobs"]))))
    elif args.action == "validate":
        Controller(args.study).validate_inputs()
        print("Frozen inputs and feature-only configuration contracts passed")
    elif args.action == "start":
        print(json.dumps(dict(pid=start_background(args.study), study=str(args.study.resolve()))))
    else:
        Controller(args.study).run()


if __name__ == "__main__":
    main()
