"""Exercise queue safety without launching training or touching live runs."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("feature_ablation_runner", SCRIPTS / "run_feature_ablation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def complete_run(run, step=32, world_size=2, temporal=0):
    """Write independently consistent completion records for a fake run."""
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "cluster_status.json", {"status": "complete", "completed_steps": step,
                                            "world_size": world_size})
    write_json(run / "training_complete.json", {"completed_steps": step})
    state = run / "checkpoints" / f"steps_{step}_training_state"
    write_json(state / "complete.json", {"step": step, "world_size": world_size})
    for rank in range(world_size):
        (state / f"random_states_{rank}.pkl").write_bytes(b"rank RNG record")
    (run / "checkpoints" / f"steps_{step}_pytorch_model.pt").write_bytes(b"model checkpoint")
    OmegaConf.save(OmegaConf.create({"framework": {"world_model": {
        "dense_temporal_smoothness_weight": temporal}}}), run / "config.full.yaml")
    row = {"step": step, "dino_future_loss": .2, "predicted_latent_rms": 1.,
           "temporal_dense_loss": .1 if temporal else 0.}
    (run / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    write_json(run / "dataset_statistics.json", {"franka": {"q01": [0.], "q99": [1.]}})
    return state


class FeatureQueueSafetyTest(unittest.TestCase):
    def test_dependencies_require_successful_completion_of_every_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"{name}.json" for name in ("act_l", "enc_b", "enc_s")]
            self.assertFalse(runner.dependencies_ready(paths)[0])
            for path in paths:
                write_json(path, {"status": "complete"})
            self.assertTrue(runner.dependencies_ready(paths)[0])
            for status in ("training", "waiting_evaluation_gpus", "failed", "stopped"):
                with self.subTest(status=status):
                    write_json(paths[-1], {"status": status})
                    ready, rows = runner.dependencies_ready(paths)
                    self.assertFalse(ready)
                    self.assertEqual(rows[str(paths[-1])], status)

    def test_waiting_dependencies_never_probes_or_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            write_json(study / "plan.json", {"dependencies": [str(study / "not_complete.json")], "jobs": []})
            control = runner.Controller(study)
            with patch.object(control, "validate_inputs"), \
                 patch.object(control, "pause", side_effect=InterruptedError("test stop")), \
                 patch.object(control, "probe") as probe, \
                 patch.object(runner.subprocess, "Popen") as launch, \
                 patch.object(runner.signal, "signal"):
                control.run()
            probe.assert_not_called()
            launch.assert_not_called()
            self.assertEqual(json.loads((study / "status.json").read_text())["status"], "stopped")

    def test_free_gpu_requires_both_memory_and_utilization(self):
        self.assertEqual(runner.available_gpus("0, 0, 0\n1, 201, 0\n2, 0, 6\n3, 200, 5\n"), [0, 3])
        with self.assertRaises(ValueError):
            runner.available_gpus("0, N/A, 0\n")

    def test_missing_or_occupied_selected_gpu_does_not_allow_an_alternative_map(self):
        layout = {"first": "0,1", "second": "2,3"}
        observations = {"first": {"available": [0, 1]}, "second": {"available": [0, 1, 2]}}
        self.assertFalse(runner.layout_ready(layout, observations))
        observations["second"]["available"].append(3)
        self.assertTrue(runner.layout_ready(layout, observations))
        observations["first"] = {"error": "node unreachable"}
        self.assertFalse(runner.layout_ready(layout, observations))

    def test_wait_layout_keeps_the_original_selected_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            job_dir = study / "job"
            job_dir.mkdir()
            write_json(study / "plan.json", {"jobs": []})
            control = runner.Controller(study)
            control.job = {"name": "job", "run": str(study / "run"), "directory": str(job_dir)}
            baseline = {"gpu_map": {"controller": "4,5,6,7"}, "controller_host": "controller"}
            with patch.object(control, "probe", side_effect=[{"available": [0, 1, 2, 3]},
                                                               {"available": list(range(8))}]) as probe, \
                 patch.object(control, "pause") as pause, \
                 patch.object(runner.subprocess, "Popen") as launch:
                control.wait_layout(baseline)
            self.assertEqual(probe.call_count, 2)
            pause.assert_called_once()
            launch.assert_not_called()
            self.assertEqual(baseline["gpu_map"], {"controller": "4,5,6,7"})
            self.assertEqual(json.loads((job_dir / "resource_check.json").read_text()),
                             {"controller": {"available": list(range(8))}})

    def test_unreachable_probe_returns_no_available_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            write_json(study / "plan.json", {"jobs": []})
            control = runner.Controller(study)
            with patch.object(runner.subprocess, "check_output", side_effect=subprocess.TimeoutExpired("ssh", 25)):
                result = control.probe("remote", "local")
            self.assertEqual(result["available"], [])
            self.assertIn("error", result)

    def test_frozen_source_rejects_file_and_directory_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "model.py").write_text("frozen implementation\n")
            for directory in (False, True):
                with self.subTest(directory=directory):
                    target = root / ("environment" if directory else "weights")
                    target.mkdir() if directory else target.write_bytes(b"asset")
                    link = source / "external"
                    link.symlink_to(target, target_is_directory=directory)
                    with self.assertRaisesRegex(ValueError, "symlinks"):
                        runner.source_manifest(source)
                    link.unlink()

    def test_input_hash_check_rejects_changed_and_missing_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / "frozen.json"
            asset.write_text("original")
            hashes = {str(asset): runner.digest(asset)}
            runner.check_hashes(hashes)
            asset.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.check_hashes(hashes)
            asset.unlink()
            with self.assertRaises(OSError):
                runner.check_hashes(hashes)

    def test_controller_validation_rejects_a_mutated_frozen_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            asset = study / "official_stats.json"
            asset.write_text("original normalization statistics")
            write_json(study / "plan.json", {"jobs": []})
            write_json(study / "frozen_inputs.json", {str(asset): runner.digest(asset)})
            control = runner.Controller(study)
            control.validate_inputs()
            asset.write_text("changed normalization statistics")
            with self.assertRaisesRegex(ValueError, "changed"):
                control.validate_inputs()

    def test_candidate_source_and_normalization_must_match_frozen_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            source = study / "source"
            source.mkdir()
            (source / "model.py").write_text("baseline implementation\n")
            run = study / "run"
            shutil.copytree(source, run / "source_snapshot")
            reference = study / "reference"
            write_json(reference / "dataset_statistics.json", {"franka": {"q01": [0.]}})
            write_json(run / "dataset_statistics.json", {"franka": {"q01": [0.]}})
            write_json(study / "plan.json", {"jobs": []})
            control = runner.Controller(study)
            baseline = {"source": str(source), "run": str(reference)}
            control.validate_run_source(run, baseline)
            control.validate_statistics(run, baseline)
            (run / "source_snapshot/model.py").write_text("changed world model implementation\n")
            with self.assertRaisesRegex(ValueError, "source differs"):
                control.validate_run_source(run, baseline)
            write_json(run / "dataset_statistics.json", {"franka": {"q01": [.1]}})
            with self.assertRaisesRegex(ValueError, "normalization statistics"):
                control.validate_statistics(run, baseline)

    def test_checkpoint_requires_successful_final_training_and_rank_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            state = complete_run(run)
            expected = run / "checkpoints/steps_32_pytorch_model.pt"
            self.assertEqual(runner.validate_checkpoint(run, 32), expected)
            for bad_status in ("running", "failed", "stopped"):
                write_json(run / "cluster_status.json", {"status": bad_status, "completed_steps": 32, "world_size": 2})
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    runner.validate_checkpoint(run, 32)
            complete_run(run)
            (state / "random_states_1.pkl").unlink()
            with self.assertRaisesRegex(ValueError, "RNG"):
                runner.validate_checkpoint(run, 32)

    def test_checkpoint_rejects_wrong_step_world_size_empty_model_and_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            for metadata in ({"step": 31, "world_size": 2}, {"step": 32, "world_size": 1}):
                state = complete_run(run)
                write_json(state / "complete.json", metadata)
                with self.assertRaisesRegex(ValueError, "step/world-size"):
                    runner.validate_checkpoint(run, 32)
            complete_run(run)
            (run / "checkpoints/steps_32_pytorch_model.pt").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "Incomplete model"):
                runner.validate_checkpoint(run, 32)
            complete_run(run)
            write_json(run / "training_complete.json", {"completed_steps": 31})
            with self.assertRaisesRegex(ValueError, "marker"):
                runner.validate_checkpoint(run, 32)

    def test_smoke_rejects_nonfinite_unbounded_or_absent_teacher_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            complete_run(run)
            self.assertTrue(runner.validate_smoke(run)["finite"])
            valid = {"step": 32, "dino_future_loss": .2, "predicted_latent_rms": 1.}
            bad_rows = [{**valid, "other_loss": float("nan")},
                        {**valid, "other_loss": float("inf")},
                        {**valid, "predicted_latent_rms": 0.},
                        {"step": 32, "predicted_latent_rms": 1.},
                        {**valid, "step": 31}]
            for row in bad_rows:
                with self.subTest(row=row):
                    (run / "metrics.jsonl").write_text(json.dumps(row) + "\n")
                    with self.assertRaises(ValueError):
                        runner.validate_smoke(run)

    def test_robotwin_smoke_preserves_existing_dense_temporal_supervision(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            complete_run(run, temporal=.1)
            runner.validate_smoke(run)
            (run / "metrics.jsonl").write_text(json.dumps({"step": 32, "dino_future_loss": .2,
                                                            "predicted_latent_rms": 1.,
                                                            "temporal_dense_loss": 0.}) + "\n")
            with self.assertRaisesRegex(ValueError, "temporal supervision"):
                runner.validate_smoke(run)

    def test_both_benchmark_smokes_use_reference_seed_and_reject_partial_rollouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            write_json(study / "plan.json", {"jobs": []})
            control = runner.Controller(study)
            for benchmark in ("libero", "robotwin"):
                with self.subTest(benchmark=benchmark):
                    run = study / benchmark
                    complete_run(run)
                    (run / "source_snapshot").mkdir()
                    directory = study / f"job_{benchmark}"
                    directory.mkdir()
                    control.job = {"name": benchmark, "benchmark": benchmark, "run": str(run),
                                   "directory": str(directory)}
                    captured = []

                    def simulate(command, phase, cwd, env=None):
                        captured.append(command)
                        output = Path(command[command.index("--output") + 1])
                        result = ({"status": "complete", "episodes": 40} if benchmark == "libero"
                                  else {"state": "complete", "sources_unchanged": True, "trials": 1})
                        write_json(output / "summary.json", result)

                    with patch.object(control, "probe", return_value={"available": [0, 1]}), \
                         patch.object(control, "execute", side_effect=simulate), \
                         patch.object(runner, "choose_port_base", return_value=27000), \
                         patch.object(runner.subprocess, "Popen") as launch:
                        summary = control.evaluate(run, 32, {"controller_host": "local"}, smoke=True)
                        command = captured[0]
                        self.assertEqual(command[command.index("--seed") + 1], "7" if benchmark == "libero" else "0")
                        if benchmark == "libero":
                            self.assertEqual(command[command.index("--trials") + 1], "1")
                            self.assertEqual(command[command.index("--gpus") + 1], "0")
                            bad_results = [{"status": "complete", "episodes": 39}]
                        else:
                            self.assertEqual(command[command.index("--episodes") + 1], "1")
                            self.assertEqual(command[command.index("--image-channel-order") + 1], "rgb")
                            self.assertEqual(command[command.index("--tasks") + 1:], ["adjust_bottle"])
                            bad_results = [{"state": "complete", "sources_unchanged": True, "trials": 0},
                                           {"state": "complete", "sources_unchanged": False, "trials": 1}]
                        for bad_result in bad_results:
                            write_json(summary, bad_result)
                            with self.assertRaisesRegex(ValueError, "Incomplete"):
                                control.evaluate(run, 32, {"controller_host": "local"}, smoke=True)
                    launch.assert_not_called()

    def test_robotwin_selection_requires_every_libero_arm_to_be_complete_and_fully_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            jobs = []
            for variant in ("last_layer", "late_layers", "wide_layers"):
                directory = study / variant
                jobs.append({"benchmark": "libero", "variant": variant, "directory": str(directory),
                             "max_steps": 80000})
                write_json(directory / "comparison_step80000.json", {
                    "complete": True, "fully_paired": True, "candidate": {"successes": 300},
                    "paired": {"gains": 10, "losses": 10}})
            write_json(study / "plan.json", {"jobs": jobs})
            control = runner.Controller(study)
            for field in ("complete", "fully_paired"):
                invalid = {"complete": True, "fully_paired": True, "candidate": {"successes": 399},
                           "paired": {"gains": 99, "losses": 0}}
                invalid[field] = False
                write_json(Path(jobs[-1]["directory"]) / "comparison_step80000.json", invalid)
                with self.assertRaisesRegex(ValueError, "complete and use identical initial states"):
                    control.select_robotwin_variant()
                self.assertFalse((study / "robotwin_selection.json").exists())

    def test_robotwin_selection_uses_the_80k_primary_results_and_stable_queue_ties(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            jobs = []
            for variant, primary_successes in (("last_layer", 290), ("late_layers", 310), ("wide_layers", 310)):
                directory = study / variant
                jobs.append({"benchmark": "libero", "variant": variant, "directory": str(directory),
                             "max_steps": 80000})
                write_json(directory / "comparison_step40000.json", {
                    "complete": True, "fully_paired": True,
                    "candidate": {"successes": 400 if variant == "last_layer" else 0},
                    "paired": {"gains": 0, "losses": 0}})
                write_json(directory / "comparison_step80000.json", {
                    "complete": True, "fully_paired": True, "candidate": {"successes": primary_successes},
                    "paired": {"gains": max(primary_successes - 300, 0),
                               "losses": max(300 - primary_successes, 0)}})
            # A RoboTwin report is never an input to the LIBERO-based selector.
            jobs.append({"benchmark": "robotwin", "variant": "last_layer", "directory": "not-created"})
            write_json(study / "plan.json", {"jobs": jobs})
            control = runner.Controller(study)
            self.assertEqual(control.select_robotwin_variant(), "late_layers")
            self.assertEqual(control.select_robotwin_variant(), "late_layers")
            selection = json.loads((study / "robotwin_selection.json").read_text())
            self.assertEqual(selection["variant"], "late_layers")
            self.assertEqual([score[0] for score in selection["scores"]], [290, 310, 310])

    def test_frozen_support_controller_uses_plan_root_for_runtime_links_and_eval_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "study"
            runtime_root = Path(tmp) / "project"
            assets = (".venv", "playground/LIBERO", ".cache/libero_eval/egl")
            for relative in assets:
                (runtime_root / relative).mkdir(parents=True)
            write_json(study / "plan.json", {"root": str(runtime_root), "jobs": []})
            with patch.object(runner, "ROOT", study / "support"):
                control = runner.Controller(study)
            self.assertEqual(control.root, runtime_root)
            run = study / "run"
            complete_run(run)
            snapshot = run / "source_snapshot"
            snapshot.mkdir()
            directory = study / "job"
            directory.mkdir()
            control.job = {"benchmark": "libero", "name": "libero", "directory": str(directory), "run": str(run)}

            def simulate(command, phase, cwd, env):
                self.assertEqual(cwd, snapshot)
                self.assertEqual(env["PYTHONPATH"], str(snapshot))
                write_json(Path(command[command.index("--output") + 1]) / "summary.json",
                           {"status": "complete", "episodes": 40})

            with patch.object(control, "probe", return_value={"available": [0]}), \
                 patch.object(control, "execute", side_effect=simulate), \
                 patch.object(runner, "choose_port_base", return_value=27000), \
                 patch.object(runner.subprocess, "Popen") as launch:
                control.evaluate(run, 32, {"controller_host": "local"}, smoke=True)
            launch.assert_not_called()
            for relative in assets:
                self.assertTrue((snapshot / relative).is_symlink())
                self.assertEqual((snapshot / relative).resolve(), runtime_root / relative)

    def test_shared_gpu_lock_remains_held_through_smoke_training_and_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            directory = study / "job"
            source = study / "source"
            source.mkdir()
            (source / "model.py").write_text("frozen model implementation\n")
            run = study / "new_run"
            lock_path = study / "shared_gpu.lock"
            reference_run = study / "reference_run"
            write_json(reference_run / "dataset_statistics.json", {"franka": {"q01": [0.], "q99": [1.]}})
            baseline = {"run": str(reference_run), "source": str(source), "config": str(study / "base.yaml"),
                        "gpu_map": {"local": "0,1"}, "controller_host": "local",
                        "summaries": {"50": "reference_50", "100": "reference_100"}}
            job = {"name": "libero_late_layers", "benchmark": "libero", "variant": "late_layers",
                   "run": str(run), "directory": str(directory), "config": str(study / "arm.yaml"),
                   "max_steps": 100, "eval_steps": [50, 100]}
            write_json(directory / "status.json", {"status": "queued"})
            write_json(study / "plan.json", {"gpu_lock": str(lock_path), "jobs": [job],
                                             "baselines": {"libero": baseline},
                                             "support": {"compare_feature_ablation.py": "compare.py"}})
            control = runner.Controller(study)
            phases = []

            def require_held_lock():
                with lock_path.open("a") as other:
                    with self.assertRaises(BlockingIOError):
                        runner.fcntl.flock(other, runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)

            def execute(command, phase, cwd, env=None):
                require_held_lock()
                phases.append(phase)
                if phase == "smoke_training":
                    smoke = run.with_name(run.name + "_preflight")
                    complete_run(smoke)
                    shutil.copytree(source, smoke / "source_snapshot")
                elif phase == "training":
                    complete_run(run, step=100)
                    shutil.copytree(source, run / "source_snapshot")

            def evaluate(eval_run, step, eval_baseline, smoke=False):
                require_held_lock()
                phases.append("evaluation_smoke" if smoke else f"evaluation_{step}")
                return directory / f"summary_{step}.json"

            with patch.object(control, "validate_inputs"), \
                 patch.object(control, "wait_layout", side_effect=lambda _: require_held_lock()) as resources, \
                 patch.object(control, "execute", side_effect=execute), \
                 patch.object(control, "evaluate", side_effect=evaluate), \
                 patch.object(runner, "make_config", return_value=OmegaConf.create({"run_id": "test"})), \
                 patch.object(runner.subprocess, "Popen") as launch:
                control.run_job(job)
            launch.assert_not_called()
            self.assertEqual(resources.call_count, 2)
            self.assertEqual(phases, ["smoke_training", "evaluation_smoke", "training", "evaluation_50",
                                      "compare_step50", "evaluation_100", "compare_step100"])
            self.assertEqual(json.loads((directory / "status.json").read_text())["status"], "complete")
            with lock_path.open("a") as other:
                runner.fcntl.flock(other, runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)


if __name__ == "__main__":
    unittest.main()
