import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_feature_ablation import ComparisonError, compare_results, exact_mcnemar, main


# The historical LIBERO identity can be reconstructed only from an archived
# evaluator with this verified contract. This fixture is parsed, never run.
ARCHIVED_LIBERO_SOURCE = '''
def _get_libero_env(task, resolution, seed):
    env.seed(seed)
    return env, task_description

def eval_libero(args):
    for task_id in task_ids:
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            obs = env.set_init_state(initial_states[episode_idx])
            logging.info(f"Starting episode {task_episodes + 1}...")
            while not done:
                if waiting:
                    continue
                break
            task_episodes += 1
'''


class FeatureAblationComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def robotwin(self, name, episodes, *, requested=None, state="complete", changes=None):
        directory = self.root / name
        directory.mkdir()
        requested = len(episodes) if requested is None else requested
        protocol = {"tasks": ["task_a"], "task_config": "demo_clean", "episodes_per_task": requested,
                    "seed": 0, "execute_horizon": 16, "action_chunk_size": 32,
                    "smooth_actions": True, "task_success": "official eval_success", "expert_filter": True,
                    "simulator_image_channel_order": "rgb", "policy_image_channel_order": "rgb",
                    "swap_rb_before_normalization": False, "robotwin_commit": "same_simulator",
                    "environment": {"packages": {"sapien": "3.0.0b1"}},
                    "source_sha256": {"/snapshot/eval_policy.py": "same_hash"}}
        protocol.update(changes or {})
        successes = sum(success for seed, success in episodes)
        summary = {"state": state, "completed_tasks": 1, "total_tasks": 1, "trials": len(episodes),
                   "successes": successes, "sources_unchanged": True, "failed_tasks": [],
                   "results": [{"task": "task_a", "state": "complete", "trials": len(episodes),
                                "successes": successes,
                                "episodes": [{"seed": seed, "success": success} for seed, success in episodes]}]}
        (directory / "protocol.json").write_text(json.dumps(protocol))
        path = directory / "summary.json"
        path.write_text(json.dumps(summary))
        return path

    def libero(self, name, successes, *, identities=None, snapshot=False, requested=None,
               state="complete", argument_changes=None):
        directory = self.root / name / "evaluations" / "test"
        directory.mkdir(parents=True)
        requested = len(successes) if requested is None else requested
        arguments = {"task_suite_name": "libero_10", "start_task": 0, "max_tasks": 1,
                     "num_trials_per_task": requested, "seed": 7, "num_steps_wait": 10,
                     "unnorm_key": "franka", "post_process_action": True, "execute_horizon": 8,
                     "temporal_action_ensemble": False, "adaptive_ensemble_alpha": 0.0}
        arguments.update(argument_changes or {})
        lines = ["INFO | Arguments: " + json.dumps(arguments, indent=2)]
        for index, success in enumerate(successes):
            lines.append(f"INFO | Starting episode {index + 1}...")
            if identities is not None:
                seed, init_index = identities[index]
                lines.append(f"INFO | Initial state index: {init_index}, seed: {seed}")
            lines += [f"INFO | Success: {success}", f"INFO | # episodes completed so far: {index + 1}",
                      f"INFO | # successes: {sum(successes[:index + 1])} (rate)"]
        if state == "complete":
            lines.append(f"INFO | Total episodes: {len(successes)}")
        log = directory / "libero_10_task00.log"
        log.write_text("\n".join(lines))
        if snapshot:
            source = self.root / name / "source_snapshot/examples/LIBERO/eval_files/eval_libero.py"
            source.parent.mkdir(parents=True)
            source.write_text(ARCHIVED_LIBERO_SOURCE)
        plan = {"seed": 7, "execute_horizon": 8, "unnorm_key": "franka", "trials_per_task": requested,
                "suites": ["libero_10"], "total_tasks": 1, "total_episodes": requested,
                "libero_commit": "same_simulator", "precision": "same_fp32_bf16"}
        summary = {"status": state, "plan": plan, "episodes": len(successes), "successes": sum(successes),
                   "suites": {"libero_10": {"episodes": len(successes), "successes": sum(successes)}},
                   "tasks": [{"suite": "libero_10", "task_id": 0, "episodes": len(successes),
                              "successes": sum(successes), "exit_code": 0, "log": str(log)}]}
        path = directory / "summary.json"
        path.write_text(json.dumps(summary))
        return path

    def test_exact_mcnemar_two_sided(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_mcnemar(1, 1), 1.0)
        self.assertEqual(exact_mcnemar(6, 0), 0.03125)
        self.assertEqual(exact_mcnemar(0, 6), 0.03125)

    def test_robotwin_pairs_actual_seeds_despite_order(self):
        baseline = self.robotwin("baseline", [(10, False), (11, True), (12, False)])
        candidate = self.robotwin("candidate", [(12, True), (10, False), (11, False)])
        report = compare_results("robotwin", baseline, candidate)
        self.assertEqual(report["paired"]["gains"], 1)
        self.assertEqual(report["paired"]["losses"], 1)
        self.assertTrue(report["fully_paired"])
        self.assertEqual(report["tasks"][0]["paired"], report["paired"])

    def test_robotwin_different_seeds_never_pair_by_ordinal(self):
        baseline = self.robotwin("baseline", [(10, False), (11, True)])
        candidate = self.robotwin("candidate", [(11, True), (12, True)])
        report = compare_results("robotwin", baseline, candidate)
        self.assertEqual(report["paired"]["intersection_episodes"], 1)
        self.assertEqual(report["paired"]["baseline_coverage"], 0.5)
        self.assertEqual(report["paired"]["gains"], 0)
        self.assertEqual(report["delta_percentage_points"], 50)
        self.assertFalse(report["paired_screening_signal"])
        self.assertEqual(report["evidence_status"], "preliminary_partial_episode_overlap")

    def test_incomplete_robotwin_result_cannot_pass_screening(self):
        baseline = self.robotwin("baseline", [(10, False), (11, False)])
        candidate = self.robotwin("candidate", [(10, True)], requested=2, state="running")
        report = compare_results("robotwin", baseline, candidate)
        self.assertFalse(report["complete"])
        self.assertFalse(report["effective"])
        self.assertFalse(report["paired_screening_signal"])
        self.assertEqual(report["evidence_status"], "incomplete_or_unverified")

    def test_positive_signal_remains_single_seed_preliminary(self):
        baseline = self.robotwin("baseline", [(seed, False) for seed in range(6)])
        candidate = self.robotwin("candidate", [(seed, True) for seed in range(6)])
        report = compare_results("robotwin", baseline, candidate)
        self.assertTrue(report["paired_screening_signal"])
        self.assertFalse(report["effective"])
        self.assertFalse(report["feature_attribution_verified"])
        self.assertEqual(report["evidence_status"], "preliminary_positive_signal")
        self.assertEqual(report["evidence_scope"], "preliminary_single_training_seed")

    def test_protocol_changes_reject_comparison(self):
        for key, value in (("execute_horizon", 8), ("robotwin_commit", "other_simulator"),
                           ("environment", {"packages": {"sapien": "different"}}),
                           ("source_sha256", {"/another/eval_policy.py": "different_hash"})):
            with self.subTest(key=key):
                baseline = self.robotwin(f"baseline_{key}", [(10, False)])
                candidate = self.robotwin(f"candidate_{key}", [(10, True)], changes={key: value})
                with self.assertRaisesRegex(ComparisonError, "Incompatible evaluation protocols"):
                    compare_results("robotwin", baseline, candidate)

    def test_robotwin_duplicate_seed_rejected(self):
        baseline = self.robotwin("baseline", [(10, False), (11, False)])
        candidate = self.robotwin("candidate", [(10, True), (10, True)])
        with self.assertRaisesRegex(ComparisonError, "Duplicate RoboTwin"):
            compare_results("robotwin", baseline, candidate)

    def test_robotwin_corrupt_summary_count_rejected(self):
        baseline = self.robotwin("baseline", [(10, False)])
        candidate = self.robotwin("candidate", [(10, True)])
        summary = json.loads(candidate.read_text())
        summary["successes"] = 0
        candidate.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ComparisonError, "summary totals disagree"):
            compare_results("robotwin", baseline, candidate)

    def test_libero_explicit_init_indices_pair_real_identities(self):
        baseline = self.libero("baseline", [False, True], identities=[(7, 5), (7, 9)])
        candidate = self.libero("candidate", [False, True], identities=[(7, 9), (7, 5)])
        report = compare_results("libero", baseline, candidate)
        self.assertTrue(report["complete"])
        self.assertEqual(report["paired"]["gains"], 1)
        self.assertEqual(report["paired"]["losses"], 1)
        self.assertEqual(report["tasks"][0]["task"], 0)

    def test_libero_archived_source_verifies_old_log_init_mapping(self):
        baseline = self.libero("baseline", [False, True], snapshot=True)
        candidate = self.libero("candidate", [True, True], snapshot=True)
        report = compare_results("libero", baseline, candidate)
        self.assertTrue(report["complete"])
        self.assertEqual(report["paired"]["intersection_episodes"], 2)
        self.assertEqual(report["paired"]["gains"], 1)
        self.assertEqual(report["validation"]["baseline"]["identity_provenance"][0]["method"],
                         "archived_fixed_official_init_state_loop")

    def test_libero_without_snapshot_does_not_invent_init_identity(self):
        baseline = self.libero("baseline", [False, True])
        candidate = self.libero("candidate", [True, True])
        report = compare_results("libero", baseline, candidate)
        self.assertEqual(report["baseline"]["successes"], 1)
        self.assertEqual(report["candidate"]["successes"], 2)
        self.assertEqual(report["paired"]["intersection_episodes"], 0)
        self.assertIsNone(report["paired"]["mcnemar_exact_two_sided_p"])
        self.assertFalse(report["complete"])
        self.assertFalse(report["paired_screening_signal"])

    def test_libero_unrecognized_snapshot_cannot_justify_ordinal_pairing(self):
        baseline = self.libero("baseline", [False, True], snapshot=True)
        candidate = self.libero("candidate", [True, True], snapshot=True)
        source = self.root / "candidate/source_snapshot/examples/LIBERO/eval_files/eval_libero.py"
        source.write_text(ARCHIVED_LIBERO_SOURCE.replace("initial_states[episode_idx]", "initial_states[episode_idx + 10]"))
        report = compare_results("libero", baseline, candidate)
        self.assertEqual(report["paired"]["intersection_episodes"], 0)
        self.assertFalse(report["protocol_comparable"])
        self.assertFalse(report["complete"])

    def test_libero_missing_final_log_summary_blocks_complete_claim(self):
        baseline = self.libero("baseline", [False], snapshot=True)
        candidate = self.libero("candidate", [True], snapshot=True)
        log = candidate.parent / "libero_10_task00.log"
        log.write_text(log.read_text().replace("INFO | Total episodes: 1", ""))
        report = compare_results("libero", baseline, candidate)
        self.assertEqual(report["candidate"]["successes"], 1)
        self.assertFalse(report["complete"])

    def test_libero_missing_log_preserves_counts_without_claiming_improvement(self):
        baseline = self.libero("baseline", [False], snapshot=True)
        candidate = self.libero("candidate", [True], snapshot=True)
        (candidate.parent / "libero_10_task00.log").unlink()
        report = compare_results("libero", baseline, candidate)
        self.assertEqual(report["candidate"]["successes"], 1)
        self.assertFalse(report["complete"])
        self.assertFalse(report["paired_screening_signal"])

    def test_libero_plan_log_seed_disagreement_rejected(self):
        baseline = self.libero("baseline", [False], snapshot=True)
        candidate = self.libero("candidate", [True], snapshot=True, argument_changes={"seed": 8})
        with self.assertRaisesRegex(ComparisonError, "Arguments disagree"):
            compare_results("libero", baseline, candidate)

    def test_libero_corrupt_counters_rejected(self):
        baseline = self.libero("baseline", [False], snapshot=True)
        candidate = self.libero("candidate", [True], snapshot=True)
        log = candidate.parent / "libero_10_task00.log"
        log.write_text(log.read_text().replace("# successes: 1", "# successes: 0"))
        with self.assertRaisesRegex(ComparisonError, "counters disagree"):
            compare_results("libero", baseline, candidate)

    def test_libero_incomplete_episode_budget_is_reported(self):
        baseline = self.libero("baseline", [False, False], snapshot=True)
        candidate = self.libero("candidate", [True], snapshot=True, requested=2, state="running")
        report = compare_results("libero", baseline, candidate)
        self.assertFalse(report["complete"])
        self.assertEqual(report["paired"]["intersection_episodes"], 1)
        self.assertEqual(report["validation"]["candidate"]["expected_episodes"], 2)

    def test_cli_writes_standard_library_report(self):
        baseline = self.robotwin("baseline", [(10, False)])
        candidate = self.robotwin("candidate", [(10, True)])
        output = self.root / "report/comparison.json"
        self.assertEqual(main(["--benchmark", "robotwin", "--baseline", str(baseline),
                               "--candidate", str(candidate), "--output", str(output)]), 0)
        report = json.loads(output.read_text())
        self.assertFalse(report["effective"])
        self.assertEqual(report["paired"]["mcnemar_exact_two_sided_p"], 1)

    def test_cli_rejects_overwriting_input_summary(self):
        baseline = self.robotwin("baseline", [(10, False)])
        candidate = self.robotwin("candidate", [(10, True)])
        original = baseline.read_bytes()
        with self.assertRaises(SystemExit):
            main(["--benchmark", "robotwin", "--baseline", str(baseline),
                  "--candidate", str(candidate), "--output", str(baseline)])
        self.assertEqual(baseline.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
