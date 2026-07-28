import unittest

from examples.LIBERO.eval_files.evaluate_latent_progress import (
    apply_deployed_ema,
    calibration_table,
    regression_metrics,
    trajectory_metrics,
)


class ProgressEvaluationMetricsTest(unittest.TestCase):
    def test_regression_metrics_identify_perfect_and_constant_predictions(self):
        target = [0.0, 0.25, 0.5, 0.75, 1.0]
        perfect = regression_metrics(target, target)
        constant = regression_metrics(target, [0.5] * len(target))

        self.assertEqual(perfect["mae"], 0.0)
        self.assertAlmostEqual(perfect["pearson"], 1.0)
        self.assertAlmostEqual(perfect["spearman"], 1.0)
        self.assertAlmostEqual(perfect["r_squared"], 1.0)
        self.assertIsNone(constant["pearson"])
        self.assertAlmostEqual(constant["r_squared"], 0.0)

    def test_calibration_includes_right_edge_in_final_bin(self):
        rows = calibration_table([0.0, 0.1, 0.99, 1.0], [0.1, 0.2, 0.9, 0.8])
        self.assertEqual(sum(row["count"] for row in rows), 4)
        self.assertEqual(rows[-1]["count"], 2)

    def test_trajectory_metrics_measure_monotonicity_per_episode(self):
        records = [
            {
                "episode_id": "a",
                "dataset": "suite",
                "instruction": "task a",
                "target": target,
                "prediction": prediction,
            }
            for target, prediction in ((0.0, 0.0), (0.5, 0.6), (1.0, 1.0))
        ]
        records.extend(
            {
                "episode_id": "b",
                "dataset": "suite",
                "instruction": "task b",
                "target": target,
                "prediction": prediction,
            }
            for target, prediction in ((0.0, 0.8), (0.5, 0.4), (1.0, 0.2))
        )

        metrics = trajectory_metrics(apply_deployed_ema(records, 0.5))
        self.assertEqual(metrics["num_episodes"], 2)
        self.assertAlmostEqual(
            metrics["episodes"][0]["nondecreasing_fraction"], 1.0
        )
        self.assertAlmostEqual(
            metrics["episodes"][1]["nondecreasing_fraction"], 0.0
        )

    def test_deployed_ema_resets_at_episode_boundary(self):
        records = [
            {"episode_id": "a", "target": 0.0, "prediction": 0.2},
            {"episode_id": "a", "target": 1.0, "prediction": 1.0},
            {"episode_id": "b", "target": 0.0, "prediction": 0.8},
        ]
        filtered = apply_deployed_ema(records, 0.75)
        self.assertAlmostEqual(filtered[0]["deployed_ema_prediction"], 0.2)
        self.assertAlmostEqual(filtered[1]["deployed_ema_prediction"], 0.4)
        self.assertAlmostEqual(filtered[2]["deployed_ema_prediction"], 0.8)


if __name__ == "__main__":
    unittest.main()
