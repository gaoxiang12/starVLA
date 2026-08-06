import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.training_dashboard import JsonlMetricStore, RunCatalog, read_max_train_steps, run_status


class JsonlMetricStoreTest(unittest.TestCase):
    def test_incremental_reads_and_incomplete_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.jsonl"
            path.write_text('{"step": 0, "loss": 2.0}\n{"step": 1, "loss":', encoding="utf-8")
            store = JsonlMetricStore(path)

            first = store.payload(None, 0)
            self.assertTrue(first["reset"])
            self.assertEqual(first["records"], [{"step": 0, "loss": 2.0}])
            self.assertEqual(first["parse_errors"], 0)

            with path.open("a", encoding="utf-8") as stream:
                stream.write(" 1.5}\n")
            second = store.payload(first["generation"], first["cursor"])

            self.assertFalse(second["reset"])
            self.assertEqual(second["records"], [{"step": 1, "loss": 1.5}])
            self.assertEqual(second["cursor"], 2)
            self.assertEqual(second["metric_names"], ["loss"])

    def test_truncation_starts_new_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.jsonl"
            path.write_text('{"step": 10, "loss": 1}\n{"step": 20, "loss": 2}\n', encoding="utf-8")
            store = JsonlMetricStore(path)
            first = store.payload(None, 0)

            path.write_text('{"step": 0, "new_loss": 3}\n', encoding="utf-8")
            second = store.payload(first["generation"], first["cursor"])

            self.assertTrue(second["reset"])
            self.assertNotEqual(second["generation"], first["generation"])
            self.assertEqual(second["records"], [{"step": 0, "new_loss": 3}])
            self.assertEqual(second["metric_names"], ["new_loss"])

    def test_malformed_and_non_finite_records_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.jsonl"
            lines = [
                "not json",
                json.dumps({"loss": 1.0}),
                json.dumps({"step": 3, "loss": 2.0, "label": "ignored", "flag": True}),
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            payload = JsonlMetricStore(path).payload(None, 0)

            self.assertEqual(payload["parse_errors"], 2)
            self.assertEqual(payload["records"], [{"step": 3, "loss": 2.0}])


class RunCatalogTest(unittest.TestCase):
    def test_discovers_runs_newest_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = root / "old"
            new = root / "nested" / "new"
            old.mkdir()
            new.mkdir(parents=True)
            (old / "metrics.jsonl").write_text('{"step": 0}\n', encoding="utf-8")
            (new / "metrics.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
            (old / "metrics.jsonl").touch()
            (new / "metrics.jsonl").touch()
            old_time = (old / "metrics.jsonl").stat().st_mtime - 10
            os.utime(old / "metrics.jsonl", (old_time, old_time))

            runs = RunCatalog(root, scan_interval=0).scan(force=True)

            self.assertEqual([run.run_id for run in runs], ["nested/new", "old"])
            self.assertEqual(runs[0].name, "new")

    def test_reads_target_step_and_reports_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "config.yaml").write_text(
                "trainer:\n  max_train_steps: 12345  # experiment target\n",
                encoding="utf-8",
            )
            (run_dir / "STATUS.running").touch()
            (run_dir / "train.pid").write_text("4321\n", encoding="utf-8")

            with patch("scripts.training_dashboard.pid_is_alive", return_value=False):
                status = run_status(run_dir, metrics_mtime=1.0)

            self.assertEqual(read_max_train_steps(run_dir), 12345)
            self.assertEqual(status["state"], "stale")
            self.assertEqual(status["pid"], 4321)


if __name__ == "__main__":
    unittest.main()
