import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.cluster_dashboard import ClusterMonitor, load_nodes, probe_node
from scripts.cluster_probe import collect_history, collect_run, number, read_records


class ProbeTest(unittest.TestCase):
    def test_full_history_exceeds_tail_limit_and_appends_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.jsonl"
            path.write_text(''.join(json.dumps({"step": i, "loss": 1 / (i + 1), "padding": "x" * 3000})
                                    + '\n' for i in range(800)))
            rows, offset, identity = [], 0, None
            while True:
                page = collect_history(temp, offset, identity)
                rows.extend(page["records"])
                offset, identity = page["offset"], page["identity"]
                if page["done"]:
                    break
            self.assertEqual([row["step"] for row in rows], list(range(800)))
            self.assertNotIn("padding", rows[0])
            self.assertGreater(path.stat().st_size, 2 * 1024 * 1024)
            with path.open("a") as stream:
                stream.write('{"step":800,"loss":')
            partial = collect_history(temp, offset, identity)
            self.assertEqual(partial["offset"], offset)
            self.assertEqual(partial["records"], [])
            with path.open("a") as stream:
                stream.write('0.1}\n')
            appended = collect_history(temp, offset, identity)
            self.assertFalse(appended["reset"])
            self.assertEqual(appended["records"], [{"step": 800, "loss": 0.1}])
            path.write_text('{"step":0,"loss":2}\n')
            reset = collect_history(temp, appended["offset"], identity)
            self.assertTrue(reset["reset"])
            self.assertEqual(reset["records"], [{"step": 0, "loss": 2}])

    def test_history_filters_invalid_and_unfinished_lines(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.jsonl"
            path.write_text('bad\n{"step":true}\n{"step":1,"loss":NaN,"mse":2}\n{"step":2')
            page = collect_history(temp)
            self.assertEqual(page["parse_errors"], 2)
            self.assertEqual(page["records"], [{"step": 1, "mse": 2}])

    def test_partial_malformed_nonfinite_and_bounded_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.jsonl"
            path.write_text('oops\n{"step":true}\n{"step":"1"}\n{"step":1,"loss":NaN}\n'
                            '{"step":2,"loss":0.5}\n{"step":3,"loss":')
            records, errors = read_records(path, limit=2)
            self.assertEqual(errors, 3)
            self.assertEqual(records, [{"step": 1}, {"step": 2, "loss": 0.5}])
            self.assertIsNone(number("[N/A]"))
            self.assertIsNone(number("inf"))

    def test_status_does_not_claim_running_or_complete_from_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "metrics.jsonl"
            path.write_text('{"step":6,"loss":0.2}\n')
            self.assertEqual(collect_run(path)["state"], "unknown")
            (root / "config.full.yaml").write_text("trainer:\n  max_train_steps: 6\n")
            self.assertEqual(collect_run(path)["state"], "target_reached")
            (root / "STATUS.failed").touch()
            self.assertEqual(collect_run(path)["state"], "failed")
            (root / "train.pid").write_text("123")
            with patch("scripts.cluster_probe.os.kill", return_value=None):
                self.assertEqual(collect_run(path)["state"], "running")

    def test_log_rotation_and_partial_write(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.jsonl"
            path.write_text('{"step":100,"loss":0.1}\n')
            self.assertEqual(read_records(path)[0][0]["step"], 100)
            path.write_text('{"step":1,"loss":')
            self.assertEqual(read_records(path)[0], [])
            with path.open("a") as stream:
                stream.write('0.9}\n')
            self.assertEqual(read_records(path)[0][0]["step"], 1)


class ClusterTest(unittest.TestCase):
    def test_history_only_reads_discovered_runs(self):
        node = {"id": "local", "name": "local", "paths": []}
        monitor = ClusterMonitor([node])
        monitor.snapshots["local"]["runs"] = [{"id": "/known/run"}]
        with patch("scripts.cluster_dashboard.probe_node", return_value={"records": []}) as probe:
            with self.assertRaises(KeyError):
                monitor.history("local", "/etc")
            probe.assert_not_called()
            monitor.history("local", "/known/run", 123, "version")
            self.assertEqual(probe.call_args.kwargs["history"],
                             {"directory": "/known/run", "offset": 123, "identity": "version"})

    def test_failed_node_retains_last_sample_with_offline_status(self):
        node = {"id": "test", "name": "test", "paths": []}
        monitor = ClusterMonitor([node])
        with patch("scripts.cluster_dashboard.probe_node", return_value={"gpus": [{"index": "0"}]}):
            monitor.sample(node)
        with patch("scripts.cluster_dashboard.probe_node", side_effect=subprocess.TimeoutExpired("ssh", 20)):
            monitor.sample(node)
        data = monitor.payload()["nodes"][0]
        self.assertEqual(data["connection"], "offline")
        self.assertEqual(data["gpus"], [{"index": "0"}])
        self.assertIn("received_at", data)
        self.assertIn("timed out", data["error"])
        data["gpus"].clear()
        self.assertEqual(len(monitor.payload()["nodes"][0]["gpus"]), 1)

    def test_config_rejects_duplicate_ids_and_ssh_options(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nodes.json"
            for nodes in ([{"id": "x"}, {"id": "x"}], [{"id": "x", "host": "-oProxyCommand=bad"}]):
                path.write_text(json.dumps({"nodes": nodes}))
                with self.assertRaises(ValueError):
                    load_nodes(path)

    def test_remote_probe_is_streamed_and_shell_quotes_python(self):
        node = {"host": "192.0.2.1", "python": "/opt/my python", "paths": ["/tmp/my 'run"]}
        with patch("scripts.cluster_dashboard.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, '{"gpus":[]}', '')
            self.assertEqual(probe_node(node), {"gpus": []})
        args, kwargs = run.call_args
        self.assertEqual(args[0][-1], "'/opt/my python' -")
        self.assertIn("StrictHostKeyChecking=yes", args[0])
        self.assertIn("collect([", kwargs["input"])
        self.assertEqual(kwargs["timeout"], 20)


if __name__ == "__main__":
    unittest.main()
