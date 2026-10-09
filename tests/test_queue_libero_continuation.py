import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'queue_libero', Path(__file__).parents[1] / 'scripts/queue_libero_continuation.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class QueueSafetyTest(unittest.TestCase):
    def test_memory_and_utilization_both_required(self):
        self.assertEqual(queue.available_gpus('0, 0, 0\n1, 201, 0\n2, 0, 6\n3, 200, 5\n'), [0, 3])

    def check_waits_without_launching(self, probe_result=None, error=None):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            control = run / 'continuation'
            control.mkdir()
            plan = dict(root=folder, hosts=['local'], controller_host='local', gpus=[0, 1])
            (control / 'plan.json').write_text(json.dumps(plan))
            handlers = {}

            def register(sig, handler):
                handlers[sig] = handler

            def stop_when_waiting(_):
                state = json.loads((control / 'status.json').read_text())
                self.assertEqual(state['status'], 'waiting_gpus')
                handlers[signal.SIGTERM](signal.SIGTERM, None)

            with patch.object(queue.sys, 'argv', ['queue', '--run', folder]), \
                 patch.object(queue.signal, 'signal', side_effect=register), \
                 patch.object(queue.subprocess, 'run', return_value=probe_result, side_effect=error), \
                 patch.object(queue.subprocess, 'Popen') as launch, \
                 patch.object(queue.time, 'sleep', side_effect=stop_when_waiting):
                queue.main()
            launch.assert_not_called()
            self.assertEqual(json.loads((control / 'status.json').read_text())['status'], 'stopped')

    def test_busy_gpu_prevents_launch(self):
        self.check_waits_without_launching(subprocess.CompletedProcess([], 0, '0, 0, 0\n1, 5000, 80\n'))

    def test_unreachable_node_prevents_launch(self):
        self.check_waits_without_launching(error=subprocess.TimeoutExpired('ssh', 20))

    def test_missing_gpu_prevents_launch(self):
        self.check_waits_without_launching(subprocess.CompletedProcess([], 0, '0, 0, 0\n'))

    def test_free_cluster_launches_once_and_checks_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            control = run / 'continuation'
            control.mkdir()
            plan = dict(root=folder, hosts=['local'], controller_host='local', gpus=[0, 1],
                        target_step=240000, training_command=['trainer'], evaluation_steps=[])
            (control / 'plan.json').write_text(json.dumps(plan))
            (run / 'training_complete.json').write_text(json.dumps({'completed_steps': 240000}))
            probe = subprocess.CompletedProcess([], 0, '0, 0, 0\n1, 0, 0\n')
            with patch.object(queue.sys, 'argv', ['queue', '--run', folder]), \
                 patch.object(queue.signal, 'signal'), \
                 patch.object(queue.subprocess, 'run', return_value=probe), \
                 patch.object(queue.subprocess, 'Popen') as launch:
                launch.return_value.poll.return_value = 0
                launch.return_value.returncode = 0
                launch.return_value.pid = 123
                queue.main()
            launch.assert_called_once()
            self.assertEqual(json.loads((control / 'status.json').read_text())['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
