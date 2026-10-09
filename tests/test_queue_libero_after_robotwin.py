import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import signal

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import queue_libero_after_robotwin as queue


class DependencyQueueTest(unittest.TestCase):
    def test_only_successful_complete_checkpoint_unlocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(queue.dependency_complete(root, 10))
            (root / 'cluster_status.json').write_text(json.dumps({'status': 'running'}))
            self.assertFalse(queue.dependency_complete(root, 10))
            for state in ('failed', 'stopped'):
                (root / 'cluster_status.json').write_text(json.dumps({'status': state}))
                with self.assertRaises(RuntimeError):
                    queue.dependency_complete(root, 10)
            (root / 'cluster_status.json').write_text(json.dumps({'status': 'complete', 'world_size': 2}))
            (root / 'training_complete.json').write_text(json.dumps({'completed_steps': 10}))
            saved = root / 'checkpoints/steps_10_training_state'
            saved.mkdir(parents=True)
            (saved / 'complete.json').write_text(json.dumps({'step': 10, 'world_size': 2}))
            with self.assertRaises(ValueError):
                queue.dependency_complete(root, 10)
            for rank in range(2):
                (saved / f'random_states_{rank}.pkl').write_bytes(b'complete')
            self.assertTrue(queue.dependency_complete(root, 10))
            (saved / 'complete.json').write_text(json.dumps({'step': 9, 'world_size': 2}))
            with self.assertRaises(ValueError):
                queue.dependency_complete(root, 10)

    def test_selects_40_or_falls_back_without_using_busy_gpus(self):
        full = [{'host': str(i), 'available': list(range(8))} for i in range(5)]
        self.assertEqual(sum(len(g.split(',')) for g in queue.select_layout(full, '0').values()), 40)
        full[0]['available'] = list(range(1, 8))
        layout = queue.select_layout(full, '0')
        self.assertEqual(sum(len(g.split(',')) for g in layout.values()), 32)
        self.assertNotIn('0', layout.get('0', '').split(','))
        offset = 0
        for devices in layout.values():
            count = len(devices.split(','))
            self.assertEqual(offset % count, 0)
            offset += count
        for row in full:
            row['available'] = []
        self.assertIsNone(queue.select_layout(full, '0'))

    def test_waiting_parent_never_probes_or_launches_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'plan.json').write_text(json.dumps({'root': tmp, 'run': str(base / 'new_run'), 'dependency_run': str(base / 'parent'), 'dependency_target_step': 10}))
            handlers = {}
            def sleep(_):
                self.assertEqual(json.loads((base / 'status.json').read_text())['status'], 'waiting_robotwin')
                handlers[signal.SIGTERM]()
            with patch.object(queue.sys, 'argv', ['queue', '--queue', tmp]), patch.object(queue.signal, 'signal', side_effect=lambda sig, handler: handlers.update({sig: handler})), patch.object(queue.time, 'sleep', side_effect=sleep), patch.object(queue.subprocess, 'Popen') as launch, patch.object(queue.subprocess, 'run') as probe:
                queue.main()
            launch.assert_not_called()
            probe.assert_not_called()
            self.assertEqual(json.loads((base / 'status.json').read_text())['status'], 'stopped')


if __name__ == '__main__':
    unittest.main()
