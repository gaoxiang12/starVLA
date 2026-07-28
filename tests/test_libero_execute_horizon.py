import unittest
from unittest.mock import patch

import numpy as np

from examples.LIBERO.eval_files.model2libero_interface import ModelClient


class _FakePolicy:
    def __init__(self, *args, **kwargs):
        self.calls = 0
        self.payloads = []

    def get_server_metadata(self):
        return {"action_chunk_size": 8}

    def predict_action(self, payload):
        self.payloads.append(payload)
        base = 100 * self.calls
        self.calls += 1
        chunk = np.stack(
            [np.full(7, base + offset, dtype=np.float32) for offset in range(8)]
        )
        return {
            "data": {
                "actions": chunk[None],
                "progress": np.asarray([0.25 * self.calls]),
                "raw_progress": np.asarray([0.3 * self.calls]),
                "conditioning_progress": np.asarray([0.2 * self.calls]),
            }
        }


def _action_value(response):
    return float(response["raw_action"]["world_vector"][0])


class ExecuteHorizonTest(unittest.TestCase):
    @patch(
        "examples.LIBERO.eval_files.model2libero_interface.WebsocketClientPolicy",
        _FakePolicy,
    )
    def test_replans_from_start_of_new_chunk(self):
        client = ModelClient(execute_horizon=4, action_ensemble=False)
        example = {"image": [], "lang": "test"}

        self.assertEqual(
            [_action_value(client.step(example, step=i)) for i in range(6)],
            [0.0, 1.0, 2.0, 3.0, 100.0, 101.0],
        )
        self.assertEqual(client.client.calls, 2)

    @patch(
        "examples.LIBERO.eval_files.model2libero_interface.WebsocketClientPolicy",
        _FakePolicy,
    )
    def test_must_fit_model_chunk(self):
        for execute_horizon in (0, 9):
            with self.subTest(execute_horizon=execute_horizon):
                with self.assertRaisesRegex(ValueError, "execute_horizon"):
                    ModelClient(execute_horizon=execute_horizon)

    @patch(
        "examples.LIBERO.eval_files.model2libero_interface.WebsocketClientPolicy",
        _FakePolicy,
    )
    def test_episode_start_and_progress_are_forwarded(self):
        client = ModelClient(execute_horizon=4, action_ensemble=False)
        example = {"image": [], "lang": "test"}

        first = client.step(example, step=0)
        cached = client.step(example, step=1)

        self.assertTrue(
            client.client.payloads[0]["examples"][0]["episode_start"]
        )
        self.assertEqual(first["progress"], 0.25)
        self.assertAlmostEqual(first["raw_progress"], 0.3)
        self.assertAlmostEqual(first["conditioning_progress"], 0.2)
        self.assertTrue(first["progress_updated"])
        self.assertFalse(cached["progress_updated"])
        self.assertEqual(cached["progress"], first["progress"])
        self.assertEqual(client.client.calls, 1)

    @patch(
        "examples.LIBERO.eval_files.model2libero_interface.WebsocketClientPolicy",
        _FakePolicy,
    )
    def test_temporal_ensemble_aligns_overlapping_chunks(self):
        client = ModelClient(
            execute_horizon=4,
            action_ensemble=True,
            adaptive_ensemble_alpha=0.0,
        )
        example = {"image": [], "lang": "test"}

        values = [_action_value(client.step(example, step=i)) for i in range(6)]

        self.assertEqual(values[:4], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(values[4:], [52.0, 53.0])
        self.assertEqual(client.client.calls, 2)


if __name__ == "__main__":
    unittest.main()
