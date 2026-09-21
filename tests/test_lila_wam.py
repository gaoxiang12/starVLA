import unittest
from collections import deque
from unittest.mock import Mock, patch

import numpy as np

from examples.Robotwin.eval_files.lila_wam_interface import ModelClient, endpose_state
from examples.Robotwin.eval_files.run_lila_benchmark import parse_counters


class LiLaContractTest(unittest.TestCase):
    def setUp(self):
        capture = patch("examples.Robotwin.eval_files.lila_wam_interface.capture_cuda_rng",
                        return_value=np.zeros(16, dtype=np.uint8))
        restore = patch("examples.Robotwin.eval_files.lila_wam_interface.restore_cuda_rng")
        self.capture_rng, self.restore_rng = capture.start(), restore.start()
        self.addCleanup(capture.stop)
        self.addCleanup(restore.stop)

    def observation(self):
        return {"endpose": {"left_endpose": np.arange(7), "left_gripper": 7,
                            "right_endpose": np.arange(8, 15), "right_gripper": 15},
                "observation": {"head_camera": {"rgb": np.zeros((240, 320, 3), np.uint8)}}}

    def test_pose_state_preserves_official_order(self):
        np.testing.assert_array_equal(endpose_state(self.observation()), np.arange(16))

    def test_chunk_execution_and_episode_reset(self):
        client = ModelClient.__new__(ModelClient)
        client.client = Mock()
        chunk = np.arange(32 * 14).reshape(1, 32, 14)
        client.client.predict_action.return_value = {"ok": True, "data": {
            "actions": chunk, "cuda_rng_state": np.ones(16, dtype=np.uint8)}}
        client.horizon, client.chunk_size, client.actions = 16, 32, deque()
        obs = self.observation()
        for index in range(16):
            np.testing.assert_array_equal(client.step(obs, "click_bell"), chunk[0, index])
        self.assertEqual(client.client.predict_action.call_count, 1)
        np.testing.assert_array_equal(client.step(obs, "click_bell"), chunk[0, 0])
        client.reset()
        np.testing.assert_array_equal(client.step(obs, "click_bell"), chunk[0, 0])
        self.assertEqual(client.client.predict_action.call_count, 3)
        self.assertEqual(self.capture_rng.call_count, 3)
        self.assertEqual(self.restore_rng.call_count, 3)
        np.testing.assert_array_equal(self.restore_rng.call_args.args[0], np.ones(16, dtype=np.uint8))

    def test_inference_error_is_not_a_policy_failure(self):
        client = ModelClient.__new__(ModelClient)
        client.client = Mock()
        client.client.predict_action.return_value = {"ok": False, "error": "OOM"}
        client.actions = deque()
        with self.assertRaises(RuntimeError):
            client.step(self.observation(), "click_bell")

    def test_counters_require_complete_unselected_episodes(self):
        text = "Success rate: \x1b[96m1/1\x1b[0m => 100%, current seed: 100002\n"
        text += "Success rate: 1/2 => 50%, current seed: 100004\n"
        self.assertEqual(parse_counters(text, 2), [
            {"seed": 100002, "success": True}, {"seed": 100004, "success": False}])
        for bad_text, count in [(text, 3), (text + text, 4), (text.replace("1/2", "3/2"), 2)]:
            with self.assertRaises(ValueError):
                parse_counters(bad_text, count)


if __name__ == "__main__":
    unittest.main()
