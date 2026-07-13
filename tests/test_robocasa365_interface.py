"""Request and action-shape tests for the RoboCasa365 eval bridge."""

import unittest
from unittest import mock

import numpy as np

from examples.Robocasa_365.eval_files import model2robocasa365_interface as m2r


def _fake_observations(batch_size: int = 1, image_hw: tuple = (32, 48)) -> dict:
    height, width = image_hw
    observations = {
        "annotation.human.task_description": ("open the drawer",) * batch_size,
        "video.robot0_agentview_left": np.full((batch_size, 1, height, width, 3), 10, dtype=np.uint8),
        "video.robot0_agentview_right": np.full((batch_size, 1, height, width, 3), 20, dtype=np.uint8),
        "video.robot0_eye_in_hand": np.full((batch_size, 1, height, width, 3), 30, dtype=np.uint8),
    }
    dimensions = [3, 4, 3, 4, 2]
    for key, dimension in zip(m2r.STATE_KEY_ORDER, dimensions):
        observations[key] = np.zeros((batch_size, 1, dimension), dtype=np.float32)
    return observations


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.requests = []

    def get_server_metadata(self) -> dict:
        return {"env": "test"}

    def predict_action(self, request: dict) -> dict:
        self.requests.append(request)
        batch_size = len(request["examples"])
        actions = np.arange(batch_size * 16 * 12).reshape(batch_size, 16, 12)
        return {"data": {"actions": actions.tolist()}}


class PolicyWarperRequestTest(unittest.TestCase):
    def _make_warper(self, **kwargs) -> m2r.PolicyWarper:
        with mock.patch.object(m2r, "WebsocketClientPolicy", _FakeClient):
            return m2r.PolicyWarper(policy_ckpt_path="unused", **kwargs)

    def test_three_views_follow_training_order(self):
        warper = self._make_warper()
        warper.step(_fake_observations())
        images = warper.client.requests[-1]["examples"][0]["image"]
        self.assertEqual(len(images), 3)
        self.assertEqual([int(image[0, 0, 0]) for image in images], [10, 20, 30])
        self.assertTrue(all(image.shape == (224, 224, 3) for image in images))

    def test_state_is_raw_16d_by_default(self):
        warper = self._make_warper()
        warper.step(_fake_observations())
        state = warper.client.requests[-1]["examples"][0]["state"]
        self.assertEqual(state.shape, (16,))

    def test_state_can_be_disabled(self):
        warper = self._make_warper(send_state=False)
        warper.step(_fake_observations())
        self.assertNotIn("state", warper.client.requests[-1]["examples"][0])

    def test_action_slices_match_gym_schema(self):
        warper = self._make_warper(n_action_steps=8)
        output = warper.step(_fake_observations())["actions"]
        self.assertEqual(output["action.end_effector_position"].shape, (1, 8, 3))
        self.assertEqual(output["action.end_effector_rotation"].shape, (1, 8, 3))
        self.assertEqual(output["action.gripper_close"].shape, (1, 8, 1))
        self.assertEqual(output["action.base_motion"].shape, (1, 8, 4))
        self.assertEqual(output["action.control_mode"].shape, (1, 8, 1))


if __name__ == "__main__":
    unittest.main()
