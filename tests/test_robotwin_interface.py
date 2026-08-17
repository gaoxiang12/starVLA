import unittest
from collections import deque

import numpy as np

from examples.Robotwin.eval_files import model2robotwin_interface as interface


class _FakePolicyClient:
    def __init__(self):
        self.last_request = None

    def predict_action(self, request):
        self.last_request = request
        actions = np.tile(np.arange(14, dtype=np.float32), (16, 1))
        return {"data": {"actions": [actions]}}


class _FakeTask:
    take_action_cnt = 0
    task_name = "click_bell"

    def __init__(self):
        self.action = None

    def get_instruction(self):
        return "test instruction"

    def take_action(self, action):
        self.action = action


class _RecordingModel:
    def __init__(self, task_language_mode="metadata"):
        self.example = None
        self.task_language_mode = task_language_mode

    def step(self, example, step=0):
        self.example = example
        return np.zeros(14, dtype=np.float32)


class RobotwinInterfaceTest(unittest.TestCase):
    def test_eval_reorders_raw_robotwin_state_for_model(self):
        task = _FakeTask()
        model = _RecordingModel()
        raw_state = np.arange(14, dtype=np.float32)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        observation = {
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image},
                "right_camera": {"rgb": image},
            },
            "joint_action": {"vector": raw_state},
        }

        interface.eval(task, model, observation)

        np.testing.assert_array_equal(
            model.example["state"],
            raw_state[interface.ROBOTWIN_TO_MODEL_JOINT_ORDER],
        )

    def test_step_preserves_state_and_reorders_action_for_robotwin(self):
        model = interface.ModelClient.__new__(interface.ModelClient)
        model.client = _FakePolicyClient()
        model.task_description = "test instruction"
        model.image_size = [224, 224]
        model.use_ddim = True
        model.num_ddim_steps = 10
        model.unnorm_key = "new_embodiment"
        model.action_chunk_size = 16
        model.raw_actions = None
        model.action_mode = "abs"
        model.initial_state = None
        model.prev_action = None
        model.visual_context_length = 1
        model.image_history = deque(maxlen=1)
        model.num_image_history = 0

        state = np.arange(14, dtype=np.float32)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        action = model.step(
            {
                "lang": "test instruction",
                "image": [image, image, image],
                "state": state,
            }
        )

        sent = model.client.last_request["examples"][0]
        np.testing.assert_array_equal(sent["state"], state)
        np.testing.assert_array_equal(
            action,
            np.arange(14, dtype=np.float32)[interface.MODEL_TO_ROBOTWIN_JOINT_ORDER],
        )

    def test_eval_uses_canonical_task_text_when_checkpoint_requests_it(self):
        task = _FakeTask()
        model = _RecordingModel(task_language_mode="dataset_name")
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        observation = {
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image},
                "right_camera": {"rgb": image},
            },
            "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
        }

        interface.eval(task, model, observation)

        self.assertEqual(model.example["lang"], "click bell")


if __name__ == "__main__":
    unittest.main()
