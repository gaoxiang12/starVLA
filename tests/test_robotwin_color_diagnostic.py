import numpy as np
import pytest

from examples.Robotwin.eval_files.robotwin_color_eval_runner import policy_observation


def test_color_counterfactual_changes_only_policy_camera_arrays():
    frames = {camera: {"rgb": np.array([[[230, 50, 10]]], dtype=np.uint8), "depth": np.ones((1, 1))}
              for camera in ("head_camera", "left_camera", "right_camera")}
    observation = dict(observation=frames, joint_action=np.arange(14), instruction="rank blocks")
    changed = policy_observation(observation, "bgr")
    for camera in frames:
        np.testing.assert_array_equal(changed["observation"][camera]["rgb"], [[[10, 50, 230]]])
        np.testing.assert_array_equal(observation["observation"][camera]["rgb"], [[[230, 50, 10]]])
        assert not np.shares_memory(changed["observation"][camera]["rgb"], frames[camera]["rgb"])
        assert changed["observation"][camera]["depth"] is frames[camera]["depth"]
    assert changed["joint_action"] is observation["joint_action"]
    assert changed["instruction"] == observation["instruction"]
    assert policy_observation(observation, "rgb") is observation
    with pytest.raises(ValueError):
        policy_observation(observation, "rbg")
