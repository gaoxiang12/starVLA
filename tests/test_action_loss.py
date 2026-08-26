import unittest

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer
from starVLA.model.modules.action_model.action_loss import (
    action_l1_diagnostics,
    masked_action_l1_loss,
)


class ActionLossTest(unittest.TestCase):
    def test_masked_l1_ignores_padded_timesteps(self):
        target = torch.zeros(1, 3, 2)
        pred = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]]])
        valid = torch.tensor([[True, True, False]])

        loss = masked_action_l1_loss(pred, target, valid)

        self.assertAlmostEqual(loss.item(), 1.5)

    def test_action_diagnostics_split_continuous_and_gripper(self):
        target = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
        pred = torch.tensor([[[1.0, 0.0], [3.0, 0.75], [99.0, 1.0]]])
        valid = torch.tensor([[True, True, False]])

        metrics = action_l1_diagnostics(
            pred, target, valid, gripper_indices=(1,)
        )

        self.assertAlmostEqual(metrics["continuous_action_l1"].item(), 2.0)
        self.assertAlmostEqual(metrics["gripper_action_l1"].item(), 0.125)
        self.assertAlmostEqual(metrics["gripper_action_accuracy"].item(), 1.0)
        self.assertAlmostEqual(metrics["first_action_l1"].item(), 0.5)
        self.assertAlmostEqual(metrics["valid_action_fraction"].item(), 2 / 3)

    def test_q99_clip_can_match_tanh_output_range(self):
        normalizer = Normalizer(
            "q99",
            {"q01": [0.0], "q99": [1.0]},
            q99_clip=1.0,
        )

        normalized = normalizer.forward(torch.tensor([[-1.0], [2.0]]))

        torch.testing.assert_close(normalized, torch.tensor([[-1.0], [1.0]]))


class ActionValidityTest(unittest.TestCase):
    def test_action_validity_marks_episode_tail_padding(self):
        dataset = object.__new__(LeRobotSingleDataset)
        dataset.data_cfg = {"action_valid_mask": True}
        dataset._modality_keys = {"action": ["action.x", "action.gripper"]}
        dataset._delta_indices = {
            "action.x": np.array([0, 1, 2]),
            "action.gripper": np.array([0, 1, 2]),
        }
        dataset._trajectory_ids = np.array([7])
        dataset._trajectory_lengths = np.array([5])
        sample = {"action": np.zeros((3, 2), dtype=np.float32)}

        result = dataset._attach_action_validity(sample, 7, 4)

        np.testing.assert_array_equal(
            result["action_valid_mask"], np.array([True, False, False])
        )


if __name__ == "__main__":
    unittest.main()
