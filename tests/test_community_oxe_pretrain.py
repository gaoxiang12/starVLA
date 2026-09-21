import json
import unittest
from pathlib import Path

import numpy as np

from examples.UnifiedPretrain.train_files.data_registry.data_config import (
    DATASET_NAMED_MIXTURES,
    UnifiedBcZWMDataConfig,
    UnifiedFmbWMDataConfig,
    UnifiedFractalWMDataConfig,
    UnifiedTacoPlayWMDataConfig,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata


REPO_ROOT = Path(__file__).resolve().parents[1]


class CommunityOxePretrainTest(unittest.TestCase):
    def test_independent_configs_and_mixtures(self):
        expected = {
            "unified_taco_play_wm": (
                UnifiedTacoPlayWMDataConfig,
                EmbodimentTag.TACO_FRANKA,
                15,
                [0, 3, 6],
                2,
            ),
            "unified_bc_z_wm": (
                UnifiedBcZWMDataConfig,
                EmbodimentTag.GOOGLE_BCZ,
                10,
                [0, 2, 4],
                1,
            ),
            "unified_fractal_wm": (
                UnifiedFractalWMDataConfig,
                EmbodimentTag.GOOGLE_RT1,
                3,
                [0, 1, 1],
                1,
            ),
            "unified_fmb_wm": (
                UnifiedFmbWMDataConfig,
                EmbodimentTag.FMB_FRANKA,
                10,
                [0, 2, 4],
                3,
            ),
        }
        for mixture_name, (klass, tag, hz, video_steps, views) in expected.items():
            config = klass()
            modalities = config.modality_config()
            self.assertEqual(config.embodiment_tag, tag)
            self.assertEqual(config.control_hz, hz)
            self.assertEqual(list(modalities["video"].delta_indices), video_steps)
            self.assertEqual(len(config.video_keys), views)
            self.assertEqual(sum(config.action_key_dims.values()), 7)
            self.assertEqual(sum(config.state_key_dims.values()), 8)
            self.assertEqual(len(modalities["action"].delta_indices), 8)
            self.assertEqual(len(DATASET_NAMED_MIXTURES[mixture_name]), 1)
        combined = DATASET_NAMED_MIXTURES["unified_community_oxe_candidate_wm"]
        self.assertEqual(len(combined), 4)
        self.assertEqual(len({entry[2] for entry in combined}), 4)

    def test_modality_templates_match_config_dimensions(self):
        templates = (
            "taco_play_modality.json",
            "bc_z_modality.json",
            "fractal_modality.json",
            "fmb_modality.json",
        )
        for filename in templates:
            path = REPO_ROOT / "examples/UnifiedPretrain/train_files" / filename
            metadata = LeRobotModalityMetadata.model_validate(json.loads(path.read_text()))
            self.assertEqual(sum(row.end - row.start for row in metadata.state.values()), 8)
            self.assertEqual(sum(row.end - row.start for row in metadata.action.values()), 7)
            self.assertTrue(metadata.action["gripper_open"].absolute)

    def test_repeated_low_rate_future_frame_is_masked(self):
        dataset = object.__new__(LeRobotSingleDataset)
        dataset.data_cfg = {"future_obs_valid_mask": True}
        dataset._modality_keys = {"video": ["video.image"]}
        dataset._delta_indices = {"video.image": np.asarray([0, 1, 1])}
        dataset._trajectory_lengths = np.asarray([10])
        dataset.get_trajectory_index = lambda trajectory_id: 0
        sample = dataset._attach_future_frame_validity({}, 0, 0)
        np.testing.assert_array_equal(
            sample["future_frame_valid_mask"], np.asarray([True, True, False])
        )

if __name__ == "__main__":
    unittest.main()
