import json
import unittest
from pathlib import Path

import numpy as np

from examples.UnifiedPretrain.data_tools.prepare_kuka_pretrain import _statistics
from examples.UnifiedPretrain.train_files.data_registry.data_config import (
    UnifiedKukaWMDataConfig,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import (
    EMBODIMENT_TAG_MAPPING,
    EmbodimentTag,
)
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata


REPO_ROOT = Path(__file__).resolve().parents[1]


class KukaPretrainTest(unittest.TestCase):
    def test_schema_and_head_contract(self):
        config = UnifiedKukaWMDataConfig()
        modalities = config.modality_config()
        self.assertEqual(config.embodiment_tag, EmbodimentTag.KUKA)
        self.assertEqual(config.control_hz, 10)
        self.assertEqual(list(modalities["video"].delta_indices), [0, 2, 4])
        self.assertEqual(len(modalities["action"].delta_indices), 8)
        self.assertEqual(sum(config.action_key_dims.values()), 7)
        self.assertEqual(sum(config.state_key_dims.values()), 8)
        self.assertEqual(EMBODIMENT_TAG_MAPPING["kuka"], 20)

    def test_modality_declares_xyzw_and_mixed_action_semantics(self):
        path = (
            REPO_ROOT
            / "examples/UnifiedPretrain/train_files/kuka_modality.json"
        )
        metadata = LeRobotModalityMetadata.model_validate(
            json.loads(path.read_text())
        )
        self.assertEqual(metadata.state["eef_quaternion_xyzw"].start, 3)
        self.assertEqual(metadata.state["eef_quaternion_xyzw"].end, 7)
        self.assertFalse(metadata.action["eef_position_delta"].absolute)
        self.assertFalse(metadata.action["eef_rotation_delta_rpy"].absolute)
        self.assertTrue(metadata.action["gripper_open"].absolute)

    def test_exact_statistics_include_quantiles(self):
        values = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        stats = _statistics(values)
        self.assertEqual(stats["mean"], [1.0, 3.0])
        self.assertEqual(stats["min"], [0.0, 2.0])
        self.assertEqual(stats["max"], [2.0, 4.0])
        self.assertIn("q01", stats)
        self.assertIn("q99", stats)


if __name__ == "__main__":
    unittest.main()
