import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata


def _statistics(values):
    return {
        name: values
        for name in ("mean", "std", "max", "min", "q01", "q99")
    }


class _FakeDataset:
    tag = "new_embodiment"
    modality_keys = {
        "action": ["action.arm", "action.gripper"],
        "state": ["state.arm", "state.gripper"],
    }

    def __init__(self):
        self.applied_metadata = None

    def set_transforms_metadata(self, metadata):
        self.applied_metadata = metadata


class NormalizationStatisticsOverrideTest(unittest.TestCase):
    def setUp(self):
        statistical_values = {
            "max": [1.0], "min": [-1.0], "mean": [0.0],
            "std": [1.0], "q01": [-0.9], "q99": [0.9],
        }
        self.metadata = DatasetMetadata.model_validate({
            "embodiment_tag": "new_embodiment",
            "statistics": {
                "action": {"arm": statistical_values, "gripper": statistical_values},
                "state": {"arm": statistical_values, "gripper": statistical_values},
            },
            "modalities": {
                "video": {},
                "action": {
                    "arm": {"absolute": True, "shape": [2], "continuous": True},
                    "gripper": {"absolute": True, "shape": [1], "continuous": False},
                },
                "state": {
                    "arm": {"absolute": True, "shape": [2], "continuous": True},
                    "gripper": {"absolute": True, "shape": [1], "continuous": False},
                },
            },
        })

    def _mixture(self):
        mixture = LeRobotMixtureDataset.__new__(LeRobotMixtureDataset)
        mixture.datasets = [_FakeDataset()]
        mixture.merged_metadata = {"new_embodiment": self.metadata}
        return mixture

    def test_flat_statistics_are_split_and_applied_in_modality_order(self):
        payload = {
            "new_embodiment": {
                "action": _statistics([10.0, 20.0, 30.0]),
                "state": _statistics([40.0, 50.0, 60.0]),
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset_statistics.json"
            path.write_text(json.dumps(payload))
            mixture = self._mixture()
            mixture.apply_normalization_statistics(path)

        action = mixture.merged_metadata["new_embodiment"].statistics.action
        state = mixture.merged_metadata["new_embodiment"].statistics.state
        self.assertEqual(action["arm"].mean.tolist(), [10.0, 20.0])
        self.assertEqual(action["gripper"].mean.tolist(), [30.0])
        self.assertEqual(state["arm"].mean.tolist(), [40.0, 50.0])
        self.assertEqual(state["gripper"].mean.tolist(), [60.0])
        self.assertIs(mixture.datasets[0].applied_metadata, mixture.merged_metadata["new_embodiment"])

    def test_dimension_mismatch_fails_instead_of_falling_back(self):
        payload = {
            "new_embodiment": {
                "action": _statistics([1.0, 2.0]),
                "state": _statistics([1.0, 2.0, 3.0]),
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset_statistics.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "expected \\(3,\\)"):
                self._mixture().apply_normalization_statistics(path)

    def test_metadata_weights_are_renormalized_within_each_embodiment(self):
        class TaggedDataset:
            def __init__(self, tag, metadata):
                self.tag = tag
                self.metadata = metadata
                self.applied_metadata = None

            def set_transforms_metadata(self, metadata):
                self.applied_metadata = metadata

        mixture = LeRobotMixtureDataset.__new__(LeRobotMixtureDataset)
        mixture.datasets = [
            TaggedDataset("franka", "f1"),
            TaggedDataset("franka", "f2"),
            TaggedDataset("aloha", "a1"),
        ]
        mixture._dataset_sampling_weights = __import__("numpy").asarray(
            [0.2, 0.3, 0.5]
        )
        calls = []

        def merge_metadata(*, metadatas, dataset_sampling_weights, **_):
            calls.append((metadatas, dataset_sampling_weights))
            return tuple(metadatas)

        mixture.merge_metadata = merge_metadata
        mixture.update_metadata({"percentile_mixing_method": "min_max"})

        self.assertEqual(calls[0][0], ["f1", "f2"])
        self.assertEqual(calls[0][1], [0.4, 0.6])
        self.assertEqual(calls[1], (["a1"], [1.0]))


if __name__ == "__main__":
    unittest.main()
