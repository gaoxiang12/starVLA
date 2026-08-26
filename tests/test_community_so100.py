import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from examples.UnifiedPretrain.data_tools.community_so100 import (
    SO100_ACTION_NAMES,
    SO_FOLLOWER_ACTION_NAMES,
    classify_so_family_metadata,
    select_so100_video_keys,
    validate_so_family_dataset_metadata,
    validate_so100_dataset_metadata,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata
from starVLA.dataloader.lerobot_datasets import _expand_dataset_manifest_entries


def _info():
    vector = {"dtype": "float32", "shape": [6], "names": list(SO100_ACTION_NAMES)}
    return {
        "codebase_version": "v3.0",
        "robot_type": "so100",
        "fps": 30,
        "features": {
            "action": vector,
            "observation.state": vector,
            "observation.images.top": {"dtype": "video", "info": {}},
            "observation.images.wrist": {"dtype": "video", "info": {}},
        },
    }


class CommunitySo100Test(unittest.TestCase):
    def test_candidate_requires_canonical_so100_metadata(self):
        info = _info()
        self.assertIsNone(
            validate_so100_dataset_metadata(info, "owner/pick_cube", ["Pick cube"])
        )
        info["features"]["action"]["names"] = list(reversed(SO100_ACTION_NAMES))
        self.assertEqual(
            validate_so100_dataset_metadata(
                info, "owner/pick_cube", ["Pick cube"]
            ),
            "noncanonical_action_order",
        )

    def test_video_selection_ignores_explicitly_unusable_views(self):
        info = _info()
        info["features"]["observation.images.bad"] = {
            "dtype": "video",
            "info": {"curation": {"usable": False}},
        }
        self.assertEqual(
            select_so100_video_keys(info),
            ("observation.images.top", "observation.images.wrist"),
        )

    def test_single_usable_view_is_valid_for_expanded_subset(self):
        info = _info()
        del info["features"]["observation.images.wrist"]
        self.assertEqual(
            select_so100_video_keys(info), ("observation.images.top",)
        )
        self.assertIsNone(validate_so_family_dataset_metadata(info, ["Pick cube"]))
        self.assertEqual(
            validate_so100_dataset_metadata(info, "owner/pick_cube", ["Pick cube"]),
            "fewer_than_two_usable_views",
        )

    def test_so101_and_follower_route_to_separate_configs(self):
        info = _info()
        info["robot_type"] = "so101"
        self.assertEqual(classify_so_family_metadata(info), "unified_so101_wm")

        info["robot_type"] = "so101_follower"
        info["features"]["action"]["names"] = list(SO_FOLLOWER_ACTION_NAMES)
        info["features"]["observation.state"]["names"] = list(
            SO_FOLLOWER_ACTION_NAMES
        )
        self.assertEqual(
            classify_so_family_metadata(info), "unified_so_follower_wm"
        )

    def test_manifest_expansion_keeps_paths_under_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "so100.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "source_root": "community_dataset_v3",
                        "datasets": [{"path": "owner/dataset"}],
                    }
                )
            )
            expanded = _expand_dataset_manifest_entries(
                [("@manifest:community_so100", 1.0, "unified_so100_wm")],
                {"dataset_manifests": {"community_so100": str(manifest)}},
                root,
            )
        self.assertEqual(
            expanded,
            [
                (
                    "community_dataset_v3/owner/dataset",
                    1.0,
                    "unified_so100_wm",
                )
            ],
        )

    def test_manifest_entry_can_override_data_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "so_family.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "source_root": "community_dataset_v3",
                        "datasets": [
                            {
                                "path": "owner/so101_dataset",
                                "data_config": "unified_so101_wm",
                            }
                        ],
                    }
                )
            )
            expanded = _expand_dataset_manifest_entries(
                [("@manifest:community_so_family", 1.0, "unified_so100_wm")],
                {"dataset_manifests": {"community_so_family": str(manifest)}},
                root,
            )
        self.assertEqual(
            expanded,
            [
                (
                    "community_dataset_v3/owner/so101_dataset",
                    1.0,
                    "unified_so101_wm",
                )
            ],
        )

    def test_v3_video_path_uses_video_file_indices(self):
        dataset = object.__new__(LeRobotSingleDataset)
        dataset._dataset_path = Path("/dataset")
        dataset._lerobot_version = "v3.0"
        dataset._video_path_pattern = (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        dataset._chunk_size = 1000
        dataset._lerobot_modality_meta = LeRobotModalityMetadata.model_validate(
            {
                "state": {},
                "action": {},
                "video": {
                    "primary": {"original_key": "observation.images.primary"}
                },
            }
        )
        dataset.trajectory_ids_to_metadata = {
            4: {
                "data/chunk_index": 0,
                "data/file_index": 1,
                "videos/file_indices": {
                    "observation.images.primary": {
                        "chunk_index": 2,
                        "file_index": 7,
                    }
                },
            }
        }
        self.assertEqual(
            dataset.get_video_path(4, "primary"),
            Path(
                "/dataset/videos/observation.images.primary/"
                "chunk-002/file-007.mp4"
            ),
        )

    def test_v3_trajectory_listing_applies_episode_blacklist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_dir = root / "meta/episodes/chunk-000"
            episode_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "episode_index": [0, 1, 2],
                    "length": [10, 20, 30],
                    "data/chunk_index": [0, 0, 0],
                    "data/file_index": [0, 0, 0],
                }
            ).to_parquet(episode_dir / "file-000.parquet")

            dataset = object.__new__(LeRobotSingleDataset)
            dataset._dataset_path = root
            dataset._lerobot_version = "v3.0"
            dataset._episode_blacklist = {1}
            trajectory_ids, trajectory_lengths = dataset._get_trajectories()

        self.assertEqual(trajectory_ids.tolist(), [0, 2])
        self.assertEqual(trajectory_lengths.tolist(), [10, 30])


if __name__ == "__main__":
    unittest.main()
