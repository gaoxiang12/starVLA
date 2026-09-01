import json
import tempfile
import unittest
from pathlib import Path

from examples.UnifiedPretrain.data_tools.scan_bridge_videos import (
    probe_video,
    read_lfs_pointer,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


class EpisodeBlacklistTest(unittest.TestCase):
    def test_blacklist_filters_trajectory_metadata_and_cached_steps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            metadata_dir = dataset_path / "meta"
            metadata_dir.mkdir()
            with (metadata_dir / "episodes.jsonl").open("w") as handle:
                for episode_index, length in ((0, 2), (1, 3), (2, 4)):
                    handle.write(
                        json.dumps(
                            {"episode_index": episode_index, "length": length}
                        )
                        + "\n"
                    )
            blacklist_path = metadata_dir / "bad_episodes.jsonl"
            blacklist_path.write_text(json.dumps({"episode_index": 1}) + "\n")

            dataset = object.__new__(LeRobotSingleDataset)
            dataset._dataset_path = dataset_path
            dataset._dataset_name = "bridge"
            dataset._lerobot_version = "v2.0"
            dataset.episode_blacklist_path = "meta/bad_episodes.jsonl"
            dataset._episode_blacklist = dataset._load_episode_blacklist()

            trajectory_ids, trajectory_lengths = dataset._get_trajectories()
            self.assertEqual(trajectory_ids.tolist(), [0, 2])
            self.assertEqual(trajectory_lengths.tolist(), [2, 4])
            self.assertEqual(
                dataset._filter_blacklisted_steps([(0, 0), (1, 0), (2, 0)]),
                [(0, 0), (2, 0)],
            )

    def test_probe_identifies_git_lfs_pointer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "episode.mp4"
            path.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:abc\n"
                "size 1234\n"
            )
            self.assertEqual(probe_video(path), "git_lfs_pointer")
            self.assertEqual(read_lfs_pointer(path), ("abc", 1234))


if __name__ == "__main__":
    unittest.main()
