import json
import tempfile
import unittest
from pathlib import Path

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.task_language import canonical_task_text, resolve_task_language


class TaskLanguageTest(unittest.TestCase):
    def test_canonical_task_text_is_stable_and_readable(self):
        self.assertEqual(canonical_task_text("Clean/click_bell"), "click bell")
        self.assertEqual(
            resolve_task_language("episode-specific wording", "click_bell", "dataset_name"),
            "click bell",
        )

    def test_dataset_name_mode_replaces_all_episode_descriptions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "click_bell"
            metadata_dir = dataset_path / "meta"
            metadata_dir.mkdir(parents=True)
            with (metadata_dir / "tasks.jsonl").open("w") as handle:
                for task_index, task in enumerate(("first wording", "second wording")):
                    handle.write(json.dumps({"task_index": task_index, "task": task}) + "\n")

            dataset = object.__new__(LeRobotSingleDataset)
            dataset._dataset_path = dataset_path
            dataset._dataset_name = dataset_path.name
            dataset._lerobot_version = "v2.0"
            dataset.data_cfg = {"task_language_mode": "dataset_name"}

            tasks = dataset._get_tasks()

            self.assertEqual(tasks["task"].tolist(), ["click bell", "click bell"])

    def test_metadata_mode_preserves_episode_descriptions(self):
        self.assertEqual(
            resolve_task_language("Keep This Wording", "click_bell", "metadata"),
            "Keep This Wording",
        )


if __name__ == "__main__":
    unittest.main()
