import json
import tempfile
import unittest
from pathlib import Path

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.task_language import (
    canonical_bridge_task,
    canonical_bridge_taxonomy_task,
    canonical_metadata_text,
    canonical_task_text,
    classify_bridge_task,
    configured_task_language_mode,
    resolve_task_language,
)


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

    def test_per_embodiment_mode_overrides_global_fallback(self):
        config = {
            "task_language_mode": "metadata",
            "task_language_modes": {
                "aloha": "dataset_name",
                "oxe_bridge": "bridge_canonical",
            },
        }
        self.assertEqual(
            configured_task_language_mode(config, "aloha"), "dataset_name"
        )
        self.assertEqual(
            configured_task_language_mode(config, "oxe_bridge"), "bridge_canonical"
        )
        self.assertEqual(configured_task_language_mode(config, "franka"), "metadata")
        self.assertEqual(
            configured_task_language_mode(config, EmbodimentTag.ALOHA), "dataset_name"
        )

    def test_bridge_surface_aliases_share_one_task_label(self):
        aliases = (
            "Put the red object into the pot.",
            "place red object inside of pot",
            "Moved a red object in the pot",
        )
        self.assertEqual(
            {canonical_bridge_task(alias) for alias in aliases},
            {"move red object in pot"},
        )

    def test_bridge_semantic_differences_are_not_merged(self):
        self.assertNotEqual(
            canonical_bridge_task("put cup in left drawer"),
            canonical_bridge_task("put cup in right drawer"),
        )
        self.assertNotEqual(
            canonical_bridge_task("open the top drawer"),
            canonical_bridge_task("close the top drawer"),
        )

    def test_bridge_taxonomy_merges_role_preserving_aliases(self):
        aliases = (
            "Pick up the brush and place it on the left of the red fruit",
            "put brush to the left side of red fruit",
        )
        self.assertEqual(
            {canonical_bridge_taxonomy_task(alias) for alias in aliases},
            {"place brush left of red fruit"},
        )

    def test_bridge_taxonomy_normalizes_ordered_directions(self):
        self.assertEqual(
            canonical_bridge_taxonomy_task("move pot to the right top burner"),
            "place pot to top right burner",
        )
        self.assertNotEqual(
            canonical_bridge_taxonomy_task("fold cloth from top left to bottom right"),
            canonical_bridge_taxonomy_task("fold cloth from bottom right to top left"),
        )

    def test_bridge_taxonomy_exposes_unlabeled_tasks(self):
        label = classify_bridge_task("")
        self.assertEqual(label.status, "unlabeled")
        self.assertEqual(label.canonical_text, "")
        self.assertEqual(
            resolve_task_language("", "bridge", "bridge_taxonomy"), ""
        )

    def test_bridge_taxonomy_groups_no_op_aliases(self):
        self.assertEqual(
            {
                canonical_bridge_taxonomy_task("Nothing"),
                canonical_bridge_taxonomy_task("the robot arm did nothing"),
                canonical_bridge_taxonomy_task("no change in the image"),
            },
            {"no_op"},
        )

    def test_canonical_metadata_matches_text_encoder_normalization(self):
        self.assertEqual(canonical_metadata_text("  PICK\u3000UP  Cup  "), "pick up cup")


if __name__ == "__main__":
    unittest.main()
