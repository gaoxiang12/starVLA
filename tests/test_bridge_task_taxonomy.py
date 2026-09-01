import json
from pathlib import Path
import tempfile
import unittest

from examples.UnifiedPretrain.data_tools.build_bridge_task_taxonomy import (
    build_catalog,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class BridgeTaskTaxonomyTest(unittest.TestCase):
    def test_training_filter_uses_canonical_count_and_keeps_exact_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            meta_dir = dataset_dir / "meta"
            meta_dir.mkdir()
            _write_jsonl(
                meta_dir / "tasks.jsonl",
                [
                    {
                        "task_index": 0,
                        "task": (
                            "Pick up the brush and place it on the left "
                            "of the red fruit"
                        ),
                    },
                    {
                        "task_index": 1,
                        "task": "put brush to the left side of red fruit",
                    },
                    {"task_index": 2, "task": "open drawer"},
                    {"task_index": 3, "task": ""},
                ],
            )

            episodes = []
            episode_index = 0
            for task, count in (
                ("Pick up the brush and place it on the left of the red fruit", 10),
                ("put brush to the left side of red fruit", 10),
                ("open drawer", 19),
                ("", 21),
            ):
                for _ in range(count):
                    episodes.append(
                        {
                            "episode_index": episode_index,
                            "tasks": [task],
                            "length": 5,
                        }
                    )
                    episode_index += 1
            _write_jsonl(meta_dir / "episodes.jsonl", episodes)
            (meta_dir / "info.json").write_text(
                json.dumps(
                    {
                        "features": {
                            f"observation.images.image_{index}": {"dtype": "video"}
                            for index in range(4)
                        }
                    }
                ),
                encoding="utf-8",
            )
            _write_jsonl(
                meta_dir / "video_health" / "bad_episodes.jsonl",
                [{"episode_index": 0, "reason": "test"}],
            )

            mappings, catalog, blacklist, summary = build_catalog(
                dataset_dir,
                ["image_0", "image_1", "image_2"],
                min_episodes=20,
            )

            included = [row for row in catalog if row["include_for_training"]]
            self.assertEqual(len(included), 1)
            self.assertEqual(included[0]["episode_count"], 20)
            self.assertEqual(summary["training_task_count"], 1)
            self.assertEqual(summary["training_episode_count"], 19)
            self.assertEqual(summary["excluded_episode_count"], 41)
            self.assertEqual(summary["training_selected_camera_video_count"], 57)
            self.assertEqual(len(blacklist), 41)

            mapping_by_raw = {row["raw_description"]: row for row in mappings}
            self.assertTrue(
                mapping_by_raw[
                    "Pick up the brush and place it on the left of the red fruit"
                ]["include_for_training"]
            )
            self.assertTrue(
                mapping_by_raw["put brush to the left side of red fruit"][
                    "include_for_training"
                ]
            )
            self.assertEqual(
                mapping_by_raw["open drawer"]["exclusion_reason"],
                "below_min_episodes",
            )
            self.assertEqual(
                mapping_by_raw[""]["exclusion_reason"],
                "empty_task_description",
            )


if __name__ == "__main__":
    unittest.main()
