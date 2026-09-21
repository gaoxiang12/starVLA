import fcntl
import io
import json
import tempfile
import unittest
from pathlib import Path

import av
import h5py
import numpy as np
import pandas as pd
from PIL import Image

from examples.Robotwin.data_preparation import CAMERA_MAP, convert_extracted
from examples.Robotwin.generate_local_dataset import (
    DEFAULT_TARGET,
    REPO_ROOT,
    collection_lock_available,
    default_robotwin_python,
    selected_tasks,
    validate_converted,
)
from examples.Robotwin.train_files.data_registry.data_config import DATASET_NAMED_MIXTURES


class RoboTwinDataGenerationTest(unittest.TestCase):
    @staticmethod
    def _jpeg(value: int) -> np.ndarray:
        image = Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return np.frombuffer(buffer.getvalue(), dtype=np.uint8)

    def _write_episode(self, root: Path, episode_index: int) -> None:
        length = 4
        action = np.arange(length * 14, dtype=np.float32).reshape(length, 14)
        action += episode_index * 100
        variable_bytes = h5py.vlen_dtype(np.dtype("uint8"))
        with h5py.File(root / "data" / f"episode{episode_index}.hdf5", "w") as handle:
            handle.create_dataset("joint_action/vector", data=action)
            for camera_index, camera_name in enumerate(CAMERA_MAP.values()):
                dataset = handle.create_dataset(
                    f"observation/{camera_name}/rgb",
                    shape=(length,),
                    dtype=variable_bytes,
                )
                for frame_index in range(length):
                    dataset[frame_index] = self._jpeg(
                        20 * episode_index + 5 * camera_index + frame_index
                    )
        (root / "instructions" / f"episode{episode_index}.json").write_text(
            json.dumps({"seen": [f"Surface wording {episode_index}"], "unseen": []})
        )

    def test_clean_default_is_500_successes(self):
        self.assertEqual(DEFAULT_TARGET["clean"], 500)

    def test_generated_clean500_scope_excludes_incomplete_open_laptop(self):
        tasks = selected_tasks(["all"], ["open_laptop"])
        self.assertEqual(len(tasks), 49)
        self.assertNotIn("open_laptop", tasks)

        mixture = DATASET_NAMED_MIXTURES["robotwin_generated_clean500_wm"]
        self.assertEqual(len(mixture), 49)
        self.assertNotIn("Clean/open_laptop", {dataset for dataset, _, _ in mixture})

    def test_collector_python_keeps_virtual_environment_path(self):
        expected = (REPO_ROOT.parent / ".venvs" / "RoboTwin" / "bin" / "python").absolute()
        if expected.is_file():
            self.assertEqual(default_robotwin_python(), expected)

    def test_prefinalize_waits_for_collector_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            lock_path = run_dir / "collect.lock"
            lock_path.touch()
            self.assertTrue(collection_lock_available(run_dir))
            with lock_path.open("a+") as holder:
                fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(collection_lock_available(run_dir))
                fcntl.flock(holder, fcntl.LOCK_UN)
            self.assertTrue(collection_lock_available(run_dir))

    def test_conversion_emits_canonical_lerobot_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "raw"
            destination = temp / "click_bell"
            (source / "data").mkdir(parents=True)
            (source / "instructions").mkdir()
            for episode_index in range(2):
                self._write_episode(source, episode_index)

            modality = (
                Path(__file__).parents[1]
                / "examples"
                / "Robotwin"
                / "train_files"
                / "modality.json"
            )
            convert_extracted(source, destination, modality, task_name="click_bell")
            validate_converted(destination, 2, deep=True)

            tasks = [
                json.loads(line)
                for line in (destination / "meta" / "tasks.jsonl").read_text().splitlines()
            ]
            self.assertEqual(tasks, [{"task_index": 0, "task": "click bell"}])
            self.assertTrue((destination / "meta" / "stats.json").is_file())
            self.assertTrue((destination / "meta" / "stats_gr00t.json").is_file())
            self.assertTrue((destination / "meta" / "audit" / "validation_audit.json").is_file())

            parquet = pd.read_parquet(
                destination / "data" / "chunk-000" / "episode_000000.parquet"
            )
            self.assertNotIn("observation.images.ego_view", parquet.columns)
            for video_key in CAMERA_MAP:
                video_path = (
                    destination
                    / "videos"
                    / "chunk-000"
                    / video_key
                    / "episode_000000.mp4"
                )
                with av.open(str(video_path)) as container:
                    self.assertEqual(sum(1 for _ in container.decode(video=0)), 4)

    def test_modality_packing_places_grippers_last(self):
        modality_path = (
            Path(__file__).parents[1]
            / "examples"
            / "Robotwin"
            / "train_files"
            / "modality.json"
        )
        modality = json.loads(modality_path.read_text())
        self.assertEqual(
            list(modality["action"]),
            ["left_joints", "right_joints", "left_gripper", "right_gripper"],
        )
        slices = [
            (modality["action"][key]["start"], modality["action"][key]["end"])
            for key in modality["action"]
        ]
        self.assertEqual(slices, [(0, 6), (7, 13), (6, 7), (13, 14)])


if __name__ == "__main__":
    unittest.main()
