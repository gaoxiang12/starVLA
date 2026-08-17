import unittest
from types import SimpleNamespace

import numpy as np
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from deployment.model_server.policy_norm_processor import _resolve_robot_type
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
)
from starVLA.dataloader.lerobot_datasets import EmbodimentBatchSampler
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


class _FakeMixture:
    def __init__(self):
        self.datasets = [
            SimpleNamespace(tag="franka"),
            SimpleNamespace(tag="franka"),
            SimpleNamespace(tag="aloha"),
            SimpleNamespace(tag="oxe_bridge"),
        ]
        self.dataset_sampling_weights = np.asarray([0.2, 0.1, 0.4, 0.3])
        self.epoch = 0

    def __len__(self):
        return 48

    def set_epoch(self, epoch):
        self.epoch = epoch


class _FakeDeepSpeedPlugin:
    def __init__(self):
        self.deepspeed_config = {"train_micro_batch_size_per_gpu": "auto"}
        self.is_train_batch_min = True

    def is_auto(self, key):
        return self.deepspeed_config.get(key) == "auto"


class _FakeAccelerator:
    def __init__(self):
        self.state = SimpleNamespace(deepspeed_plugin=_FakeDeepSpeedPlugin())
        self.split_batches = False
        self.num_processes = 1

    def prepare(self, *components):
        return components


class MultiEmbodimentPretrainTest(unittest.TestCase):
    def test_registry_has_aligned_native_schemas(self):
        mixture = DATASET_NAMED_MIXTURES["unified_libero_robotwin_bridge_wm"]
        self.assertEqual(len(mixture), 109)

        expected = {
            "unified_libero_wm": ("franka", 2, [0, 4, 8], 8),
            "unified_robotwin_wm": ("aloha", 3, [0, 6, 12], 16),
            "unified_bridge_wm": ("oxe_bridge", 3, [0, 1, 2], 3),
        }
        for robot_type, (tag, views, video_steps, horizon) in expected.items():
            config = ROBOT_TYPE_CONFIG_MAP[robot_type]
            modalities = config.modality_config()
            self.assertEqual(config.embodiment_tag.value, tag)
            self.assertEqual(len(config.video_keys), views)
            self.assertEqual(list(modalities["video"].delta_indices), video_steps)
            self.assertEqual(len(modalities["action"].delta_indices), horizon)
            self.assertEqual(list(config.future_time_offsets_s), [0.0, 0.2, 0.4])

    def test_batch_sampler_never_mixes_embodiments(self):
        mixture = _FakeMixture()
        sampler = EmbodimentBatchSampler(
            mixture,
            batch_size=4,
            embodiment_weights={"franka": 1, "aloha": 1, "oxe_bridge": 1},
            seed=7,
        )
        for batch in sampler:
            tags = {mixture.datasets[dataset_index].tag for dataset_index, _ in batch}
            self.assertEqual(len(tags), 1)
        sampler.set_epoch(3)
        self.assertEqual(mixture.epoch, 3)

    def test_deepspeed_infers_micro_batch_size_from_batch_sampler(self):
        batch_sampler = BatchSampler(
            SequentialSampler(range(8)), batch_size=2, drop_last=True
        )
        dataloader = DataLoader(range(8), batch_sampler=batch_sampler)
        self.assertIsNone(dataloader.batch_size)

        accelerator = _FakeAccelerator()
        TrainerUtils.setup_distributed_training(accelerator, dataloader)

        self.assertEqual(
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ],
            2,
        )

    def test_deployment_resolves_statistics_tag_to_robot_type(self):
        cfg = {
            "datasets": {
                "vla_data": {"data_mix": "unified_libero_robotwin_bridge_wm"}
            }
        }
        self.assertEqual(_resolve_robot_type(cfg, "franka"), "unified_libero_wm")
        self.assertEqual(_resolve_robot_type(cfg, "aloha"), "unified_robotwin_wm")
        self.assertEqual(
            _resolve_robot_type(cfg, "oxe_bridge"), "unified_bridge_wm"
        )


if __name__ == "__main__":
    unittest.main()
