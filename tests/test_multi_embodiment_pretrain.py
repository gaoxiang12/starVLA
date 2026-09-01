import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from deployment.model_server.policy_norm_processor import _resolve_robot_type
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from starVLA.dataloader.lerobot_datasets import EmbodimentBatchSampler
from starVLA.model.framework.WM4A.GAWM import GAWM
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
    def test_mixture_clears_inactive_trajectory_cache(self):
        mixture = LeRobotMixtureDataset.__new__(LeRobotMixtureDataset)
        mixture._clear_inactive_trajectory_cache = True
        mixture._active_cached_dataset = None
        first = SimpleNamespace(curr_traj_data=object(), curr_traj_id=3)
        second = SimpleNamespace(curr_traj_data=object(), curr_traj_id=5)

        mixture._activate_dataset_cache(first)
        mixture._activate_dataset_cache(second)

        self.assertIsNone(first.curr_traj_data)
        self.assertIsNone(first.curr_traj_id)
        self.assertIs(mixture._active_cached_dataset, second)
        self.assertIsNotNone(second.curr_traj_data)

    def test_checkpoint_expansion_preserves_existing_embodiment_rows(self):
        model = object.__new__(GAWM)
        nn.Module.__init__(model)
        model.embodiment_tags = (
            "aloha",
            "franka",
            "kuka",
            "oxe_bridge",
            "oxe_droid",
            "so100",
            "so101",
            "so_follower",
        )
        model.embodiment_embedding = nn.Embedding(len(model.embodiment_tags), 2)
        with torch.no_grad():
            model.embodiment_embedding.weight.fill_(-1.0)

        source_rows = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        checkpoint = {
            "embodiment_embedding.weight": source_rows,
            "action_models.aloha.action_queries.weight": torch.zeros(1),
            "action_models.franka.action_queries.weight": torch.zeros(1),
            "action_models.oxe_bridge.action_queries.weight": torch.zeros(1),
        }

        remapped = model.remap_checkpoint_state_dict(checkpoint)
        expanded = remapped["embodiment_embedding.weight"]
        target_index = {
            tag: index for index, tag in enumerate(model.embodiment_tags)
        }
        for source_index, tag in enumerate(("aloha", "franka", "oxe_bridge")):
            torch.testing.assert_close(
                expanded[target_index[tag]], source_rows[source_index]
            )
        for tag in ("kuka", "oxe_droid", "so100", "so101", "so_follower"):
            torch.testing.assert_close(
                expanded[target_index[tag]], torch.full((2,), -1.0)
            )

    def test_registry_has_aligned_native_schemas(self):
        mixture = DATASET_NAMED_MIXTURES["unified_libero_robotwin_bridge_wm"]
        self.assertEqual(len(mixture), 109)

        expected = {
            "unified_libero_wm": ("franka", 2, [0, 4, 8], 8),
            "unified_robotwin_wm": ("aloha", 3, [0, 6, 12], 16),
            "unified_bridge_wm": ("oxe_bridge", 3, [0, 1, 2], 3),
            "unified_kuka_wm": ("kuka", 1, [0, 2, 4], 8),
            "unified_so100_wm": ("so100", 2, [0, 6, 12], 16),
            "unified_so101_wm": ("so101", 2, [0, 6, 12], 16),
            "unified_so_follower_wm": ("so_follower", 2, [0, 6, 12], 16),
        }
        for robot_type, (tag, views, video_steps, horizon) in expected.items():
            config = ROBOT_TYPE_CONFIG_MAP[robot_type]
            modalities = config.modality_config()
            self.assertEqual(config.embodiment_tag.value, tag)
            self.assertEqual(len(config.video_keys), views)
            self.assertEqual(list(modalities["video"].delta_indices), video_steps)
            self.assertEqual(len(modalities["action"].delta_indices), horizon)
            self.assertEqual(list(config.future_time_offsets_s), [0.0, 0.2, 0.4])

        so100_mixture = DATASET_NAMED_MIXTURES["unified_so100_wm"]
        self.assertEqual(
            so100_mixture,
            [("@manifest:community_so100", 1.0, "unified_so100_wm")],
        )
        self.assertEqual(
            DATASET_NAMED_MIXTURES["unified_so_family_wm"],
            [("@manifest:community_so_family", 1.0, "unified_so100_wm")],
        )
        self.assertEqual(
            DATASET_NAMED_MIXTURES["unified_kuka_wm"],
            [("kuka_lerobot", 1.0, "unified_kuka_wm")],
        )

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

        cfg["datasets"]["vla_data"]["data_mix"] = (
            "unified_libero_robotwin_bridge_droid_so100_wm"
        )
        self.assertEqual(
            _resolve_robot_type(cfg, "so100"), "unified_so100_wm"
        )


if __name__ == "__main__":
    unittest.main()
