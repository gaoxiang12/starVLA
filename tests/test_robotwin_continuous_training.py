import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn

from deployment.model_server.policy_norm_processor import _build_dataset_metadata, _resolve_robot_type
from examples.UnifiedPretrain.train_files.data_registry.data_config import (
    RoboTwinContinuousNextWMDataConfig,
    UnifiedRoboTwinWMDataConfig,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer
from starVLA.dataloader.lerobot_datasets import episode_split_blacklist
from starVLA.model.modules.action_model.ACT_ActionHeader import TurboStyleACTActionHead
from starVLA.training.trainer_utils.action_validation import evaluate_action_batch
from starVLA.training.train_starvla import VLATrainer


class RobotwinContinuousTrainingTest(unittest.TestCase):
    def test_continuous_gripper_and_joint_tail_round_trip(self):
        config = RoboTwinContinuousNextWMDataConfig()
        stats = {
            modality: {key: [-1.] * 12 + [0., 0.] if key in {'min', 'q01'} else [1.] * 14
                       for key in ('mean', 'std', 'min', 'max', 'q01', 'q99')}
            for modality in ('action', 'state')
        }
        meta = _build_dataset_metadata(stats, config.embodiment_tag, config.action_keys,
                                       config.state_keys, config.action_key_dims, config.state_key_dims)
        transform = config.transform()
        transform.set_metadata(meta)
        inputs = {}
        for keys in (config.action_keys, config.state_keys):
            for key in keys:
                inputs[key] = np.full((3, 1 if 'gripper' in key else 6), .5 if 'gripper' in key else 1.3, np.float32)
        normalized = transform({key: value.copy() for key, value in inputs.items()})
        self.assertAlmostEqual(float(normalized['action.left_gripper'][0, 0]), .5)
        self.assertGreater(float(normalized['action.right_joints'][0, 0]), 1.)
        restored = transform.unapply(normalized)
        for key, expected in inputs.items():
            np.testing.assert_allclose(restored[key], expected, atol=1e-6)

    def test_legacy_binary_and_clipping_are_unchanged(self):
        binary = Normalizer('binary', {})
        self.assertEqual(binary.forward(torch.tensor([.5])).item(), 0.)
        bounded = Normalizer('q99', {'q01': [-1.], 'q99': [1.]}, q99_clip=1.)
        self.assertEqual(bounded.forward(torch.tensor([1.3])).item(), 1.)
        self.assertEqual(UnifiedRoboTwinWMDataConfig.action_indices, list(range(16)))
        self.assertEqual(RoboTwinContinuousNextWMDataConfig.action_indices, list(range(1, 17)))
        self.assertNotEqual(UnifiedRoboTwinWMDataConfig.action_spec_id, RoboTwinContinuousNextWMDataConfig.action_spec_id)

    def test_deployment_selects_new_transform_only_for_new_recipe(self):
        for mixture, expected in [
            ('unified_robotwin_generated_clean500_wm', 'unified_robotwin_wm'),
            ('robotwin_generated_clean1000_continuous_next_wm', 'robotwin_continuous_next_wm'),
        ]:
            cfg = {'datasets': {'vla_data': {'data_mix': mixture}}}
            self.assertEqual(_resolve_robot_type(cfg, 'aloha'), expected)

    def test_positive_action_offset_never_samples_terminal_anchor(self):
        dataset = SimpleNamespace(trajectory_ids=np.array([0]), trajectory_lengths=np.array([2]), minimum_action_offset=1)
        mixture = LeRobotMixtureDataset.__new__(LeRobotMixtureDataset)
        mixture.datasets = [dataset]
        mixture._dataset_sampling_weights = np.array([1.])
        mixture._trajectory_sampling_weights = [np.array([1.])]
        mixture.epoch, mixture.seed, mixture.mode = 0, 42, 'train'
        for index in range(100):
            self.assertEqual(mixture.sample_step(index)[2], 0)

    def test_future_action_tail_mask_is_shifted(self):
        dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
        dataset.data_cfg = {'action_valid_mask': True}
        dataset._modality_keys = {'action': ['action.joints']}
        dataset._delta_indices = {'action.joints': np.arange(1, 17)}
        dataset._trajectory_ids = np.array([0])
        dataset._trajectory_lengths = np.array([20])
        result = dataset._attach_action_validity({'action': np.zeros((16, 14))}, 0, 18)
        np.testing.assert_array_equal(result['action_valid_mask'], [True] + [False] * 15)

    def test_episode_split_is_disjoint_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'meta').mkdir()
            (root / 'meta/episodes.jsonl').write_text(''.join(json.dumps({'episode_index': i}) + '\n' for i in range(100)))
            train_excluded = set(episode_split_blacklist(root, {'validation_episode_stride': 20}))
            val_excluded = set(episode_split_blacklist(root, {'validation_episode_stride': 20, 'episode_split': 'validation'}))
            self.assertEqual(train_excluded, {0, 20, 40, 60, 80})
            self.assertFalse(train_excluded & val_excluded)
            self.assertEqual(train_excluded | val_excluded, set(range(100)))

    def test_identity_action_output_can_represent_normalized_joint_tails(self):
        head = TurboStyleACTActionHead(token_dim=8, hidden_dim=8, action_dim=14, horizon=16,
                                      num_frames=3, num_visual_tokens=4, num_heads=2, num_layers=1,
                                      output_activation='identity')
        head.action_projection = nn.Identity()
        value = torch.full((1, 16, 14), 1.3)
        torch.testing.assert_close(head.predict_action(value), value)

    def test_validation_disables_dropout_masks_padding_and_restores_train_mode(self):
        class FakePolicy(nn.Module):
            def predict_action(self, examples):
                assert not self.training
                return {'normalized_actions': np.array([[[.5, .5], [999., 999.]]])}
        model = FakePolicy().train()
        examples = [{'action': np.array([[.5, .5], [0., 0.]]), 'action_valid_mask': [True, False]}]
        scores = evaluate_action_batch(model, examples, [0, 1])
        self.assertEqual(scores['mse_score'], 0.)
        self.assertEqual(scores['l1_action_loss'], 0.)
        self.assertEqual(scores['gripper_position_accuracy'], 1.)
        self.assertTrue(model.training)

    def test_linear_tails_preserve_central_pretrained_mapping_and_allow_gradients(self):
        head = TurboStyleACTActionHead(token_dim=8, hidden_dim=8, action_dim=14, horizon=16,
                                      num_frames=3, num_visual_tokens=4, num_heads=2, num_layers=1,
                                      output_activation='tanh_linear_tail', gripper_indices=(12, 13))
        head.action_projection = nn.Identity()
        central = torch.linspace(-3, 3, 14).reshape(1, 1, 14)
        torch.testing.assert_close(head.predict_action(central), central.tanh())
        tails = torch.full((1, 1, 14), 8., requires_grad=True)
        predicted = head.predict_action(tails)
        self.assertTrue(bool((predicted[..., :12] > 1).all()))
        self.assertTrue(bool((predicted[..., :12] < 1.05).all()))
        self.assertTrue(bool((predicted[..., 12:] <= 1).all()))
        predicted.sum().backward()
        self.assertTrue(bool((tails.grad[..., :12] > 0.009).all()))

    @patch('starVLA.training.train_starvla.dist.barrier')
    @patch('starVLA.training.trainer_utils.action_validation.HeldOutActionEvaluator')
    def test_nonmain_rank_participates_in_validation_dataset_construction(self, evaluator, barrier):
        trainer = VLATrainer.__new__(VLATrainer)
        trainer.config = SimpleNamespace(datasets=SimpleNamespace(vla_data={'validation_episode_stride': 20}))
        trainer.accelerator = SimpleNamespace(is_main_process=False, unwrap_model=MagicMock(),
                                              wait_for_everyone=barrier)
        trainer.model = MagicMock()
        trainer.completed_steps = 10
        self.assertEqual(trainer.eval_action_model({}), {})
        evaluator.assert_called_once_with(trainer.config)
        evaluator.return_value.evaluate.assert_not_called()
        barrier.assert_called_once()


if __name__ == '__main__':
    unittest.main()
