import importlib.util
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.modules.world_model.smooth_spatial_latent_world_model import (
    GridSpatialLatentProjector,
    SmoothSpatialLatentWorldModel,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class GridSpatialLatentProjectorTest(unittest.TestCase):
    def test_fixed_grid_projection_is_detached_from_dino(self):
        projector = GridSpatialLatentProjector(
            patch_dim=8,
            spatial_token_dim=4,
            latent_dim=4,
            num_views=2,
            grid_size=2,
        )
        patches = torch.randn(3, 4, 2, 16, 8, requires_grad=True)

        latent = projector(patches)
        latent.square().mean().backward()

        self.assertEqual(latent.shape, (3, 4, 4))
        self.assertEqual(projector.spatial_token_count, 8)
        self.assertEqual(projector.num_tokens, 1)
        self.assertIsNone(patches.grad)
        self.assertIsNotNone(projector.token_projection.weight.grad)
        self.assertIsNotNone(projector.global_projection.weight.grad)
        token_row_gram = (
            projector.token_projection.weight.detach()
            @ projector.token_projection.weight.detach().T
        )
        global_row_gram = (
            projector.global_projection.weight.detach()
            @ projector.global_projection.weight.detach().T
        )
        torch.testing.assert_close(
            token_row_gram, torch.eye(4), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            global_row_gram, torch.eye(4), atol=1e-5, rtol=1e-5
        )


class SmoothSpatialLatentWorldModelTest(unittest.TestCase):
    def _make_model(self, **weight_overrides):
        weights = {
            "prediction_weight": 1.0,
            "sigreg_weight": 0.02,
            "slow_weight": 0.05,
            "acceleration_weight": 0.10,
            "temporal_order_weight": 0.05,
        }
        rollout_steps = int(weight_overrides.pop("rollout_steps", 1))
        rollout_weight = float(weight_overrides.pop("rollout_weight", 0.0))
        n_future = int(weight_overrides.pop("n_future", 2))
        weights.update(weight_overrides)
        return SmoothSpatialLatentWorldModel(
            patch_dim=8,
            num_views=2,
            goal_dim=5,
            latent_dim=4,
            spatial_token_dim=4,
            grid_size=2,
            n_future=n_future,
            rollout_steps=rollout_steps,
            rollout_weight=rollout_weight,
            predictor_dim=8,
            predictor_depth=1,
            predictor_heads=2,
            predictor_ffn_dim=16,
            temporal_order_margin=0.10,
            sigreg_knots=5,
            sigreg_num_proj=16,
            **weights,
        )

    def test_direct_future_l2_objective_masking_and_gradients(self):
        torch.manual_seed(11)
        model = self._make_model()
        patches = torch.randn(4, 4, 2, 16, 8, requires_grad=True)
        goal = torch.randn(4, 5)
        valid_mask = torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=torch.bool,
        )

        output = model(patches, goal=goal, valid_mask=valid_mask)

        # Zero-initialized predictor output means the direct future prediction
        # starts exactly at the copy-current baseline.
        torch.testing.assert_close(output["prediction_mse"], output["copy_mse"])
        torch.testing.assert_close(
            output["prediction_to_copy_ratio"], torch.ones(())
        )
        torch.testing.assert_close(output["valid_fraction_horizon_1"], torch.tensor(0.75))
        torch.testing.assert_close(output["valid_fraction_horizon_2"], torch.tensor(0.50))
        torch.testing.assert_close(output["valid_far_fraction"], torch.tensor(0.25))
        torch.testing.assert_close(output["sigreg_sample_count"], torch.tensor(10.0))
        self.assertLessEqual(
            float(output["latent_effective_rank_fraction"]), 1.0 + 1e-5
        )

        expected_total = (
            output["latent_prediction_loss"]
            + 0.02 * output["sigreg_loss"]
            + 0.05 * output["slow_loss"]
            + 0.10 * output["acceleration_loss"]
            + 0.05 * output["temporal_order_loss"]
        )
        torch.testing.assert_close(output["loss"], expected_total)
        self.assertFalse(
            any("reconstruction" in key or "state_loss" in key for key in output)
        )

        output["loss"].backward()
        self.assertIsNone(patches.grad)
        self.assertGreater(
            model.projector.token_projection.weight.grad.abs().sum(), 0
        )
        self.assertGreater(
            model.projector.global_projection.weight.grad.abs().sum(), 0
        )
        self.assertGreater(model.predictor.out.weight.grad.abs().sum(), 0)

    def test_second_difference_and_far_order_constraints(self):
        class FixedProjector(nn.Module):
            latent_dim = 4
            num_tokens = 1

            def forward(self, latent):
                return latent

        acceleration_model = self._make_model(
            prediction_weight=0.0,
            sigreg_weight=0.0,
            slow_weight=0.0,
            acceleration_weight=1.0,
            temporal_order_weight=0.0,
        )
        acceleration_model.projector = FixedProjector()
        linear = torch.tensor([0.0, 1.0, 2.0, 4.0]).view(1, 4, 1).expand(-1, -1, 4)
        jerk = torch.tensor([0.0, 1.0, 3.0, 4.0]).view(1, 4, 1).expand(-1, -1, 4)

        linear_output = acceleration_model(linear, goal=torch.zeros(1, 5))
        jerk_output = acceleration_model(jerk, goal=torch.zeros(1, 5))

        torch.testing.assert_close(linear_output["acceleration_loss"], torch.zeros(()))
        torch.testing.assert_close(jerk_output["acceleration_loss"], torch.ones(()))

        order_model = self._make_model(
            prediction_weight=0.0,
            sigreg_weight=0.0,
            slow_weight=0.0,
            acceleration_weight=0.0,
            temporal_order_weight=1.0,
        )
        order_model.projector = FixedProjector()
        constant = torch.zeros(1, 4, 4)

        constant_output = order_model(constant, goal=torch.zeros(1, 5))
        ordered_output = order_model(linear, goal=torch.zeros(1, 5))

        torch.testing.assert_close(
            constant_output["temporal_order_loss"], torch.tensor(0.10)
        )
        torch.testing.assert_close(
            ordered_output["temporal_order_loss"], torch.zeros(())
        )

    def test_two_step_rollout_supervision_and_diagnostics(self):
        torch.manual_seed(7)
        model = self._make_model(rollout_steps=2, rollout_weight=1.0)
        self.assertEqual(model.required_frames, 6)
        patches = torch.randn(4, 6, 2, 16, 8, requires_grad=True)
        goal = torch.randn(4, 5)
        valid_mask = torch.ones(4, 6, dtype=torch.bool)

        output = model(patches, goal=goal, valid_mask=valid_mask)

        # Step 1 stays the zero-initialized copy baseline.
        torch.testing.assert_close(
            output["prediction_to_copy_ratio"], torch.ones(())
        )
        # Step 2 rollout diagnostics exist and are finite.
        self.assertTrue(torch.isfinite(output["rollout_latent_loss"]))
        self.assertTrue(torch.isfinite(output["rollout_to_copy_ratio_step_2"]))
        self.assertTrue(
            torch.isfinite(output["rollout_direction_cosine_step_2"])
        )
        self.assertLessEqual(
            float(output["rollout_to_copy_ratio_step_2"].abs()), 10.0
        )
        # The rollout step is part of the total loss.
        expected_total = (
            output["latent_prediction_loss"]
            + 1.0 * output["rollout_latent_loss"]
            + 0.02 * output["sigreg_loss"]
            + 0.05 * output["slow_loss"]
            + 0.10 * output["acceleration_loss"]
            + 0.05 * output["temporal_order_loss"]
        )
        torch.testing.assert_close(output["loss"], expected_total)

        output["loss"].backward()
        self.assertGreater(model.predictor.out.weight.grad.abs().sum(), 0)

    def test_rollout_frame_budget_and_weight_rules(self):
        with self.assertRaises(ValueError):
            self._make_model(rollout_steps=1, rollout_weight=1.0)
        with self.assertRaises(ValueError):
            self._make_model(n_future=0)
        with self.assertRaises(ValueError):
            self._make_model(rollout_steps=0)


class LiberoSmoothLatentDataConfigTest(unittest.TestCase):
    def test_temporal_schema_and_mixture(self):
        config_path = (
            REPO_ROOT / "examples/LIBERO/train_files/data_registry/data_config.py"
        )
        spec = importlib.util.spec_from_file_location(
            "libero_smooth_latent_data_config", config_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        config = module.ROBOT_TYPE_CONFIG_MAP["libero_franka_smooth_latent_wm"]
        modalities = config.modality_config()
        self.assertEqual(list(modalities["video"].delta_indices), [0, 1, 2, 8])
        self.assertEqual(list(modalities["state"].delta_indices), [0])
        mixture = module.DATASET_NAMED_MIXTURES[
            "libero_all_smooth_latent_wm_l10_augmented"
        ]
        self.assertGreater(len(mixture), 0)
        self.assertTrue(
            all(
                robot_type == "libero_franka_smooth_latent_wm"
                for _, _, robot_type in mixture
            )
        )

        rollout_config = module.ROBOT_TYPE_CONFIG_MAP[
            "libero_franka_smooth_latent_wm_rollout"
        ]
        rollout_modalities = rollout_config.modality_config()
        self.assertEqual(
            list(rollout_modalities["video"].delta_indices), [0, 1, 2, 3, 4, 8]
        )
        rollout_mixture = module.DATASET_NAMED_MIXTURES[
            "libero_all_smooth_latent_wm_l10_augmented_rollout"
        ]
        self.assertGreater(len(rollout_mixture), 0)
        self.assertTrue(
            all(
                robot_type == "libero_franka_smooth_latent_wm_rollout"
                for _, _, robot_type in rollout_mixture
            )
        )


class SmoothGlobalLatentActionIntegrationTest(unittest.TestCase):
    def test_joint_action_path_uses_only_one_global_token_per_frame(self):
        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=32)
                self.anchor = nn.Parameter(torch.zeros(()))

        config = OmegaConf.load(
            REPO_ROOT
            / "examples/LIBERO/train_files/"
            "starvla_smooth_global_latent_action_joint_libero_200k.yaml"
        )
        wm = config.framework.world_model
        config.framework.lang_cond.num_buckets = 32
        config.framework.lang_cond.embed_dim = 16
        wm.visual_token_dim = 24
        wm.residual_predictor_dim = 24
        wm.residual_predictor_depth = 1
        wm.residual_predictor_heads = 4
        wm.residual_predictor_ffn = 48
        wm.smooth_spatial_token_dim = 8
        wm.smooth_latent_dim = 16
        wm.smooth_predictor_dim = 16
        wm.smooth_predictor_depth = 1
        wm.smooth_predictor_heads = 2
        wm.smooth_predictor_ffn = 32
        wm.smooth_sigreg_knots = 5
        wm.smooth_sigreg_num_proj = 16

        with patch(
            "starVLA.model.modules.world_model.dinov3_loader.load_dinov3",
            return_value=(FakeEncoder(), None, 0),
        ):
            from starVLA.model.framework.WM4A.LeWMOFT import LeWM_OFT

            model = LeWM_OFT(config=config)

        def fake_encode(_self, frames):
            return torch.randn(len(frames), len(frames[0]), 2, 16, 32)

        model.backbone.encode_patch_frames = MethodType(
            fake_encode, model.backbone
        )
        image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        examples = [
            {
                "image": [image, image],
                "future_images": [[image, image] for _ in range(3)],
                "future_frame_valid_mask": [True, True, True, True],
                "lang": f"task {index}",
                "action": np.random.randn(8, 7).astype(np.float32),
            }
            for index in range(2)
        ]

        output = model(examples)
        torch.testing.assert_close(
            output["action_loss"].detach(),
            output["l1_action_loss"] + output["smooth_loss"],
        )
        output["action_loss"].backward()

        allowed_prefixes = (
            "smooth_world_model.",
            "task_embedding.",
            "visual_action_head.",
            "action_model.",
        )
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith(allowed_prefixes) for name in trainable)
        )
        self.assertEqual(model.visual_action_head.num_tokens, 1)
        self.assertIsNotNone(
            model.smooth_world_model.projector.global_projection.weight.grad
        )
        self.assertIsNotNone(model.smooth_world_model.predictor.out.weight.grad)
        self.assertIsNotNone(model.visual_action_head.kv_proj.weight.grad)
        self.assertIsNotNone(model.action_model.model.fc2.weight.grad)

        prediction = model.predict_action(examples)
        self.assertEqual(prediction["normalized_actions"].shape, (2, 8, 7))


if __name__ == "__main__":
    unittest.main()
