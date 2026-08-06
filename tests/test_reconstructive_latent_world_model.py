import importlib.util
import unittest
from pathlib import Path

import torch

from starVLA.model.modules.world_model.reconstructive_latent_world_model import (
    ReconstructiveSpatialLatentWorldModel,
    SpatialBlockCodec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class SpatialBlockCodecTest(unittest.TestCase):
    def test_spatial_block_rearrangement_is_lossless(self):
        codec = SpatialBlockCodec(
            patch_dim=3,
            num_views=2,
            state_dim=4,
            pool_grid_size=4,
            latent_grid_size=2,
            latent_dim=8,
            codec_hidden_dim=16,
            state_hidden_dim=16,
        )
        spatial = torch.arange(2 * 3 * 2 * 4 * 4 * 3, dtype=torch.float32)
        spatial = spatial.reshape(2, 3, 2, 4, 4, 3)

        blocks = codec._grid_to_blocks(spatial)
        restored = codec._blocks_to_grid(blocks)

        self.assertEqual(blocks.shape, (2, 3, 8, 12))
        torch.testing.assert_close(restored, spatial)

    def test_patch_target_is_detached_and_normalized(self):
        codec = SpatialBlockCodec(
            patch_dim=8,
            num_views=2,
            state_dim=3,
            pool_grid_size=4,
            latent_grid_size=2,
            latent_dim=8,
            codec_hidden_dim=16,
            state_hidden_dim=16,
        )
        patches = torch.randn(2, 3, 2, 36, 8, requires_grad=True)

        target = codec.spatial_target(patches)

        self.assertEqual(target.shape, (2, 3, 2, 4, 4, 8))
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(
            target.mean(dim=-1),
            torch.zeros_like(target.mean(dim=-1)),
            atol=2e-5,
            rtol=0,
        )


class ReconstructiveSpatialLatentWorldModelTest(unittest.TestCase):
    def _make_model(self):
        return ReconstructiveSpatialLatentWorldModel(
            patch_dim=8,
            num_views=2,
            goal_dim=5,
            state_dim=3,
            n_future=2,
            context_len=1,
            pool_grid_size=4,
            latent_grid_size=2,
            latent_dim=8,
            codec_hidden_dim=16,
            state_hidden_dim=16,
            predictor_dim=16,
            predictor_depth=1,
            predictor_heads=2,
            predictor_ffn_dim=32,
            reconstruction_weight=1.0,
            prediction_weight=1.0,
            state_weight=0.5,
        )

    def test_three_loss_objective_shapes_and_gradient_flow(self):
        torch.manual_seed(7)
        model = self._make_model()
        model.train()
        patches = torch.randn(3, 3, 2, 16, 8, requires_grad=True)
        state = torch.randn(3, 3, 3)
        goal = torch.randn(3, 5)

        output = model(
            patches,
            state=state,
            goal=goal,
            update_stats=False,
        )

        expected = (
            output["reconstruction_loss"]
            + output["latent_prediction_loss"]
            + 0.5 * output["state_loss"]
        )
        torch.testing.assert_close(output["loss"], expected)
        self.assertEqual(output["latent"].shape, (3, 3, 8, 8))
        self.assertEqual(output["pred_future_latent"].shape, (3, 2, 8, 8))
        self.assertEqual(output["reconstructed_dino"].shape, (3, 3, 2, 4, 4, 8))
        self.assertEqual(output["decoded_state"].shape, (3, 3, 3))
        self.assertFalse(
            any("variance" in key or "covariance" in key for key in output)
        )

        # Non-affine LayerNorm fixes each token's scale without adding another
        # optimization objective.
        latent = output["latent"].detach()
        torch.testing.assert_close(
            latent.mean(dim=-1),
            torch.zeros_like(latent.mean(dim=-1)),
            atol=2e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            latent.square().mean(dim=-1),
            torch.ones_like(latent.square().mean(dim=-1)),
            atol=2e-3,
            rtol=0,
        )

        output["loss"].backward()
        self.assertIsNone(patches.grad)
        for module in (
            model.codec.encoder,
            model.codec.decoder,
            model.codec.state_decoder,
            model.predictor.residual_predictor,
        ):
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    and parameter.grad.abs().sum() > 0
                    for parameter in module.parameters()
                )
            )

    def test_state_must_align_with_every_image_frame(self):
        model = self._make_model()
        patches = torch.randn(2, 3, 2, 16, 8)

        with self.assertRaisesRegex(ValueError, "expected aligned state shape"):
            model(
                patches,
                state=torch.randn(2, 1, 3),
                goal=torch.randn(2, 5),
                update_stats=False,
            )


class RoboTwinReconstructiveDataConfigTest(unittest.TestCase):
    def test_image_and_state_indices_are_identical(self):
        config_path = (
            REPO_ROOT
            / "examples/Robotwin/train_files/data_registry/data_config.py"
        )
        spec = importlib.util.spec_from_file_location(
            "robotwin_reconstructive_data_config", config_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        config = module.ROBOT_TYPE_CONFIG_MAP["robotwin_reconstructive_wm"]
        modalities = config.modality_config()
        self.assertEqual(list(modalities["video"].delta_indices), [0, 1, 2])
        self.assertEqual(list(modalities["state"].delta_indices), [0, 1, 2])
        mixture = module.DATASET_NAMED_MIXTURES[
            "robotwin_clean_reconstructive_wm"
        ]
        self.assertGreater(len(mixture), 0)
        self.assertTrue(
            all(robot_type == "robotwin_reconstructive_wm" for _, _, robot_type in mixture)
        )


if __name__ == "__main__":
    unittest.main()
