import unittest

import torch

from starVLA.model.framework.WM4A.LeWMOFT import (
    LeWM_OFT,
    VisualActionCrossAttn,
    VisualTokenPooler,
    prefix_l1_loss,
)
from starVLA.model.modules.world_model.visual_token_delta_world_model import (
    VisualTokenLatentWorldModel,
)


class VisualTokenComponentsTest(unittest.TestCase):
    def test_prefix_l1_loss_masks_tail_per_sample(self):
        pred = torch.zeros(2, 4, 1, requires_grad=True)
        target = torch.tensor(
            [[[1.0], [2.0], [30.0], [40.0]], [[1.0], [2.0], [3.0], [4.0]]]
        )
        loss = prefix_l1_loss(pred, target, torch.tensor([2, 4]))

        self.assertTrue(torch.isclose(loss, torch.tensor(13.0 / 6.0)))
        loss.backward()
        self.assertTrue(torch.equal(pred.grad[0, 2:], torch.zeros(2, 1)))
        self.assertTrue(torch.all(pred.grad[0, :2] != 0))
        self.assertTrue(torch.all(pred.grad[1] != 0))

    def test_full_prefix_matches_regular_l1(self):
        pred = torch.randn(3, 8, 7)
        target = torch.randn_like(pred)
        prefix = prefix_l1_loss(pred, target, torch.full((3,), 8))
        self.assertTrue(torch.allclose(prefix, torch.nn.functional.l1_loss(pred, target)))

    def test_spatial_pooling_and_action_readout(self):
        pooler = VisualTokenPooler(
            patch_dim=32,
            token_dim=24,
            num_views=2,
            tokens_per_view=16,
        )
        patches = torch.randn(2, 3, 2, 196, 32, requires_grad=True)
        tokens, content = pooler(patches, return_content=True)

        self.assertEqual(tokens.shape, (2, 3, 32, 24))
        self.assertTrue(torch.allclose(pooler.remove_position(tokens), content))
        self.assertFalse(torch.allclose(tokens[:, :, 0], tokens[:, :, 1]))

        action_head = VisualActionCrossAttn(
            token_dim=24,
            action_hidden_dim=32,
            chunk_len=8,
            num_frames=3,
            num_tokens=32,
            num_heads=4,
        )
        queries = action_head(tokens)
        self.assertEqual(queries.shape, (2, 8, 32))

        framework = object.__new__(LeWM_OFT)
        framework.visual_token_min_std = 0.1
        diversity, variance, mean_cosine = LeWM_OFT._visual_token_regularization(
            framework, content
        )
        diagnostics = LeWM_OFT._visual_content_diagnostics(framework, content)
        loss = queries.square().mean() + 0.02 * diversity + 0.02 * variance
        loss.backward()
        self.assertTrue(torch.isfinite(patches.grad).all())
        self.assertTrue(torch.isfinite(mean_cosine))
        self.assertGreater(float(diagnostics["effective_rank"]), 1.0)

    def test_state_conditioning_is_zero_initialized(self):
        torch.manual_seed(1)
        base = VisualActionCrossAttn(
            token_dim=24,
            action_hidden_dim=32,
            chunk_len=8,
            num_frames=3,
            num_tokens=32,
            num_heads=4,
        )
        conditioned = VisualActionCrossAttn(
            token_dim=24,
            action_hidden_dim=32,
            chunk_len=8,
            num_frames=3,
            num_tokens=32,
            num_heads=4,
            state_dim=8,
            state_hidden_dim=16,
            state_dropout=0.1,
        )
        conditioned.load_state_dict(base.state_dict(), strict=False)
        tokens = torch.randn(2, 3, 32, 24)
        state = torch.randn(2, 8)

        base.eval()
        conditioned.eval()
        self.assertTrue(torch.allclose(base(tokens), conditioned(tokens, state=state)))

        torch.nn.init.normal_(conditioned.state_encoder[-1].weight, std=0.01)
        self.assertFalse(torch.allclose(base(tokens), conditioned(tokens, state=state)))

    def test_state_conditioning_requires_current_state(self):
        action_head = VisualActionCrossAttn(
            token_dim=24,
            action_hidden_dim=32,
            chunk_len=8,
            num_frames=3,
            num_tokens=32,
            num_heads=4,
            state_dim=8,
        )
        with self.assertRaisesRegex(ValueError, "requires current state"):
            action_head(torch.randn(1, 3, 32, 24))


class VisualTokenLatentWorldModelTest(unittest.TestCase):
    def test_legacy_delta_head_checkpoint_keys_are_remapped(self):
        framework = object.__new__(LeWM_OFT)
        legacy_weight = torch.randn(3, 4)
        remapped = LeWM_OFT.remap_checkpoint_state_dict(
            framework,
            {"world_model.delta_head.out.weight": legacy_weight},
        )

        self.assertNotIn("world_model.delta_head.out.weight", remapped)
        self.assertIs(
            remapped["world_model.residual_predictor.out.weight"], legacy_weight
        )

    def test_delta_training_and_inference(self):
        torch.manual_seed(0)
        model = VisualTokenLatentWorldModel(
            latent_dim=16,
            goal_dim=12,
            n_future=2,
            num_tokens=8,
            dim=32,
            depth=2,
            num_heads=4,
            ffn_dim=64,
            sigreg_weight=0.02,
        )
        latent = torch.randn(2, 3, 8, 16, requires_grad=True)
        goal = torch.randn(2, 12)

        output = model(latent, ctx_len=1, goal=goal)
        self.assertEqual(output["pred_future_latent"].shape, (2, 2, 8, 16))
        self.assertIn("sigreg_loss", output)
        self.assertIn("delta_to_copy_ratio", output)
        self.assertTrue(torch.isfinite(output["delta_direction_cosine"]))
        loss = output["latent_loss"] + 0.02 * output["sigreg_loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(latent.grad).all())

        model.eval()
        context = latent[:, :1].detach()
        future = model.regress_future(context, goal=goal)
        self.assertEqual(future.shape, (2, 2, 8, 16))
        self.assertTrue(torch.allclose(future, output["pred_future_latent"].detach()))
        state_keys = tuple(model.state_dict())
        self.assertTrue(any("residual_predictor." in key for key in state_keys))
        self.assertFalse(any("delta_head." in key for key in state_keys))
        for removed_name in (
            "predictor",
            "scheduler",
            "action_scheduler",
            "flow_loss",
            "sample_future",
        ):
            self.assertFalse(hasattr(model, removed_name))


if __name__ == "__main__":
    unittest.main()
