import unittest

import torch

from starVLA.model.framework.WM4A.LeWMOFT import (
    LeWM_OFT,
    VisualActionCrossAttn,
    VisualTokenPooler,
    prefix_l1_loss,
)
from starVLA.model.modules.world_model.token_wan_world_model import TokenWanWorldModel


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


class TokenWanWorldModelTest(unittest.TestCase):
    def test_delta_training_and_inference(self):
        torch.manual_seed(0)
        model = TokenWanWorldModel(
            latent_dim=16,
            action_dim=14,
            goal_dim=12,
            dim=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=64,
            num_tokens=8,
            token_grid_shape=(2, 2, 2),
            ctx_len=1,
            flow_sample_steps=2,
            num_train_timesteps=32,
            scheduler_kwargs={"shift": 5.0, "sigma_min": 0.0, "extra_one_step": True},
            action_scheduler_kwargs={
                "shift": 0.05,
                "sigma_min": 0.0,
                "extra_one_step": True,
            },
            delta_head_futures=2,
            delta_head_inference=True,
            delta_head_dim=32,
            delta_head_depth=2,
            delta_head_heads=4,
            delta_head_ffn=64,
            delta_head_sigreg_weight=0.02,
        )
        latent = torch.randn(2, 3, 8, 16, requires_grad=True)
        action = torch.randn(2, 2, 14)
        goal = torch.randn(2, 12)

        output = model.flow_loss(latent, action, goal=goal)
        self.assertEqual(output["delta_future_latent"].shape, (2, 2, 8, 16))
        self.assertIn("delta_sigreg_loss", output)
        self.assertIn("delta_to_copy_ratio", output)
        self.assertTrue(torch.isfinite(output["delta_direction_cosine"]))
        loss = (
            output["flow_latent_loss"]
            + output["flow_action_loss"]
            + output["delta_latent_loss"]
            + 0.02 * output["delta_sigreg_loss"]
        )
        loss.backward()
        self.assertTrue(torch.isfinite(latent.grad).all())

        model.eval()
        context = latent[:, :1].detach()
        future, sampled_action = model.sample_future(
            context,
            goal=goal,
            n_future=2,
            num_steps=2,
            return_action=True,
        )
        self.assertEqual(future.shape, (2, 2, 8, 16))
        self.assertEqual(sampled_action.shape, (2, 2, 14))
        self.assertTrue(torch.allclose(future, model.regress_future(context, goal=goal)))


if __name__ == "__main__":
    unittest.main()
