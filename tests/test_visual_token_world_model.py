import unittest

import torch

from starVLA.model.framework.WM4A.LeWMOFT import (
    DensePatchActionAdapter,
    LeWM_OFT,
    VisualActionCrossAttn,
    VisualTokenPooler,
    prefix_l1_loss,
)
from starVLA.model.modules.world_model.visual_token_delta_world_model import (
    VisualTokenLatentWorldModel,
)
from starVLA.model.modules.world_model.wala_transition_auxiliary import (
    WALAVisualTransitionAuxiliary,
    token_cosine_loss,
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

    def test_dense_grid_preserves_every_patch_without_pooling(self):
        pooler = VisualTokenPooler(
            patch_dim=8,
            token_dim=8,
            num_views=2,
            tokens_per_view=196,
        )
        patches = torch.randn(2, 3, 2, 196, 8, requires_grad=True)
        tokens, content = pooler(patches, return_content=True)

        manual = pooler.out_norm(pooler.patch_proj(pooler.patch_norm(patches)))
        manual = manual.reshape(2, 3, 392, 8)
        self.assertEqual(tokens.shape, (2, 3, 392, 8))
        self.assertTrue(torch.equal(content, manual))
        self.assertTrue(torch.allclose(pooler.remove_position(tokens), content))

        tokens.square().mean().backward()
        self.assertTrue(torch.isfinite(patches.grad).all())

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

    def test_dense_patch_action_is_zero_gated_then_learns_residual(self):
        torch.manual_seed(4)
        adapter = DensePatchActionAdapter(
            patch_dim=12,
            action_hidden_dim=32,
            hidden_dim=16,
            num_views=2,
            patch_grid_size=4,
            num_heads=4,
            gate_init=1.0,
        )
        base_queries = torch.randn(2, 8, 32, requires_grad=True)
        patches = torch.randn(2, 2, 16, 12, requires_grad=True)

        output, metrics = adapter(base_queries, patches)
        self.assertTrue(torch.equal(output, base_queries))
        self.assertGreater(float(metrics["dense_patch_gate"]), 0.7)
        output.square().mean().backward()
        self.assertGreater(float(adapter.out_proj.weight.grad.abs().sum()), 0.0)
        self.assertTrue(torch.equal(patches.grad, torch.zeros_like(patches.grad)))

        adapter.zero_grad(set_to_none=True)
        base_queries.grad = None
        patches.grad = None
        torch.nn.init.normal_(adapter.out_proj.weight, std=0.01)
        output, metrics = adapter(base_queries, patches)
        self.assertFalse(torch.allclose(output, base_queries))
        self.assertGreater(float(metrics["dense_patch_query_update_ratio"]), 0.0)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(patches.grad).all())
        self.assertGreater(float(patches.grad.abs().sum()), 0.0)

    def test_dense_patch_action_never_reads_true_future_patches(self):
        torch.manual_seed(5)
        framework = object.__new__(LeWM_OFT)
        torch.nn.Module.__init__(framework)
        framework.wm_ctx_len = 1
        framework.dense_patch_action = DensePatchActionAdapter(
            patch_dim=12,
            action_hidden_dim=32,
            hidden_dim=16,
            num_views=2,
            patch_grid_size=4,
            num_heads=4,
            gate_init=0.3,
        )
        torch.nn.init.normal_(framework.dense_patch_action.out_proj.weight, std=0.01)
        base_queries = torch.randn(2, 8, 32)
        temporal_patches = torch.randn(2, 3, 2, 16, 12)

        output, _ = LeWM_OFT._augment_action_queries_with_dense_patches(
            framework, base_queries, temporal_patches
        )
        future_changed = temporal_patches.clone()
        future_changed[:, 1:] = torch.randn_like(future_changed[:, 1:]) * 100.0
        output_future_changed, _ = (
            LeWM_OFT._augment_action_queries_with_dense_patches(
                framework, base_queries, future_changed
            )
        )
        self.assertTrue(torch.equal(output, output_future_changed))

        current_changed = temporal_patches.clone()
        current_changed[:, 0] = torch.randn_like(current_changed[:, 0]) * 100.0
        output_current_changed, _ = (
            LeWM_OFT._augment_action_queries_with_dense_patches(
                framework, base_queries, current_changed
            )
        )
        self.assertFalse(torch.allclose(output, output_current_changed))


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

    def test_delta_scale_can_be_frozen_for_auxiliary_stages(self):
        model = VisualTokenLatentWorldModel(
            latent_dim=8,
            goal_dim=6,
            n_future=2,
            num_tokens=4,
            dim=16,
            depth=1,
            num_heads=4,
            ffn_dim=32,
        )
        model.train()
        model.delta_scale.fill_(3.0)
        model._delta_scale_ready.fill_(1.0)
        latent = torch.randn(2, 3, 4, 8)

        model(latent, ctx_len=1, update_stats=False)

        self.assertEqual(float(model.delta_scale), 3.0)


class WALAVisualTransitionAuxiliaryTest(unittest.TestCase):
    def _make_module(self):
        return WALAVisualTransitionAuxiliary(
            latent_dim=12,
            action_hidden_dim=16,
            hidden_dim=24,
            num_visual_tokens=6,
            num_future=2,
            num_action_queries=4,
            num_transition_tokens=3,
            encoder_depth=1,
            decoder_depth=1,
            resampler_depth=1,
            num_heads=4,
        )

    def test_teacher_reconstructs_ground_truth_transition_shape(self):
        module = self._make_module()
        current = torch.randn(2, 6, 12)
        future_delta = torch.randn(2, 2, 6, 12)

        tokens, reconstruction = module.teacher_forward(current, future_delta)
        loss = torch.nn.functional.smooth_l1_loss(reconstruction, future_delta)
        loss = loss + 0.1 * token_cosine_loss(reconstruction, future_delta)
        loss.backward()

        self.assertEqual(tokens.shape, (2, 3, 24))
        self.assertEqual(reconstruction.shape, future_delta.shape)
        self.assertTrue(
            any(parameter.grad is not None for parameter in module.teacher_encoder.parameters())
        )

    def test_student_gradient_uses_frozen_teacher_decoder(self):
        module = self._make_module()
        module.teacher_encoder.requires_grad_(False)
        module.teacher_decoder.requires_grad_(False)
        current = torch.randn(2, 6, 12)
        action_queries = torch.randn(2, 4, 16, requires_grad=True)

        student_tokens, prediction = module.student_forward(action_queries, current)
        loss = student_tokens.square().mean() + prediction.square().mean()
        loss.backward()

        self.assertEqual(student_tokens.shape, (2, 3, 24))
        self.assertEqual(prediction.shape, (2, 2, 6, 12))
        self.assertIsNotNone(action_queries.grad)
        self.assertTrue(torch.isfinite(action_queries.grad).all())
        self.assertTrue(
            all(parameter.grad is None for parameter in module.teacher_decoder.parameters())
        )

    def test_combined_mode_updates_teacher_student_and_action_queries(self):
        class _FrozenWorldModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("delta_scale", torch.tensor(2.0))
                self._stats_eps = 1e-6

        framework = object.__new__(LeWM_OFT)
        torch.nn.Module.__init__(framework)
        framework.transition_auxiliary = self._make_module()
        framework.transition_mode = "combined"
        framework.wm_ctx_len = 1
        framework.n_future = 2
        framework.world_model = _FrozenWorldModel()
        framework.transition_cosine_weight = 0.1
        framework.transition_alignment_l1_weight = 0.1
        framework.transition_detach_action_queries = False

        action_queries = torch.randn(2, 4, 16, requires_grad=True)
        latent = torch.randn(2, 3, 6, 12, requires_grad=True)

        losses = LeWM_OFT._transition_auxiliary_losses(
            framework,
            action_queries,
            latent,
        )
        total = (
            losses["transition_teacher_recon_loss"]
            + 0.005 * losses["transition_alignment_loss"]
            + 0.05 * losses["transition_decode_loss"]
        )
        total.backward()

        self.assertIsNotNone(action_queries.grad)
        self.assertIsNone(latent.grad)
        for submodule in (
            framework.transition_auxiliary.teacher_encoder,
            framework.transition_auxiliary.teacher_decoder,
            framework.transition_auxiliary.student_resampler,
        ):
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    for parameter in submodule.parameters()
                )
            )

    def test_teacher_validates_future_shape(self):
        module = self._make_module()
        with self.assertRaisesRegex(ValueError, "future_delta"):
            module.teacher_forward(
                torch.randn(2, 6, 12),
                torch.randn(2, 1, 6, 12),
            )


if __name__ == "__main__":
    unittest.main()
