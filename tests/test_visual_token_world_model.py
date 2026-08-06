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

    def test_delta_scale_keeps_exact_fp32_value_under_bfloat16_cast(self):
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
        expected = torch.tensor(1.680324673652649, dtype=torch.float32)
        model.delta_scale.copy_(expected)
        model._delta_scale_ready.fill_(1.0)

        model.to(dtype=torch.bfloat16)

        self.assertEqual(model.delta_scale.dtype, torch.float32)
        self.assertTrue(torch.equal(model.delta_scale.cpu(), expected.reshape_as(model.delta_scale)))
        self.assertEqual(model._delta_scale_ready.dtype, torch.float32)

    def test_history_and_state_condition_future_prediction(self):
        torch.manual_seed(7)
        model = VisualTokenLatentWorldModel(
            latent_dim=12,
            goal_dim=10,
            n_future=2,
            num_tokens=6,
            context_len=2,
            state_dim=8,
            dim=24,
            depth=2,
            num_heads=4,
            ffn_dim=48,
        )
        torch.nn.init.normal_(model.residual_predictor.out.weight, std=0.01)
        latent = torch.randn(3, 4, 6, 12, requires_grad=True)
        goal = torch.randn(3, 10)
        state = torch.randn(3, 8)

        output = model(latent, ctx_len=2, goal=goal, state=state)
        self.assertEqual(output["pred_future_latent"].shape, (3, 2, 6, 12))
        self.assertIn("latent_cosine_loss", output)
        self.assertIn("latent_loss_horizon_1", output)
        self.assertIn("latent_loss_horizon_2", output)
        (output["latent_loss"] + 0.1 * output["latent_cosine_loss"]).backward()
        self.assertIsNotNone(model.residual_predictor.history_proj.weight.grad)
        self.assertGreater(
            float(model.residual_predictor.history_proj.weight.grad.abs().sum()),
            0.0,
        )
        self.assertIsNotNone(model.residual_predictor.state_proj.weight.grad)

        with self.assertRaisesRegex(ValueError, "requires current state"):
            model(latent.detach(), ctx_len=2, goal=goal)

    def test_rollout_first_step_matches_single_shot_regression(self):
        torch.manual_seed(11)
        model = VisualTokenLatentWorldModel(
            latent_dim=12,
            goal_dim=10,
            n_future=2,
            num_tokens=6,
            dim=24,
            depth=2,
            num_heads=4,
            ffn_dim=48,
        )
        torch.nn.init.normal_(model.residual_predictor.out.weight, std=0.02)
        model.delta_scale.fill_(1.7)
        model._delta_scale_ready.fill_(1.0)
        context = torch.randn(2, 1, 6, 12)
        goal = torch.randn(2, 10)

        single = model.regress_future(context, goal=goal)
        rolled = model.rollout_future(context, steps=3, goal=goal)

        self.assertEqual(rolled.shape, (2, 6, 6, 12))
        self.assertTrue(torch.equal(rolled[:, :2], single))
        with self.assertRaisesRegex(ValueError, "at least 1"):
            model.rollout_future(context, steps=0, goal=goal)

    def test_rollout_losses_supervise_later_steps_only(self):
        torch.manual_seed(12)
        model = VisualTokenLatentWorldModel(
            latent_dim=12,
            goal_dim=10,
            n_future=2,
            num_tokens=6,
            dim=24,
            depth=2,
            num_heads=4,
            ffn_dim=48,
        )
        torch.nn.init.normal_(model.residual_predictor.out.weight, std=0.02)
        latent = torch.randn(2, 7, 6, 12)
        goal = torch.randn(2, 10)

        single = model(latent, ctx_len=1, goal=goal, update_stats=False)
        rolled = model(
            latent, ctx_len=1, goal=goal, update_stats=False, rollout_steps=3
        )

        # Headline objective stays identical so old checkpoints stay comparable.
        self.assertTrue(torch.equal(single["latent_loss"], rolled["latent_loss"]))
        self.assertNotIn("rollout_latent_loss", single)
        self.assertNotIn("rollout_latent_loss_step_1", rolled)
        for step in (2, 3):
            self.assertIn(f"rollout_latent_loss_step_{step}", rolled)
        for step in (1, 2, 3):
            self.assertIn(f"rollout_to_copy_ratio_step_{step}", rolled)
            self.assertIn(f"rollout_direction_cosine_step_{step}", rolled)

        rolled["rollout_latent_loss"].backward()
        self.assertGreater(
            float(model.residual_predictor.out.weight.grad.abs().sum()), 0.0
        )

    def test_rollout_requires_enough_future_frames(self):
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
        latent = torch.randn(2, 5, 4, 8)

        with self.assertRaisesRegex(ValueError, "rollout_steps=3 needs 6 future"):
            model(latent, ctx_len=1, update_stats=False, rollout_steps=3)

    def test_history_and_state_adapters_preserve_one_frame_warm_start(self):
        torch.manual_seed(8)
        base = VisualTokenLatentWorldModel(
            latent_dim=12,
            goal_dim=10,
            n_future=2,
            num_tokens=6,
            dim=24,
            depth=2,
            num_heads=4,
            ffn_dim=48,
        )
        enhanced = VisualTokenLatentWorldModel(
            latent_dim=12,
            goal_dim=10,
            n_future=2,
            num_tokens=6,
            context_len=2,
            state_dim=8,
            dim=24,
            depth=2,
            num_heads=4,
            ffn_dim=48,
        )
        enhanced.load_state_dict(base.state_dict(), strict=False)
        current = torch.randn(2, 1, 6, 12)
        history = torch.randn_like(current)
        goal = torch.randn(2, 10)

        base_future = base.regress_future(current, goal=goal)
        enhanced_future = enhanced.regress_future(
            torch.cat([history, current], dim=1),
            goal=goal,
            state=torch.randn(2, 8),
        )
        self.assertTrue(torch.equal(base_future, enhanced_future))

    def test_context_correction_preserves_base_then_learns_from_histories(self):
        torch.manual_seed(9)
        base = VisualTokenLatentWorldModel(
            latent_dim=12,
            goal_dim=10,
            n_future=2,
            num_tokens=6,
            context_len=3,
            dim=24,
            depth=2,
            num_heads=4,
            ffn_dim=48,
        )
        boosted = VisualTokenLatentWorldModel(
            latent_dim=12,
            goal_dim=10,
            n_future=2,
            num_tokens=6,
            context_len=3,
            dim=24,
            depth=2,
            num_heads=4,
            ffn_dim=48,
            context_correction_depth=2,
            context_correction_state_dim=8,
            context_correction_dim=24,
            context_correction_heads=4,
            context_correction_ffn_dim=48,
        )
        boosted.load_state_dict(base.state_dict(), strict=False)
        boosted.delta_scale.copy_(base.delta_scale)
        context = torch.randn(2, 3, 6, 12)
        goal = torch.randn(2, 10)
        state_history = torch.randn(2, 3, 8)

        expected = base.regress_future(context, goal=goal)
        actual = boosted.regress_future(
            context, goal=goal, state=state_history
        )
        self.assertTrue(torch.equal(expected, actual))

        latent = torch.randn(2, 5, 6, 12)
        output = boosted(
            latent, ctx_len=3, goal=goal, state=state_history
        )
        self.assertIn("latent_base_loss", output)
        self.assertEqual(float(output["context_correction_rms"]), 0.0)
        output["latent_loss"].backward()
        correction = boosted.context_correction
        self.assertGreater(float(correction.out.weight.grad.abs().sum()), 0.0)

        with torch.no_grad():
            correction.out.weight.normal_(std=0.01)
        changed_history = context.clone()
        changed_history[:, 0] += 3.0
        changed_state = state_history.clone()
        changed_state[:, 0] += 2.0
        changed = boosted.regress_future(
            changed_history, goal=goal, state=changed_state
        )
        original = boosted.regress_future(
            context, goal=goal, state=state_history
        )
        self.assertFalse(torch.allclose(changed, original))

    def test_predictable_innovation_is_exact_warm_start_and_orthonormal(self):
        torch.manual_seed(10)
        base = VisualTokenLatentWorldModel(
            latent_dim=12,
            goal_dim=10,
            n_future=2,
            num_tokens=6,
            dim=24,
            depth=2,
            num_heads=4,
            ffn_dim=48,
        )
        innovation = VisualTokenLatentWorldModel(
            latent_dim=12,
            goal_dim=10,
            n_future=2,
            num_tokens=6,
            dim=24,
            depth=2,
            num_heads=4,
            ffn_dim=48,
            predictable_innovation_enabled=True,
            innovation_rank=4,
            innovation_predictor_dim=24,
            innovation_predictor_depth=2,
            innovation_predictor_heads=4,
            innovation_predictor_ffn_dim=48,
        )
        missing, unexpected = innovation.load_state_dict(base.state_dict(), strict=False)
        self.assertTrue(missing)
        self.assertTrue(
            all(key.startswith("predictable_innovation.") for key in missing)
        )
        self.assertFalse(unexpected)

        latent = torch.randn(3, 3, 6, 12)
        goal = torch.randn(3, 10)
        base.eval()
        innovation.eval()
        base_output = base(latent, ctx_len=1, goal=goal, update_stats=False)
        output = innovation(latent, ctx_len=1, goal=goal, update_stats=False)

        self.assertTrue(
            torch.equal(output["pred_future_latent"], base_output["pred_future_latent"])
        )
        self.assertTrue(torch.equal(output["latent_loss"], output["latent_base_loss"]))
        self.assertEqual(float(output["innovation_pred_code_std"]), 0.0)
        self.assertEqual(float(output["innovation_code_cosine"]), 0.0)
        self.assertTrue(
            torch.equal(output["innovation_code_nmse"], output["innovation_code_loss"])
        )
        self.assertLessEqual(
            float(output["innovation_oracle_raw_loss"]),
            float(output["latent_base_loss"]) + 1e-6,
        )
        self.assertTrue(
            torch.equal(
                innovation.regress_future(latent[:, :1], goal=goal),
                output["pred_future_latent"],
            )
        )

        basis = innovation.predictable_innovation.basis()
        gram = basis @ basis.transpose(-1, -2)
        identity = torch.eye(4).expand_as(gram)
        self.assertTrue(torch.allclose(gram, identity, atol=1e-5, rtol=1e-5))
        spatial_basis = innovation.predictable_innovation.spatial_basis()
        spatial_gram = spatial_basis @ spatial_basis.transpose(-1, -2)
        spatial_identity = torch.eye(4).expand_as(spatial_gram)
        self.assertTrue(
            torch.allclose(
                spatial_gram, spatial_identity, atol=1e-5, rtol=1e-5
            )
        )

    def test_predictable_innovation_has_no_future_leakage_and_trains_only_branch(self):
        torch.manual_seed(11)
        model = VisualTokenLatentWorldModel(
            latent_dim=12,
            goal_dim=10,
            n_future=2,
            num_tokens=6,
            dim=24,
            depth=1,
            num_heads=4,
            ffn_dim=48,
            predictable_innovation_enabled=True,
            innovation_rank=4,
            innovation_predictor_dim=24,
            innovation_predictor_depth=1,
            innovation_predictor_heads=4,
            innovation_predictor_ffn_dim=48,
            innovation_min_std=0.25,
        )
        model.requires_grad_(False)
        model.predictable_innovation.requires_grad_(True)
        context = torch.randn(2, 1, 6, 12)
        goal = torch.randn(2, 10)
        with torch.no_grad():
            base_prediction = model.residual_predictor(context, goal=goal)
        target_one = torch.randn_like(base_prediction)
        target_two = target_one + torch.randn_like(target_one)

        first = model.predictable_innovation(
            context.detach(), base_prediction, target_delta=target_one, goal=goal
        )
        second = model.predictable_innovation(
            context.detach(), base_prediction, target_delta=target_two, goal=goal
        )
        self.assertTrue(torch.equal(first["predicted_code"], second["predicted_code"]))
        self.assertTrue(torch.equal(first["final_prediction"], second["final_prediction"]))
        self.assertFalse(torch.allclose(first["target_code"], second["target_code"]))

        loss = (
            first["innovation_final_mse"]
            + first["innovation_code_loss"]
            + 0.25 * first["innovation_capture_loss"]
            + first["innovation_variance_loss"]
            + 0.01 * first["innovation_covariance_loss"]
        )
        loss.backward()
        self.assertIsNone(model.residual_predictor.out.weight.grad)
        self.assertGreater(
            float(model.predictable_innovation.predictor.output.weight.grad.abs().sum()),
            0.0,
        )
        self.assertTrue(
            torch.isfinite(model.predictable_innovation.basis.raw_basis.grad).all()
        )
        self.assertTrue(
            torch.isfinite(
                model.predictable_innovation.spatial_basis.raw_basis.grad
            ).all()
        )

        constant_target = torch.ones_like(target_one)
        constant = model.predictable_innovation(
            context.detach() * 0,
            torch.zeros_like(base_prediction),
            target_delta=constant_target,
            goal=goal * 0,
        )
        self.assertGreater(float(constant["innovation_variance_loss"]), 0.2)

    def test_local_increment_projection_and_cumulative_round_trip(self):
        torch.manual_seed(12)
        model = VisualTokenLatentWorldModel(
            latent_dim=8,
            goal_dim=6,
            n_future=2,
            num_tokens=4,
            dim=16,
            depth=1,
            num_heads=4,
            ffn_dim=32,
            predictable_innovation_enabled=True,
            innovation_rank=8,
            innovation_transition_tokens=4,
            innovation_predictor_dim=16,
            innovation_predictor_depth=1,
            innovation_predictor_heads=4,
            innovation_predictor_ffn_dim=32,
        )
        module = model.predictable_innovation
        cumulative = torch.randn(3, 2, 4, 8)
        local = module._to_local(cumulative)
        self.assertTrue(
            torch.allclose(
                module._to_cumulative(local), cumulative, atol=1e-6, rtol=1e-6
            )
        )

        channel_basis = module.basis()
        spatial_basis = module.spatial_basis()
        code = module._project(local, spatial_basis, channel_basis)
        decoded = module._decode(code, spatial_basis, channel_basis)
        self.assertTrue(torch.allclose(decoded, local, atol=1e-5, rtol=1e-5))
        self.assertTrue(
            torch.allclose(
                module._to_cumulative(decoded),
                cumulative,
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_fixed_mean_target_is_independent_of_cobatch_and_removes_token_bias(self):
        torch.manual_seed(13)
        model = VisualTokenLatentWorldModel(
            latent_dim=8,
            goal_dim=6,
            n_future=2,
            num_tokens=4,
            dim=16,
            depth=1,
            num_heads=4,
            ffn_dim=32,
            predictable_innovation_enabled=True,
            innovation_rank=4,
            innovation_transition_tokens=2,
            innovation_predictor_dim=16,
            innovation_predictor_depth=1,
            innovation_predictor_heads=4,
            innovation_predictor_ffn_dim=32,
            innovation_min_std=0.25,
        )
        module = model.predictable_innovation
        fixed_mean = torch.randn(1, 2, 4, 8)
        module.set_fixed_error_mean(fixed_mean, sample_count=4096)
        base = torch.zeros(2, 2, 4, 8)
        context = torch.randn(2, 1, 4, 8)
        goal = torch.randn(2, 6)
        sample_local = fixed_mean + torch.randn_like(fixed_mean) * 0.1
        sample_target = module._to_cumulative(sample_local)

        target_a = torch.cat((sample_target, torch.randn_like(sample_target)), dim=0)
        target_b = torch.cat((sample_target, torch.randn_like(sample_target) * 100), dim=0)
        out_a = module(context, base, target_delta=target_a, goal=goal)
        out_b = module(context, base, target_delta=target_b, goal=goal)
        self.assertTrue(torch.equal(out_a["target_code"][0], out_b["target_code"][0]))

        fixed_bias_target = module._to_cumulative(fixed_mean).expand(2, -1, -1, -1)
        fixed_bias = module(
            context,
            base,
            target_delta=fixed_bias_target,
            goal=goal,
        )
        self.assertLess(float(fixed_bias["target_code"].abs().max()), 1e-6)
        self.assertGreater(float(fixed_bias["innovation_variance_loss"]), 0.2)

        module.to(dtype=torch.bfloat16)
        self.assertEqual(module.fixed_local_error_mean.dtype, torch.float32)
        self.assertEqual(module.fixed_local_error_count.dtype, torch.float32)

    def test_private_ctx3_keeps_legacy_ctx1_frame_semantics(self):
        framework = object.__new__(LeWM_OFT)
        framework.predictable_innovation_enabled = True
        framework.innovation_context_len = 3
        framework.wm_ctx_len = 1
        framework.n_future = 2
        sentinel = torch.arange(5.0).view(1, 5, 1, 1)

        legacy, innovation_context, indices = (
            LeWM_OFT._select_innovation_training_sequence(framework, sentinel)
        )

        self.assertEqual(indices, (2, 3, 4))
        self.assertTrue(torch.equal(innovation_context.flatten(), torch.tensor([0.0, 1.0, 2.0])))
        self.assertTrue(torch.equal(legacy.flatten(), torch.tensor([2.0, 3.0, 4.0])))

    def test_private_ctx3_forward_and_regress_match_after_nonzero_correction(self):
        torch.manual_seed(14)
        model = VisualTokenLatentWorldModel(
            latent_dim=8,
            goal_dim=6,
            n_future=2,
            num_tokens=4,
            context_len=1,
            dim=16,
            depth=1,
            num_heads=4,
            ffn_dim=32,
            predictable_innovation_enabled=True,
            innovation_rank=4,
            innovation_transition_tokens=2,
            innovation_context_len=3,
            innovation_predictor_dim=16,
            innovation_predictor_depth=1,
            innovation_predictor_heads=4,
            innovation_predictor_ffn_dim=32,
        )
        torch.nn.init.normal_(
            model.predictable_innovation.predictor.output.weight, std=0.01
        )
        legacy_latent = torch.randn(2, 3, 4, 8)
        private_context = torch.randn(2, 3, 4, 8)
        private_context[:, -1:] = legacy_latent[:, :1]
        goal = torch.randn(2, 6)
        model.eval()

        output = model(
            legacy_latent,
            ctx_len=1,
            goal=goal,
            innovation_context=private_context,
            update_stats=False,
        )
        regressed = model.regress_future(
            legacy_latent[:, :1],
            goal=goal,
            innovation_context=private_context,
        )
        self.assertTrue(torch.equal(output["pred_future_latent"], regressed))


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

    def test_latent_stats_follow_trainable_visual_coordinates(self):
        framework = object.__new__(LeWM_OFT)
        torch.nn.Module.__init__(framework)
        framework.use_dense_patch_action = False
        framework.dense_patch_freeze_base = True

        cases = (
            ("off", True, True),
            ("teacher", False, False),
            ("student", False, False),
            ("joint", True, False),
            ("joint", False, True),
            ("combined", True, False),
            ("combined", False, True),
        )
        for mode, freeze_base, expected in cases:
            with self.subTest(mode=mode, freeze_base=freeze_base):
                framework.transition_mode = mode
                framework.transition_joint_freeze_base = freeze_base
                self.assertEqual(
                    LeWM_OFT._should_update_latent_stats(framework),
                    expected,
                )

        framework.transition_mode = "off"
        framework.use_dense_patch_action = True
        self.assertFalse(LeWM_OFT._should_update_latent_stats(framework))

    def test_teacher_validates_future_shape(self):
        module = self._make_module()
        with self.assertRaisesRegex(ValueError, "future_delta"):
            module.teacher_forward(
                torch.randn(2, 6, 12),
                torch.randn(2, 1, 6, 12),
            )


if __name__ == "__main__":
    unittest.main()
