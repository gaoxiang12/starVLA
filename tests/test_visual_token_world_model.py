import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from starVLA.model.framework.WM4A.LeWMOFT import (
    CompositionalTextEncoder,
    LeWM_OFT,
    VisualActionCrossAttn,
    VisualTokenPooler,
    prefix_l1_loss,
)
from starVLA.model.modules.action_model.ACT_ActionHeader import (
    TurboStyleACTActionHead,
)
from starVLA.model.modules.world_model.visual_token_delta_world_model import (
    VisualTokenLatentWorldModel,
)
from starVLA.model.modules.world_model.wala_transition_auxiliary import (
    WALAVisualTransitionAuxiliary,
    token_cosine_loss,
)


class VisualTokenComponentsTest(unittest.TestCase):
    def test_multi_embodiment_act_heads_route_native_shapes(self):
        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=32)
                self.anchor = nn.Parameter(torch.zeros(()))

        repo_root = Path(__file__).resolve().parents[1]
        config = OmegaConf.load(
            repo_root
            / "examples/UnifiedPretrain/train_files/"
            "starvla_lewm_unified_pretrain.yaml"
        )
        wm = config.framework.world_model
        wm.base_wm = "fake_dinov3.pth"
        wm.train_encoder = False
        wm.visual_token_dim = 24
        wm.visual_tokens_per_view = 4
        wm.residual_predictor_dim = 24
        wm.residual_predictor_depth = 1
        wm.residual_predictor_heads = 4
        wm.residual_predictor_ffn = 48
        action = config.framework.action_model
        action.action_hidden_dim = 32
        action.act_num_heads = 4
        action.act_num_layers = 1
        action.act_dim_feedforward = 64
        action.act_mlp_hidden_dim = 48
        config.framework.lang_cond.embed_dim = 16
        config.framework.lang_cond.text_hidden_dim = 16
        config.framework.lang_cond.text_heads = 4
        config.framework.lang_cond.text_ffn_dim = 32

        with patch(
            "starVLA.model.modules.world_model.dinov3_loader.load_dinov3",
            return_value=(FakeEncoder(), None, 0),
        ):
            model = LeWM_OFT(config=config)

        self.assertEqual(set(model.action_models), {"franka", "oxe_bridge", "aloha"})
        visual = torch.randn(2, 3, 12, 24)
        expected = {
            "franka": (8, 7, 8),
            "oxe_bridge": (3, 7, 8),
            "aloha": (16, 14, 14),
        }
        for tag, (horizon, action_dim, state_dim) in expected.items():
            selected_tag = model._resolve_batch_embodiment(
                [{"robot_tag": tag}, {"robot_tag": tag}]
            )
            head, selected_horizon, selected_state_dim, _ = model._action_runtime(
                selected_tag
            )
            hidden = model._pool_visual_tokens_to_action_queries(
                visual,
                state=torch.randn(2, state_dim),
                action_model=head,
            )
            self.assertEqual(selected_horizon, horizon)
            self.assertEqual(selected_state_dim, state_dim)
            self.assertEqual(head.predict_action(hidden).shape, (2, horizon, action_dim))

        with self.assertRaisesRegex(ValueError, "homogeneous batch"):
            model._resolve_batch_embodiment(
                [{"robot_tag": "franka"}, {"robot_tag": "aloha"}]
            )
        with self.assertRaisesRegex(ValueError, "action_spec_id"):
            model._resolve_batch_embodiment(
                [
                    {
                        "robot_tag": "franka",
                        "action_spec_id": "bridge_eef_delta_7",
                    }
                ]
            )

    def test_lewm_selects_turbo_style_act_from_config(self):
        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=32)
                self.anchor = nn.Parameter(torch.zeros(()))

        repo_root = Path(__file__).resolve().parents[1]
        config = OmegaConf.load(
            repo_root
            / "examples/LIBERO/train_files/starvla_lewm_oft_libero.yaml"
        )
        wm = config.framework.world_model
        wm.base_wm = "fake_dinov3.pth"
        wm.visual_token_dim = 24
        wm.visual_tokens_per_view = 4
        wm.num_visual_tokens = 4
        wm.residual_predictor_dim = 24
        wm.residual_predictor_depth = 1
        wm.residual_predictor_heads = 4
        wm.residual_predictor_ffn = 48
        wm.visual_diagnostics = False
        config.framework.lang_cond.num_buckets = 32
        config.framework.lang_cond.embed_dim = 16
        action = config.framework.action_model
        action.action_hidden_dim = 32
        action.act_num_heads = 4
        action.act_num_layers = 2
        action.act_dim_feedforward = 64
        action.act_mlp_hidden_dim = 48

        with patch(
            "starVLA.model.modules.world_model.dinov3_loader.load_dinov3",
            return_value=(FakeEncoder(), None, 0),
        ):
            model = LeWM_OFT(config=config)

        visual = torch.randn(2, 3, 8, 24, requires_grad=True)
        state = torch.randn(2, 8, requires_grad=True)
        hidden = model._pool_visual_tokens_to_action_queries(visual, state=state)
        actions = model.action_model.predict_action(hidden)

        self.assertEqual(model.action_model_type, "ACT")
        self.assertIsNone(model.visual_action_head)
        self.assertEqual(len(model.action_model.decoder.layers), 2)
        self.assertEqual(hidden.shape, (2, 8, 32))
        self.assertEqual(actions.shape, (2, 8, 7))

        actions.square().mean().backward()
        self.assertTrue(torch.isfinite(visual.grad).all())
        self.assertTrue(torch.isfinite(state.grad).all())

    def test_turbo_style_act_decodes_visual_and_state_memory(self):
        head = TurboStyleACTActionHead(
            token_dim=24,
            hidden_dim=32,
            action_dim=7,
            horizon=8,
            num_frames=3,
            num_visual_tokens=32,
            num_heads=4,
            num_layers=3,
            dim_feedforward=64,
            mlp_hidden_dim=48,
            dropout=0.0,
            state_dim=8,
            state_hidden_dim=16,
            num_state_tokens=2,
        )
        visual = torch.randn(2, 3, 32, 24, requires_grad=True)
        state = torch.randn(2, 8, requires_grad=True)

        memory = head.build_memory(visual, state=state)
        hidden = head.decode_action_queries(visual, state=state)
        actions = head(visual, state=state)

        self.assertEqual(memory.shape, (2, 3 * 32 + 2, 32))
        self.assertEqual(hidden.shape, (2, 8, 32))
        self.assertEqual(actions.shape, (2, 8, 7))
        self.assertEqual(len(head.decoder.layers), 3)
        self.assertTrue(torch.all(actions <= 1.0))
        self.assertTrue(torch.all(actions >= -1.0))

        actions.square().mean().backward()
        self.assertTrue(torch.isfinite(visual.grad).all())
        self.assertTrue(torch.isfinite(state.grad).all())

        with self.assertRaisesRegex(ValueError, "requires current state"):
            head(visual.detach())

    def test_compositional_text_encoder_is_collision_free_and_trainable(self):
        encoder = CompositionalTextEncoder(
            output_dim=24,
            hidden_dim=32,
            depth=1,
            num_heads=4,
            ffn_dim=64,
            max_length=32,
            dropout=0.0,
        )
        token_ids, valid = encoder.tokenize(
            [" Pick  up   the cup ", "pick up the bowl", "拿起杯子"],
            device=torch.device("cpu"),
        )
        normalized_ids, normalized_valid = encoder.tokenize(
            ["pick up the cup"], device=torch.device("cpu")
        )
        self.assertTrue(torch.equal(token_ids[0], normalized_ids[0]))
        self.assertTrue(torch.equal(valid[0], normalized_valid[0]))
        self.assertFalse(torch.equal(token_ids[0], token_ids[1]))
        self.assertGreater(int(valid[2].sum()), 2)

        output = encoder(
            ["pick up the cup", "pick up the bowl"],
            device=torch.device("cpu"),
        )
        self.assertEqual(output.shape, (2, 24))
        output.square().mean().backward()
        self.assertIsNotNone(encoder.token_embedding.weight.grad)

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

    def test_spatial_pooling_masks_padded_views(self):
        pooler = VisualTokenPooler(
            patch_dim=8,
            token_dim=8,
            num_views=3,
            tokens_per_view=4,
        )
        patches = torch.randn(2, 3, 3, 16, 8)
        view_mask = torch.tensor([[True, True, False], [True, True, True]])
        tokens, content = pooler(
            patches, return_content=True, view_valid_mask=view_mask
        )
        self.assertTrue(torch.equal(tokens[0, :, 8:], torch.zeros_like(tokens[0, :, 8:])))
        self.assertTrue(torch.equal(content[0, :, 8:], torch.zeros_like(content[0, :, 8:])))
        recovered = pooler.remove_position(tokens, view_valid_mask=view_mask)
        torch.testing.assert_close(recovered, content)

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

    def test_latent_loss_is_direct_unnormalized_future_mse(self):
        model = VisualTokenLatentWorldModel(
            latent_dim=2,
            goal_dim=None,
            n_future=2,
            num_tokens=1,
            dim=4,
            depth=1,
            num_heads=1,
            ffn_dim=8,
        )
        model.delta_scale.fill_(3.0)
        model._delta_scale_ready.fill_(1.0)
        # The residual predictor is zero-initialized, so the model copies the
        # current latent. Direct future-latent MSE is mean([1^2, 2^2]) = 2.5;
        # the removed normalized-residual objective would instead be 2.5 / 9.
        latent = torch.tensor(
            [
                [
                    [[0.0, 0.0]],
                    [[1.0, 1.0]],
                    [[2.0, 2.0]],
                    [[3.0, 3.0]],
                    [[4.0, 4.0]],
                ]
            ]
        )

        output = model(
            latent, ctx_len=1, update_stats=False, rollout_steps=2
        )

        torch.testing.assert_close(output["latent_loss"], torch.tensor(2.5))
        torch.testing.assert_close(output["delta_pred_mse"], output["latent_loss"])
        torch.testing.assert_close(
            output["latent_loss_horizon_1"], torch.tensor(1.0)
        )
        torch.testing.assert_close(
            output["latent_loss_horizon_2"], torch.tensor(4.0)
        )
        # The second rollout step still predicts a copy of zero, so its direct
        # MSE against future values [3, 4] is mean([3^2, 4^2]) = 12.5.
        torch.testing.assert_close(
            output["rollout_latent_loss_step_2"], torch.tensor(12.5)
        )
        torch.testing.assert_close(
            output["rollout_latent_loss"], torch.tensor(12.5)
        )

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
    def test_teacher_validates_future_shape(self):
        module = self._make_module()
        with self.assertRaisesRegex(ValueError, "future_delta"):
            module.teacher_forward(
                torch.randn(2, 6, 12),
                torch.randn(2, 1, 6, 12),
            )


if __name__ == "__main__":
    unittest.main()
