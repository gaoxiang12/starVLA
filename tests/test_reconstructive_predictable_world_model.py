import unittest

import torch

from starVLA.model.modules.world_model.reconstructive_predictable_world_model import (
    ReconstructivePredictableWorldModel,
    smooth_targets,
)


def _build(**overrides):
    kwargs = dict(
        token_dim=24,
        num_tokens=8,
        latent_dim=16,
        goal_dim=10,
        n_future=2,
        context_len=1,
        state_dim=6,
        hidden_dim=32,
        encoder_depth=1,
        decoder_depth=1,
        predictor_depth=2,
        num_heads=4,
        ffn_dim=48,
        target_smooth_width=3,
    )
    kwargs.update(overrides)
    return ReconstructivePredictableWorldModel(**kwargs)


class SmoothTargetsTest(unittest.TestCase):
    def test_width_one_is_identity(self):
        tokens = torch.randn(2, 5, 8, 24)
        self.assertTrue(torch.equal(smooth_targets(tokens, 1), tokens))

    def test_even_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be odd"):
            smooth_targets(torch.randn(1, 5, 2, 3), 2)

    def test_interior_frame_is_the_local_mean(self):
        tokens = torch.randn(2, 5, 8, 24)
        smoothed = smooth_targets(tokens, 3)
        self.assertEqual(smoothed.shape, tokens.shape)
        self.assertTrue(
            torch.allclose(smoothed[:, 2], tokens[:, 1:4].mean(dim=1), atol=1e-6)
        )

    def test_edge_frames_use_replicate_padding(self):
        tokens = torch.randn(1, 4, 2, 3)
        smoothed = smooth_targets(tokens, 3)
        expected = (tokens[:, 0] * 2 + tokens[:, 1]) / 3
        self.assertTrue(torch.allclose(smoothed[:, 0], expected, atol=1e-6))

    def test_smoothing_suppresses_white_noise_far_more_than_signal(self):
        torch.manual_seed(0)
        steps = torch.arange(64, dtype=torch.float32).reshape(1, 64, 1, 1)
        signal = (steps * 0.05).expand(4, 64, 8, 24).clone()
        noisy = signal + torch.randn_like(signal)

        raw_delta = (noisy[:, 1:] - noisy[:, :-1]).square().mean()
        smoothed = smooth_targets(noisy, 5)
        smooth_delta = (smoothed[:, 1:] - smoothed[:, :-1]).square().mean()
        signal_delta = (signal[:, 1:] - signal[:, :-1]).square().mean()

        # The incoherent part collapses; the ramp survives essentially intact.
        self.assertLess(float(smooth_delta), float(raw_delta) * 0.2)
        self.assertGreater(float(smooth_delta), float(signal_delta) * 0.5)


class ReconstructivePredictableWorldModelTest(unittest.TestCase):
    def test_forward_shapes_and_losses(self):
        model = _build()
        model.train()
        tokens = torch.randn(3, 4, 8, 24)
        goal = torch.randn(3, 10)
        state = torch.randn(3, 4, 6)

        out = model(tokens, goal=goal, state=state)

        self.assertEqual(out["latent"].shape, (3, 4, 16))
        self.assertEqual(out["pred_future_latent"].shape, (3, 2, 16))
        for key in ("recon_loss", "latent_loss", "state_loss", "variance_loss"):
            self.assertTrue(torch.isfinite(out[key]))

    def test_predictor_starts_as_exact_copy_baseline(self):
        model = _build()
        tokens = torch.randn(2, 4, 8, 24)
        goal = torch.randn(2, 10)

        out = model(tokens, goal=goal, state=torch.randn(2, 4, 6))

        anchor = out["latent"][:, :1].expand(-1, 2, -1)
        self.assertTrue(torch.allclose(out["pred_future_latent"], anchor, atol=1e-6))
        self.assertAlmostEqual(float(out["latent_to_copy_ratio"]), 1.0, places=4)

    def test_prediction_loss_shapes_the_encoder_but_not_via_the_target(self):
        model = _build()
        torch.nn.init.normal_(model.predictor[-1].weight, std=0.05)
        tokens = torch.randn(2, 4, 8, 24)
        out = model(tokens, goal=torch.randn(2, 10), state=torch.randn(2, 4, 6))

        out["latent_loss"].backward()

        # Requirement (2) is a property of the latent, so this loss must reach
        # the encoder -- but only through the predictor's input, never through
        # the detached target.
        self.assertGreater(float(model.token_in.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.predictor[0].weight.grad.abs().sum()), 0.0)
        self.assertIsNone(model.token_out.weight.grad)

    def test_prediction_loss_is_invariant_to_shrinking_temporal_motion(self):
        model = _build()
        torch.nn.init.normal_(model.predictor[-1].weight, std=0.05)
        model.eval()
        latent = torch.randn(6, 3, 16)
        goal = torch.randn(6, 10)

        def ratio(sequence: torch.Tensor, scale: float) -> float:
            model.delta_scale.fill_(scale)
            model._delta_scale_ready.fill_(1.0)
            context = sequence[:, :1]
            residual = sequence[:, 1:3] - context
            predicted = model.predict_latent(context, goal=goal)
            target = residual / model.delta_scale
            got = (predicted - context) / model.delta_scale
            return float((got - target).square().mean())

        # Halving every temporal increment also halves delta_scale, so a latent
        # that freezes in time gains nothing.
        anchor = latent[:, :1]
        shrunk = anchor + (latent - anchor) * 0.5
        self.assertAlmostEqual(ratio(latent, 1.0), ratio(shrunk, 0.5), places=4)

    def test_temporal_variance_floor_penalises_a_time_frozen_latent(self):
        model = _build()
        model.train()
        tokens = torch.randn(4, 4, 8, 24)
        frozen = tokens[:, :1].expand(-1, 4, -1, -1).contiguous()

        moving = model(tokens, goal=torch.randn(4, 10), state=torch.randn(4, 4, 6))
        still = model(frozen, goal=torch.randn(4, 10), state=torch.randn(4, 4, 6))

        self.assertAlmostEqual(float(still["latent_temporal_std_ratio"]), 0.0, places=5)
        self.assertAlmostEqual(
            float(still["temporal_variance_loss"]), model.temporal_floor, places=5
        )
        self.assertLess(
            float(moving["temporal_variance_loss"]),
            float(still["temporal_variance_loss"]),
        )

    def test_reconstruction_and_state_losses_train_the_encoder(self):
        model = _build()
        tokens = torch.randn(2, 4, 8, 24)
        out = model(tokens, goal=torch.randn(2, 10), state=torch.randn(2, 4, 6))

        (out["recon_loss"] + out["state_loss"]).backward()

        self.assertGreater(float(model.token_in.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.to_latent.weight.grad.abs().sum()), 0.0)

    def test_reconstruction_target_is_smoothed_not_raw(self):
        model = _build(target_smooth_width=3)
        tokens = torch.randn(2, 6, 8, 24)
        out = model(tokens, goal=torch.randn(2, 10), state=torch.randn(2, 6, 6))

        smoothed = smooth_targets(tokens, 3)
        decoded = model.decode(out["latent"])
        self.assertTrue(
            torch.allclose(
                out["recon_loss"], (decoded - smoothed).square().mean(), atol=1e-6
            )
        )
        self.assertFalse(
            torch.allclose(
                out["recon_loss"], (decoded - tokens).square().mean(), atol=1e-4
            )
        )

    def test_rollout_first_step_matches_single_prediction(self):
        model = _build()
        torch.nn.init.normal_(model.predictor[-1].weight, std=0.05)
        context = torch.randn(2, 1, 16)
        goal = torch.randn(2, 10)

        single = model.predict_latent(context, goal=goal)
        rolled = model.rollout_latent(context, steps=3, goal=goal)

        self.assertEqual(rolled.shape, (2, 6, 16))
        self.assertTrue(torch.equal(rolled[:, :2], single))
        with self.assertRaisesRegex(ValueError, "at least 1"):
            model.rollout_latent(context, steps=0, goal=goal)

    def test_delta_scale_stays_fp32_under_bfloat16_cast(self):
        model = _build()
        expected = torch.tensor(1.6803246, dtype=torch.float32)
        model.delta_scale.copy_(expected)
        model._delta_scale_ready.fill_(1.0)

        model.to(dtype=torch.bfloat16)

        self.assertEqual(model.delta_scale.dtype, torch.float32)
        self.assertTrue(torch.equal(model.delta_scale.cpu(), expected.reshape(())))

    def test_variance_floor_penalises_a_collapsed_latent(self):
        model = _build()
        healthy = torch.randn(64, 16) * 3.0
        collapsed = torch.zeros(64, 16)
        floor = model.variance_floor

        self.assertAlmostEqual(
            float(torch.relu(floor - healthy.std(dim=0)).mean()), 0.0, places=5
        )
        self.assertAlmostEqual(
            float(torch.relu(floor - collapsed.std(dim=0)).mean()), floor, places=5
        )

    def test_missing_state_is_rejected_when_head_is_enabled(self):
        model = _build()
        with self.assertRaisesRegex(ValueError, "requires a state tensor"):
            model(torch.randn(2, 4, 8, 24), goal=torch.randn(2, 10))

    def test_too_few_frames_is_rejected(self):
        model = _build()
        with self.assertRaisesRegex(ValueError, "need at least 3 frames"):
            model(torch.randn(2, 2, 8, 24), goal=torch.randn(2, 10))


if __name__ == "__main__":
    unittest.main()
