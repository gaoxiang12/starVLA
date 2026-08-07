"""Deterministic future-latent prediction for spatial visual tokens."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class SIGReg(nn.Module):
    """Regularize projected features toward an isotropic unit Gaussian."""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = int(num_proj)
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        projected = projected.float()
        directions = torch.randn(
            projected.size(-1), self.num_proj, device=projected.device
        )
        directions = directions.div_(directions.norm(p=2, dim=0))
        projected_t = (projected @ directions).unsqueeze(-1) * self.t
        error = (
            projected_t.cos().mean(-3) - self.phi
        ).square() + projected_t.sin().mean(-3).square()
        statistic = (error @ self.weights) * projected.size(-2)
        return statistic.mean()


class _TokenResidualBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(hidden)
        hidden = hidden + self.attn(
            normalized, normalized, normalized, need_weights=False
        )[0]
        return hidden + self.mlp(self.norm2(hidden))


class TokenResidualPredictor(nn.Module):
    """Predict future visual-token residuals from current tokens and task."""

    def __init__(
        self,
        *,
        latent_dim: int,
        goal_dim: Optional[int],
        n_future: int,
        num_tokens: int,
        context_len: int = 1,
        state_dim: int = 0,
        dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.n_future = int(n_future)
        self.num_tokens = int(num_tokens)
        self.context_len = int(context_len)
        self.state_dim = int(state_dim)
        if self.context_len < 1:
            raise ValueError("context_len must be at least 1")
        self.anchor_proj = nn.Linear(latent_dim, dim)
        self.goal_proj = nn.Linear(goal_dim, dim) if goal_dim else None
        self.state_proj = nn.Linear(self.state_dim, dim) if self.state_dim > 0 else None
        self.history_proj = (
            nn.Linear(latent_dim, dim) if self.context_len > 1 else None
        )
        if self.state_proj is not None:
            nn.init.zeros_(self.state_proj.weight)
            nn.init.zeros_(self.state_proj.bias)
        if self.history_proj is not None:
            nn.init.zeros_(self.history_proj.weight)
            nn.init.zeros_(self.history_proj.bias)
        self.future_query = nn.Parameter(
            torch.randn(1, self.n_future, 1, dim) * 0.02
        )
        self.frame_embedding = nn.Parameter(
            torch.randn(1, 1 + self.n_future, 1, dim) * 0.02
        )
        self.token_embedding = nn.Parameter(
            torch.randn(1, 1, self.num_tokens, dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [_TokenResidualBlock(dim, num_heads, ffn_dim) for _ in range(int(depth))]
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        context: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, context_len, num_tokens, _ = context.shape
        if context_len != self.context_len:
            raise ValueError(
                f"expected {self.context_len} context frames, got {context_len}"
            )
        if num_tokens != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} context tokens, got {num_tokens}")

        current_tokens = self.anchor_proj(context[:, -1:])
        future_tokens = self.future_query.expand(batch_size, -1, num_tokens, -1)
        if self.history_proj is not None:
            # A zero-initialized motion adapter preserves a one-frame
            # checkpoint exactly at initialization, then learns how the most
            # recent visual-token displacement should alter future queries.
            history_delta = context[:, -1] - context[:, -2]
            future_tokens = future_tokens + self.history_proj(history_delta).unsqueeze(1)
        current_and_future = torch.cat([current_tokens, future_tokens], dim=1)
        current_and_future = (
            current_and_future + self.frame_embedding + self.token_embedding
        )
        hidden = current_and_future
        if goal is not None and self.goal_proj is not None:
            hidden = hidden + self.goal_proj(goal.to(hidden.dtype)).view(
                batch_size, 1, 1, -1
            )
        if self.state_proj is not None:
            if state is None:
                raise ValueError("state-conditioned latent predictor requires current state")
            if state.ndim != 2 or state.shape != (batch_size, self.state_dim):
                raise ValueError(
                    "expected state shape "
                    f"({batch_size}, {self.state_dim}), got {tuple(state.shape)}"
                )
            hidden = hidden + self.state_proj(state.to(hidden.dtype)).view(
                batch_size, 1, 1, -1
            )

        hidden = hidden.reshape(batch_size, (1 + self.n_future) * num_tokens, -1)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden).view(
            batch_size, 1 + self.n_future, num_tokens, -1
        )
        return self.out(hidden[:, 1:])


class ContextResidualCorrection(nn.Module):
    """Boost a frozen one-frame predictor with causal history and state motion.

    The base predictor remains the deterministic warm-start model.  This branch
    sees a longer visual history, the base future prediction, and (optionally)
    proprioceptive history, then predicts only a residual correction.  Its
    zero-initialized output makes enabling the branch an exact checkpoint-safe
    no-op before optimization.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        goal_dim: Optional[int],
        n_future: int,
        num_tokens: int,
        context_len: int,
        state_dim: int = 0,
        dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.n_future = int(n_future)
        self.num_tokens = int(num_tokens)
        self.context_len = int(context_len)
        self.state_dim = int(state_dim)
        if self.context_len < 2:
            raise ValueError("context residual correction requires context_len >= 2")

        self.context_proj = nn.Linear(latent_dim, dim)
        self.prediction_proj = nn.Linear(latent_dim, dim)
        self.goal_proj = nn.Linear(goal_dim, dim) if goal_dim else None
        self.state_history_proj = (
            nn.Linear(self.context_len * self.state_dim, dim)
            if self.state_dim > 0
            else None
        )
        self.frame_embedding = nn.Parameter(
            torch.randn(1, self.context_len + self.n_future, 1, dim) * 0.02
        )
        self.token_embedding = nn.Parameter(
            torch.randn(1, 1, self.num_tokens, dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [_TokenResidualBlock(dim, num_heads, ffn_dim) for _ in range(int(depth))]
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _state_motion_features(
        self, state: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        if state.ndim == 2:
            if state.shape != (batch_size, self.state_dim):
                raise ValueError(
                    f"expected state shape ({batch_size}, {self.state_dim}), "
                    f"got {tuple(state.shape)}"
                )
            state_history = state.unsqueeze(1).expand(-1, self.context_len, -1)
        elif state.ndim == 3:
            if state.shape[0] != batch_size or state.shape[2] != self.state_dim:
                raise ValueError(
                    "expected state history shape "
                    f"({batch_size}, T, {self.state_dim}), got {tuple(state.shape)}"
                )
            state_history = state
            if state_history.shape[1] < self.context_len:
                padding = state_history[:, :1].expand(
                    -1, self.context_len - state_history.shape[1], -1
                )
                state_history = torch.cat([padding, state_history], dim=1)
            state_history = state_history[:, -self.context_len :]
        else:
            raise ValueError(
                "state must have shape (B, D) or (B, T, D), "
                f"got {tuple(state.shape)}"
            )

        # Current pose plus finite differences retains absolute configuration
        # and exposes velocity without using action labels.
        state_motion = torch.cat(
            [
                state_history[:, -1],
                (state_history[:, 1:] - state_history[:, :-1]).flatten(1),
            ],
            dim=-1,
        )
        return state_motion

    def forward(
        self,
        context: torch.Tensor,
        base_prediction: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, context_len, num_tokens, _ = context.shape
        if context_len != self.context_len:
            raise ValueError(
                f"expected {self.context_len} context frames, got {context_len}"
            )
        if num_tokens != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} context tokens, got {num_tokens}")
        expected_prediction_shape = (batch_size, self.n_future, self.num_tokens)
        if base_prediction.shape[:3] != expected_prediction_shape:
            raise ValueError(
                "expected base prediction prefix "
                f"{expected_prediction_shape}, got {tuple(base_prediction.shape[:3])}"
            )

        # Represent the causal history as current appearance followed by all
        # consecutive visual motions. This keeps the token count fixed while
        # making more than the final displacement available to the adapter.
        context_motion = torch.cat(
            [context[:, -1:], context[:, 1:] - context[:, :-1]], dim=1
        )
        hidden = torch.cat(
            [
                self.context_proj(context_motion),
                self.prediction_proj(base_prediction),
            ],
            dim=1,
        )
        hidden = hidden + self.frame_embedding + self.token_embedding
        if goal is not None and self.goal_proj is not None:
            hidden = hidden + self.goal_proj(goal.to(hidden.dtype)).view(
                batch_size, 1, 1, -1
            )
        if self.state_history_proj is not None:
            if state is None:
                raise ValueError(
                    "state-conditioned context correction requires state history"
                )
            state_motion = self._state_motion_features(
                state.to(hidden.dtype), batch_size
            )
            hidden = hidden + self.state_history_proj(state_motion).view(
                batch_size, 1, 1, -1
            )

        hidden = hidden.reshape(
            batch_size, (self.context_len + self.n_future) * num_tokens, -1
        )
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden).view(
            batch_size, self.context_len + self.n_future, num_tokens, -1
        )
        return self.out(hidden[:, -self.n_future :])


class VisualTokenLatentWorldModel(nn.Module):
    """Predict all future visual-token latents in one deterministic pass."""

    def __init__(
        self,
        *,
        latent_dim: int,
        goal_dim: Optional[int],
        n_future: int,
        num_tokens: int,
        context_len: int = 1,
        state_dim: int = 0,
        dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 1024,
        context_correction_depth: int = 0,
        context_correction_state_dim: int = 0,
        context_correction_dim: int = 384,
        context_correction_heads: int = 6,
        context_correction_ffn_dim: int = 1024,
        sigreg_weight: float = 0.0,
        stats_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.n_future = int(n_future)
        self.num_tokens = int(num_tokens)
        self.context_len = int(context_len)
        self.stats_momentum = float(stats_momentum)
        self._stats_eps = 1e-4
        self.residual_predictor = TokenResidualPredictor(
            latent_dim=latent_dim,
            goal_dim=goal_dim,
            n_future=self.n_future,
            num_tokens=self.num_tokens,
            context_len=self.context_len,
            state_dim=state_dim,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
        )
        self.context_correction = (
            ContextResidualCorrection(
                latent_dim=latent_dim,
                goal_dim=goal_dim,
                n_future=self.n_future,
                num_tokens=self.num_tokens,
                context_len=self.context_len,
                state_dim=context_correction_state_dim,
                dim=context_correction_dim,
                depth=context_correction_depth,
                num_heads=context_correction_heads,
                ffn_dim=context_correction_ffn_dim,
            )
            if context_correction_depth > 0
            else None
        )
        self.sigreg = SIGReg() if sigreg_weight > 0 else None
        self.register_buffer("delta_scale", torch.ones(1))
        self.register_buffer("_delta_scale_ready", torch.zeros(1))

    def _apply(self, *args, **kwargs):
        # DeepSpeed/Accelerate bf16 training casts module buffers to bfloat16.
        # The per-step EMA increment ((1 - momentum) * rms) is smaller than the
        # bf16 quantization step near 1.0, so an in-place bf16 EMA underflows and
        # freezes the normalizer once it reaches ~1.0. That makes ``latent_loss``
        # (delta_pred_mse / delta_scale**2) an unreliable progress signal while
        # the encoder-drifting residual scale keeps growing. Keep the running
        # delta-scale statistics in float32 regardless of module-wide casting.
        # Save the fp32 values before ``super()._apply`` can quantize them to
        # bf16. Merely casting the already-quantized result back to fp32 loses
        # the checkpoint's exact normalizer (e.g. 1.6803247 -> 1.6796875).
        delta_scale_fp32 = self.delta_scale.detach().float().clone()
        delta_scale_ready_fp32 = self._delta_scale_ready.detach().float().clone()
        module = super()._apply(*args, **kwargs)
        target_device = module.delta_scale.device
        module.delta_scale = delta_scale_fp32.to(device=target_device)
        module._delta_scale_ready = delta_scale_ready_fp32.to(device=target_device)
        return module

    @torch.no_grad()
    def _update_delta_scale(self, residual: torch.Tensor) -> None:
        rms = residual.float().square().mean().clamp_min(self._stats_eps).sqrt()
        if float(self._delta_scale_ready) < 1.0:
            self.delta_scale.fill_(float(rms))
            self._delta_scale_ready.fill_(1.0)
        else:
            self.delta_scale.mul_(self.stats_momentum).add_(
                rms, alpha=1 - self.stats_momentum
            )

    def regress_future(
        self,
        context: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context.shape[1] != self.context_len:
            raise ValueError(
                f"expected {self.context_len} context frames, got {context.shape[1]}"
            )
        anchor = context[:, -1:]
        predicted_delta = self.residual_predictor(context, goal=goal, state=state)
        if self.context_correction is not None:
            predicted_delta = predicted_delta + self.context_correction(
                context, predicted_delta, goal=goal, state=state
            )
        scale = self.delta_scale.clamp_min(self._stats_eps)
        return anchor + predicted_delta * scale

    def rollout_future(
        self,
        context: torch.Tensor,
        *,
        steps: int,
        goal: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extend the single-shot predictor by re-anchoring on its own output.

        Each step emits ``n_future`` latents at the training cadence, so the
        returned tensor covers ``steps * n_future`` horizons. ``state`` is the
        current proprio observation and is deliberately held fixed: no future
        proprio exists at deployment, so a rollout beyond the first step is
        conditioned on a stale pose.
        """
        if int(steps) < 1:
            raise ValueError(f"rollout steps must be at least 1, got {steps}")
        predictions = []
        window = context
        for _ in range(int(steps)):
            step_prediction = self.regress_future(window, goal=goal, state=state)
            predictions.append(step_prediction)
            window = torch.cat([window, step_prediction], dim=1)[
                :, -self.context_len :
            ]
        return torch.cat(predictions, dim=1)

    def _rollout_losses(
        self,
        latent: torch.Tensor,
        *,
        ctx_len: int,
        rollout_steps: int,
        scale: torch.Tensor,
        goal: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Supervise steps 2..K of a rollout that re-anchors on predictions.

        Step 1 is left to the ordinary teacher-forced objective so the headline
        ``latent_loss`` stays comparable with every single-shot checkpoint.
        """
        current = latent[:, ctx_len - 1 : ctx_len]
        window = latent[:, :ctx_len]
        losses: dict[str, torch.Tensor] = {}
        total = latent.new_zeros(())
        for step in range(int(rollout_steps)):
            predicted_delta = self.residual_predictor(window, goal=goal, state=state)
            if self.context_correction is not None:
                predicted_delta = predicted_delta + self.context_correction(
                    window, predicted_delta, goal=goal, state=state
                )
            anchor = window[:, -1:]
            predicted_future = anchor + predicted_delta * scale
            start = ctx_len + step * self.n_future
            true_future = latent[:, start : start + self.n_future]
            if step > 0:
                # Ground the residual on the *predicted* anchor: this is exactly
                # the quantity the network must emit when it is run on its own
                # output at deployment.
                target_delta = (true_future - anchor).detach() / scale
                step_loss = (
                    predicted_delta.float() - target_delta.float()
                ).square().mean()
                total = total + step_loss
                losses[f"rollout_latent_loss_step_{step + 1}"] = step_loss
            with torch.no_grad():
                error = (predicted_future.float() - true_future.float()).square().mean()
                copy = (current.float() - true_future.float()).square().mean()
                losses[f"rollout_to_copy_ratio_step_{step + 1}"] = error / copy.clamp_min(
                    1e-8
                )
                losses[f"rollout_direction_cosine_step_{step + 1}"] = (
                    F.cosine_similarity(
                        (predicted_future - current).float().flatten(2),
                        (true_future - current).float().flatten(2),
                        dim=-1,
                        eps=1e-8,
                    ).mean()
                )
            window = torch.cat([window, predicted_future], dim=1)[
                :, -self.context_len :
            ]
        losses["rollout_latent_loss"] = total / max(int(rollout_steps) - 1, 1)
        return losses

    def forward(
        self,
        latent: torch.Tensor,
        *,
        ctx_len: int,
        goal: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        update_stats: bool = True,
        rollout_steps: int = 1,
    ) -> dict[str, torch.Tensor]:
        batch_size, total_frames, num_tokens = latent.shape[:3]
        if num_tokens != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} visual tokens, got {num_tokens}")
        if ctx_len != self.context_len:
            raise ValueError(f"expected ctx_len={self.context_len}, got {ctx_len}")
        if total_frames - ctx_len < self.n_future:
            raise ValueError(
                f"need {self.n_future} future frames, got {total_frames - ctx_len}"
            )
        rollout_steps = int(rollout_steps)
        if rollout_steps < 1:
            raise ValueError(f"rollout_steps must be at least 1, got {rollout_steps}")
        if total_frames - ctx_len < self.n_future * rollout_steps:
            raise ValueError(
                f"rollout_steps={rollout_steps} needs "
                f"{self.n_future * rollout_steps} future frames, "
                f"got {total_frames - ctx_len}"
            )
        anchor = latent[:, ctx_len - 1 : ctx_len]
        future = latent[:, ctx_len : ctx_len + self.n_future]
        residual = (future - anchor).detach()
        if self.training and update_stats:
            self._update_delta_scale(residual)

        scale = self.delta_scale.clamp_min(self._stats_eps)
        target_delta = residual / scale
        base_predicted_delta = self.residual_predictor(
            latent[:, :ctx_len], goal=goal, state=state
        )
        correction = None
        predicted_delta = base_predicted_delta
        if self.context_correction is not None:
            correction = self.context_correction(
                latent[:, :ctx_len], base_predicted_delta, goal=goal, state=state
            )
            predicted_delta = base_predicted_delta + correction
        predicted_residual = predicted_delta * scale
        predicted_future = anchor + predicted_residual
        squared_error = (predicted_delta.float() - target_delta.float()).square()
        latent_loss = squared_error.mean()
        latent_cosine_loss = 1.0 - F.cosine_similarity(
            predicted_delta.float().flatten(2),
            target_delta.float().flatten(2),
            dim=-1,
            eps=1e-8,
        ).mean()
        output = {
            "latent_loss": latent_loss,
            "latent_cosine_loss": latent_cosine_loss,
            "pred_future_latent": predicted_future,
        }
        if correction is not None:
            with torch.no_grad():
                output["latent_base_loss"] = (
                    base_predicted_delta.float() - target_delta.float()
                ).square().mean()
                output["context_correction_rms"] = correction.float().square().mean().sqrt()
                output["context_correction_to_base_ratio"] = (
                    correction.float().square().mean().sqrt()
                    / base_predicted_delta.float().square().mean().sqrt().clamp_min(1e-8)
                )
        per_horizon_loss = squared_error.mean(dim=(0, 2, 3))
        for horizon_index, horizon_loss in enumerate(per_horizon_loss, start=1):
            output[f"latent_loss_horizon_{horizon_index}"] = horizon_loss

        with torch.no_grad():
            copy_mse = residual.float().square().mean()
            pred_mse = (predicted_residual.float() - residual.float()).square().mean()
            mean_residual = residual.float().mean(dim=0, keepdim=True)
            mean_baseline_mse = (residual.float() - mean_residual).square().mean()
            direction_cosine = F.cosine_similarity(
                predicted_residual.float().flatten(2),
                residual.float().flatten(2),
                dim=-1,
                eps=1e-8,
            ).mean()
            output.update(
                {
                    "delta_scale": scale.detach().mean(),
                    "delta_target_rms": residual.float().square().mean().sqrt(),
                    "delta_pred_rms": predicted_residual.float().square().mean().sqrt(),
                    "delta_copy_mse": copy_mse,
                    "delta_pred_mse": pred_mse,
                    "delta_mean_baseline_mse": mean_baseline_mse,
                    "delta_to_copy_ratio": pred_mse / copy_mse.clamp_min(1e-8),
                    "delta_direction_cosine": direction_cosine,
                }
            )

        if self.sigreg is not None:
            future_frames, channels = predicted_delta.shape[1], predicted_delta.shape[-1]
            sigreg_input = predicted_delta.permute(1, 0, 2, 3).reshape(
                future_frames, batch_size * num_tokens, channels
            )
            output["sigreg_loss"] = self.sigreg(sigreg_input)
        if rollout_steps > 1:
            output.update(
                self._rollout_losses(
                    latent,
                    ctx_len=ctx_len,
                    rollout_steps=rollout_steps,
                    scale=scale,
                    goal=goal,
                    state=state,
                )
            )
        return output
