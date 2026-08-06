"""Predictable, temporally smooth spatial latents over frozen DINOv3 patches.

The trainable representation is intentionally small: fixed 4x4 spatial cells
per camera are first compressed by a shared token projection, then their fixed
camera/grid ordering is flattened and projected to one latent vector per
frame. There is no reconstruction, state decoding, action conditioning, or
residual normalization. A deterministic world model predicts future latent
values directly, while SIGReg and temporal objectives keep the learned
coordinates non-collapsed and locally smooth.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_token_delta_world_model import SIGReg, TokenResidualPredictor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average one scalar per sample, preserving a zero-gradient empty case."""

    if values.ndim != 1 or mask.ndim != 1 or values.shape != mask.shape:
        raise ValueError(
            f"masked mean expects matching vectors, got {values.shape} and {mask.shape}"
        )
    weights = mask.to(device=values.device, dtype=values.dtype)
    denominator = weights.sum()
    if bool(denominator > 0):
        return (values * weights).sum() / denominator
    return values.sum() * 0.0


class GridSpatialLatentProjector(nn.Module):
    """Compress fixed DINO spatial cells into one position-aware frame vector."""

    def __init__(
        self,
        *,
        patch_dim: int,
        spatial_token_dim: int,
        latent_dim: int,
        num_views: int,
        grid_size: int = 4,
    ) -> None:
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.spatial_token_dim = int(spatial_token_dim)
        self.latent_dim = int(latent_dim)
        self.num_views = int(num_views)
        self.grid_size = int(grid_size)
        if min(
            self.patch_dim,
            self.spatial_token_dim,
            self.latent_dim,
            self.num_views,
            self.grid_size,
        ) < 1:
            raise ValueError("projector dimensions must be positive")
        if self.spatial_token_dim > self.patch_dim:
            raise ValueError(
                "the first spatial projection must be a bottleneck; expected "
                "spatial_token_dim <= patch_dim, got "
                f"{self.spatial_token_dim} > {self.patch_dim}"
            )

        self.tokens_per_view = self.grid_size**2
        self.spatial_token_count = self.num_views * self.tokens_per_view
        self.num_tokens = 1
        self.input_norm = nn.LayerNorm(self.patch_dim)
        # Stage 1 is shared across every camera/grid position. Stage 2 sees the
        # fixed flattened order, so it can retain position-specific information
        # without leaving an explicit 32-token axis in z_t.
        self.token_projection = nn.Linear(
            self.patch_dim, self.spatial_token_dim, bias=False
        )
        flattened_dim = self.spatial_token_count * self.spatial_token_dim
        self.global_norm = nn.LayerNorm(flattened_dim)
        self.global_projection = nn.Linear(
            flattened_dim, self.latent_dim, bias=False
        )
        nn.init.orthogonal_(self.token_projection.weight)
        nn.init.orthogonal_(self.global_projection.weight)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 5:
            raise ValueError(
                "expected DINO patches (B,T,V,N,D), got "
                f"{tuple(patches.shape)}"
            )
        batch, frames, views, patch_count, channels = patches.shape
        if views != self.num_views or channels != self.patch_dim:
            raise ValueError(
                "DINO patch suffix mismatch: expected "
                f"V={self.num_views}, D={self.patch_dim}, got V={views}, D={channels}"
            )
        patch_grid = math.isqrt(patch_count)
        if patch_grid * patch_grid != patch_count:
            raise ValueError(f"DINO patch count must form a square grid, got {patch_count}")

        # The DINO encoder is a fixed teacher.  Detaching here makes that
        # invariant local to this branch even if an outer config is wrong.
        spatial = patches.detach().float().reshape(
            batch * frames * views, patch_grid, patch_grid, channels
        )
        spatial = spatial.permute(0, 3, 1, 2)
        if patch_grid != self.grid_size:
            spatial = F.adaptive_avg_pool2d(
                spatial, (self.grid_size, self.grid_size)
            )
        spatial = spatial.permute(0, 2, 3, 1).reshape(
            batch, frames, self.spatial_token_count, channels
        )
        spatial = self.token_projection(self.input_norm(spatial))
        flattened = spatial.reshape(batch, frames, -1)
        return self.global_projection(self.global_norm(flattened))


class SmoothSpatialLatentWorldModel(nn.Module):
    """Direct future-latent prediction with SIGReg and temporal smoothness.

    Step 1 is teacher-forced: the predictor sees the true current latent and
    emits ``n_future`` consecutive future latents, keeping the headline loss
    comparable with every single-shot checkpoint.  When ``rollout_steps > 1``,
    steps 2..K additionally re-anchor the predictor on its *own* output and
    supervise the residual against the true future, so the model learns to stay
    stable when the loop is closed at deployment.  The model consumes
    ``1 + n_future * rollout_steps`` consecutive frames plus one same-episode
    far frame used only by the temporal-order ranking.
    """

    def __init__(
        self,
        *,
        patch_dim: int,
        num_views: int,
        goal_dim: Optional[int],
        latent_dim: int = 384,
        spatial_token_dim: int = 128,
        grid_size: int = 4,
        n_future: int = 2,
        rollout_steps: int = 1,
        rollout_weight: float = 0.0,
        predictor_dim: int = 384,
        predictor_depth: int = 4,
        predictor_heads: int = 6,
        predictor_ffn_dim: int = 1024,
        prediction_weight: float = 1.0,
        sigreg_weight: float = 0.02,
        slow_weight: float = 0.05,
        acceleration_weight: float = 0.10,
        temporal_order_weight: float = 0.05,
        temporal_order_margin: float = 0.10,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
    ) -> None:
        super().__init__()
        self.n_future = int(n_future)
        self.rollout_steps = int(rollout_steps)
        self.rollout_weight = float(rollout_weight)
        if self.n_future < 1:
            raise ValueError(
                f"n_future must be at least 1, got {self.n_future}"
            )
        if self.rollout_steps < 1:
            raise ValueError(
                f"rollout_steps must be at least 1, got {self.rollout_steps}"
            )
        if self.rollout_steps == 1 and self.rollout_weight > 0:
            raise ValueError("rollout_weight requires rollout_steps >= 2")
        # t, t+1, ..., t+n_future*rollout_steps, then one same-episode far frame.
        self.required_frames = 1 + self.n_future * self.rollout_steps + 1
        self.prediction_weight = float(prediction_weight)
        self.sigreg_weight = float(sigreg_weight)
        self.slow_weight = float(slow_weight)
        self.acceleration_weight = float(acceleration_weight)
        self.temporal_order_weight = float(temporal_order_weight)
        self.temporal_order_margin = float(temporal_order_margin)
        weights = (
            self.prediction_weight,
            self.rollout_weight,
            self.sigreg_weight,
            self.slow_weight,
            self.acceleration_weight,
            self.temporal_order_weight,
        )
        if min(weights) < 0:
            raise ValueError("smooth latent loss weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one smooth latent loss weight must be positive")
        if self.temporal_order_margin < 0:
            raise ValueError("temporal_order_margin must be non-negative")

        self.projector = GridSpatialLatentProjector(
            patch_dim=patch_dim,
            spatial_token_dim=spatial_token_dim,
            latent_dim=latent_dim,
            num_views=num_views,
            grid_size=grid_size,
        )
        self.predictor = TokenResidualPredictor(
            latent_dim=latent_dim,
            goal_dim=goal_dim,
            n_future=self.n_future,
            num_tokens=self.projector.num_tokens,
            context_len=1,
            state_dim=0,
            dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            ffn_dim=predictor_ffn_dim,
        )
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    def _sigreg_loss(
        self, latent: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        # A frame now contributes one vector rather than 32 token samples.
        # Pool all valid time groups in the micro-batch so SIGReg has the
        # largest honest sample set available without adding a stateful queue.
        valid_latent = latent[valid_mask]
        if valid_latent.shape[0] == 0:
            return latent.sum() * 0.0
        return self.sigreg(valid_latent.reshape(1, -1, latent.shape[-1]))

    @torch.no_grad()
    def _representation_diagnostics(
        self, latent: torch.Tensor, valid_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        valid_latent = latent[valid_mask]
        samples = valid_latent.reshape(-1, latent.shape[-1]).float()
        centered = samples - samples.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
        trace = covariance.diagonal().sum()
        effective_rank = trace.square() / covariance.square().sum().clamp_min(1e-8)
        effective_rank_cap = max(
            min(samples.shape[-1], samples.shape[0] - 1), 1
        )
        sample_std = samples.std(dim=0, unbiased=False).mean()

        consecutive_deltas = torch.stack(
            [latent[:, 1] - latent[:, 0], latent[:, 2] - latent[:, 1]], dim=1
        )
        consecutive_valid = torch.stack(
            [
                valid_mask[:, 0] & valid_mask[:, 1],
                valid_mask[:, 1] & valid_mask[:, 2],
            ],
            dim=1,
        )
        valid_deltas = consecutive_deltas[consecutive_valid]
        if valid_deltas.numel() == 0:
            delta_dim_std_mean = latent.new_zeros(())
            dead_dim_fraction = latent.new_ones(())
        else:
            delta_samples = valid_deltas.reshape(-1, latent.shape[-1]).float()
            delta_dim_std = delta_samples.std(dim=0, unbiased=False)
            delta_dim_std_mean = delta_dim_std.mean()
            dead_dim_fraction = (delta_dim_std < 1e-3).float().mean()
        return {
            "latent_sample_std": sample_std,
            "latent_effective_rank": effective_rank,
            "latent_effective_rank_fraction": effective_rank
            / effective_rank_cap,
            "latent_diagnostic_sample_count": latent.new_tensor(
                samples.shape[0], dtype=torch.float32
            ),
            "temporal_delta_dim_std_mean": delta_dim_std_mean,
            "temporal_dead_dim_fraction": dead_dim_fraction,
        }

    def forward(
        self,
        patches: torch.Tensor,
        *,
        goal: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        latent = self.projector(patches)
        if latent.shape[1] != self.required_frames:
            raise ValueError(
                "smooth latent training expects "
                f"{self.required_frames} frames "
                "(consecutive targets plus one far frame), "
                f"got T={latent.shape[1]}"
            )
        batch = latent.shape[0]
        if valid_mask is None:
            valid_mask = torch.ones(
                batch, self.required_frames, device=latent.device, dtype=torch.bool
            )
        else:
            valid_mask = valid_mask.to(device=latent.device, dtype=torch.bool)
            if valid_mask.shape != (batch, self.required_frames):
                raise ValueError(
                    "expected valid_mask shape "
                    f"{(batch, self.required_frames)}, got {tuple(valid_mask.shape)}"
                )

        current = latent[:, :1]
        future = latent[:, 1 : 1 + self.n_future * self.rollout_steps]
        predicted_delta = self.predictor(
            current.unsqueeze(2), goal=goal, state=None
        ).squeeze(2)
        predicted_future = current + predicted_delta

        prediction_losses = []
        copy_losses = []
        direction_cosines = []
        horizon_valid_masks = []
        for horizon_index in range(self.n_future):
            target = future[:, horizon_index]
            prediction_error = (
                predicted_future[:, horizon_index].float() - target.detach().float()
            ).square().mean(dim=1)
            copy_error = (
                current[:, 0].detach().float() - target.detach().float()
            ).square().mean(dim=1)
            horizon_mask = valid_mask[:, 0] & valid_mask[:, horizon_index + 1]
            horizon_valid_masks.append(horizon_mask)
            prediction_losses.append(_masked_mean(prediction_error, horizon_mask))
            copy_losses.append(_masked_mean(copy_error, horizon_mask))

            predicted_motion = predicted_future[:, horizon_index] - current[:, 0]
            true_motion = target.detach() - current[:, 0].detach()
            direction = F.cosine_similarity(
                predicted_motion.float().flatten(1),
                true_motion.float().flatten(1),
                dim=-1,
                eps=1e-8,
            )
            direction_cosines.append(_masked_mean(direction, horizon_mask))
        prediction_loss = torch.stack(prediction_losses).mean()

        output: dict[str, torch.Tensor] = {}
        # Steps 2..K close the loop: the predictor is re-anchored on its own
        # output and the residual is supervised against the true future.  Step
        # 1 stays the teacher-forced headline so ``latent_prediction_loss`` and
        # the copy-ratio diagnostics remain comparable with single-shot runs.
        rollout_step_losses = []
        if self.rollout_steps > 1:
            window = torch.cat([current, predicted_future], dim=1)[:, -1:]
            for step in range(1, self.rollout_steps):
                rollout_delta = self.predictor(
                    window.unsqueeze(2), goal=goal, state=None
                ).squeeze(2)
                anchor = window[:, -1:]
                rollout_future = anchor + rollout_delta
                start = step * self.n_future
                true_future = future[:, start : start + self.n_future]
                target_delta = (true_future - anchor).detach()
                step_errors = []
                for horizon_index in range(self.n_future):
                    horizon_mask = (
                        valid_mask[:, 0]
                        & valid_mask[:, start + horizon_index + 1]
                    )
                    step_errors.append(
                        _masked_mean(
                            (
                                rollout_delta[:, horizon_index].float()
                                - target_delta[:, horizon_index].float()
                            )
                            .square()
                            .mean(dim=1),
                            horizon_mask,
                        )
                    )
                rollout_step_losses.append(torch.stack(step_errors).mean())
                with torch.no_grad():
                    step_mask = valid_mask[
                        :, : start + self.n_future + 1
                    ].all(dim=1)
                    error = _masked_mean(
                        (rollout_future.float() - true_future.float())
                        .square()
                        .mean(dim=(1, 2)),
                        step_mask,
                    )
                    copy = _masked_mean(
                        (current[:, 0].unsqueeze(1).float() - true_future.float())
                        .square()
                        .mean(dim=(1, 2)),
                        step_mask,
                    )
                    direction = F.cosine_similarity(
                        (rollout_future - current[:, 0].unsqueeze(1))
                        .float()
                        .flatten(1),
                        (true_future - current[:, 0].unsqueeze(1))
                        .float()
                        .flatten(1),
                        dim=-1,
                        eps=1e-8,
                    )
                    output[f"rollout_to_copy_ratio_step_{step + 1}"] = (
                        error / copy.clamp_min(1e-8)
                    )
                    output[f"rollout_direction_cosine_step_{step + 1}"] = (
                        _masked_mean(direction, step_mask)
                    )
                window = torch.cat([window, rollout_future], dim=1)[:, -1:]
        rollout_latent_loss = (
            torch.stack(rollout_step_losses).mean()
            if rollout_step_losses
            else latent.sum() * 0.0
        )

        delta_01 = latent[:, 1] - latent[:, 0]
        delta_12 = latent[:, 2] - latent[:, 1]
        valid_01 = valid_mask[:, 0] & valid_mask[:, 1]
        valid_12 = valid_mask[:, 1] & valid_mask[:, 2]
        slow_01 = delta_01.float().square().mean(dim=1)
        slow_12 = delta_12.float().square().mean(dim=1)
        slow_loss = 0.5 * (
            _masked_mean(slow_01, valid_01) + _masked_mean(slow_12, valid_12)
        )

        acceleration = delta_12 - delta_01
        valid_acceleration = valid_mask[:, :3].all(dim=1)
        acceleration_per_sample = acceleration.float().square().mean(dim=1)
        acceleration_loss = _masked_mean(
            acceleration_per_sample, valid_acceleration
        )

        far_delta = latent[:, -1] - latent[:, 0]
        near_distance = 0.5 * (slow_01 + slow_12)
        far_distance = far_delta.float().square().mean(dim=1)
        valid_order = valid_mask.all(dim=1)
        temporal_order_per_sample = F.relu(
            self.temporal_order_margin + near_distance - far_distance
        )
        temporal_order_loss = _masked_mean(
            temporal_order_per_sample, valid_order
        )
        sigreg_loss = self._sigreg_loss(latent, valid_mask)

        total_loss = (
            self.prediction_weight * prediction_loss
            + self.rollout_weight * rollout_latent_loss
            + self.sigreg_weight * sigreg_loss
            + self.slow_weight * slow_loss
            + self.acceleration_weight * acceleration_loss
            + self.temporal_order_weight * temporal_order_loss
        )

        with torch.no_grad():
            prediction_mse = torch.stack(prediction_losses).mean()
            copy_mse = torch.stack(copy_losses).mean()
            adjacent_rms = slow_loss.sqrt()
            acceleration_rms = acceleration_loss.sqrt()
            far_mse = _masked_mean(far_distance, valid_order)
            far_rms = far_mse.sqrt()
            far_to_near_ratio = far_mse / slow_loss.detach().clamp_min(1e-8)
            order_satisfied = _masked_mean(
                (far_distance > near_distance).float(), valid_order
            )
            diagnostics = self._representation_diagnostics(latent, valid_mask)

        output.update({
            "loss": total_loss,
            "latent_prediction_loss": prediction_loss,
            "rollout_latent_loss": rollout_latent_loss,
            "rollout_steps": latent.new_tensor(self.rollout_steps),
            "sigreg_loss": sigreg_loss,
            "slow_loss": slow_loss,
            "acceleration_loss": acceleration_loss,
            "temporal_order_loss": temporal_order_loss,
            "prediction_mse": prediction_mse,
            "copy_mse": copy_mse,
            "prediction_to_copy_ratio": prediction_mse
            / copy_mse.clamp_min(1e-8),
            "direction_cosine": torch.stack(direction_cosines).mean(),
            "adjacent_rms": adjacent_rms,
            "acceleration_rms": acceleration_rms,
            "far_rms": far_rms,
            "far_to_near_ratio": far_to_near_ratio,
            "temporal_order_satisfied_fraction": order_satisfied,
            "valid_near_fraction": valid_acceleration.float().mean(),
            "valid_far_fraction": valid_order.float().mean(),
            "sigreg_sample_count": valid_mask.sum().to(dtype=torch.float32),
            "latent": latent,
            "pred_future_latent": predicted_future,
        })
        output.update(diagnostics)
        for horizon_index, (horizon_loss, copy_loss, direction) in enumerate(
            zip(prediction_losses, copy_losses, direction_cosines), start=1
        ):
            output[f"latent_loss_horizon_{horizon_index}"] = horizon_loss
            output[f"copy_mse_horizon_{horizon_index}"] = copy_loss
            output[f"direction_cosine_horizon_{horizon_index}"] = direction
            output[f"valid_fraction_horizon_{horizon_index}"] = (
                horizon_valid_masks[horizon_index - 1].float().mean()
            )
        return output
