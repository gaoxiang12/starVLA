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
        dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.n_future = int(n_future)
        self.num_tokens = int(num_tokens)
        self.anchor_proj = nn.Linear(latent_dim, dim)
        self.goal_proj = nn.Linear(goal_dim, dim) if goal_dim else None
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
        self, anchor: torch.Tensor, goal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, _, num_tokens, _ = anchor.shape
        if num_tokens != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} anchor tokens, got {num_tokens}")

        anchor_tokens = self.anchor_proj(anchor)
        future_tokens = self.future_query.expand(batch_size, -1, num_tokens, -1)
        hidden = torch.cat([anchor_tokens, future_tokens], dim=1)
        hidden = hidden + self.frame_embedding + self.token_embedding
        if goal is not None and self.goal_proj is not None:
            hidden = hidden + self.goal_proj(goal.to(hidden.dtype)).view(
                batch_size, 1, 1, -1
            )

        hidden = hidden.reshape(batch_size, (1 + self.n_future) * num_tokens, -1)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden).view(
            batch_size, 1 + self.n_future, num_tokens, -1
        )
        return self.out(hidden[:, 1:])


class VisualTokenLatentWorldModel(nn.Module):
    """Predict all future visual-token latents in one deterministic pass."""

    def __init__(
        self,
        *,
        latent_dim: int,
        goal_dim: Optional[int],
        n_future: int,
        num_tokens: int,
        dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 1024,
        sigreg_weight: float = 0.0,
        stats_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.n_future = int(n_future)
        self.num_tokens = int(num_tokens)
        self.stats_momentum = float(stats_momentum)
        self._stats_eps = 1e-4
        self.residual_predictor = TokenResidualPredictor(
            latent_dim=latent_dim,
            goal_dim=goal_dim,
            n_future=self.n_future,
            num_tokens=self.num_tokens,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
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
        module = super()._apply(*args, **kwargs)
        module.delta_scale = module.delta_scale.float()
        module._delta_scale_ready = module._delta_scale_ready.float()
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
        self, context: torch.Tensor, goal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        anchor = context[:, -1:]
        predicted_delta = self.residual_predictor(anchor, goal=goal)
        scale = self.delta_scale.clamp_min(self._stats_eps)
        return anchor + predicted_delta * scale

    def forward(
        self,
        latent: torch.Tensor,
        *,
        ctx_len: int,
        goal: Optional[torch.Tensor] = None,
        update_stats: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch_size, total_frames, num_tokens = latent.shape[:3]
        if num_tokens != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} visual tokens, got {num_tokens}")
        if total_frames - ctx_len < self.n_future:
            raise ValueError(
                f"need {self.n_future} future frames, got {total_frames - ctx_len}"
            )

        anchor = latent[:, ctx_len - 1 : ctx_len]
        future = latent[:, ctx_len : ctx_len + self.n_future]
        residual = (future - anchor).detach()
        if self.training and update_stats:
            self._update_delta_scale(residual)

        scale = self.delta_scale.clamp_min(self._stats_eps)
        target_delta = residual / scale
        predicted_delta = self.residual_predictor(anchor, goal=goal)
        predicted_residual = predicted_delta * scale
        predicted_future = anchor + predicted_residual
        output = {
            "latent_loss": F.mse_loss(predicted_delta, target_delta),
            "pred_future_latent": predicted_future,
        }

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
        return output
