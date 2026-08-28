"""Future-latent prediction used by the checkpoint-compatible GAWM recipe."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Predict future visual-token residuals from one current frame and task."""

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
        self,
        context: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, context_len, num_tokens, _ = context.shape
        if context_len != 1:
            raise ValueError(f"GAWM expects one context frame, got {context_len}")
        if num_tokens != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} context tokens, got {num_tokens}")

        current_tokens = self.anchor_proj(context)
        future_tokens = self.future_query.expand(batch_size, -1, num_tokens, -1)
        hidden = torch.cat([current_tokens, future_tokens], dim=1)
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
        stats_momentum: float = 0.9,
        detach_input: bool = True,
    ) -> None:
        super().__init__()
        self.n_future = int(n_future)
        self.num_tokens = int(num_tokens)
        self.stats_momentum = float(stats_momentum)
        self.detach_input = bool(detach_input)
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
        self.register_buffer("delta_scale", torch.ones(1))
        self.register_buffer("_delta_scale_ready", torch.zeros(1))

    def _apply(self, *args, **kwargs):
        # Keep EMA statistics in fp32 under module-wide bf16 conversion.
        delta_scale_fp32 = self.delta_scale.detach().float().clone()
        delta_scale_ready_fp32 = self._delta_scale_ready.detach().float().clone()
        module = super()._apply(*args, **kwargs)
        target_device = module.delta_scale.device
        module.delta_scale = delta_scale_fp32.to(device=target_device)
        module._delta_scale_ready = delta_scale_ready_fp32.to(device=target_device)
        return module

    @torch.no_grad()
    def _update_delta_scale(
        self,
        residual: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if mask is not None:
            weights = mask.to(device=residual.device, dtype=torch.float32)
            denominator = (weights.sum() * residual.shape[-1]).clamp_min(1.0)
            rms = ((residual.float().square() * weights).sum() / denominator).clamp_min(
                self._stats_eps
            ).sqrt()
        else:
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
    ) -> torch.Tensor:
        if context.shape[1] != 1:
            raise ValueError(f"GAWM expects one context frame, got {context.shape[1]}")
        predicted_delta = self.residual_predictor(context, goal=goal)
        return context + predicted_delta * self.delta_scale.clamp_min(self._stats_eps)

    def forward(
        self,
        latent: torch.Tensor,
        *,
        ctx_len: int,
        goal: Optional[torch.Tensor] = None,
        update_stats: bool = True,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, total_frames, num_tokens = latent.shape[:3]
        if num_tokens != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} visual tokens, got {num_tokens}")
        if ctx_len != 1:
            raise ValueError(f"GAWM expects ctx_len=1, got {ctx_len}")
        if total_frames != 1 + self.n_future:
            raise ValueError(
                f"expected {1 + self.n_future} temporal frames, got {total_frames}"
            )
        if loss_mask is not None:
            if tuple(loss_mask.shape) != (batch_size, total_frames, num_tokens):
                raise ValueError(
                    "expected loss_mask shape "
                    f"{(batch_size, total_frames, num_tokens)}, got {tuple(loss_mask.shape)}"
                )
            if loss_mask.dtype not in {torch.bool, torch.float32}:
                raise ValueError(f"loss_mask must be bool or float32, got {loss_mask.dtype}")

        context = latent[:, :1]
        anchor = context
        if self.detach_input:
            context = context.detach()
            anchor = anchor.detach()
        future = latent[:, 1:]
        residual = (future - anchor).detach()
        residual_mask = loss_mask[:, 1:].unsqueeze(-1) if loss_mask is not None else None
        if self.training and update_stats:
            self._update_delta_scale(residual, mask=residual_mask)

        scale = self.delta_scale.clamp_min(self._stats_eps)
        predicted_residual = self.residual_predictor(context, goal=goal) * scale
        predicted_future = anchor + predicted_residual
        squared_error = (predicted_future.float() - future.detach().float()).square()
        if residual_mask is not None:
            weights = residual_mask.to(dtype=squared_error.dtype)
            latent_loss = (squared_error * weights).sum() / (
                weights.sum() * squared_error.shape[-1]
            ).clamp_min(1.0)
            pred_cosine_input = predicted_residual.float() * weights
            true_cosine_input = residual.float() * weights
        else:
            latent_loss = squared_error.mean()
            pred_cosine_input = predicted_residual.float()
            true_cosine_input = residual.float()
        direction_cosine = F.cosine_similarity(
            pred_cosine_input.flatten(2),
            true_cosine_input.flatten(2),
            dim=-1,
            eps=1e-8,
        ).mean()
        output = {
            "latent_loss": latent_loss,
            "latent_cosine_loss": 1.0 - direction_cosine,
            "pred_future_latent": predicted_future,
        }
        if residual_mask is not None:
            weights = residual_mask.to(dtype=squared_error.dtype)
            numerator = (squared_error * weights).sum(dim=(0, 2, 3))
            denominator = (
                weights.sum(dim=(0, 2, 3)) * squared_error.shape[-1]
            ).clamp_min(1.0)
            per_horizon_loss = numerator / denominator
        else:
            per_horizon_loss = squared_error.mean(dim=(0, 2, 3))
        for horizon_index, horizon_loss in enumerate(per_horizon_loss, start=1):
            output[f"latent_loss_horizon_{horizon_index}"] = horizon_loss

        with torch.no_grad():
            if residual_mask is not None:
                weights = residual_mask.to(dtype=torch.float32)
                denominator = (weights.sum() * residual.shape[-1]).clamp_min(1.0)

                def masked_mse(values: torch.Tensor) -> torch.Tensor:
                    return (values.float().square() * weights).sum() / denominator

                copy_mse = masked_mse(residual)
                pred_mse = masked_mse(predicted_residual - residual)
                mean_residual = (residual.float() * weights).sum(
                    dim=0, keepdim=True
                ) / weights.sum(dim=0, keepdim=True).clamp_min(1.0)
                mean_baseline_mse = masked_mse(residual - mean_residual)
                pred_rms = masked_mse(predicted_residual).sqrt()
            else:
                copy_mse = residual.float().square().mean()
                pred_mse = (predicted_residual.float() - residual.float()).square().mean()
                mean_residual = residual.float().mean(dim=0, keepdim=True)
                mean_baseline_mse = (residual.float() - mean_residual).square().mean()
                pred_rms = predicted_residual.float().square().mean().sqrt()
            output.update(
                {
                    "delta_scale": scale.detach().mean(),
                    "delta_target_rms": copy_mse.sqrt(),
                    "delta_pred_rms": pred_rms,
                    "delta_copy_mse": copy_mse,
                    "delta_pred_mse": pred_mse,
                    "delta_mean_baseline_mse": mean_baseline_mse,
                    "delta_to_copy_ratio": pred_mse / copy_mse.clamp_min(1e-8),
                    "delta_direction_cosine": direction_cosine,
                }
            )
        return output
