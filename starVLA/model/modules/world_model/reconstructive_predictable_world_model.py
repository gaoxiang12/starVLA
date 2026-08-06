"""A reconstructive latent whose temporal evolution is actually predictable.

Motivation, measured on ``steps_220000`` over held-out LIBERO episodes:

* The pooled DINOv3 token stream moves a lot in time (a 32-frame window travels
  1.39x further than the spread across completely different scenes) but its
  frame-to-frame delta has lag-1 autocorrelation -0.52, i.e. it is
  indistinguishable from white.  An MSE world model on it is pinned at 0.5,
  which is the exact algebraic floor when ``L_t = S_t + e_t`` with white ``e_t``:
  ``copy_mse = 2v`` and the best reachable error is ``v``.
* A plain reconstructive bottleneck does *not* fix this.  PCA ranks directions
  by variance and the incoherent component sits in exactly the high-variance
  directions, so shrinking to 8 dims made prediction *worse* (copy ratio 0.68
  against 0.53 in the full space).  Reconstruction and predictability fight.
* They stop fighting once the reconstruction target is smoothed over time.
  Against a width-5 temporal mean of the future tokens, a linear map from the
  current frame reaches copy ratio 0.2163 and cuts absolute MSE from 1.58 to
  0.39, because smoothing suppresses the incoherent term by ``~m^2`` while
  leaving the scene content intact.

So this module keeps a low-dimensional latent that must (1) decode the
*temporally smoothed* DINOv3 tokens, (2) predict its own future, and (3) decode
the robot state.  Requirement (1) is what stops (2) from collapsing to a
constant, and (3) grounds the latent in physical configuration rather than
appearance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def smooth_targets(tokens: torch.Tensor, width: int) -> torch.Tensor:
    """Box-average ``tokens`` (B, T, K, C) over time with edge padding.

    The incoherent part of the DINOv3 delta shrinks like ``1/width**2`` under
    this average while scene content is preserved, which is what turns the
    reconstruction target into something a predictor can actually chase.
    """
    if width <= 1:
        return tokens
    if width % 2 == 0:
        raise ValueError(f"target_smooth_width must be odd, got {width}")
    batch, frames, num_tokens, channels = tokens.shape
    flat = tokens.permute(0, 2, 3, 1).reshape(-1, 1, frames)
    padded = F.pad(flat, (width // 2, width // 2), mode="replicate")
    kernel = tokens.new_full((1, 1, width), 1.0 / width)
    smoothed = F.conv1d(padded, kernel)
    return smoothed.reshape(batch, num_tokens, channels, frames).permute(0, 3, 1, 2)


class _TransformerStack(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "norm1": nn.LayerNorm(dim),
                    "attn": nn.MultiheadAttention(
                        dim, num_heads, batch_first=True
                    ),
                    "norm2": nn.LayerNorm(dim),
                    "ffn": nn.Sequential(
                        nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim)
                    ),
                }
            )
            for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            normed = block["norm1"](x)
            x = x + block["attn"](normed, normed, normed, need_weights=False)[0]
            x = x + block["ffn"](block["norm2"](x))
        return x


class ReconstructivePredictableWorldModel(nn.Module):
    """Encode spatial tokens into a compact, temporally predictable latent.

    The encoder/decoder pair and the predictor are trained jointly, but the
    predictor's target is detached: gradients from the prediction loss must not
    be able to reshape the latent into something trivially predictable.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        latent_dim: int,
        goal_dim: int,
        n_future: int,
        context_len: int = 1,
        state_dim: int = 0,
        hidden_dim: int = 384,
        encoder_depth: int = 2,
        decoder_depth: int = 2,
        predictor_depth: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 1024,
        target_smooth_width: int = 5,
        variance_floor: float = 1.0,
        temporal_floor: float = 0.1,
        stats_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.latent_dim = int(latent_dim)
        self.n_future = int(n_future)
        self.context_len = int(context_len)
        self.state_dim = int(state_dim)
        self.target_smooth_width = int(target_smooth_width)
        self.variance_floor = float(variance_floor)
        self.temporal_floor = float(temporal_floor)
        self.stats_momentum = float(stats_momentum)
        self._stats_eps = 1e-4

        self.token_in = nn.Linear(token_dim, hidden_dim)
        self.token_position = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        self.encoder = _TransformerStack(hidden_dim, encoder_depth, num_heads, ffn_dim)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder_queries = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        self.decoder = _TransformerStack(hidden_dim, decoder_depth, num_heads, ffn_dim)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.token_out = nn.Linear(hidden_dim, token_dim)

        self.state_head = (
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, state_dim)
            )
            if state_dim > 0
            else None
        )

        predictor_input = latent_dim * self.context_len + goal_dim
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input, hidden_dim),
            nn.GELU(),
            *[
                layer
                for _ in range(predictor_depth - 1)
                for layer in (nn.Linear(hidden_dim, hidden_dim), nn.GELU())
            ],
            nn.Linear(hidden_dim, latent_dim * self.n_future),
        )
        # Zero init makes the predictor start as an exact copy baseline, so the
        # reported ratio begins at 1.0 and any drop is a real gain.
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

        nn.init.normal_(self.token_position, std=0.02)
        nn.init.normal_(self.decoder_queries, std=0.02)
        self.register_buffer("delta_scale", torch.ones(()))
        self.register_buffer("_delta_scale_ready", torch.zeros(()))

    def _apply(self, fn, recurse: bool = True):
        # Mirrors VisualTokenLatentWorldModel: bf16 casting quantises the EMA
        # away, so these two buffers are pinned to fp32 after any dtype change.
        scale = self.delta_scale.detach().clone().float()
        ready = self._delta_scale_ready.detach().clone().float()
        module = super()._apply(fn, recurse)
        device = module.delta_scale.device
        module.delta_scale = scale.to(device=device)
        module._delta_scale_ready = ready.to(device=device)
        return module

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, frames = tokens.shape[:2]
        x = self.token_in(tokens).reshape(batch * frames, self.num_tokens, -1)
        x = self.encoder(x + self.token_position)
        pooled = self.encoder_norm(x).mean(dim=1)
        return self.to_latent(pooled).reshape(batch, frames, self.latent_dim)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        batch, frames = latent.shape[:2]
        seed = self.from_latent(latent).reshape(batch * frames, 1, -1)
        x = self.decoder(seed + self.decoder_queries)
        tokens = self.token_out(self.decoder_norm(x))
        return tokens.reshape(batch, frames, self.num_tokens, self.token_dim)

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

    def predict_latent(
        self, context: torch.Tensor, goal: torch.Tensor | None = None
    ) -> torch.Tensor:
        if context.shape[1] != self.context_len:
            raise ValueError(
                f"expected {self.context_len} context frames, got {context.shape[1]}"
            )
        batch = context.shape[0]
        features = context.reshape(batch, -1)
        if goal is not None:
            features = torch.cat([features, goal], dim=-1)
        delta = self.predictor(features).reshape(batch, self.n_future, self.latent_dim)
        scale = self.delta_scale.clamp_min(self._stats_eps)
        return context[:, -1:] + delta * scale

    def rollout_latent(
        self, context: torch.Tensor, *, steps: int, goal: torch.Tensor | None = None
    ) -> torch.Tensor:
        if int(steps) < 1:
            raise ValueError(f"rollout steps must be at least 1, got {steps}")
        predictions = []
        window = context
        for _ in range(int(steps)):
            step = self.predict_latent(window, goal=goal)
            predictions.append(step)
            window = torch.cat([window, step], dim=1)[:, -self.context_len :]
        return torch.cat(predictions, dim=1)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        goal: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        update_stats: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch, frames, num_tokens = tokens.shape[:3]
        if num_tokens != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} tokens, got {num_tokens}")
        required = self.context_len + self.n_future
        if frames < required:
            raise ValueError(f"need at least {required} frames, got {frames}")

        latent = self.encode(tokens)
        reconstruction = self.decode(latent)
        target_tokens = smooth_targets(tokens, self.target_smooth_width).detach()
        recon_loss = (reconstruction.float() - target_tokens.float()).square().mean()

        context = latent[:, : self.context_len]
        future = latent[:, self.context_len : self.context_len + self.n_future]
        # The target is detached but the context is not: requirement (2) asks the
        # latent itself to become predictable, so this loss is meant to shape the
        # encoder.  Two things stop it buying that by freezing the latent in
        # time -- dividing by the EMA ``delta_scale`` makes it invariant to the
        # residual's magnitude, and ``temporal_variance_loss`` floors how much of
        # each dimension's variability may leave the temporal axis.
        residual = (future - context[:, -1:]).detach()
        if self.training and update_stats:
            self._update_delta_scale(residual)
        scale = self.delta_scale.clamp_min(self._stats_eps)

        predicted = self.predict_latent(context, goal=goal)
        predicted_delta = (predicted - context[:, -1:]) / scale
        target_delta = residual / scale
        latent_loss = (predicted_delta.float() - target_delta.float()).square().mean()

        flat_latent = latent.reshape(-1, self.latent_dim).float()
        spread = flat_latent.std(dim=0)
        variance_loss = F.relu(self.variance_floor - spread).mean()
        temporal_spread = (latent[:, 1:] - latent[:, :-1]).float().std(dim=(0, 1))
        temporal_variance_loss = F.relu(
            self.temporal_floor - temporal_spread / spread.clamp_min(1e-6)
        ).mean()

        output = {
            "latent": latent,
            "pred_future_latent": predicted,
            "recon_loss": recon_loss,
            "latent_loss": latent_loss,
            "variance_loss": variance_loss,
            "temporal_variance_loss": temporal_variance_loss,
        }

        if self.state_head is not None:
            if state is None:
                raise ValueError("state_dim > 0 requires a state tensor")
            predicted_state = self.state_head(latent[:, : state.shape[1]])
            output["state_loss"] = (
                predicted_state.float() - state.float()
            ).square().mean()

        with torch.no_grad():
            copy_mse = target_delta.float().square().mean()
            pred_mse = (predicted_delta.float() - target_delta.float()).square().mean()
            token_variance = target_tokens.float().var()
            output.update(
                {
                    "delta_scale": scale.detach().clone(),
                    "latent_to_copy_ratio": pred_mse / copy_mse.clamp_min(1e-8),
                    "latent_direction_cosine": F.cosine_similarity(
                        predicted_delta.float().flatten(1),
                        target_delta.float().flatten(1),
                        dim=-1,
                        eps=1e-8,
                    ).mean(),
                    "recon_nmse": recon_loss / token_variance.clamp_min(1e-8),
                    "latent_std": spread.mean(),
                    "latent_temporal_std_ratio": (
                        temporal_spread / spread.clamp_min(1e-6)
                    ).mean(),
                }
            )
        return output
