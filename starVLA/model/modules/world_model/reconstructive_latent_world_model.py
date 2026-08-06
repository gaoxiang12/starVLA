"""Joint reconstructive latent world model over frozen DINO spatial patches.

The module deliberately keeps the objective small:

1. a compact latent reconstructs fixed, normalized DINO patch features;
2. a deterministic dynamics model predicts future compact latents; and
3. the same compact latent decodes aligned robot proprioceptive state.

There are no variance/covariance auxiliary objectives.  A non-affine
LayerNorm fixes the per-token latent scale, while the reconstruction and state
losses rule out the constant-code solution.  DINO features are detached at the
module boundary so enabling this branch can never fine-tune the image encoder.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_token_delta_world_model import VisualTokenLatentWorldModel


class SpatialBlockCodec(nn.Module):
    """Compress fixed spatial DINO cells and decode both DINO and robot state.

    The raw square DINO patch grid is first pooled without parameters to a
    ``pool_grid_size x pool_grid_size`` grid.  Each non-overlapping spatial
    block is encoded independently, which gives every compact latent token a
    stable ``(view, row, col)`` identity without learned slot matching.
    """

    def __init__(
        self,
        *,
        patch_dim: int,
        num_views: int,
        state_dim: int,
        pool_grid_size: int = 4,
        latent_grid_size: int = 2,
        latent_dim: int = 256,
        codec_hidden_dim: int = 512,
        state_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.num_views = int(num_views)
        self.state_dim = int(state_dim)
        self.pool_grid_size = int(pool_grid_size)
        self.latent_grid_size = int(latent_grid_size)
        self.latent_dim = int(latent_dim)
        if self.patch_dim < 1 or self.num_views < 1 or self.state_dim < 1:
            raise ValueError("patch_dim, num_views, and state_dim must be positive")
        if self.pool_grid_size < 1 or self.latent_grid_size < 1:
            raise ValueError("spatial grid sizes must be positive")
        if self.pool_grid_size % self.latent_grid_size != 0:
            raise ValueError(
                "pool_grid_size must be divisible by latent_grid_size, got "
                f"{self.pool_grid_size} and {self.latent_grid_size}"
            )
        if self.latent_dim < 2:
            raise ValueError("latent_dim must be at least 2 for non-affine LayerNorm")

        self.block_size = self.pool_grid_size // self.latent_grid_size
        self.tokens_per_view = self.latent_grid_size**2
        self.num_latent_tokens = self.num_views * self.tokens_per_view
        self.block_feature_dim = self.block_size**2 * self.patch_dim

        self.encoder = nn.Sequential(
            nn.LayerNorm(self.block_feature_dim),
            nn.Linear(self.block_feature_dim, int(codec_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(codec_hidden_dim), self.latent_dim),
            # Fix code scale without introducing a variance/covariance loss.
            nn.LayerNorm(self.latent_dim, elementwise_affine=False),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, int(codec_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(codec_hidden_dim), self.block_feature_dim),
        )
        flattened_latent_dim = self.num_latent_tokens * self.latent_dim
        self.state_decoder = nn.Sequential(
            nn.LayerNorm(flattened_latent_dim),
            nn.Linear(flattened_latent_dim, int(state_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(state_hidden_dim), self.state_dim),
        )

    def spatial_target(self, patches: torch.Tensor) -> torch.Tensor:
        """Return fixed normalized DINO content as ``(B,T,V,G,G,D)``."""

        if patches.ndim != 5:
            raise ValueError(
                "expected DINO patches with shape (B,T,V,N,D), got "
                f"{tuple(patches.shape)}"
            )
        batch, frames, views, patch_count, channels = patches.shape
        if views != self.num_views or channels != self.patch_dim:
            raise ValueError(
                "DINO patch shape mismatch: expected "
                f"V={self.num_views}, D={self.patch_dim}, got V={views}, D={channels}"
            )
        patch_grid = math.isqrt(patch_count)
        if patch_grid * patch_grid != patch_count:
            raise ValueError(f"DINO patch count must be square, got {patch_count}")

        # DINO is a fixed teacher.  Detach before any casting/pooling so no
        # reconstruction or prediction gradient can reach the encoder.
        x = patches.detach().float().reshape(
            batch * frames * views, patch_grid, patch_grid, channels
        )
        x = x.permute(0, 3, 1, 2)
        if patch_grid != self.pool_grid_size:
            x = F.adaptive_avg_pool2d(
                x, (self.pool_grid_size, self.pool_grid_size)
            )
        x = x.permute(0, 2, 3, 1).reshape(
            batch,
            frames,
            views,
            self.pool_grid_size,
            self.pool_grid_size,
            channels,
        )
        # Per-cell normalization is deterministic and avoids an extra
        # calibration/training stage while preserving DINO feature direction.
        return F.layer_norm(x, (channels,))

    def _grid_to_blocks(self, spatial: torch.Tensor) -> torch.Tensor:
        if spatial.ndim != 6:
            raise ValueError(
                "expected spatial features (B,T,V,G,G,D), got "
                f"{tuple(spatial.shape)}"
            )
        batch, frames, views, rows, cols, channels = spatial.shape
        expected = (
            self.num_views,
            self.pool_grid_size,
            self.pool_grid_size,
            self.patch_dim,
        )
        if (views, rows, cols, channels) != expected:
            raise ValueError(
                "spatial target shape mismatch: expected suffix "
                f"{expected}, got {(views, rows, cols, channels)}"
            )
        latent_grid = self.latent_grid_size
        block = self.block_size
        return (
            spatial.reshape(
                batch,
                frames,
                views,
                latent_grid,
                block,
                latent_grid,
                block,
                channels,
            )
            .permute(0, 1, 2, 3, 5, 4, 6, 7)
            .reshape(
                batch,
                frames,
                self.num_latent_tokens,
                self.block_feature_dim,
            )
        )

    def _blocks_to_grid(self, blocks: torch.Tensor) -> torch.Tensor:
        if blocks.ndim != 4:
            raise ValueError(
                "expected decoded blocks (B,T,K,F), got "
                f"{tuple(blocks.shape)}"
            )
        batch, frames, tokens, features = blocks.shape
        if tokens != self.num_latent_tokens or features != self.block_feature_dim:
            raise ValueError(
                "decoded block shape mismatch: expected "
                f"K={self.num_latent_tokens}, F={self.block_feature_dim}, "
                f"got K={tokens}, F={features}"
            )
        latent_grid = self.latent_grid_size
        block = self.block_size
        return (
            blocks.reshape(
                batch,
                frames,
                self.num_views,
                latent_grid,
                latent_grid,
                block,
                block,
                self.patch_dim,
            )
            .permute(0, 1, 2, 3, 5, 4, 6, 7)
            .reshape(
                batch,
                frames,
                self.num_views,
                self.pool_grid_size,
                self.pool_grid_size,
                self.patch_dim,
            )
        )

    def encode_spatial(self, spatial: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._grid_to_blocks(spatial))

    def encode_patches(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.spatial_target(patches)
        return self.encode_spatial(target), target

    def decode_spatial(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4:
            raise ValueError(
                "expected compact latent (B,T,K,C), got "
                f"{tuple(latent.shape)}"
            )
        if latent.shape[2:] != (self.num_latent_tokens, self.latent_dim):
            raise ValueError(
                "compact latent shape mismatch: expected suffix "
                f"{(self.num_latent_tokens, self.latent_dim)}, "
                f"got {tuple(latent.shape[2:])}"
            )
        return self._blocks_to_grid(self.decoder(latent))

    def decode_state(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4:
            raise ValueError(
                "expected compact latent (B,T,K,C), got "
                f"{tuple(latent.shape)}"
            )
        batch, frames, tokens, channels = latent.shape
        if (tokens, channels) != (self.num_latent_tokens, self.latent_dim):
            raise ValueError(
                "compact latent shape mismatch for state decode: expected "
                f"{(self.num_latent_tokens, self.latent_dim)}, "
                f"got {(tokens, channels)}"
            )
        return self.state_decoder(latent.reshape(batch, frames, -1))


class ReconstructiveSpatialLatentWorldModel(nn.Module):
    """Joint DINO reconstruction, compact-latent prediction, and state decode."""

    def __init__(
        self,
        *,
        patch_dim: int,
        num_views: int,
        goal_dim: Optional[int],
        state_dim: int,
        n_future: int = 2,
        context_len: int = 1,
        pool_grid_size: int = 4,
        latent_grid_size: int = 2,
        latent_dim: int = 256,
        codec_hidden_dim: int = 512,
        state_hidden_dim: int = 512,
        predictor_dim: int = 256,
        predictor_depth: int = 4,
        predictor_heads: int = 8,
        predictor_ffn_dim: int = 1024,
        reconstruction_weight: float = 1.0,
        prediction_weight: float = 1.0,
        state_weight: float = 0.5,
        stats_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.n_future = int(n_future)
        self.context_len = int(context_len)
        self.state_dim = int(state_dim)
        if self.n_future < 1 or self.context_len < 1:
            raise ValueError("n_future and context_len must be positive")
        self.reconstruction_weight = float(reconstruction_weight)
        self.prediction_weight = float(prediction_weight)
        self.state_weight = float(state_weight)
        if min(
            self.reconstruction_weight,
            self.prediction_weight,
            self.state_weight,
        ) < 0:
            raise ValueError("reconstructive loss weights must be non-negative")
        if (
            self.reconstruction_weight
            + self.prediction_weight
            + self.state_weight
            <= 0
        ):
            raise ValueError("at least one reconstructive loss weight must be positive")

        self.codec = SpatialBlockCodec(
            patch_dim=patch_dim,
            num_views=num_views,
            state_dim=state_dim,
            pool_grid_size=pool_grid_size,
            latent_grid_size=latent_grid_size,
            latent_dim=latent_dim,
            codec_hidden_dim=codec_hidden_dim,
            state_hidden_dim=state_hidden_dim,
        )
        self.predictor = VisualTokenLatentWorldModel(
            latent_dim=latent_dim,
            goal_dim=goal_dim,
            n_future=self.n_future,
            num_tokens=self.codec.num_latent_tokens,
            context_len=self.context_len,
            state_dim=0,
            dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            ffn_dim=predictor_ffn_dim,
            sigreg_weight=0.0,
            stats_momentum=stats_momentum,
        )

    def forward(
        self,
        patches: torch.Tensor,
        *,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        update_stats: bool = True,
    ) -> dict[str, torch.Tensor]:
        latent, dino_target = self.codec.encode_patches(patches)
        total_frames = latent.shape[1]
        required_frames = self.context_len + self.n_future
        if total_frames < required_frames:
            raise ValueError(
                f"reconstructive world model needs {required_frames} frames, "
                f"got {total_frames}"
            )
        latent = latent[:, :required_frames]
        dino_target = dino_target[:, :required_frames]
        if state.ndim != 3:
            raise ValueError(
                "aligned robot state must have shape (B,T,D), got "
                f"{tuple(state.shape)}"
            )
        expected_state_shape = (latent.shape[0], required_frames, self.state_dim)
        if tuple(state.shape[:3]) != expected_state_shape:
            raise ValueError(
                f"expected aligned state shape {expected_state_shape}, "
                f"got {tuple(state.shape)}"
            )
        state_target = state.float()

        reconstructed_dino = self.codec.decode_spatial(latent)
        decoded_state = self.codec.decode_state(latent)
        reconstruction_loss = F.mse_loss(
            reconstructed_dino.float(), dino_target.float()
        )
        state_loss = F.mse_loss(decoded_state.float(), state_target)

        prediction = self.predictor(
            latent,
            ctx_len=self.context_len,
            goal=goal,
            state=None,
            update_stats=update_stats,
            rollout_steps=1,
        )
        latent_prediction_loss = prediction["latent_loss"]
        predicted_future_latent = prediction["pred_future_latent"]
        total_loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.prediction_weight * latent_prediction_loss
            + self.state_weight * state_loss
        )

        future_slice = slice(self.context_len, required_frames)
        current_dino = dino_target[:, self.context_len - 1 : self.context_len]
        future_dino = dino_target[:, future_slice]
        current_state = state_target[:, self.context_len - 1 : self.context_len]
        future_state = state_target[:, future_slice]
        with torch.no_grad():
            predicted_future_dino = self.codec.decode_spatial(
                predicted_future_latent
            )
            predicted_future_state = self.codec.decode_state(
                predicted_future_latent
            )
            predicted_dino_mse = F.mse_loss(
                predicted_future_dino.float(), future_dino.float()
            )
            dino_copy_mse = F.mse_loss(
                current_dino.expand_as(future_dino).float(), future_dino.float()
            )
            predicted_state_mse = F.mse_loss(
                predicted_future_state.float(), future_state.float()
            )
            state_copy_mse = F.mse_loss(
                current_state.expand_as(future_state).float(), future_state.float()
            )
            target_energy = dino_target.float().square().mean().clamp_min(1e-8)
            latent_sample_std = (
                latent.float()
                .reshape(latent.shape[0] * latent.shape[1], -1)
                .std(dim=0, unbiased=False)
                .mean()
            )
            latent_dynamic_rms = (
                (latent[:, 1:] - latent[:, :-1])
                .float()
                .square()
                .mean()
                .sqrt()
            )

        output = {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "latent_prediction_loss": latent_prediction_loss,
            "state_loss": state_loss,
            "reconstruction_nmse": reconstruction_loss / target_energy,
            "predicted_dino_mse": predicted_dino_mse,
            "dino_copy_mse": dino_copy_mse,
            "predicted_dino_to_copy_ratio": predicted_dino_mse
            / dino_copy_mse.clamp_min(1e-8),
            "predicted_state_mse": predicted_state_mse,
            "state_copy_mse": state_copy_mse,
            "predicted_state_to_copy_ratio": predicted_state_mse
            / state_copy_mse.clamp_min(1e-8),
            "latent_sample_std": latent_sample_std,
            "latent_dynamic_rms": latent_dynamic_rms,
            "latent": latent,
            "pred_future_latent": predicted_future_latent,
            "reconstructed_dino": reconstructed_dino,
            "decoded_state": decoded_state,
        }
        for name in (
            "delta_scale",
            "delta_target_rms",
            "delta_pred_rms",
            "delta_copy_mse",
            "delta_pred_mse",
            "delta_mean_baseline_mse",
            "delta_to_copy_ratio",
            "delta_direction_cosine",
        ):
            if name in prediction:
                output[name] = prediction[name]
        for name, value in prediction.items():
            if name.startswith("latent_loss_horizon_"):
                output[name] = value
        return output
