# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""LeWM-OFT with deterministic visual-token future prediction.

A ViT encodes spatially anchored tokens for every camera view. An action-free
transformer predicts all future-token residuals in one pass, and an OFT head
cross-attends current and predicted tokens to produce the action chunk.
"""

import hashlib
import math
import os
import sys
import unicodedata
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.ACT_ActionHeader import (
    TurboStyleACTActionHead,
)
from starVLA.model.modules.action_model.action_loss import (
    action_l1_diagnostics,
    masked_action_l1_loss,
)
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.modules.world_model.visual_token_delta_world_model import (
    VisualTokenLatentWorldModel,
)
from starVLA.model.modules.world_model.smooth_spatial_latent_world_model import (
    SmoothSpatialLatentWorldModel,
)
from starVLA.model.modules.world_model.latent_progress import (
    LatentGoalPredictor,
    LatentProgressChecker,
    ProgressActionConditioner,
    latent_goal_loss,
    progress_ranking_loss,
)
from starVLA.model.modules.world_model.wala_transition_auxiliary import (
    WALAVisualTransitionAuxiliary,
    token_cosine_loss,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def prefix_l1_loss(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    prefix_lengths: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Average L1 loss over a per-sample action prefix."""
    if pred_actions.shape != target_actions.shape or pred_actions.ndim != 3:
        raise ValueError(
            "pred_actions and target_actions must have matching (B, H, D) shapes, "
            f"got {tuple(pred_actions.shape)} and {tuple(target_actions.shape)}"
        )
    batch_size, horizon, action_dim = pred_actions.shape
    if prefix_lengths.shape != (batch_size,):
        raise ValueError(
            f"prefix_lengths must have shape {(batch_size,)}, got {tuple(prefix_lengths.shape)}"
        )
    if torch.any(prefix_lengths < 1) or torch.any(prefix_lengths > horizon):
        raise ValueError(f"prefix lengths must be in [1, {horizon}]")

    steps = torch.arange(horizon, device=pred_actions.device).view(1, horizon, 1)
    mask = steps < prefix_lengths.to(device=pred_actions.device).view(batch_size, 1, 1)
    if valid_mask is not None:
        valid_mask = torch.as_tensor(valid_mask, device=pred_actions.device)
        if valid_mask.shape == (batch_size, horizon):
            valid_mask = valid_mask.unsqueeze(-1)
        if valid_mask.shape not in {
            (batch_size, horizon, 1),
            (batch_size, horizon, action_dim),
        }:
            raise ValueError(
                "valid_mask must have shape [B,H], [B,H,1], or [B,H,D], "
                f"got {tuple(valid_mask.shape)}"
            )
        mask = mask & valid_mask.bool()
    return masked_action_l1_loss(pred_actions, target_actions, mask)


class VisualTokenPooler(nn.Module):
    """Create spatially anchored tokens from each view's patch grid.

    Learned pooling queries can all attend to the same salient region and make
    the resulting tokens interchangeable. Here each output token owns a fixed
    ``(view, row, col)`` cell, so token identity survives training.
    """

    def __init__(
        self,
        patch_dim: int,
        token_dim: int,
        num_views: int,
        tokens_per_view: int,
    ) -> None:
        super().__init__()
        self.num_views = int(num_views)
        self.tokens_per_view = int(tokens_per_view)
        self.token_dim = int(token_dim)
        self.grid_size = math.isqrt(self.tokens_per_view)
        if self.grid_size * self.grid_size != self.tokens_per_view:
            raise ValueError(
                "visual_tokens_per_view must be a perfect square for spatial grid pooling, "
                f"got {self.tokens_per_view}"
            )
        self.num_tokens = self.num_views * self.tokens_per_view

        self.patch_norm = nn.LayerNorm(patch_dim)
        self.patch_proj = nn.Linear(patch_dim, token_dim)
        self.view_embedding = nn.Embedding(self.num_views, token_dim)
        self.row_embedding = nn.Embedding(self.grid_size, token_dim)
        self.col_embedding = nn.Embedding(self.grid_size, token_dim)
        self.out_norm = nn.LayerNorm(token_dim)
        nn.init.normal_(self.view_embedding.weight, std=0.02)
        nn.init.normal_(self.row_embedding.weight, std=0.02)
        nn.init.normal_(self.col_embedding.weight, std=0.02)

    def position_tokens(self) -> torch.Tensor:
        """Return flattened static ``(view, row, col)`` embeddings."""
        device = self.view_embedding.weight.device
        view_ids = torch.arange(self.num_views, device=device)
        row_ids = torch.arange(self.grid_size, device=device)
        col_ids = torch.arange(self.grid_size, device=device)
        position = self.view_embedding(view_ids).view(
            self.num_views, 1, 1, self.token_dim
        )
        position = position + self.row_embedding(row_ids).view(
            1, self.grid_size, 1, self.token_dim
        )
        position = position + self.col_embedding(col_ids).view(
            1, 1, self.grid_size, self.token_dim
        )
        return position.reshape(self.num_tokens, self.token_dim)

    def _token_mask(
        self, view_valid_mask: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        if view_valid_mask.ndim != 2 or view_valid_mask.shape[1] != self.num_views:
            raise ValueError(
                "view_valid_mask must have shape [B,V] with "
                f"V={self.num_views}, got {tuple(view_valid_mask.shape)}"
            )
        return view_valid_mask.to(dtype=dtype).repeat_interleave(
            self.tokens_per_view, dim=1
        )[:, None, :, None]

    def remove_position(
        self,
        tokens: torch.Tensor,
        view_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        position = self.position_tokens().to(dtype=tokens.dtype)
        content = tokens - position.view(1, 1, self.num_tokens, self.token_dim)
        if view_valid_mask is not None:
            content = content * self._token_mask(
                view_valid_mask.to(device=tokens.device), dtype=tokens.dtype
            )
        return content

    def forward(
        self,
        patches: torch.Tensor,
        return_content: bool = False,
        view_valid_mask: Optional[torch.Tensor] = None,
    ):
        # patches: (B, T, V, N, D_patch)
        B, T, V, N, D = patches.shape
        if V != self.num_views:
            raise ValueError(f"expected {self.num_views} views, got {V}")
        patch_grid = math.isqrt(N)
        if patch_grid * patch_grid != N:
            raise ValueError(f"expected a square patch grid, got {N} patch tokens")

        x = self.patch_norm(patches)
        x = x.reshape(B * T * V, patch_grid, patch_grid, D).permute(0, 3, 1, 2)
        # Dense mode keeps every DINO patch. Avoid even an identity pooling op
        # so the 14x14 experiment is explicitly unpooled.
        if self.grid_size != patch_grid:
            x = F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))
        x = x.permute(0, 2, 3, 1).reshape(
            B, T, V, self.grid_size, self.grid_size, D
        )
        content = self.out_norm(self.patch_proj(x)).reshape(
            B, T, self.num_tokens, self.token_dim
        )
        position = self.position_tokens().to(dtype=content.dtype)
        tokens = content + position.view(1, 1, self.num_tokens, self.token_dim)
        if view_valid_mask is not None:
            token_mask = self._token_mask(
                view_valid_mask.to(device=tokens.device), dtype=tokens.dtype
            )
            tokens = tokens * token_mask
            content = content * token_mask
        if return_content:
            return tokens, content
        return tokens


class VisualActionCrossAttn(nn.Module):
    """Read the world model's prediction into the OFT action head.

    ``chunk_len`` learnable action queries cross-attend the flattened latent
    tokens ``[current latent, WM-predicted future latents]`` (T*K tokens), so
    the head sees every visual token and can distinguish current vs. future
    frames -- instead of collapsing all tokens into a single mean vector.
    """

    def __init__(
        self,
        token_dim: int,
        action_hidden_dim: int,
        chunk_len: int,
        num_frames: int,
        num_tokens: int,
        num_heads: int = 8,
        state_dim: int = 0,
        state_hidden_dim: int = 256,
        state_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.chunk_len = int(chunk_len)
        self.action_hidden_dim = int(action_hidden_dim)
        self.num_frames = int(num_frames)
        self.num_tokens = int(num_tokens)
        self.state_dim = int(state_dim)
        self.state_dropout = float(state_dropout)
        if not 0.0 <= self.state_dropout < 1.0:
            raise ValueError(f"state_dropout must be in [0, 1), got {self.state_dropout}")

        self.action_queries = nn.Parameter(torch.randn(self.chunk_len, action_hidden_dim) * 0.02)
        self.kv_proj = nn.Linear(token_dim, action_hidden_dim)
        self.frame_embedding = nn.Embedding(self.num_frames, action_hidden_dim)
        self.token_embedding = nn.Embedding(self.num_tokens, action_hidden_dim)
        self.query_norm = nn.LayerNorm(action_hidden_dim)
        self.kv_norm = nn.LayerNorm(action_hidden_dim)
        self.cross_attn = nn.MultiheadAttention(action_hidden_dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(action_hidden_dim)
        if self.state_dim > 0:
            self.state_encoder = nn.Sequential(
                nn.LayerNorm(self.state_dim),
                nn.Linear(self.state_dim, int(state_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(state_hidden_dim), self.chunk_len * action_hidden_dim),
            )
            nn.init.zeros_(self.state_encoder[-1].weight)
            nn.init.zeros_(self.state_encoder[-1].bias)
        else:
            self.state_encoder = None

    def forward(
        self,
        head_tokens: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # head_tokens: (B, T, K, C) == [current latent, predicted future latents]
        B, T, K, _ = head_tokens.shape
        if T > self.num_frames or K != self.num_tokens:
            raise ValueError(
                f"visual action head expected T<={self.num_frames}, K={self.num_tokens}; "
                f"got T={T}, K={K}"
            )
        kv = self.kv_proj(head_tokens)  # (B, T, K, H)
        frame_ids = torch.arange(T, device=head_tokens.device)
        token_ids = torch.arange(K, device=head_tokens.device)
        kv = kv + self.frame_embedding(frame_ids).view(1, T, 1, self.action_hidden_dim)
        kv = kv + self.token_embedding(token_ids).view(1, 1, K, self.action_hidden_dim)
        kv = self.kv_norm(kv.reshape(B, T * K, self.action_hidden_dim))
        q = self.action_queries.to(kv.dtype).unsqueeze(0).expand(B, -1, -1)
        q = self.query_norm(q)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        out = self.out_norm(attn_out + q)
        if self.state_encoder is not None:
            if state is None:
                raise ValueError("state-conditioned visual action head requires current state")
            if state.ndim != 2 or state.shape != (B, self.state_dim):
                raise ValueError(
                    f"expected current state shape {(B, self.state_dim)}, got {tuple(state.shape)}"
                )
            state = state.to(device=out.device, dtype=out.dtype)
            if self.training and self.state_dropout > 0:
                keep = torch.rand(B, 1, device=state.device) >= self.state_dropout
                state = state * keep.to(state.dtype)
            state_residual = self.state_encoder(state).view(
                B, self.chunk_len, self.action_hidden_dim
            )
            out = out + state_residual
        return out  # (B, chunk_len, action_hidden_dim)


class CompositionalTextEncoder(nn.Module):
    """Encode instructions without assigning an opaque ID to each sentence.

    UTF-8 bytes provide a deterministic, collision-free vocabulary for every
    language and allow related instructions to share parameters. A compact
    Transformer composes those tokens into the task vector consumed by the
    latent predictor and action path.
    """

    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    BYTE_OFFSET = 3
    VOCAB_SIZE = BYTE_OFFSET + 256

    def __init__(
        self,
        *,
        output_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 512,
        max_length: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_length = int(max_length)
        if self.max_length < 2:
            raise ValueError("text encoder max_length must be at least 2")
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError(
                "text encoder hidden_dim must be divisible by num_heads, got "
                f"{self.hidden_dim} and {num_heads}"
            )

        self.token_embedding = nn.Embedding(
            self.VOCAB_SIZE, self.hidden_dim, padding_idx=self.PAD_TOKEN
        )
        self.position_embedding = nn.Embedding(self.max_length, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(depth), enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)

    @staticmethod
    def normalize(instruction: Optional[str]) -> str:
        text = unicodedata.normalize("NFKC", instruction or "")
        return " ".join(text.strip().lower().split())

    def tokenize(
        self, instructions: List[str], *, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.full(
            (len(instructions), self.max_length),
            self.PAD_TOKEN,
            device=device,
            dtype=torch.long,
        )
        for row, instruction in enumerate(instructions):
            byte_values = list(self.normalize(instruction).encode("utf-8"))
            byte_values = byte_values[: self.max_length - 2]
            values = (
                [self.BOS_TOKEN]
                + [value + self.BYTE_OFFSET for value in byte_values]
                + [self.EOS_TOKEN]
            )
            token_ids[row, : len(values)] = torch.tensor(
                values, device=device, dtype=torch.long
            )
        valid_mask = token_ids.ne(self.PAD_TOKEN)
        return token_ids, valid_mask

    def forward(self, instructions: List[str], *, device: torch.device) -> torch.Tensor:
        token_ids, valid_mask = self.tokenize(instructions, device=device)
        positions = torch.arange(self.max_length, device=device).unsqueeze(0)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~valid_mask)
        pooled = (hidden * valid_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.out_proj(self.out_norm(pooled))


@dataclass
class LeWMOFTDefaultConfig:
    """LeWM-OFT default parameters."""

    name: str = "LeWMOFT"

    # === World Model backbone (LeWM ViT encoder) ===
    world_model: dict = field(
        default_factory=lambda: {
            # ``base_wm`` is required and has no default: every launcher sets it
            # explicitly (DINOv2/DINOv3 .pth, TAESD, Qwen3-VL). The old
            # vit-tiny default / qwenvl.base_vlm fallback was removed.
            "train_encoder": False,  # frozen ViT by default; flip for joint finetune
            "num_views": 2,          # camera views per frame (e.g. primary + wrist)
            "n_future": 2,            # number of future latents to predict
            "ctx_len": 1,             # clean context frames (current frame only)
            "future_time_offsets_s": None,
            "world_model_only": False,
            "freeze_latent_stats": False,
            # Predictable per-cell 4x4 spatial tokens with direct future-token
            # L2, content-only SIGReg, and explicit temporal smoothness.
            "smooth_latent_enabled": False,
            "smooth_action_enabled": False,
            "smooth_action_loss_weight": 1.0,
            "smooth_world_model_loss_weight": 1.0,
            "smooth_latent_grid_size": 4,
            "smooth_spatial_token_dim": 384,
            "smooth_latent_dim": 384,
            "smooth_predictor_dim": 384,
            "smooth_predictor_depth": 4,
            "smooth_predictor_heads": 6,
            "smooth_predictor_ffn": 1024,
            "smooth_prediction_weight": 1.0,
            "smooth_sigreg_weight": 0.02,
            "smooth_slow_weight": 0.05,
            "smooth_acceleration_weight": 0.10,
            "smooth_temporal_order_weight": 0.05,
            "smooth_temporal_order_margin": 0.10,
            "smooth_sigreg_knots": 17,
            "smooth_sigreg_num_proj": 1024,
            "loss_latent_weight": 1.0,
            "latent_cosine_weight": 0.0,
            # Detach the world-model context/anchor so the latent loss trains
            # only the predictor. Otherwise the encoder/pooler can lower the
            # loss by erasing temporal variation (current == future collapse).
            "detach_wm_input": False,
            "latent_stats_momentum": 0.99,
            # Multi-step rollout supervision. ``rollout_steps=1`` reproduces the
            # validated single-shot objective exactly; higher values re-anchor
            # the predictor on its own output and need a dataset that provides
            # ``n_future * rollout_steps`` future frames.
            "rollout_steps": 1,
            "loss_rollout_weight": 0.0,
            "predictor_state_dim": 0,
            "predictor_state_history": 1,
            "predictor_state_current_index": 0,
            "residual_predictor_dim": 384,
            "residual_predictor_depth": 4,
            "residual_predictor_heads": 6,
            "residual_predictor_ffn": 1024,
            "residual_predictor_sigreg_weight": 0.0,

            # Optional residual booster that consumes longer causal visual and
            # proprio histories while preserving the warm-start prediction.
            "context_correction_depth": 0,
            "context_correction_state_dim": 0,
            "context_correction_dim": 384,
            "context_correction_heads": 6,
            "context_correction_ffn": 1024,
            "context_correction_freeze_base": False,
            # Spatial tokens are fixed grid cells, not learned pooling queries.
            # ``num_visual_tokens`` remains a legacy alias for per-view count.
            "visual_tokens_per_view": 16,
            "num_visual_tokens": 16,
            "visual_token_dim": None,
            "visual_token_diversity_weight": 0.02,
            "visual_token_variance_weight": 0.02,
            "visual_token_min_std": 0.1,
            "visual_diagnostics": True,

            # Add a zero-initialized proprio residual to each action query.
            "use_state_cond": False,
            "state_cond_dim": 8,
            "state_cond_hidden_dim": 256,
            "state_cond_dropout": 0.1,
            "state_cond_only": False,

            # === Optional state probe (align latents to future proprio) ===
            "use_state_probe": False,
            "state_dim": 8,
            "loss_state_weight": 0.5,

            # === Optional start/current/goal latent progress checker ===
            # Training additionally requires datasets.vla_data.include_progress.
            # The goal predictor prevents true terminal frames leaking into
            # deployed action conditioning.
            "use_progress_checker": False,
            "progress_hidden_dim": 256,
            "progress_action_hidden_dim": 128,
            "progress_action_dropout": 0.05,
            "progress_detach_latents": True,
            "progress_detach_action": True,
            "progress_loss_weight": 0.2,
            "progress_anchor_weight": 0.5,
            "progress_ranking_weight": 0.1,
            "progress_goal_weight": 0.2,
            "progress_ema": 0.8,

            # === Optional WALA-style future-transition teacher ===
            # Disabled by default, so existing checkpoints and inference are
            # unchanged. ``combined`` trains teacher, student, and deployed
            # action modules together in one run. The legacy staged modes
            # teacher -> student -> joint remain available for ablations.
            "transition_mode": "off",
            "transition_hidden_dim": 384,
            "transition_num_tokens": 8,
            "transition_encoder_depth": 2,
            "transition_decoder_depth": 2,
            "transition_resampler_depth": 2,
            "transition_heads": 6,
            "transition_teacher_recon_weight": 1.0,
            "transition_alignment_weight": 0.005,
            "transition_decode_weight": 0.05,
            "transition_cosine_weight": 0.1,
            "transition_alignment_l1_weight": 0.1,
            "transition_detach_action_queries": False,
            "transition_joint_freeze_base": True,
        }
    )

    # === Action shape config ===
    action_model: dict = field(
        default_factory=lambda: {
            # ``MLP`` preserves legacy checkpoints. New recipes should use
            # ``ACT`` for the TurboVLA-style multi-layer action decoder.
            "action_model_type": "MLP",
            "action_dim": 7,
            "action_hidden_dim": 384,
            "act_num_heads": 8,
            "act_num_layers": 3,
            "act_dim_feedforward": 2048,
            "act_mlp_hidden_dim": 512,
            "act_dropout": 0.1,
            "act_num_state_tokens": 2,
            "future_action_window_size": 8,
            "past_action_window_size": 0,
            # Optional auxiliary objective for policies deployed with shorter
            # replanning horizons. Empty choices or zero weight preserves the
            # original full-chunk L1 objective exactly.
            "random_execution_horizons": [1, 2, 4, 8],
            "random_prefix_loss_weight": 0.0,
            # Optional embodiment-tag -> native ACT shape map.  When set, a
            # homogeneous batch is routed to one independent ACT head while
            # the visual/language/world-model trunk remains shared.
            "embodiment_heads": {},
        }
    )

    # === Language / task conditioning ===
    # Legacy runs use a whole-sentence hash table. New runs may select a compact
    # compositional text encoder. ``embed_dim: null`` -> world-model hidden size.
    lang_cond: dict = field(
        default_factory=lambda: {
            # ``hash`` preserves legacy checkpoints. New training recipes
            # should explicitly select ``text`` for compositional language
            # conditioning instead of treating every full sentence as an ID.
            "type": "hash",
            "num_buckets": 4096,
            "embed_dim": None,
            "text_hidden_dim": 256,
            "text_depth": 2,
            "text_heads": 4,
            "text_ffn_dim": 512,
            "text_max_length": 128,
            "text_dropout": 0.1,
        }
    )


@FRAMEWORK_REGISTRY.register("LeWMOFT")
class LeWM_OFT(baseframework):
    """LeWM visual encoder + deterministic latent predictor + OFT action head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(LeWMOFTDefaultConfig, config)

        wm_cfg = self.config.framework.get("world_model", {}) or {}
        enabled_removed_branches = [
            option
            for option in (
                "reconstructive_latent_enabled",
                "predictable_innovation_enabled",
                "use_dense_patch_action",
            )
            if bool(wm_cfg.get(option, False))
        ]
        if enabled_removed_branches:
            raise ValueError(
                "removed experimental branches are enabled in world_model: "
                + ", ".join(enabled_removed_branches)
            )

        self.backbone = get_world_model(config=self.config)

        wm_hidden = self.backbone.model.config.hidden_size
        self.num_views = int(wm_cfg.get("num_views", 2))

        self.use_state_cond = bool(wm_cfg.get("use_state_cond", False))
        # The deployment wrapper normalizes state only when this flag is set.
        # Cover every state consumer: the action head (use_state_cond) and the
        # world-model state paths (predictor_state_dim /
        # context_correction_state_dim). Predictor state history is read below
        # via wm_cfg, so the raw dict values are sufficient here.
        self.expects_normalized_state = (
            self.use_state_cond
            or int(wm_cfg.get("predictor_state_dim", 0)) > 0
            or int(wm_cfg.get("context_correction_state_dim", 0)) > 0
        )
        visual_token_dim_cfg = wm_cfg.get("visual_token_dim", None)
        self.visual_token_dim = int(visual_token_dim_cfg) if visual_token_dim_cfg else wm_hidden

        # `action_horizon` is the single source of truth for chunk length;
        # legacy aliases are normalised upstream by share_tools.apply_config_compat.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon
        prefix_choices = self.config.framework.action_model.get(
            "random_execution_horizons", []
        )
        self.random_execution_horizons = tuple(int(value) for value in prefix_choices)
        self.random_prefix_loss_weight = float(
            self.config.framework.action_model.get("random_prefix_loss_weight", 0.0)
        )
        if self.random_prefix_loss_weight < 0:
            raise ValueError("random_prefix_loss_weight must be non-negative")
        if any(
            horizon < 1 or horizon > self.action_horizon
            for horizon in self.random_execution_horizons
        ):
            raise ValueError(
                "random_execution_horizons must be between 1 and action_horizon "
                f"({self.action_horizon}), got {self.random_execution_horizons}"
            )

        configured_action_hidden = self.config.framework.action_model.get(
            "action_hidden_dim", None
        )
        self.action_hidden_dim = int(configured_action_hidden or wm_hidden)
        self.config.framework.action_model.action_hidden_dim = self.action_hidden_dim
        action_model_type = str(
            self.config.framework.action_model.get("action_model_type", "MLP")
        ).strip().upper()
        action_model_aliases = {
            "MLP": "MLP",
            "ACT": "ACT",
            "TURBO_ACT": "ACT",
            "TURBOVLA_ACT": "ACT",
        }
        if action_model_type not in action_model_aliases:
            raise ValueError(
                "LeWMOFT action_model_type must be MLP or ACT, got "
                f"{action_model_type!r}"
            )
        self.action_model_type = action_model_aliases[action_model_type]
        raw_embodiment_heads = self.config.framework.action_model.get(
            "embodiment_heads", {}
        ) or {}
        self.embodiment_head_specs = {
            str(tag): {str(key): value for key, value in spec.items()}
            for tag, spec in raw_embodiment_heads.items()
        }
        self.multi_embodiment_actions = bool(self.embodiment_head_specs)
        if self.multi_embodiment_actions and self.action_model_type != "ACT":
            raise ValueError(
                "action_model.embodiment_heads currently requires action_model_type=ACT"
            )
        # ACT needs the final visual-token layout, so it is constructed below.
        self.action_model = (
            get_action_model(config=self.config)
            if self.action_model_type == "MLP"
            else None
        )

        self.l1_loss = nn.L1Loss()

        # === Language / task conditioning ===
        # Legacy checkpoints use a whole-sentence hash embedding. New recipes
        # can use a compact compositional text encoder so related instructions
        # share token parameters and arbitrary text has no bucket collisions.
        lang_cfg = self.config.framework.get("lang_cond", {}) or {}
        self.lang_cond_type = str(lang_cfg.get("type", "hash")).strip().lower()
        if self.lang_cond_type not in {"hash", "text"}:
            raise ValueError(
                "framework.lang_cond.type must be 'hash' or 'text', got "
                f"{self.lang_cond_type!r}"
            )
        self.num_task_buckets = int(lang_cfg.get("num_buckets", 4096))
        _emb_dim = lang_cfg.get("embed_dim", None)
        self.task_emb_dim = int(_emb_dim) if _emb_dim else wm_hidden
        if self.lang_cond_type == "hash":
            self.task_embedding = nn.Embedding(
                self.num_task_buckets, self.task_emb_dim
            )
        else:
            self.task_embedding = CompositionalTextEncoder(
                output_dim=self.task_emb_dim,
                hidden_dim=int(lang_cfg.get("text_hidden_dim", 256)),
                depth=int(lang_cfg.get("text_depth", 2)),
                num_heads=int(lang_cfg.get("text_heads", 4)),
                ffn_dim=int(lang_cfg.get("text_ffn_dim", 512)),
                max_length=int(lang_cfg.get("text_max_length", 128)),
                dropout=float(lang_cfg.get("text_dropout", 0.1)),
            )
        self.embodiment_tags = tuple(sorted(self.embodiment_head_specs))
        self.embodiment_tag_to_index = {
            tag: index for index, tag in enumerate(self.embodiment_tags)
        }
        self.embodiment_embedding = (
            nn.Embedding(len(self.embodiment_tags), self.task_emb_dim)
            if self.multi_embodiment_actions
            else None
        )
        if self.embodiment_embedding is not None:
            nn.init.normal_(self.embodiment_embedding.weight, std=0.02)

        self.n_future = int(wm_cfg.get("n_future", 2))
        self.wm_ctx_len = int(wm_cfg.get("ctx_len", 1))
        configured_time_offsets = wm_cfg.get("future_time_offsets_s", None)
        self.future_time_offsets_s = (
            tuple(float(value) for value in configured_time_offsets)
            if configured_time_offsets is not None
            else None
        )
        if self.future_time_offsets_s is not None and len(
            self.future_time_offsets_s
        ) != 1 + self.n_future:
            raise ValueError(
                "world_model.future_time_offsets_s must contain current plus "
                f"n_future={self.n_future} offsets"
            )
        self.world_model_only = bool(wm_cfg.get("world_model_only", False))
        self.smooth_latent_enabled = bool(
            wm_cfg.get("smooth_latent_enabled", False)
        )
        self.smooth_action_enabled = bool(
            wm_cfg.get("smooth_action_enabled", False)
        )
        self.smooth_action_loss_weight = float(
            wm_cfg.get("smooth_action_loss_weight", 1.0)
        )
        self.smooth_world_model_loss_weight = float(
            wm_cfg.get("smooth_world_model_loss_weight", 1.0)
        )
        if min(
            self.smooth_action_loss_weight,
            self.smooth_world_model_loss_weight,
        ) < 0:
            raise ValueError("smooth joint loss weights must be non-negative")
        if self.smooth_action_enabled and not self.smooth_latent_enabled:
            raise ValueError(
                "smooth_action_enabled requires smooth_latent_enabled=true"
            )
        self.freeze_latent_stats = bool(wm_cfg.get("freeze_latent_stats", False))
        self.predictor_state_dim = int(wm_cfg.get("predictor_state_dim", 0))
        self.predictor_state_history = int(wm_cfg.get("predictor_state_history", 1))
        self.predictor_state_current_index = int(
            wm_cfg.get("predictor_state_current_index", 0)
        )
        self.context_correction_state_dim = int(
            wm_cfg.get("context_correction_state_dim", 0)
        )
        self.context_correction_freeze_base = bool(
            wm_cfg.get("context_correction_freeze_base", False)
        )
        if self.predictor_state_history < 1:
            raise ValueError("predictor_state_history must be at least 1")
        self.loss_latent_weight = float(
            wm_cfg.get("loss_delta_weight", wm_cfg.get("loss_latent_weight", 1.0))
        )
        self.latent_cosine_weight = float(wm_cfg.get("latent_cosine_weight", 0.0))
        self.rollout_steps = int(wm_cfg.get("rollout_steps", 1))
        self.loss_rollout_weight = float(wm_cfg.get("loss_rollout_weight", 0.0))
        if self.rollout_steps < 1:
            raise ValueError("world_model.rollout_steps must be at least 1")
        if self.rollout_steps == 1 and self.loss_rollout_weight > 0:
            raise ValueError(
                "world_model.loss_rollout_weight requires rollout_steps >= 2"
            )
        # The deployed action path always consumes exactly ``n_future``
        # predictions, so extra rollout frames stay inside the world model and
        # never change the action-head token layout or checkpoint shapes.
        if self.smooth_latent_enabled:
            if self.smooth_action_enabled and self.world_model_only:
                raise ValueError(
                    "smooth_action_enabled requires world_model_only=false"
                )
            if not self.smooth_action_enabled and not self.world_model_only:
                raise ValueError(
                    "a representation-only smooth latent run requires "
                    "world_model_only=true"
                )
            # Joint encoder finetuning is allowed for the smooth path when the
            # run explicitly opts in (train_encoder=true); the base coordinate
            # system then adapts jointly with the projector/predictor.
            smooth_train_encoder = bool(wm_cfg.get("train_encoder", False))
            if smooth_train_encoder:
                logger.warning(
                    "smooth latent training with train_encoder=true: DINO "
                    "backbone will be finetuned jointly with the smooth "
                    "world model and action head"
                )
            if self.wm_ctx_len != 1 or self.n_future != 2:
                raise ValueError(
                    "smooth latent training requires ctx_len=1 and n_future=2"
                )
            # The smooth world-model path itself stays vision-only, but the
            # action head may consume normalized proprio state when the run
            # opts in via use_state_cond (mirrors the classic path).
            if (
                self.predictor_state_dim > 0
                or self.context_correction_state_dim > 0
                or bool(wm_cfg.get("use_state_probe", False))
            ):
                raise ValueError(
                    "smooth latent world-model state consumption requires "
                    "use_state_cond; predictor_state_dim / "
                    "context_correction_state_dim / use_state_probe must be 0 "
                    "for the smooth latent path"
                )
            smooth_transition_mode = wm_cfg.get("transition_mode", "off")
            smooth_transition_enabled = (
                smooth_transition_mode is not False
                and str(smooth_transition_mode).lower() != "off"
            )
            if self.smooth_action_enabled and (
                bool(wm_cfg.get("use_progress_checker", False))
                or smooth_transition_enabled
            ):
                raise ValueError(
                    "smooth action training accepts only current/predicted global "
                    "latents; progress and transition auxiliaries must be disabled"
                )
        self.loss_sigreg_weight = float(
            wm_cfg.get(
                "residual_predictor_sigreg_weight",
                wm_cfg.get("delta_head_sigreg_weight", 0.0),
            )
        )
        self.detach_wm_input = bool(wm_cfg.get("detach_wm_input", False))

        # DINOv3 reports its patch width through the HF encoder config; the
        # TAESD / Qwen vision interfaces expose it directly.
        patch_dim = getattr(self.backbone, "patch_feature_dim", None) or getattr(
            self.backbone, "feature_dim", None
        )
        if patch_dim is None:
            patch_dim = self.backbone.encoder.config.hidden_size
        patch_dim = int(patch_dim)
        self.visual_tokens_per_view = int(
            wm_cfg.get("visual_tokens_per_view", wm_cfg.get("num_visual_tokens", 16))
        )
        self.visual_token_pooler = VisualTokenPooler(
            patch_dim=patch_dim,
            token_dim=self.visual_token_dim,
            num_views=self.num_views,
            tokens_per_view=self.visual_tokens_per_view,
        )
        self.num_visual_tokens = self.visual_token_pooler.num_tokens
        self.visual_token_diversity_weight = float(
            wm_cfg.get("visual_token_diversity_weight", 0.02)
        )
        self.visual_token_variance_weight = float(
            wm_cfg.get("visual_token_variance_weight", 0.02)
        )
        self.visual_token_min_std = float(wm_cfg.get("visual_token_min_std", 0.1))
        self.visual_diagnostics = bool(wm_cfg.get("visual_diagnostics", True))
        self.world_model = VisualTokenLatentWorldModel(
            latent_dim=self.visual_token_dim,
            goal_dim=self.task_emb_dim,
            n_future=self.n_future,
            num_tokens=self.num_visual_tokens,
            context_len=self.wm_ctx_len,
            state_dim=self.predictor_state_dim,
            dim=int(
                wm_cfg.get("residual_predictor_dim", wm_cfg.get("delta_head_dim", 384))
            ),
            depth=int(
                wm_cfg.get("residual_predictor_depth", wm_cfg.get("delta_head_depth", 4))
            ),
            num_heads=int(
                wm_cfg.get("residual_predictor_heads", wm_cfg.get("delta_head_heads", 6))
            ),
            ffn_dim=int(
                wm_cfg.get("residual_predictor_ffn", wm_cfg.get("delta_head_ffn", 1024))
            ),
            context_correction_depth=int(wm_cfg.get("context_correction_depth", 0)),
            context_correction_state_dim=self.context_correction_state_dim,
            context_correction_dim=int(wm_cfg.get("context_correction_dim", 384)),
            context_correction_heads=int(wm_cfg.get("context_correction_heads", 6)),
            context_correction_ffn_dim=int(wm_cfg.get("context_correction_ffn", 1024)),
            sigreg_weight=self.loss_sigreg_weight,
            stats_momentum=float(wm_cfg.get("latent_stats_momentum", 0.99)),
            detach_input=self.detach_wm_input,
        )
        self.smooth_world_model = (
            SmoothSpatialLatentWorldModel(
                patch_dim=patch_dim,
                num_views=self.num_views,
                goal_dim=self.task_emb_dim,
                latent_dim=int(wm_cfg.get("smooth_latent_dim", 384)),
                spatial_token_dim=int(
                    wm_cfg.get("smooth_spatial_token_dim", 384)
                ),
                grid_size=int(wm_cfg.get("smooth_latent_grid_size", 4)),
                n_future=self.n_future,
                rollout_steps=self.rollout_steps,
                rollout_weight=self.loss_rollout_weight,
                predictor_dim=int(wm_cfg.get("smooth_predictor_dim", 384)),
                predictor_depth=int(wm_cfg.get("smooth_predictor_depth", 4)),
                predictor_heads=int(wm_cfg.get("smooth_predictor_heads", 6)),
                predictor_ffn_dim=int(
                    wm_cfg.get("smooth_predictor_ffn", 1024)
                ),
                prediction_weight=float(
                    wm_cfg.get("smooth_prediction_weight", 1.0)
                ),
                sigreg_weight=float(wm_cfg.get("smooth_sigreg_weight", 0.02)),
                slow_weight=float(wm_cfg.get("smooth_slow_weight", 0.05)),
                acceleration_weight=float(
                    wm_cfg.get("smooth_acceleration_weight", 0.10)
                ),
                temporal_order_weight=float(
                    wm_cfg.get("smooth_temporal_order_weight", 0.05)
                ),
                temporal_order_margin=float(
                    wm_cfg.get("smooth_temporal_order_margin", 0.10)
                ),
                sigreg_knots=int(wm_cfg.get("smooth_sigreg_knots", 17)),
                sigreg_num_proj=int(
                    wm_cfg.get("smooth_sigreg_num_proj", 1024)
                ),
                detach_input=not bool(wm_cfg.get("train_encoder", False)),
            )
            if self.smooth_latent_enabled
            else None
        )
        action_visual_token_dim = (
            int(wm_cfg.get("smooth_latent_dim", 384))
            if self.smooth_action_enabled
            else self.visual_token_dim
        )
        action_visual_num_tokens = (
            self.smooth_world_model.projector.num_tokens
            if self.smooth_action_enabled
            else self.num_visual_tokens
        )
        action_cfg = self.config.framework.action_model
        self.action_models = nn.ModuleDict()
        if self.multi_embodiment_actions:
            self.visual_action_head = None
            self.action_model = None
            for tag, spec in self.embodiment_head_specs.items():
                self.action_models[tag] = TurboStyleACTActionHead(
                    token_dim=action_visual_token_dim,
                    hidden_dim=self.action_hidden_dim,
                    action_dim=int(spec["action_dim"]),
                    horizon=int(spec["action_horizon"]),
                    num_frames=self.wm_ctx_len + self.n_future,
                    num_visual_tokens=action_visual_num_tokens,
                    num_heads=int(action_cfg.get("act_num_heads", 8)),
                    num_layers=int(action_cfg.get("act_num_layers", 3)),
                    dim_feedforward=int(action_cfg.get("act_dim_feedforward", 2048)),
                    mlp_hidden_dim=int(action_cfg.get("act_mlp_hidden_dim", 512)),
                    dropout=float(action_cfg.get("act_dropout", 0.1)),
                    state_dim=(int(spec.get("state_dim", 0)) if self.use_state_cond else 0),
                    state_hidden_dim=int(wm_cfg.get("state_cond_hidden_dim", 256)),
                    num_state_tokens=int(action_cfg.get("act_num_state_tokens", 2)),
                )
        elif self.action_model_type == "ACT":
            self.visual_action_head = None
            self.action_model = TurboStyleACTActionHead(
                token_dim=action_visual_token_dim,
                hidden_dim=self.action_hidden_dim,
                action_dim=int(action_cfg.action_dim),
                horizon=self.chunk_len,
                num_frames=self.wm_ctx_len + self.n_future,
                num_visual_tokens=action_visual_num_tokens,
                num_heads=int(action_cfg.get("act_num_heads", 8)),
                num_layers=int(action_cfg.get("act_num_layers", 3)),
                dim_feedforward=int(action_cfg.get("act_dim_feedforward", 2048)),
                mlp_hidden_dim=int(action_cfg.get("act_mlp_hidden_dim", 512)),
                dropout=float(action_cfg.get("act_dropout", 0.1)),
                state_dim=(
                    int(wm_cfg.get("state_cond_dim", 8))
                    if self.use_state_cond
                    else 0
                ),
                state_hidden_dim=int(wm_cfg.get("state_cond_hidden_dim", 256)),
                num_state_tokens=int(action_cfg.get("act_num_state_tokens", 2)),
            )
        else:
            self.visual_action_head = VisualActionCrossAttn(
                token_dim=action_visual_token_dim,
                action_hidden_dim=self.action_hidden_dim,
                chunk_len=self.chunk_len,
                num_frames=self.wm_ctx_len + self.n_future,
                num_tokens=action_visual_num_tokens,
                num_heads=int(wm_cfg.get("visual_action_heads", 8)),
                state_dim=(
                    int(wm_cfg.get("state_cond_dim", 8))
                    if self.use_state_cond
                    else 0
                ),
                state_hidden_dim=int(wm_cfg.get("state_cond_hidden_dim", 256)),
                state_dropout=float(wm_cfg.get("state_cond_dropout", 0.1)),
            )
        self.use_progress_checker = bool(wm_cfg.get("use_progress_checker", False))
        if self.multi_embodiment_actions and self.use_progress_checker:
            raise ValueError(
                "multi-embodiment ACT heads do not yet support use_progress_checker"
            )
        self.progress_goal_predictor = None
        self.progress_checker = None
        self.progress_action_conditioner = None
        if self.use_progress_checker:
            progress_hidden_dim = int(wm_cfg.get("progress_hidden_dim", 256))
            self.progress_goal_predictor = LatentGoalPredictor(
                latent_dim=self.visual_token_dim,
                task_dim=self.task_emb_dim,
                hidden_dim=progress_hidden_dim,
                num_tokens=self.num_visual_tokens,
            )
            self.progress_checker = LatentProgressChecker(
                latent_dim=self.visual_token_dim,
                hidden_dim=progress_hidden_dim,
            )
            self.progress_action_conditioner = ProgressActionConditioner(
                action_hidden_dim=self.action_hidden_dim,
                chunk_len=self.chunk_len,
                hidden_dim=int(wm_cfg.get("progress_action_hidden_dim", 128)),
                dropout=float(wm_cfg.get("progress_action_dropout", 0.05)),
            )
        self.progress_detach_latents = bool(
            wm_cfg.get("progress_detach_latents", True)
        )
        self.progress_detach_action = bool(
            wm_cfg.get("progress_detach_action", True)
        )
        self.progress_loss_weight = float(wm_cfg.get("progress_loss_weight", 0.2))
        self.progress_anchor_weight = float(
            wm_cfg.get("progress_anchor_weight", 0.5)
        )
        self.progress_ranking_weight = float(
            wm_cfg.get("progress_ranking_weight", 0.1)
        )
        self.progress_goal_weight = float(wm_cfg.get("progress_goal_weight", 0.2))
        self.progress_ema = float(wm_cfg.get("progress_ema", 0.8))
        if not 0.0 <= self.progress_ema < 1.0:
            raise ValueError(
                f"world_model.progress_ema must be in [0, 1), got {self.progress_ema}"
            )
        # Deployment-only controls used for causal closed-loop ablations.
        # These are intentionally not checkpoint parameters.
        self.progress_inference_mode = "learned"
        self.progress_fixed_value = 0.5
        for name, weight in (
            ("progress_loss_weight", self.progress_loss_weight),
            ("progress_anchor_weight", self.progress_anchor_weight),
            ("progress_ranking_weight", self.progress_ranking_weight),
            ("progress_goal_weight", self.progress_goal_weight),
        ):
            if weight < 0:
                raise ValueError(f"{name} must be non-negative")
        self.reset_progress_state()

        transition_mode = wm_cfg.get("transition_mode", "off")
        # OmegaConf's dotlist parser follows YAML boolean aliases and parses
        # the unquoted CLI value `off` as False. Normalize that representation
        # so the ordinary non-transition launcher remains usable.
        self.transition_mode = (
            "off" if transition_mode is False else str(transition_mode).lower()
        )
        if self.multi_embodiment_actions and self.transition_mode != "off":
            raise ValueError(
                "multi-embodiment ACT heads require transition_mode=off"
            )
        if self.transition_mode not in {
            "off",
            "teacher",
            "student",
            "joint",
            "combined",
        }:
            raise ValueError(
                "world_model.transition_mode must be one of "
                "off/teacher/student/joint/combined, "
                f"got {self.transition_mode!r}"
            )
        self.transition_auxiliary = None
        if self.transition_mode != "off":
            transition_hidden_dim = int(wm_cfg.get("transition_hidden_dim", 384))
            self.transition_auxiliary = WALAVisualTransitionAuxiliary(
                latent_dim=self.visual_token_dim,
                action_hidden_dim=self.action_hidden_dim,
                hidden_dim=transition_hidden_dim,
                num_visual_tokens=self.num_visual_tokens,
                num_future=self.n_future,
                num_action_queries=self.chunk_len,
                num_transition_tokens=int(wm_cfg.get("transition_num_tokens", 8)),
                encoder_depth=int(wm_cfg.get("transition_encoder_depth", 2)),
                decoder_depth=int(wm_cfg.get("transition_decoder_depth", 2)),
                resampler_depth=int(wm_cfg.get("transition_resampler_depth", 2)),
                num_heads=int(wm_cfg.get("transition_heads", 6)),
            )
            self.transition_teacher_recon_weight = float(
                wm_cfg.get("transition_teacher_recon_weight", 1.0)
            )
            self.transition_alignment_weight = float(
                wm_cfg.get("transition_alignment_weight", 0.005)
            )
            self.transition_decode_weight = float(
                wm_cfg.get("transition_decode_weight", 0.05)
            )
            self.transition_cosine_weight = float(
                wm_cfg.get("transition_cosine_weight", 0.1)
            )
            self.transition_alignment_l1_weight = float(
                wm_cfg.get("transition_alignment_l1_weight", 0.1)
            )
            self.transition_detach_action_queries = bool(
                wm_cfg.get(
                    "transition_detach_action_queries",
                    self.transition_mode == "student",
                )
            )
            self.transition_joint_freeze_base = bool(
                wm_cfg.get("transition_joint_freeze_base", True)
            )

        self.use_state_probe = bool(wm_cfg.get("use_state_probe", False))
        if self.multi_embodiment_actions and self.use_state_probe:
            raise ValueError(
                "multi-embodiment pretraining requires use_state_probe=false; "
                "state dimensions differ by embodiment"
            )
        if self.use_state_probe:
            self.state_probe_dim = int(wm_cfg.get("state_dim", 8))
            self.loss_state_weight = float(wm_cfg.get("loss_state_weight", 0.5))
            self.state_probe = nn.Sequential(
                nn.Linear(self.visual_token_dim, wm_hidden),
                nn.GELU(),
                nn.Linear(wm_hidden, self.state_probe_dim),
            )
            self.state_loss_fn = nn.MSELoss()

        if self.use_state_cond and bool(wm_cfg.get("state_cond_only", False)):
            self.requires_grad_(False)
            if self.multi_embodiment_actions:
                for action_model in self.action_models.values():
                    action_model.state_projection.requires_grad_(True)
            elif self.action_model_type == "ACT":
                self.action_model.state_projection.requires_grad_(True)
            else:
                self.visual_action_head.state_encoder.requires_grad_(True)

        # Stage isolation is enforced here rather than relying on a long and
        # error-prone freeze_modules string in launch scripts.
        if self.transition_mode == "teacher":
            self.requires_grad_(False)
            self.transition_auxiliary.teacher_encoder.requires_grad_(True)
            self.transition_auxiliary.teacher_decoder.requires_grad_(True)
        elif self.transition_mode == "student":
            self.requires_grad_(False)
            self.transition_auxiliary.student_resampler.requires_grad_(True)
        elif self.transition_mode == "joint":
            self.transition_auxiliary.teacher_encoder.requires_grad_(False)
            self.transition_auxiliary.teacher_decoder.requires_grad_(False)
            self.transition_auxiliary.student_resampler.requires_grad_(True)
            if self.transition_joint_freeze_base:
                # Preserve the validated DINO -> pooled-token -> latent-world
                # coordinate system. Joint training adapts only the deployed
                # action readout/model plus the transition-token student.
                self.backbone.requires_grad_(False)
                self.visual_token_pooler.requires_grad_(False)
                self.world_model.requires_grad_(False)
                self.task_embedding.requires_grad_(False)
                if self.use_state_probe:
                    self.state_probe.requires_grad_(False)
        elif self.transition_mode == "combined":
            # One-run variant: learn the future-aware tokenizer/decoder and
            # the action-query student concurrently while adapting the
            # deployed action readout. Keep the validated visual coordinate
            # system fixed exactly as in the staged joint phase.
            self.transition_auxiliary.teacher_encoder.requires_grad_(True)
            self.transition_auxiliary.teacher_decoder.requires_grad_(True)
            self.transition_auxiliary.student_resampler.requires_grad_(True)
            if self.transition_joint_freeze_base:
                self.backbone.requires_grad_(False)
                self.visual_token_pooler.requires_grad_(False)
                self.world_model.requires_grad_(False)
                self.task_embedding.requires_grad_(False)
                if self.use_state_probe:
                    self.state_probe.requires_grad_(False)

        # Stage-specific freezing above must not accidentally disable an
        # explicitly enabled progress experiment.
        if self.use_progress_checker:
            self.progress_goal_predictor.requires_grad_(True)
            self.progress_checker.requires_grad_(True)
            self.progress_action_conditioner.requires_grad_(True)

        if self.world_model_only:
            if self.transition_mode != "off" or self.use_progress_checker:
                raise ValueError(
                    "world_model_only requires transition_mode=off and "
                    "use_progress_checker=false"
                )
            # Keep the pretrained visual coordinate system fixed. Ordinary
            # world-model-only runs retain their historical trainable set.
            # Innovation runs instead optimize only the new bottleneck: the
            # mature base predictor and its task embedding remain immutable.
            self.requires_grad_(False)
            if self.smooth_latent_enabled:
                if self.smooth_world_model is None:
                    raise RuntimeError(
                        "smooth_latent_enabled requires smooth_world_model"
                    )
                self.smooth_world_model.requires_grad_(True)
                self.task_embedding.requires_grad_(True)
            else:
                self.world_model.requires_grad_(True)
                self.task_embedding.requires_grad_(True)
            if self.embodiment_embedding is not None:
                self.embodiment_embedding.requires_grad_(True)
            if self.context_correction_freeze_base:
                if self.world_model.context_correction is None:
                    raise ValueError(
                        "context_correction_freeze_base requires "
                        "context_correction_depth > 0"
                    )
                self.world_model.residual_predictor.requires_grad_(False)
                self.task_embedding.requires_grad_(False)

        if self.smooth_action_enabled:
            if self.smooth_world_model is None:
                raise RuntimeError(
                    "smooth_action_enabled requires smooth_world_model"
                )
            # This joint policy has exactly one visual path. DINO feeds fixed
            # per-view spatial cells; the world model and action head retain
            # the explicit token grid instead of flattening it globally.
            # When train_encoder=true, the DINO backbone is additionally
            # unfrozen for joint finetuning.
            self.requires_grad_(False)
            self.smooth_world_model.requires_grad_(True)
            self.task_embedding.requires_grad_(True)
            if self.visual_action_head is not None:
                self.visual_action_head.requires_grad_(True)
            if self.multi_embodiment_actions:
                self.action_models.requires_grad_(True)
                self.embodiment_embedding.requires_grad_(True)
            else:
                self.action_model.requires_grad_(True)
            if bool(wm_cfg.get("train_encoder", False)):
                self.backbone.requires_grad_(True)

    def reset_progress_state(self) -> None:
        """Clear episode-local start, goal, and filtered progress caches."""
        self._progress_start_latent = None
        self._progress_goal_latent = None
        self._progress_previous = None
        self._progress_instruction_signature = None

    def configure_progress_inference(
        self,
        *,
        mode: str = "learned",
        fixed_value: float = 0.5,
        ema: Optional[float] = None,
    ) -> None:
        """Configure a weight-preserving progress-conditioning ablation."""
        valid_modes = {"learned", "disabled", "fixed"}
        if mode not in valid_modes:
            raise ValueError(
                f"progress inference mode must be one of {sorted(valid_modes)}, "
                f"got {mode!r}"
            )
        if mode != "disabled" and not self.use_progress_checker:
            raise ValueError(
                f"progress mode {mode!r} requires a checkpoint with progress enabled"
            )
        if not 0.0 <= fixed_value <= 1.0:
            raise ValueError(f"fixed progress must be in [0, 1], got {fixed_value}")
        if ema is not None:
            if not 0.0 <= ema < 1.0:
                raise ValueError(f"progress EMA must be in [0, 1), got {ema}")
            self.progress_ema = float(ema)
        self.progress_inference_mode = mode
        self.progress_fixed_value = float(fixed_value)
        self.reset_progress_state()

    def _condition_action_queries_on_progress(
        self,
        action_queries: torch.Tensor,
        progress: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.progress_action_conditioner is None:
            return action_queries
        if progress is None:
            raise ValueError("progress conditioning is enabled but progress is missing")
        if self.progress_detach_action:
            progress = progress.detach()
        return self.progress_action_conditioner(action_queries, progress)

    def _select_inference_progress_for_action(
        self, learned_progress: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Select the scalar exposed to the conditioner for an A/B mode."""
        if self.progress_inference_mode == "learned":
            return learned_progress
        if self.progress_inference_mode == "fixed":
            return torch.full_like(learned_progress, self.progress_fixed_value)
        if self.progress_inference_mode == "disabled":
            return None
        raise RuntimeError(
            f"unknown progress inference mode {self.progress_inference_mode!r}"
        )

    def _training_progress(
        self,
        *,
        start_latent: torch.Tensor,
        current_latent: torch.Tensor,
        target_goal_latent: torch.Tensor,
        task_embedding: torch.Tensor,
        target: torch.Tensor,
        episode_ids: List[object],
    ) -> dict[str, torch.Tensor]:
        if self.progress_goal_predictor is None or self.progress_checker is None:
            raise RuntimeError("progress checker is disabled")
        if self.progress_detach_latents:
            start_latent = start_latent.detach()
            current_latent = current_latent.detach()
            target_goal_latent = target_goal_latent.detach()
            task_embedding = task_embedding.detach()

        predicted_goal = self.progress_goal_predictor(start_latent, task_embedding)
        current_output = self.progress_checker(
            start_latent, current_latent, predicted_goal
        )
        start_output = self.progress_checker(
            start_latent, start_latent, predicted_goal
        )
        goal_output = self.progress_checker(
            start_latent, target_goal_latent, predicted_goal
        )
        regression_loss = F.smooth_l1_loss(
            current_output["progress"].float(), target.float()
        )
        anchor_loss = 0.5 * (
            start_output["progress"].float().square().mean()
            + (1.0 - goal_output["progress"].float()).square().mean()
        )
        ranking_loss = progress_ranking_loss(
            current_output["progress"], target, episode_ids
        )
        goal_loss = latent_goal_loss(predicted_goal, target_goal_latent)
        auxiliary_loss = (
            self.progress_loss_weight * regression_loss
            + self.progress_anchor_weight * anchor_loss
            + self.progress_ranking_weight * ranking_loss
            + self.progress_goal_weight * goal_loss
        )
        return {
            **current_output,
            "progress_regression_loss": regression_loss,
            "progress_anchor_loss": anchor_loss,
            "progress_ranking_loss": ranking_loss,
            "progress_goal_loss": goal_loss,
            "progress_auxiliary_loss": auxiliary_loss,
        }

    def _inference_progress(
        self,
        *,
        current_latent: torch.Tensor,
        task_embedding: torch.Tensor,
        instructions: List[str],
        examples: List[dict],
    ) -> dict[str, torch.Tensor]:
        if self.progress_goal_predictor is None or self.progress_checker is None:
            raise RuntimeError("progress checker is disabled")
        signature = tuple((text or "").strip().lower() for text in instructions)
        explicit_reset = any(bool(example.get("episode_start", False)) for example in examples)
        cache_mismatch = (
            self._progress_start_latent is None
            or self._progress_start_latent.shape != current_latent.shape
            or self._progress_instruction_signature != signature
        )
        if explicit_reset or cache_mismatch:
            self.reset_progress_state()
            self._progress_start_latent = current_latent.detach().clone()
            self._progress_goal_latent = self.progress_goal_predictor(
                self._progress_start_latent, task_embedding.detach()
            ).detach()
            self._progress_instruction_signature = signature

        output = self.progress_checker(
            self._progress_start_latent,
            current_latent,
            self._progress_goal_latent,
        )
        raw_progress = output["progress"]
        if self._progress_previous is None:
            filtered_progress = raw_progress
        else:
            filtered_progress = (
                self.progress_ema * self._progress_previous
                + (1.0 - self.progress_ema) * raw_progress
            )
        self._progress_previous = filtered_progress.detach()
        return {**output, "raw_progress": raw_progress, "progress": filtered_progress}

    def remap_checkpoint_state_dict(self, state_dict: dict) -> dict:
        """Load checkpoints written before delta_head was renamed."""
        remapped = dict(state_dict)
        legacy_marker = "world_model.delta_head."
        current_marker = "world_model.residual_predictor."
        for key in tuple(remapped):
            if legacy_marker not in key:
                continue
            current_key = key.replace(legacy_marker, current_marker)
            remapped.setdefault(current_key, remapped[key])
            del remapped[key]
        return remapped

    def _pool_visual_tokens_to_action_queries(
        self,
        visual_tokens: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        action_model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        # visual_tokens: (B, T, K, C) == [current latent, WM-predicted future latents].
        # chunk_len action queries cross-attend all T*K tokens so the action
        # head reads the world model's prediction with full token/temporal
        # structure. ACT additionally appends projected state memory tokens and
        # refines the queries through a multi-layer Transformer decoder.
        selected_action_model = action_model or self.action_model
        if self.action_model_type == "ACT":
            if selected_action_model is None:
                raise RuntimeError("an ACT action model must be selected")
            return selected_action_model.decode_action_queries(
                visual_tokens,
                state=state,
            )
        return self.visual_action_head(visual_tokens, state=state)

    def _resolve_batch_embodiment(
        self,
        examples: Optional[List[dict]] = None,
        robot_tag: Optional[str] = None,
    ) -> Optional[str]:
        if not self.multi_embodiment_actions:
            return None
        tags = set()
        if robot_tag is not None:
            tags.add(str(robot_tag))
        if examples is not None:
            tags.update(str(example.get("robot_tag", "")) for example in examples)
        tags.discard("")
        if len(tags) != 1:
            raise ValueError(
                "multi-embodiment LeWMOFT requires a homogeneous batch with one "
                f"robot_tag, got {sorted(tags) if tags else 'none'}"
            )
        tag = next(iter(tags))
        if tag not in self.action_models:
            raise KeyError(
                f"robot_tag={tag!r} has no configured ACT head; "
                f"available={list(self.action_models.keys())}"
            )
        action_specs = {
            str(example["action_spec_id"])
            for example in (examples or [])
            if example.get("action_spec_id") is not None
        }
        expected_action_spec = self.embodiment_head_specs[tag].get("action_spec_id")
        if len(action_specs) > 1 or (
            action_specs
            and expected_action_spec is not None
            and action_specs != {str(expected_action_spec)}
        ):
            raise ValueError(
                f"robot_tag={tag!r} expects action_spec_id={expected_action_spec!r}, "
                f"got {sorted(action_specs)}"
            )
        return tag

    def _action_runtime(
        self, robot_tag: Optional[str]
    ) -> tuple[nn.Module, int, int, tuple[int, ...]]:
        if self.multi_embodiment_actions:
            if robot_tag is None:
                raise ValueError("robot_tag is required for multi-embodiment actions")
            spec = self.embodiment_head_specs[robot_tag]
            horizons = tuple(
                int(value)
                for value in spec.get(
                    "random_execution_horizons",
                    range(1, int(spec["action_horizon"]) + 1),
                )
            )
            return (
                self.action_models[robot_tag],
                int(spec["action_horizon"]),
                int(spec.get("state_dim", 0)),
                horizons,
            )
        state_dim = (
            int(self.config.framework.world_model.get("state_cond_dim", 8))
            if self.use_state_cond
            else 0
        )
        return (
            self.action_model,
            self.action_horizon,
            state_dim,
            self.random_execution_horizons,
        )

    def _action_gripper_indices(
        self, robot_tag: Optional[str]
    ) -> tuple[int, ...]:
        action_cfg = self.config.framework.action_model
        if self.multi_embodiment_actions:
            if robot_tag is None:
                raise ValueError("robot_tag is required for embodiment action metadata")
            configured = self.embodiment_head_specs[robot_tag].get(
                "gripper_indices", ()
            )
        else:
            configured = action_cfg.get("gripper_indices", ())
        return tuple(int(index) for index in configured)

    @staticmethod
    def _action_valid_mask_tensor(
        examples: List[dict],
        device: torch.device,
        action_horizon: int,
    ) -> Optional[torch.Tensor]:
        raw_masks = [example.get("action_valid_mask") for example in examples]
        if all(mask is None for mask in raw_masks):
            return None
        if any(mask is None for mask in raw_masks):
            raise ValueError(
                "action_valid_mask must be present for every example in a batch"
            )
        masks = np.asarray(raw_masks, dtype=np.bool_)
        if masks.ndim != 2 or masks.shape[1] < action_horizon:
            raise ValueError(
                "action_valid_mask must have shape [B,H] with H at least "
                f"{action_horizon}, got {tuple(masks.shape)}"
            )
        return torch.as_tensor(
            masks[:, -action_horizon:], device=device, dtype=torch.bool
        )

    def _condition_task_on_embodiment(
        self, task_embedding: torch.Tensor, robot_tag: Optional[str]
    ) -> torch.Tensor:
        if not self.multi_embodiment_actions:
            return task_embedding
        if robot_tag is None or self.embodiment_embedding is None:
            raise ValueError("robot_tag is required for embodiment conditioning")
        index = torch.tensor(
            self.embodiment_tag_to_index[robot_tag],
            device=task_embedding.device,
            dtype=torch.long,
        )
        return task_embedding + self.embodiment_embedding(index).to(
            dtype=task_embedding.dtype
        ).unsqueeze(0)

    def _view_valid_mask_tensor(
        self, examples: List[dict], device: torch.device
    ) -> Optional[torch.Tensor]:
        masks = [example.get("view_valid_mask") for example in examples]
        if all(mask is None for mask in masks):
            return None
        if any(mask is None for mask in masks):
            raise ValueError("view_valid_mask must be present for every example or none")
        mask = torch.as_tensor(masks, device=device, dtype=torch.bool)
        expected = (len(examples), self.num_views)
        if tuple(mask.shape) != expected:
            raise ValueError(
                f"view_valid_mask must have shape {expected}, got {tuple(mask.shape)}"
            )
        return mask

    def _wm_loss_mask_tensor(
        self,
        examples: List[dict],
        view_valid_mask: Optional[torch.Tensor],
        device: torch.device,
        frame_count: int,
    ) -> Optional[torch.Tensor]:
        """Combine view and future-frame validity into a (B, T, K) loss mask.

        Blank padded views carry zero residual targets and end-of-episode
        padding frames carry repeated frames; both must be excluded from the
        world-model loss, its statistics, and the copy-ratio diagnostics.
        """
        has_frame_mask = any(
            example.get("future_frame_valid_mask") is not None for example in examples
        )
        if view_valid_mask is None and not has_frame_mask:
            return None
        batch_size = len(examples)
        if view_valid_mask is None:
            token_valid = torch.ones(
                batch_size,
                self.num_visual_tokens,
                device=device,
                dtype=torch.bool,
            )
        else:
            token_valid = view_valid_mask.repeat_interleave(
                self.visual_tokens_per_view, dim=1
            )
        frame_masks = []
        for example in examples:
            raw_mask = example.get("future_frame_valid_mask")
            if raw_mask is None:
                frame_masks.append(np.ones(frame_count, dtype=np.bool_))
                continue
            mask = np.asarray(raw_mask, dtype=np.bool_)
            if mask.shape != (frame_count,):
                raise ValueError(
                    "expected future_frame_valid_mask shape "
                    f"({frame_count},), got {mask.shape}"
                )
            frame_masks.append(mask)
        frame_valid = torch.as_tensor(
            np.stack(frame_masks), device=device, dtype=torch.bool
        )
        if not bool(frame_valid[:, 0].all()):
            raise ValueError("the current frame must always be valid")
        loss_mask = frame_valid[:, :, None] & token_valid[:, None, :]
        return loss_mask.to(dtype=torch.float32)

    def _validate_future_time_offsets(self, examples: List[dict]) -> None:
        if self.future_time_offsets_s is None:
            return
        expected = np.asarray(self.future_time_offsets_s, dtype=np.float32)
        for example in examples:
            actual = example.get("future_time_offsets_s")
            actual_array = (
                np.asarray(actual, dtype=np.float32) if actual is not None else None
            )
            if (
                actual_array is None
                or actual_array.shape != expected.shape
                or not np.allclose(actual_array, expected)
            ):
                raise ValueError(
                    f"robot_tag={example.get('robot_tag')!r} must use shared "
                    f"future_time_offsets_s={list(self.future_time_offsets_s)}, "
                    f"got {actual}"
                )

    def _pad_inference_views(self, frame) -> tuple[list, list[bool]]:
        views = list(frame) if isinstance(frame, (list, tuple)) else [frame]
        views = [to_pil_preserve(view) for view in views]
        if len(views) > self.num_views:
            raise ValueError(
                f"inference provides {len(views)} views, model expects {self.num_views}"
            )
        mask = [True] * len(views) + [False] * (self.num_views - len(views))
        if not views:
            raise ValueError("inference requires at least one camera view")
        blank = Image.new("RGB", views[0].size)
        views.extend(blank.copy() for _ in range(self.num_views - len(views)))
        return views, mask

    def _current_state_tensor(
        self,
        examples: List[dict],
        device: torch.device,
        state_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Stack only the current normalized proprio state from each example."""
        if state_dim is None:
            state_dim = (
                self.predictor_state_dim
                if self.predictor_state_dim > 0
                else int(self.config.framework.world_model.get("state_cond_dim", 8))
            )
        current_states = []
        for example in examples:
            raw_state = example.get("state")
            if raw_state is None:
                raise KeyError(
                    "LeWMOFT.use_state_cond=True requires normalized 'state' in every example"
                )
            state = np.asarray(raw_state, dtype=np.float32)
            if state.ndim == 2:
                state = state[0]
            if state.ndim != 1 or state.shape[0] != state_dim:
                raise ValueError(
                    f"expected state shape ({state_dim},) or (T, {state_dim}), got {state.shape}"
                )
            current_states.append(state)
        return torch.as_tensor(np.stack(current_states), device=device, dtype=torch.float32)

    def _predictor_state_tensor(
        self, examples: List[dict], device: torch.device
    ) -> torch.Tensor:
        """Stack current state or a causal state history for the world model."""
        state_dim = max(self.predictor_state_dim, self.context_correction_state_dim)
        if state_dim <= 0:
            raise ValueError("predictor state requested with no configured state dimension")
        histories = []
        for example in examples:
            raw_state = example.get("state")
            if raw_state is None:
                raise KeyError(
                    "state-conditioned latent prediction requires normalized "
                    "'state' in every example"
                )
            state = np.asarray(raw_state, dtype=np.float32)
            if state.ndim == 1:
                state = state[None]
            if state.ndim != 2 or state.shape[1] != state_dim:
                raise ValueError(
                    f"expected state shape ({state_dim},) or (T, {state_dim}), "
                    f"got {state.shape}"
                )
            if self.predictor_state_history == 1:
                histories.append(state[self.predictor_state_current_index])
                continue
            if state.shape[0] < self.predictor_state_history:
                padding = np.repeat(
                    state[:1], self.predictor_state_history - state.shape[0], axis=0
                )
                state = np.concatenate([padding, state], axis=0)
            histories.append(state[-self.predictor_state_history :])
        return torch.as_tensor(
            np.stack(histories), device=device, dtype=torch.float32
        )

    def _smooth_future_valid_mask_tensor(
        self,
        examples: List[dict],
        device: torch.device,
        *,
        frame_count: int,
    ) -> torch.Tensor:
        """Stack validity for the smooth frames before padded-frame losses."""

        masks = []
        for example in examples:
            raw_mask = example.get("future_frame_valid_mask")
            if raw_mask is None:
                raise KeyError(
                    "smooth latent training requires 'future_frame_valid_mask'; "
                    "enable datasets.vla_data.future_obs_valid_mask"
                )
            mask = np.asarray(raw_mask, dtype=np.bool_)
            if mask.shape != (frame_count,):
                raise ValueError(
                    f"expected future frame valid mask ({frame_count},), "
                    f"got {mask.shape}"
                )
            if not bool(mask[0]):
                raise ValueError("the current frame must always be valid")
            masks.append(mask)
        return torch.as_tensor(
            np.stack(masks), device=device, dtype=torch.bool
        )

    def _visual_token_regularization(self, tokens: torch.Tensor):
        """Return diversity/variance penalties and a collapse diagnostic."""
        x = tokens.float()
        K = x.shape[2]
        if K < 2:
            zero = x.new_zeros(())
            return zero, zero, x.new_ones(())

        off_diag = ~torch.eye(K, device=x.device, dtype=torch.bool).view(1, 1, K, K)
        # Penalize pairwise cosine on the raw normalized content. The previous
        # per-sample centering (subtract the token-axis mean before normalizing)
        # had a blind spot: tokens that collapse to a shared direction have
        # centered values of zero, so the penalty vanished exactly in the
        # collapse regime it was meant to prevent. The pooler's LayerNorm keeps
        # per-sample channel variance at one, so the loss cannot be trivially
        # satisfied by shrinking token norms.
        normalized = F.normalize(x, dim=-1, eps=1e-6)
        cosine = torch.matmul(normalized, normalized.transpose(-1, -2))
        diversity_loss = cosine.square().masked_select(off_diag).mean()

        token_std = x.var(dim=2, unbiased=False).add(1e-4).sqrt()
        variance_loss = F.relu(self.visual_token_min_std - token_std).mean()

        mean_cosine = cosine.masked_select(off_diag).mean()
        return diversity_loss, variance_loss, mean_cosine

    @torch.no_grad()
    def _visual_content_diagnostics(self, content: torch.Tensor):
        """Diagnose content collapse before static position embeddings."""
        x = content.float()
        B, T, K, C = x.shape
        spatial_std = x.var(dim=2, unbiased=False).add(1e-8).sqrt().mean()
        sample_std = (
            x.reshape(B * T, K, C)
            .var(dim=0, unbiased=False)
            .add(1e-8)
            .sqrt()
            .mean()
        )

        normalized = F.normalize(x, dim=-1, eps=1e-6)
        cosine = torch.matmul(normalized, normalized.transpose(-1, -2))
        off_diag = ~torch.eye(K, device=x.device, dtype=torch.bool).view(1, 1, K, K)
        mean_cosine = cosine.masked_select(off_diag).mean()

        centered = x - x.mean(dim=2, keepdim=True)
        gram = torch.matmul(centered, centered.transpose(-1, -2)) / max(C, 1)
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
        probs = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        effective_rank = torch.exp(
            -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        ).mean()
        return {
            "spatial_std": spatial_std,
            "sample_std": sample_std,
            "mean_cosine": mean_cosine,
            "effective_rank": effective_rank,
        }

    def _hash_instruction(self, instruction: Optional[str]) -> int:
        """Map an instruction string to a stable embedding-table bucket.

        Uses md5 (not Python ``hash``) so the mapping is deterministic across
        processes/runs regardless of ``PYTHONHASHSEED``.
        """
        text = (instruction or "").strip().lower()
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.num_task_buckets

    def _embed_task(self, instructions: List[str], device: torch.device) -> torch.Tensor:
        if self.lang_cond_type == "text":
            return self.task_embedding(instructions, device=device)
        ids = torch.tensor(
            [self._hash_instruction(s) for s in instructions],
            device=device,
            dtype=torch.long,
        )
        return self.task_embedding(ids)  # (B, task_emb_dim)

    def _transition_auxiliary_losses(
        self,
        action_queries: torch.Tensor,
        latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute training-only teacher/student transition objectives.

        Ground-truth future tokens are detached before entering the teacher.
        The mature baseline's visual coordinate system therefore cannot move to
        make the auxiliary reconstruction easier.
        """

        if self.transition_auxiliary is None:
            raise RuntimeError("transition auxiliary is disabled")
        anchor = latent[:, self.wm_ctx_len - 1].detach()
        future = latent[
            :, self.wm_ctx_len : self.wm_ctx_len + self.n_future
        ].detach()
        scale = self.world_model.delta_scale.detach().clamp_min(
            self.world_model._stats_eps
        )
        target_delta = (future - anchor[:, None]) / scale

        if self.transition_mode in {"teacher", "combined"}:
            teacher_tokens, teacher_prediction = self.transition_auxiliary.teacher_forward(
                anchor, target_delta
            )
        else:
            with torch.no_grad():
                teacher_tokens, teacher_prediction = (
                    self.transition_auxiliary.teacher_forward(anchor, target_delta)
                )

        teacher_l1 = F.smooth_l1_loss(
            teacher_prediction.float(), target_delta.float()
        )
        teacher_cosine = token_cosine_loss(teacher_prediction, target_delta)
        teacher_reconstruction = (
            teacher_l1 + self.transition_cosine_weight * teacher_cosine
        )

        zero = teacher_reconstruction.detach() * 0.0
        output = {
            "transition_teacher_recon_loss": teacher_reconstruction,
            "transition_teacher_l1_loss": teacher_l1,
            "transition_teacher_cosine_loss": teacher_cosine,
            "transition_alignment_loss": zero,
            "transition_alignment_cosine_loss": zero,
            "transition_alignment_l1_loss": zero,
            "transition_decode_loss": zero,
            "transition_decode_l1_loss": zero,
            "transition_decode_cosine_loss": zero,
        }
        if self.transition_mode == "teacher":
            return output

        student_input = (
            action_queries.detach()
            if self.transition_detach_action_queries
            else action_queries
        )
        student_tokens, student_prediction = self.transition_auxiliary.student_forward(
            student_input, anchor
        )
        target_tokens = teacher_tokens.detach()
        alignment_cosine = token_cosine_loss(student_tokens, target_tokens)
        student_ln = F.layer_norm(
            student_tokens.float(), student_tokens.shape[-1:]
        )
        target_ln = F.layer_norm(target_tokens.float(), target_tokens.shape[-1:])
        alignment_l1 = F.smooth_l1_loss(student_ln, target_ln)
        alignment = (
            alignment_cosine
            + self.transition_alignment_l1_weight * alignment_l1
        )

        decode_l1 = F.smooth_l1_loss(
            student_prediction.float(), target_delta.float()
        )
        decode_cosine = token_cosine_loss(student_prediction, target_delta)
        decode = decode_l1 + self.transition_cosine_weight * decode_cosine
        output.update(
            {
                "transition_alignment_loss": alignment,
                "transition_alignment_cosine_loss": alignment_cosine,
                "transition_alignment_l1_loss": alignment_l1,
                "transition_decode_loss": decode,
                "transition_decode_l1_loss": decode_l1,
                "transition_decode_cosine_loss": decode_cosine,
            }
        )
        return output

    def _should_update_latent_stats(self) -> bool:
        """Whether the latent-delta normalizer should track this training run."""

        if getattr(self, "freeze_latent_stats", False):
            return False

        transition_freezes_visual_coordinates = (
            self.transition_mode in {"teacher", "student"}
            or (
                self.transition_mode in {"joint", "combined"}
                and getattr(self, "transition_joint_freeze_base", True)
            )
        )
        return not transition_freezes_visual_coordinates

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        instructions = [example["lang"] for example in examples]
        self._validate_future_time_offsets(examples)
        device = next(self.parameters()).device
        robot_tag = self._resolve_batch_embodiment(examples)
        action_model, action_horizon, action_state_dim, execution_horizons = (
            self._action_runtime(robot_tag)
        )
        gripper_indices = self._action_gripper_indices(robot_tag)
        actions_target = None
        action_valid_mask = None
        if not self.world_model_only:
            actions = [example["action"] for example in examples]
            actions = torch.tensor(np.array(actions), device=device, dtype=torch.float32)
            if self.multi_embodiment_actions and actions.shape[1] != action_horizon:
                raise ValueError(
                    f"robot_tag={robot_tag!r} expects action horizon {action_horizon}, "
                    f"got {actions.shape[1]}"
                )
            if actions.shape[1] < action_horizon:
                raise ValueError(
                    f"action target has {actions.shape[1]} steps, fewer than "
                    f"action_horizon={action_horizon}"
                )
            expected_action_dim = int(action_model.action_dim)
            if actions.shape[2] != expected_action_dim:
                raise ValueError(
                    f"robot_tag={robot_tag!r} expects action dim {expected_action_dim}, "
                    f"got {actions.shape[2]}"
                )
            actions_target = actions[:, -action_horizon:, :]
            action_valid_mask = self._action_valid_mask_tensor(
                examples, device, action_horizon
            )

        # === Encode the current frame + future frames into a latent sequence ===
        # frames_per_example[b] = [current_views, future_views_1, ..., future_views_Tf]
        frames_per_example = []
        for example in examples:
            current = example["image"]
            future = example.get("future_images")
            if future is None:
                raise KeyError(
                    "LeWMOFT requires 'future_images' in each "
                    "example (enable future-frame loading in the data config)."
                )
            policy_frames = [current] + list(future)
            if self.use_progress_checker:
                start_image = example.get("progress_start_image")
                goal_image = example.get("progress_goal_image")
                if start_image is None or goal_image is None:
                    raise KeyError(
                        "LeWMOFT.use_progress_checker=True requires "
                        "'progress_start_image' and 'progress_goal_image' in every "
                        "training example (set datasets.vla_data.include_progress: true)."
                    )
                policy_frames.extend([start_image, goal_image])
            frames_per_example.append(policy_frames)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            patch_tokens = self.backbone.encode_patch_frames(
                frames_per_example
            )  # (B, 1+Tf, V, N, D)

        if self.smooth_latent_enabled:
            if self.smooth_world_model is None:
                raise RuntimeError(
                    "smooth_latent_enabled requires a constructed branch"
                )
            required_frames = self.smooth_world_model.required_frames
            if patch_tokens.shape[1] != required_frames:
                raise ValueError(
                    "smooth latent training requires exactly "
                    f"{required_frames} image frames "
                    "(consecutive targets plus one far frame), "
                    f"got {patch_tokens.shape[1]}"
                )
            with torch.autocast("cuda", dtype=torch.float32):
                task_emb = self._embed_task(
                    instructions, device=patch_tokens.device
                )
                task_emb = self._condition_task_on_embodiment(task_emb, robot_tag)
                valid_mask = self._smooth_future_valid_mask_tensor(
                    examples,
                    patch_tokens.device,
                    frame_count=required_frames,
                )
                smooth = self.smooth_world_model(
                    patch_tokens.float(),
                    goal=task_emb,
                    valid_mask=valid_mask,
                )
                if self.smooth_action_enabled:
                    if actions_target is None:
                        raise RuntimeError(
                            "smooth action training requires action targets"
                        )
                    # The policy sees three full spatial token grids: current
                    # z_t and the two action-free predicted futures. True
                    # future content remains a training target only.
                    action_latents = torch.cat(
                        [smooth["latent"][:, :1], smooth["pred_future_latent"]],
                        dim=1,
                    )
                    current_state = (
                        self._current_state_tensor(
                            examples, patch_tokens.device, state_dim=action_state_dim
                        )
                        if self.use_state_cond
                        else None
                    )
                    action_queries = self._pool_visual_tokens_to_action_queries(
                        action_latents,
                        state=current_state,
                        action_model=action_model,
                    )
                    pred_actions = action_model.predict_action(action_queries)
                    full_l1_action_loss = masked_action_l1_loss(
                        pred_actions, actions_target, action_valid_mask
                    )
                    action_metrics = action_l1_diagnostics(
                        pred_actions,
                        actions_target,
                        action_valid_mask,
                        gripper_indices=gripper_indices,
                    )
                    total_loss = (
                        self.smooth_action_loss_weight * full_l1_action_loss
                        + self.smooth_world_model_loss_weight * smooth["loss"]
                    )
                    output = {
                        "action_loss": total_loss,
                        "l1_action_loss": full_l1_action_loss.detach(),
                        "full_l1_action_loss": full_l1_action_loss.detach(),
                        "latent_loss": smooth[
                            "latent_prediction_loss"
                        ].detach(),
                        "latent_cosine_loss": total_loss.detach() * 0.0,
                        "sigreg_loss": smooth["sigreg_loss"].detach(),
                        "smooth_action_l1_loss": full_l1_action_loss.detach(),
                        "smooth_world_model_loss": smooth["loss"].detach(),
                        "smooth_joint_total_loss": total_loss.detach(),
                    }
                    output.update(
                        {
                            name: value.detach()
                            for name, value in action_metrics.items()
                        }
                    )
                else:
                    zero = smooth["loss"].detach() * 0.0
                    output = {
                        "action_loss": smooth["loss"],
                        "l1_action_loss": zero,
                        "full_l1_action_loss": zero,
                        "latent_loss": smooth[
                            "latent_prediction_loss"
                        ].detach(),
                        "latent_cosine_loss": zero,
                        "sigreg_loss": smooth["sigreg_loss"].detach(),
                        "world_model_only_loss": smooth["loss"].detach(),
                    }
            for name, value in smooth.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    output[f"smooth_{name}"] = value.detach()
            return output

        with torch.autocast("cuda", dtype=torch.float32):
            view_valid_mask = self._view_valid_mask_tensor(examples, patch_tokens.device)
            latent, content_latent = self.visual_token_pooler(
                patch_tokens.float(),
                return_content=True,
                view_valid_mask=view_valid_mask,
            )
            task_emb = self._embed_task(instructions, device=latent.device)
            task_emb = self._condition_task_on_embodiment(task_emb, robot_tag)
            progress_start_latent = None
            progress_goal_latent = None
            if self.use_progress_checker:
                progress_start_latent = latent[:, -2]
                progress_goal_latent = latent[:, -1]
                # Endpoint frames supervise only the progress branch. They must
                # never enter world-model targets or the ordinary action path.
                latent = latent[:, :-2]
                content_latent = content_latent[:, :-2]

            required_frames = self.wm_ctx_len + self.n_future * self.rollout_steps
            if latent.shape[1] < required_frames:
                raise ValueError(
                    f"expected at least {required_frames} temporal frames, "
                    f"got {latent.shape[1]}"
                )
            latent = latent[:, :required_frames]
            content_latent = content_latent[:, :required_frames]
            current_state = (
                self._current_state_tensor(
                    examples, latent.device, state_dim=action_state_dim
                )
                if self.use_state_cond
                else None
            )
            predictor_state = (
                self._predictor_state_tensor(examples, latent.device)
                if self.predictor_state_dim > 0
                or self.context_correction_state_dim > 0
                else None
            )
            wm_loss_mask = self._wm_loss_mask_tensor(
                examples, view_valid_mask, latent.device, required_frames
            )

            B = latent.shape[0]
            wm_out = self.world_model(
                latent,
                ctx_len=self.wm_ctx_len,
                goal=task_emb,
                state=predictor_state,
                # Freeze the EMA only when the visual coordinate system itself
                # is frozen. In joint/combined from-scratch training the
                # encoder and pooler move, so delta_scale must track that drift.
                update_stats=self._should_update_latent_stats(),
                rollout_steps=self.rollout_steps,
                loss_mask=wm_loss_mask,
            )
            pred_future_latent = wm_out["pred_future_latent"]

            # State-probe input: [current real latent, predicted future latents].
            head_tokens = torch.cat([latent[:, : self.wm_ctx_len], pred_future_latent], dim=1)

            pred_content = self.visual_token_pooler.remove_position(
                pred_future_latent, view_valid_mask=view_valid_mask
            )
            source_div, source_var, source_cos = self._visual_token_regularization(
                content_latent
            )
            pred_div, pred_var, pred_cos = self._visual_token_regularization(pred_content)
            visual_token_diversity_loss = 0.5 * (source_div + pred_div)
            visual_token_variance_loss = 0.5 * (source_var + pred_var)
            visual_token_mean_cosine = 0.5 * (source_cos + pred_cos)
            if self.visual_diagnostics:
                content_diag = self._visual_content_diagnostics(content_latent)
                pred_content_diag = self._visual_content_diagnostics(pred_content)

            progress_output = None
            transition_losses = None
            prefix_l1_action_loss = None
            sampled_prefix_mean = None
            future_action_sensitivity = None
            future_action_sensitivity_ratio = None
            if self.world_model_only:
                full_l1_action_loss = latent.new_zeros(())
                l1_action_loss = full_l1_action_loss
            else:
                action_queries = self._pool_visual_tokens_to_action_queries(
                    head_tokens, state=current_state, action_model=action_model
                )
                if self.use_progress_checker:
                    progress_target = torch.as_tensor(
                        [example["progress_target"] for example in examples],
                        device=latent.device,
                        dtype=torch.float32,
                    )
                    episode_ids = [
                        example.get("progress_episode_id", index)
                        for index, example in enumerate(examples)
                    ]
                    progress_output = self._training_progress(
                        start_latent=progress_start_latent,
                        current_latent=latent[:, self.wm_ctx_len - 1],
                        target_goal_latent=progress_goal_latent,
                        task_embedding=task_emb,
                        target=progress_target,
                        episode_ids=episode_ids,
                    )
                    action_queries = self._condition_action_queries_on_progress(
                        action_queries, progress_output["progress"]
                    )
                transition_losses = (
                    self._transition_auxiliary_losses(action_queries, latent)
                    if self.transition_mode != "off"
                    else None
                )
                pred_actions = action_model.predict_action(action_queries)
                full_l1_action_loss = masked_action_l1_loss(
                    pred_actions, actions_target, action_valid_mask
                )
                action_metrics = action_l1_diagnostics(
                    pred_actions,
                    actions_target,
                    action_valid_mask,
                    gripper_indices=gripper_indices,
                )
                if self.random_prefix_loss_weight > 0 and execution_horizons:
                    choices = torch.tensor(
                        execution_horizons,
                        device=pred_actions.device,
                        dtype=torch.long,
                    )
                    choice_indices = torch.randint(
                        choices.numel(), (B,), device=pred_actions.device
                    )
                    sampled_prefix_lengths = choices[choice_indices]
                    sampled_prefix_mean = sampled_prefix_lengths.float().mean()
                    prefix_l1_action_loss = prefix_l1_loss(
                        pred_actions,
                        actions_target,
                        sampled_prefix_lengths,
                        action_valid_mask,
                    )
                    l1_action_loss = (
                        full_l1_action_loss
                        + self.random_prefix_loss_weight * prefix_l1_action_loss
                    ) / (1.0 + self.random_prefix_loss_weight)
                else:
                    l1_action_loss = full_l1_action_loss
                if self.visual_diagnostics:
                    with torch.no_grad():
                        ablated_tokens = head_tokens[:1].detach().clone()
                        ablated_tokens[:, self.wm_ctx_len :] = ablated_tokens[
                            :, self.wm_ctx_len - 1 : self.wm_ctx_len
                        ].expand(
                            -1,
                            ablated_tokens.shape[1] - self.wm_ctx_len,
                            -1,
                            -1,
                        )
                        ablated_queries = self._pool_visual_tokens_to_action_queries(
                            ablated_tokens,
                            state=current_state[:1]
                            if current_state is not None
                            else None,
                            action_model=action_model,
                        )
                        if progress_output is not None:
                            ablated_queries = self._condition_action_queries_on_progress(
                                ablated_queries, progress_output["progress"][:1]
                            )
                        ablated_actions = action_model.predict_action(ablated_queries)
                        future_action_sensitivity = (
                            pred_actions[:1].detach() - ablated_actions
                        ).abs().mean()
                        future_action_sensitivity_ratio = future_action_sensitivity / (
                            pred_actions[:1].detach().abs().mean() + 1e-6
                        )

            latent_loss = wm_out["latent_loss"]
            latent_cosine_loss = wm_out["latent_cosine_loss"]
            rollout_latent_loss = wm_out.get("rollout_latent_loss")
            total_loss = (
                l1_action_loss
                + self.loss_latent_weight * latent_loss
                + self.latent_cosine_weight * latent_cosine_loss
                + self.visual_token_diversity_weight * visual_token_diversity_loss
                + self.visual_token_variance_weight * visual_token_variance_loss
            )
            if rollout_latent_loss is not None:
                total_loss = total_loss + self.loss_rollout_weight * rollout_latent_loss
            sigreg_loss = wm_out.get("sigreg_loss")
            if progress_output is not None:
                total_loss = total_loss + progress_output["progress_auxiliary_loss"]
            if sigreg_loss is not None:
                total_loss = total_loss + self.loss_sigreg_weight * sigreg_loss

            # === State probe: ground latents in (future) physical state ===
            # Decode [current real latent, predicted future latents] back to the
            # robot's proprioceptive state and supervise against the aligned
            # current+future states. This pushes the world model's imagined
            # future latents to encode where the arm will actually be.
            if self.use_state_probe:
                raw_states = [example.get("state") for example in examples]
                if any(s is None for s in raw_states):
                    raise KeyError(
                        "LeWMOFT.use_state_probe=True requires 'state' in each example "
                        "(set datasets.vla_data.include_state: true)."
                    )
                states = torch.tensor(
                    np.array(raw_states), device=head_tokens.device, dtype=torch.float32
                )  # (B, 1+Tf, D_state)
                T_head = head_tokens.shape[1]
                state_target = states[:, :T_head, : self.state_probe_dim]
                state_input = head_tokens.mean(dim=2)
                state_pred = self.state_probe(state_input.to(torch.float32))
                state_loss = self.state_loss_fn(state_pred, state_target)
                total_loss = total_loss + self.loss_state_weight * state_loss

            if transition_losses is not None:
                if self.transition_mode == "teacher":
                    total_loss = (
                        self.transition_teacher_recon_weight
                        * transition_losses["transition_teacher_recon_loss"]
                    )
                else:
                    total_loss = (
                        total_loss
                        + (
                            self.transition_teacher_recon_weight
                            * transition_losses["transition_teacher_recon_loss"]
                            if self.transition_mode == "combined"
                            else 0.0
                        )
                        + self.transition_alignment_weight
                        * transition_losses["transition_alignment_loss"]
                        + self.transition_decode_weight
                        * transition_losses["transition_decode_loss"]
                    )

        out = {
            "action_loss": total_loss,
            "l1_action_loss": l1_action_loss.detach(),
            "full_l1_action_loss": full_l1_action_loss.detach(),
            "latent_loss": latent_loss.detach(),
            "latent_cosine_loss": latent_cosine_loss.detach(),
        }
        if not self.world_model_only:
            out.update(
                {name: value.detach() for name, value in action_metrics.items()}
            )
        if self.world_model_only:
            out["world_model_only_loss"] = total_loss.detach()
        for metric_name, metric_value in wm_out.items():
            if metric_name.startswith("latent_loss_horizon_"):
                out[metric_name] = metric_value.detach()
        for metric_name, metric_value in wm_out.items():
            if metric_name.startswith("rollout_") and torch.is_tensor(metric_value):
                out[metric_name] = metric_value.detach()
        if prefix_l1_action_loss is not None:
            out["prefix_l1_action_loss"] = prefix_l1_action_loss.detach()
            out["sampled_prefix_mean"] = sampled_prefix_mean.detach()
        if sigreg_loss is not None:
            out["sigreg_loss"] = sigreg_loss.detach()
        for metric_name in (
            "latent_base_loss",
            "context_correction_rms",
            "context_correction_to_base_ratio",
            "delta_scale",
            "delta_target_rms",
            "delta_pred_rms",
            "delta_copy_mse",
            "delta_pred_mse",
            "delta_mean_baseline_mse",
            "delta_to_copy_ratio",
            "delta_direction_cosine",
        ):
            if metric_name in wm_out:
                out[metric_name] = wm_out[metric_name].detach()
        out["visual_token_diversity_loss"] = visual_token_diversity_loss.detach()
        out["visual_token_variance_loss"] = visual_token_variance_loss.detach()
        out["visual_token_mean_cosine"] = visual_token_mean_cosine.detach()
        if self.visual_diagnostics:
            out["visual_content_spatial_std"] = content_diag["spatial_std"]
            out["visual_content_sample_std"] = content_diag["sample_std"]
            out["visual_content_mean_cosine"] = content_diag["mean_cosine"]
            out["visual_content_effective_rank"] = content_diag["effective_rank"]
            out["visual_pred_content_spatial_std"] = pred_content_diag["spatial_std"]
            out["visual_pred_content_mean_cosine"] = pred_content_diag["mean_cosine"]
            out["visual_pred_content_effective_rank"] = pred_content_diag[
                "effective_rank"
            ]
            if future_action_sensitivity is not None:
                out["future_action_sensitivity"] = future_action_sensitivity
                out["future_action_sensitivity_ratio"] = (
                    future_action_sensitivity_ratio
                )
        if self.use_state_probe:
            out["state_loss"] = state_loss.detach()
        if transition_losses is not None:
            for name, value in transition_losses.items():
                out[name] = value.detach()
        if progress_output is not None:
            out["progress_mean"] = progress_output["progress"].mean().detach()
            out["progress_target_mean"] = progress_target.mean().detach()
            out["progress_geometric_mean"] = progress_output[
                "geometric_progress"
            ].mean().detach()
            for name in (
                "progress_regression_loss",
                "progress_anchor_loss",
                "progress_ranking_loss",
                "progress_goal_loss",
                "progress_auxiliary_loss",
            ):
                out[name] = progress_output[name].detach()
        return out

    def forward_policy_tensor(
        self,
        examples: Optional[List[dict]] = None,
        *,
        images: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        task_bucket_ids: Optional[torch.Tensor] = None,
        robot_tag: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        """Run the deployed smooth LeWM action path without disabling gradients.

        RL trainers should call this method instead of :meth:`predict_action` so
        action log-probabilities can backpropagate into the LeWM action path.
        Callers may either provide ordinary StarVLA ``examples`` or tensor-only
        replay inputs. Tensor images are expected as raw ``[B,V,H,W,C]`` (or
        channels-first) data; states must already use the checkpoint's training
        normalization.

        Returns:
            A dictionary containing ``action_mean`` with shape ``[B,H,D]`` and
            ``policy_features`` with shape ``[B,H,C]``.
        """
        if not self.smooth_action_enabled:
            raise NotImplementedError(
                "forward_policy_tensor currently supports the deployed smooth "
                "LeWM action branch only."
            )
        if self.smooth_world_model is None:
            raise RuntimeError("smooth action inference requires smooth_world_model")
        if examples is not None and images is not None:
            raise ValueError("Pass either examples or tensor images, not both.")
        if examples is None and images is None:
            raise ValueError("forward_policy_tensor requires examples or images.")
        resolved_robot_tag = self._resolve_batch_embodiment(examples, robot_tag)
        action_model, _, action_state_dim, _ = self._action_runtime(
            resolved_robot_tag
        )

        instructions: Optional[List[str]] = None
        if examples is not None:
            if type(examples) is not list:
                examples = [examples]
            instructions = [example["lang"] for example in examples]
            train_obs_image_size = getattr(
                self.config.datasets.vla_data, "obs_image_size", None
            )
            frames_per_example = []
            for example in examples:
                current, _ = self._pad_inference_views(example["image"])
                if train_obs_image_size:
                    current = resize_images(
                        current, target_size=train_obs_image_size
                    )
                frames_per_example.append([current])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                patch_tokens = self.backbone.encode_patch_frames(frames_per_example)
        else:
            if not isinstance(images, torch.Tensor) or images.ndim != 5:
                raise ValueError(
                    "tensor images must have shape [B,V,H,W,C] or [B,V,C,H,W], "
                    f"got {getattr(images, 'shape', None)}"
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                patch_tokens = self.backbone.encode_patch_image_tensor(
                    images.unsqueeze(1)
                )

        with torch.autocast("cuda", dtype=torch.float32):
            current_latent, current_content = self.smooth_world_model.projector(
                patch_tokens.float(), return_content=True
            )
            if task_bucket_ids is not None:
                if self.lang_cond_type != "hash":
                    raise ValueError(
                        "task_bucket_ids are only valid for hash language conditioning"
                    )
                task_ids = task_bucket_ids.to(
                    device=current_latent.device, dtype=torch.long
                ).reshape(-1)
                if task_ids.shape[0] != current_latent.shape[0]:
                    raise ValueError(
                        "task_bucket_ids batch size does not match images: "
                        f"{task_ids.shape[0]} != {current_latent.shape[0]}"
                    )
                task_emb = self.task_embedding(task_ids)
            else:
                if instructions is None:
                    raise ValueError(
                        "Tensor replay requires task_bucket_ids for task conditioning."
                    )
                task_emb = self._embed_task(
                    instructions, device=current_latent.device
                )
            task_emb = self._condition_task_on_embodiment(
                task_emb, resolved_robot_tag
            )

            predicted_delta = self.smooth_world_model.predictor(
                current_latent, goal=task_emb, state=None
            )
            predicted_future_content = current_content + predicted_delta
            predicted_future = self.smooth_world_model.projector.add_position(
                predicted_future_content
            )
            action_latents = torch.cat([current_latent, predicted_future], dim=1)

            current_state = None
            if self.use_state_cond:
                if state is not None:
                    current_state = state.to(
                        device=current_latent.device, dtype=torch.float32
                    )
                    if current_state.ndim == 3 and current_state.shape[1] == 1:
                        current_state = current_state[:, 0]
                    expected_dim = action_state_dim
                    if (
                        current_state.ndim != 2
                        or current_state.shape[-1] != expected_dim
                    ):
                        raise ValueError(
                            "normalized state must have shape "
                            f"[B,{expected_dim}], got {tuple(current_state.shape)}"
                        )
                elif examples is not None:
                    current_state = self._current_state_tensor(
                        examples,
                        current_latent.device,
                        state_dim=action_state_dim,
                    )
                else:
                    raise ValueError(
                        "Tensor replay for a state-conditioned LeWM requires state."
                    )

            action_queries = self._pool_visual_tokens_to_action_queries(
                action_latents, state=current_state, action_model=action_model
            )
            action_mean = action_model.predict_action(action_queries)

        return {
            "action_mean": action_mean,
            "policy_features": action_queries,
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        robot_tag = self._resolve_batch_embodiment(
            examples, kwargs.get("robot_tag")
        )
        action_model, _, action_state_dim, _ = self._action_runtime(robot_tag)
        if self.smooth_latent_enabled and not self.smooth_action_enabled:
            raise RuntimeError(
                "the smooth spatial latent branch is world-model-only and is "
                "not connected to the action head"
            )
        if self.smooth_action_enabled:
            policy_output = self.forward_policy_tensor(examples=examples)
            return {
                "normalized_actions": policy_output["action_mean"].detach().cpu().numpy()
            }

        instructions = [example["lang"] for example in examples]
        train_obs_image_size = getattr(
            self.config.datasets.vla_data, "obs_image_size", None
        )

        inference_history_len = self.wm_ctx_len
        frames_per_example = []
        inference_view_masks = []
        for example in examples:
            history = example.get("image_history") or [example["image"]]
            frames = []
            inferred_masks = []
            for frame in history[-inference_history_len:]:
                padded_frame, inferred_mask = self._pad_inference_views(frame)
                frames.append(padded_frame)
                inferred_masks.append(inferred_mask)
            if len(frames) < inference_history_len:
                frames = [frames[0]] * (inference_history_len - len(frames)) + frames
            if train_obs_image_size:
                frames = resize_images(frames, target_size=train_obs_image_size)
            frames_per_example.append(frames)
            inference_view_masks.append(
                example.get("view_valid_mask", inferred_masks[-1])
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            patch_tokens = self.backbone.encode_patch_frames(
                frames_per_example
            )  # (B, ctx, V, N, D)

        with torch.autocast("cuda", dtype=torch.float32):
            view_valid_mask = torch.as_tensor(
                inference_view_masks,
                device=patch_tokens.device,
                dtype=torch.bool,
            )
            latent_history = self.visual_token_pooler(
                patch_tokens.float(), view_valid_mask=view_valid_mask
            )
            latent = latent_history[:, -self.wm_ctx_len :]
            task_emb = self._embed_task(instructions, device=latent.device)
            task_emb = self._condition_task_on_embodiment(task_emb, robot_tag)
            current_state = (
                self._current_state_tensor(
                    examples, latent.device, state_dim=action_state_dim
                )
                if self.use_state_cond
                else None
            )
            predictor_state = (
                self._predictor_state_tensor(examples, latent.device)
                if self.predictor_state_dim > 0
                or self.context_correction_state_dim > 0
                else None
            )
            pred_future_latent = self.world_model.regress_future(
                latent,
                goal=task_emb,
                state=predictor_state,
            )
            head_tokens = torch.cat(
                [latent[:, : self.wm_ctx_len], pred_future_latent], dim=1
            )
            action_queries = self._pool_visual_tokens_to_action_queries(
                head_tokens, state=current_state, action_model=action_model
            )
            progress_output = None
            conditioning_progress = None
            if self.use_progress_checker:
                progress_output = self._inference_progress(
                    current_latent=latent[:, self.wm_ctx_len - 1],
                    task_embedding=task_emb,
                    instructions=instructions,
                    examples=examples,
                )
                conditioning_progress = self._select_inference_progress_for_action(
                    progress_output["progress"]
                )
                if conditioning_progress is not None:
                    action_queries = self._condition_action_queries_on_progress(
                        action_queries, conditioning_progress
                    )
            pred_actions = action_model.predict_action(action_queries)

        normalized_actions = pred_actions.detach().cpu().numpy()
        output = {"normalized_actions": normalized_actions}
        if progress_output is not None:
            output["progress"] = progress_output["progress"].detach().cpu().numpy()
            output["raw_progress"] = (
                progress_output["raw_progress"].detach().cpu().numpy()
            )
            if conditioning_progress is not None:
                output["conditioning_progress"] = (
                    conditioning_progress.detach().cpu().numpy()
                )
        return output


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf
    from PIL import Image

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.name = "LeWMOFT"
    cfg.framework.world_model = {
        "base_wm": "WinKawaks/vit-tiny-patch16-224",
        "train_encoder": False,
    }

    model: LeWM_OFT = LeWM_OFT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "future_images": [[image, image], [image, image]],  # 2 future frames x 2 views
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."
    out = model([sample, sample2])
    print(out)
