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

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.modules.world_model.visual_token_delta_world_model import (
    VisualTokenLatentWorldModel,
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
    absolute_error = (pred_actions - target_actions).abs()
    return (absolute_error * mask).sum() / (mask.sum() * action_dim)


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

    def remove_position(self, tokens: torch.Tensor) -> torch.Tensor:
        position = self.position_tokens().to(dtype=tokens.dtype)
        return tokens - position.view(1, 1, self.num_tokens, self.token_dim)

    def forward(self, patches: torch.Tensor, return_content: bool = False):
        # patches: (B, T, V, N, D_patch)
        B, T, V, N, D = patches.shape
        if V != self.num_views:
            raise ValueError(f"expected {self.num_views} views, got {V}")
        patch_grid = math.isqrt(N)
        if patch_grid * patch_grid != N:
            raise ValueError(f"expected a square patch grid, got {N} patch tokens")

        x = self.patch_norm(patches)
        x = x.reshape(B * T * V, patch_grid, patch_grid, D).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))
        x = x.permute(0, 2, 3, 1).reshape(
            B, T, V, self.grid_size, self.grid_size, D
        )
        content = self.out_norm(self.patch_proj(x)).reshape(
            B, T, self.num_tokens, self.token_dim
        )
        position = self.position_tokens().to(dtype=content.dtype)
        tokens = content + position.view(1, 1, self.num_tokens, self.token_dim)
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


@dataclass
class LeWMOFTDefaultConfig:
    """LeWM-OFT default parameters."""

    name: str = "LeWMOFT"

    # === World Model backbone (LeWM ViT encoder) ===
    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "WinKawaks/vit-tiny-patch16-224",
            "train_encoder": False,  # frozen ViT by default; flip for joint finetune
            "num_views": 2,          # camera views per frame (e.g. primary + wrist)
            "n_future": 2,            # number of future latents to predict
            "ctx_len": 1,             # clean context frames (current frame only)
            "loss_latent_weight": 1.0,
            "residual_predictor_dim": 384,
            "residual_predictor_depth": 4,
            "residual_predictor_heads": 6,
            "residual_predictor_ffn": 1024,
            "residual_predictor_sigreg_weight": 0.0,
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

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "WinKawaks/vit-tiny-patch16-224",
        }
    )

    # === Action shape config ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "MLP",
            "action_dim": 7,
            "action_hidden_dim": 384,
            "future_action_window_size": 8,
            "past_action_window_size": 0,
            # Optional auxiliary objective for policies deployed with shorter
            # replanning horizons. Empty choices or zero weight preserves the
            # original full-chunk L1 objective exactly.
            "random_execution_horizons": [1, 2, 4, 8],
            "random_prefix_loss_weight": 0.0,
        }
    )

    # === Language / task conditioning (le-wm-style task embedding) ===
    # le-wm conditions its action head on a per-task one-hot. starVLA only
    # exposes language strings, so we hash each instruction into a fixed bucket
    # and look up a learnable embedding (an implicit task-id embedding that needs
    # no predefined task list). ``embed_dim: null`` -> world-model hidden size.
    lang_cond: dict = field(
        default_factory=lambda: {
            "num_buckets": 4096,
            "embed_dim": None,
        }
    )


@FRAMEWORK_REGISTRY.register("LeWMOFT")
class LeWM_OFT(baseframework):
    """LeWM visual encoder + deterministic latent predictor + OFT action head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(LeWMOFTDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)

        wm_hidden = self.backbone.model.config.hidden_size
        wm_cfg = self.config.framework.get("world_model", {}) or {}
        self.num_views = int(wm_cfg.get("num_views", 2))

        self.use_state_cond = bool(wm_cfg.get("use_state_cond", False))
        self.expects_normalized_state = self.use_state_cond
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

        self.config.framework.action_model.action_hidden_dim = wm_hidden
        self.action_model = get_action_model(config=self.config)
        self.action_hidden_dim = wm_hidden

        self.l1_loss = nn.L1Loss()

        # === Language / task conditioning ===
        # Hash each instruction into a fixed bucket and look up a learnable task
        # embedding (an implicit one-hot / task-id embedding requiring no
        # predefined task list; stable across train/eval because identical
        # instruction strings hash identically). Feeds the world model as the
        # per-task ``goal`` conditioning.
        lang_cfg = self.config.framework.get("lang_cond", {}) or {}
        self.num_task_buckets = int(lang_cfg.get("num_buckets", 4096))
        _emb_dim = lang_cfg.get("embed_dim", None)
        self.task_emb_dim = int(_emb_dim) if _emb_dim else wm_hidden
        self.task_embedding = nn.Embedding(self.num_task_buckets, self.task_emb_dim)

        self.n_future = int(wm_cfg.get("n_future", 2))
        self.wm_ctx_len = int(wm_cfg.get("ctx_len", 1))
        self.loss_latent_weight = float(
            wm_cfg.get("loss_delta_weight", wm_cfg.get("loss_latent_weight", 1.0))
        )
        self.loss_sigreg_weight = float(
            wm_cfg.get(
                "residual_predictor_sigreg_weight",
                wm_cfg.get("delta_head_sigreg_weight", 0.0),
            )
        )

        patch_dim = int(self.backbone.encoder.config.hidden_size)
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
            sigreg_weight=self.loss_sigreg_weight,
            stats_momentum=float(wm_cfg.get("latent_stats_momentum", 0.99)),
        )
        self.visual_action_head = VisualActionCrossAttn(
            token_dim=self.visual_token_dim,
            action_hidden_dim=wm_hidden,
            chunk_len=self.chunk_len,
            num_frames=self.wm_ctx_len + self.n_future,
            num_tokens=self.num_visual_tokens,
            num_heads=int(wm_cfg.get("visual_action_heads", 8)),
            state_dim=int(wm_cfg.get("state_cond_dim", 8)) if self.use_state_cond else 0,
            state_hidden_dim=int(wm_cfg.get("state_cond_hidden_dim", 256)),
            state_dropout=float(wm_cfg.get("state_cond_dropout", 0.1)),
        )

        self.transition_mode = str(wm_cfg.get("transition_mode", "off")).lower()
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
    ) -> torch.Tensor:
        # visual_tokens: (B, T, K, C) == [current latent, WM-predicted future latents].
        # chunk_len action queries cross-attend all T*K tokens so the OFT head
        # reads the world model's prediction with full token/temporal structure.
        return self.visual_action_head(visual_tokens, state=state)

    def _current_state_tensor(self, examples: List[dict], device: torch.device) -> torch.Tensor:
        """Stack only the current normalized proprio state from each example."""
        state_dim = int(self.config.framework.world_model.get("state_cond_dim", 8))
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

    def _visual_token_regularization(self, tokens: torch.Tensor):
        """Return diversity/variance penalties and a collapse diagnostic."""
        x = tokens.float()
        K = x.shape[2]
        if K < 2:
            zero = x.new_zeros(())
            return zero, zero, x.new_ones(())

        off_diag = ~torch.eye(K, device=x.device, dtype=torch.bool).view(1, 1, K, K)
        centered = F.normalize(x - x.mean(dim=2, keepdim=True), dim=-1, eps=1e-6)
        centered_cos = torch.matmul(centered, centered.transpose(-1, -2))
        diversity_loss = centered_cos.square().masked_select(off_diag).mean()

        token_std = x.var(dim=2, unbiased=False).add(1e-4).sqrt()
        variance_loss = F.relu(self.visual_token_min_std - token_std).mean()

        normalized = F.normalize(x, dim=-1, eps=1e-6)
        raw_cos = torch.matmul(normalized, normalized.transpose(-1, -2))
        mean_cosine = raw_cos.masked_select(off_diag).mean()
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

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        device = next(self.parameters()).device
        actions = torch.tensor(np.array(actions), device=device, dtype=torch.float32)
        actions_target = actions[:, -self.action_horizon :, :]

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
            frames_per_example.append([current] + list(future))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            patch_tokens = self.backbone.encode_patch_frames(
                frames_per_example
            )  # (B, 1+Tf, V, N, D)

        with torch.autocast("cuda", dtype=torch.float32):
            latent, content_latent = self.visual_token_pooler(
                patch_tokens.float(), return_content=True
            )  # (B, 1+Tf, K, C)
            task_emb = self._embed_task(instructions, device=latent.device)
            current_state = (
                self._current_state_tensor(examples, latent.device)
                if self.use_state_cond
                else None
            )

            B = latent.shape[0]
            wm_out = self.world_model(
                latent,
                ctx_len=self.wm_ctx_len,
                goal=task_emb,
                # A frozen parameter set can still mutate EMA buffers. Keep
                # the 200k baseline's normalization fixed in every auxiliary
                # stage so its deployed predictions do not drift implicitly.
                update_stats=self.transition_mode == "off",
            )
            pred_future_latent = wm_out["pred_future_latent"]

            # State-probe input: [current real latent, predicted future latents].
            head_tokens = torch.cat([latent[:, : self.wm_ctx_len], pred_future_latent], dim=1)

            pred_content = self.visual_token_pooler.remove_position(pred_future_latent)
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

            action_queries = self._pool_visual_tokens_to_action_queries(
                head_tokens, state=current_state
            )
            transition_losses = (
                self._transition_auxiliary_losses(action_queries, latent)
                if self.transition_mode != "off"
                else None
            )
            pred_actions = self.action_model.predict_action(action_queries)
            full_l1_action_loss = self.l1_loss(pred_actions, actions_target)
            prefix_l1_action_loss = None
            sampled_prefix_mean = None
            if self.random_prefix_loss_weight > 0 and self.random_execution_horizons:
                choices = torch.tensor(
                    self.random_execution_horizons,
                    device=pred_actions.device,
                    dtype=torch.long,
                )
                choice_indices = torch.randint(
                    choices.numel(), (B,), device=pred_actions.device
                )
                sampled_prefix_lengths = choices[choice_indices]
                sampled_prefix_mean = sampled_prefix_lengths.float().mean()
                prefix_l1_action_loss = prefix_l1_loss(
                    pred_actions, actions_target, sampled_prefix_lengths
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
                    ].expand(-1, ablated_tokens.shape[1] - self.wm_ctx_len, -1, -1)
                    ablated_queries = self._pool_visual_tokens_to_action_queries(
                        ablated_tokens, state=current_state[:1] if current_state is not None else None
                    )
                    ablated_actions = self.action_model.predict_action(ablated_queries)
                    future_action_sensitivity = (
                        pred_actions[:1].detach() - ablated_actions
                    ).abs().mean()
                    future_action_sensitivity_ratio = future_action_sensitivity / (
                        pred_actions[:1].detach().abs().mean() + 1e-6
                    )

            latent_loss = wm_out["latent_loss"]
            total_loss = (
                l1_action_loss
                + self.loss_latent_weight * latent_loss
                + self.visual_token_diversity_weight * visual_token_diversity_loss
                + self.visual_token_variance_weight * visual_token_variance_loss
            )
            sigreg_loss = wm_out.get("sigreg_loss")
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
        }
        if prefix_l1_action_loss is not None:
            out["prefix_l1_action_loss"] = prefix_l1_action_loss.detach()
            out["sampled_prefix_mean"] = sampled_prefix_mean.detach()
        if sigreg_loss is not None:
            out["sigreg_loss"] = sigreg_loss.detach()
        for metric_name in (
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
            out["future_action_sensitivity"] = future_action_sensitivity
            out["future_action_sensitivity_ratio"] = future_action_sensitivity_ratio
        if self.use_state_probe:
            out["state_loss"] = state_loss.detach()
        if transition_losses is not None:
            for name, value in transition_losses.items():
                out[name] = value.detach()
        return out

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        frames_per_example = []
        for example in examples:
            history = example.get("image_history") or [example["image"]]
            frames = [to_pil_preserve(frame) for frame in history[-self.wm_ctx_len :]]
            if len(frames) < self.wm_ctx_len:
                frames = [frames[0]] * (self.wm_ctx_len - len(frames)) + frames
            if train_obs_image_size:
                frames = resize_images(frames, target_size=train_obs_image_size)
            frames_per_example.append(frames)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            patch_tokens = self.backbone.encode_patch_frames(
                frames_per_example
            )  # (B, ctx, V, N, D)

        with torch.autocast("cuda", dtype=torch.float32):
            latent = self.visual_token_pooler(patch_tokens.float())
            task_emb = self._embed_task(instructions, device=latent.device)
            current_state = (
                self._current_state_tensor(examples, latent.device)
                if self.use_state_cond
                else None
            )
            pred_future_latent = self.world_model.regress_future(latent, goal=task_emb)
            head_tokens = torch.cat(
                [latent[:, : self.wm_ctx_len], pred_future_latent], dim=1
            )
            action_queries = self._pool_visual_tokens_to_action_queries(
                head_tokens, state=current_state
            )
            pred_actions = self.action_model.predict_action(action_queries)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


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
    cfg.framework.qwenvl.base_vlm = "WinKawaks/vit-tiny-patch16-224"
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
