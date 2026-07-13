# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
LeWM-OFT Framework — LeWorldModel ViT encoder + flow-matching world model.

Uses the LeWM front-end (a pretrained ViT encoder) as the perception backbone,
a Wan-style flow-matching world model to roll out future latents, and an
OFT-style MLP regression head for action prediction. The world model's action
flow loss is kept as an auxiliary objective and can still be used directly for
flow-only ablations via ``world_model.action_source: flow``.

Architecture:
  ViT encoder → per-view latent concat[CLS, mean-pool] → [B, V, 2*hidden]
    → view fusion → [B, 1+Tf, hidden]
    → WanWorldModel flow predictor
        → future latents          (flow_latent_loss)
        → flow-sampled actions    (flow_action_loss, auxiliary)
        → OFT MLP action head from [current raw latent, predicted future latent]
                → l1_action_loss
    → optional state probe on [current, predicted future] latents (state_loss)

The ViT encoder is frozen by default; set ``world_model.train_encoder: true``
to keep the joint fine-tuning interface available.
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
from starVLA.model.modules.world_model.token_wan_world_model import TokenWanWorldModel
from starVLA.model.modules.world_model.wan_world_model import WanWorldModel
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


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
            # === flow-matching latent world model (ported le-wm WanPredictor) ===
            # When ``use_future_latent`` is on, a Wan-style predictor rolls out
            # future latents (flow_latent_loss) and the OFT head conditions on
            # ``[current latent, predicted future latents]`` before producing the
            # action chunk. Requires the dataloader to supply ``future_images``.
            "use_future_latent": True,
            "n_future": 2,            # number of future latents to predict
            "ctx_len": 1,             # clean context frames (current frame only)
            "predictor_dim": 384,
            "predictor_layers": 4,
            "predictor_heads": 6,
            "predictor_ffn": 1024,
            "flow_sample_steps": 20,
            # lingbot-va scheduler recipe: separate SNR shifts for latent vs
            # action flow + noisy-context augmentation probability.
            "latent_snr_shift": 5.0,
            "action_snr_shift": 0.05,
            "noisy_cond_prob": 0.5,
            # === Direct delta-regression auxiliary head (oracle-probe inspired) ===
            # Deterministically predicts future latent residuals from the current
            # latent (+goal) by default. Keep action conditioning off for the
            # deployed OFT path, otherwise GT actions leak through the future latent.
            "use_delta_head": False,
            "loss_delta_weight": 0.5,
            "delta_head_hidden": 1024,
            "delta_head_inference": False,  # replace flow-sampled latent at rollout
            # transformer delta head (le-wm ARPredictor style) + SIGReg; when
            # ``oft_future_from_delta`` the OFT head conditions on the delta
            # head's future latent instead of the flow-sampled one.
            "delta_head_type": "mlp",       # "mlp" | "transformer"
            "delta_head_dim": 384,
            "delta_head_depth": 4,
            "delta_head_heads": 6,
            "delta_head_ffn": 1024,
            "delta_head_sigreg_weight": 0.0,
            "delta_head_condition_on_action": False,
            "oft_future_from_delta": False,
            "loss_latent_weight": 1.0,
            "loss_action_weight": 1.0,
            "action_source": "oft",
            "use_visual_token_wm": False,
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
        }
    )

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "WinKawaks/vit-tiny-patch16-224",
        }
    )

    # === Action shape config (action_dim / horizon consumed by the flow model) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "MLP",
            "action_dim": 7,
            "action_hidden_dim": 384,
            "future_action_window_size": 8,
            "past_action_window_size": 0,
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
    """World-Model-for-Action framework: LeWM ViT encoder + flow-matching world model."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(LeWMOFTDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)

        wm_hidden = self.backbone.model.config.hidden_size
        wm_cfg = self.config.framework.get("world_model", {}) or {}
        self.num_views = int(wm_cfg.get("num_views", 2))

        # 是否使用visual token, 有的话visual_token_dim就是visual token的维度, 没有的话就是wm_hidden
        self.use_visual_token_wm = bool(wm_cfg.get("use_visual_token_wm", False))
        self.use_state_cond = bool(wm_cfg.get("use_state_cond", False))
        self.expects_normalized_state = self.use_state_cond
        if self.use_state_cond and not self.use_visual_token_wm:
            raise ValueError("use_state_cond currently requires use_visual_token_wm=True")
        visual_token_dim_cfg = wm_cfg.get("visual_token_dim", None)
        self.visual_token_dim = int(visual_token_dim_cfg) if visual_token_dim_cfg else wm_hidden

        # `action_horizon` is the single source of truth for chunk length;
        # legacy aliases are normalised upstream by share_tools.apply_config_compat.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon

        self.config.framework.action_model.action_hidden_dim = wm_hidden
        self.action_model = get_action_model(config=self.config)
        self.action_hidden_dim = wm_hidden

        # Keep the OFT projection shape compatible with vfuse checkpoints:
        # raw LeWM features are concatenated across views (V * wm_hidden), then
        # projected to per-action tokens of width wm_hidden.
        self.action_context_dim = self.num_views * wm_hidden
        self.action_query_proj = nn.Linear(self.num_views * wm_hidden, self.chunk_len * wm_hidden)
        self.future_action_context_proj = nn.Linear(wm_hidden, self.action_context_dim)
        with torch.no_grad():
            self.future_action_context_proj.weight.zero_()
            eye = torch.eye(wm_hidden)
            for v in range(self.num_views):
                self.future_action_context_proj.weight[v * wm_hidden : (v + 1) * wm_hidden, :] = eye
            self.future_action_context_proj.bias.zero_()
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

        # === Flow-matching latent world model (le-wm WanPredictor) ===
        # Predicts future latents so the OFT head can "see the future" before
        # producing actions, and supplies a real latent-prediction loss.
        self.action_source = str(wm_cfg.get("action_source", "oft")).lower()
        if self.action_source not in {"oft", "flow"}:
            raise ValueError(
                f"LeWMOFT world_model.action_source must be 'oft' or 'flow', got {self.action_source!r}"
            )
        self.use_future_latent = bool(wm_cfg.get("use_future_latent", True))

        if self.use_future_latent:
            if self.use_visual_token_wm:
                self.action_query_proj.requires_grad_(False)
                self.future_action_context_proj.requires_grad_(False)
            self.n_future = int(wm_cfg.get("n_future", 2))
            self.wm_ctx_len = int(wm_cfg.get("ctx_len", 1))
            raw_action_dim = int(self.config.framework.action_model.action_dim)
            assert (
                self.action_horizon % self.n_future == 0
            ), f"action_horizon ({self.action_horizon}) must be divisible by n_future ({self.n_future})"
            self.wm_segment_len = self.action_horizon // self.n_future
            self.loss_latent_weight = float(wm_cfg.get("loss_latent_weight", 1.0))
            self.loss_action_weight = float(wm_cfg.get("loss_action_weight", 1.0))
            self.use_delta_head = bool(wm_cfg.get("use_delta_head", False))
            self.loss_delta_weight = float(wm_cfg.get("loss_delta_weight", 0.5))
            self.oft_future_from_delta = bool(wm_cfg.get("oft_future_from_delta", False))
            self.loss_delta_sigreg_weight = float(wm_cfg.get("delta_head_sigreg_weight", 0.0))

            predictor_kwargs = {
                "latent_dim": self.visual_token_dim if self.use_visual_token_wm else wm_hidden,
                "action_dim": self.wm_segment_len * raw_action_dim,
                "goal_dim": self.task_emb_dim,
                "dim": int(wm_cfg.get("predictor_dim", 384)),
                "num_layers": int(wm_cfg.get("predictor_layers", 4)),
                "num_heads": int(wm_cfg.get("predictor_heads", 6)),
                "ffn_dim": int(wm_cfg.get("predictor_ffn", 1024)),
                "ctx_len": self.wm_ctx_len,
                "flow_sample_steps": int(wm_cfg.get("flow_sample_steps", 20)),
            }
            if self.use_visual_token_wm:
                patch_dim = int(self.backbone.encoder.config.hidden_size)
                self.visual_tokens_per_view = int(
                    wm_cfg.get(
                        "visual_tokens_per_view",
                        wm_cfg.get("num_visual_tokens", 16),
                    )
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
                self.world_model = TokenWanWorldModel(
                    **predictor_kwargs,
                    num_tokens=self.num_visual_tokens,
                    token_grid_shape=(
                        self.num_views,
                        self.visual_token_pooler.grid_size,
                        self.visual_token_pooler.grid_size,
                    ),
                    scheduler_kwargs={
                        "shift": float(wm_cfg.get("latent_snr_shift", 5.0)),
                        "sigma_min": 0.0,
                        "extra_one_step": True,
                    },
                    action_scheduler_kwargs={
                        "shift": float(wm_cfg.get("action_snr_shift", 0.05)),
                        "sigma_min": 0.0,
                        "extra_one_step": True,
                    },
                    stats_momentum=float(wm_cfg.get("latent_stats_momentum", 0.99)),
                    delta_head_futures=self.n_future if self.use_delta_head else 0,
                    delta_head_inference=bool(wm_cfg.get("delta_head_inference", False)),
                    delta_head_dim=int(wm_cfg.get("delta_head_dim", 384)),
                    delta_head_depth=int(wm_cfg.get("delta_head_depth", 4)),
                    delta_head_heads=int(wm_cfg.get("delta_head_heads", 6)),
                    delta_head_ffn=int(wm_cfg.get("delta_head_ffn", 1024)),
                    delta_head_sigreg_weight=float(wm_cfg.get("delta_head_sigreg_weight", 0.0)),
                )
                self.visual_action_head = VisualActionCrossAttn(
                    token_dim=self.visual_token_dim,
                    action_hidden_dim=wm_hidden,
                    chunk_len=self.chunk_len,
                    num_frames=self.wm_ctx_len + self.n_future,
                    num_tokens=self.num_visual_tokens,
                    num_heads=int(wm_cfg.get("visual_action_heads", 8)),
                    state_dim=int(wm_cfg.get("state_cond_dim", 8))
                    if self.use_state_cond
                    else 0,
                    state_hidden_dim=int(wm_cfg.get("state_cond_hidden_dim", 256)),
                    state_dropout=float(wm_cfg.get("state_cond_dropout", 0.1)),
                )
                if self.use_state_cond and bool(wm_cfg.get("state_cond_only", False)):
                    for name, parameter in self.visual_action_head.named_parameters():
                        parameter.requires_grad_(name.startswith("state_encoder."))
            else:
                self.world_model = WanWorldModel(
                    **predictor_kwargs,
                    scheduler_kwargs={
                        "shift": float(wm_cfg.get("latent_snr_shift", 5.0)),
                        "sigma_min": 0.0,
                        "extra_one_step": True,
                    },
                    action_scheduler_kwargs={
                        "shift": float(wm_cfg.get("action_snr_shift", 0.05)),
                        "sigma_min": 0.0,
                        "extra_one_step": True,
                    },
                    noisy_cond_prob=float(wm_cfg.get("noisy_cond_prob", 0.5)),
                    whiten_latent=bool(wm_cfg.get("whiten_latent", True)),
                    predict_residual=bool(wm_cfg.get("predict_residual", True)),
                    stats_momentum=float(wm_cfg.get("latent_stats_momentum", 0.99)),
                    delta_head_futures=self.n_future if self.use_delta_head else 0,
                    delta_head_hidden=int(wm_cfg.get("delta_head_hidden", 1024)),
                    delta_head_inference=bool(wm_cfg.get("delta_head_inference", False)),
                    delta_head_type=str(wm_cfg.get("delta_head_type", "mlp")),
                    delta_head_dim=int(wm_cfg.get("delta_head_dim", 384)),
                    delta_head_depth=int(wm_cfg.get("delta_head_depth", 4)),
                    delta_head_heads=int(wm_cfg.get("delta_head_heads", 6)),
                    delta_head_ffn=int(wm_cfg.get("delta_head_ffn", 1024)),
                    delta_head_sigreg_weight=float(wm_cfg.get("delta_head_sigreg_weight", 0.0)),
                    delta_head_condition_on_action=bool(wm_cfg.get("delta_head_condition_on_action", False)),
                )

            # === Multi-view fusion ===
            # encode_frames concatenates the V camera views per frame
            # (V * wm_hidden). Project back to wm_hidden so the world model keeps
            # its original width. Initialized to equal-weight
            # averaging so the fused latent initially reproduces the previous
            # mean-over-views behavior exactly -> smooth warm-start from a
            # mean-pool checkpoint; the model then learns per-view weighting.
            self.view_fuse = nn.Linear(self.num_views * wm_hidden, wm_hidden)
            with torch.no_grad():
                self.view_fuse.weight.zero_()
                eye = torch.eye(wm_hidden)
                for v in range(self.num_views):
                    self.view_fuse.weight[:, v * wm_hidden : (v + 1) * wm_hidden] = eye / self.num_views
                self.view_fuse.bias.zero_()

            # === Latent normalization for the flow world model ===
            # The fused view latent has per-dim std ~0.19 (<< 1), scale-mismatched
            # with the flow-matching N(0,1) noise: flow_latent_loss barely drops
            # and sample_future's scale explodes (~3.7x). A LayerNorm anchors the
            # latent to ~unit variance so the flow target is well-scaled and
            # sampling is scale-calibrated. The visual-token path already
            # normalizes inside the pooler (out_norm), so it uses Identity.
            self.latent_norm = (
                nn.Identity() if self.use_visual_token_wm else nn.LayerNorm(wm_hidden)
            )

            # === Optional state probe (ground latents in physical state) ===
            # Decodes each latent frame back to the robot's proprioceptive
            # state so predicted future latents can be supervised against the
            # *future* physical state. Training-only; predict_action never uses
            # it. Requires the dataloader to pack aligned future ``state``.
            self.use_state_probe = bool(wm_cfg.get("use_state_probe", False))
            if self.use_state_probe:
                self.state_probe_dim = int(wm_cfg.get("state_dim", 8))
                self.loss_state_weight = float(wm_cfg.get("loss_state_weight", 0.5))
                state_input_dim = self.visual_token_dim if self.use_visual_token_wm else wm_hidden
                self.state_probe = nn.Sequential(
                    nn.Linear(state_input_dim, wm_hidden),
                    nn.GELU(),
                    nn.Linear(wm_hidden, self.state_probe_dim),
                )
                self.state_loss_fn = nn.MSELoss()
        else:
            self.use_visual_token_wm = False
            self.use_state_probe = False

        if self.use_state_cond and bool(wm_cfg.get("state_cond_only", False)):
            self.requires_grad_(False)
            self.visual_action_head.state_encoder.requires_grad_(True)

        logger.info(f"[LeWMOFT] action source = {self.action_source}")

    def _pool_to_action_queries(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B = hidden_states.shape[0]
        pooled = hidden_states.mean(dim=1)
        queries = self.action_query_proj(pooled)
        return queries.view(B, self.chunk_len, self.action_hidden_dim)

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

    def _build_oft_context(
        self, raw_latent: torch.Tensor, pred_future_latent: torch.Tensor
    ) -> torch.Tensor:
        current_latent = raw_latent[:, : self.wm_ctx_len]
        future_latent = self.future_action_context_proj(pred_future_latent)
        return torch.cat([current_latent, future_latent], dim=1)

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
                    "LeWMOFT.use_future_latent=True requires 'future_images' in each "
                    "example (enable future-frame loading in the data config)."
                )
            frames_per_example.append([current] + list(future))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.use_visual_token_wm:
                patch_tokens = self.backbone.encode_patch_frames(frames_per_example)  # (B, 1+Tf, V, N, D)
            else:
                raw_latent = self.backbone.encode_frames(frames_per_example)  # (B, 1+Tf, V*C)

        with torch.autocast("cuda", dtype=torch.float32):
            if self.use_visual_token_wm:
                latent, content_latent = self.visual_token_pooler(
                    patch_tokens.float(), return_content=True
                )  # (B, 1+Tf, K, C)
            else:
                raw_latent = raw_latent.float()
                latent = self.view_fuse(raw_latent)  # fuse camera views -> (B, 1+Tf, C)
                latent = self.latent_norm(latent)  # normalize to ~unit variance for the flow WM
            task_emb = self._embed_task(instructions, device=latent.device)
            current_state = (
                self._current_state_tensor(examples, latent.device)
                if self.use_state_cond
                else None
            )

            # Macro-actions: split the per-step action chunk into n_future
            # segments, one per predicted future latent (drives frame i->i+1).
            B = latent.shape[0]
            macro_actions = actions_target.reshape(
                B, self.n_future, self.wm_segment_len * actions_target.shape[-1]
            )

            wm_out = self.world_model.flow_loss(
                latent, macro_actions, ctx_len=self.wm_ctx_len, goal=task_emb
            )
            pred_future_latent = wm_out["pred_future_latent"]  # (B, Tf, C)
            # When the latent flow is disabled, the flow z0 estimate is
            # meaningless -> condition the OFT head (and state probe) on the
            # deterministic delta-head prediction instead.
            if self.oft_future_from_delta and "delta_future_latent" in wm_out:
                pred_future_latent = wm_out["delta_future_latent"]

            # State-probe input: [current real latent, predicted future latents].
            head_tokens = torch.cat([latent[:, : self.wm_ctx_len], pred_future_latent], dim=1)

            if self.use_visual_token_wm:
                pred_content = self.visual_token_pooler.remove_position(pred_future_latent)
                source_div, source_var, source_cos = self._visual_token_regularization(
                    content_latent
                )
                pred_div, pred_var, pred_cos = self._visual_token_regularization(
                    pred_content
                )
                visual_token_diversity_loss = 0.5 * (source_div + pred_div)
                visual_token_variance_loss = 0.5 * (source_var + pred_var)
                visual_token_mean_cosine = 0.5 * (source_cos + pred_cos)
                if self.visual_diagnostics:
                    content_diag = self._visual_content_diagnostics(content_latent)
                    pred_content_diag = self._visual_content_diagnostics(pred_content)

            if self.use_visual_token_wm:
                action_queries = self._pool_visual_tokens_to_action_queries(
                    head_tokens, state=current_state
                )
            else:
                oft_context = self._build_oft_context(raw_latent, pred_future_latent)
                action_queries = self._pool_to_action_queries(oft_context)
            pred_actions = self.action_model.predict_action(action_queries)
            l1_action_loss = self.l1_loss(pred_actions, actions_target)
            if self.use_visual_token_wm and self.visual_diagnostics:
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

            flow_latent_loss = wm_out["flow_latent_loss"]
            flow_action_loss = wm_out["flow_action_loss"]
            flow_aux_loss = self.loss_latent_weight * flow_latent_loss + self.loss_action_weight * flow_action_loss
            delta_latent_loss = wm_out.get("delta_latent_loss")
            if delta_latent_loss is not None:
                flow_aux_loss = flow_aux_loss + self.loss_delta_weight * delta_latent_loss
            delta_sigreg_loss = wm_out.get("delta_sigreg_loss")
            if delta_sigreg_loss is not None and self.loss_delta_sigreg_weight > 0:
                flow_aux_loss = flow_aux_loss + self.loss_delta_sigreg_weight * delta_sigreg_loss
            if self.use_visual_token_wm:
                flow_aux_loss = (
                    flow_aux_loss
                    + self.visual_token_diversity_weight * visual_token_diversity_loss
                    + self.visual_token_variance_weight * visual_token_variance_loss
                )
            if self.action_source == "oft":
                total_loss = l1_action_loss + flow_aux_loss
            else:
                total_loss = flow_aux_loss

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
                state_input = head_tokens.mean(dim=2) if self.use_visual_token_wm else head_tokens
                state_pred = self.state_probe(state_input.to(torch.float32))
                state_loss = self.state_loss_fn(state_pred, state_target)
                total_loss = total_loss + self.loss_state_weight * state_loss

        out = {
            "action_loss": total_loss,
            "l1_action_loss": l1_action_loss.detach(),
            "flow_latent_loss": flow_latent_loss.detach(),
            "flow_action_loss": flow_action_loss.detach(),
        }
        if delta_latent_loss is not None:
            out["delta_latent_loss"] = delta_latent_loss.detach()
        if delta_sigreg_loss is not None:
            out["delta_sigreg_loss"] = delta_sigreg_loss.detach()
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
        if self.use_visual_token_wm:
            out["visual_token_diversity_loss"] = visual_token_diversity_loss.detach()
            out["visual_token_variance_loss"] = visual_token_variance_loss.detach()
            out["visual_token_mean_cosine"] = visual_token_mean_cosine.detach()
            if self.visual_diagnostics:
                out["visual_content_spatial_std"] = content_diag["spatial_std"]
                out["visual_content_sample_std"] = content_diag["sample_std"]
                out["visual_content_mean_cosine"] = content_diag["mean_cosine"]
                out["visual_content_effective_rank"] = content_diag["effective_rank"]
                out["visual_pred_content_spatial_std"] = pred_content_diag[
                    "spatial_std"
                ]
                out["visual_pred_content_mean_cosine"] = pred_content_diag[
                    "mean_cosine"
                ]
                out["visual_pred_content_effective_rank"] = pred_content_diag[
                    "effective_rank"
                ]
                out["future_action_sensitivity"] = future_action_sensitivity
                out["future_action_sensitivity_ratio"] = future_action_sensitivity_ratio
        if self.use_state_probe:
            out["state_loss"] = state_loss.detach()
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

        # === World-model path: imagine future latents + flow-sample actions ===
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.use_visual_token_wm:
                patch_tokens = self.backbone.encode_patch_frames(frames_per_example)  # (B, 1, V, N, D)
            else:
                raw_latent = self.backbone.encode_frames(frames_per_example)  # (B, 1, V*C)

        with torch.autocast("cuda", dtype=torch.float32):
            if self.use_visual_token_wm:
                latent = self.visual_token_pooler(patch_tokens.float())  # (B, 1, K, C)
            else:
                raw_latent = raw_latent.float()
                latent = self.view_fuse(raw_latent)  # fuse camera views -> (B, 1, C)
                latent = self.latent_norm(latent)  # normalize to ~unit variance for the flow WM
            task_emb = self._embed_task(instructions, device=latent.device)
            current_state = (
                self._current_state_tensor(examples, latent.device)
                if self.use_state_cond
                else None
            )
            # The visual-token OFT path consumes only the deterministic delta
            # rollout. Avoid running a multi-step flow sampler whose latent and
            # action outputs would both be discarded.
            direct_visual_delta = (
                self.use_visual_token_wm
                and self.action_source == "oft"
                and self.use_delta_head
                and self.oft_future_from_delta
            )
            if direct_visual_delta:
                pred_future_latent = self.world_model.regress_future(latent, goal=task_emb)
                pred_future_action = None
            else:
                pred_future_latent, pred_future_action = self.world_model.sample_future(
                    latent, goal=task_emb, n_future=self.n_future, return_action=True
                )  # (B, Tf, wm_segment_len*action_dim)
            B = latent.shape[0]
            raw_action_dim = int(self.config.framework.action_model.action_dim)
            if self.action_source == "flow":
                # Reshape the flow macro-actions into a per-step chunk:
                # (B, n_future, seg*A) -> (B, horizon, A).
                pred_actions = pred_future_action.reshape(B, self.action_horizon, raw_action_dim)
            else:
                if self.use_visual_token_wm:
                    head_tokens = torch.cat([latent[:, : self.wm_ctx_len], pred_future_latent], dim=1)
                    action_queries = self._pool_visual_tokens_to_action_queries(
                        head_tokens, state=current_state
                    )
                else:
                    oft_context = self._build_oft_context(raw_latent, pred_future_latent)
                    action_queries = self._pool_to_action_queries(oft_context)
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
