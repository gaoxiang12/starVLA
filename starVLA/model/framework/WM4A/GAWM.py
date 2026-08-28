# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""GAWM: the geometry-aware world model used by the unified checkpoint.

A DINOv3 encoder produces spatial patch features for three camera views. Fixed
grid pooling preserves their geometry, a task-conditioned residual transformer
predicts two future token grids, and an embodiment-specific ACT head decodes an
action chunk from current and predicted tokens.
"""

import math
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.ACT_ActionHeader import (
    TurboStyleACTActionHead,
)
from starVLA.model.modules.action_model.action_loss import (
    action_l1_diagnostics,
    masked_action_l1_loss,
)
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.modules.world_model.visual_token_delta_world_model import (
    VisualTokenLatentWorldModel,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


class VisualTokenPooler(nn.Module):
    """Pool every view to fixed spatial cells and add view/row/column identity."""

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
                "visual_tokens_per_view must be a perfect square, got "
                f"{self.tokens_per_view}"
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
                f"view_valid_mask must have shape [B,{self.num_views}], got "
                f"{tuple(view_valid_mask.shape)}"
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
        batch_size, frame_count, view_count, patch_count, patch_dim = patches.shape
        if view_count != self.num_views:
            raise ValueError(f"expected {self.num_views} views, got {view_count}")
        patch_grid = math.isqrt(patch_count)
        if patch_grid * patch_grid != patch_count:
            raise ValueError(f"expected a square patch grid, got {patch_count} tokens")

        content = self.patch_norm(patches).reshape(
            batch_size * frame_count * view_count,
            patch_grid,
            patch_grid,
            patch_dim,
        ).permute(0, 3, 1, 2)
        if self.grid_size != patch_grid:
            content = F.adaptive_avg_pool2d(
                content, (self.grid_size, self.grid_size)
            )
        content = content.permute(0, 2, 3, 1).reshape(
            batch_size,
            frame_count,
            view_count,
            self.grid_size,
            self.grid_size,
            patch_dim,
        )
        content = self.out_norm(self.patch_proj(content)).reshape(
            batch_size, frame_count, self.num_tokens, self.token_dim
        )
        position = self.position_tokens().to(dtype=content.dtype)
        tokens = content + position.view(1, 1, self.num_tokens, self.token_dim)
        if view_valid_mask is not None:
            token_mask = self._token_mask(
                view_valid_mask.to(device=tokens.device), dtype=tokens.dtype
            )
            tokens = tokens * token_mask
            content = content * token_mask
        return (tokens, content) if return_content else tokens


class CompositionalTextEncoder(nn.Module):
    """Encode normalized UTF-8 instruction bytes with a compact Transformer."""

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
            raise ValueError("text encoder hidden_dim must be divisible by num_heads")
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
            values = [self.BOS_TOKEN] + [
                value + self.BYTE_OFFSET
                for value in byte_values[: self.max_length - 2]
            ] + [self.EOS_TOKEN]
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
class GAWMDefaultConfig:
    """Defaults for the single supported unified GAWM architecture."""

    name: str = "GAWM"
    world_model: dict = field(
        default_factory=lambda: {
            "encoder_spec": "vitb16",
            "train_encoder": True,
            "num_views": 3,
            "n_future": 2,
            "ctx_len": 1,
            "future_time_offsets_s": [0.0, 0.2, 0.4],
            "loss_latent_weight": 1.0,
            "latent_cosine_weight": 0.1,
            "detach_wm_input": True,
            "latent_stats_momentum": 0.9,
            "residual_predictor_dim": 384,
            "residual_predictor_depth": 4,
            "residual_predictor_heads": 6,
            "residual_predictor_ffn": 1024,
            "visual_tokens_per_view": 16,
            "visual_token_dim": 384,
            "visual_token_diversity_weight": 0.02,
            "visual_token_variance_weight": 0.02,
            "visual_token_min_std": 0.1,
            "use_state_cond": True,
            "state_cond_hidden_dim": 256,
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "ACT",
            "action_horizon": 8,
            "action_hidden_dim": 384,
            "act_num_heads": 8,
            "act_num_layers": 3,
            "act_dim_feedforward": 2048,
            "act_mlp_hidden_dim": 512,
            "act_dropout": 0.1,
            "act_num_state_tokens": 2,
            "embodiment_heads": {},
        }
    )
    lang_cond: dict = field(
        default_factory=lambda: {
            "type": "text",
            "embed_dim": 384,
            "text_hidden_dim": 256,
            "text_depth": 2,
            "text_heads": 4,
            "text_ffn_dim": 512,
            "text_max_length": 128,
            "text_dropout": 0.1,
        }
    )


@FRAMEWORK_REGISTRY.register("GAWM")
class GAWM(baseframework):
    """Checkpoint-compatible unified geometry-aware world model."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(GAWMDefaultConfig, config)
        wm_cfg = self.config.framework.world_model
        action_cfg = self.config.framework.action_model
        lang_cfg = self.config.framework.lang_cond

        if int(wm_cfg.ctx_len) != 1 or int(wm_cfg.n_future) != 2:
            raise ValueError("GAWM supports only ctx_len=1 and n_future=2")
        if not bool(wm_cfg.use_state_cond):
            raise ValueError("GAWM requires normalized proprioceptive state conditioning")
        if str(action_cfg.action_model_type).strip().upper() != "ACT":
            raise ValueError("GAWM supports only ACT action heads")
        if str(lang_cfg.type).strip().lower() != "text":
            raise ValueError("GAWM supports only compositional text conditioning")

        self.backbone = get_world_model(config=self.config)
        wm_hidden = int(self.backbone.model.config.hidden_size)
        self.num_views = int(wm_cfg.num_views)
        self.n_future = 2
        self.wm_ctx_len = 1
        self.use_state_cond = True
        self.expects_normalized_state = True
        self.action_model_type = "ACT"
        self.action_horizon = int(action_cfg.action_horizon)
        self.chunk_len = self.action_horizon
        self.action_hidden_dim = int(action_cfg.action_hidden_dim or wm_hidden)
        self.config.framework.action_model.action_hidden_dim = self.action_hidden_dim
        self.visual_token_dim = int(wm_cfg.visual_token_dim or wm_hidden)

        raw_heads = action_cfg.get("embodiment_heads", {}) or {}
        if not raw_heads:
            raise ValueError("GAWM requires at least one embodiment-specific ACT head")
        self.embodiment_head_specs = {
            str(tag): {str(key): value for key, value in spec.items()}
            for tag, spec in raw_heads.items()
        }
        self.multi_embodiment_actions = True
        self.action_model = None

        self.task_emb_dim = int(lang_cfg.embed_dim or wm_hidden)
        self.task_embedding = CompositionalTextEncoder(
            output_dim=self.task_emb_dim,
            hidden_dim=int(lang_cfg.text_hidden_dim),
            depth=int(lang_cfg.text_depth),
            num_heads=int(lang_cfg.text_heads),
            ffn_dim=int(lang_cfg.text_ffn_dim),
            max_length=int(lang_cfg.text_max_length),
            dropout=float(lang_cfg.text_dropout),
        )
        self.embodiment_tags = tuple(sorted(self.embodiment_head_specs))
        self.embodiment_tag_to_index = {
            tag: index for index, tag in enumerate(self.embodiment_tags)
        }
        self.embodiment_embedding = nn.Embedding(
            len(self.embodiment_tags), self.task_emb_dim
        )
        nn.init.normal_(self.embodiment_embedding.weight, std=0.02)

        offsets = wm_cfg.get("future_time_offsets_s", None)
        self.future_time_offsets_s = (
            tuple(float(value) for value in offsets) if offsets is not None else None
        )
        if self.future_time_offsets_s is not None and len(self.future_time_offsets_s) != 3:
            raise ValueError("future_time_offsets_s must contain current and two futures")
        self.loss_latent_weight = float(wm_cfg.loss_latent_weight)
        self.latent_cosine_weight = float(wm_cfg.latent_cosine_weight)
        self.visual_token_diversity_weight = float(
            wm_cfg.visual_token_diversity_weight
        )
        self.visual_token_variance_weight = float(
            wm_cfg.visual_token_variance_weight
        )
        self.visual_token_min_std = float(wm_cfg.visual_token_min_std)

        patch_dim = getattr(self.backbone, "patch_feature_dim", None) or getattr(
            self.backbone, "feature_dim", None
        )
        if patch_dim is None:
            patch_dim = self.backbone.encoder.config.hidden_size
        self.visual_tokens_per_view = int(wm_cfg.visual_tokens_per_view)
        self.visual_token_pooler = VisualTokenPooler(
            patch_dim=int(patch_dim),
            token_dim=self.visual_token_dim,
            num_views=self.num_views,
            tokens_per_view=self.visual_tokens_per_view,
        )
        self.num_visual_tokens = self.visual_token_pooler.num_tokens
        self.world_model = VisualTokenLatentWorldModel(
            latent_dim=self.visual_token_dim,
            goal_dim=self.task_emb_dim,
            n_future=self.n_future,
            num_tokens=self.num_visual_tokens,
            dim=int(wm_cfg.residual_predictor_dim),
            depth=int(wm_cfg.residual_predictor_depth),
            num_heads=int(wm_cfg.residual_predictor_heads),
            ffn_dim=int(wm_cfg.residual_predictor_ffn),
            stats_momentum=float(wm_cfg.latent_stats_momentum),
            detach_input=bool(wm_cfg.detach_wm_input),
        )

        self.visual_action_head = None
        self.action_models = nn.ModuleDict()
        for tag, spec in self.embodiment_head_specs.items():
            self.action_models[tag] = TurboStyleACTActionHead(
                token_dim=self.visual_token_dim,
                hidden_dim=self.action_hidden_dim,
                action_dim=int(spec["action_dim"]),
                horizon=int(spec["action_horizon"]),
                num_frames=3,
                num_visual_tokens=self.num_visual_tokens,
                num_heads=int(action_cfg.act_num_heads),
                num_layers=int(action_cfg.act_num_layers),
                dim_feedforward=int(action_cfg.act_dim_feedforward),
                mlp_hidden_dim=int(action_cfg.act_mlp_hidden_dim),
                dropout=float(action_cfg.act_dropout),
                state_dim=int(spec["state_dim"]),
                state_hidden_dim=int(wm_cfg.state_cond_hidden_dim),
                num_state_tokens=int(action_cfg.act_num_state_tokens),
            )

    def remap_checkpoint_state_dict(self, state_dict: dict) -> dict:
        """Map the pre-rename residual predictor keys to the GAWM layout."""
        remapped = dict(state_dict)
        legacy_marker = "world_model.delta_head."
        current_marker = "world_model.residual_predictor."
        for key in tuple(remapped):
            if legacy_marker in key:
                remapped.setdefault(
                    key.replace(legacy_marker, current_marker), remapped[key]
                )
                del remapped[key]

        embedding_key = "embodiment_embedding.weight"
        source_embedding = remapped.get(embedding_key)
        source_tags = tuple(
            sorted(
                {
                    key.split(".", 2)[1]
                    for key in remapped
                    if key.startswith("action_models.") and key.count(".") >= 2
                }
            )
        )
        if (
            not source_tags
            and source_embedding is not None
            and source_embedding.ndim == 2
            and source_embedding.shape[0] <= len(self.embodiment_tags)
        ):
            source_tags = self.embodiment_tags[: source_embedding.shape[0]]
        target_embedding = self.embodiment_embedding.weight
        if (
            source_embedding is not None
            and source_embedding.ndim == target_embedding.ndim == 2
            and source_embedding.shape[0] == len(source_tags)
            and source_embedding.shape[1:] == target_embedding.shape[1:]
            and set(source_tags).issubset(self.embodiment_tags)
            and source_embedding.shape != target_embedding.shape
        ):
            expanded = target_embedding.detach().clone()
            target_index = {
                tag: index for index, tag in enumerate(self.embodiment_tags)
            }
            for source_index, tag in enumerate(source_tags):
                expanded[target_index[tag]].copy_(source_embedding[source_index])
            remapped[embedding_key] = expanded
        return remapped

    def _resolve_batch_embodiment(
        self,
        examples: Optional[List[dict]] = None,
        robot_tag: Optional[str] = None,
    ) -> str:
        tags = {str(robot_tag)} if robot_tag is not None else set()
        if examples is not None:
            tags.update(str(example.get("robot_tag", "")) for example in examples)
        tags.discard("")
        if len(tags) != 1:
            raise ValueError(
                "multi-embodiment GAWM requires a homogeneous batch with one "
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
        expected = self.embodiment_head_specs[tag].get("action_spec_id")
        if len(action_specs) > 1 or (
            action_specs and expected is not None and action_specs != {str(expected)}
        ):
            raise ValueError(
                f"robot_tag={tag!r} expects action_spec_id={expected!r}, "
                f"got {sorted(action_specs)}"
            )
        return tag

    def _action_runtime(self, robot_tag: str) -> tuple[nn.Module, int, int]:
        spec = self.embodiment_head_specs[robot_tag]
        return (
            self.action_models[robot_tag],
            int(spec["action_horizon"]),
            int(spec["state_dim"]),
        )

    def _pool_visual_tokens_to_action_queries(
        self,
        visual_tokens: torch.Tensor,
        state: torch.Tensor,
        action_model: nn.Module,
    ) -> torch.Tensor:
        return action_model.decode_action_queries(visual_tokens, state=state)

    def _action_gripper_indices(self, robot_tag: str) -> tuple[int, ...]:
        return tuple(
            int(index)
            for index in self.embodiment_head_specs[robot_tag].get(
                "gripper_indices", ()
            )
        )

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
            raise ValueError("action_valid_mask must be present for every example")
        masks = np.asarray(raw_masks, dtype=np.bool_)
        if masks.ndim != 2 or masks.shape[1] < action_horizon:
            raise ValueError(
                f"action_valid_mask must have shape [B,H>={action_horizon}], got "
                f"{tuple(masks.shape)}"
            )
        return torch.as_tensor(
            masks[:, -action_horizon:], device=device, dtype=torch.bool
        )

    def _condition_task_on_embodiment(
        self, task_embedding: torch.Tensor, robot_tag: str
    ) -> torch.Tensor:
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
    ) -> Optional[torch.Tensor]:
        has_frame_mask = any(
            example.get("future_frame_valid_mask") is not None for example in examples
        )
        if view_valid_mask is None and not has_frame_mask:
            return None
        if view_valid_mask is None:
            token_valid = torch.ones(
                len(examples), self.num_visual_tokens, device=device, dtype=torch.bool
            )
        else:
            token_valid = view_valid_mask.repeat_interleave(
                self.visual_tokens_per_view, dim=1
            )
        frame_masks = []
        for example in examples:
            raw_mask = example.get("future_frame_valid_mask")
            mask = (
                np.ones(3, dtype=np.bool_)
                if raw_mask is None
                else np.asarray(raw_mask, dtype=np.bool_)
            )
            if mask.shape != (3,):
                raise ValueError(
                    f"future_frame_valid_mask must have shape (3,), got {mask.shape}"
                )
            frame_masks.append(mask)
        frame_valid = torch.as_tensor(
            np.stack(frame_masks), device=device, dtype=torch.bool
        )
        if not bool(frame_valid[:, 0].all()):
            raise ValueError("the current frame must always be valid")
        return (frame_valid[:, :, None] & token_valid[:, None, :]).float()

    def _validate_future_time_offsets(self, examples: List[dict]) -> None:
        if self.future_time_offsets_s is None:
            return
        expected = np.asarray(self.future_time_offsets_s, dtype=np.float32)
        for example in examples:
            actual = example.get("future_time_offsets_s")
            actual = np.asarray(actual, dtype=np.float32) if actual is not None else None
            if actual is None or actual.shape != expected.shape or not np.allclose(
                actual, expected
            ):
                raise ValueError(
                    f"robot_tag={example.get('robot_tag')!r} must use "
                    f"future_time_offsets_s={list(self.future_time_offsets_s)}, got {actual}"
                )

    def _pad_inference_views(self, frame) -> tuple[list, list[bool]]:
        views = list(frame) if isinstance(frame, (list, tuple)) else [frame]
        views = [to_pil_preserve(view) for view in views]
        if not views:
            raise ValueError("inference requires at least one camera view")
        if len(views) > self.num_views:
            raise ValueError(
                f"inference provides {len(views)} views, model expects {self.num_views}"
            )
        mask = [True] * len(views) + [False] * (self.num_views - len(views))
        blank = Image.new("RGB", views[0].size)
        views.extend(blank.copy() for _ in range(self.num_views - len(views)))
        return views, mask

    @staticmethod
    def _current_state_tensor(
        examples: List[dict], device: torch.device, state_dim: int
    ) -> torch.Tensor:
        current_states = []
        for example in examples:
            raw_state = example.get("state")
            if raw_state is None:
                raise KeyError("GAWM requires normalized 'state' in every example")
            state = np.asarray(raw_state, dtype=np.float32)
            if state.ndim == 2:
                state = state[0]
            if state.shape != (state_dim,):
                raise ValueError(
                    f"expected state shape ({state_dim},) or (T,{state_dim}), got "
                    f"{state.shape}"
                )
            current_states.append(state)
        return torch.as_tensor(
            np.stack(current_states), device=device, dtype=torch.float32
        )

    def _visual_token_regularization(self, tokens: torch.Tensor):
        x = tokens.float()
        token_count = x.shape[2]
        if token_count < 2:
            zero = x.new_zeros(())
            return zero, zero, x.new_ones(())
        off_diag = ~torch.eye(
            token_count, device=x.device, dtype=torch.bool
        ).view(1, 1, token_count, token_count)
        normalized = F.normalize(x, dim=-1, eps=1e-6)
        cosine = torch.matmul(normalized, normalized.transpose(-1, -2))
        diversity_loss = cosine.square().masked_select(off_diag).mean()
        token_std = x.var(dim=2, unbiased=False).add(1e-4).sqrt()
        variance_loss = F.relu(self.visual_token_min_std - token_std).mean()
        return diversity_loss, variance_loss, cosine.masked_select(off_diag).mean()

    def _embed_task(
        self, instructions: List[str], device: torch.device
    ) -> torch.Tensor:
        return self.task_embedding(instructions, device=device)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("GAWM.forward requires a non-empty example batch")
        instructions = [example["lang"] for example in examples]
        self._validate_future_time_offsets(examples)
        device = next(self.parameters()).device
        robot_tag = self._resolve_batch_embodiment(examples)
        action_model, action_horizon, state_dim = self._action_runtime(robot_tag)
        gripper_indices = self._action_gripper_indices(robot_tag)

        actions = torch.as_tensor(
            np.asarray([example["action"] for example in examples]),
            device=device,
            dtype=torch.float32,
        )
        if actions.ndim != 3 or actions.shape[1] != action_horizon:
            raise ValueError(
                f"robot_tag={robot_tag!r} expects action horizon {action_horizon}, "
                f"got {tuple(actions.shape)}"
            )
        if actions.shape[2] != int(action_model.action_dim):
            raise ValueError(
                f"robot_tag={robot_tag!r} expects action dim {action_model.action_dim}, "
                f"got {actions.shape[2]}"
            )
        action_valid_mask = self._action_valid_mask_tensor(
            examples, device, action_horizon
        )

        frames_per_example = []
        for example in examples:
            future = example.get("future_images")
            if future is None or len(future) != 2:
                raise KeyError("GAWM requires exactly two 'future_images'")
            frames_per_example.append([example["image"], *list(future)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            patch_tokens = self.backbone.encode_patch_frames(frames_per_example)
        if patch_tokens.shape[1] != 3:
            raise ValueError(f"expected three temporal frames, got {patch_tokens.shape[1]}")

        with torch.autocast("cuda", dtype=torch.float32):
            view_valid_mask = self._view_valid_mask_tensor(
                examples, patch_tokens.device
            )
            latent, content_latent = self.visual_token_pooler(
                patch_tokens.float(),
                return_content=True,
                view_valid_mask=view_valid_mask,
            )
            task_emb = self._condition_task_on_embodiment(
                self._embed_task(instructions, device=latent.device), robot_tag
            )
            current_state = self._current_state_tensor(
                examples, latent.device, state_dim
            )
            wm_out = self.world_model(
                latent,
                ctx_len=1,
                goal=task_emb,
                update_stats=True,
                loss_mask=self._wm_loss_mask_tensor(
                    examples, view_valid_mask, latent.device
                ),
            )
            pred_future_latent = wm_out["pred_future_latent"]
            head_tokens = torch.cat([latent[:, :1], pred_future_latent], dim=1)
            pred_content = self.visual_token_pooler.remove_position(
                pred_future_latent, view_valid_mask=view_valid_mask
            )
            source_div, source_var, source_cos = self._visual_token_regularization(
                content_latent
            )
            pred_div, pred_var, pred_cos = self._visual_token_regularization(
                pred_content
            )
            diversity_loss = 0.5 * (source_div + pred_div)
            variance_loss = 0.5 * (source_var + pred_var)
            mean_cosine = 0.5 * (source_cos + pred_cos)

            action_queries = self._pool_visual_tokens_to_action_queries(
                head_tokens, state=current_state, action_model=action_model
            )
            pred_actions = action_model.predict_action(action_queries)
            l1_action_loss = masked_action_l1_loss(
                pred_actions, actions, action_valid_mask
            )
            action_metrics = action_l1_diagnostics(
                pred_actions,
                actions,
                action_valid_mask,
                gripper_indices=gripper_indices,
            )
            total_loss = (
                l1_action_loss
                + self.loss_latent_weight * wm_out["latent_loss"]
                + self.latent_cosine_weight * wm_out["latent_cosine_loss"]
                + self.visual_token_diversity_weight * diversity_loss
                + self.visual_token_variance_weight * variance_loss
            )

        output = {
            "action_loss": total_loss,
            "l1_action_loss": l1_action_loss.detach(),
            "full_l1_action_loss": l1_action_loss.detach(),
            "latent_loss": wm_out["latent_loss"].detach(),
            "latent_cosine_loss": wm_out["latent_cosine_loss"].detach(),
            "visual_token_diversity_loss": diversity_loss.detach(),
            "visual_token_variance_loss": variance_loss.detach(),
            "visual_token_mean_cosine": mean_cosine.detach(),
        }
        output.update({name: value.detach() for name, value in action_metrics.items()})
        for name, value in wm_out.items():
            if name.startswith("latent_loss_horizon_") or name.startswith("delta_"):
                output[name] = value.detach()
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict[str, np.ndarray]:
        if type(examples) is not list:
            examples = [examples]
        robot_tag = self._resolve_batch_embodiment(
            examples, kwargs.get("robot_tag")
        )
        action_model, _, state_dim = self._action_runtime(robot_tag)
        instructions = [example["lang"] for example in examples]
        train_obs_image_size = getattr(
            self.config.datasets.vla_data, "obs_image_size", None
        )
        frames_per_example = []
        inference_view_masks = []
        for example in examples:
            current, inferred_mask = self._pad_inference_views(example["image"])
            if train_obs_image_size:
                current = resize_images(current, target_size=train_obs_image_size)
            frames_per_example.append([current])
            inference_view_masks.append(
                example.get("view_valid_mask", inferred_mask)
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            patch_tokens = self.backbone.encode_patch_frames(frames_per_example)
        with torch.autocast("cuda", dtype=torch.float32):
            view_valid_mask = torch.as_tensor(
                inference_view_masks,
                device=patch_tokens.device,
                dtype=torch.bool,
            )
            latent = self.visual_token_pooler(
                patch_tokens.float(), view_valid_mask=view_valid_mask
            )
            task_emb = self._condition_task_on_embodiment(
                self._embed_task(instructions, device=latent.device), robot_tag
            )
            current_state = self._current_state_tensor(
                examples, latent.device, state_dim
            )
            predicted_future = self.world_model.regress_future(
                latent, goal=task_emb
            )
            action_queries = self._pool_visual_tokens_to_action_queries(
                torch.cat([latent, predicted_future], dim=1),
                state=current_state,
                action_model=action_model,
            )
            pred_actions = action_model.predict_action(action_queries)
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
