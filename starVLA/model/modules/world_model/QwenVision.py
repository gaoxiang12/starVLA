"""Frozen Qwen3-VL vision tower interface for spatial latent world models."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _QwenVision_Interface(nn.Module):
    """Load only Qwen3-VL's frozen vision tower, without the language model."""

    checkpoint_prefix = "model.visual."

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoImageProcessor
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get("base_wm", "Qwen/Qwen3-VL-4B-Instruct")
        if bool(wm_cfg.get("train_encoder", False)):
            raise ValueError("Qwen vision encoder fine-tuning is not supported; set train_encoder=false")

        self.config = config
        self.train_encoder = False
        self.image_size = int(wm_cfg.get("encoder_image_size", 256))
        self.policy_hidden_size = int(wm_cfg.get("policy_hidden_dim", 1536))
        if self.image_size % 32 != 0:
            raise ValueError(
                "Qwen encoder_image_size must be divisible by patch_size * spatial_merge_size "
                f"(32), got {self.image_size}"
            )

        logger.info(f"Loading frozen Qwen3-VL vision encoder from {model_name}")
        full_config = AutoConfig.from_pretrained(model_name)
        if full_config.model_type != "qwen3_vl":
            raise ValueError(
                f"QwenVision requires a Qwen3-VL checkpoint, got model_type={full_config.model_type!r}"
            )
        vision_config = full_config.vision_config
        vision_config._attn_implementation = wm_cfg.get("attn_implementation", "sdpa")

        with torch.device("meta"):
            self.encoder = Qwen3VLVisionModel(vision_config)
        self._load_visual_weights(self.encoder, model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

        self.merge_size = int(vision_config.spatial_merge_size)
        self.feature_dim = int(vision_config.out_hidden_size)
        self._model_config = SimpleNamespace(hidden_size=self.policy_hidden_size)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @property
    def model(self):
        """Compatibility shim for framework code that reads model.config."""
        return SimpleNamespace(config=self._model_config)

    @classmethod
    def _load_visual_weights(cls, encoder: nn.Module, model_name: str) -> None:
        """Materialize a meta vision tower from only ``model.visual.*`` tensors."""
        from safetensors import safe_open

        model_path = Path(model_name).expanduser()
        if not model_path.is_dir():
            raise ValueError(
                "QwenVision currently requires a local checkpoint directory so it can load only "
                f"the vision weights, got {model_name!r}"
            )

        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            visual_weight_map = {
                key: model_path / shard
                for key, shard in weight_map.items()
                if key.startswith(cls.checkpoint_prefix)
            }
        else:
            weight_path = model_path / "model.safetensors"
            if not weight_path.exists():
                raise FileNotFoundError(
                    f"no model.safetensors index or weight file found under {model_path}"
                )
            with safe_open(weight_path, framework="pt", device="cpu") as handle:
                visual_weight_map = {
                    key: weight_path
                    for key in handle.keys()
                    if key.startswith(cls.checkpoint_prefix)
                }

        expected_keys = set(encoder.state_dict())
        checkpoint_keys = {
            key[len(cls.checkpoint_prefix) :] for key in visual_weight_map
        }
        missing = expected_keys - checkpoint_keys
        unexpected = checkpoint_keys - expected_keys
        if missing or unexpected:
            raise RuntimeError(
                "Qwen vision checkpoint mapping mismatch: "
                f"missing={sorted(missing)[:8]} unexpected={sorted(unexpected)[:8]}"
            )

        state_dict = {}
        by_shard = {}
        for full_key, shard in visual_weight_map.items():
            by_shard.setdefault(shard, []).append(full_key)
        for shard, keys in by_shard.items():
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for full_key in keys:
                    state_dict[full_key[len(cls.checkpoint_prefix) :]] = handle.get_tensor(full_key)
        encoder.load_state_dict(state_dict, strict=True, assign=True)
        rope_dim = encoder.rotary_pos_emb.inv_freq.numel() * 2
        encoder.rotary_pos_emb.inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @staticmethod
    def _flatten_frames(frames_per_example: List):
        flat = []
        num_frames = None
        num_views = None
        for frames in frames_per_example:
            if num_frames is None:
                num_frames = len(frames)
            elif len(frames) != num_frames:
                raise ValueError("all examples must contain the same number of frames")
            for frame in frames:
                views = frame if isinstance(frame, (list, tuple)) else [frame]
                if num_views is None:
                    num_views = len(views)
                elif len(views) != num_views:
                    raise ValueError("all frames must contain the same number of views")
                flat.extend(views)
        return flat, int(num_frames or 0), int(num_views or 0)

    def _preprocess(self, flat: List, device: torch.device):
        image_area = self.image_size * self.image_size
        inputs = self.processor(
            images=flat,
            return_tensors="pt",
            size={"shortest_edge": image_area, "longest_edge": image_area},
        )
        return (
            inputs.pixel_values.to(device=device, dtype=self.encoder.dtype),
            inputs.image_grid_thw.to(device=device),
        )

    def _reshape_tokens(
        self,
        image_embeds: torch.Tensor,
        grid_thw: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_views: int,
    ) -> torch.Tensor:
        merged_grids = grid_thw.clone()
        merged_grids[:, 1:] = merged_grids[:, 1:] // self.merge_size
        if torch.any(merged_grids[:, 0] != 1):
            raise ValueError(f"expected image temporal grid size 1, got {grid_thw.tolist()}")
        spatial_shapes = merged_grids[:, 1:]
        if not torch.all(spatial_shapes == spatial_shapes[0]):
            raise ValueError(f"all Qwen image grids must match, got {grid_thw.tolist()}")
        if int(spatial_shapes[0, 0]) != int(spatial_shapes[0, 1]):
            raise ValueError(f"Qwen visual tokens must form a square grid, got {grid_thw.tolist()}")

        tokens_per_image = merged_grids.prod(dim=-1)
        expected_images = batch_size * num_frames * num_views
        if len(tokens_per_image) != expected_images:
            raise ValueError(
                f"expected {expected_images} image grids, got {len(tokens_per_image)}"
            )
        if image_embeds.shape[0] != int(tokens_per_image.sum()):
            raise ValueError(
                f"Qwen returned {image_embeds.shape[0]} tokens for grids requiring "
                f"{int(tokens_per_image.sum())}"
            )

        tokens_per_image_value = int(tokens_per_image[0])
        return image_embeds.reshape(
            batch_size,
            num_frames,
            num_views,
            tokens_per_image_value,
            self.feature_dim,
        )

    def _encode_pixels(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_views: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            image_embeds, _ = self.encoder(pixel_values, grid_thw=grid_thw)
        return self._reshape_tokens(
            image_embeds, grid_thw, batch_size, num_frames, num_views
        )

    def encode_patch_frames(self, frames_per_example: List) -> torch.Tensor:
        """Return merger-output tokens as ``(B, T, V, N, C)``."""
        flat, num_frames, num_views = self._flatten_frames(frames_per_example)
        if not flat:
            raise ValueError("frames_per_example must not be empty")
        device = next(self.encoder.parameters()).device
        pixel_values, grid_thw = self._preprocess(flat, device)
        return self._encode_pixels(
            pixel_values,
            grid_thw,
            len(frames_per_example),
            num_frames,
            num_views,
        )

    def build_inputs(self, images: List, instructions: List, **kwargs):
        frames_per_example = [
            [[*sample]] if isinstance(sample, (list, tuple)) else [[[sample]]]
            for sample in images
        ]
        flat, _, num_views = self._flatten_frames(frames_per_example)
        device = next(self.encoder.parameters()).device
        pixel_values, grid_thw = self._preprocess(flat, device)
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "n_views": num_views,
            "_is_wm_input": True,
        }

    def forward(self, **kwargs):
        kwargs.pop("_is_wm_input", False)
        num_views = int(kwargs.pop("n_views", 1))
        pixel_values = kwargs["pixel_values"]
        grid_thw = kwargs["image_grid_thw"]
        batch_size = len(grid_thw) // num_views
        tokens = self._encode_pixels(
            pixel_values, grid_thw, batch_size, 1, num_views
        )
        return SimpleNamespace(
            hidden_states=(tokens.reshape(batch_size, num_views * tokens.shape[3], self.feature_dim),)
        )
