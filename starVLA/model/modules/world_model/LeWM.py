# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
LeWM World Model Interface — DINOv3 encoder frontend.

Wraps the LeWorldModel (LeWM) front-end: a raw facebookresearch/dinov3
torchhub checkpoint converted to a HuggingFace ``DINOv3ViTModel``. The
flow-matching predictor of the original LeWM is intentionally dropped — in
starVLA the action head is provided by a separate (OFT) module.

Per-frame latent (matches the LeWM `encode()` convention):
    latent = concat([CLS_token, mean_pool(patch_tokens)])  -> dim = 2 * hidden
so a `vit-tiny` (hidden=192) yields a 384-d latent per view, identical to the
LeWM `embed_dim`.

The wrapper exposes the standard starVLA world-model contract:
  - ``build_inputs(images, instructions)`` -> dict of tensors
  - ``forward(**inputs)``                  -> object with ``.hidden_states``
  - ``model.config.hidden_size``           -> latent dim used by the action head

The ViT is frozen by default; set ``world_model.train_encoder: true`` in the
framework config to keep the joint fine-tuning interface open.
"""

import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _LeWM_Interface(nn.Module):
    """World-model wrapper exposing a ViT encoder as a feature backbone."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get("base_wm")
        if not model_name:
            raise ValueError(
                "framework.world_model.base_wm is required (e.g. "
                "facebook/dinov2-base or a DINOv3 .pth); the legacy "
                "qwenvl.base_vlm / vit-tiny fallback was removed"
            )
        model_path = Path(model_name).expanduser()
        if not model_path.is_absolute() and not model_path.exists():
            repo_relative = Path(__file__).resolve().parents[4] / model_path
            if repo_relative.exists():
                model_path = repo_relative
        model_name = os.fspath(model_path)
        self.config = config
        self.train_encoder = bool(wm_cfg.get("train_encoder", False))

        # DINOv3 is the only supported encoder: raw facebookresearch/dinov3
        # torchhub ``.pth`` files whose token layout is [CLS, register_tokens,
        # patches]. The HF ViT / DINO v1 / DINOv2 AutoModel path was removed.
        is_dinov3_raw = (
            model_name.endswith(".pth")
            and "dinov3" in os.path.basename(model_name).lower()
        )
        if not is_dinov3_raw:
            raise ValueError(
                "LeWM world model now supports only raw DINOv3 checkpoints "
                "(*.pth with 'dinov3' in the filename); "
                f"got {model_name!r}"
            )

        from .dinov3_loader import load_dinov3

        logger.info(f"Loading DINOv3 vision encoder from raw checkpoint {model_name}")
        self.encoder, self.processor, num_register = load_dinov3(model_name)
        self.num_prefix_tokens = 1 + num_register

        vit_hidden = self.encoder.config.hidden_size
        # LeWM latent = concat(cls, mean-pool patches) -> 2 * hidden
        self._hidden_size = vit_hidden * 2

        # Frozen by default; keep joint fine-tuning interface available.
        self.encoder.requires_grad_(self.train_encoder)
        self.encoder.train(self.train_encoder)

        class _FakeConfig:
            pass

        self._model_config = _FakeConfig()
        self._model_config.hidden_size = self._hidden_size

    @property
    def model(self):
        """Compat shim: framework reads self.backbone.model.config.hidden_size."""

        class _ModelShim:
            pass

        shim = _ModelShim()
        shim.config = self._model_config
        return shim

    def _to_pixel_values(self, flat: List, device=None) -> torch.Tensor:
        """Preprocess a flat list of PIL images into a (N, C, H, W) tensor."""
        if self.processor is not None:
            # Run resize/normalize on the encoder's device (GPU) when possible to
            # avoid a CPU-bound bottleneck (worsened by OMP_NUM_THREADS=1) when
            # encoding many views per forward pass.
            try:
                kwargs = {"images": flat, "return_tensors": "pt"}
                if device is not None and getattr(device, "type", None) == "cuda":
                    kwargs["device"] = device
                pixel_values = self.processor(**kwargs).pixel_values
            except TypeError:
                pixel_values = self.processor(images=flat, return_tensors="pt").pixel_values
        else:
            from torchvision.transforms import v2 as T

            tf = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize((224, 224), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            pixel_values = torch.stack([tf(im) for im in flat], dim=0)
        return pixel_values

    def build_inputs(self, images: List, instructions: List, **kwargs):
        """Preprocess multi-view images into ViT pixel values.

        ``images`` is a list of B examples; each example is a single PIL image
        or a list of V views. All views are flattened to (B * V, C, H, W).
        ``instructions`` is ignored (no text encoder in this front-end).
        """
        flat, n_views = [], None
        for sample in images:
            views = sample if isinstance(sample, (list, tuple)) else [sample]
            n_views = len(views) if n_views is None else n_views
            flat.extend(views)

        device = next(self.encoder.parameters()).device
        pixel_values = self._to_pixel_values(flat, device)

        return {
            "pixel_values": pixel_values.to(device),
            "n_views": n_views,
            "_is_wm_input": True,
        }

    def encode_frames(self, frames_per_example: List) -> torch.Tensor:
        """Encode a temporal sequence of multi-view frames into per-frame latents.

        ``frames_per_example`` is a list of B examples; each example is a list
        of T frames; each frame is a single image or a list of V views. All
        ``B * T * V`` views are encoded in a single ViT pass.

        Returns ``latent (B, T, V*2*hidden)`` where each frame's latent is the
        concatenation over its V views of ``concat([CLS, mean-pool(patches)])``.
        The framework projects this back to ``2*hidden`` via a learned
        view-fusion layer (preserving per-camera identity).
        """
        flat, T, V = [], None, None
        for frames in frames_per_example:
            T = len(frames) if T is None else T
            for frame in frames:
                views = frame if isinstance(frame, (list, tuple)) else [frame]
                V = len(views) if V is None else V
                flat.extend(views)

        device = next(self.encoder.parameters()).device
        pixel_values = self._to_pixel_values(flat, device).to(device)

        with torch.set_grad_enabled(self.train_encoder):
            out = self.encoder(pixel_values=pixel_values)
            hidden = out.last_hidden_state            # (B*T*V, prefix+N, D)
            cls = hidden[:, 0]                         # (B*T*V, D)
            pooled = hidden[:, self.num_prefix_tokens:].mean(dim=1)  # (B*T*V, D)
            vec = torch.cat([cls, pooled], dim=-1)     # (B*T*V, 2D)

        B = len(frames_per_example)
        # Concatenate views along the feature dim to preserve per-camera
        # identity (mean-pool would discard which camera saw what). The
        # framework fuses V*2D -> 2D via a learned projection.
        vec = vec.view(B, T, V, vec.shape[-1]).reshape(B, T, V * vec.shape[-1])  # (B, T, V*2D)
        return vec

    def encode_patch_frames(self, frames_per_example: List) -> torch.Tensor:
        """Encode frames into raw per-view patch tokens.

        ``frames_per_example`` has the same structure as ``encode_frames``.
        Returns ``patches (B, T, V, N, D)`` where ``N`` is the encoder patch
        grid length and ``D`` is the ViT/DINO hidden size.
        """
        flat, T, V = [], None, None
        for frames in frames_per_example:
            T = len(frames) if T is None else T
            for frame in frames:
                views = frame if isinstance(frame, (list, tuple)) else [frame]
                V = len(views) if V is None else V
                flat.extend(views)

        device = next(self.encoder.parameters()).device
        pixel_values = self._to_pixel_values(flat, device).to(device)

        return self._encode_patch_pixel_values(
            pixel_values,
            batch_size=len(frames_per_example),
            time_steps=T,
            num_views=V,
        )

    def encode_patch_image_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of raw image tensors into per-view patch tokens.

        This is the replay-friendly counterpart of :meth:`encode_patch_frames`.
        It avoids converting rollout images back to PIL during RL updates while
        preserving the exact HuggingFace DINOv3 image processor used at
        inference time.

        Args:
            images: Raw images shaped ``[B, T, V, H, W, C]`` or
                ``[B, T, V, C, H, W]``. ``uint8`` and floating tensors are
                accepted by the DINOv3 fast image processor.

        Returns:
            Patch tokens shaped ``[B, T, V, N, D]``.
        """
        if not isinstance(images, torch.Tensor) or images.ndim != 6:
            raise ValueError(
                "images must be a rank-6 tensor [B,T,V,H,W,C] or "
                f"[B,T,V,C,H,W], got {type(images).__name__} "
                f"with shape {getattr(images, 'shape', None)}"
            )

        batch_size, time_steps, num_views = images.shape[:3]
        if images.shape[-1] in (1, 3, 4):
            flat = images.reshape(-1, *images.shape[-3:])
        elif images.shape[3] in (1, 3, 4):
            flat = images.reshape(-1, *images.shape[-3:])
        else:
            raise ValueError(
                "Cannot infer image channel axis from shape "
                f"{tuple(images.shape)}; expected C in {{1,3,4}}."
            )

        device = next(self.encoder.parameters()).device
        pixel_values = self._to_pixel_values(flat, device).to(device)
        return self._encode_patch_pixel_values(
            pixel_values,
            batch_size=batch_size,
            time_steps=time_steps,
            num_views=num_views,
        )

    def _encode_patch_pixel_values(
        self,
        pixel_values: torch.Tensor,
        *,
        batch_size: int,
        time_steps: int,
        num_views: int,
    ) -> torch.Tensor:
        """Run DINOv3 on already preprocessed pixels and restore B/T/V axes."""

        with torch.set_grad_enabled(self.train_encoder):
            out = self.encoder(pixel_values=pixel_values)
            patches = out.last_hidden_state[:, self.num_prefix_tokens:]  # (B*T*V, N, D)

        return patches.view(
            batch_size,
            time_steps,
            num_views,
            patches.shape[-2],
            patches.shape[-1],
        )

    def forward(self, **kwargs):
        """Encode views; return hidden states (B, V, 2*hidden) for the action head."""
        kwargs.pop("_is_wm_input", False)
        kwargs.pop("output_hidden_states", False)
        kwargs.pop("return_dict", True)
        n_views = int(kwargs.pop("n_views", 1))
        pixel_values = kwargs["pixel_values"]

        with torch.set_grad_enabled(self.train_encoder):
            out = self.encoder(pixel_values=pixel_values)
            hidden = out.last_hidden_state            # (BV, prefix+N, D)
            cls = hidden[:, 0]                         # (BV, D)
            pooled = hidden[:, self.num_prefix_tokens:].mean(dim=1)  # (BV, D)
            vec = torch.cat([cls, pooled], dim=-1)     # (BV, 2D)

        bv = vec.shape[0]
        b = bv // n_views
        latent = vec.view(b, n_views, vec.shape[-1])   # (B, V, 2D)

        class _WMOutput:
            def __init__(self, hidden_states_tuple):
                self.hidden_states = hidden_states_tuple

        return _WMOutput(hidden_states_tuple=(latent,))
