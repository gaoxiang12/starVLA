# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Build DINOv3 encoders and optionally load original torchhub weights.

The ``dinov3_weights/*.pth`` files are the *original* DINOv3 state_dicts
(keys like ``blocks.0.attn.qkv.weight`` / ``storage_tokens`` / ``ls1.gamma``),
not HuggingFace format. This module converts a raw checkpoint into the
HuggingFace ``DINOv3ViTModel`` key layout and returns a ready-to-use encoder,
its image processor, and the number of register (storage) tokens.

DINOv3 emits tokens as ``[CLS, register_tokens, patch_tokens]``; callers must
skip ``1 + num_register_tokens`` prefix tokens before pooling patch features.
"""

from __future__ import annotations

import os
import re
from typing import Tuple

import torch

# DINOv3 uses a fixed attention head dim of 64 across ViT-S/B/L.
HEAD_DIM = 64

# spec -> architecture hyper-parameters (register tokens = 4 for every variant).
SPECS = {
    "vits16": dict(hidden=384, layers=12, intermediate=1536, gated=False),
    "vits16plus": dict(hidden=384, layers=12, intermediate=1536, gated=True),
    "vitb16": dict(hidden=768, layers=12, intermediate=3072, gated=False),
    "vitl16": dict(hidden=1024, layers=24, intermediate=4096, gated=False),
}


def spec_from_filename(path: str) -> str:
    name = os.path.basename(path)
    m = re.search(r"dinov3_(vit[sbl]16(?:plus)?)_", name)
    if not m:
        raise ValueError(f"cannot infer DINOv3 spec from filename: {name}")
    return m.group(1)


def build_config(spec: str):
    from transformers import DINOv3ViTConfig

    p = SPECS[spec]
    hidden = p["hidden"]
    return DINOv3ViTConfig(
        patch_size=16,
        hidden_size=hidden,
        intermediate_size=p["intermediate"],
        num_hidden_layers=p["layers"],
        num_attention_heads=hidden // HEAD_DIM,
        use_gated_mlp=p["gated"],
        num_register_tokens=4,
        image_size=224,
        rope_theta=100.0,
    )


def convert_state_dict(raw: dict, cfg, gated: bool) -> dict:
    """Map an original DINOv3 torchhub state_dict to HF ``DINOv3ViTModel`` keys."""
    h = cfg.hidden_size
    out: dict = {}

    # --- embeddings / final norm -------------------------------------------
    out["embeddings.cls_token"] = raw["cls_token"]
    # original mask_token is (1, hidden); HF expects (1, 1, hidden).
    out["embeddings.mask_token"] = raw["mask_token"].reshape(1, 1, -1)
    out["embeddings.register_tokens"] = raw["storage_tokens"]
    out["embeddings.patch_embeddings.weight"] = raw["patch_embed.proj.weight"]
    out["embeddings.patch_embeddings.bias"] = raw["patch_embed.proj.bias"]
    out["norm.weight"] = raw["norm.weight"]
    out["norm.bias"] = raw["norm.bias"]
    # ``rope_embed.periods`` is intentionally dropped: HF regenerates inv_freq
    # from ``rope_theta`` as a non-persistent buffer.

    for i in range(cfg.num_hidden_layers):
        src = f"blocks.{i}."
        dst = f"layer.{i}."

        out[dst + "norm1.weight"] = raw[src + "norm1.weight"]
        out[dst + "norm1.bias"] = raw[src + "norm1.bias"]
        out[dst + "norm2.weight"] = raw[src + "norm2.weight"]
        out[dst + "norm2.bias"] = raw[src + "norm2.bias"]

        # fused qkv -> separate q/k/v projections.
        qkv_w = raw[src + "attn.qkv.weight"]  # (3h, h)
        out[dst + "attention.q_proj.weight"] = qkv_w[0:h]
        out[dst + "attention.k_proj.weight"] = qkv_w[h : 2 * h]
        out[dst + "attention.v_proj.weight"] = qkv_w[2 * h : 3 * h]

        # bias_mask zeroes out the key bias (DINOv3 uses key_bias=False); apply
        # it before splitting so the effective bias matches HF's q/v-only bias.
        qkv_b = raw[src + "attn.qkv.bias"]
        mask = raw.get(src + "attn.qkv.bias_mask")
        if mask is not None:
            qkv_b = qkv_b * mask
        out[dst + "attention.q_proj.bias"] = qkv_b[0:h]
        out[dst + "attention.v_proj.bias"] = qkv_b[2 * h : 3 * h]
        # k_proj has no bias parameter in HF config -> nothing to assign.

        out[dst + "attention.o_proj.weight"] = raw[src + "attn.proj.weight"]
        out[dst + "attention.o_proj.bias"] = raw[src + "attn.proj.bias"]

        out[dst + "layer_scale1.lambda1"] = raw[src + "ls1.gamma"]
        out[dst + "layer_scale2.lambda1"] = raw[src + "ls2.gamma"]

        if gated:
            # SwiGLU: w1 -> gate_proj, w2 -> up_proj, w3 -> down_proj.
            out[dst + "mlp.gate_proj.weight"] = raw[src + "mlp.w1.weight"]
            out[dst + "mlp.gate_proj.bias"] = raw[src + "mlp.w1.bias"]
            out[dst + "mlp.up_proj.weight"] = raw[src + "mlp.w2.weight"]
            out[dst + "mlp.up_proj.bias"] = raw[src + "mlp.w2.bias"]
            out[dst + "mlp.down_proj.weight"] = raw[src + "mlp.w3.weight"]
            out[dst + "mlp.down_proj.bias"] = raw[src + "mlp.w3.bias"]
        else:
            out[dst + "mlp.up_proj.weight"] = raw[src + "mlp.fc1.weight"]
            out[dst + "mlp.up_proj.bias"] = raw[src + "mlp.fc1.bias"]
            out[dst + "mlp.down_proj.weight"] = raw[src + "mlp.fc2.weight"]
            out[dst + "mlp.down_proj.bias"] = raw[src + "mlp.fc2.bias"]

    return out


def build_dinov3(spec: str) -> Tuple[object, object, int]:
    """Build an uninitialized HF DINOv3 encoder for a known architecture.

    This path is used when a complete starVLA checkpoint supplies the encoder
    weights. It deliberately performs no file or network access.
    """
    from transformers import DINOv3ViTImageProcessorFast, DINOv3ViTModel

    if spec not in SPECS:
        raise ValueError(
            f"unsupported DINOv3 encoder_spec {spec!r}; "
            f"expected one of {sorted(SPECS)}"
        )
    cfg = build_config(spec)
    return (
        DINOv3ViTModel(cfg),
        DINOv3ViTImageProcessorFast(),
        cfg.num_register_tokens,
    )


def load_dinov3(path: str) -> Tuple[object, object, int]:
    """Build a HF ``DINOv3ViTModel`` from a raw torchhub ``.pth`` checkpoint.

    Returns ``(encoder, image_processor, num_register_tokens)``.
    """
    spec = spec_from_filename(path)
    gated = SPECS[spec]["gated"]
    model, processor, num_register_tokens = build_dinov3(spec)
    cfg = model.config
    raw = torch.load(path, map_location="cpu")
    hf_sd = convert_state_dict(raw, cfg, gated)
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)

    # ``k_proj.bias`` and rope buffers are legitimately absent from our converted
    # dict; anything else missing/unexpected indicates a broken mapping.
    real_missing = [k for k in missing if not (k.endswith("k_proj.bias") or "rope" in k.lower())]
    if real_missing or unexpected:
        raise RuntimeError(
            f"DINOv3 state_dict mapping mismatch for {os.path.basename(path)}: "
            f"missing={real_missing[:8]} unexpected={unexpected[:8]}"
        )

    return model, processor, num_register_tokens
