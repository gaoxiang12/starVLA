#!/usr/bin/env python
"""Standalone sanity check for the raw DINOv3 checkpoints in ``dinov3_weights/``.

These files are the *original* facebookresearch/dinov3 torchhub state_dicts
(keys like ``blocks.0.attn.qkv.weight`` / ``storage_tokens`` / ``ls1.gamma``),
not HuggingFace format. This script converts a raw checkpoint into the
HuggingFace ``DINOv3ViTModel`` key layout, loads it, and runs a forward pass on
a dummy image to verify that:

  1. the weights load with no *unexpected* real parameters left over, and
  2. the model produces a sane ``last_hidden_state`` of shape
     ``[B, 1 (CLS) + num_register_tokens + num_patches, hidden]``.

Usage:
    python scripts/try_load_dinov3.py                 # test all 4 checkpoints
    python scripts/try_load_dinov3.py --spec vitb16   # test just ViT-B/16
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from starVLA.model.modules.world_model.dinov3_loader import (
    SPECS,
    load_dinov3,
    spec_from_filename,
)


def check_one(path: str, device: torch.device) -> bool:
    spec = spec_from_filename(path)
    print("=" * 78)
    print(f"[{spec}] {os.path.basename(path)}")

    model, processor, num_register = load_dinov3(path)
    model = model.to(device).eval()
    cfg = model.config

    # --- forward on a dummy image ------------------------------------------
    dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    inputs = processor(images=[dummy], return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs)

    lhs = out.last_hidden_state
    n_patches = (cfg.image_size // cfg.patch_size) ** 2
    expected_tokens = 1 + num_register + n_patches
    print(f"  pixel_values: {tuple(inputs['pixel_values'].shape)}")
    print(f"  last_hidden_state: {tuple(lhs.shape)} "
          f"(expected = 1 CLS + {num_register} reg + {n_patches} patch = {expected_tokens})")
    print(f"  stats: mean={lhs.mean().item():+.4f} std={lhs.std().item():.4f} "
          f"has_nan={bool(torch.isnan(lhs).any())}")

    ok = (
        lhs.shape[1] == expected_tokens
        and lhs.shape[-1] == cfg.hidden_size
        and not bool(torch.isnan(lhs).any())
    )
    print(f"  RESULT: {'OK' if ok else 'FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", choices=list(SPECS), help="only test this variant")
    ap.add_argument("--weights-dir", default="dinov3_weights")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}")

    files = sorted(glob.glob(os.path.join(args.weights_dir, "*.pth")))
    if args.spec:
        files = [f for f in files if spec_from_filename(f) == args.spec]
    if not files:
        raise SystemExit(f"no .pth found in {args.weights_dir}")

    results = {os.path.basename(f): check_one(f, device) for f in files}
    print("=" * 78)
    print("SUMMARY:")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
