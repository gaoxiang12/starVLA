#!/usr/bin/env python3
"""Calibrate a fixed local-error mean for the action-free dynamics bottleneck.

Run this once with the frozen encoder, spatial pooler, task embedding and 220k
base predictor.  The ctx3 dataset is explicitly split as:

    innovation history: [t-2, t-1, t]
    legacy base anchor:  [0]
    target futures:      [+4, +8]

The resulting fp32 ``mean[1,H,K,D]`` is immutable training data statistics.
It prevents spatial-token identity or co-batch composition from being learned
as "dynamic" latent signal.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm

from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


def _load_config(path: Path, output: Path, batch_size: int):
    cfg = apply_config_compat(OmegaConf.load(path))
    # Calibration uses the exact legacy ctx1 architecture. The new module is
    # intentionally absent so a missing mean file cannot affect construction.
    cfg.framework.world_model.predictable_innovation_enabled = False
    cfg.framework.world_model.ctx_len = 1
    cfg.datasets.vla_data.per_device_batch_size = int(batch_size)
    cfg.datasets.vla_data.num_workers = int(
        cfg.datasets.vla_data.get("num_workers", 4)
    )
    cfg.output_dir = str(output.parent / "loader_metadata")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg


@torch.inference_mode()
def calibrate(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("calibration requires one CUDA GPU")
    if not dist.is_initialized():
        rendezvous = f"file:///tmp/starvla_calibration_dist_{os.getpid()}"
        dist.init_process_group(
            backend="gloo", init_method=rendezvous, rank=0, world_size=1
        )
    device = torch.device("cuda", 0)

    cfg = _load_config(args.config, args.output, args.batch_size)
    torch.manual_seed(int(cfg.get("seed", 42)))
    model = build_framework(cfg)
    model = TrainerUtils.load_pretrained_backbones(
        model, str(args.checkpoint), reload_modules=None
    )
    model.requires_grad_(False).eval().to(device)

    dataloader = build_dataloader(
        cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py
    )
    total_sum = None
    total_square_sum = None
    first_sum = None
    second_sum = None
    first_count = 0
    second_count = 0
    sample_count = 0
    split_at = args.num_samples // 2

    progress = tqdm(
        total=args.num_samples,
        disable=False,
        desc="calibrating fixed local-error mean",
    )
    for examples in dataloader:
        remaining = args.num_samples - sample_count
        if remaining <= 0:
            break
        examples = examples[:remaining]
        frames_per_example = [
            [example["image"]] + list(example["future_images"])
            for example in examples
        ]
        if any(len(frames) != 5 for frames in frames_per_example):
            lengths = [len(frames) for frames in frames_per_example]
            raise ValueError(
                "ctx3 calibration expects exactly [t-2,t-1,t,t+4,t+8]; "
                f"got frame counts {lengths}"
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            patches = model.backbone.encode_patch_frames(frames_per_example)
        latent = model.visual_token_pooler(patches.float())
        task = model._embed_task(
            [example["lang"] for example in examples], device=latent.device
        )

        current = latent[:, 2:3]
        future = latent[:, 3:5]
        base_delta = model.world_model.residual_predictor(
            current, goal=task, state=None
        )
        scale = model.world_model.delta_scale.float().clamp_min(
            model.world_model._stats_eps
        )
        cumulative_target = (future - current) / scale
        cumulative_error = cumulative_target.float() - base_delta.float()
        local_error = torch.cat(
            (
                cumulative_error[:, :1],
                cumulative_error[:, 1:] - cumulative_error[:, :-1],
            ),
            dim=1,
        ).double()

        batch = local_error.shape[0]
        batch_sum = local_error.sum(dim=0, keepdim=True)
        batch_square_sum = local_error.square().sum(dim=0, keepdim=True)
        if total_sum is None:
            total_sum = torch.zeros_like(batch_sum)
            total_square_sum = torch.zeros_like(batch_square_sum)
            first_sum = torch.zeros_like(batch_sum)
            second_sum = torch.zeros_like(batch_sum)
        total_sum += batch_sum
        total_square_sum += batch_square_sum

        first_take = min(batch, max(split_at - sample_count, 0))
        if first_take:
            first_sum += local_error[:first_take].sum(dim=0, keepdim=True)
            first_count += first_take
        if first_take < batch:
            second_sum += local_error[first_take:].sum(dim=0, keepdim=True)
            second_count += batch - first_take

        sample_count += batch
        progress.update(batch)
    progress.close()

    if sample_count != args.num_samples:
        raise RuntimeError(
            f"requested {args.num_samples} samples but collected {sample_count}"
        )
    mean = (total_sum / sample_count).float().cpu()
    second_moment = (total_square_sum / sample_count).float().cpu()
    variance = (second_moment - mean.square()).clamp_min(0.0)
    first_mean = (first_sum / first_count).float().cpu()
    second_mean = (second_sum / second_count).float().cpu()
    half_mean_rms_difference = (first_mean - second_mean).square().mean().sqrt()

    payload = {
        "mean": mean,
        "sample_count": int(sample_count),
        "second_moment": second_moment,
        "std": variance.sqrt(),
        "half_mean_rms_difference": half_mean_rms_difference,
        "frame_indices": torch.tensor([-2, -1, 0, 4, 8]),
        "source_checkpoint": str(args.checkpoint),
        "data_mix": str(cfg.datasets.vla_data.data_mix),
        "delta_scale": model.world_model.delta_scale.detach().float().cpu(),
        "seed": int(cfg.get("seed", 42)),
    }
    for name in ("mean", "second_moment", "std", "half_mean_rms_difference"):
        if not torch.isfinite(torch.as_tensor(payload[name])).all():
            raise RuntimeError(f"calibration statistic {name!r} is non-finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary_output)
    temporary_output.replace(args.output)
    print(f"saved: {args.output}")
    print(f"samples: {sample_count}")
    print(f"mean RMS: {float(mean.square().mean().sqrt()):.8f}")
    print(f"dynamic std mean: {float(payload['std'].mean()):.8f}")
    print(
        "first-half vs second-half mean RMS difference: "
        f"{float(half_mean_rms_difference):.8f}"
    )
    print(f"delta_scale: {float(payload['delta_scale'].item()):.9f}")
    dist.destroy_process_group()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "examples/LIBERO/train_files/"
            "starvla_lewm_oft_dinov3_libero_wm_only_predictable_innovation_10k.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "playground/Checkpoints/"
            "lewm_oft_libero_dinov3b_l10aug489_spatial4x4_trainenc1e6_statecond_220k/"
            "checkpoints/steps_220000_pytorch_model.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "playground/Checkpoints/lewm_oft_libero_dinov3b_localinc_calibration/"
            "train_local_error_mean_8192.pt"
        ),
    )
    parser.add_argument("--num-samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.num_samples < 2 or args.num_samples % 2:
        parser.error("--num-samples must be an even integer >= 2")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    calibrate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
