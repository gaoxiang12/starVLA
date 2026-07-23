"""Compare staged or combined WALA checkpoints on identical LIBERO batches."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import baseframework


METRICS = (
    "transition_teacher_recon_loss",
    "transition_teacher_l1_loss",
    "transition_teacher_cosine_loss",
    "transition_alignment_loss",
    "transition_alignment_cosine_loss",
    "transition_alignment_l1_loss",
    "transition_decode_loss",
    "transition_decode_l1_loss",
    "transition_decode_cosine_loss",
    "l1_action_loss",
    "latent_loss",
    "delta_scale",
)


def _transition_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float,
) -> torch.Tensor:
    l1 = F.smooth_l1_loss(prediction.float(), target.float())
    prediction = F.normalize(prediction.float(), dim=-1, eps=1e-6)
    target = F.normalize(target.float(), dim=-1, eps=1e-6)
    cosine = 1.0 - (prediction * target).sum(dim=-1).mean()
    return l1 + cosine_weight * cosine


def _init_single_process_group() -> None:
    if dist.is_initialized():
        return
    file_descriptor, rendezvous = tempfile.mkstemp(
        prefix="starvla_wala_eval_", suffix=".rdzv"
    )
    os.close(file_descriptor)
    os.unlink(rendezvous)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> tuple[int, int]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = model.remap_checkpoint_state_dict(state)
    model_state = model.state_dict()
    compatible = {
        name: value
        for name, value in state.items()
        if name in model_state and model_state[name].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    transition_loaded = sum(
        name.startswith("transition_auxiliary.") for name in compatible
    )
    if transition_loaded == 0:
        raise RuntimeError(f"{checkpoint} contains no compatible transition weights")
    # Legacy flow-only weights are intentionally absent from the simplified
    # deployed baseline, so report counts instead of requiring global strictness.
    return len(missing), len(unexpected)


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    batches: list[list[dict]],
) -> dict[str, float]:
    dependence_metrics = (
        "transition_teacher_shuffled_recon_loss",
        "transition_teacher_shuffle_gap",
        "transition_teacher_zero_recon_loss",
        "transition_teacher_zero_gap",
    )
    totals = {name: 0.0 for name in (*METRICS, *dependence_metrics)}
    model.eval()
    for batch in batches:
        output = model(batch)
        for name in METRICS:
            totals[name] += float(output[name])
        frames_per_example = [
            [example["image"]] + list(example["future_images"])
            for example in batch
        ]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            patch_tokens = model.backbone.encode_patch_frames(frames_per_example)
        latent = model.visual_token_pooler(patch_tokens.float())
        anchor = latent[:, model.wm_ctx_len - 1].detach()
        future = latent[
            :, model.wm_ctx_len : model.wm_ctx_len + model.n_future
        ].detach()
        scale = model.world_model.delta_scale.detach().clamp_min(
            model.world_model._stats_eps
        )
        target_delta = (future - anchor[:, None]) / scale
        teacher_tokens, prediction = model.transition_auxiliary.teacher_forward(
            anchor, target_delta
        )
        normal = _transition_reconstruction_loss(
            prediction, target_delta, model.transition_cosine_weight
        )
        shuffled_prediction = model.transition_auxiliary.teacher_decoder(
            anchor, teacher_tokens.roll(1, dims=0)
        )
        shuffled = _transition_reconstruction_loss(
            shuffled_prediction, target_delta, model.transition_cosine_weight
        )
        zero_prediction = model.transition_auxiliary.teacher_decoder(
            anchor, torch.zeros_like(teacher_tokens)
        )
        zero = _transition_reconstruction_loss(
            zero_prediction, target_delta, model.transition_cosine_weight
        )
        totals["transition_teacher_shuffled_recon_loss"] += float(shuffled)
        totals["transition_teacher_shuffle_gap"] += float(shuffled - normal)
        totals["transition_teacher_zero_recon_loss"] += float(zero)
        totals["transition_teacher_zero_gap"] += float(zero - normal)
    return {name: value / len(batches) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    for checkpoint in args.checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if args.num_batches < 1 or args.batch_size < 1:
        raise ValueError("num-batches and batch-size must be positive")

    _init_single_process_group()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = baseframework.from_pretrained(str(args.checkpoints[0]))
    if model.transition_mode == "off":
        raise ValueError(
            "expected a checkpoint with transition supervision enabled, "
            f"got mode={model.transition_mode!r}"
        )
    model = model.to(args.device)

    run_dir = args.checkpoints[0].parent.parent
    full_config = run_dir / "config.full.yaml"
    if not full_config.is_file():
        raise FileNotFoundError(full_config)
    cfg = OmegaConf.load(full_config)
    cfg.datasets.vla_data.per_device_batch_size = args.batch_size
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.pin_memory = False
    cfg.output_dir = tempfile.mkdtemp(prefix="starvla_wala_dataset_")
    loader = build_dataloader(
        cfg=cfg,
        dataset_py=cfg.datasets.vla_data.dataset_py,
    )
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(args.num_batches)]

    results = []
    for checkpoint in args.checkpoints:
        missing, unexpected = _load_checkpoint(model, checkpoint)
        metrics = _evaluate(model, batches)
        results.append(
            {
                "checkpoint": str(checkpoint),
                "num_batches": args.num_batches,
                "batch_size": args.batch_size,
                "missing_model_keys": missing,
                "unexpected_loaded_keys": unexpected,
                **metrics,
            }
        )
        print(json.dumps(results[-1], sort_keys=True), flush=True)

    payload = {"seed": args.seed, "results": results}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
