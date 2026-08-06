#!/usr/bin/env python3
"""Overfit the Stage-B predictor on a frozen, encoded anchor set.

This is a diagnostic, not a generalization experiment.  It deliberately removes
the mixture sampler, video decoding, DINO encoder, and latent-target construction
from the optimization loop.  If the original predictor cannot memorize these
fixed tensors, the Stage-B failure may be an implementation/optimization issue.
If it can, this rules out a basic capacity/gradient-path failure and motivates
a separate generalization test of whether the code is action-free predictable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat


DEFAULT_CONFIG = (
    "examples/LIBERO/train_files/"
    "starvla_lewm_oft_dinov3_libero_wm_only_predictable_innovation_10k.yaml"
)
DEFAULT_CHECKPOINT = (
    "playground/Checkpoints/"
    "lewm_oft_libero_dinov3b_wmonly_localbasis_m4r64_fixedmean_2k/"
    "checkpoints/steps_2000_pytorch_model.pt"
)
DEFAULT_OUTPUT = (
    "playground/Checkpoints/"
    "lewm_oft_libero_dinov3b_localinc_stageb_overfit128_fp32_seed42_2k"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument(
        "--raw-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional Stage-B raw-latent loss weight. The capacity probe defaults "
            "to zero so it directly optimizes the frozen target-code gate."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pass-nmse", type=float, default=0.10)
    parser.add_argument("--pass-cosine", type=float, default=0.90)
    parser.add_argument("--pass-realized-headroom", type=float, default=0.80)
    parser.add_argument("--pass-patience", type=int, default=3)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "anchors",
        "encode_batch_size",
        "train_batch_size",
        "eval_batch_size",
        "steps",
        "eval_interval",
        "pass_patience",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive")
    if args.raw_loss_weight < 0:
        raise ValueError("--raw-loss-weight must be non-negative")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("only scalar tensors can be serialized as metrics")
        return value.detach().float().cpu().item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_jsonable, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(payload, sort_keys=True, default=_jsonable, allow_nan=False)
            + "\n"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_jsonable
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _sample_image_sha256(sample: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    frames = [sample["image"]] + list(sample["future_images"])
    for frame_index, views in enumerate(frames):
        for view_index, image in enumerate(views):
            array = np.asarray(image)
            digest.update(f"{frame_index}:{view_index}".encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _load_model(
    config_path: Path, checkpoint_path: Path, device: torch.device
) -> tuple[Any, Any]:
    cfg = OmegaConf.load(config_path)
    cfg = apply_config_compat(cfg)
    torch.manual_seed(int(cfg.get("seed", 42)))
    model = build_framework(cfg)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = model.remap_checkpoint_state_dict(state_dict)
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    mismatched = sorted(
        key
        for key in set(model_state).intersection(state_dict)
        if tuple(model_state[key].shape) != tuple(state_dict[key].shape)
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "checkpoint is not an exact architecture match: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"mismatched={mismatched[:8]}"
        )
    model.load_state_dict(state_dict, strict=True)
    model.requires_grad_(False)
    model.eval()
    model.to(device)

    bottleneck = getattr(model.world_model, "predictable_innovation", None)
    if bottleneck is None:
        raise RuntimeError("configured model has no predictable-innovation bottleneck")
    if float(bottleneck.fixed_local_error_count.item()) <= 0:
        raise RuntimeError("checkpoint has no calibrated fixed local-error mean")
    calibration_scale = getattr(model, "_innovation_calibration_delta_scale", None)
    current_scale = float(model.world_model.delta_scale.detach().item())
    if calibration_scale is not None and not math.isclose(
        current_scale, float(calibration_scale), rel_tol=1e-6, abs_tol=1e-6
    ):
        raise RuntimeError(
            "fixed-mean calibration delta_scale mismatch: "
            f"checkpoint={current_scale:.9f}, calibration={float(calibration_scale):.9f}"
        )
    return cfg, model


def _valid_step_bounds(
    dataset: Any, trajectory_index: int
) -> tuple[int, int, list[int]]:
    offsets: list[int] = []
    for modality_keys in dataset.modality_keys.values():
        for key in modality_keys:
            if key in dataset.delta_indices:
                offsets.extend(int(value) for value in dataset.delta_indices[key])
    if not offsets:
        raise RuntimeError(f"dataset {dataset.dataset_name!r} has no temporal offsets")
    length = int(dataset.trajectory_lengths[trajectory_index])
    first = max(0, -min(offsets))
    last = length - 1 - max(offsets)
    return first, last, sorted(set(offsets))


def _sample_anchor(
    mixture: Any,
    rng: np.random.Generator,
    used_episodes: set[tuple[int, int]],
    expected_noncurrent_frames: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for _ in range(10000):
        dataset_index = int(
            rng.choice(len(mixture.datasets), p=mixture.dataset_sampling_weights)
        )
        dataset = mixture.datasets[dataset_index]
        trajectory_index = int(
            rng.choice(
                len(dataset.trajectory_ids),
                p=mixture.trajectory_sampling_weights[dataset_index],
            )
        )
        first, last, temporal_offsets = _valid_step_bounds(dataset, trajectory_index)
        if last < first:
            continue
        step = int(rng.integers(first, last + 1))
        trajectory_id = int(dataset.trajectory_ids[trajectory_index])
        episode_identity = (dataset_index, trajectory_id)
        if episode_identity in used_episodes:
            continue

        # Fail fast after choosing a valid identity. Silently replacing a
        # decode failure would make the same seed refer to different anchors as
        # the underlying dataset is repaired.
        raw_data = dataset.get_step_data(trajectory_id, step)
        transformed = dataset.transforms(raw_data)
        sample = dataset._pack_sample(transformed)
        if len(sample.get("future_images", ())) != expected_noncurrent_frames:
            raise RuntimeError(
                f"anchor {dataset.dataset_name}:{trajectory_id}:{step} did not "
                f"produce {expected_noncurrent_frames} non-current frames"
            )

        used_episodes.add(episode_identity)
        manifest = {
            "anchor_index": len(used_episodes) - 1,
            "dataset_index": dataset_index,
            "dataset_name": str(dataset.dataset_name),
            "dataset_path": str(Path(dataset.dataset_path).resolve()),
            "trajectory_id": trajectory_id,
            "trajectory_index": trajectory_index,
            "step": step,
            "valid_step_first": first,
            "valid_step_last": last,
            "temporal_offsets": temporal_offsets,
            "instruction": str(sample.get("lang", "")),
            "images_sha256": _sample_image_sha256(sample),
        }
        return sample, manifest
    raise RuntimeError("could not sample a unique, unpadded anchor after 10000 tries")


@torch.inference_mode()
def _encode_batch(model: Any, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    frames_per_example = [
        [example["image"]] + list(example["future_images"]) for example in examples
    ]
    device = next(model.parameters()).device
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        patch_tokens = model.backbone.encode_patch_frames(frames_per_example)

    latent = model.visual_token_pooler(patch_tokens.float())
    goal = model._embed_task(
        [str(example["lang"]) for example in examples], device=latent.device
    )
    legacy_sequence, innovation_context, _ = model._select_innovation_training_sequence(
        latent
    )
    if innovation_context is None:
        raise RuntimeError("innovation context was not produced")

    anchor = legacy_sequence[:, :1]
    future = legacy_sequence[:, 1 : 1 + model.n_future]
    scale = model.world_model.delta_scale.detach().float().clamp_min(
        model.world_model._stats_eps
    )
    target_delta = (future - anchor).float() / scale
    base_prediction = model.world_model.residual_predictor(
        legacy_sequence[:, : model.wm_ctx_len], goal=goal, state=None
    ).float()

    bottleneck = model.world_model.predictable_innovation
    channel_basis = bottleneck.basis()
    spatial_basis = bottleneck.spatial_basis()
    cumulative_error = target_delta - base_prediction
    local_error = bottleneck._to_local(cumulative_error)
    dynamic_local_error = bottleneck._center_local_error(local_error)
    target_code = bottleneck._project(
        dynamic_local_error, spatial_basis, channel_basis
    )
    native = bottleneck(
        innovation_context,
        base_prediction,
        target_delta=target_delta,
        goal=goal,
    )
    native_difference = (native["target_code"].float() - target_code).abs().max()
    if float(native_difference) > 1e-6:
        raise RuntimeError(
            "cached target code differs from the production bottleneck path: "
            f"max_abs={float(native_difference):.3e}"
        )
    return {
        "context": innovation_context.detach().float().cpu(),
        "base_prediction": base_prediction.detach().float().cpu(),
        "goal": goal.detach().float().cpu(),
        "target_delta": target_delta.detach().float().cpu(),
        "target_code": target_code.detach().float().cpu(),
    }


def _build_anchor_cache(
    cfg: Any,
    model: Any,
    *,
    count: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    data_cfg = cfg.datasets.vla_data
    mixture = get_vla_dataset(
        data_cfg=data_cfg,
        mode="val",
        balance_dataset_weights=bool(data_cfg.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(
            data_cfg.get("balance_trajectory_weights", False)
        ),
        seed=seed,
    )
    for dataset in mixture.datasets:
        dataset.transforms.eval()

    rng = np.random.default_rng(seed)
    # One anchor per episode prevents a short temporal neighborhood from making
    # the memorization test artificially easy.
    used_episodes: set[tuple[int, int]] = set()
    manifest: list[dict[str, Any]] = []
    chunks: dict[str, list[torch.Tensor]] = {}
    pending: list[dict[str, Any]] = []
    expected_noncurrent_frames = model.innovation_context_len + model.n_future - 1
    while len(manifest) < count:
        sample, record = _sample_anchor(
            mixture, rng, used_episodes, expected_noncurrent_frames
        )
        record["anchor_index"] = len(manifest)
        manifest.append(record)
        pending.append(sample)
        if len(pending) == batch_size or len(manifest) == count:
            encoded = _encode_batch(model, pending)
            for name, tensor in encoded.items():
                chunks.setdefault(name, []).append(tensor)
            pending.clear()
            print(f"cached {len(manifest)}/{count} anchors", flush=True)

    cache = {name: torch.cat(values, dim=0) for name, values in chunks.items()}
    if any(tensor.shape[0] != count for tensor in cache.values()):
        raise RuntimeError("anchor cache tensors have inconsistent sample counts")
    non_finite = [
        name for name, tensor in cache.items() if not torch.isfinite(tensor).all()
    ]
    if non_finite:
        raise FloatingPointError(f"anchor cache contains non-finite tensors: {non_finite}")
    return cache, manifest


@torch.inference_mode()
def _evaluate(
    predictor: torch.nn.Module,
    bottleneck: Any,
    cache: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> dict[str, float]:
    was_training = predictor.training
    predictor.eval()
    predictions = []
    count = cache["context"].shape[0]
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        predictions.append(
            predictor(
                cache["context"][start:stop],
                cache["base_prediction"][start:stop],
                goal=cache["goal"][start:stop],
            ).float()
        )
    predicted_code = torch.cat(predictions, dim=0)
    target_code = cache["target_code"]
    code_error = predicted_code - target_code
    target_energy = target_code.square().mean(dim=(0, 2, 3))
    nmse_per_horizon = code_error.square().mean(dim=(0, 2, 3)) / target_energy.clamp_min(
        bottleneck.eps
    )
    cosine = F.cosine_similarity(
        predicted_code, target_code, dim=-1, eps=bottleneck.eps
    ).mean()

    channel_basis = bottleneck.basis()
    spatial_basis = bottleneck.spatial_basis()
    predicted_local = bottleneck._decode(
        predicted_code, spatial_basis, channel_basis
    )
    target_local = bottleneck._decode(target_code, spatial_basis, channel_basis)
    predicted_cumulative = bottleneck._to_cumulative(predicted_local)
    oracle_cumulative = bottleneck._to_cumulative(target_local)
    base_prediction = cache["base_prediction"]
    target_delta = cache["target_delta"]
    base_mse = (base_prediction - target_delta).square().mean()
    final_mse = (
        base_prediction + predicted_cumulative - target_delta
    ).square().mean()
    oracle_mse = (base_prediction + oracle_cumulative - target_delta).square().mean()
    headroom = base_mse - oracle_mse
    improvement = base_mse - final_mse
    minimum_headroom = torch.maximum(base_mse * 1e-4, base_mse.new_tensor(1e-8))
    if bool(headroom <= minimum_headroom):
        raise RuntimeError(
            "fixed anchor set has non-positive/negligible oracle headroom: "
            f"base={float(base_mse):.6f}, oracle={float(oracle_mse):.6f}"
        )

    predicted_diag = bottleneck._distribution_diagnostics(predicted_code)
    target_diag = bottleneck._distribution_diagnostics(target_code)
    metrics = {
        "code_nmse": float(nmse_per_horizon.mean()),
        "code_nmse_horizon_1": float(nmse_per_horizon[0]),
        "code_nmse_horizon_2": float(nmse_per_horizon[1]),
        "code_cosine": float(cosine),
        "base_mse": float(base_mse),
        "final_mse": float(final_mse),
        "oracle_mse": float(oracle_mse),
        "improvement": float(improvement),
        "oracle_headroom": float(headroom),
        "realized_headroom_fraction": float(improvement / headroom),
        "predicted_std": float(predicted_diag["std"]),
        "target_std": float(target_diag["std"]),
        "predicted_effective_rank": float(predicted_diag["effective_rank"]),
        "target_effective_rank": float(target_diag["effective_rank"]),
        "predicted_abs_max": float(predicted_code.abs().max()),
    }
    if was_training:
        predictor.train()
    return metrics


def _next_indices(
    order: torch.Tensor,
    cursor: int,
    *,
    batch_size: int,
    count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    pieces = []
    needed = batch_size
    while needed > 0:
        available = count - cursor
        take = min(available, needed)
        pieces.append(order[cursor : cursor + take])
        cursor += take
        needed -= take
        if cursor == count:
            order = torch.randperm(count, generator=generator)
            cursor = 0
    return torch.cat(pieces), order, cursor


def _passes(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        metrics["code_nmse"] <= args.pass_nmse
        and metrics["code_cosine"] >= args.pass_cosine
        and metrics["realized_headroom_fraction"] >= args.pass_realized_headroom
    )


def _gradient_norm(
    predictor: torch.nn.Module, *, include_output: bool
) -> float:
    squared = None
    for name, parameter in predictor.named_parameters():
        is_output = name.startswith("output.")
        if is_output != include_output or parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return float(squared.sqrt()) if squared is not None else 0.0


def _run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"missing config/checkpoint: {config_path}, {checkpoint_path}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to use a non-empty diagnostic directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    running_path = output_dir / "STATUS.running"
    running_path.write_text(f"pid={Path('/proc/self').resolve().name}\n", encoding="utf-8")
    _write_json(output_dir / "arguments.json", vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    started = time.time()
    cfg, model = _load_model(config_path, checkpoint_path, device)
    bottleneck = model.world_model.predictable_innovation
    cache, manifest = _build_anchor_cache(
        cfg,
        model,
        count=args.anchors,
        batch_size=args.encode_batch_size,
        seed=args.seed,
    )
    checkpoint_sha256 = _sha256(checkpoint_path)
    config_sha256 = _sha256(config_path)
    workspace_root = Path(__file__).resolve().parents[3]
    source_paths = (
        Path(__file__).resolve(),
        workspace_root
        / "starVLA/model/modules/world_model/predictable_innovation_bottleneck.py",
        workspace_root / "starVLA/model/framework/WM4A/LeWMOFT.py",
    )
    source_code_sha256 = {str(path): _sha256(path) for path in source_paths}
    cache_sha256 = {name: _tensor_sha256(tensor) for name, tensor in cache.items()}
    seen_inputs: dict[str, tuple[int, str]] = {}
    for index, record in enumerate(manifest):
        input_digest = hashlib.sha256()
        for name in ("context", "base_prediction", "goal"):
            input_digest.update(bytes.fromhex(_tensor_sha256(cache[name][index])))
        input_sha256 = input_digest.hexdigest()
        target_sha256 = _tensor_sha256(cache["target_code"][index])
        if input_sha256 in seen_inputs:
            previous_index, previous_target = seen_inputs[input_sha256]
            if previous_target != target_sha256:
                raise RuntimeError(
                    "identical encoded predictor input has conflicting targets: "
                    f"anchors {previous_index} and {index}"
                )
            raise RuntimeError(
                "duplicate encoded predictor input in fixed anchor set: "
                f"anchors {previous_index} and {index}"
            )
        seen_inputs[input_sha256] = (index, target_sha256)
        record["encoded_input_sha256"] = input_sha256
        record["target_code_sha256"] = target_sha256

    manifest_payload = {
        "metadata": {
            "data_mix": str(cfg.datasets.vla_data.data_mix),
            "source_config": str(config_path),
            "source_config_sha256": config_sha256,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": checkpoint_sha256,
            "source_code_sha256": source_code_sha256,
            "cache_tensor_sha256": cache_sha256,
        },
        "anchors": manifest,
    }
    cache_path = output_dir / "anchors.pt"
    torch.save(
        {
            "cache": cache,
            "config": {
                "source_config": str(config_path),
                "source_checkpoint": str(checkpoint_path),
                "source_config_sha256": config_sha256,
                "source_checkpoint_sha256": checkpoint_sha256,
                "source_code_sha256": source_code_sha256,
                "cache_tensor_sha256": cache_sha256,
                "anchor_count": args.anchors,
                "seed": args.seed,
            },
        },
        cache_path,
    )
    manifest_payload["metadata"]["cache_file_sha256"] = _sha256(cache_path)
    manifest_payload["manifest_sha256"] = _payload_sha256(manifest_payload)
    _write_json(output_dir / "anchors_manifest.json", manifest_payload)

    cache = {name: tensor.to(device) for name, tensor in cache.items()}
    predictor = bottleneck.predictor
    predictor.requires_grad_(True)
    predictor.train()
    trainable = sum(parameter.numel() for parameter in predictor.parameters())
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        fused=device.type == "cuda",
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    order = torch.randperm(args.anchors, generator=generator)
    cursor = 0
    # The gate is aggregated over the complete fixed set, so use the same
    # deterministic per-horizon normalization during every minibatch update.
    # Re-estimating this denominator from 16 samples makes the capacity probe
    # needlessly noisy and does not change the target mapping being tested.
    global_target_energy = cache["target_code"].square().mean(dim=(0, 2, 3)).detach()

    initial_metrics = _evaluate(
        predictor, bottleneck, cache, batch_size=args.eval_batch_size
    )
    initial_invariants = {
        "zero_output": initial_metrics["predicted_abs_max"] <= 1e-8,
        "unit_nmse": abs(initial_metrics["code_nmse"] - 1.0) <= 1e-6,
        "final_equals_base": abs(
            initial_metrics["final_mse"] - initial_metrics["base_mse"]
        )
        <= 1e-7,
        "target_noncollapsed": initial_metrics["target_std"] > 1e-3,
        "positive_oracle_headroom": initial_metrics["oracle_headroom"]
        > max(1e-4, 0.01 * initial_metrics["base_mse"]),
    }
    if not all(initial_invariants.values()):
        raise RuntimeError(f"step-0 invariants failed: {initial_invariants}")
    _append_jsonl(metrics_path, {"step": 0, **initial_metrics})
    print(json.dumps({"step": 0, **initial_metrics}, sort_keys=True), flush=True)

    consecutive_passes = 0
    final_step = 0
    final_metrics = initial_metrics
    last_grad_norm = 0.0
    gradient_probes: dict[str, dict[str, float]] = {}
    for step in range(1, args.steps + 1):
        cpu_indices, order, cursor = _next_indices(
            order,
            cursor,
            batch_size=args.train_batch_size,
            count=args.anchors,
            generator=generator,
        )
        indices = cpu_indices.to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted = predictor(
            cache["context"][indices],
            cache["base_prediction"][indices],
            goal=cache["goal"][indices],
        ).float()
        target = cache["target_code"][indices]
        error = (predicted - target).square().mean(dim=(0, 2, 3))
        code_loss = (
            error / global_target_energy.clamp_min(bottleneck.eps)
        ).mean()
        decoded_local = bottleneck._decode(
            predicted, bottleneck.spatial_basis(), bottleneck.basis()
        )
        final_prediction = cache["base_prediction"][indices] + bottleneck._to_cumulative(
            decoded_local
        )
        final_mse = (
            final_prediction - cache["target_delta"][indices]
        ).square().mean()
        loss = code_loss + args.raw_loss_weight * final_mse
        loss.backward()
        if step in (1, 2):
            output_gradient = _gradient_norm(predictor, include_output=True)
            internal_gradient = _gradient_norm(predictor, include_output=False)
            gradient_probes[str(step)] = {
                "output_gradient_norm": output_gradient,
                "internal_gradient_norm": internal_gradient,
            }
            if not math.isfinite(output_gradient) or output_gradient <= 0:
                raise RuntimeError(f"invalid output gradient at step {step}")
            if step == 2 and (
                not math.isfinite(internal_gradient) or internal_gradient <= 0
            ):
                raise RuntimeError("predictor internal gradient remained zero at step 2")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            predictor.parameters(), args.max_grad_norm
        )
        last_grad_norm = float(grad_norm.detach())
        optimizer.step()

        should_eval = step % args.eval_interval == 0 or step == args.steps
        if not should_eval:
            continue
        final_step = step
        final_metrics = _evaluate(
            predictor, bottleneck, cache, batch_size=args.eval_batch_size
        )
        record = {
            "step": step,
            "minibatch_loss": float(loss.detach()),
            "minibatch_code_loss": float(code_loss.detach()),
            "minibatch_final_mse": float(final_mse.detach()),
            "grad_norm": last_grad_norm,
            **final_metrics,
        }
        _append_jsonl(metrics_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if _passes(final_metrics, args):
            consecutive_passes += 1
        else:
            consecutive_passes = 0
        if consecutive_passes >= args.pass_patience:
            break

    passed = consecutive_passes >= args.pass_patience
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in predictor.state_dict().items()},
        output_dir / "predictor_final.pt",
    )
    summary = {
        "status": "PASS" if passed else "FAIL",
        "diagnostic_scope": "fixed_anchor_training_set_only",
        "final_step": final_step,
        "requested_steps": args.steps,
        "elapsed_seconds": time.time() - started,
        "trainable_predictor_parameters": trainable,
        "anchor_count": args.anchors,
        "optimization_recipe": {
            "learning_rate": args.learning_rate,
            "max_grad_norm": args.max_grad_norm,
            "raw_loss_weight": args.raw_loss_weight,
            "nmse_normalization": "fixed_full_anchor_per_horizon_energy",
            "precision": "fp32",
        },
        "thresholds": {
            "code_nmse_max": args.pass_nmse,
            "code_cosine_min": args.pass_cosine,
            "realized_headroom_fraction_min": args.pass_realized_headroom,
            "consecutive_evaluations": args.pass_patience,
        },
        "initial_metrics": initial_metrics,
        "initial_invariants": initial_invariants,
        "final_metrics": final_metrics,
        "last_grad_norm": last_grad_norm,
        "gradient_probes": gradient_probes,
        "consecutive_passing_evaluations": consecutive_passes,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / ("GATE.pass" if passed else "GATE.fail")).write_text(
        f"{summary['status']} at step {final_step}\n", encoding="utf-8"
    )
    running_path.replace(output_dir / "STATUS.complete")
    return summary


def main() -> None:
    args = _parse_args()
    try:
        summary = _run(args)
    except Exception:
        output_dir = Path(args.output_dir).resolve()
        error = traceback.format_exc()
        running = output_dir / "STATUS.running"
        if running.exists():
            error_path = output_dir / "error.txt"
            if error_path.exists():
                error_path = output_dir / f"error.{time.time_ns()}.txt"
        elif output_dir.exists() and any(output_dir.iterdir()):
            error_path = output_dir.parent / (
                f"{output_dir.name}.launch_error.{time.time_ns()}.txt"
            )
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            error_path = output_dir / "error.txt"
        error_path.write_text(error, encoding="utf-8")
        if running.exists():
            running.replace(output_dir / "STATUS.failed")
        print(error, flush=True)
        raise
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
