#!/usr/bin/env python3
"""Is the ~0.5 copy-ratio an intrinsic floor, or only the action-free ceiling?

The deployed LeWM-OFT world model is action-free: it predicts the future
visual-token residual from the current tokens plus a task embedding.  Its
training metric has been stuck near ``delta_to_copy_ratio ~ 0.50`` while several
capacity-oriented follow-ups (longer context, an 11M residual booster, a
low-rank innovation code) produced almost no gain.  Two very different
explanations survive that evidence:

  * the residual is genuinely aleatoric given the observation, so 0.5 is an
    intrinsic floor, or
  * the residual is dominated by *intent*, which an action-free predictor
    cannot observe, so 0.5 is only the action-free information ceiling.

This probe separates them with closed-form kernel ridge regressions fitted on
train episodes, selected on validation episodes and reported once on disjoint
test episodes.  The frozen checkpoint supplies the encoder, the spatial token
pooler, the task embedding and its own deployed prediction; nothing is
fine-tuned.  The decisive comparison is between an observation-only probe and
the same probe with the ground-truth action chunk appended: ground-truth
actions are unavailable at deployment, so they act purely as an oracle that
upper-bounds what *any* intent-conditioned predictor could recover.

Run (single GPU, no training job disturbed):

    .venv/bin/python examples/LIBERO/train_files/probe_action_conditioned_ceiling.py \
        --checkpoint playground/Checkpoints/<run>/checkpoints/steps_220000_pytorch_model.pt \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat

SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.60, "val": 0.20, "test": 0.20}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default=None,
        help="defaults to <checkpoint>/../../config.yaml, matching the policy server",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-episodes", type=int, default=900)
    parser.add_argument("--val-episodes", type=int, default=300)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--anchors-per-episode", type=int, default=2)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--ridge", default="1e-3,1e-2,1e-1,1,10,100")
    parser.add_argument(
        "--kernel",
        default="rbf",
        choices=("linear", "rbf"),
        help="a linear probe only bounds linearly available information; "
        "rbf is required for the ceiling claim to be meaningful",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _hash_u64(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    cfg = apply_config_compat(OmegaConf.load(config_path))
    torch.manual_seed(int(cfg.get("seed", 42)))
    model = build_framework(cfg)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = model.remap_checkpoint_state_dict(state_dict)
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint architecture mismatch: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    model.load_state_dict(state_dict, strict=True)
    model.requires_grad_(False)
    model.eval().to(device)
    if int(getattr(model, "predictor_state_dim", 0)) != 0:
        raise RuntimeError("probe assumes the deployed predictor is state-free")
    if bool(getattr(model, "predictable_innovation_enabled", False)):
        raise RuntimeError("probe targets the plain deployed residual predictor")
    return cfg, model


def _valid_step_bounds(dataset: Any, trajectory_index: int) -> tuple[int, int]:
    offsets: list[int] = []
    for modality_keys in dataset.modality_keys.values():
        for key in modality_keys:
            if key in dataset.delta_indices:
                offsets.extend(int(value) for value in dataset.delta_indices[key])
    length = int(dataset.trajectory_lengths[trajectory_index])
    return max(0, -min(offsets)), length - 1 - max(offsets)


def _episode_pools(mixture: Any, *, seed: int) -> dict[str, list[dict[str, Any]]]:
    train_cut = SPLIT_FRACTIONS["train"]
    val_cut = train_cut + SPLIT_FRACTIONS["val"]
    pools: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    seen: set[str] = set()
    for dataset in mixture.datasets:
        for trajectory_index, trajectory_id_raw in enumerate(dataset.trajectory_ids):
            trajectory_id = int(trajectory_id_raw)
            episode_key = f"{dataset.dataset_name}::{trajectory_id}"
            if episode_key in seen:
                continue
            seen.add(episode_key)
            unit = _hash_u64(seed, "split", episode_key) / float(1 << 64)
            split = "train" if unit < train_cut else "val" if unit < val_cut else "test"
            pools[split].append(
                {
                    "dataset": dataset,
                    "dataset_name": str(dataset.dataset_name),
                    "trajectory_index": trajectory_index,
                    "trajectory_id": trajectory_id,
                    "episode_key": episode_key,
                    "priority": _hash_u64(seed, "priority", episode_key),
                }
            )
    for split in SPLITS:
        pools[split].sort(key=lambda record: record["priority"])
    return pools


def _episode_samples(
    candidate: dict[str, Any], *, seed: int, anchors: int, expected_future: int
) -> list[dict[str, Any]]:
    dataset = candidate["dataset"]
    trajectory_index = int(candidate["trajectory_index"])
    trajectory_id = int(candidate["trajectory_id"])
    first, last = _valid_step_bounds(dataset, trajectory_index)
    if last < first:
        return []
    width = last - first + 1
    samples = []
    for anchor_index in range(anchors):
        step = first + _hash_u64(
            seed, "anchor", candidate["episode_key"], anchor_index
        ) % width
        raw = dataset.get_step_data(trajectory_id, int(step))
        sample = dataset._pack_sample(dataset.transforms(raw))
        future_images = sample.get("future_images")
        if future_images is None or len(future_images) != expected_future:
            raise RuntimeError(
                f"expected {expected_future} future frames, got "
                f"{0 if future_images is None else len(future_images)}"
            )
        samples.append(sample)
    return samples


@torch.inference_mode()
def _encode_batch(model: Any, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    frames_per_example = [
        [example["image"]] + list(example["future_images"]) for example in examples
    ]
    device = next(model.parameters()).device
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        patch_tokens = model.backbone.encode_patch_frames(frames_per_example)

    latent = model.visual_token_pooler(patch_tokens.float())
    goal = model._embed_task(
        [str(example["lang"]) for example in examples], device=latent.device
    )
    anchor = latent[:, : model.wm_ctx_len]
    future = latent[:, model.wm_ctx_len : model.wm_ctx_len + model.n_future]
    scale = model.world_model.delta_scale.detach().float().clamp_min(
        model.world_model._stats_eps
    )
    target_delta = (future - anchor[:, -1:]).float() / scale
    base_prediction = model.world_model.residual_predictor(
        anchor, goal=goal, state=None
    ).float()

    state = torch.as_tensor(
        np.stack([np.asarray(e["state"], dtype=np.float32)[0] for e in examples]),
        dtype=torch.float32,
    )
    action = torch.as_tensor(
        np.stack([np.asarray(e["action"], dtype=np.float32) for e in examples]),
        dtype=torch.float32,
    )
    return {
        "current": anchor[:, -1].detach().float().cpu(),
        "goal": goal.detach().float().cpu(),
        "state": state,
        "action": action.flatten(1),
        "base": base_prediction.detach().float().cpu(),
        "target": target_delta.detach().float().cpu(),
    }


def _build_cache(
    model: Any,
    candidates: list[dict[str, Any]],
    *,
    split: str,
    seed: int,
    anchors: int,
    expected_future: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    buffers: dict[str, list[torch.Tensor]] = {}
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        encoded = _encode_batch(model, pending)
        for key, value in encoded.items():
            buffers.setdefault(key, []).append(value)
        pending.clear()

    started = time.time()
    for index, candidate in enumerate(candidates):
        for sample in _episode_samples(
            candidate, seed=seed, anchors=anchors, expected_future=expected_future
        ):
            pending.append(sample)
            if len(pending) >= batch_size:
                flush()
        if (index + 1) % 100 == 0:
            print(
                f"[{split}] {index + 1}/{len(candidates)} episodes "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )
    flush()
    return {key: torch.cat(value, dim=0) for key, value in buffers.items()}


def _standardize(block: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor] | None):
    flat = block.flatten(1)
    if stats is None:
        mean = flat.mean(dim=0)
        std = flat.std(dim=0).clamp_min(1e-6)
        stats = (mean, std)
    mean, std = stats
    normalized = (flat - mean) / std
    # Equalize block influence so the combined kernel is not dominated by the
    # 12288-dim visual block purely because of its dimensionality.
    return normalized / float(normalized.shape[1]) ** 0.5, stats


def _metrics(prediction: torch.Tensor, target: torch.Tensor, n_future: int) -> dict:
    error = prediction - target
    mse = float(error.square().mean())
    copy_mse = float(target.square().mean())
    per_horizon = error.view(error.shape[0], n_future, -1).square().mean(dim=(0, 2))
    copy_horizon = target.view(target.shape[0], n_future, -1).square().mean(dim=(0, 2))
    cosine = float(
        F.cosine_similarity(prediction, target, dim=-1, eps=1e-8).mean()
    )
    return {
        "mse": mse,
        "copy_ratio": mse / max(copy_mse, 1e-12),
        "direction_cosine": cosine,
        "pred_rms": float(prediction.square().mean().sqrt()),
        "target_rms": float(target.square().mean().sqrt()),
        "per_horizon_copy_ratio": [
            float(per_horizon[i] / copy_horizon[i].clamp_min(1e-12))
            for i in range(n_future)
        ],
    }


class _KernelRidge:
    """Dual-form ridge with a shared eigendecomposition across penalties.

    A linear kernel only measures how much information is *linearly* available.
    The RBF kernel is a universal approximator on the training support, which
    is what makes an "oracle ceiling" claim meaningful, because the deployed
    world model is itself nonlinear.
    """

    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        kernel: str = "linear",
        gamma: float | None = None,
    ) -> None:
        self.features = features
        self.kernel = kernel
        self.gamma = gamma
        self.target_mean = targets.mean(dim=0, keepdim=True)
        centered = targets - self.target_mean
        gram = self._gram(features)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram.double())
        self.eigenvalues = eigenvalues.clamp_min(0)
        self.eigenvectors = eigenvectors
        self.projected = eigenvectors.T @ centered.double()

    def _gram(self, other: torch.Tensor) -> torch.Tensor:
        if self.kernel == "linear":
            return other @ self.features.T
        squared = (
            other.square().sum(dim=1, keepdim=True)
            + self.features.square().sum(dim=1)
            - 2.0 * (other @ self.features.T)
        )
        return torch.exp(-float(self.gamma) * squared.clamp_min(0))

    def predict(self, other: torch.Tensor, ridge: float) -> torch.Tensor:
        weights = self.projected / (self.eigenvalues + ridge).unsqueeze(-1)
        dual = (self.eigenvectors @ weights).float()
        return self.target_mean + self._gram(other) @ dual


def _median_squared_distance(features: torch.Tensor, samples: int = 1024) -> float:
    subset = features[torch.randperm(features.shape[0], device=features.device)[:samples]]
    distances = torch.cdist(subset, subset).square()
    mask = torch.triu(torch.ones_like(distances), diagonal=1) > 0
    return float(distances[mask].median().clamp_min(1e-8))


def _fit_probe(
    caches: dict[str, dict[str, torch.Tensor]],
    blocks: tuple[str, ...],
    target_key: str,
    ridges: list[float],
    n_future: int,
    device: torch.device,
    kernel: str = "linear",
    gamma_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> dict:
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    features: dict[str, torch.Tensor] = {}
    for split in SPLITS:
        parts = []
        for name in blocks:
            block = caches[split][name].to(device)
            standardized, block_stats = _standardize(
                block, stats.get(name) if split != "train" else None
            )
            if split == "train":
                stats[name] = block_stats
            parts.append(standardized)
        features[split] = torch.cat(parts, dim=1)

    targets = {
        split: caches[split][target_key].to(device).flatten(1) for split in SPLITS
    }

    if kernel == "linear":
        gammas: list[float | None] = [None]
    else:
        median = _median_squared_distance(features["train"])
        gammas = [scale / median for scale in gamma_scales]

    best = None
    for gamma in gammas:
        solver = _KernelRidge(
            features["train"], targets["train"], kernel=kernel, gamma=gamma
        )
        for ridge in ridges:
            prediction = solver.predict(features["val"], ridge)
            mse = float((prediction - targets["val"]).square().mean())
            if best is None or mse < best[1]:
                best = (ridge, mse, gamma, solver)
    ridge, val_mse, gamma, solver = best
    test_prediction = solver.predict(features["test"], float(ridge))
    result = _metrics(test_prediction, targets["test"], n_future)
    result["selected_ridge"] = float(ridge)
    result["kernel"] = kernel
    if gamma is not None:
        result["selected_gamma"] = float(gamma)
    result["val_mse"] = float(val_mse)
    result["blocks"] = list(blocks)
    result["target"] = target_key
    return result, test_prediction


def _run(args: argparse.Namespace) -> dict:
    checkpoint = Path(args.checkpoint).resolve()
    config_path = (
        Path(args.config).resolve()
        if args.config
        else checkpoint.parents[1] / "config.yaml"
    )
    output_dir = Path(
        args.output_dir or (checkpoint.parents[1] / "action_conditioned_ceiling_probe")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ridges = sorted({float(value) for value in args.ridge.split(",") if value.strip()})

    cfg, model = _load_model(config_path, checkpoint, device)
    data_cfg = cfg.datasets.vla_data
    # The launcher passes these on the command line, so a run's saved
    # ``config.yaml`` does not necessarily record them.
    data_cfg.future_obs_frames = True
    data_cfg.include_state = True
    mixture = get_vla_dataset(
        data_cfg=data_cfg,
        mode="val",
        balance_dataset_weights=bool(data_cfg.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(
            data_cfg.get("balance_trajectory_weights", False)
        ),
        seed=args.seed,
    )
    for dataset in mixture.datasets:
        dataset.transforms.eval()
    pools = _episode_pools(mixture, seed=args.seed)
    requested = {
        "train": args.train_episodes,
        "val": args.val_episodes,
        "test": args.test_episodes,
    }
    for split in SPLITS:
        if len(pools[split]) < requested[split]:
            raise RuntimeError(
                f"split {split} has {len(pools[split])} episodes < {requested[split]}"
            )

    caches: dict[str, dict[str, torch.Tensor]] = {}
    cache_path = output_dir / (
        f"cache_seed{args.seed}_e{args.train_episodes}-{args.val_episodes}"
        f"-{args.test_episodes}_a{args.anchors_per_episode}.pt"
    )
    if cache_path.exists():
        print(f"reusing encoder cache {cache_path}", flush=True)
        caches = torch.load(cache_path, map_location="cpu")
    else:
        for split in SPLITS:
            caches[split] = _build_cache(
                model,
                pools[split][: requested[split]],
                split=split,
                seed=args.seed,
                anchors=args.anchors_per_episode,
                expected_future=model.n_future,
                batch_size=args.encode_batch_size,
            )
            print(
                f"[{split}] cached {caches[split]['target'].shape[0]} anchors",
                flush=True,
            )
        torch.save(caches, cache_path)

    n_future = int(model.n_future)
    for split in SPLITS:
        cache = caches[split]
        cache["model_residual"] = cache["target"] - cache["base"]

    test_target = caches["test"]["target"].to(device).flatten(1)
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "seed": args.seed,
        "kernel": args.kernel,
        "anchors": {split: int(caches[split]["target"].shape[0]) for split in SPLITS},
        "episodes": requested,
        "delta_scale": float(model.world_model.delta_scale.detach().float().item()),
        "probes": {},
    }

    zero = torch.zeros_like(test_target)
    report["probes"]["copy_current_frame"] = _metrics(zero, test_target, n_future)
    train_mean = caches["train"]["target"].to(device).flatten(1).mean(dim=0, keepdim=True)
    report["probes"]["train_mean_residual"] = _metrics(
        train_mean.expand_as(test_target), test_target, n_future
    )
    report["probes"]["deployed_world_model"] = _metrics(
        caches["test"]["base"].to(device).flatten(1), test_target, n_future
    )

    probe_specs = [
        ("ridge_observation_only", ("current", "goal"), "target"),
        ("ridge_observation_state", ("current", "goal", "state"), "target"),
        ("ridge_action_only", ("action",), "target"),
        (
            "ridge_observation_state_action",
            ("current", "goal", "state", "action"),
            "target",
        ),
    ]
    for name, blocks, target_key in probe_specs:
        result, _ = _fit_probe(
            caches, blocks, target_key, ridges, n_future, device, kernel=args.kernel
        )
        report["probes"][name] = result
        print(f"{name}: copy_ratio={result['copy_ratio']:.4f}", flush=True)

    # The most deployment-relevant question: how much of the *trained model's*
    # remaining error is explained by intent that the model cannot observe.
    residual_result, residual_prediction = _fit_probe(
        caches,
        ("current", "goal", "state", "action"),
        "model_residual",
        ridges,
        n_future,
        device,
        kernel=args.kernel,
    )
    report["probes"]["ridge_residual_given_action"] = residual_result
    refined = caches["test"]["base"].to(device).flatten(1) + residual_prediction
    report["probes"]["deployed_plus_action_refinement"] = _metrics(
        refined, test_target, n_future
    )
    observation_residual, observation_prediction = _fit_probe(
        caches,
        ("current", "goal", "state"),
        "model_residual",
        ridges,
        n_future,
        device,
        kernel=args.kernel,
    )
    report["probes"]["ridge_residual_observation_only"] = observation_residual
    report["probes"]["deployed_plus_observation_refinement"] = _metrics(
        caches["test"]["base"].to(device).flatten(1) + observation_prediction,
        test_target,
        n_future,
    )

    output_path = output_dir / f"report_{args.kernel}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {output_path}")
    return report


def main() -> None:
    args = _parse_args()
    report = _run(args)
    print("\n=== held-out test copy ratios (lower is better) ===")
    for name, metrics in report["probes"].items():
        print(
            f"{name:42s} ratio={metrics['copy_ratio']:.4f} "
            f"cos={metrics['direction_cosine']:.4f} "
            f"per_horizon={[round(v, 4) for v in metrics['per_horizon_copy_ratio']]}"
        )


if __name__ == "__main__":
    main()
