#!/usr/bin/env python3
"""Episode-disjoint conditional probe for a predictable raw-error subspace.

This diagnostic deliberately ignores the Stage-A fixed mean, learned bases and
compact target code.  The Stage-A checkpoint is used only as a frozen source of
the visual encoder, visual-token pooler, base world-model prediction and task
embedding.  Targets are the uncentered *raw local error* remaining after that
base prediction.

Episodes are assigned to train/validation/test before a single unpadded anchor
is chosen from each episode.  A supervised reduced-rank ridge regression (RRR)
then asks whether current appearance, recent visual motion, the frozen base
prediction and task embedding identify a low-rank raw-error subspace that
generalizes to held-out episodes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

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
    "lewm_oft_libero_dinov3b_raw_predictive_subspace_rrr_"
    "train640_val192_test256"
)
SPLITS = ("train", "val", "test")
BLOCK_NAMES = ("current", "history_motion", "base", "goal")
SCOPE = "frozen 220k full-mixture conditional probe"
SPLIT_FRACTIONS = {"train": 0.60, "val": 0.20, "test": 0.20}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--train-episodes", "--train-anchors", dest="train_episodes", type=int, default=640
    )
    parser.add_argument(
        "--val-episodes", "--val-anchors", dest="val_episodes", type=int, default=192
    )
    parser.add_argument(
        "--test-episodes", "--test-anchors", dest="test_episodes", type=int, default=256
    )
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--ranks", default="4,8,16,32,64")
    parser.add_argument("--ridge", default="1e-3,1e-2,1e-1,1,10")
    parser.add_argument("--minimum-validation-gain", type=float, default=1.0e-3)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_number_list(text: str, *, integer: bool) -> list[int] | list[float]:
    values = []
    for field in str(text).split(","):
        field = field.strip()
        if not field:
            continue
        values.append(int(field) if integer else float(field))
    if not values:
        raise ValueError("numeric list cannot be empty")
    if any(value <= 0 for value in values):
        raise ValueError(f"numeric list values must be positive: {values}")
    return sorted(set(values))


def _validate_args(args: argparse.Namespace) -> tuple[list[int], list[float]]:
    for name in (
        "train_episodes",
        "val_episodes",
        "test_episodes",
        "encode_batch_size",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    ranks = _parse_number_list(args.ranks, integer=True)
    ridges = _parse_number_list(args.ridge, integer=False)
    if args.minimum_validation_gain < 0:
        raise ValueError("--minimum-validation-gain must be non-negative")
    if args.bootstrap_repeats < 100:
        raise ValueError("--bootstrap-repeats must be at least 100")
    return list(ranks), list(ridges)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().float().cpu().item()
        return value.detach().float().cpu().tolist()
    raise TypeError(f"cannot JSON-serialize {type(value).__name__}")


def _optional_difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_jsonable,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_u64(*parts: Any) -> int:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def _sample_image_sha256(sample: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    frames = [sample["image"]] + list(sample["future_images"])
    for frame_index, views in enumerate(frames):
        for view_index, image in enumerate(views):
            array = np.ascontiguousarray(np.asarray(image))
            digest.update(f"{frame_index}:{view_index}".encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _claim_empty_output(output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to use non-empty diagnostic output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    running = output_dir / "STATUS.running"
    running.write_text(
        f"pid={os.getpid()}\nstarted={time.time():.6f}\n", encoding="utf-8"
    )
    return running


def _load_model(
    config_path: Path, checkpoint_path: Path, device: torch.device
) -> tuple[Any, Any]:
    cfg = apply_config_compat(OmegaConf.load(config_path))
    # Keep the predictable-innovation module only so the checkpoint remains an
    # exact architectural match and the private history split stays available.
    # The external calibration artifact is intentionally not loaded: this probe
    # defines its own train-only raw mean and never reads the Stage-A bases/code.
    cfg.framework.world_model.innovation_fixed_mean_path = None
    cfg.framework.world_model.innovation_require_fixed_mean = False
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

    if not bool(getattr(model, "predictable_innovation_enabled", False)):
        raise RuntimeError("config must expose the private three-frame context")
    if int(getattr(model, "predictor_state_dim", 0)) != 0:
        raise RuntimeError("raw predictive-subspace probe requires predictor_state_dim=0")
    if int(model.innovation_context_len) < 2:
        raise RuntimeError("at least two history frames are required")
    return cfg, model


def _valid_step_bounds(dataset: Any, trajectory_index: int) -> tuple[int, int]:
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
    return first, last


def _episode_pools(
    mixture: Any,
    *,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Assign every episode by hash before selecting an anchor."""

    # Split membership must not depend on how many anchors a smoke/formal run
    # requests.  Otherwise changing a count silently changes which episodes
    # are considered held out.  These fixed cuts are part of the protocol.
    train_cut = SPLIT_FRACTIONS["train"]
    val_cut = train_cut + SPLIT_FRACTIONS["val"]
    pools: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    seen_keys: set[str] = set()

    for dataset_index, dataset in enumerate(mixture.datasets):
        for trajectory_index, trajectory_id_raw in enumerate(dataset.trajectory_ids):
            trajectory_id = int(trajectory_id_raw)
            episode_key = f"{dataset.dataset_name}::{trajectory_id}"
            if episode_key in seen_keys:
                continue
            seen_keys.add(episode_key)

            split_hash = _hash_u64(seed, "split", episode_key)
            unit = split_hash / float(1 << 64)
            split = "train" if unit < train_cut else "val" if unit < val_cut else "test"
            pools[split].append(
                {
                    "dataset": dataset,
                    "dataset_index": dataset_index,
                    "dataset_name": str(dataset.dataset_name),
                    "dataset_path": str(Path(dataset.dataset_path).resolve()),
                    "trajectory_index": trajectory_index,
                    "trajectory_id": trajectory_id,
                    "episode_key": episode_key,
                    "split_hash": f"{split_hash:016x}",
                    "priority": _hash_u64(seed, "priority", episode_key),
                }
            )

    requested = {"train": train_count, "val": val_count, "test": test_count}
    for split in SPLITS:
        pools[split].sort(key=lambda record: record["priority"])
        if len(pools[split]) < requested[split]:
            raise RuntimeError(
                f"episode-hash split {split} has {len(pools[split])} candidates, "
                f"fewer than requested {requested[split]}"
            )
    return pools


def _sample_episode_anchor(
    candidate: dict[str, Any], *, seed: int, expected_future_count: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = candidate["dataset"]
    trajectory_index = int(candidate["trajectory_index"])
    trajectory_id = int(candidate["trajectory_id"])
    first, last = _valid_step_bounds(dataset, trajectory_index)
    if last < first:
        raise RuntimeError("episode has no unpadded current/history/future anchor")
    width = last - first + 1
    step = first + _hash_u64(seed, "anchor", candidate["episode_key"]) % width

    raw_data = dataset.get_step_data(trajectory_id, int(step))
    transformed = dataset.transforms(raw_data)
    sample = dataset._pack_sample(transformed)
    future_images = sample.get("future_images")
    if future_images is None or len(future_images) != expected_future_count:
        raise RuntimeError(
            f"expected {expected_future_count} packed future/history frames, got "
            f"{0 if future_images is None else len(future_images)}"
        )
    manifest = {
        "dataset_index": int(candidate["dataset_index"]),
        "dataset_name": candidate["dataset_name"],
        "dataset_path": candidate["dataset_path"],
        "trajectory_index": trajectory_index,
        "trajectory_id": trajectory_id,
        "episode_key": candidate["episode_key"],
        "split_hash": candidate["split_hash"],
        "step": int(step),
        "valid_step_first": int(first),
        "valid_step_last": int(last),
        "progress": float((step - first) / max(last - first, 1)),
        "instruction": str(sample.get("lang", "")),
        "images_sha256": _sample_image_sha256(sample),
    }
    return sample, manifest


def _to_local(cumulative: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (cumulative[:, :1], cumulative[:, 1:] - cumulative[:, :-1]), dim=1
    )


def _to_cumulative(local: torch.Tensor) -> torch.Tensor:
    return local.cumsum(dim=1)


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
    cumulative_error = target_delta - base_prediction
    raw_local_error = _to_local(cumulative_error)
    reconstructed_target = base_prediction + _to_cumulative(raw_local_error)
    decode_error = (reconstructed_target - target_delta).abs().max()
    if not torch.isfinite(decode_error) or float(decode_error) > 1e-5:
        raise RuntimeError(
            "raw-local decode invariant failed: expected "
            "base + cumsum(raw_local_error) == target_delta, "
            f"max_abs_error={float(decode_error):.6e}"
        )
    history_motion = innovation_context[:, 1:] - innovation_context[:, :-1]

    return {
        "current": innovation_context[:, -1].detach().float().cpu(),
        "history_motion": history_motion.detach().float().cpu(),
        "base": base_prediction.detach().float().cpu(),
        "goal": goal.detach().float().cpu(),
        "target_delta": target_delta.detach().float().cpu(),
        "raw_local_error": raw_local_error.detach().float().cpu(),
    }


def _collect_split_cache(
    model: Any,
    candidates: list[dict[str, Any]],
    *,
    split: str,
    count: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    expected_future_count = int(model.innovation_context_len + model.n_future - 1)
    chunks: dict[str, list[torch.Tensor]] = {}
    manifest: list[dict[str, Any]] = []
    pending_examples: list[dict[str, Any]] = []
    pending_records: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending_examples:
            return
        encoded = _encode_batch(model, pending_examples)
        for name, tensor in encoded.items():
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"non-finite cached tensor {split}/{name}")
            chunks.setdefault(name, []).append(tensor)
        for record in pending_records:
            record["split"] = split
            record["anchor_index"] = len(manifest)
            manifest.append(record)
        pending_examples.clear()
        pending_records.clear()
        print(f"cached {split} {len(manifest)}/{count}", flush=True)

    if len(candidates) != count:
        raise RuntimeError(
            f"frozen {split} manifest has {len(candidates)} candidates, expected {count}"
        )
    for candidate in candidates:
        # The selected episode identities were written before decoding.  A
        # decode failure must stop the run rather than silently substitute a
        # different episode and change the formal test set.
        sample, record = _sample_episode_anchor(
            candidate,
            seed=seed,
            expected_future_count=expected_future_count,
        )
        pending_examples.append(sample)
        pending_records.append(record)
        if len(pending_examples) >= batch_size:
            flush()

    flush()
    if len(manifest) != count:
        raise RuntimeError(f"cached only {len(manifest)}/{count} anchors for {split}")
    cache = {name: torch.cat(parts, dim=0) for name, parts in chunks.items()}
    if any(tensor.shape[0] != count for tensor in cache.values()):
        raise RuntimeError(f"inconsistent cache tensor counts in split {split}")
    return cache, manifest


def _build_cache(
    cfg: Any,
    model: Any,
    *,
    counts: dict[str, int],
    batch_size: int,
    seed: int,
    planned_manifest_path: Path,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
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

    pools = _episode_pools(
        mixture,
        train_count=counts["train"],
        val_count=counts["val"],
        test_count=counts["test"],
        seed=seed,
    )
    selected = {split: pools[split][: counts[split]] for split in SPLITS}
    planned_payload = {
        "protocol": {
            "seed": seed,
            "split_fractions": SPLIT_FRACTIONS,
            "selection": "sha256 episode split, then sha256 priority",
            "one_anchor_per_episode": True,
            "decode_failure_policy": "fail_fast_no_replacement",
        },
        "candidate_episode_counts": {split: len(pools[split]) for split in SPLITS},
        "selected": {
            split: [
                {key: value for key, value in record.items() if key != "dataset"}
                for record in selected[split]
            ]
            for split in SPLITS
        },
    }
    _write_json(planned_manifest_path, planned_payload)
    caches: dict[str, dict[str, torch.Tensor]] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}
    diagnostics: dict[str, Any] = {
        "candidate_episode_counts": {split: len(pools[split]) for split in SPLITS},
        "split_fractions": SPLIT_FRACTIONS,
        "planned_manifest_sha256": _sha256_file(planned_manifest_path),
        "decode_failure_policy": "fail_fast_no_replacement",
    }
    for split in SPLITS:
        cache, manifest = _collect_split_cache(
            model,
            selected[split],
            split=split,
            count=counts[split],
            batch_size=batch_size,
            seed=seed,
        )
        caches[split] = cache
        manifests[split] = manifest
    return caches, manifests, diagnostics


def _normalise_task(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _audit_split_manifests(
    caches: dict[str, dict[str, torch.Tensor]],
    manifests: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    episode_sets = {
        split: {record["episode_key"] for record in manifests[split]}
        for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = episode_sets[left].intersection(episode_sets[right])
            if overlap:
                raise RuntimeError(
                    f"episode leakage between {left}/{right}: {sorted(overlap)[:4]}"
                )

    image_owner: dict[str, tuple[str, str]] = {}
    current_owner: dict[str, tuple[str, str]] = {}
    task_counts: dict[str, Counter[str]] = {}
    dataset_counts: dict[str, Counter[str]] = {}
    for split in SPLITS:
        task_counts[split] = Counter()
        dataset_counts[split] = Counter()
        for index, record in enumerate(manifests[split]):
            task = _normalise_task(record["instruction"])
            task_counts[split][task] += 1
            dataset_counts[split][str(record["dataset_name"])] += 1
            for digest_name, digest, owners in (
                ("image", record["images_sha256"], image_owner),
                ("encoded-current", _tensor_sha256(caches[split]["current"][index]), current_owner),
            ):
                if digest in owners:
                    previous_split, previous_episode = owners[digest]
                    if previous_split != split:
                        raise RuntimeError(
                            f"cross-split duplicate {digest_name}: "
                            f"{previous_split}/{previous_episode} and "
                            f"{split}/{record['episode_key']}"
                        )
                owners[digest] = (split, str(record["episode_key"]))

    missing_train_tasks = sorted(set(task_counts["test"]) - set(task_counts["train"]))
    missing_val_tasks = sorted(set(task_counts["test"]) - set(task_counts["val"]))
    if missing_train_tasks or missing_val_tasks:
        raise RuntimeError(
            "test tasks are not covered by development splits: "
            f"missing_train={missing_train_tasks}, missing_val={missing_val_tasks}"
        )
    return {
        "episode_overlap": False,
        "cross_split_exact_image_duplicates": False,
        "cross_split_exact_current_duplicates": False,
        "task_counts": {
            split: dict(sorted(task_counts[split].items())) for split in SPLITS
        },
        "dataset_counts": {
            split: dict(sorted(dataset_counts[split].items())) for split in SPLITS
        },
        "task_count": len(set().union(*(set(value) for value in task_counts.values()))),
        "minimum_task_samples": {
            split: min(task_counts[split].values()) for split in SPLITS
        },
    }


def _fit_input_normalisation(
    train_cache: dict[str, torch.Tensor], device: torch.device
) -> dict[str, dict[str, torch.Tensor | float | int]]:
    statistics: dict[str, dict[str, torch.Tensor | float | int]] = {}
    for block in BLOCK_NAMES:
        train = train_cache[block].reshape(train_cache[block].shape[0], -1)
        train = train.to(device=device, dtype=torch.float32)
        mean = train.mean(dim=0)
        std = (train - mean).square().mean(dim=0).clamp_min(0.0).sqrt()
        positive = std[std > 0]
        median_positive = float(positive.median()) if positive.numel() else 0.0
        floor = max(1e-6, median_positive * 1e-3)
        active = std > floor
        active_count = int(active.sum().item())
        if active_count == 0:
            raise RuntimeError(f"input block {block} has no non-constant dimensions")
        statistics[block] = {
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
            "active": active.detach().cpu(),
            "active_count": active_count,
            "std_floor": floor,
        }
    return statistics


def _apply_input_normalisation(
    caches: dict[str, dict[str, torch.Tensor]],
    statistics: dict[str, dict[str, torch.Tensor | float | int]],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    normalised: dict[str, dict[str, torch.Tensor]] = {
        split: {} for split in caches
    }
    for block in BLOCK_NAMES:
        block_stats = statistics[block]
        mean = block_stats["mean"].to(device=device, dtype=torch.float32)
        std = block_stats["std"].to(device=device, dtype=torch.float32)
        active = block_stats["active"].to(device=device, dtype=torch.bool)
        active_count = int(block_stats["active_count"])
        floor = float(block_stats["std_floor"])
        divisor = std[active].clamp_min(floor)
        for split, cache in caches.items():
            values = cache[block].reshape(cache[block].shape[0], -1)
            values = values.to(device=device, dtype=torch.float32)
            values = (values[:, active] - mean[active]) / divisor
            values = values / math.sqrt(active_count)
            if not torch.isfinite(values).all():
                raise FloatingPointError(f"non-finite normalized input {split}/{block}")
            normalised[split][block] = values
    return normalised


def _matched_history_permutation(
    manifest: list[dict[str, Any]], *, seed: int, split: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a deterministic, task-stratified bijective derangement.

    Every history is used exactly once, so the shuffle preserves its empirical
    marginal distribution.  A non-zero cyclic shift inside each task keeps the
    instruction fixed while guaranteeing that an episode never donates its own
    history.  Singleton tasks are pooled as an explicit fallback (or merged
    with one non-singleton stratum if there is only one); this is the minimum
    relaxation needed for a global derangement.  Among possible shifts, prefer
    the one retaining the most task/source/progress matches.
    """

    def fields(index: int) -> tuple[str, str, int]:
        record = manifest[index]
        task = _normalise_task(record.get("instruction", ""))
        source = str(record.get("dataset_name", ""))
        progress_bin = min(3, max(0, int(float(record.get("progress", 0.0)) * 4)))
        return task, source, progress_bin

    if len(manifest) < 2:
        raise RuntimeError(
            f"history derangement requires at least two samples in {split}"
        )

    task_groups: dict[str, list[int]] = {}
    for index in range(len(manifest)):
        task_groups.setdefault(fields(index)[0], []).append(index)
    singleton_tasks = sorted(
        task for task, indices in task_groups.items() if len(indices) < 2
    )

    strata = [
        (f"task:{task}", list(task_groups[task]))
        for task in sorted(task_groups)
        if len(task_groups[task]) >= 2
    ]
    fallback_strategy = "none"
    if len(singleton_tasks) >= 2:
        strata.append(
            (
                "singleton_task_pool",
                [task_groups[task][0] for task in singleton_tasks],
            )
        )
        fallback_strategy = "pool_singleton_tasks"
    elif len(singleton_tasks) == 1:
        singleton_task = singleton_tasks[0]
        merge_index = min(
            range(len(strata)),
            key=lambda index: _hash_u64(
                seed,
                "matched-history-singleton-merge",
                split,
                singleton_task,
                strata[index][0],
            ),
        )
        merged_label, merged_indices = strata.pop(merge_index)
        strata.append(
            (
                f"singleton_merge:{singleton_task}:{merged_label}",
                merged_indices + [task_groups[singleton_task][0]],
            )
        )
        fallback_strategy = "merge_singleton_with_non_singleton_task"

    permutation = torch.full((len(manifest),), -1, dtype=torch.long)
    match_counts: Counter[str] = Counter()
    donor_records: list[dict[str, Any]] = []
    stratum_shifts: dict[str, int] = {}
    for stratum, stratum_indices in sorted(strata):
        ordered = sorted(
            stratum_indices,
            key=lambda index: _hash_u64(
                seed,
                "matched-history-order",
                split,
                stratum,
                manifest[index]["episode_key"],
            ),
        )

        def shift_key(shift: int) -> tuple[int, int, int, int]:
            same_task = 0
            exact = 0
            same_task_source = 0
            for position, destination in enumerate(ordered):
                donor = ordered[(position + shift) % len(ordered)]
                destination_fields = fields(destination)
                donor_fields = fields(donor)
                same_task += int(destination_fields[0] == donor_fields[0])
                exact += int(destination_fields == donor_fields)
                same_task_source += int(
                    destination_fields[:2] == donor_fields[:2]
                )
            return (
                -same_task,
                -exact,
                -same_task_source,
                _hash_u64(seed, "matched-history-shift", split, stratum, shift),
            )

        shift = min(range(1, len(ordered)), key=shift_key)
        stratum_shifts[stratum] = shift
        for position, destination in enumerate(ordered):
            donor = ordered[(position + shift) % len(ordered)]
            destination_fields = fields(destination)
            donor_fields = fields(donor)
            if destination_fields == donor_fields:
                match_level = "task_source_progress"
            elif destination_fields[:2] == donor_fields[:2]:
                match_level = "task_source"
            elif destination_fields[0] == donor_fields[0]:
                match_level = "task"
            else:
                match_level = "cross_task_singleton_fallback"
            permutation[destination] = donor
            match_counts[match_level] += 1
            donor_records.append(
                {
                    "target_index": destination,
                    "target_episode_key": manifest[destination]["episode_key"],
                    "donor_index": int(donor),
                    "donor_episode_key": manifest[donor]["episode_key"],
                    "match_level": match_level,
                }
            )

    expected = torch.arange(len(manifest), dtype=torch.long)
    fixed = int((permutation == torch.arange(len(manifest))).sum().item())
    unique_donors = int(permutation.unique().numel())
    bijective = bool(torch.equal(permutation.sort().values, expected))
    if fixed or not bijective:
        raise RuntimeError(
            f"invalid history derangement in {split}: fixed_points={fixed}, "
            f"unique_donors={unique_donors}, sample_count={len(manifest)}"
        )
    return permutation, {
        "strategy": "task_stratified_bijective_cyclic_derangement",
        "sample_count": len(manifest),
        "unique_donors": unique_donors,
        "bijective": bijective,
        "fixed_points": fixed,
        "fixed_fraction": fixed / max(len(manifest), 1),
        "task_group_count": len(task_groups),
        "minimum_task_group_size": min(map(len, task_groups.values())),
        "same_task_count": len(manifest)
        - match_counts["cross_task_singleton_fallback"],
        "cross_task_count": match_counts["cross_task_singleton_fallback"],
        "singleton_tasks": singleton_tasks,
        "singleton_fallback_strategy": fallback_strategy,
        "stratum_count": len(strata),
        "match_counts": dict(sorted(match_counts.items())),
        "stratum_shifts": stratum_shifts,
        "donors": sorted(donor_records, key=lambda record: record["target_index"]),
    }


def _block_kernels(
    normalised: dict[str, dict[str, torch.Tensor]],
    manifests: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, Any],
    dict[str, torch.Tensor],
]:
    split_names = tuple(normalised)
    if "train" not in split_names or "val" not in split_names:
        raise ValueError("selection kernels require train and val splits")
    kernels: dict[str, dict[str, torch.Tensor]] = {}
    for block in BLOCK_NAMES:
        # Cast before GEMM.  Casting an fp32 Gram matrix afterwards preserves
        # spurious low-rank eigenvalues and can manufacture supervised signal
        # in an exactly null response direction.
        train = normalised["train"][block].double()
        kernels[block] = {"train": train @ train.transpose(0, 1)}
        for split in split_names:
            if split == "train":
                continue
            query = normalised[split][block].double()
            kernels[block][split] = query @ train.transpose(0, 1)

    shuffle_info: dict[str, Any] = {}
    shuffled_motion: dict[str, torch.Tensor] = {}
    for split in split_names:
        permutation, diagnostics = _matched_history_permutation(
            manifests[split], seed=seed, split=split
        )
        permutation = permutation.to(normalised[split]["history_motion"].device)
        shuffled_motion[split] = normalised[split]["history_motion"][permutation]
        shuffle_info[split] = diagnostics
    shuffled_train = shuffled_motion["train"].double()
    kernels["history_motion_shuffled"] = {
        "train": shuffled_train @ shuffled_train.transpose(0, 1)
    }
    for split in split_names:
        if split == "train":
            continue
        kernels["history_motion_shuffled"][split] = (
            shuffled_motion[split].double() @ shuffled_train.transpose(0, 1)
        )
    return kernels, shuffle_info, shuffled_motion


def _combine_kernel(
    kernels: dict[str, dict[str, torch.Tensor]], blocks: Iterable[str]
) -> dict[str, torch.Tensor]:
    selected = tuple(blocks)
    if not selected:
        raise ValueError("kernel variant must contain at least one input block")
    combined = {}
    split_names = tuple(kernels[selected[0]])
    for split in split_names:
        value = sum(kernels[block][split] for block in selected) / len(selected)
        if split == "train":
            value = 0.5 * (value + value.transpose(0, 1))
        combined[split] = value
    return combined


def _response_tensors(
    caches: dict[str, dict[str, torch.Tensor]],
    raw_mean: torch.Tensor,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    raw_mean = raw_mean.to(device=device, dtype=torch.float32)
    response: dict[str, dict[str, torch.Tensor]] = {}
    for split, cache in caches.items():
        raw = cache["raw_local_error"].to(device=device, dtype=torch.float32)
        base = cache["base"].to(device=device, dtype=torch.float32)
        target_delta = cache["target_delta"].to(
            device=device, dtype=torch.float32
        )
        decode_error = (base + _to_cumulative(raw) - target_delta).abs().max()
        if not torch.isfinite(decode_error) or float(decode_error) > 1e-5:
            raise RuntimeError(
                f"cached raw-local decode invariant failed in {split}: "
                "base + cumsum(raw_local_error) must equal target_delta, "
                f"max_abs_error={float(decode_error):.6e}"
            )
        centered = (raw - raw_mean).reshape(raw.shape[0], -1)
        response[split] = {
            "raw_local_error": raw,
            "centered": centered,
            "base": base,
            "target_delta": target_delta,
        }
    return response


@torch.inference_mode()
def _probe_metrics(
    split_data: dict[str, torch.Tensor],
    raw_mean: torch.Tensor,
    subspace: torch.Tensor,
    predicted_z: torch.Tensor,
    *,
    include_per_sample: bool = False,
) -> dict[str, Any]:
    y_centered = split_data["centered"]
    subspace_fp32 = subspace.float()
    target_z = y_centered @ subspace_fp32
    predicted_z_fp32 = predicted_z.float()

    centered_energy = y_centered.square().sum()
    if subspace.shape[1] == 0:
        z_nmse = None
        z_cosine = None
        subspace_energy_fraction = y_centered.new_zeros(())
        predicted_z_energy_fraction = y_centered.new_zeros(())
        predictable_energy_fraction = y_centered.new_zeros(())
    else:
        z_error = predicted_z_fp32 - target_z
        z_energy = target_z.square().sum()
        z_nmse = z_error.square().sum() / z_energy.clamp_min(1e-12)
        z_cosine = F.cosine_similarity(
            predicted_z_fp32, target_z, dim=-1, eps=1e-8
        ).mean()
        subspace_energy_fraction = z_energy / centered_energy.clamp_min(1e-12)
        predicted_z_energy_fraction = (
            predicted_z_fp32.square().sum() / z_energy.clamp_min(1e-12)
        )
        predictable_energy_fraction = subspace_energy_fraction * (1.0 - z_nmse)

    shape = split_data["raw_local_error"].shape
    predicted_centered = (predicted_z_fp32 @ subspace_fp32.transpose(0, 1)).reshape(
        shape
    )
    oracle_centered = (target_z @ subspace_fp32.transpose(0, 1)).reshape(shape)
    mean_local = raw_mean.unsqueeze(0)

    base = split_data["base"]
    target_delta = split_data["target_delta"]
    mean_prediction = base + _to_cumulative(mean_local)
    final_prediction = base + _to_cumulative(mean_local + predicted_centered)
    oracle_prediction = base + _to_cumulative(mean_local + oracle_centered)

    def mse(prediction: torch.Tensor) -> torch.Tensor:
        return (prediction - target_delta).square().mean()

    def sample_mse(prediction: torch.Tensor) -> torch.Tensor:
        return (prediction - target_delta).square().mean(dim=(1, 2, 3))

    def per_horizon(prediction: torch.Tensor) -> list[float]:
        values = (prediction - target_delta).square().mean(dim=(0, 2, 3))
        return [float(value) for value in values]

    base_mse = mse(base)
    mean_mse = mse(mean_prediction)
    final_mse = mse(final_prediction)
    oracle_mse = mse(oracle_prediction)
    headroom = mean_mse - oracle_mse
    improvement = mean_mse - final_mse
    improvement_samples = sample_mse(mean_prediction) - sample_mse(final_prediction)
    improvement_se = (
        improvement_samples.std(unbiased=True) / math.sqrt(shape[0])
        if shape[0] > 1
        else improvement_samples.new_zeros(())
    )
    realized = (
        float(improvement / headroom)
        if float(headroom) > 1e-12
        else None
    )
    metrics = {
        "sample_count": int(shape[0]),
        "rank": int(subspace.shape[1]),
        "base_raw_mse": float(base_mse),
        "mean_raw_mse": float(mean_mse),
        "final_raw_mse": float(final_mse),
        "subspace_oracle_raw_mse": float(oracle_mse),
        "base_raw_mse_per_horizon": per_horizon(base),
        "mean_raw_mse_per_horizon": per_horizon(mean_prediction),
        "final_raw_mse_per_horizon": per_horizon(final_prediction),
        "subspace_oracle_raw_mse_per_horizon": per_horizon(oracle_prediction),
        "mean_improvement_over_base": float(base_mse - mean_mse),
        "final_improvement_over_mean": float(improvement),
        "final_improvement_over_mean_se": float(improvement_se),
        "subspace_oracle_headroom": float(headroom),
        "realized_subspace_headroom_fraction": realized,
        "z_nmse": None if z_nmse is None else float(z_nmse),
        "z_cosine": None if z_cosine is None else float(z_cosine),
        "subspace_energy_fraction": float(subspace_energy_fraction),
        "predicted_z_energy_fraction": float(predicted_z_energy_fraction),
        "predictable_energy_fraction": float(predictable_energy_fraction),
    }
    if include_per_sample:
        metrics["per_sample_raw_mse"] = {
            "base": sample_mse(base).detach().float().cpu().tolist(),
            "mean": sample_mse(mean_prediction).detach().float().cpu().tolist(),
            "final": sample_mse(final_prediction).detach().float().cpu().tolist(),
            "oracle": sample_mse(oracle_prediction).detach().float().cpu().tolist(),
        }
    return metrics


def _kernel_eigendecomposition(kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel = kernel.double()
    kernel = 0.5 * (kernel + kernel.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    largest = float(eigenvalues.abs().max()) if eigenvalues.numel() else 0.0
    zero_tolerance = max(largest * 1e-10, 1e-12)
    if float(eigenvalues.min()) < -zero_tolerance:
        raise RuntimeError(
            f"normalized input kernel is not PSD: min={float(eigenvalues.min()):.6e}"
        )
    eigenvalues = torch.where(
        eigenvalues > zero_tolerance, eigenvalues, torch.zeros_like(eigenvalues)
    )
    return eigenvalues, eigenvectors


def _supervised_subspace(
    y_train: torch.Tensor,
    projected_y_gram: torch.Tensor,
    kernel_eigenvalues: torch.Tensor,
    kernel_eigenvectors: torch.Tensor,
    *,
    alpha: float,
    max_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top raw-output RRR vectors without forming a Q x Q matrix."""

    u = kernel_eigenvectors
    h_sqrt = (kernel_eigenvalues / (kernel_eigenvalues + alpha)).sqrt()
    supervised_gram = (
        h_sqrt[:, None] * projected_y_gram * h_sqrt[None, :]
    )
    supervised_gram = 0.5 * (
        supervised_gram + supervised_gram.transpose(0, 1)
    )
    values, vectors = torch.linalg.eigh(supervised_gram)
    order = torch.argsort(values, descending=True)
    values = values[order]
    vectors = vectors[:, order]
    response_energy = max(float(torch.trace(projected_y_gram).abs()), 1.0)
    tolerance = max(response_energy * 1e-12, 1e-14)
    usable = int((values > tolerance).sum().item())
    retained = min(max_rank, usable)
    if retained < 1:
        return (
            torch.empty(
                y_train.shape[1],
                0,
                device=y_train.device,
                dtype=torch.float64,
            ),
            torch.empty(0, device=y_train.device, dtype=torch.float64),
        )
    values = values[:retained]
    vectors = vectors[:, :retained]

    sample_coefficients = u @ (h_sqrt[:, None] * vectors)
    raw_vectors = y_train.double().transpose(0, 1) @ sample_coefficients
    raw_vectors = raw_vectors / values.sqrt().unsqueeze(0)
    subspace, _ = torch.linalg.qr(raw_vectors, mode="reduced")
    orthogonality_error = (
        subspace.transpose(0, 1) @ subspace
        - torch.eye(retained, device=subspace.device, dtype=subspace.dtype)
    ).abs().max()
    if float(orthogonality_error) > 1e-6:
        raise RuntimeError(
            f"RRR subspace is not orthonormal: {float(orthogonality_error):.3e}"
        )
    return subspace, values


def _ridge_dual_coefficients(
    kernel_eigenvalues: torch.Tensor,
    kernel_eigenvectors: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    projected = kernel_eigenvectors.transpose(0, 1) @ targets
    projected = projected / (kernel_eigenvalues + alpha).unsqueeze(1)
    return kernel_eigenvectors @ projected


def _select_variant(
    variant: str,
    kernel: dict[str, torch.Tensor],
    response: dict[str, dict[str, torch.Tensor]],
    raw_mean: torch.Tensor,
    *,
    ranks: list[int],
    ridge_multipliers: list[float],
    minimum_validation_gain: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, torch.Tensor]]:
    k_train = kernel["train"]
    k_val = kernel["val"]
    eigenvalues, eigenvectors = _kernel_eigendecomposition(k_train)
    kernel_scale = float(torch.trace(k_train) / k_train.shape[0])
    if not math.isfinite(kernel_scale) or kernel_scale <= 0:
        raise RuntimeError(f"variant {variant} has invalid kernel scale {kernel_scale}")

    y_train = response["train"]["centered"]
    y_train_double = y_train.double()
    y_gram = y_train_double @ y_train_double.transpose(0, 1)
    y_gram = 0.5 * (y_gram + y_gram.transpose(0, 1))
    projected_y_gram = eigenvectors.transpose(0, 1) @ y_gram @ eigenvectors
    projected_y_gram = 0.5 * (
        projected_y_gram + projected_y_gram.transpose(0, 1)
    )
    max_rank = min(max(ranks), y_train.shape[0], y_train.shape[1])
    grid: list[dict[str, Any]] = []
    rank_zero_subspace = torch.empty(
        y_train.shape[1], 0, device=y_train.device, dtype=torch.float64
    )
    rank_zero_dual = torch.empty(
        y_train.shape[0], 0, device=y_train.device, dtype=torch.float64
    )
    rank_zero_prediction = torch.empty(
        response["val"]["centered"].shape[0],
        0,
        device=y_train.device,
        dtype=torch.float64,
    )
    rank_zero_metrics = _probe_metrics(
        response["val"], raw_mean, rank_zero_subspace, rank_zero_prediction
    )
    grid.append(
        {
            "variant": variant,
            "ridge_multiplier": None,
            "alpha": None,
            "rank": 0,
            "kernel_scale": kernel_scale,
            "supervised_eigenvalue_sum": 0.0,
            **rank_zero_metrics,
        }
    )
    best_key: tuple[float, int, float] | None = (
        rank_zero_metrics["final_raw_mse"],
        0,
        0.0,
    )
    best: dict[str, Any] | None = {
        "ridge_multiplier": None,
        "alpha": None,
        "rank": 0,
        "kernel_scale": kernel_scale,
        "subspace": rank_zero_subspace,
        "dual": rank_zero_dual,
        "validation_metrics": rank_zero_metrics,
        "supervised_eigenvalues": torch.empty(
            0, device=y_train.device, dtype=torch.float64
        ),
    }

    for ridge_multiplier in ridge_multipliers:
        alpha = float(ridge_multiplier * kernel_scale)
        subspace_max, supervised_values = _supervised_subspace(
            y_train,
            projected_y_gram,
            eigenvalues,
            eigenvectors,
            alpha=alpha,
            max_rank=max_rank,
        )
        if subspace_max.shape[1] == 0:
            continue
        z_train_max = y_train_double @ subspace_max
        dual_max = _ridge_dual_coefficients(
            eigenvalues,
            eigenvectors,
            z_train_max,
            alpha=alpha,
        )
        predicted_val_max = k_val @ dual_max

        for rank in ranks:
            if rank > subspace_max.shape[1]:
                continue
            subspace = subspace_max[:, :rank]
            predicted_val = predicted_val_max[:, :rank]
            metrics = _probe_metrics(
                response["val"], raw_mean, subspace, predicted_val
            )
            record = {
                "variant": variant,
                "ridge_multiplier": ridge_multiplier,
                "alpha": alpha,
                "rank": rank,
                "kernel_scale": kernel_scale,
                "supervised_eigenvalue_sum": float(supervised_values[:rank].sum()),
                **metrics,
            }
            grid.append(record)
            required_gain = max(
                minimum_validation_gain,
                float(metrics["final_improvement_over_mean_se"]),
            )
            if metrics["final_improvement_over_mean"] < required_gain:
                continue
            key = (metrics["final_raw_mse"], rank, -ridge_multiplier)
            if best_key is None or key < best_key:
                best_key = key
                best = {
                    "ridge_multiplier": ridge_multiplier,
                    "alpha": alpha,
                    "rank": rank,
                    "kernel_scale": kernel_scale,
                    "subspace": subspace.detach().clone(),
                    "dual": dual_max[:, :rank].detach().clone(),
                    "validation_metrics": metrics,
                    "supervised_eigenvalues": supervised_values[:rank]
                    .detach()
                    .clone(),
                }

    if best is None:
        raise RuntimeError(f"no valid ridge/rank candidate for variant {variant}")

    selection = {
        "variant": variant,
        "ridge_multiplier": best["ridge_multiplier"],
        "alpha": best["alpha"],
        "rank": best["rank"],
        "kernel_scale": best["kernel_scale"],
        "validation_metrics": best["validation_metrics"],
        "minimum_validation_gain": minimum_validation_gain,
        "positive_rank_requires_one_standard_error": True,
        "supervised_eigenvalues": best["supervised_eigenvalues"],
    }
    tensors = {
        "subspace": best["subspace"].detach().cpu(),
        "dual_coefficients": best["dual"].detach().cpu(),
    }
    return selection, grid, tensors


def _build_test_cross_kernels(
    normalised_development: dict[str, dict[str, torch.Tensor]],
    normalised_test: dict[str, dict[str, torch.Tensor]],
    shuffled_development_motion: dict[str, torch.Tensor],
    test_manifest: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    train = normalised_development["train"]
    test = normalised_test["test"]
    block_cross = {
        block: test[block].double() @ train[block].double().transpose(0, 1)
        for block in BLOCK_NAMES
    }
    permutation, shuffle_diagnostics = _matched_history_permutation(
        test_manifest, seed=seed, split="test"
    )
    permutation = permutation.to(test["history_motion"].device)
    shuffled_test_motion = test["history_motion"][permutation].double()
    original_train_motion = train["history_motion"].double()
    shuffled_train_motion = shuffled_development_motion["train"].double()
    block_cross["history_motion_test_shuffled"] = (
        shuffled_test_motion @ original_train_motion.transpose(0, 1)
    )
    block_cross["history_motion_refit_shuffled"] = (
        shuffled_test_motion @ shuffled_train_motion.transpose(0, 1)
    )
    return block_cross, shuffle_diagnostics


def _combine_test_cross(
    block_cross: dict[str, torch.Tensor], blocks: Iterable[str]
) -> torch.Tensor:
    selected = tuple(blocks)
    if not selected:
        raise ValueError("test kernel must contain at least one input block")
    return sum(block_cross[block] for block in selected) / len(selected)


def _evaluate_selection(
    test_kernel: torch.Tensor,
    test_response: dict[str, torch.Tensor],
    raw_mean: torch.Tensor,
    tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    device = test_kernel.device
    subspace = tensors["subspace"].to(device=device, dtype=torch.float64)
    dual = tensors["dual_coefficients"].to(device=device, dtype=torch.float64)
    metrics = _probe_metrics(
        test_response,
        raw_mean,
        subspace,
        test_kernel @ dual,
        include_per_sample=True,
    )
    per_sample = metrics.pop("per_sample_raw_mse")
    return metrics, per_sample


def _paired_task_bootstrap(
    left: list[float],
    right: list[float],
    manifest: list[dict[str, Any]],
    *,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    if difference.shape != (len(manifest),):
        raise ValueError("paired bootstrap loss arrays do not match test manifest")
    groups: dict[str, np.ndarray] = {}
    for index, record in enumerate(manifest):
        groups.setdefault(_normalise_task(record["instruction"]), []).append(index)
    groups = {
        task: np.asarray(indices, dtype=np.int64) for task, indices in groups.items()
    }
    tasks = sorted(groups)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled_tasks = rng.integers(0, len(tasks), size=len(tasks))
        values = []
        for task_index in sampled_tasks:
            indices = groups[tasks[int(task_index)]]
            sampled = rng.choice(indices, size=len(indices), replace=True)
            values.append(difference[sampled])
        draws[repeat] = np.concatenate(values).mean()
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return {
        "mean": float(difference.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_repeats": int(repeats),
        "task_count": len(tasks),
    }


def _task_macro_difference(
    left: list[float],
    right: list[float],
    manifest: list[dict[str, Any]],
) -> dict[str, float]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    values: dict[str, list[float]] = {}
    for index, record in enumerate(manifest):
        values.setdefault(_normalise_task(record["instruction"]), []).append(
            float(difference[index])
        )
    task_means = np.asarray(
        [np.mean(values[task]) for task in sorted(values)], dtype=np.float64
    )
    return {
        "task_macro_mean": float(task_means.mean()),
        "positive_task_fraction": float((task_means > 0).mean()),
        "task_count": int(task_means.size),
        "minimum_task_gain": float(task_means.min()),
        "maximum_task_gain": float(task_means.max()),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ranks, ridges = _validate_args(args)
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"missing config/checkpoint: {config_path}, {checkpoint_path}"
        )
    running_path = _claim_empty_output(output_dir)
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
    counts = {
        "train": args.train_episodes,
        "val": args.val_episodes,
        "test": args.test_episodes,
    }
    planned_manifest_path = output_dir / "episode_split_plan.json"
    caches, manifests, cache_diagnostics = _build_cache(
        cfg,
        model,
        counts=counts,
        batch_size=args.encode_batch_size,
        seed=args.seed,
        planned_manifest_path=planned_manifest_path,
    )
    split_audit = _audit_split_manifests(caches, manifests)
    manifest_path = output_dir / "episodes_manifest.json"
    _write_json(
        manifest_path,
        {
            "splits": manifests,
            "diagnostics": cache_diagnostics,
            "audit": split_audit,
        },
    )
    development_cache_path = output_dir / "development_anchor_cache.pt"
    test_cache_path = output_dir / "sealed_test_anchor_cache.pt"
    common_cache_metadata = {
        "scope": SCOPE,
        "coordinate": "normalized_delta_raw_local_error",
        "uses_stage_a_fixed_mean": False,
        "uses_stage_a_basis": False,
        "uses_stage_a_target_code": False,
        "source_config": str(config_path),
        "source_checkpoint": str(checkpoint_path),
        "seed": args.seed,
        "planned_manifest_sha256": _sha256_file(planned_manifest_path),
    }
    torch.save(
        {
            "splits": {"train": caches["train"], "val": caches["val"]},
            "metadata": {**common_cache_metadata, "sealed_split": "development"},
        },
        development_cache_path,
    )
    torch.save(
        {
            "split": caches["test"],
            "metadata": {**common_cache_metadata, "sealed_split": "test"},
        },
        test_cache_path,
    )
    development_caches = {"train": caches["train"], "val": caches["val"]}
    del caches

    # Encoding is complete.  The frozen DINO/base model is not part of the
    # linear probe and can release its GPU memory before kernel construction.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    normalisation_statistics = _fit_input_normalisation(
        development_caches["train"], device
    )
    normalisation_path = output_dir / "input_normalisation.pt"
    torch.save(normalisation_statistics, normalisation_path)
    normalised_development = _apply_input_normalisation(
        development_caches, normalisation_statistics, device
    )
    development_manifests = {"train": manifests["train"], "val": manifests["val"]}
    kernels, development_shuffle_info, shuffled_development_motion = _block_kernels(
        normalised_development, development_manifests, seed=args.seed
    )
    raw_mean = development_caches["train"]["raw_local_error"].float().mean(dim=0)
    raw_mean = raw_mean.to(device)
    development_response = _response_tensors(
        development_caches, raw_mean, device
    )
    raw_mean_path = output_dir / "train_raw_local_error_mean.pt"
    torch.save(raw_mean.detach().cpu(), raw_mean_path)

    variants = {
        "full": ("current", "history_motion", "base", "goal"),
        "no_history": ("current", "base", "goal"),
        "task_only": ("goal",),
        "fit_time_history_motion_shuffle_null": (
            "current",
            "history_motion_shuffled",
            "base",
            "goal",
        ),
    }
    selections: dict[str, Any] = {}
    selected_tensors: dict[str, dict[str, torch.Tensor]] = {}
    validation_grid: list[dict[str, Any]] = []
    for variant, blocks in variants.items():
        print(f"fitting probe variant={variant} blocks={blocks}", flush=True)
        combined = _combine_kernel(kernels, blocks)
        selection, grid, tensors = _select_variant(
            variant,
            combined,
            development_response,
            raw_mean,
            ranks=ranks,
            ridge_multipliers=ridges,
            minimum_validation_gain=args.minimum_validation_gain,
        )
        selections[variant] = selection
        selected_tensors[variant] = tensors
        validation_grid.extend(grid)
        print(json.dumps(selection, default=_jsonable, sort_keys=True), flush=True)

    grid_path = output_dir / "validation_grid.json"
    _write_json(grid_path, {"records": validation_grid})
    subspace_path = output_dir / "selected_subspaces.pt"
    torch.save(
        {
            "variants": selected_tensors,
            "raw_mean": raw_mean.detach().cpu(),
            "ranks": ranks,
            "ridge_multipliers": ridges,
            "input_variants": variants,
        },
        subspace_path,
    )
    selection_path = output_dir / "selection.json"
    _write_json(
        selection_path,
        {
            "scope": SCOPE,
            "test_used": False,
            "input_variants": variants,
            "selections": selections,
            "selected_subspaces_sha256": _sha256_file(subspace_path),
            "minimum_validation_gain": args.minimum_validation_gain,
        },
    )
    selection_sha256 = _sha256_file(selection_path)
    (output_dir / "SELECTION.frozen").write_text(
        f"selection_sha256={selection_sha256}\n", encoding="utf-8"
    )

    # Selection is now immutable.  Load the sealed test cache exactly once and
    # evaluate the complete predeclared bundle without further tuning.
    sealed_test = torch.load(test_cache_path, map_location="cpu", weights_only=True)
    test_caches = {"test": sealed_test["split"]}
    normalised_test = _apply_input_normalisation(
        test_caches, normalisation_statistics, device
    )
    test_response = _response_tensors(test_caches, raw_mean, device)["test"]
    test_block_cross, test_shuffle_info = _build_test_cross_kernels(
        normalised_development,
        normalised_test,
        shuffled_development_motion,
        manifests["test"],
        seed=args.seed,
    )
    test_kernel_blocks = {
        "full": ("current", "history_motion", "base", "goal"),
        "no_history": ("current", "base", "goal"),
        "task_only": ("goal",),
        "fit_time_history_motion_shuffle_null": (
            "current",
            "history_motion_refit_shuffled",
            "base",
            "goal",
        ),
    }
    test_results: dict[str, Any] = {}
    test_per_sample: dict[str, dict[str, list[float]]] = {}
    for variant, blocks in test_kernel_blocks.items():
        metrics, per_sample = _evaluate_selection(
            _combine_test_cross(test_block_cross, blocks),
            test_response,
            raw_mean,
            selected_tensors[variant],
        )
        test_results[variant] = metrics
        test_per_sample[variant] = per_sample

    direct_variant = "full_test_history_motion_shuffle"
    direct_metrics, direct_per_sample = _evaluate_selection(
        _combine_test_cross(
            test_block_cross,
            ("current", "history_motion_test_shuffled", "base", "goal"),
        ),
        test_response,
        raw_mean,
        selected_tensors["full"],
    )
    test_results[direct_variant] = {
        **direct_metrics,
        "fit_source": "full",
        "retrained": False,
        "changed_input": "test_history_motion_only",
    }
    test_per_sample[direct_variant] = direct_per_sample

    full_losses = test_per_sample["full"]
    comparison_pairs = {
        "full_gain_over_base": (full_losses["base"], full_losses["final"]),
        "full_gain_over_train_mean": (full_losses["mean"], full_losses["final"]),
        "full_gain_over_no_history": (
            test_per_sample["no_history"]["final"],
            full_losses["final"],
        ),
        "full_gain_over_task_only": (
            test_per_sample["task_only"]["final"],
            full_losses["final"],
        ),
        "full_gain_over_fit_time_shuffle_null": (
            test_per_sample["fit_time_history_motion_shuffle_null"]["final"],
            full_losses["final"],
        ),
        "direct_history_shuffle_penalty": (
            test_per_sample[direct_variant]["final"],
            full_losses["final"],
        ),
    }
    comparisons = {}
    for index, (name, (left, right)) in enumerate(comparison_pairs.items()):
        comparisons[name] = {
            "paired_task_bootstrap": _paired_task_bootstrap(
                left,
                right,
                manifests["test"],
                repeats=args.bootstrap_repeats,
                seed=args.seed + 1000 + index,
            ),
            "task_macro": _task_macro_difference(left, right, manifests["test"]),
        }
    comparisons["direct_history_shuffle_z_nmse_penalty"] = _optional_difference(
        test_results[direct_variant]["z_nmse"], test_results["full"]["z_nmse"]
    )

    test_results_path = output_dir / "test_results.json"
    _write_json(
        test_results_path,
        {
            "selection_sha256": selection_sha256,
            "selection_frozen_before_test": True,
            "results": test_results,
            "comparisons": comparisons,
            "test_history_motion_shuffle": test_shuffle_info,
        },
    )
    per_sample_path = output_dir / "test_per_sample_raw_mse.json"
    _write_json(per_sample_path, {"variants": test_per_sample})

    full_test = test_results["full"]
    full_r2 = (
        None if full_test["z_nmse"] is None else 1.0 - float(full_test["z_nmse"])
    )
    gate_criteria = {
        "selected_nonzero_rank": int(selections["full"]["rank"]) > 0,
        "test_z_r2_at_least_0_05": full_r2 is not None and full_r2 >= 0.05,
        "overall_gain_ci_positive": comparisons["full_gain_over_base"]
        ["paired_task_bootstrap"]["ci95_lower"]
        > 0,
        "dynamic_gain_ci_positive": comparisons["full_gain_over_train_mean"]
        ["paired_task_bootstrap"]["ci95_lower"]
        > 0,
        "history_gain_over_no_history_ci_positive": comparisons[
            "full_gain_over_no_history"
        ]["paired_task_bootstrap"]["ci95_lower"]
        > 0,
        "direct_history_shuffle_penalty_ci_positive": comparisons[
            "direct_history_shuffle_penalty"
        ]["paired_task_bootstrap"]["ci95_lower"]
        > 0,
        "realized_headroom_at_least_0_05": (
            full_test["realized_subspace_headroom_fraction"] is not None
            and full_test["realized_subspace_headroom_fraction"] >= 0.05
        ),
        "positive_history_gain_on_60_percent_tasks": comparisons[
            "full_gain_over_no_history"
        ]["task_macro"]["positive_task_fraction"]
        >= 0.60,
    }
    scientific_status = "PASS" if all(gate_criteria.values()) else "FAIL"

    dimensions = {
        "current": list(development_caches["train"]["current"].shape[1:]),
        "history_motion": list(
            development_caches["train"]["history_motion"].shape[1:]
        ),
        "base": list(development_caches["train"]["base"].shape[1:]),
        "goal": list(development_caches["train"]["goal"].shape[1:]),
        "raw_local_error": list(
            development_caches["train"]["raw_local_error"].shape[1:]
        ),
        "raw_output_dimensions": int(
            np.prod(development_caches["train"]["raw_local_error"].shape[1:])
        ),
    }
    summary = {
        "status": "COMPLETE",
        "scientific_status": scientific_status,
        "scope": SCOPE,
        "diagnostic": "episode_disjoint_raw_predictive_subspace_rrr",
        "elapsed_seconds": time.time() - started,
        "source_config": str(config_path),
        "source_checkpoint": str(checkpoint_path),
        "split_episode_counts": counts,
        "one_anchor_per_episode": True,
        "padding_allowed": False,
        "dimensions": dimensions,
        "ranks": ranks,
        "ridge_multipliers": ridges,
        "minimum_validation_gain": args.minimum_validation_gain,
        "input_variants": variants,
        "cache_diagnostics": cache_diagnostics,
        "split_audit": split_audit,
        "development_history_motion_shuffle": development_shuffle_info,
        "test_history_motion_shuffle": test_shuffle_info,
        "selections": selections,
        "test_results": test_results,
        "test_comparisons": comparisons,
        "gate_criteria": gate_criteria,
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)

    hashes = {
        "script": _sha256_file(Path(__file__).resolve()),
        "config": _sha256_file(config_path),
        "checkpoint": _sha256_file(checkpoint_path),
        "episode_split_plan": _sha256_file(planned_manifest_path),
        "episode_manifest": _sha256_file(manifest_path),
        "development_anchor_cache": _sha256_file(development_cache_path),
        "sealed_test_anchor_cache": _sha256_file(test_cache_path),
        "input_normalisation": _sha256_file(normalisation_path),
        "train_raw_local_error_mean": _sha256_file(raw_mean_path),
        "validation_grid": _sha256_file(grid_path),
        "selected_subspaces": _sha256_file(subspace_path),
        "selection": _sha256_file(selection_path),
        "test_results": _sha256_file(test_results_path),
        "test_per_sample_raw_mse": _sha256_file(per_sample_path),
        "summary": _sha256_file(summary_path),
    }
    _write_json(output_dir / "artifact_hashes.json", hashes)
    (output_dir / f"GATE.{scientific_status.lower()}").write_text(
        f"{scientific_status}\n", encoding="utf-8"
    )
    running_path.replace(output_dir / "STATUS.complete")
    return summary


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    claimed = False
    try:
        # _run claims the directory only after validating its inputs.
        summary = _run(args)
        claimed = True
    except Exception:
        error = traceback.format_exc()
        running = output_dir / "STATUS.running"
        running_is_ours = False
        if running.exists():
            try:
                running_is_ours = f"pid={os.getpid()}" in running.read_text(
                    encoding="utf-8"
                )
            except OSError:
                running_is_ours = False
        if running_is_ours:
            claimed = True
            (output_dir / "error.txt").write_text(error, encoding="utf-8")
            running.replace(output_dir / "STATUS.failed")
        print(error, flush=True)
        raise
    if claimed:
        print(json.dumps(summary, indent=2, sort_keys=True, default=_jsonable), flush=True)


if __name__ == "__main__":
    main()
