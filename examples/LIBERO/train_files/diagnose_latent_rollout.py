"""Measure how far the LeWM-OFT latent world model can roll out.

The deployed objective only ever supervises a single re-anchoring step: the
predictor sees the *true* current latent and emits ``n_future`` latents.  Nothing
in that objective asks the predictor to stay stable when it is fed its own
output, so "can it roll out?" is an open empirical question rather than a design
property.

This script answers it directly on held-out episodes by comparing three curves
at each horizon:

``copy``
    the trivial baseline that predicts "nothing moves".  Normalising by this is
    what makes numbers comparable across horizons, since the true motion grows
    with the horizon.
``teacher_forced``
    re-anchor on the *true* latent at every step.  This isolates how hard the
    horizon itself is.
``rollout``
    re-anchor on the model's *own* prediction.  The gap against
    ``teacher_forced`` is exactly the compounding error introduced by closing
    the loop, which is the quantity a rollout objective is supposed to shrink.

Both curves are reported as ``mse / copy_mse``: below 1.0 means the model beats
"the world is static", above 1.0 means it is actively harmful.

A fourth, zero-parameter curve settles a separate question.  ``interpolation``
predicts frame ``t`` as the midpoint of frames ``t-1`` and ``t+1``: it is an
oracle that gets to see the future, so whatever error it leaves is *aleatoric*
noise in the latent space itself rather than information the predictor is
missing.  If interpolation is near-perfect, a forward predictor stuck at 0.5 is
limited by missing information; if interpolation is also stuck near 0.5, then
half of every frame-to-frame delta is unpredictable-in-principle jitter and no
amount of conditioning will drive the forward loss lower.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset

from probe_action_conditioned_ceiling import (  # noqa: E402  (same-directory script)
    _episode_pools,
    _hash_u64,
    _load_model,
    _valid_step_bounds,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--data-mix",
        default="libero_all_wm_l10_augmented_rollout_h32",
        help="mixture whose video_indices span the full rollout horizon",
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--anchors-per-episode", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
        step = (
            first
            + _hash_u64(seed, "anchor", candidate["episode_key"], anchor_index) % width
        )
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
def _encode(model: Any, examples: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
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
    return latent, goal


class _Accumulator:
    """Streaming sums so the diagnostic never holds every latent in memory."""

    def __init__(self) -> None:
        self.error = 0.0
        self.copy = 0.0
        self.cosine = 0.0
        self.pred_displacement = 0.0
        self.true_displacement = 0.0
        self.count = 0

    def update(
        self, prediction: torch.Tensor, truth: torch.Tensor, current: torch.Tensor
    ) -> None:
        batch = prediction.shape[0]
        self.error += float((prediction - truth).square().mean()) * batch
        self.copy += float((current - truth).square().mean()) * batch
        # Displacement magnitude separates "accurate" from "frozen": a predictor
        # that shrinks to zero motion also lands near copy_ratio 1.0, and one
        # that drifts lands above 1.0 with a large displacement.
        self.pred_displacement += float((prediction - current).square().mean()) * batch
        self.true_displacement += float((truth - current).square().mean()) * batch
        self.cosine += (
            float(
                F.cosine_similarity(
                    (prediction - current).flatten(1),
                    (truth - current).flatten(1),
                    dim=-1,
                    eps=1e-8,
                ).mean()
            )
            * batch
        )
        self.count += batch

    def summary(self) -> dict[str, float]:
        count = max(self.count, 1)
        error = self.error / count
        copy = self.copy / count
        pred_displacement = (self.pred_displacement / count) ** 0.5
        true_displacement = (self.true_displacement / count) ** 0.5
        return {
            "mse": error,
            "copy_mse": copy,
            "copy_ratio": error / max(copy, 1e-12),
            "direction_cosine": self.cosine / count,
            "pred_displacement_rms": pred_displacement,
            "true_displacement_rms": true_displacement,
            "displacement_ratio": pred_displacement / max(true_displacement, 1e-12),
            "anchors": self.count,
        }


@torch.inference_mode()
def _evaluate(model: Any, latent: torch.Tensor, goal: torch.Tensor, steps: int):
    """Return per-step rollout and teacher-forced predictions."""
    world_model = model.world_model
    ctx_len = int(model.wm_ctx_len)
    n_future = int(model.n_future)

    rollout = world_model.rollout_future(latent[:, :ctx_len], steps=steps, goal=goal)

    teacher = []
    for step in range(steps):
        start = ctx_len + step * n_future
        # Anchor on the ground-truth window that ends right before this step.
        window = latent[:, start - ctx_len : start]
        teacher.append(world_model.regress_future(window, goal=goal))
    return rollout, torch.cat(teacher, dim=1)


def _run(args: argparse.Namespace) -> dict:
    checkpoint = Path(args.checkpoint).resolve()
    config_path = (
        Path(args.config).resolve()
        if args.config
        else checkpoint.parents[1] / "config.yaml"
    )
    output_dir = Path(args.output_dir or (checkpoint.parents[1] / "latent_rollout_diagnostic"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg, model = _load_model(config_path, checkpoint, device)
    data_cfg = cfg.datasets.vla_data
    data_cfg.future_obs_frames = True
    data_cfg.include_state = True
    data_cfg.data_mix = args.data_mix
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

    n_future = int(model.n_future)
    ctx_len = int(model.wm_ctx_len)
    probe = mixture.datasets[0]
    video_indices = None
    for key in probe.modality_keys["video"]:
        if key in probe.delta_indices:
            video_indices = [int(value) for value in probe.delta_indices[key]]
            break
    if video_indices is None:
        raise RuntimeError("dataset exposes no video delta indices")
    future_frames = len(video_indices) - 1
    steps = future_frames // n_future
    if steps < 2:
        raise RuntimeError(
            f"data_mix {args.data_mix!r} only supplies {future_frames} future frames, "
            f"which is not enough for a multi-step rollout with n_future={n_future}"
        )

    pools = _episode_pools(mixture, seed=args.seed)
    candidates = pools[args.split][: args.episodes]
    if len(candidates) < args.episodes:
        raise RuntimeError(
            f"split {args.split} has {len(candidates)} episodes < {args.episodes}"
        )

    rollout_stats = [_Accumulator() for _ in range(steps)]
    teacher_stats = [_Accumulator() for _ in range(steps)]
    interpolation_stats = _Accumulator()

    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        latent, goal = _encode(model, pending)
        rollout, teacher = _evaluate(model, latent, goal, steps)
        current = latent[:, ctx_len - 1 : ctx_len]
        for step in range(steps):
            start = ctx_len + step * n_future
            truth = latent[:, start : start + n_future]
            window = slice(step * n_future, (step + 1) * n_future)
            rollout_stats[step].update(rollout[:, window], truth, current)
            teacher_stats[step].update(teacher[:, window], truth, current)
        for index in range(1, latent.shape[1] - 1):
            previous = latent[:, index - 1 : index]
            following = latent[:, index + 1 : index + 2]
            interpolation_stats.update(
                0.5 * (previous + following), latent[:, index : index + 1], previous
            )
        pending.clear()

    started = time.time()
    skipped = 0
    for index, candidate in enumerate(candidates):
        try:
            samples = _episode_samples(
                candidate,
                seed=args.seed,
                anchors=args.anchors_per_episode,
                expected_future=future_frames,
            )
        except Exception as error:  # noqa: BLE001 - a few source videos are corrupt
            skipped += 1
            print(
                f"[{args.split}] skipping {candidate['episode_key']}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            continue
        for sample in samples:
            pending.append(sample)
            if len(pending) >= args.batch_size:
                flush()
        if (index + 1) % 50 == 0:
            print(
                f"[{args.split}] {index + 1}/{len(candidates)} episodes "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )
    flush()

    frame_stride = video_indices[1] - video_indices[0]
    report = {
        "checkpoint": str(checkpoint),
        "data_mix": args.data_mix,
        "split": args.split,
        "seed": args.seed,
        "n_future": n_future,
        "context_len": ctx_len,
        "frame_stride": frame_stride,
        "skipped_episodes": skipped,
        # Oracle that sees both neighbours: its error is the latent noise floor
        # for one ``frame_stride`` step, normalised by the same copy baseline.
        "interpolation_oracle": interpolation_stats.summary(),
        "steps": [
            {
                "step": step + 1,
                "horizon_frames": (step + 1) * n_future * frame_stride,
                "rollout": rollout_stats[step].summary(),
                "teacher_forced": teacher_stats[step].summary(),
            }
            for step in range(steps)
        ],
    }
    output_path = output_dir / f"rollout_{args.split}_{args.data_mix}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {output_path}")
    return report


def main() -> None:
    args = _parse_args()
    report = _run(args)
    print("\n=== copy-normalised error (1.0 == predicting a static world) ===")
    print(
        f"{'horizon':>8}  {'teacher':>9}  {'rollout':>9}  {'compounding':>12}"
        f"  {'roll_disp':>10}  {'roll_cos':>9}"
    )
    for entry in report["steps"]:
        teacher = entry["teacher_forced"]["copy_ratio"]
        rollout = entry["rollout"]
        print(
            f"{'+' + str(entry['horizon_frames']):>8}  {teacher:9.4f}  "
            f"{rollout['copy_ratio']:9.4f}  {rollout['copy_ratio'] - teacher:12.4f}"
            f"  {rollout['displacement_ratio']:10.4f}"
            f"  {rollout['direction_cosine']:9.4f}"
        )
    oracle = report["interpolation_oracle"]
    print(
        f"\ntwo-sided interpolation oracle over one {report['frame_stride']}-frame "
        f"step: copy_ratio={oracle['copy_ratio']:.4f} "
        f"cos={oracle['direction_cosine']:.4f}"
    )


if __name__ == "__main__":
    main()
