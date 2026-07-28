"""Offline diagnostics and visualizations for a trained latent progress checker.

This evaluates deterministic samples from the configured LIBERO training
mixture. It is a calibration/behavior diagnostic, not a held-out benchmark:
the current project config does not define a separate progress validation set.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import baseframework


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, including correct handling of ties."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size < 2 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def regression_metrics(target: Iterable[float], prediction: Iterable[float]) -> dict:
    """Compute scale-independent progress regression diagnostics."""
    target = np.asarray(list(target), dtype=np.float64)
    prediction = np.asarray(list(prediction), dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 1 or target.size == 0:
        raise ValueError("target and prediction must be non-empty vectors of equal size")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("target and prediction must be finite")

    error = prediction - target
    target_variance = np.square(target - target.mean()).sum()
    r_squared = (
        1.0 - np.square(error).sum() / target_variance
        if target_variance > 1e-12
        else None
    )
    return {
        "count": int(target.size),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "pearson": _correlation(target, prediction),
        "spearman": _correlation(_rankdata(target), _rankdata(prediction)),
        "r_squared": float(r_squared) if r_squared is not None else None,
    }


def calibration_table(
    target: Iterable[float],
    prediction: Iterable[float],
    *,
    num_bins: int = 10,
) -> list[dict]:
    target = np.asarray(list(target), dtype=np.float64)
    prediction = np.asarray(list(prediction), dtype=np.float64)
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    rows = []
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (target >= left) & (
            (target <= right) if index == num_bins - 1 else (target < right)
        )
        if not mask.any():
            continue
        rows.append(
            {
                "bin": index,
                "left": float(left),
                "right": float(right),
                "count": int(mask.sum()),
                "target_mean": float(target[mask].mean()),
                "prediction_mean": float(prediction[mask].mean()),
                "mae": float(np.abs(prediction[mask] - target[mask]).mean()),
            }
        )
    return rows


def apply_deployed_ema(records: list[dict], momentum: float) -> list[dict]:
    """Apply the same causal per-episode EMA used by online inference."""
    if not 0.0 <= momentum < 1.0:
        raise ValueError("EMA momentum must be in [0, 1)")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["episode_id"])].append(record)
    for episode_records in grouped.values():
        previous = None
        for record in sorted(episode_records, key=lambda item: item["target"]):
            raw = float(record["prediction"])
            filtered = (
                raw if previous is None else momentum * previous + (1 - momentum) * raw
            )
            record["deployed_ema_prediction"] = filtered
            previous = filtered
    return records


def trajectory_metrics(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["episode_id"])].append(record)

    episodes = []
    for episode_id, episode_records in grouped.items():
        ordered = sorted(episode_records, key=lambda item: item["target"])
        target = np.asarray([item["target"] for item in ordered])
        prediction = np.asarray([item["prediction"] for item in ordered])
        deployed = np.asarray(
            [item["deployed_ema_prediction"] for item in ordered]
        )
        differences = np.diff(prediction)
        deployed_differences = np.diff(deployed)
        episodes.append(
            {
                "episode_id": episode_id,
                "dataset": ordered[0]["dataset"],
                "instruction": ordered[0]["instruction"],
                "count": len(ordered),
                "mae": float(np.abs(prediction - target).mean()),
                "spearman": _correlation(_rankdata(target), _rankdata(prediction)),
                "nondecreasing_fraction": (
                    float((differences >= -1e-3).mean())
                    if differences.size
                    else 1.0
                ),
                "predicted_endpoint_gap": float(prediction[-1] - prediction[0]),
                "deployed_ema_mae": float(np.abs(deployed - target).mean()),
                "deployed_ema_spearman": _correlation(
                    _rankdata(target), _rankdata(deployed)
                ),
                "deployed_ema_nondecreasing_fraction": (
                    float((deployed_differences >= -1e-3).mean())
                    if deployed_differences.size
                    else 1.0
                ),
                "deployed_ema_endpoint_gap": float(deployed[-1] - deployed[0]),
            }
        )

    valid_spearman = [
        row["spearman"] for row in episodes if row["spearman"] is not None
    ]
    valid_deployed_spearman = [
        row["deployed_ema_spearman"]
        for row in episodes
        if row["deployed_ema_spearman"] is not None
    ]
    return {
        "num_episodes": len(episodes),
        "mean_mae": float(np.mean([row["mae"] for row in episodes])),
        "mean_spearman": (
            float(np.mean(valid_spearman)) if valid_spearman else None
        ),
        "mean_nondecreasing_fraction": float(
            np.mean([row["nondecreasing_fraction"] for row in episodes])
        ),
        "mean_predicted_endpoint_gap": float(
            np.mean([row["predicted_endpoint_gap"] for row in episodes])
        ),
        "deployed_ema_mean_mae": float(
            np.mean([row["deployed_ema_mae"] for row in episodes])
        ),
        "deployed_ema_mean_spearman": (
            float(np.mean(valid_deployed_spearman))
            if valid_deployed_spearman
            else None
        ),
        "deployed_ema_mean_nondecreasing_fraction": float(
            np.mean(
                [row["deployed_ema_nondecreasing_fraction"] for row in episodes]
            )
        ),
        "deployed_ema_mean_endpoint_gap": float(
            np.mean([row["deployed_ema_endpoint_gap"] for row in episodes])
        ),
        "episodes": episodes,
    }


def _make_trajectory_sample(dataset, trajectory_id: int, step: int) -> dict:
    raw_data = dataset.get_step_data(trajectory_id, step)
    data = dataset.transforms(raw_data)
    sample = dataset._pack_sample(data)
    return dataset._attach_progress_fields(sample, trajectory_id, step)


@torch.inference_mode()
def _predict_batch(model, examples: list[dict], device: torch.device) -> list[dict]:
    frames = [
        [
            example["image"],
            example["progress_start_image"],
            example["progress_goal_image"],
        ]
        for example in examples
    ]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        patch_tokens = model.backbone.encode_patch_frames(frames)
    latent = model.visual_token_pooler(patch_tokens.float())
    current_latent = latent[:, 0]
    start_latent = latent[:, 1]
    true_goal_latent = latent[:, 2]
    task_embedding = model._embed_task(
        [example["lang"] for example in examples], device=device
    )
    predicted_goal_latent = model.progress_goal_predictor(
        start_latent, task_embedding
    )

    predicted = model.progress_checker(
        start_latent, current_latent, predicted_goal_latent
    )
    oracle = model.progress_checker(
        start_latent, current_latent, true_goal_latent
    )
    start_anchor = model.progress_checker(
        start_latent, start_latent, predicted_goal_latent
    )
    goal_anchor = model.progress_checker(
        start_latent, true_goal_latent, predicted_goal_latent
    )
    goal_cosine = F.cosine_similarity(
        predicted_goal_latent.flatten(1),
        true_goal_latent.flatten(1),
        dim=-1,
        eps=1e-6,
    )

    # Verify the deployed action path actually responds to the scalar progress.
    predicted_future_latent = model.world_model.regress_future(
        current_latent[:, None], goal=task_embedding
    )
    head_tokens = torch.cat(
        [current_latent[:, None], predicted_future_latent], dim=1
    )
    current_state = (
        model._current_state_tensor(examples, device)
        if model.use_state_cond
        else None
    )
    base_queries = model._pool_visual_tokens_to_action_queries(
        head_tokens, state=current_state
    )
    base_action = model.action_model.predict_action(base_queries)
    zero_progress = torch.zeros(len(examples), device=device)
    one_progress = torch.ones(len(examples), device=device)
    zero_action = model.action_model.predict_action(
        model.progress_action_conditioner(base_queries, zero_progress)
    )
    one_action = model.action_model.predict_action(
        model.progress_action_conditioner(base_queries, one_progress)
    )
    predicted_action = model.action_model.predict_action(
        model.progress_action_conditioner(base_queries, predicted["progress"])
    )
    action_endpoint_swing = (one_action - zero_action).abs().mean(dim=(1, 2))
    action_conditioning_shift = (predicted_action - base_action).abs().mean(
        dim=(1, 2)
    )

    rows = []
    for index, example in enumerate(examples):
        episode_id = str(example["progress_episode_id"])
        rows.append(
            {
                "dataset": episode_id.rsplit(":", 1)[0],
                "episode_id": episode_id,
                "instruction": str(example["lang"]),
                "target": float(example["progress_target"]),
                "prediction": float(predicted["progress"][index]),
                "geometric_prediction": float(
                    predicted["geometric_progress"][index]
                ),
                # This is an ablation, not a true oracle: the checker and goal
                # predictor were co-trained, so swapping in a real goal latent
                # changes the checker's input distribution.
                "true_goal_substitution_prediction": float(
                    oracle["progress"][index]
                ),
                "start_anchor_prediction": float(start_anchor["progress"][index]),
                "goal_anchor_prediction": float(goal_anchor["progress"][index]),
                "goal_latent_cosine": float(goal_cosine[index]),
                "action_progress_0_to_1_mae": float(action_endpoint_swing[index]),
                "action_conditioning_shift_mae": float(
                    action_conditioning_shift[index]
                ),
            }
        )
    return rows


def _batched_predictions(model, samples: list[dict], batch_size: int, device):
    records = []
    for start in range(0, len(samples), batch_size):
        records.extend(
            _predict_batch(model, samples[start : start + batch_size], device)
        )
        print(
            f"evaluated {min(start + batch_size, len(samples))}/{len(samples)}",
            flush=True,
        )
    return records


def _moving_average(values: np.ndarray, window: int = 25) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    averaged = np.convolve(values, kernel, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), averaged])


def _plot_training_curves(metrics_path: Path, output_path: Path) -> None:
    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    steps = np.asarray([row["step"] for row in records])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, label in (
        ("progress_regression_loss", "regression"),
        ("progress_anchor_loss", "anchor"),
        ("progress_goal_loss", "goal"),
        ("progress_auxiliary_loss", "weighted auxiliary"),
    ):
        values = np.asarray([row[name] for row in records])
        axes[0].plot(steps, _moving_average(values), label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("loss (25-log moving average)")
    axes[0].set_title("Progress losses during training")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    prediction = np.asarray([row["progress_mean"] for row in records])
    target = np.asarray([row["progress_target_mean"] for row in records])
    axes[1].plot(steps, _moving_average(prediction), label="predicted")
    axes[1].plot(steps, _moving_average(target), label="target")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("batch mean progress")
    axes[1].set_title("Predicted and target batch means")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_global_diagnostics(
    records: list[dict],
    calibration: list[dict],
    output_path: Path,
) -> None:
    target = np.asarray([row["target"] for row in records])
    prediction = np.asarray([row["prediction"] for row in records])
    action_swing = np.asarray(
        [row["action_progress_0_to_1_mae"] for row in records]
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].scatter(target, prediction, s=12, alpha=0.25, edgecolors="none")
    axes[0, 0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0, 0].set(xlim=(0, 1), ylim=(0, 1))
    axes[0, 0].set_xlabel("target progress")
    axes[0, 0].set_ylabel("predicted progress")
    axes[0, 0].set_title("Per-sample progress predictions")
    axes[0, 0].grid(alpha=0.2)

    bin_target = [row["target_mean"] for row in calibration]
    bin_prediction = [row["prediction_mean"] for row in calibration]
    axes[0, 1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0, 1].plot(bin_target, bin_prediction, "o-", color="#2b6cb0")
    axes[0, 1].set(xlim=(0, 1), ylim=(0, 1))
    axes[0, 1].set_xlabel("mean target progress")
    axes[0, 1].set_ylabel("mean predicted progress")
    axes[0, 1].set_title("10-bin calibration")
    axes[0, 1].grid(alpha=0.2)

    centers = [0.5 * (row["left"] + row["right"]) for row in calibration]
    errors = [row["mae"] for row in calibration]
    axes[1, 0].bar(centers, errors, width=0.085, color="#38a169")
    axes[1, 0].set(xlim=(0, 1))
    axes[1, 0].set_xlabel("target progress bin")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].set_title("Error by task stage")
    axes[1, 0].grid(axis="y", alpha=0.2)

    axes[1, 1].hist(action_swing, bins=30, color="#805ad5", alpha=0.85)
    axes[1, 1].axvline(
        action_swing.mean(),
        color="black",
        linestyle="--",
        label=f"mean={action_swing.mean():.5f}",
    )
    axes[1, 1].set_xlabel("mean |action(progress=1) - action(progress=0)|")
    axes[1, 1].set_ylabel("sample count")
    axes[1, 1].set_title("Action-head sensitivity to progress")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_trajectories(records: list[dict], output_path: Path) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["episode_id"]].append(record)
    episode_items = list(grouped.items())
    columns = 2
    rows = int(np.ceil(len(episode_items) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.7 * rows), squeeze=False)
    for axis, (episode_id, episode_records) in zip(axes.flat, episode_items):
        ordered = sorted(episode_records, key=lambda item: item["target"])
        target = [item["target"] for item in ordered]
        axis.plot(target, target, "k--", label="target")
        axis.plot(
            target,
            [item["prediction"] for item in ordered],
            "o-",
            markersize=3,
            label="raw prediction",
        )
        axis.plot(
            target,
            [item["deployed_ema_prediction"] for item in ordered],
            ".-",
            color="#805ad5",
            alpha=0.85,
            label="deployed EMA",
        )
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.set_title(episode_id, fontsize=9)
        axis.set_xlabel("normalized episode time")
        axis.set_ylabel("progress")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(episode_items) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Progress along complete episodes", y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_csv(records: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument(
        "--curve-stride",
        type=int,
        default=0,
        help="Episode steps between progress queries; 0 uses action_horizon.",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if min(
        args.num_samples,
        args.batch_size,
        args.num_trajectories,
    ) < 1:
        raise ValueError("sample, batch, and trajectory counts must be positive")
    if args.curve_stride < 0:
        raise ValueError("curve-stride must be non-negative")

    run_dir = args.checkpoint.parent.parent
    output_dir = args.output_dir or run_dir / "progress_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    full_config = run_dir / "config.full.yaml"
    if not full_config.is_file():
        raise FileNotFoundError(full_config)

    device = torch.device(args.device)
    model = baseframework.from_pretrained(str(args.checkpoint)).to(device).eval()
    if not model.use_progress_checker:
        raise ValueError("checkpoint config has use_progress_checker=false")
    curve_stride = args.curve_stride or model.action_horizon

    cfg = OmegaConf.load(full_config)
    cfg.datasets.vla_data.include_progress = True
    cfg.datasets.vla_data.future_obs_frames = False
    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        mode="test",
        balance_dataset_weights=cfg.datasets.vla_data.get(
            "balance_dataset_weights", False
        ),
        balance_trajectory_weights=cfg.datasets.vla_data.get(
            "balance_trajectory_weights", False
        ),
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(
        len(dataset), size=min(args.num_samples, len(dataset)), replace=False
    )
    global_samples = [dataset[int(index)] for index in indices]
    global_records = _batched_predictions(
        model, global_samples, args.batch_size, device
    )

    curve_samples = []
    per_dataset = max(1, int(np.ceil(args.num_trajectories / len(dataset.datasets))))
    for single_dataset in dataset.datasets:
        available = len(single_dataset.trajectory_ids)
        chosen = rng.choice(
            available, size=min(per_dataset, available), replace=False
        )
        for trajectory_index in chosen:
            trajectory_id = int(single_dataset.trajectory_ids[trajectory_index])
            length = int(single_dataset.trajectory_lengths[trajectory_index])
            steps = np.unique(
                np.concatenate(
                    [
                        np.arange(0, length, curve_stride, dtype=int),
                        np.asarray([max(length - 1, 0)], dtype=int),
                    ]
                )
            )
            curve_samples.extend(
                _make_trajectory_sample(single_dataset, trajectory_id, int(step))
                for step in steps
            )
            if len(
                {sample["progress_episode_id"] for sample in curve_samples}
            ) >= args.num_trajectories:
                break
        if len(
            {sample["progress_episode_id"] for sample in curve_samples}
        ) >= args.num_trajectories:
            break
    curve_records = _batched_predictions(
        model, curve_samples, args.batch_size, device
    )
    curve_records = apply_deployed_ema(curve_records, model.progress_ema)

    target = [row["target"] for row in global_records]
    prediction = [row["prediction"] for row in global_records]
    calibration = calibration_table(target, prediction)
    summary = {
        "scope": (
            "deterministic offline diagnostic samples from the configured "
            "training mixture; not a held-out validation benchmark"
        ),
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "trajectory_query_stride": int(curve_stride),
        "prediction": regression_metrics(target, prediction),
        "geometric_baseline": regression_metrics(
            target, [row["geometric_prediction"] for row in global_records]
        ),
        "true_goal_substitution_ablation": regression_metrics(
            target,
            [
                row["true_goal_substitution_prediction"]
                for row in global_records
            ],
        ),
        "constant_half_baseline": regression_metrics(
            target, np.full(len(global_records), 0.5)
        ),
        "anchors": {
            "start_prediction_mean": float(
                np.mean([row["start_anchor_prediction"] for row in global_records])
            ),
            "goal_prediction_mean": float(
                np.mean([row["goal_anchor_prediction"] for row in global_records])
            ),
            "goal_latent_cosine_mean": float(
                np.mean([row["goal_latent_cosine"] for row in global_records])
            ),
        },
        "action_conditioning": {
            "progress_0_to_1_action_mae_mean": float(
                np.mean(
                    [row["action_progress_0_to_1_mae"] for row in global_records]
                )
            ),
            "predicted_progress_vs_no_conditioning_action_mae_mean": float(
                np.mean(
                    [
                        row["action_conditioning_shift_mae"]
                        for row in global_records
                    ]
                )
            ),
        },
        "calibration": calibration,
        "trajectories": trajectory_metrics(curve_records),
    }

    (output_dir / "progress_evaluation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    _write_csv(global_records, output_dir / "progress_predictions.csv")
    _write_csv(curve_records, output_dir / "progress_trajectory_predictions.csv")
    _plot_global_diagnostics(
        global_records,
        calibration,
        output_dir / "progress_scatter_calibration.png",
    )
    _plot_trajectories(
        curve_records, output_dir / "progress_trajectory_curves.png"
    )
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.is_file():
        _plot_training_curves(
            metrics_path, output_dir / "progress_training_curves.png"
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote progress diagnostics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
