#!/usr/bin/env python3
"""Plot scalar metrics from starVLA's rich-formatted training log.

Example:
    .venv/bin/python scripts/plot_training_losses.py \
        playground/Checkpoints/<run>/train_200k.log
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STEP_RE = re.compile(r"Step\s+(\d+),\s+Loss:")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

DEFAULT_GROUPS = (
    (
        "Losses",
        (
            "action_dit_loss",
            "l1_action_loss",
            "latent_loss",
            "sigreg_loss",
        ),
    ),
    (
        "Delta diagnostics",
        (
            "delta_scale",
            "delta_target_rms",
            "delta_pred_rms",
            "delta_copy_mse",
            "delta_pred_mse",
            "delta_to_copy_ratio",
            "delta_direction_cosine",
        ),
    ),
    (
        "Learning rates",
        (
            "learning_rate/action_model",
            "learning_rate/backbone.encoder",
            "learning_rate/base",
        ),
    ),
)


def parse_log(path: Path, max_step: int | None = None) -> dict[str, np.ndarray]:
    """Parse one scalar record for every ``Step N, Loss:`` block."""
    text = path.read_text(errors="replace")
    matches = list(STEP_RE.finditer(text))
    records: list[dict[str, float]] = []

    for index, match in enumerate(matches):
        step = int(match.group(1))
        if max_step is not None and step > max_step:
            break
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end() : block_end]
        record: dict[str, float] = {"step": float(step)}
        for key, value in re.findall(rf"'([^']+)':\s*({NUMBER})", block):
            record[key] = float(value)
        records.append(record)

    if not records:
        raise ValueError(f"No 'Step N, Loss:' records found in {path}")

    keys = sorted({key for record in records for key in record if key != "step"})
    result = {"step": np.array([record["step"] for record in records])}
    for key in keys:
        result[key] = np.array([record.get(key, np.nan) for record in records])
    return result


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    window = min(window, len(values))
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.array([np.nanmedian(padded[index : index + window]) for index in range(len(values))])


def plot_metrics(data: dict[str, np.ndarray], output: Path, smooth: int, log_y: bool) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, constrained_layout=True)
    steps = data["step"]
    plotted = 0

    for axis, (title, metric_names) in zip(axes, DEFAULT_GROUPS):
        for metric in metric_names:
            values = data.get(metric)
            if values is None or not np.isfinite(values).any():
                continue
            valid = np.isfinite(values)
            axis.plot(steps[valid], values[valid], alpha=0.28, linewidth=0.9)
            if smooth > 1 and valid.sum() > 1:
                smoothed = rolling_median(values, smooth)
                axis.plot(steps[valid], smoothed[valid], linewidth=1.8, label=f"{metric} (median {smooth})")
            else:
                axis.plot(steps[valid], values[valid], linewidth=1.5, label=metric)
            plotted += 1
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)
        if log_y:
            axis.set_yscale("log")

    if plotted == 0:
        raise ValueError("None of the default metrics were found in the log")
    axes[-1].set_xlabel("optimization step")
    figure.suptitle("starVLA training metrics", fontsize=16)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to train_*.log")
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path")
    parser.add_argument("--smooth", type=int, default=1, help="Rolling median window; 1 disables smoothing")
    parser.add_argument("--log-y", action="store_true", help="Use logarithmic y axes")
    parser.add_argument("--max-step", type=int, help="Only plot records up to this step")
    args = parser.parse_args()

    output = args.output or args.log.with_suffix(".losses.png")
    data = parse_log(args.log, max_step=args.max_step)
    plot_metrics(data, output, smooth=args.smooth, log_y=args.log_y)
    print(f"parsed {len(data['step'])} records, steps {int(data['step'][0])}-{int(data['step'][-1])}")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
