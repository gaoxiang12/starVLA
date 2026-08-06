#!/usr/bin/env python3
"""Gate Stage A of the predictable local-dynamics experiment.

Stage A only learns the shared spatial/channel projection bases.  The causal
student is frozen at zero, so this gate asks a deliberately narrow question:
does the compact code retain materially more *dynamic* frozen-base error than
a random separable projection, without changing the deployed prediction?

Metrics are first averaged within each optimizer step.  Only complete
eight-micro-batch groups enter the final quality window; the single record at
step 2000 is progress evidence only because it is emitted before the final
optimizer update and checkpoint save.  A PASS promotes the checkpoint to
Stage B; it is not evidence that the code is causally predictable and is not a
held-out result.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import torch


RANDOM_SEPARABLE_FRACTION = (4.0 / 32.0) * (64.0 / 384.0)
THRESHOLDS = {
    # 0.25 is 12x the 1/48 isotropic random separable expectation and means
    # that the 512-D code captures a material fraction of the 24,576-D error.
    "explained_fraction": 0.25,
    "explained_p10": 0.20,
    "initial_explained_gain": 0.15,
    "oracle_improvement": 0.10,
    "oracle_improvement_per_horizon": 0.07,
    "orthogonality_error_max": 1.0e-4,
    "target_median_std": 0.20,
    "target_effective_rank": 6.0,
    "mean_only_abs_improvement": 0.002,
    "max_explained_window_decline": 0.03,
    "zero_student_tolerance": 1.0e-6,
    "zero_predicted_std_tolerance": 1.0e-8,
    "fixed_mean_count": 8192.0,
    "code_dimensions": 512.0,
    "transition_tokens": 4.0,
    "delta_scale": 1.680324673652649,
    "delta_scale_tolerance": 1.0e-6,
    "capture_identity_tolerance": 1.0e-5,
    "records_per_complete_step": 8,
    "expected_quality_groups": 20,
}

ALIASES = {
    "final": ("innovation_final_raw_loss", "innovation_final_mse", "latent_loss"),
    "base": ("innovation_base_raw_loss", "innovation_base_mse", "latent_base_loss"),
    "oracle": ("innovation_oracle_raw_loss", "innovation_oracle_mse"),
    "explained": ("innovation_explained_fraction",),
    "capture": ("innovation_capture_loss",),
    "dynamic_base": ("innovation_dynamic_base_mse",),
    "dynamic_oracle": ("innovation_dynamic_oracle_mse",),
    "mean_only_improvement": ("innovation_mean_only_improvement",),
    "base_h1": ("innovation_base_mse_horizon_1",),
    "base_h2": ("innovation_base_mse_horizon_2",),
    "oracle_h1": ("innovation_oracle_mse_horizon_1",),
    "oracle_h2": ("innovation_oracle_mse_horizon_2",),
    "orthogonality": (
        "innovation_orthogonality_error",
        "innovation_orthogonality",
    ),
    "target_median_std": (
        "innovation_target_std_median",
        "innovation_target_code_std",
        "innovation_target_std",
    ),
    "target_effective_rank": ("innovation_effective_rank",),
    "predicted_std": (
        "innovation_pred_code_std",
        "innovation_predicted_std",
    ),
    "predicted_effective_rank": ("innovation_predicted_effective_rank",),
    "fixed_mean_count": ("innovation_fixed_mean_count",),
    "code_dimensions": ("innovation_code_dimensions",),
    "transition_tokens": ("innovation_transition_tokens",),
    "delta_scale": ("delta_scale",),
}


def _finite(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric(record: dict, aliases: Sequence[str]) -> Optional[float]:
    for name in aliases:
        value = _finite(record.get(name))
        if value is not None:
            return value
    return None


def _values(records: Sequence[dict], name: str) -> list[float]:
    values = []
    for record in records:
        value = _metric(record, ALIASES[name])
        if value is not None:
            values.append(value)
    return values


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _paired_mean(records: Sequence[dict], left: str, right: str) -> Optional[float]:
    differences = []
    for record in records:
        left_value = _metric(record, ALIASES[left])
        right_value = _metric(record, ALIASES[right])
        if left_value is not None and right_value is not None:
            differences.append(left_value - right_value)
    return _mean(differences)


def _aggregate_by_step(records: Sequence[dict]) -> list[dict]:
    """Average duplicate accumulation records before comparing optimizer steps."""

    grouped: dict[float, list[dict]] = {}
    for record in records:
        grouped.setdefault(float(record["step"]), []).append(record)

    aggregated = []
    for step in sorted(grouped):
        members = grouped[step]
        record = {"step": step, "_record_count": len(members)}
        for canonical, aliases in ALIASES.items():
            values = []
            for member in members:
                value = _metric(member, aliases)
                if value is not None:
                    values.append(value)
            if values:
                # Store under the first alias so the ordinary metric resolver
                # can consume both raw and per-step records.
                record[aliases[0]] = statistics.fmean(values)
        aggregated.append(record)
    return aggregated


def _energy_weighted_explained(records: Sequence[dict]) -> Optional[float]:
    dynamic_base = []
    dynamic_oracle = []
    for record in records:
        base = _metric(record, ALIASES["dynamic_base"])
        oracle = _metric(record, ALIASES["dynamic_oracle"])
        if base is not None and oracle is not None:
            dynamic_base.append(base)
            dynamic_oracle.append(oracle)
    denominator = sum(dynamic_base)
    if not dynamic_base or denominator <= 0:
        return None
    return 1.0 - sum(dynamic_oracle) / denominator


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary_contains_step(path: Path, expected_steps: int) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if int(payload.get("steps", -1)) == expected_steps:
                    return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return False


def _find_unique_suffix(state: dict, suffix: str) -> Optional[torch.Tensor]:
    matches = [value for key, value in state.items() if str(key).endswith(suffix)]
    if len(matches) != 1 or not torch.is_tensor(matches[0]):
        return None
    return matches[0]


def _verify_checkpoint(path: Path, *, inspect_innovation: bool) -> dict:
    result = {
        "exists": path.is_file(),
        "loadable": False,
        "finite": False,
        "innovation_shapes": False if inspect_innovation else None,
        "predictor_output_zero": False if inspect_innovation else None,
        "error": None,
    }
    if not path.is_file() or path.stat().st_size <= 0:
        result["error"] = "missing or empty"
        return result
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"expected state dict, got {type(state).__name__}")
        tensors = [value for value in state.values() if torch.is_tensor(value)]
        result["loadable"] = True
        result["finite"] = bool(tensors) and all(
            bool(torch.isfinite(value).all())
            for value in tensors
            if value.is_floating_point() or value.is_complex()
        )
        if inspect_innovation:
            channel = _find_unique_suffix(
                state, "predictable_innovation.basis.raw_basis"
            )
            spatial = _find_unique_suffix(
                state, "predictable_innovation.spatial_basis.raw_basis"
            )
            output_weight = _find_unique_suffix(
                state, "predictable_innovation.predictor.output.weight"
            )
            output_bias = _find_unique_suffix(
                state, "predictable_innovation.predictor.output.bias"
            )
            result["innovation_shapes"] = (
                channel is not None
                and spatial is not None
                and output_weight is not None
                and output_bias is not None
                and tuple(channel.shape) == (384, 64)
                and tuple(spatial.shape) == (32, 4)
                and tuple(output_weight.shape) == (64, 384)
                and tuple(output_bias.shape) == (64,)
            )
            result["predictor_output_zero"] = (
                output_weight is not None
                and output_bias is not None
                and int(torch.count_nonzero(output_weight)) == 0
                and int(torch.count_nonzero(output_bias)) == 0
            )
        del state
    except Exception as exc:  # gate must turn corrupt artifacts into FAIL
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _read_metrics(path: Path) -> tuple[list[dict], int]:
    records = []
    invalid_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                invalid_lines += 1
                continue
            if not isinstance(record, dict) or _finite(record.get("step")) is None:
                invalid_lines += 1
                continue
            records.append(record)
    return records, invalid_lines


def _terminal_state(run_dir: Path) -> Optional[str]:
    # A stale STATUS.running marker is harmless.  Failure always takes priority
    # if a manually resumed run happened to leave more than one marker behind.
    for state in ("failed", "stopped", "complete"):
        if (run_dir / f"STATUS.{state}").exists():
            return state
    return None


def summarize(
    run_dir: Path,
    *,
    expected_steps: int = 2000,
    window_steps: int = 500,
    checkpoint: Optional[Path] = None,
) -> dict:
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint = checkpoint or (
        run_dir / "checkpoints" / f"steps_{expected_steps}_pytorch_model.pt"
    )
    terminal_state = _terminal_state(run_dir)

    if not metrics_path.is_file():
        return {
            "status": "FAIL" if terminal_state else "PENDING",
            "reason": f"missing {metrics_path}",
            "run_dir": str(run_dir),
            "terminal_state": terminal_state,
            "criteria": {},
        }

    records, invalid_lines = _read_metrics(metrics_path)
    if not records:
        return {
            "status": "FAIL" if terminal_state else "PENDING",
            "reason": "metrics.jsonl contains no valid records",
            "run_dir": str(run_dir),
            "terminal_state": terminal_state,
            "invalid_lines": invalid_lines,
            "criteria": {},
        }

    latest_step = max(float(record["step"]) for record in records)
    per_step = _aggregate_by_step(records)
    quality_end = min(latest_step, float(expected_steps))
    window_start = max(0.0, quality_end - float(window_steps))
    expected_records = int(THRESHOLDS["records_per_complete_step"])
    # Exclude the final step: train_starvla logs its lone pre-update record
    # before saving. Normal logged steps contain exactly grad_accum=8 records.
    window = [
        record
        for record in per_step
        if window_start <= float(record["step"]) < quality_end
        and int(record["_record_count"]) == expected_records
    ]
    previous_start = max(0.0, window_start - float(window_steps))
    previous_window = [
        record
        for record in per_step
        if previous_start <= float(record["step"]) < window_start
        and int(record["_record_count"]) == expected_records
    ]
    raw_window = [
        record
        for record in records
        if window_start <= float(record["step"]) < quality_end
    ]
    all_group_counts = {
        str(int(record["step"])): int(record["_record_count"])
        for record in per_step
    }

    paired_zero_errors = []
    for record in records:
        final = _metric(record, ALIASES["final"])
        base = _metric(record, ALIASES["base"])
        if final is not None and base is not None:
            paired_zero_errors.append(abs(final - base))

    orthogonality = _values(window, "orthogonality")
    fixed_mean_counts = _values(window, "fixed_mean_count")
    code_dimensions = _values(window, "code_dimensions")
    transition_tokens = _values(window, "transition_tokens")
    explained_values = _values(window, "explained")
    capture_identity_errors = []
    for record in window:
        capture = _metric(record, ALIASES["capture"])
        explained = _metric(record, ALIASES["explained"])
        if capture is not None and explained is not None:
            capture_identity_errors.append(abs(capture + explained - 1.0))
    delta_scales = _values(window, "delta_scale")
    initial_records = [record for record in per_step if float(record["step"]) == 0.0]
    initial_explained = _energy_weighted_explained(initial_records)
    explained = _energy_weighted_explained(window)
    previous_explained = _energy_weighted_explained(previous_window)
    mean_only = _mean(_values(window, "mean_only_improvement"))
    checkpoint_verification = _verify_checkpoint(
        checkpoint, inspect_innovation=True
    ) if terminal_state == "complete" else {
        "exists": checkpoint.is_file(),
        "loadable": False,
        "finite": False,
        "innovation_shapes": False,
        "predictor_output_zero": False,
        "error": "verification deferred until STATUS.complete",
    }
    final_checkpoint = run_dir / "final_model" / "pytorch_model.pt"
    final_verification = _verify_checkpoint(
        final_checkpoint, inspect_innovation=False
    ) if terminal_state == "complete" else {
        "exists": final_checkpoint.is_file(),
        "loadable": False,
        "finite": False,
        "innovation_shapes": None,
        "predictor_output_zero": None,
        "error": "verification deferred until STATUS.complete",
    }
    metrics = {
        "base_raw_loss_mean": _mean(_values(window, "base")),
        "oracle_raw_loss_mean": _mean(_values(window, "oracle")),
        "oracle_improvement_mean": _paired_mean(window, "base", "oracle"),
        "oracle_improvement_horizon_1": _paired_mean(
            window, "base_h1", "oracle_h1"
        ),
        "oracle_improvement_horizon_2": _paired_mean(
            window, "base_h2", "oracle_h2"
        ),
        "explained_fraction_energy_weighted": explained,
        "explained_fraction_step_mean": _mean(explained_values),
        "explained_fraction_p10": _percentile(explained_values, 0.10),
        "initial_explained_fraction": initial_explained,
        "initial_explained_gain": (
            None
            if explained is None or initial_explained is None
            else explained - initial_explained
        ),
        "previous_window_explained_fraction": previous_explained,
        "explained_window_change": (
            None
            if explained is None or previous_explained is None
            else explained - previous_explained
        ),
        "orthogonality_error_max": max(orthogonality) if orthogonality else None,
        "target_median_std": _median(_values(window, "target_median_std")),
        "target_effective_rank_median": _median(
            _values(window, "target_effective_rank")
        ),
        "mean_only_improvement_mean": mean_only,
        "mean_only_abs_improvement": abs(mean_only) if mean_only is not None else None,
        "capture_identity_error_max": (
            max(capture_identity_errors) if capture_identity_errors else None
        ),
        "predicted_std_max": (
            max(_values(window, "predicted_std"))
            if _values(window, "predicted_std")
            else None
        ),
        "predicted_effective_rank_max": (
            max(_values(window, "predicted_effective_rank"))
            if _values(window, "predicted_effective_rank")
            else None
        ),
        "delta_scale_max_abs_error": (
            max(
                abs(value - THRESHOLDS["delta_scale"])
                for value in delta_scales
            )
            if delta_scales
            else None
        ),
        "student_base_max_abs_difference": (
            max(paired_zero_errors) if paired_zero_errors else None
        ),
        "fixed_mean_count_min": min(fixed_mean_counts) if fixed_mean_counts else None,
        "fixed_mean_count_max": max(fixed_mean_counts) if fixed_mean_counts else None,
        "code_dimensions_min": min(code_dimensions) if code_dimensions else None,
        "code_dimensions_max": max(code_dimensions) if code_dimensions else None,
        "transition_tokens_min": min(transition_tokens) if transition_tokens else None,
        "transition_tokens_max": max(transition_tokens) if transition_tokens else None,
    }

    def at_least(name: str, threshold: float) -> Optional[bool]:
        value = metrics[name]
        return None if value is None else value >= threshold

    def at_most(name: str, threshold: float) -> Optional[bool]:
        value = metrics[name]
        return None if value is None else value <= threshold

    criteria = {
        "terminal_complete": terminal_state == "complete",
        "reached_expected_step": latest_step >= expected_steps,
        "summary_contains_expected_step": _summary_contains_step(
            run_dir / "summary.jsonl", expected_steps
        ),
        "checkpoint_loadable": bool(
            checkpoint_verification["loadable"]
            and checkpoint_verification["finite"]
            and checkpoint_verification["innovation_shapes"]
        ),
        "final_model_loadable": bool(
            final_verification["loadable"] and final_verification["finite"]
        ),
        "checkpoint_predictor_still_zero": bool(
            checkpoint_verification["predictor_output_zero"]
        ),
        "complete_quality_groups": len(window)
        == int(THRESHOLDS["expected_quality_groups"]),
        "complete_previous_groups": len(previous_window)
        == int(THRESHOLDS["expected_quality_groups"]),
        "final_step_single_progress_record": all_group_counts.get(
            str(expected_steps)
        ) == 1,
        "no_invalid_metric_lines": invalid_lines == 0,
        "explained_fraction": at_least(
            "explained_fraction_energy_weighted", THRESHOLDS["explained_fraction"]
        ),
        "explained_fraction_p10": at_least(
            "explained_fraction_p10", THRESHOLDS["explained_p10"]
        ),
        "gain_over_initial_basis": at_least(
            "initial_explained_gain", THRESHOLDS["initial_explained_gain"]
        ),
        "oracle_improvement": at_least(
            "oracle_improvement_mean", THRESHOLDS["oracle_improvement"]
        ),
        "oracle_improvement_horizon_1": at_least(
            "oracle_improvement_horizon_1",
            THRESHOLDS["oracle_improvement_per_horizon"],
        ),
        "oracle_improvement_horizon_2": at_least(
            "oracle_improvement_horizon_2",
            THRESHOLDS["oracle_improvement_per_horizon"],
        ),
        "stable_vs_previous_window": (
            None
            if metrics["explained_window_change"] is None
            else metrics["explained_window_change"]
            >= -THRESHOLDS["max_explained_window_decline"]
        ),
        "orthogonality": at_most(
            "orthogonality_error_max", THRESHOLDS["orthogonality_error_max"]
        ),
        "target_not_collapsed": at_least(
            "target_median_std", THRESHOLDS["target_median_std"]
        ),
        "target_effective_rank": at_least(
            "target_effective_rank_median", THRESHOLDS["target_effective_rank"]
        ),
        "fixed_mean_not_the_gain": at_most(
            "mean_only_abs_improvement", THRESHOLDS["mean_only_abs_improvement"]
        ),
        "capture_identity": at_most(
            "capture_identity_error_max", THRESHOLDS["capture_identity_tolerance"]
        ),
        "student_remained_exact_base": at_most(
            "student_base_max_abs_difference",
            THRESHOLDS["zero_student_tolerance"],
        ),
        "student_code_remained_zero": (
            None
            if metrics["predicted_std_max"] is None
            or metrics["predicted_effective_rank_max"] is None
            else metrics["predicted_std_max"]
            <= THRESHOLDS["zero_predicted_std_tolerance"]
            and metrics["predicted_effective_rank_max"]
            <= THRESHOLDS["zero_predicted_std_tolerance"]
        ),
        "delta_scale_unchanged": at_most(
            "delta_scale_max_abs_error", THRESHOLDS["delta_scale_tolerance"]
        ),
        "fixed_mean_calibrated": (
            None
            if metrics["fixed_mean_count_min"] is None
            or metrics["fixed_mean_count_max"] is None
            else metrics["fixed_mean_count_min"] == THRESHOLDS["fixed_mean_count"]
            and metrics["fixed_mean_count_max"] == THRESHOLDS["fixed_mean_count"]
        ),
        "code_shape": (
            None
            if metrics["code_dimensions_min"] is None
            or metrics["code_dimensions_max"] is None
            or metrics["transition_tokens_min"] is None
            or metrics["transition_tokens_max"] is None
            else metrics["code_dimensions_min"] == THRESHOLDS["code_dimensions"]
            and metrics["code_dimensions_max"] == THRESHOLDS["code_dimensions"]
            and metrics["transition_tokens_min"] == THRESHOLDS["transition_tokens"]
            and metrics["transition_tokens_max"] == THRESHOLDS["transition_tokens"]
        ),
    }

    missing = [name for name, passed in criteria.items() if passed is None]
    terminal_failure = terminal_state in {"failed", "stopped"}
    if terminal_failure:
        status = "FAIL"
        reason = f"Stage A terminal state is {terminal_state}"
    elif terminal_state != "complete":
        status = "PENDING"
        reason = f"Stage A has reached step {latest_step:g}/{expected_steps}"
    elif missing:
        status = "FAIL"
        reason = "completed run is missing gate metrics: " + ", ".join(missing)
    elif all(criteria.values()):
        status = "PASS"
        reason = "compact coordinates passed the Stage-A representation gate"
    else:
        status = "FAIL"
        failed = [name for name, passed in criteria.items() if passed is False]
        reason = "failed criteria: " + ", ".join(failed)

    return {
        "status": status,
        "reason": reason,
        "scope": "representation capture only; predictability and held-out validation remain",
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        "terminal_state": terminal_state,
        "latest_step": latest_step,
        "expected_steps": expected_steps,
        "window_start_step": window_start,
        "previous_window_start_step": previous_start,
        "window_steps": window_steps,
        "window_records": len(raw_window),
        "window_complete_groups": len(window),
        "previous_window_complete_groups": len(previous_window),
        "optimizer_step_record_counts": all_group_counts,
        "invalid_lines": invalid_lines,
        "random_separable_fraction": RANDOM_SEPARABLE_FRACTION,
        "checkpoint_verification": checkpoint_verification,
        "final_model_verification": final_verification,
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "criteria": criteria,
        "missing_criteria": missing,
    }


def print_human(summary: dict) -> None:
    print(f"Local basis gate: {summary['status']}")
    print("Scope: representation capture only; predictability remains untested")
    print(f"Reason: {summary['reason']}")
    print(f"Run: {summary['run_dir']}")
    if "latest_step" not in summary:
        return
    print(
        f"Progress: {summary['latest_step']:g}/{summary['expected_steps']}; "
        f"window={summary['window_records']} records / "
        f"{summary['window_complete_groups']} complete optimizer-step groups"
    )
    for name, value in summary["metrics"].items():
        rendered = "N/A" if value is None else f"{float(value):.8f}"
        print(f"  {name}: {rendered}")
    print("Criteria:")
    for name, passed in summary["criteria"].items():
        result = "PENDING" if passed is None else "PASS" if passed else "FAIL"
        print(f"  {result}: {name}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _fixture_record(step: int, **overrides: float) -> dict:
    record = {
        "step": step,
        "innovation_final_raw_loss": 0.50,
        "innovation_base_raw_loss": 0.50,
        "innovation_oracle_raw_loss": 0.38,
        "innovation_dynamic_base_mse": 0.50,
        "innovation_dynamic_oracle_mse": 0.35,
        "innovation_capture_loss": 0.70,
        "innovation_explained_fraction": 0.30,
        "innovation_mean_only_improvement": 0.0001,
        "innovation_base_mse_horizon_1": 0.50,
        "innovation_base_mse_horizon_2": 0.50,
        "innovation_oracle_mse_horizon_1": 0.40,
        "innovation_oracle_mse_horizon_2": 0.40,
        "innovation_orthogonality_error": 5.0e-7,
        "innovation_target_std_median": 2.0,
        "innovation_effective_rank": 8.0,
        "innovation_pred_code_std": 0.0,
        "innovation_predicted_effective_rank": 0.0,
        "innovation_fixed_mean_count": 8192.0,
        "innovation_code_dimensions": 512.0,
        "innovation_transition_tokens": 4.0,
        "delta_scale": 1.680324673652649,
    }
    record.update(overrides)
    return record


def _make_fixture(root: Path, *, complete: bool, passing: bool = True) -> Path:
    run_dir = root / ("passing" if passing else "failing")
    run_dir.mkdir()
    records = []
    initial_overrides = {
        "innovation_dynamic_oracle_mse": 0.475,
        "innovation_capture_loss": 0.95,
        "innovation_explained_fraction": 0.05,
    }
    for _ in range(7):
        records.append(_fixture_record(0, **initial_overrides))
    for step in range(1000, 2000, 25):
        for _ in range(8):
            overrides = (
                {}
                if passing
                else {
                    "innovation_dynamic_oracle_mse": 0.49,
                    "innovation_capture_loss": 0.98,
                    "innovation_explained_fraction": 0.02,
                }
            )
            records.append(_fixture_record(step, **overrides))
    records.append(_fixture_record(2000))
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    if complete:
        (run_dir / "STATUS.complete").touch()
        (run_dir / "summary.jsonl").write_text(
            json.dumps({"steps": 2000}) + "\n", encoding="utf-8"
        )
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir()
        state = {
            "world_model.predictable_innovation.basis.raw_basis": torch.randn(384, 64),
            "world_model.predictable_innovation.spatial_basis.raw_basis": torch.randn(32, 4),
            "world_model.predictable_innovation.predictor.output.weight": torch.zeros(64, 384),
            "world_model.predictable_innovation.predictor.output.bias": torch.zeros(64),
        }
        torch.save(state, checkpoint_dir / "steps_2000_pytorch_model.pt")
        final_dir = run_dir / "final_model"
        final_dir.mkdir()
        torch.save(state, final_dir / "pytorch_model.pt")
    return run_dir


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="local_basis_gate_") as root:
        root_path = Path(root)
        passing = _make_fixture(root_path, complete=True)
        assert summarize(passing)["status"] == "PASS"

        pending_root = root_path / "pending_root"
        pending_root.mkdir()
        pending = _make_fixture(pending_root, complete=False)
        assert summarize(pending)["status"] == "PENDING"

        failing_root = root_path / "failing_root"
        failing_root.mkdir()
        failing = _make_fixture(failing_root, complete=True, passing=False)
        result = summarize(failing)
        assert result["status"] == "FAIL"
        assert result["criteria"]["explained_fraction"] is False
    print("fixture smoke test: PASS (PASS/PENDING/FAIL and duplicate micro-batches)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--expected-steps", type=int, default=2000)
    parser.add_argument("--window-steps", type=int, default=500)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.run_dir is None:
        parser.error("run_dir is required unless --self-test is used")
    if args.expected_steps <= 0 or args.window_steps <= 0:
        parser.error("--expected-steps and --window-steps must be positive")

    summary = summarize(
        args.run_dir,
        expected_steps=args.expected_steps,
        window_steps=args.window_steps,
        checkpoint=args.checkpoint,
    )
    if args.json_out is not None:
        _write_json_atomic(args.json_out, summary)
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human(summary)
    return {"PASS": 0, "FAIL": 1, "PENDING": 2}[summary["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
