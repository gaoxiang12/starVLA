#!/usr/bin/env python3
"""Summarize the predictable-innovation 10k training-feasibility gate.

The run directory is treated as read-only.  Metrics are aggregated over the
latest ``--window-steps`` optimization-step interval, including all logged
micro-batch records in that interval.  A run that has not reached its expected
final step is PENDING rather than prematurely failing the gate.

A PASS here is not a held-out validation result. It only promotes the run to a
separate episode-held-out predictability and shuffle-ablation evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence


METRIC_ALIASES = {
    "latent_loss": (
        "latent_loss",
        "innovation_final_raw_loss",
        "innovation_final_mse",
    ),
    "latent_base_loss": (
        "latent_base_loss",
        "innovation_base_raw_loss",
        "innovation_base_mse",
    ),
    "innovation_oracle_raw_loss": (
        "innovation_oracle_raw_loss",
        "innovation_oracle_raw_mse",
        "innovation_oracle_mse",
    ),
    "innovation_target_code_std": (
        "innovation_target_code_std",
        "innovation_target_std",
    ),
    "innovation_pred_code_std": (
        "innovation_pred_code_std",
        "innovation_pred_std",
        "innovation_prediction_std",
    ),
    "innovation_code_nmse": (
        "innovation_code_nmse",
        "innovation_code_loss",
    ),
    "innovation_code_cosine": (
        "innovation_code_cosine",
    ),
    "innovation_target_median_std": (
        "innovation_target_std_median",
        "innovation_target_code_std",
        "innovation_target_std",
    ),
    "innovation_effective_rank": (
        "innovation_effective_rank",
        "innovation_code_effective_rank",
    ),
    "innovation_predicted_effective_rank": (
        "innovation_predicted_effective_rank",
    ),
    "innovation_explained_fraction": (
        "innovation_explained_fraction",
    ),
    "innovation_realized_headroom_fraction": (
        "innovation_realized_headroom_fraction",
    ),
    "innovation_orthogonality": (
        "innovation_orthogonality",
        "innovation_orthogonality_error",
        "innovation_orthogonality_loss",
    ),
}

GATE_THRESHOLDS = {
    # The preceding ctx3 full-residual experiment gained only 0.00243 over its
    # last 5k steps. Require a materially larger optimization signal here.
    "final_improvement": 0.005,
    "code_nmse_max": 0.70,
    "code_cosine": 0.60,
    # With micro-batch 4 and four mode identities removed, per-horizon rank is
    # at most M*(B-1)=12. Six active directions is a non-trivial lower bound.
    "predicted_effective_rank": 6.0,
    # Random separable projection captures (M/K)*(R/D)=1/48 of isotropic local
    # error. Require learned capture beyond that null.
    "random_projection_fraction": (4.0 / 32.0) * (64.0 / 384.0),
    "excess_explained_fraction": 0.03,
    "target_median_std": 0.20,
    "realized_headroom_fraction": 0.30,
}


def _finite_number(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _first_metric(record: dict, aliases: Sequence[str]) -> Optional[float]:
    for name in aliases:
        value = _finite_number(record.get(name))
        if value is not None:
            return value
    return None


def _values(records: Iterable[dict], metric: str) -> list[float]:
    aliases = METRIC_ALIASES[metric]
    values = []
    for record in records:
        value = _first_metric(record, aliases)
        if value is not None:
            values.append(value)
    return values


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _paired_difference(
    records: Iterable[dict], left_metric: str, right_metric: str
) -> Optional[float]:
    differences = []
    for record in records:
        left = _first_metric(record, METRIC_ALIASES[left_metric])
        right = _first_metric(record, METRIC_ALIASES[right_metric])
        if left is not None and right is not None:
            differences.append(left - right)
    return _mean(differences)


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
            if not isinstance(record, dict) or _finite_number(record.get("step")) is None:
                invalid_lines += 1
                continue
            records.append(record)
    return records, invalid_lines


def _expected_steps_from_config(run_dir: Path) -> Optional[int]:
    pattern = re.compile(r"^\s*max_train_steps\s*:\s*([0-9]+)\s*(?:#.*)?$")
    for filename in ("config.full.yaml", "config.yaml"):
        path = run_dir / filename
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                match = pattern.match(line)
                if match:
                    return int(match.group(1))
    return None


def _terminal_state(run_dir: Path) -> Optional[str]:
    for state in ("complete", "failed", "stopped"):
        if (run_dir / f"STATUS.{state}").exists():
            return state
    return None


def summarize(
    run_dir: Path,
    *,
    window_steps: int = 2000,
    expected_steps: Optional[int] = None,
) -> dict:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return {
            "status": "PENDING",
            "reason": f"missing {metrics_path}",
            "run_dir": str(run_dir),
            "criteria": {},
            "missing_metrics": list(METRIC_ALIASES),
        }

    records, invalid_lines = _read_metrics(metrics_path)
    if not records:
        return {
            "status": "PENDING",
            "reason": "metrics.jsonl has no valid records",
            "run_dir": str(run_dir),
            "invalid_lines": invalid_lines,
            "criteria": {},
            "missing_metrics": list(METRIC_ALIASES),
        }

    latest_step = max(float(record["step"]) for record in records)
    window_start = max(0.0, latest_step - float(window_steps))
    window = [
        record
        for record in records
        if window_start <= float(record["step"]) <= latest_step
    ]

    metric_values = {name: _values(window, name) for name in METRIC_ALIASES}
    metrics = {
        "latent_loss_mean": _mean(metric_values["latent_loss"]),
        "latent_base_loss_mean": _mean(metric_values["latent_base_loss"]),
        "absolute_improvement_mean": _paired_difference(
            window, "latent_base_loss", "latent_loss"
        ),
        "innovation_oracle_raw_loss_mean": _mean(
            metric_values["innovation_oracle_raw_loss"]
        ),
        "oracle_improvement_mean": _paired_difference(
            window, "latent_base_loss", "innovation_oracle_raw_loss"
        ),
        # Remaining improvement available if the predictor reaches the oracle.
        "oracle_headroom_mean": _paired_difference(
            window, "latent_loss", "innovation_oracle_raw_loss"
        ),
        "innovation_target_code_std_median": _median(
            metric_values["innovation_target_code_std"]
        ),
        "innovation_target_median_std": _median(
            metric_values["innovation_target_median_std"]
        ),
        "innovation_pred_code_std_median": _median(
            metric_values["innovation_pred_code_std"]
        ),
        "innovation_code_nmse_mean": _mean(metric_values["innovation_code_nmse"]),
        "innovation_code_cosine_mean": _mean(
            metric_values["innovation_code_cosine"]
        ),
        "innovation_effective_rank_median": _median(
            metric_values["innovation_effective_rank"]
        ),
        "innovation_predicted_effective_rank_median": _median(
            metric_values["innovation_predicted_effective_rank"]
        ),
        "innovation_explained_fraction_mean": _mean(
            metric_values["innovation_explained_fraction"]
        ),
        "innovation_realized_headroom_fraction_mean": _mean(
            metric_values["innovation_realized_headroom_fraction"]
        ),
        "innovation_orthogonality_mean": _mean(
            metric_values["innovation_orthogonality"]
        ),
    }

    criteria = {
        "final_improvement": (
            None
            if metrics["absolute_improvement_mean"] is None
            else metrics["absolute_improvement_mean"]
            >= GATE_THRESHOLDS["final_improvement"]
        ),
        "code_nmse": (
            None
            if metrics["innovation_code_nmse_mean"] is None
            else metrics["innovation_code_nmse_mean"]
            <= GATE_THRESHOLDS["code_nmse_max"]
        ),
        "code_cosine": (
            None
            if metrics["innovation_code_cosine_mean"] is None
            else metrics["innovation_code_cosine_mean"]
            >= GATE_THRESHOLDS["code_cosine"]
        ),
        "predicted_effective_rank": (
            None
            if metrics["innovation_predicted_effective_rank_median"] is None
            else metrics["innovation_predicted_effective_rank_median"]
            >= GATE_THRESHOLDS["predicted_effective_rank"]
        ),
        "excess_explained_fraction": (
            None
            if metrics["innovation_explained_fraction_mean"] is None
            else metrics["innovation_explained_fraction_mean"]
            - GATE_THRESHOLDS["random_projection_fraction"]
            >= GATE_THRESHOLDS["excess_explained_fraction"]
        ),
        "target_median_std": (
            None
            if metrics["innovation_target_median_std"] is None
            else metrics["innovation_target_median_std"]
            >= GATE_THRESHOLDS["target_median_std"]
        ),
        "realized_headroom_fraction": (
            None
            if metrics["innovation_realized_headroom_fraction_mean"] is None
            else metrics["innovation_realized_headroom_fraction_mean"]
            >= GATE_THRESHOLDS["realized_headroom_fraction"]
        ),
    }
    missing_metrics = [
        name for name, values in metric_values.items() if not values
    ]
    missing_gate_metrics = [name for name, passed in criteria.items() if passed is None]

    configured_steps = _expected_steps_from_config(run_dir)
    expected_steps = (
        int(expected_steps)
        if expected_steps is not None
        else configured_steps if configured_steps is not None else 10000
    )
    terminal_state = _terminal_state(run_dir)
    finished = (
        terminal_state == "complete"
        or latest_step >= expected_steps
        or (run_dir / "final_model" / "pytorch_model.pt").is_file()
    )
    terminal_failure = terminal_state in {"failed", "stopped"}

    if missing_gate_metrics:
        status = "PENDING"
        reason = "required gate metrics have not been logged yet"
    elif not finished and not terminal_failure:
        status = "PENDING"
        reason = f"training has reached step {latest_step:g}/{expected_steps}"
    elif all(criteria.values()) and not terminal_failure:
        status = "PASS"
        reason = "all training-feasibility criteria passed"
    else:
        status = "FAIL"
        failed = [name for name, passed in criteria.items() if passed is False]
        reason = "failed criteria: " + ", ".join(failed)
        if terminal_failure:
            reason = f"run status is {terminal_state}; {reason}"

    return {
        "status": status,
        "reason": reason,
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "latest_step": latest_step,
        "expected_steps": expected_steps,
        "window_start_step": window_start,
        "window_steps": window_steps,
        "window_records": len(window),
        "window_unique_steps": len({float(record["step"]) for record in window}),
        "invalid_lines": invalid_lines,
        "terminal_state": terminal_state,
        "finished": finished,
        "metrics": metrics,
        "thresholds": GATE_THRESHOLDS,
        "criteria": criteria,
        "missing_metrics": missing_metrics,
        "missing_gate_metrics": missing_gate_metrics,
    }


def _format_value(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def print_human(summary: dict) -> None:
    print(f"Predictable innovation gate: {summary['status']}")
    print("Scope: training feasibility only; held-out validation is still required")
    print(f"Reason: {summary['reason']}")
    print(f"Run: {summary['run_dir']}")
    if "latest_step" not in summary:
        return
    print(
        "Progress: "
        f"{summary['latest_step']:g}/{summary['expected_steps']} "
        f"(finished={str(summary['finished']).lower()})"
    )
    print(
        "Window: "
        f"[{summary['window_start_step']:g}, {summary['latest_step']:g}], "
        f"{summary['window_records']} records / "
        f"{summary['window_unique_steps']} unique steps"
    )
    metrics = summary["metrics"]
    rows = (
        ("latent_loss", metrics["latent_loss_mean"]),
        ("latent_base_loss", metrics["latent_base_loss_mean"]),
        ("absolute improvement (base-final)", metrics["absolute_improvement_mean"]),
        ("innovation_oracle_raw_loss", metrics["innovation_oracle_raw_loss_mean"]),
        ("oracle improvement (base-oracle)", metrics["oracle_improvement_mean"]),
        ("oracle headroom (final-oracle)", metrics["oracle_headroom_mean"]),
        ("target code std (median)", metrics["innovation_target_code_std_median"]),
        ("target per-dimension std median", metrics["innovation_target_median_std"]),
        ("pred code std (median)", metrics["innovation_pred_code_std_median"]),
        ("code NMSE", metrics["innovation_code_nmse_mean"]),
        ("code cosine", metrics["innovation_code_cosine_mean"]),
        ("effective rank (median)", metrics["innovation_effective_rank_median"]),
        (
            "pred effective rank (median)",
            metrics["innovation_predicted_effective_rank_median"],
        ),
        (
            "explained error fraction",
            metrics["innovation_explained_fraction_mean"],
        ),
        (
            "realized oracle headroom fraction",
            metrics["innovation_realized_headroom_fraction_mean"],
        ),
        ("orthogonality (mean)", metrics["innovation_orthogonality_mean"]),
    )
    for label, value in rows:
        print(f"  {label}: {_format_value(value)}")

    print("Criteria:")
    labels = {
        "final_improvement": "final improvement >= 0.005",
        "code_nmse": "code NMSE <= 0.70",
        "code_cosine": "code cosine >= 0.60",
        "predicted_effective_rank": "mode-centered predicted effective rank >= 6",
        "excess_explained_fraction": (
            "explained fraction >= random (4/32)*(64/384) baseline + 0.03"
        ),
        "target_median_std": "target median std >= 0.20",
        "realized_headroom_fraction": "realized oracle headroom >= 0.30",
    }
    for name, label in labels.items():
        passed = summary["criteria"].get(name)
        result = "PENDING" if passed is None else "PASS" if passed else "FAIL"
        print(f"  {result}: {label}")
    if summary.get("missing_metrics"):
        print("Missing optional/required metrics: " + ", ".join(summary["missing_metrics"]))
    if summary.get("invalid_lines"):
        print(f"Ignored malformed records: {summary['invalid_lines']}")


def _write_fixture(run_dir: Path, records: list[dict], expected_steps: int = 10000) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.full.yaml").write_text(
        f"trainer:\n  max_train_steps: {expected_steps}\n", encoding="utf-8"
    )
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def self_test() -> None:
    common = {
        "latent_loss": 0.47,
        "latent_base_loss": 0.50,
        "innovation_oracle_raw_loss": 0.44,
        "innovation_target_code_std": 0.30,
        "innovation_pred_code_std": 0.27,
        "innovation_code_nmse": 0.35,
        "innovation_code_cosine": 0.82,
        "innovation_effective_rank": 9.0,
        "innovation_predicted_effective_rank": 8.0,
        "innovation_explained_fraction": 0.08,
        "innovation_realized_headroom_fraction": 0.50,
        "innovation_orthogonality_error": 0.02,
    }
    with tempfile.TemporaryDirectory(prefix="predictable_innovation_gate_") as root:
        root_path = Path(root)

        passing = root_path / "passing"
        _write_fixture(
            passing,
            [{"step": step, **common} for step in (7900, 8000, 9000, 10000)],
        )
        assert summarize(passing)["status"] == "PASS"

        incomplete = root_path / "incomplete"
        _write_fixture(
            incomplete,
            [{"step": step, **common} for step in (7000, 8000, 9000)],
        )
        assert summarize(incomplete)["status"] == "PENDING"

        failing = root_path / "failing"
        failed_record = {
            **common,
            "latent_loss": 0.497,
            "innovation_oracle_raw_loss": 0.495,
            "innovation_target_code_std": 0.10,
            "innovation_code_nmse": 0.95,
            "innovation_code_cosine": 0.10,
            "innovation_effective_rank": 20.0,
            "innovation_predicted_effective_rank": 3.0,
            "innovation_explained_fraction": 0.02,
            "innovation_realized_headroom_fraction": 0.05,
        }
        _write_fixture(failing, [{"step": 10000, **failed_record}])
        assert summarize(failing)["status"] == "FAIL"

        missing = root_path / "missing"
        _write_fixture(missing, [{"step": 10000, "latent_loss": 0.47}])
        assert summarize(missing)["status"] == "PENDING"

    print("fixture smoke test: PASS (PASS/PENDING/FAIL/missing-field cases)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--window-steps", type=int, default=2000)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.run_dir is None:
        parser.error("run_dir is required unless --self-test is used")
    if args.window_steps <= 0:
        parser.error("--window-steps must be positive")
    if args.expected_steps is not None and args.expected_steps <= 0:
        parser.error("--expected-steps must be positive")

    summary = summarize(
        args.run_dir,
        window_steps=args.window_steps,
        expected_steps=args.expected_steps,
    )
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human(summary)
    return {"PASS": 0, "FAIL": 1, "PENDING": 2}[summary["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
