"""Measure learned branch contribution with fixed held-out inputs and weights."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from examples.Robotwin.audits.spatial_ablation import apply_spatial_ablation
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def wait_checkpoint(run, step, process_id):
    import psutil
    process = psutil.Process(process_id) if process_id else None
    while True:
        checkpoint = run / "checkpoints" / f"steps_{step}_pytorch_model.pt"
        summary = run / "summary.jsonl"
        # The summary record is appended AFTER torch.save finishes.
        if summary.exists() and any(json.loads(row)["steps"] == step for row in summary.read_text().splitlines()):
            if not checkpoint.is_file():
                raise RuntimeError("Checkpoint summary exists but checkpoint is missing")
            return checkpoint
        if process is None or not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError(f"Checkpoint {step} is not ready and training handle is not live")
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--step", type=int)
    parser.add_argument("--training-pid", type=int)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite analysis")
    if args.samples < 1:
        raise ValueError("samples must be positive")
    checkpoint = args.checkpoint or wait_checkpoint(args.run, args.step, args.training_pid)
    cfg = OmegaConf.load(args.run / "config.full.yaml")
    cfg.datasets.vla_data.episode_split = "validation"
    if args.split_manifest:
        cfg.datasets.vla_data.episode_split_manifest = str(args.split_manifest.resolve())
    cfg.datasets.vla_data.normalization_statistics_path = str(args.run / "dataset_statistics.json")
    data = get_vla_dataset(cfg.datasets.vla_data, mode="validation")
    for child in data.datasets:
        child.transforms.eval()
        expected = (set(json.loads(args.split_manifest.read_text())["validation_episode_ids"])
                    if args.split_manifest else set(range(0, 1000, 20)))
        assert set(child.trajectory_ids) == expected
    torch.manual_seed(42)
    model = build_framework(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.cuda().eval()
    focus = model.spatial_focus
    saved_gates = focus.gates.detach().clone() if focus is not None else torch.empty(0)
    original_refine = focus.refine_queries if focus is not None else None
    full_memories = {'dense', 'local', 'goal', 'objects'}
    modes = dict(full=full_memories)
    if focus is not None:
        modes.update(no_dense=full_memories - {'dense'}, no_focus=set())
    if focus is not None and model.focus_cfg.get("use_local", False):
        modes["no_local"] = full_memories - {'local'}
    if focus is not None and focus.goal_readout is not None:
        modes['no_goal'] = full_memories - {'goal'}
    if focus is not None and focus.object_readout is not None:
        modes['no_objects'] = full_memories - {'objects'}
    predictions = {mode: [] for mode in modes}
    targets, masks, pixel_errors, per_view_errors, target_in_crop = [], [], [], [], []
    with torch.inference_mode():
        try:
            for start in range(0, args.samples, 4):
                examples = [data[(0, i)] for i in range(start, min(start + 4, args.samples))]
                inputs = [{k: v for k, v in ex.items() if not k.startswith("spatial_")
                           and "future" not in k and k not in ("action", "action_valid_mask")} for ex in examples]
                targets.extend(np.asarray(ex["action"]) for ex in examples)
                masks.extend(np.asarray(ex["action_valid_mask"], dtype=bool) for ex in examples)
                for mode in modes:
                    if focus is not None:
                        focus.refine_queries = original_refine
                        apply_spatial_ablation(model, mode)
                    response = model.predict_action(inputs)
                    predictions[mode].extend(response["normalized_actions"])
                    if mode == "full" and "spatial_predicted_xy" in response:
                        xy = np.asarray(response["spatial_predicted_xy"])
                        label = np.asarray([ex["spatial_target_xy"] for ex in examples])
                        valid = np.asarray([ex["spatial_target_valid"] for ex in examples], dtype=bool)
                        valid &= np.asarray([ex["view_valid_mask"] for ex in examples], dtype=bool)
                        distance = np.linalg.norm((xy - label) * [320, 240], axis=-1)
                        pixel_errors.extend(distance[valid].tolist())
                        per_view_errors.extend(np.where(valid, distance, np.nan).tolist())
                        fraction = float(model.focus_cfg.crop_fraction)
                        left = np.clip(xy - fraction / 2, 0, 1 - fraction)
                        inside = ((label >= left) & (label <= left + fraction)).all(-1)
                        target_in_crop.extend(inside[valid].tolist())
        finally:
            if focus is not None:
                focus.gates.copy_(saved_gates)
                focus.refine_queries = original_refine
                apply_spatial_ablation(model, 'full')
    target, mask = np.asarray(targets), np.asarray(masks)
    assert mask.any()
    full = np.asarray(predictions["full"])
    scores = {}
    for mode, values in predictions.items():
        pred = np.asarray(values)
        assert np.isfinite(pred).all()
        error = np.abs(pred - target)[mask]
        change = np.abs(pred - full)[mask]
        scores[mode] = dict(action_l1=float(error.mean()), joint_l1=float(error[:, :12].mean()),
                            gripper_l1=float(error[:, 12:].mean()), per_dimension_l1=error.mean(0).tolist(),
                            mean_change_from_full=float(change.mean()), max_change_from_full=float(change.max()),
                            per_dimension_change_from_full=change.mean(0).tolist())
    report = dict(checkpoint=str(checkpoint.resolve()), samples=args.samples,
                  split=str(args.split_manifest.resolve()) if args.split_manifest else "episode % 20 == 0; uniform validation, no event bias",
                  validation_episode_ids=sorted(int(i) for i in data.datasets[0].trajectory_ids),
                  gates_tanh=saved_gates.tanh().cpu().tolist(),
                  residual_mode=focus.residual_mode if focus is not None else 'none',
                  residual_scales=focus.residual_scales().detach().cpu().tolist() if focus is not None else [],
                  scores=scores, valid_action_steps=int(mask.sum()),
                  note="Inference only, all localization labels removed from model inputs. Gate interventions change no weights on disk. Offline action/waypoint metrics do not establish grasp or task success.")
    if pixel_errors:
        errors = np.asarray(per_view_errors)
        report["localization"] = dict(valid_points=len(pixel_errors), mean_pixel_error=float(np.mean(pixel_errors)),
            median_pixel_error=float(np.median(pixel_errors)), p90_pixel_error=float(np.percentile(pixel_errors, 90)),
            per_view=[dict(valid=int(np.isfinite(errors[:, i]).sum()),
                           mean_pixel_error=float(np.nanmean(errors[:, i])) if np.isfinite(errors[:, i]).any() else None) for i in range(3)],
            target_inside_predicted_crop_fraction=float(np.mean(target_in_crop)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
