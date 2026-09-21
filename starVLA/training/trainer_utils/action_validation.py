"""Deterministic per-task action validation on disjoint episodes."""

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.modules.action_model.action_loss import (
    action_l1_diagnostics,
    masked_action_l1_loss,
)
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig


def evaluate_action_batch(model, examples, gripper_indices=()):
    """Disable dropout temporarily and exclude padded targets from all scores."""
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            response = model.predict_action(examples=examples)
            predictions = response["normalized_actions"]
    finally:
        model.train(was_training)
    pred = torch.as_tensor(np.asarray(predictions), dtype=torch.float32)
    target = torch.as_tensor(np.asarray([x["action"] for x in examples]), dtype=torch.float32)
    masks = [x.get("action_valid_mask", np.ones(target.shape[1], dtype=bool)) for x in examples]
    mask = torch.as_tensor(np.asarray(masks), dtype=torch.bool)
    metrics = action_l1_diagnostics(pred, target, mask, gripper_indices=gripper_indices)
    metrics["l1_action_loss"] = masked_action_l1_loss(pred, target, mask)
    metrics["mse_score"] = ((pred - target).square() * mask[..., None]).sum() / (mask.sum() * pred.shape[-1])
    if "spatial_predicted_xy" in response and "spatial_target_xy" in examples[0]:
        xy = np.asarray(response["spatial_predicted_xy"])
        expected = np.asarray([x["spatial_target_xy"] for x in examples])
        valid = np.asarray([x["spatial_target_valid"] for x in examples],dtype=bool)
        distance = np.linalg.norm((xy-expected)*np.array([320.,240.]),axis=-1)
        metrics["spatial_pixel_error_sum"] = float(distance[valid].sum())
        metrics["spatial_valid_count"] = int(valid.sum())
    return {name: float(value) for name, value in metrics.items()}


class HeldOutActionEvaluator:
    def __init__(self, config):
        data_cfg = config.datasets.vla_data
        if isinstance(data_cfg, AccessTrackedConfig):
            data_cfg = data_cfg.unwrap()
        cfg = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
        if not int(cfg.get("validation_episode_stride", 0)):
            raise ValueError("Held-out validation requires validation_episode_stride")
        cfg.episode_split = "validation"
        # Reuse exactly the training normalizers, including any pinned scales.
        cfg.normalization_statistics_path = str(Path(config.output_dir) / "dataset_statistics.json")
        self.dataset = get_vla_dataset(data_cfg=cfg, mode="validation")
        for child in self.dataset.datasets:
            child.transforms.eval()
        self.samples_per_task = int(config.trainer.get("validation_samples_per_task", 16))
        self.batch_size = int(config.trainer.get("validation_batch_size", 4))
        if min(self.samples_per_task, self.batch_size) < 1:
            raise ValueError("Validation sample and batch sizes must be positive")
        self.output_path = Path(config.output_dir) / "validation_per_task.jsonl"

    def evaluate(self, model, step):
        task_scores = []
        for dataset_index, child in enumerate(self.dataset.datasets):
            batches = []
            for start in range(0, self.samples_per_task, self.batch_size):
                count = min(self.batch_size, self.samples_per_task - start)
                examples = [self.dataset[(dataset_index, i)] for i in range(start, start + count)]
                spec = model.embodiment_head_specs[child.tag]
                scores = evaluate_action_batch(model, examples, spec.get("gripper_indices", ()))
                batches.append((count, scores))
            scores = {
                key: sum(count * score[key] for count, score in batches) / self.samples_per_task
                for key in batches[0][1]
            }
            if "spatial_valid_count" in scores:
                count = sum(item[1]["spatial_valid_count"] for item in batches)
                error = sum(item[1]["spatial_pixel_error_sum"] for item in batches)
                scores.pop("spatial_pixel_error_sum")
                scores["spatial_valid_count"] = count
                scores["spatial_pixel_error"] = error/max(count,1)
            task_scores.append(scores)
            with self.output_path.open("a") as handle:
                handle.write(json.dumps({"step": step, "task": child.dataset_name, "samples": self.samples_per_task, **scores}) + "\n")
        return {
            "validation/" + key: float(np.mean([score[key] for score in task_scores]))
            for key in task_scores[0]
        }
