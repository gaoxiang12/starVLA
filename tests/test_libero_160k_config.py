"""Check the effective LIBERO budget and the cluster's batch arithmetic."""
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from scripts.run_gawm_c_cluster import validate_cluster_batch
from starVLA.training.recipe import resolve_training_budget, prepare_parameter_precision, c_lr_multiplier

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return OmegaConf.load(ROOT / "examples/LIBERO/train_files" / name)


def test_160k_profile_uses_40_gpu_batch160_and_full_budget():
    cfg = validate_cluster_batch(load("starvla_gawm_c_160k.yaml"), 40)
    assert cfg.datasets.vla_data.per_device_batch_size == 4
    assert cfg.trainer.gradient_accumulation_steps == 1
    assert cfg.trainer.expected_global_batch_size == 160
    resolve_training_budget(cfg, range(872087), 160)
    assert cfg.trainer.steps_per_epoch == 5450
    assert cfg.trainer.stage1_steps == 65400
    assert cfg.trainer.max_train_steps == 160000
    assert cfg.trainer.lr_scheduler_total_steps == 218000
    assert 160000 - cfg.trainer.stage1_steps < cfg.trainer.lr_scheduler_total_steps
    assert cfg.trainer.is_resume is False
    assert cfg.trainer.pretrained_checkpoint is None
    multiplier = c_lr_multiplier(160000, stage1_steps=65400, period=218000,
                                 minimum_ratio=.25, stage2_scale=.2)
    assert .05 < multiplier < .2


def test_original_40_gpu_profile_still_uses_batch640():
    cfg = validate_cluster_batch(load("starvla_gawm_c_12plus4.yaml"), 40)
    assert cfg.datasets.vla_data.per_device_batch_size == 16
    resolve_training_budget(cfg, range(872087), 640)
    assert cfg.trainer.max_train_steps == 21792


def test_cluster_accounts_for_gradient_accumulation():
    cfg = load("starvla_gawm_c_160k.yaml")
    cfg.training_overrides.datasets.vla_data.per_device_batch_size = 2
    cfg.training_overrides.trainer.gradient_accumulation_steps = 2
    resolved = validate_cluster_batch(cfg, 40)
    assert resolved.trainer.expected_global_batch_size == 160


def test_different_gpu_count_requires_explicit_batch_adjustment():
    with pytest.raises(ValueError, match="32 GPUs x 4 samples x 1 accumulation = 128"):
        validate_cluster_batch(load("starvla_gawm_c_160k.yaml"), 32)


def test_160k_trainable_parameters_and_adam_states_remain_fp32():
    cfg = validate_cluster_batch(load("starvla_gawm_c_160k.yaml"), 40)
    model = torch.nn.Linear(2, 1).to(torch.bfloat16)
    model.register_buffer("scale", torch.ones(1, dtype=torch.float32))
    prepare_parameter_precision(model, cfg)
    assert cfg.trainer.mixed_precision == "bf16"
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.trainer.learning_rate.base)
    model(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    assert model.scale.dtype == torch.float32
    for parameter in model.parameters():
        assert parameter.dtype == torch.float32
        assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
        assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32
