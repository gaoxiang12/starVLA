import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.run_gawm_compact_study import ARMS, GPU_MAP, STEPS, idle_devices, make_config
from starVLA.training.recipe import apply_training_recipe, resolve_training_budget


@pytest.mark.parametrize("name,spec,changes", [x for x in ARMS if x[0] != "attention_profile"])
def test_full_budget_and_small_backbone_contract(tmp_path, name, spec, changes):
    cfg = make_config(tmp_path, name, spec, changes)
    cfg = apply_training_recipe(cfg)
    resolve_training_budget(cfg, range(cfg.datasets.vla_data.expected_frames), 128)
    assert cfg.trainer.max_train_steps == STEPS
    assert list(cfg.trainer.stage_epochs) == [12, 4]
    assert cfg.trainer.expected_global_batch_size == 128
    assert cfg.datasets.vla_data.per_device_batch_size == 4
    assert sum(len(v.split(',')) for v in GPU_MAP.values()) == 32
    assert cfg.framework.world_model.train_encoder is False
    assert cfg.framework.world_model.future_objective == 'fixed_dino_patches'
    assert cfg.framework.lang_cond.type == 'vtt'
    assert cfg.framework.world_model.encoder_spec in ('vits16', 'vits16plus')
    assert list(cfg.framework.world_model.feat_layers) == [-6,-4,-2]
    assert spec in cfg.framework.lang_cond.task_vectors_path
    if name == 'adapter384x4':
        assert cfg.framework.world_model.lila_adapter_dim == 384
        assert cfg.framework.world_model.lila_adapter_heads == 8


def test_resource_detection_does_not_allow_low_memory_active_work():
    output = '0, GPU-a, 0, 100\n1, GPU-b, 1, 100\n2, GPU-c, 40, 80\n3, GPU-d, 900, 0\n4, GPU-e, 48, 0'
    assert idle_devices(output) == [(0,'GPU-a'),(1,'GPU-b'),(4,'GPU-e')]


def test_history_conditioning_is_declared_and_does_not_change_action_contract(tmp_path):
    cfg = make_config(tmp_path,'history2','vits16plus',{'history_frames':2})
    assert cfg.datasets.vla_data.history_recorded_offset == 16
    assert cfg.framework.compact_study.history_frames == 2
    assert cfg.framework.action_model.action_horizon == 32
    assert list(cfg.datasets.vla_data.future_recorded_offsets) == [0,16,32]
