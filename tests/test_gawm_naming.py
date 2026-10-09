"""Old checkpoints and pinned training sources survive the GAWM-L rename."""
import copy
from unittest.mock import patch

from omegaconf import OmegaConf
import pytest
import torch

from starVLA.model.framework.WM4A.GAWM import GAWM
from starVLA.model.gawm_config import config_for_gawm_source, migrate_gawm_config
from test_gawm_l_vision import backbone, tiny_config


def legacy_config():
    cfg, _ = tiny_config(2)
    cfg.framework.name = 'GAWM'
    wm = cfg.framework.world_model
    wm.visual_frontend = 'lila'
    wm.lila_image_size = wm.pop('gawm_l_image_size')
    wm.lila_adapter_dim = wm.pop('gawm_l_adapter_dim')
    wm.lila_adapter_depth = wm.pop('gawm_l_adapter_depth')
    wm.lila_adapter_heads = wm.pop('gawm_l_adapter_heads')
    if 'gawm_l_bridge_norm' in wm:
        wm.lila_bridge_norm = wm.pop('gawm_l_bridge_norm')
    return cfg


def test_legacy_config_preserves_weights_and_saved_config_uses_canonical_names():
    old = legacy_config()
    canonical, _ = tiny_config(2)
    with patch('starVLA.model.framework.WM4A.GAWM.get_world_model', side_effect=lambda **kw: backbone()):
        torch.manual_seed(7)
        before = GAWM(old).eval()
        torch.manual_seed(7)
        after = GAWM(canonical).eval()
        assert before.config.framework.name == 'GAWM-L'
        assert before.config.framework.world_model.visual_frontend == 'gawm_l'
        assert not any(key.startswith('lila_') for key in before.config.framework.world_model)
        assert before.state_dict().keys() == after.state_dict().keys()
        for key, value in before.state_dict().items():
            torch.testing.assert_close(value, after.state_dict()[key], rtol=0, atol=0)
        restored = GAWM(OmegaConf.create(OmegaConf.to_yaml(before.config)))
        restored.load_state_dict(before.state_dict(), strict=True)


def test_conflicting_old_and_current_keys_are_rejected():
    cfg = legacy_config()
    cfg.framework.world_model.gawm_l_adapter_dim = 999
    with pytest.raises(ValueError, match='Conflicting legacy/current'):
        migrate_gawm_config(cfg)


def test_export_for_archived_source_round_trips_without_mutating_input(tmp_path):
    original = legacy_config()
    canonical = migrate_gawm_config(copy.deepcopy(original))
    saved = copy.deepcopy(canonical)
    module = tmp_path / 'starVLA/model/modules/gawm_lila_vision.py'
    module.parent.mkdir(parents=True)
    module.touch()
    exported = config_for_gawm_source(canonical, tmp_path)
    assert exported == original
    assert canonical == saved
    assert migrate_gawm_config(exported) == canonical
    module.rename(module.with_name('gawm_l_vision.py'))
    assert config_for_gawm_source(canonical, tmp_path) == canonical


def test_small_backbones_and_historical_reference_keep_their_identity():
    cfg = legacy_config()
    cfg.framework.world_model.encoder_spec = 'vits16'
    assert migrate_gawm_config(cfg).framework.name == 'GAWM'
    reference = OmegaConf.create({'framework': {'name': 'LiLaWAMTrain', 'lila': {'image_size': [224, 224]}}})
    saved = copy.deepcopy(reference)
    assert migrate_gawm_config(reference) == saved
