"""Retired experiment configs must fail before constructing a different policy."""
from unittest.mock import patch

import pytest

from starVLA.model.framework.WM4A.GAWM import GAWM
from starVLA.model.framework.WM4A.GAWMCompactExpert import GAWMCompactExpert
from starVLA.model.framework.WM4A.GAWMObjectFusion import GAWMObjectFusion
from test_gawm_l_vision import backbone, tiny_config


@pytest.mark.parametrize('branch', ['spatial_focus', 'contact_objective'])
def test_retired_branch_rejects_config_before_loading_encoder(branch):
    cfg, _ = tiny_config(2)
    cfg.framework[branch] = {'enabled': True}
    with patch('starVLA.model.framework.WM4A.GAWM.get_world_model') as encoder:
        with pytest.raises(ValueError, match=f'{branch} has been retired'):
            GAWM(cfg)
        encoder.assert_not_called()


@pytest.mark.parametrize('framework', [GAWMCompactExpert, GAWMObjectFusion])
def test_retired_framework_has_explicit_migration_error(framework):
    with pytest.raises(ValueError, match='archived source_snapshot'):
        framework(None)


@pytest.mark.parametrize('option,value', [
    ('freeze_visual_token_pooler', True),
    ('condition_world_model_on_state', True),
    ('state_conditioning_mode', 'residual'),
])
def test_retired_latent_options_do_not_silently_change_the_experiment(option, value):
    cfg, _ = tiny_config(2)
    cfg.framework.world_model[option] = value
    with patch('starVLA.model.framework.WM4A.GAWM.get_world_model') as encoder:
        with pytest.raises(ValueError, match=f'{option} has been retired'):
            GAWM(cfg)
        encoder.assert_not_called()


@pytest.mark.parametrize('options', [{}, {'wm_state': True}])
def test_fixed_teacher_accepts_old_configs_without_building_legacy_model(options):
    cfg, _ = tiny_config(2)
    cfg.framework.compact_study = options
    cfg.framework.world_model.update(dict(
        future_objective='fixed_dino_patches', detach_wm_input=False,
        latent_cosine_weight=0., gawm_l_bridge_norm='fixed_layernorm',
        feature_decoder_dim=8, feature_decoder_depth=1, feature_decoder_heads=2,
        sync_latent_stats=True, latent_stats_momentum=.9))

    def encoder(**kwargs):
        model = backbone()
        model.encoder.config.patch_size = 8
        return model

    with (patch('starVLA.model.framework.WM4A.GAWM.get_world_model', side_effect=encoder),
          patch('starVLA.model.framework.WM4A.GAWM.VisualTokenLatentWorldModel') as legacy):
        model = GAWM(cfg)
        legacy.assert_not_called()
        assert 'sync_latent_stats' not in model.config.framework.world_model
        assert 'latent_stats_momentum' not in model.config.framework.world_model
        restored = GAWM(model.config)
        restored.load_state_dict(model.state_dict(), strict=True)
