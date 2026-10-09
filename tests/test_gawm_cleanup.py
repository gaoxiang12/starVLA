"""Retired experiment configs must fail before constructing a different policy."""
from unittest.mock import patch

import pytest

from starVLA.model.framework.WM4A.GAWM import GAWM
from starVLA.model.framework.WM4A.GAWMCompactExpert import GAWMCompactExpert
from starVLA.model.framework.WM4A.GAWMObjectFusion import GAWMObjectFusion
from test_gawm_lila_vision import tiny_config


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
