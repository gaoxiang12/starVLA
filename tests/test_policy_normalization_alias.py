import pytest

from deployment.model_server.policy_norm_processor import _resolve_robot_type, ROBOT_TYPE_CONFIG_MAP


CONFIG = {'datasets':{'vla_data':{'data_mix':'robotwin_rgb_grasp_precision'}}}


def test_explicit_identical_alias_resolves_shared_aloha_statistics():
    assert _resolve_robot_type(CONFIG,'aloha') == 'robotwin_continuous_next_wm'


def test_same_embodiment_without_alias_remains_ambiguous(monkeypatch):
    config = ROBOT_TYPE_CONFIG_MAP['robotwin_pregrasp_correction_wm']
    monkeypatch.setattr(config,'normalization_robot_type','robotwin_pregrasp_correction_wm')
    with pytest.raises(ValueError,match='matches multiple'):
        _resolve_robot_type(CONFIG,'aloha')


@pytest.mark.parametrize('field,value', [
    ('action_spec_id','different_action_semantics'),
    ('control_hz',30),
    ('action_indices',list(range(16))),
    ('action_keys',['action.right_joints','action.left_joints','action.left_gripper','action.right_gripper']),
])
def test_alias_rejects_different_units_timing_or_order(monkeypatch,field,value):
    config = ROBOT_TYPE_CONFIG_MAP['robotwin_pregrasp_correction_wm']
    monkeypatch.setattr(config,field,value)
    with pytest.raises(ValueError,match='incompatible semantics'):
        _resolve_robot_type(CONFIG,'aloha')
