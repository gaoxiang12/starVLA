import numpy as np
import pytest
import torch

from starVLA.model.modules.action_model.OFTSpatialActionHead import OFTSpatialActionHead
from starVLA.model.modules.action_model.action_loss import masked_action_l1_loss
from examples.Robotwin.audits.prepare_oft_grasp_training_20260910 import first_grasp_window


def setup():
    torch.manual_seed(73)
    head = OFTSpatialActionHead(patch_dim=24, task_dim=16, hidden_dim=32,
                               grid_size=2, horizon=4, depth=2, heads=4)
    return head, torch.randn(2, 3, 4, 24), torch.randn(2, 16)


def test_invalid_views_and_padding_do_not_change_valid_predictions():
    head, patches, task = setup()
    vv = torch.tensor([[True, True, False], [True, False, True]])
    av = torch.tensor([[True, True, False, False], [True, True, True, False]])
    reference = head(patches, task, vv, av)
    changed = patches.clone()
    changed[~vv] = float('nan')
    torch.testing.assert_close(head(changed, task, vv, av), reference, atol=0, rtol=0)
    assert torch.count_nonzero(reference[~av]) == 0
    with pytest.raises(ValueError, match='valid view'):
        head(patches, task, torch.zeros_like(vv))


def test_action_gradient_reaches_valid_spatial_features_and_task():
    head, patches, task = setup()
    patches.requires_grad_(); task.requires_grad_()
    vv = torch.tensor([[True, True, False], [True, False, True]])
    av = torch.tensor([[True, True, False, False], [True, True, True, False]])
    loss = masked_action_l1_loss(head(patches, task, vv, av), torch.randn(2, 4, 14), av)
    loss.backward()
    assert patches.grad.abs().sum() > 0 and task.grad.abs().sum() > 0
    assert torch.count_nonzero(patches.grad[~vv]) == 0
    assert torch.isfinite(patches.grad).all()
    assert head.fusion.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0


def test_action_placeholders_are_causal_and_input_prefix_is_protected():
    head, patches, task = setup()
    head.eval()
    baseline = head(patches, task)
    with torch.no_grad():
        head.action_queries.weight[-1].add_(torch.randn_like(head.action_queries.weight[-1])*10)
    changed = head(patches, task)
    torch.testing.assert_close(baseline[:, :-1], changed[:, :-1], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(baseline[:, -1], changed[:, -1])


def test_first_grasp_window_adds_closed_observations_without_second_grasp():
    grip = np.r_[np.ones(32), np.zeros(48), np.ones(20), np.zeros(40)]
    end, anchors = first_grasp_window(grip, 32)
    assert end == 80 and anchors == 64
    assert anchors-1+16 == end-1
    assert np.any(grip[:anchors] < .2)
    assert not np.any(grip[1:32+1-16] < .2)
    with pytest.raises(ValueError, match='No first reopen'):
        first_grasp_window(np.r_[np.ones(32), np.zeros(48)], 32)
    with pytest.raises(ValueError, match='boundary mismatch'):
        first_grasp_window(grip, 33)
