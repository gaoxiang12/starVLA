import json

import torch

from starVLA.model.modules.robotwin_contact_objective import RoboTwinContactObjective


def objective(tmp_path):
    path = tmp_path/'robot.urdf'
    parts = ['<robot name="test"><link name="base"/>']
    for side in ('fl', 'fr'):
        for index in range(1, 7):
            parent = 'base' if index == 1 else f'{side}_link{index-1}'
            parts.append(f'<link name="{side}_link{index}"/><joint name="{side}_joint{index}" type="revolute">'
                         f'<parent link="{parent}"/><child link="{side}_link{index}"/>'
                         '<origin xyz=".05 0 .01"/><axis xyz="0 0 1"/></joint>')
    path.write_text(''.join(parts)+'</robot>')
    stats = tmp_path/'stats.json'
    stats.write_text(json.dumps({'aloha': {'action': {'q01': [-2.]*12+[0.,0.], 'q99': [2.]*12+[1.,1.]}}}))
    return RoboTwinContactObjective(path, stats)


def test_padded_nan_targets_do_not_change_loss_or_gradients(tmp_path):
    loss_fn = objective(tmp_path)
    pred = torch.full((1, 3, 14), float('nan'))
    truth = pred.clone()
    pred[:, 0] = 0.
    pred[:, 0, 0] = .01
    truth[:, 0] = 0.
    pred.requires_grad_()
    state = torch.zeros(1, 14)
    state[:, 12:] = 1.
    mask = torch.tensor([[True, False, False]])
    loss, _ = loss_fn(pred, truth, state, mask)
    reference, _ = loss_fn(pred[:, :1], truth[:, :1], state)
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[:, 0, :6].abs().sum() > 0
    assert pred.grad[:, 1:].abs().sum() == 0


def test_all_padding_is_zero_and_geometry_survives_bfloat16_cast(tmp_path):
    loss_fn = objective(tmp_path)
    loss_fn.bfloat16()
    assert loss_fn._chains[0].origins.dtype == torch.float64
    assert not loss_fn.state_dict()  # No learned parameters or lossy BF16 geometry buffers.
    pred = torch.full((1, 2, 14), float('nan'), requires_grad=True)
    loss, metrics = loss_fn(pred, pred.detach(), torch.ones(1, 14), torch.zeros(1, 2, dtype=torch.bool))
    assert loss.item() == 0
    assert loss_fn._chains[0].origins.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() == 0
    assert metrics['contact_fraction'].item() == 0


def test_closing_mask_comes_from_targets_not_predicted_gripper(tmp_path):
    loss_fn = objective(tmp_path)
    state = torch.zeros(1, 14)
    state[:, 12:] = 1.
    target = state[:, None].expand(1, 8, 14).clone()
    target[:, 4:, 12] = 0.
    pred = target.clone()
    pred[:, :, 12] = 1.
    _, wrong = loss_fn(pred, target, state)
    _, correct = loss_fn(target, target, state)
    torch.testing.assert_close(wrong['contact_fraction'], correct['contact_fraction'])
    assert wrong['gripper_transition_l1'] > 0
    assert correct['contact_objective_loss'] == 0
