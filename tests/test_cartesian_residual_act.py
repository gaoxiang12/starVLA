import json
from pathlib import Path

import pytest
import torch

from starVLA.model.modules.action_model.ACT_ActionHeader import TurboStyleACTActionHead
from starVLA.model.modules.action_model.CartesianResidualACT import CartesianResidualACT, cartesian_context

URDF = Path('/data/gaoxiang/Code/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf')


def test_cartesian_branch_changes_executed_pose_and_receives_gradient(tmp_path):
    if not URDF.exists():
        pytest.skip('RoboTwin Aloha asset unavailable')
    path = tmp_path/'stats.json'
    path.write_text(json.dumps({'aloha': {'action': {'q01': [-3.]*14, 'q99': [3.]*14}}}))
    original = TurboStyleACTActionHead(token_dim=8, hidden_dim=16, action_dim=14, horizon=2,
        num_frames=1, num_visual_tokens=2, num_heads=2, num_layers=1, dim_feedforward=32,
        mlp_hidden_dim=16, dropout=0, state_dim=14, state_hidden_dim=16, num_state_tokens=1,
        output_activation='identity', gripper_indices=(12, 13))
    seed = torch.tensor([.1, 2., 1.7, -1., .2, .4]*2+[0., 0.])
    with torch.no_grad():
        for parameter in original.action_projection.parameters():
            parameter.zero_()
        original.action_projection.layers[-1].bias.copy_(seed/3)
    head = CartesianResidualACT.from_existing(original, {'urdf_path': str(URDF)}, str(path))
    hidden = torch.randn(1, 2, 16)
    context = dict(head=head, inference=False)
    token = cartesian_context.set(context)
    try:
        with torch.inference_mode():
            baseline = head.predict_action(hidden)
        torch.testing.assert_close(baseline, (seed/3).expand_as(baseline), atol=2e-5, rtol=0)
        with torch.no_grad():
            head.pose_projection.layers[-1].bias[0] = torch.atanh(torch.tensor(.01/.08))
        corrected = head.predict_action(hidden)
        before = head._chains[0].pose(seed[:6].expand(1, 2, 6))
        actual = head._chains[0].pose(head._normalizer.inverse(corrected[..., :12])[..., :6])
        expected = before[..., :3, 3].clone()
        expected[..., 0] += .01
        assert context['ik_converged'].all()
        torch.testing.assert_close(actual[..., :3, 3], expected, atol=2e-4, rtol=0)
        loss = (corrected-baseline.detach()).square().sum()
        loss.backward()
        gradient = head.pose_projection.layers[-1].bias.grad
        assert torch.isfinite(gradient).all() and gradient[:3].abs().sum() > 1e-5
        assert all(p.grad is None for p in head.action_projection.parameters())
    finally:
        cartesian_context.reset(token)
    with pytest.raises(RuntimeError, match='isolated'):
        head.predict_action(hidden)


@pytest.mark.parametrize('translation_m', [0., .005])
def test_shared_features_receive_true_proposal_gradient(tmp_path, translation_m):
    if not URDF.exists():
        pytest.skip('RoboTwin Aloha asset unavailable')
    path = tmp_path/'stats.json'
    path.write_text(json.dumps({'aloha': {'action': {'q01': [-3.]*14, 'q99': [3.]*14}}}))
    original = TurboStyleACTActionHead(token_dim=8, hidden_dim=16, action_dim=14, horizon=1,
        num_frames=1, num_visual_tokens=2, num_heads=2, num_layers=1, dim_feedforward=32,
        mlp_hidden_dim=16, dropout=0, state_dim=14, state_hidden_dim=16, num_state_tokens=1,
        output_activation='identity', gripper_indices=(12,13))
    seed = torch.tensor([.1,2.,1.7,-1.,.2,.4]*2+[0.,0.])
    with torch.no_grad():
        for p in original.action_projection.parameters():
            p.zero_()
        for layer in original.action_projection.layers[:2]:
            layer.weight.copy_(torch.eye(16))
        original.action_projection.layers[-1].bias.copy_(seed/3)
        original.action_projection.layers[-1].weight[0,0] = .02
    head = CartesianResidualACT.from_existing(original, {'urdf_path':str(URDF)}, str(path))
    with torch.no_grad():
        head.pose_projection.layers[-1].bias[0] = torch.atanh(torch.tensor(translation_m/.08))
    hidden = torch.ones(1,1,16, requires_grad=True)
    token = cartesian_context.set(dict(head=head, inference=False))
    try:
        predicted = head.predict_action(hidden)[0,0,0]
        derivative = torch.autograd.grad(predicted, hidden)[0][0,0,0]
        plus, minus = hidden.detach().clone(), hidden.detach().clone()
        plus[0,0,0] += .001
        minus[0,0,0] -= .001
        with torch.no_grad():
            finite_difference = (head.predict_action(plus)[0,0,0]-head.predict_action(minus)[0,0,0])/.002
        assert abs(float(finite_difference)) > .005
        torch.testing.assert_close(derivative, finite_difference, atol=3e-4, rtol=.02)
        assert all(not p.requires_grad for p in head.action_projection.parameters())
    finally:
        cartesian_context.reset(token)
