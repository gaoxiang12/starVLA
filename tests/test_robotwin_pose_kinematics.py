from pathlib import Path

import pytest
import torch

from starVLA.model.modules.robotwin_pose_kinematics import SerialPoseKinematics, rotation_exp, rotation_log

URDF = Path('/data/gaoxiang/Code/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf')


@pytest.fixture
def chain():
    if not URDF.exists():
        pytest.skip('RoboTwin Aloha asset unavailable')
    return SerialPoseKinematics(URDF, 'fl_link6', [f'fl_joint{i}' for i in range(1, 7)])


def test_rotation_zero_pi_and_gradients():
    vectors = torch.tensor([[0., 0., 0.], [.2, -.3, .1], [3.1415, 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(rotation_log(rotation_exp(vectors)), vectors, atol=1e-9, rtol=0)
    zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: rotation_log(rotation_exp(v)), (zero,), atol=1e-5)


def test_jacobian_matches_finite_pose_differences(chain):
    q = torch.tensor([.1, 2., 1.7, -1., .2, .4], dtype=torch.float64)
    pose, jacobian = chain.pose_and_jacobian(q)
    columns = []
    for index in range(6):
        perturbation = torch.zeros_like(q)
        perturbation[index] = 1e-6
        plus, minus = chain.pose(q+perturbation), chain.pose(q-perturbation)
        linear = (plus[:3, 3]-minus[:3, 3])/2e-6
        angular = rotation_log(plus[:3, :3] @ minus[:3, :3].T)/2e-6
        columns.append(torch.cat((linear, angular)))
    torch.testing.assert_close(jacobian, torch.stack(columns, -1), atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(pose[:3, 3], chain(q), atol=1e-12, rtol=0)


def test_local_pose_recovery_and_unreachable_flag(chain):
    q = torch.tensor([[.1, 2., 1.7, -1., .2, .4]], dtype=torch.float64)
    truth = chain.pose(q+.04)
    solution = chain.solve_local(q, truth[..., :3, 3], truth[..., :3, :3])
    assert solution['converged'].all()
    assert solution['position_error_m'].max() < 1e-5
    assert solution['rotation_error_rad'].max() < 1e-4
    unreachable = chain.solve_local(q, truth[..., :3, 3]+10, truth[..., :3, :3])
    assert not unreachable['converged'].any()
    assert torch.isfinite(unreachable['joints']).all()
    assert (unreachable['joints'] >= chain.joint_limits[:, 0]).all()
    assert (unreachable['joints'] <= chain.joint_limits[:, 1]).all()


def test_ik_gradient_to_cartesian_target(chain):
    q = torch.tensor([[.1, 2., 1.7, -1., .2, .4]], dtype=torch.float64)
    target = chain.pose(q+.02)
    position = target[..., :3, 3].clone().requires_grad_()
    def solve(p):
        return chain.solve_local(q, p, target[..., :3, :3], iterations=5)['joints']
    assert torch.autograd.gradcheck(solve, (position,), atol=2e-4, rtol=2e-3)
    with pytest.raises(ValueError, match='finite'):
        chain.solve_local(q, position*float('nan'), target[..., :3, :3])
