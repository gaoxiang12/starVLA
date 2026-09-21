import torch
import pytest

from starVLA.model.modules.robotwin_kinematics import SerialTCPKinematics


def test_planar_chain_positions_and_gradients(tmp_path):
    path = tmp_path / 'two_joint.urdf'
    path.write_text('''<robot name="test"><link name="base"/><link name="a"/><link name="tip"/>
      <joint name="a" type="revolute"><parent link="base"/><child link="a"/><axis xyz="0 0 1"/></joint>
      <joint name="b" type="revolute"><parent link="a"/><child link="tip"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
      </robot>''')
    chain = SerialTCPKinematics(path, 'tip', ['a', 'b'], tool_offset=(1., 0., 0.))
    q = torch.tensor([[0., 0.], [torch.pi/2, 0.], [0., torch.pi/2]], dtype=torch.float64)
    torch.testing.assert_close(chain(q), torch.tensor([[2., 0., 0.], [0., 2., 0.], [1., 1., 0.]], dtype=torch.float64))
    torch.autograd.gradcheck(chain, (torch.tensor([[.3, -.5]], dtype=torch.float64, requires_grad=True),))
    with pytest.raises(ValueError):
        SerialTCPKinematics(path, 'tip', ['a'], tool_offset=(0., 0., 0.))
