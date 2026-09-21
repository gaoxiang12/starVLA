import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from examples.LiLaWAM.gawm_official import OfficialNormalizer, prepare_precision
from starVLA.model.framework.WM4A.GAWM import VisualTokenPooler
from starVLA.model.modules.world_model.visual_token_delta_world_model import VisualTokenLatentWorldModel
from examples.Robotwin.eval_files.lila_wam_interface import ModelClient


def test_rectangular_pooling_preserves_row_major_geometry():
    pool = VisualTokenPooler(3, 3, 1, 4, input_grid_shape=(2, 4))
    pool.patch_norm = torch.nn.Identity()
    pool.patch_proj = torch.nn.Identity()
    pool.out_norm = torch.nn.Identity()
    patches = torch.arange(24.).reshape(1, 1, 1, 8, 3).requires_grad_()
    _, content = pool(patches, return_content=True)
    expected = patches.reshape(1, 1, 2, 2, 2, 3).mean(dim=4).reshape(1, 1, 4, 3)
    torch.testing.assert_close(content, expected)
    content.sum().backward()
    torch.testing.assert_close(patches.grad, torch.full_like(patches, .5))
    with pytest.raises(ValueError, match='does not match'):
        pool(torch.zeros(1, 1, 1, 9, 3))


def test_bf16_policy_preserves_frozen_dino_rope_precision():
    policy = torch.nn.Module()
    policy.backbone = torch.nn.Linear(2, 2).to(torch.bfloat16)
    frequencies = torch.tensor([.1234567, .9876543])
    policy.backbone.register_buffer('rope_frequencies', frequencies.clone(), persistent=False)
    policy.head = torch.nn.Linear(2, 2)
    prepare_precision(policy, 'cpu')
    assert policy.head.weight.dtype == torch.bfloat16
    assert policy.backbone.rope_frequencies.dtype == torch.float32
    assert torch.equal(policy.backbone.rope_frequencies, frequencies)


def test_global_delta_stats_weight_by_valid_elements():
    wm = VisualTokenLatentWorldModel(latent_dim=4, goal_dim=4, n_future=2,
        num_tokens=4, dim=8, depth=1, num_heads=2, ffn_dim=16)
    wm.sync_stats = True
    # Remote rank contributes 2 residual values of 4: sum squares 32, count 2.
    def remote(moments):
        moments.add_(torch.tensor([32., 2.]))
    residual = torch.full((1, 2, 4, 4), 2.)
    mask = torch.zeros(1, 2, 4, 1)
    mask[0, 0, 0] = 1  # Local rank contributes 4 values of 2.
    with patch('torch.distributed.is_initialized', return_value=True), patch('torch.distributed.all_reduce', side_effect=remote):
        wm._update_delta_scale(residual, mask)
    torch.testing.assert_close(wm.delta_scale, torch.tensor([8.**.5]))


def test_official_constant_channel_is_not_divided_by_epsilon(tmp_path):
    path = tmp_path / 'stats.json'
    path.write_text(json.dumps({'robotwin2': {'action': {'min': [2., 3.], 'max': [2., 5.]}}}))
    norm = OfficialNormalizer(path)
    x = torch.tensor([[2., 4.]])
    torch.testing.assert_close(norm.normalize(x, 'action'), torch.tensor([[-1., 0.]]))
    torch.testing.assert_close(norm.denormalize(norm.normalize(x, 'action')), x)


def test_gawm_client_requires_official_endpose_contract():
    metadata = dict(framework='GAWM_Experiment_B', policy_rng='simulator_cuda', ckpt_path='/tmp/model.pt',
        state_dim=16, state_representation='endpose', action_order='left_arm6,left_gripper,right_arm6,right_gripper',
        execute_horizon=16, action_chunk_size=32)
    fake = SimpleNamespace(get_server_metadata=lambda: metadata)
    with patch('examples.Robotwin.eval_files.lila_wam_interface.WebsocketClientPolicy', return_value=fake):
        client = ModelClient({'policy_ckpt_path': '/tmp/model.pt'})
        assert client.horizon == 16 and client.chunk_size == 32
        metadata['state_dim'] = 14
        with pytest.raises(ValueError, match='contract mismatch'):
            ModelClient({'policy_ckpt_path': '/tmp/model.pt'})
