import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf
from starVLA.model.framework.VLM4A.QwenGAWMDepth import pool_camera_tokens,QwenGAWMDepth


def test_raw_patch_merge_order_is_restored_before_pooling():
    spatial=torch.arange(64,dtype=torch.float32).reshape(8,8,1)
    packed=spatial.reshape(4,2,4,2,1).permute(0,2,1,3,4).reshape(64,1)
    actual=pool_camera_tokens(packed,torch.tensor([[1,8,8]]),merged=False)
    expected=torch.nn.functional.avg_pool2d(spatial.permute(2,0,1)[None],2).flatten(2).transpose(1,2)
    torch.testing.assert_close(actual,expected)


def test_merged_tokens_keep_spatial_and_camera_identity():
    features=torch.cat([torch.arange(16)[:,None],100+torch.arange(16)[:,None]]).float()
    actual=pool_camera_tokens(features,torch.tensor([[1,8,8],[1,8,8]]),merged=True)
    torch.testing.assert_close(actual,features.reshape(2,16,1))
    with pytest.raises(ValueError,match='Extra'):pool_camera_tokens(features,torch.tensor([[1,8,8]]),merged=True)


def transfer_model(variant,keys):
    m=QwenGAWMDepth.__new__(QwenGAWMDepth);nn.Module.__init__(m)
    m.depth_variant=variant;m.config=OmegaConf.create({'trainer':{'pretrained_checkpoint':'oft.pt','reload_modules':'qwen_vl_interface'}})
    m.state_dict=lambda:{k:torch.zeros(2) for k in keys}
    return m


def test_half_transfer_drops_only_later_layers_and_rejects_missing_retained_weights():
    prefix='qwen_vl_interface.model.model.language_model.layers.'
    key=prefix+'17.self_attn.q_proj.weight';late=prefix+'18.self_attn.q_proj.weight'
    m=transfer_model('half',[key]);source={key:torch.zeros(2),late:torch.ones(2)}
    assert set(m.remap_checkpoint_state_dict(source))=={key}
    with pytest.raises(ValueError,match='Missing'):m.remap_checkpoint_state_dict({late:torch.ones(2)})
    with pytest.raises(ValueError,match='Unexpected'):m.remap_checkpoint_state_dict({**source,prefix+'0.unknown.weight':torch.ones(2)})


def test_visual_transfer_rejects_shape_mismatch_and_unexpected_vision_weights():
    p='qwen_vl_interface.model.model.visual.';key=p+'patch_embed.proj.weight'
    m=transfer_model('vit',[key])
    source={key:torch.ones(2),p+'merger.linear_fc1.weight':torch.ones(2)}
    assert set(m.remap_checkpoint_state_dict(source))=={key}
    with pytest.raises(ValueError,match='incompatible'):m.remap_checkpoint_state_dict({key:torch.ones(3)})
    with pytest.raises(ValueError,match='Unexpected'):m.remap_checkpoint_state_dict({**source,p+'unknown.weight':torch.ones(2)})


def test_protocol_rejects_test_rollouts_or_different_seeds():
    from examples.Robotwin.audits.prepare_qwen_gawm_depth import validate_protocol
    protocol=dict(test_scenes=0,development_scenes=20,records=[dict(split='development',seed=x) for x in range(41000000,41000020)])
    validate_protocol(protocol)
    with pytest.raises(AssertionError):validate_protocol(dict(protocol,test_scenes=100))
    protocol['records'][-1]['split']='test'
    with pytest.raises(AssertionError):validate_protocol(protocol)
