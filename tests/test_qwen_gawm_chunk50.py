from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from starVLA.model.framework.VLM4A.QwenGAWM import QwenGAWM
from starVLA.model.framework.VLM4A.QwenGAWMChunk import QwenGAWMChunk
from starVLA.model.modules.action_model.ACT_ActionHeader import TurboStyleACTActionHead
from examples.Robotwin.eval_files.model2robotwin_interface import ModelClient


class FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(10,24)

    def forward(self,input_ids,**kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeInterface(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=torch.nn.Module();self.model.model=FakeBase()

    def build_qwenvl_inputs(self,images,instructions):
        self.counts=[s.count('🔍') for s in instructions]
        return dict(input_ids=torch.tensor([[1]+[9]*n+[2] for n in self.counts]))


def fixture(cls,horizon):
    torch.manual_seed(7)
    model=cls.__new__(cls);torch.nn.Module.__init__(model)
    model.action_horizon=horizon;model.head_type='ACT';model.action_token='🔍';model.action_token_id=9
    model.embodiment_head_specs={'aloha':{'action_spec_id':'test'}}
    model.config=OmegaConf.create({'framework':{'task_instruction':'blocks ranking rgb'},
                                 'datasets':{'vla_data':{'obs_image_size':[24,24]}}})
    model.qwen_vl_interface=FakeInterface()
    model.action_model=TurboStyleACTActionHead(token_dim=24,hidden_dim=32,action_dim=14,horizon=horizon,
        num_frames=1,num_visual_tokens=horizon,num_heads=4,num_layers=2,dim_feedforward=64,
        mlp_hidden_dim=32,dropout=0,state_dim=0)
    return model.eval()


def sample(horizon):
    return dict(image=[np.zeros((24,24,3),np.uint8)]*3,lang='blocks ranking rgb',
                action=np.zeros((horizon,14),np.float32))


def test_configurable_16_matches_frozen_framework_exactly():
    original=fixture(QwenGAWM,16);extended=fixture(QwenGAWMChunk,16)
    extended.load_state_dict(original.state_dict(),strict=True)
    np.testing.assert_array_equal(original.predict_action([sample(16)])['normalized_actions'],
                                  extended.predict_action([sample(16)])['normalized_actions'])
    torch.testing.assert_close(original([sample(16)])['action_loss'],extended([sample(16)])['action_loss'],atol=0,rtol=0)


def test_50_placeholders_and_masked_tail_have_correct_loss_and_gradient():
    model=fixture(QwenGAWMChunk,50);s=sample(50)
    s['action_valid_mask']=np.arange(50)<33;s['action'][33:]=np.nan
    result=model([s]);assert model.qwen_vl_interface.counts==[50]
    expected=model._predict_tensor([s])[:,:33].abs().mean()
    torch.testing.assert_close(result['action_loss'],expected)
    result['action_loss'].backward()
    assert torch.isfinite(model.action_model.action_queries.weight.grad).all()
    assert model.action_model.decoder.layers[0].multihead_attn.in_proj_weight.grad.abs().sum()>0
    assert model.predict_action([s])['normalized_actions'].shape==(1,50,14)
    with pytest.raises(ValueError,match='Expected target'):
        model([sample(16)])


class Policy50:
    def __init__(self,*args):self.calls=0
    def get_server_metadata(self):return dict(action_chunk_size=50)
    def predict_action(self,payload):
        self.calls+=1
        actions=np.repeat((100*(self.calls-1)+np.arange(50))[:,None],14,axis=1).astype(np.float32)
        return dict(data=dict(actions=actions[None]))


@patch('examples.Robotwin.eval_files.model2robotwin_interface.WebsocketClientPolicy',Policy50)
def test_execute_50_reobserves_at_50_and_100():
    model=ModelClient('unused',execute_horizon=50)
    s=dict(sample(50),state=np.zeros(14,np.float32))
    values=[model.step(s,step=i)[0] for i in range(101)]
    assert values==list(range(50))+list(range(100,150))+[200]
    assert model.client.calls==3


def test_50_registry_does_not_change_16_targets():
    from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
    old=ROBOT_TYPE_CONFIG_MAP['robotwin_continuous_next_wm']
    new=ROBOT_TYPE_CONFIG_MAP['robotwin_continuous_next50']
    assert old.action_indices==list(range(1,17)) and new.action_indices==list(range(1,51))
    assert old.action_spec_id==new.action_spec_id and old.state_spec_id==new.state_spec_id
