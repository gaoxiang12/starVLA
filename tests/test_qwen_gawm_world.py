import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from starVLA.model.framework.VLM4A.QwenGAWMWorld import QwenWorldActionHead,QwenGAWMWorld
from starVLA.model.modules.action_model.ACT_ActionHeader import TurboStyleACTActionHead


def head():
    act=TurboStyleACTActionHead(token_dim=24,hidden_dim=24,action_dim=14,horizon=16,
        num_frames=1,num_visual_tokens=16,num_heads=4,num_layers=2,dim_feedforward=48,
        mlp_hidden_dim=32,dropout=0,output_activation='tanh_linear_tail',gripper_indices=(12,13))
    wm=OmegaConf.create(dict(residual_predictor_dim=24,residual_predictor_depth=2,
        residual_predictor_heads=4,residual_predictor_ffn=48,latent_stats_momentum=.9))
    return QwenWorldActionHead(act,wm)


def test_world_model_and_act_train_and_predicted_memory_affects_actions():
    torch.manual_seed(42);h=head();x=torch.randn(2,1,16,24,requires_grad=True)
    opt=torch.optim.AdamW(h.parameters(),lr=.003)
    for step in range(3):
        opt.zero_grad();pred=h(x);assert pred.shape==(2,16,14)
        pred.square().mean().backward()
        assert x.grad.abs().sum()>0
        assert h.world_model.residual_predictor.out.weight.grad.abs().sum()>0
        if step:assert h.world_model.residual_predictor.blocks[0].attn.in_proj_weight.grad.abs().sum()>0
        opt.step()
    h.eval();current=h.current_latent(x[:,0]);future=h.world_model.regress_future(current)
    assert not torch.allclose(h.decode(current,future),h.decode(current,current.expand(-1,2,-1,-1)))


def fake_model():
    m=QwenGAWMWorld.__new__(QwenGAWMWorld);nn.Module.__init__(m)
    m.action_model=head();m.qwen_vl_interface=nn.Identity()
    m._encode_features=lambda xs:torch.stack([x['image'] for x in xs])
    x=dict(image=torch.randn(16,24),future_images=[torch.randn(16,24),torch.randn(16,24)],
        action=np.zeros((16,14),np.float32),action_valid_mask=np.ones(16,bool),
        future_frame_valid_mask=np.ones(3,bool))
    return m,x


def test_bf16_decoder_accepts_fp32_world_scale_memory():
    h=head().to(torch.bfloat16)
    x=torch.randn(1,1,16,24,dtype=torch.bfloat16)
    current=h.current_latent(x[:,0])
    future=h.world_model.regress_future(current)
    assert h.world_model.delta_scale.dtype==torch.float32
    assert future.dtype==torch.float32
    prediction=h.decode(current,future)
    assert prediction.dtype==torch.float32 and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert torch.isfinite(h.world_model.residual_predictor.out.weight.grad).all()


def test_future_labels_affect_diagnostics_but_not_eval_actions_or_l1():
    m,x=fake_model();m.eval();a=m([x])
    y=dict(x,future_images=[v+10 for v in x['future_images']]);b=m([y])
    torch.testing.assert_close(a['action_loss'],b['action_loss'],atol=0,rtol=0)
    torch.testing.assert_close(a['action_loss'],a['l1_action_loss'],atol=0,rtol=0)
    assert not torch.isclose(a['latent_loss'],b['latent_loss'])


def test_invalid_tail_targets_do_not_affect_loss_or_world_statistics():
    m,x=fake_model();m.eval();x['action_valid_mask'][8:]=False
    x['future_frame_valid_mask'][1:]=False
    a=m([x]);y=dict(x,action=x['action'].copy(),future_images=[v+100 for v in x['future_images']])
    y['action'][8:]=np.nan;b=m([y])
    assert a['latent_loss']==0 and b['latent_loss']==0
    torch.testing.assert_close(a['action_loss'],b['action_loss'],atol=0,rtol=0)
    assert m.action_model.world_model._delta_scale_ready.item()==0
