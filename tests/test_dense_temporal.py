from unittest.mock import patch
import numpy as np
import pytest
import torch
from PIL import Image
from starVLA.model.modules.world_model.temporal_regularization import temporal_curvature_loss
from starVLA.model.framework.WM4A.GAWM import GAWM
from test_gawm_l_vision import backbone,tiny_config

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_linear_motion_irregular_times_not_penalized_and_spike_is():
    t=torch.tensor([[0.,.05,.13]])
    x=(2*t+1)[:,:,None,None].expand(1,3,4,8).clone().requires_grad_()
    valid=torch.tensor([True]);w=torch.ones(1)
    assert temporal_curvature_loss(x,t,valid,w)<1e-10
    y=x.clone();y[:,1]+=1
    loss=temporal_curvature_loss(y,t,valid,w);assert loss>0
    loss.backward();assert x.grad.abs().sum()>0
    torch.testing.assert_close(temporal_curvature_loss(y,t,valid,w*.25),loss*.25)
    assert temporal_curvature_loss(y,torch.zeros_like(t),~valid,w)==0
    with pytest.raises(ValueError,match='increasing'):temporal_curvature_loss(y,torch.zeros_like(t),valid,w)


@pytest.mark.parametrize("views", [2, 3])
def test_dense_temporal_loss_has_gradients_but_neighbors_do_not_leak_to_actions(views):
    cfg,tag=tiny_config(views)
    cfg.framework.world_model.update(dict(future_objective='fixed_dino_patches',detach_wm_input=False,
        latent_cosine_weight=0.,gawm_l_bridge_norm='fixed_layernorm',feature_decoder_dim=8,
        feature_decoder_depth=1,feature_decoder_heads=2,dense_temporal_smoothness_weight=.03))
    cfg.datasets.vla_data.temporal_neighbors=True
    def encoder(**kwargs):
        b=backbone();b.encoder.config.patch_size=8;return b
    with patch('starVLA.model.framework.WM4A.GAWM.get_world_model',side_effect=encoder):
        model=GAWM(cfg).eval();spec=cfg.framework.action_model.embodiment_heads[tag]
        frames=[Image.new('RGB',(16,16),(20,30,90))]*views
        prev=[Image.new('RGB',(16,16),(250,20,10))]*views
        sample=dict(image=frames,future_images=[frames,frames],robot_tag=tag,lang='pick object',
            state=np.zeros(spec.state_dim,np.float32),action=np.zeros((spec.action_horizon,spec.action_dim),np.float32),
            view_valid_mask=[True]*views,action_valid_mask=[True]*spec.action_horizon,
            future_frame_valid_mask=[True,True,True],future_time_offsets_s=[0.,.2,.4],
            temporal_neighbor_images=[prev,frames],temporal_neighbor_times=[-.05,0,.05],
            temporal_neighbor_valid=True,temporal_event_weight=1.)
        if views == 3:
            sample.pop('future_time_offsets_s')
            sample['temporal_neighbor_times'] = [-1., 0., 1.]
            assert model.world_model.temporal_reference_dt == 1.
            torch.testing.assert_close(model.world_model.time_offsets, torch.tensor([0.,16.,32.]))
        else:
            assert model.world_model.temporal_reference_dt == .05
        capture={};orig=model._predict_action_chunk
        def action(*a,**kw):
            result=orig(*a,**kw);capture['actions']=result;return result
        model._predict_action_chunk=action
        hook=model.world_model.register_forward_hook(lambda m,a,o:capture.update(wm=o))
        r=model([sample]);pred=capture['actions'].detach().clone()
        loss=capture['wm']['temporal_dense_loss'];loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.visual_token_pooler.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.world_model.residual_predictor.parameters())
        model([{**sample,'temporal_neighbor_images':[frames,prev]}])
        torch.testing.assert_close(pred,capture['actions'],atol=0,rtol=0)
        np.testing.assert_allclose(pred.detach().numpy(),model.predict_action([sample])['normalized_actions'],rtol=1e-5,atol=1e-5)
        assert all(p.grad is None for p in model.backbone.parameters());hook.remove()


def test_recorded_frames_and_seconds_have_same_relative_curvature():
    x = torch.randn(2, 3, 4, 8, requires_grad=True)
    times = torch.tensor([[-1.,0.,1.]]).expand(2,-1)
    valid = torch.ones(2, dtype=torch.bool)
    weight = torch.tensor([1.,.25])
    a = temporal_curvature_loss(x,times,valid,weight,reference_dt=1.)
    b = temporal_curvature_loss(x,times*.05,valid,weight,reference_dt=.05)
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(b,x)[0])
