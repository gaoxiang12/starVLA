"""Check fixed-teacher gradients, masking, future leakage and deployment parity."""
import copy
from unittest.mock import patch
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from starVLA.model.modules.world_model.GAWM import FixedDinoWorldModel
from starVLA.model.framework.WM4A.GAWM import GAWM
from test_gawm_l_vision import backbone, tiny_config

@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def small_model():
    return FixedDinoWorldModel(latent_dim=8,goal_dim=8,n_future=2,num_tokens=8,
        num_views=2,tokens_per_view=4,feature_dim=12,num_patches=4,
        dim=8,depth=1,num_heads=2,ffn_dim=16,decoder_dim=8,decoder_depth=1,decoder_heads=2)


def test_future_teacher_is_fixed_and_prediction_loss_reaches_current_adapter():
    torch.manual_seed(6)
    model=small_model()
    adapter=nn.Linear(12,8)
    latent=adapter(torch.randn(2,3,8,12)); latent.retain_grad()
    teacher=torch.randn(2,3,2,4,12,requires_grad=True)
    result=model(latent,ctx_len=1,goal=torch.randn(2,8),teacher_patches=teacher)
    result['latent_loss'].backward()
    assert teacher.grad is None
    assert adapter.weight.grad.abs().sum()>0
    assert latent.grad[:,0].abs().sum()>0
    assert latent.grad[:,1:].abs().sum()==0
    assert model.residual_predictor.out.weight.grad.abs().sum()>0
    assert model.feature_decoder.output.weight.grad.abs().sum()>0
    assert 'delta_scale' not in model.state_dict()
    torch.testing.assert_close(result['predicted_latent_rms'],torch.tensor(1.),atol=1e-4,rtol=0)


def test_future_inputs_cannot_change_predictions_and_train_infer_match():
    model=small_model().eval()
    latent=torch.randn(2,3,8,8); teacher=torch.randn(2,3,2,4,12); goal=torch.randn(2,8)
    a=model(latent,ctx_len=1,goal=goal,teacher_patches=teacher)
    changed=latent.clone(); changed[:,1:]=torch.randn_like(changed[:,1:])*10
    b=model(changed,ctx_len=1,goal=goal,teacher_patches=torch.randn_like(teacher))
    torch.testing.assert_close(a['pred_future_latent'],b['pred_future_latent'],rtol=0,atol=0)
    torch.testing.assert_close(a['pred_future_latent'],model.regress_future(latent[:,:1],goal),rtol=0,atol=0)
    restored=small_model().eval();restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    torch.testing.assert_close(restored.regress_future(latent[:,:1],goal),a['pred_future_latent'],rtol=0,atol=0)


def test_invalid_future_is_excluded_from_loss_and_all_invalid_is_zero():
    model=small_model();latent=torch.randn(2,3,8,8);teacher=torch.randn(2,3,2,4,12)
    mask=torch.ones(2,3,8,dtype=torch.bool);mask[:,2]=False
    a=model(latent,ctx_len=1,teacher_patches=teacher,loss_mask=mask)
    teacher[:,2]=torch.randn_like(teacher[:,2])*100
    b=model(latent,ctx_len=1,teacher_patches=teacher,loss_mask=mask)
    torch.testing.assert_close(a['latent_loss'],b['latent_loss'],atol=0,rtol=0)
    assert b['latent_loss_horizon_2']==0 and b['temporal_smoothness_loss']==0
    mask[:,1:]=False
    c=model(latent,ctx_len=1,teacher_patches=teacher,loss_mask=mask)
    assert c['latent_loss']==0
    c['latent_loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_backbone_teacher_is_final_layer_patches_and_independent_of_adapter():
    enc=backbone();pixels=torch.randn(6,3,16,16)
    features,teacher=enc._encode_patch_pixel_values(pixels,batch_size=1,time_steps=3,num_views=2,return_teacher=True)
    direct=enc.encoder(pixels).last_hidden_state[:,3:].reshape(1,3,2,4,8)
    torch.testing.assert_close(teacher,direct,atol=0,rtol=0)
    assert not teacher.requires_grad
    assert features.shape==(1,3,2,3,7,8)


def test_framework_backward_and_policy_reload():
    cfg,tag=tiny_config(2)
    cfg.framework.world_model.update(dict(future_objective='fixed_dino_patches',detach_wm_input=False,
        latent_cosine_weight=0.,gawm_l_bridge_norm='fixed_layernorm',feature_decoder_dim=8,
        feature_decoder_depth=1,feature_decoder_heads=2))
    def encoder(**kwargs):
        b=backbone(); b.encoder.config.patch_size=8
        return b
    with patch('starVLA.model.framework.WM4A.GAWM.get_world_model',side_effect=encoder):
        model=GAWM(cfg).train();spec=cfg.framework.action_model.embodiment_heads[tag]
        images=[Image.new('RGB',(16,16),(20,30,90))]*2
        sample=dict(image=images,future_images=[images,images],robot_tag=tag,lang='pick object',
            state=np.zeros(spec.state_dim,np.float32),action=np.zeros((spec.action_horizon,spec.action_dim),np.float32),
            view_valid_mask=[True]*2,action_valid_mask=[True]*spec.action_horizon,
            future_frame_valid_mask=[True,True,True],future_time_offsets_s=[0.,.2,.4])
        result=model([sample]);result['action_loss'].backward()
        assert torch.isfinite(result['action_loss'])
        assert all(p.grad is None for p in model.backbone.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.visual_token_pooler.parameters())
        assert 'dino_future_loss' in result and 'latent_mse_over_delta_scale_sq' not in result
        model.eval();out=model.predict_action([sample])['normalized_actions']
        restored=GAWM(cfg).eval();restored.load_state_dict(model.state_dict(),strict=True)
        np.testing.assert_array_equal(out,restored.predict_action([sample])['normalized_actions'])
        # Real future observations are supervision only; capture ACT inputs.
        memories=[]
        handle=model.action_models[tag].register_forward_pre_hook(lambda m,args: memories.append(args))
        model([sample]);handle.remove()
        assert not model.backbone.encoder.training
