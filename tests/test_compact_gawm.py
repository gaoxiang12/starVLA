import copy
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

from starVLA.model.modules.world_model.GAWM import (
    CompactFixedDinoWorldModel,
    FixedDinoWorldModel,
    temporal_rope,
)
from starVLA.model.framework.WM4A.GAWM import GAWM
from test_gawm_l_vision import backbone, tiny_config
from test_robotwin_official_hdf5 import dataset


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small(options=None, dense=0.):
    common = dict(latent_dim=8, goal_dim=8, n_future=2, num_tokens=8, num_views=2,
        tokens_per_view=4, feature_dim=12, num_patches=4, dim=8, depth=2, num_heads=2,
        ffn_dim=16, decoder_dim=8, decoder_depth=1, decoder_heads=2,
        time_offsets=(0,16,32), dense_smoothness_weight=dense, temporal_reference_dt=1.)
    return FixedDinoWorldModel(**common) if options is None else CompactFixedDinoWorldModel(
        **common, compact_options=options, state_dim=3)


def test_zero_option_is_checkpoint_and_output_compatible():
    torch.manual_seed(42)
    original = small().eval()
    torch.manual_seed(42)
    extended = small({}).eval()
    assert original.state_dict().keys() == extended.state_dict().keys()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, extended.state_dict()[key], atol=0, rtol=0)
    latent, teacher, goal = torch.randn(2,3,8,8), torch.randn(2,3,2,4,12), torch.randn(2,8)
    a = original(latent, ctx_len=1, teacher_patches=teacher, goal=goal)
    b = extended(latent, ctx_len=1, teacher_patches=teacher, goal=goal)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_rope_relative_time_invariance_and_unrotated_channels():
    q, k = torch.randn(2,2,6,8), torch.randn(2,2,6,8)
    positions = torch.tensor([0,0,1,1,2,2.])
    rotate = lambda x,p: temporal_rope(x,p,4)
    first = rotate(q,positions) @ rotate(k,positions).transpose(-1,-2)
    shifted = rotate(q,positions+7) @ rotate(k,positions+7).transpose(-1,-2)
    torch.testing.assert_close(first, shifted, atol=3e-6, rtol=2e-6)
    torch.testing.assert_close(rotate(q,positions)[...,4:],q[...,4:],atol=0,rtol=0)


@pytest.mark.parametrize('options', [{'wm_state':True},{'temporal_rope':True},{'history_frames':2},
    {'wm_state':True,'temporal_rope':True,'history_frames':2},{'motion_weight':1.}])
def test_gradients_future_is_teacher_only_and_strict_reload(options):
    model = small(options, dense=.1).eval()
    torch.nn.init.normal_(model.residual_predictor.out.weight, std=.1)
    latent = torch.randn(2,3,8,8,requires_grad=True)
    teacher = torch.randn(2,3,2,4,12,requires_grad=True)
    state = torch.randn(2,3,requires_grad=True)
    history = torch.randn(2,1,8,8,requires_grad=True)
    args = dict(ctx_len=1, goal=torch.randn(2,8), teacher_patches=teacher,
        state=state, history=history, temporal_latent=torch.randn(2,3,8,8),
        temporal_times=torch.tensor([[-1.,0.,1.]]).repeat(2,1), temporal_valid=torch.ones(2,dtype=torch.bool),
        temporal_event_weight=torch.ones(2), temporal_state=torch.randn(2,3,3), temporal_history=torch.randn(2,3,8,8))
    output = model(latent, **args)
    changed = latent.detach().clone();changed[:,1:] *= 20
    torch.testing.assert_close(output['pred_future_latent'], model(changed,**args)['pred_future_latent'],atol=0,rtol=0)
    (output['latent_loss']+output['auxiliary_loss']).backward()
    assert teacher.grad is None
    assert latent.grad[:,0].abs().sum() > 0
    if options.get('wm_state'):
        assert state.grad.abs().sum() > 0
    if options.get('history_frames'):
        assert history.grad.abs().sum() > 0
    restored = small(options,dense=.1).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    torch.testing.assert_close(restored(latent,**args)['pred_future_latent'],output['pred_future_latent'],atol=0,rtol=0)


def test_motion_loss_excludes_invalid_future_and_is_finite_when_all_invalid():
    model = small({'motion_weight':1.}).eval()
    latent = torch.randn(2,3,8,8)
    teacher = torch.randn(2,3,2,4,12)
    mask = torch.ones(2,3,8,dtype=torch.bool);mask[:,2]=False
    first = model(latent,ctx_len=1,teacher_patches=teacher,loss_mask=mask)
    teacher[:,2] *= -100
    second = model(latent,ctx_len=1,teacher_patches=teacher,loss_mask=mask)
    torch.testing.assert_close(first['latent_loss'],second['latent_loss'],atol=0,rtol=0)
    mask[:,1:] = False
    result = model(latent,ctx_len=1,teacher_patches=teacher,loss_mask=mask)
    assert result['latent_loss'] == 0
    result['latent_loss'].backward()


@pytest.mark.parametrize('options', [{'wm_state':True},{'act_task':True},{'temporal_rope':True},
    {'history_frames':2},{'motion_weight':1.},{'wm_state':True,'act_task':True,'temporal_rope':True}])
def test_framework_train_predict_alignment_and_task_condition(options):
    cfg,tag = tiny_config(2)
    cfg.framework.compact_study = options
    cfg.framework.world_model.update(dict(future_objective='fixed_dino_patches',detach_wm_input=False,
        latent_cosine_weight=0.,gawm_l_bridge_norm='fixed_layernorm',feature_decoder_dim=8,
        feature_decoder_depth=1,feature_decoder_heads=2,dense_temporal_smoothness_weight=.03))
    cfg.datasets.vla_data.temporal_neighbors = True
    def encoder(**kwargs):
        result=backbone();result.encoder.config.patch_size=8;return result
    with patch('starVLA.model.framework.WM4A.GAWM.get_world_model',side_effect=encoder):
        model = GAWM(cfg).eval()
        torch.nn.init.normal_(model.world_model.residual_predictor.out.weight,std=.1)
        spec=cfg.framework.action_model.embodiment_heads[tag]
        images=[Image.new('RGB',(16,16),(20,30,90))]*2
        sample=dict(image=images,future_images=[images,images],robot_tag=tag,lang='pick object',
            state=np.zeros(spec.state_dim,np.float32),action=np.zeros((spec.action_horizon,spec.action_dim),np.float32),
            view_valid_mask=[True,True],action_valid_mask=[True]*spec.action_horizon,
            future_frame_valid_mask=[True]*3,future_time_offsets_s=[0.,.2,.4],
            temporal_neighbor_images=[images,images],temporal_neighbor_times=[-.05,0,.05],
            temporal_neighbor_valid=True,temporal_event_weight=1.,
            temporal_neighbor_states=np.zeros((3,spec.state_dim),np.float32),
            temporal_history_images=[images,images,images])
        capture={}; original=model._predict_action_chunk
        def capture_action(*args,**kwargs):
            result=original(*args,**kwargs);capture['actions']=result;return result
        model._predict_action_chunk=capture_action
        result=model([sample])
        actions=capture['actions'].detach().numpy().copy()
        result['action_loss'].backward()
        assert all(p.grad is None for p in model.backbone.parameters())
        out=model.predict_action([sample])
        np.testing.assert_allclose(actions,out['normalized_actions'],atol=2e-5,rtol=2e-5)
        if options.get('history_frames'):
            assert out['_current_visual_latent'].shape==(1,1,8,8)
        restored=GAWM(cfg).eval();restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
        np.testing.assert_array_equal(out['normalized_actions'],restored.predict_action([sample])['normalized_actions'])
        if options.get('act_task'):
            projection=model.action_models[tag].task_projection
            assert projection.weight.grad.abs().sum()>0


def test_history_dataset_alignment_boundaries_and_neighbor_states(dataset):
    dataset.temporal_neighbors=True
    dataset.history_offset=16
    dataset.temporal_state_metadata=True
    sample=dataset[20]
    histories=sample['temporal_history_images']
    for image,index in zip(histories,[3,4,5]):
        torch.testing.assert_close(image[0],dataset[index]['image'][0],atol=0,rtol=0)
    assert sample['temporal_neighbor_states'].shape==(3,16)
    start=dataset[0]
    for image in start['temporal_history_images']:
        torch.testing.assert_close(image[0],start['image'][0],atol=0,rtol=0)


def test_policy_history_cache_resets_between_episodes():
    from examples.Robotwin.eval_files.gawm_hdf5_server import GAWMHDF5Policy
    class Model:
        compact_history=True
        def __init__(self): self.inputs=[]
        def predict_action(self,examples):
            self.inputs.append(examples[0].get('history_latent'))
            return dict(normalized_actions=np.zeros((1,32,14),np.float32),
                _current_visual_latent=torch.ones(1,1,8,8)*len(self.inputs))
    policy=GAWMHDF5Policy.__new__(GAWMHDF5Policy)
    policy.model=Model();policy.history_latent=None;policy.history_task=None
    policy.image_channel_order='rgb';policy.image_size=(16,16);policy.cameras=['head_camera','front_camera']
    policy.smooth_actions=False
    policy.stats={k:dict(min=[0]*n,max=[1]*n) for k,n in [('state',16),('action',14)]}
    example=dict(state=np.zeros(16),task_name='pick_bottle',image=[np.zeros((16,16,3),np.uint8)]*2)
    policy.predict_action([{**example,'reset_history':True}])
    policy.predict_action([example])
    policy.predict_action([{**example,'reset_history':True}])
    assert policy.model.inputs[0] is None and policy.model.inputs[2] is None
    torch.testing.assert_close(policy.model.inputs[1],torch.ones(1,8,8))


def test_client_signals_episode_boundary_once_per_reset():
    from examples.Robotwin.eval_files.gawm_hdf5_interface import ModelClient
    model=ModelClient.__new__(ModelClient)
    model.actions=deque();model.cameras=['head_camera'];model.episode_start=False
    requests=[]
    def predict(request):
        requests.append(request)
        return dict(ok=True,data=dict(actions=np.zeros((1,32,14)).tolist()))
    model.client=SimpleNamespace(predict_action=predict)
    obs={'observation':{'head_camera':{'rgb':np.zeros((16,16,3),np.uint8)}}}
    with patch('examples.Robotwin.eval_files.gawm_hdf5_interface.endpose_state',return_value=np.zeros(16)):
        model.reset()
        for _ in range(17):model.step(obs,'pick_bottle')
        model.reset();model.step(obs,'pick_bottle')
    assert [r['examples'][0]['reset_history'] for r in requests]==[True,False,True]
