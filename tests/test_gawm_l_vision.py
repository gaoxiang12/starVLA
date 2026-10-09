from types import SimpleNamespace
from unittest.mock import patch
import copy

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch
from torch import nn

from starVLA.model.modules.gawm_l_vision import GAWMLVisualPooler, VTTConditioner, rgb_pixels
from starVLA.model.modules.world_model.GAWM import _GAWM_Interface
from starVLA.model.framework.WM4A.GAWM import GAWM
from starVLA.model.modules.world_model.GAWM import VisualTokenLatentWorldModel
from scripts.prepare_gawm_vtt import episode_difference


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(8, dtype=torch.float32), requires_grad=False)
        self.config = SimpleNamespace(hidden_size=8)

    def forward(self, pixel_values, **kwargs):
        x = pixel_values.mean((1,2,3))[:,None,None] + self.weight[None,None]
        x = x.expand(-1, 7, -1)  # 1 CLS + 2 registers + 4 patches
        return SimpleNamespace(last_hidden_state=x + 4,
                               hidden_states=tuple(x+i for i in range(5)))


def backbone():
    model = _GAWM_Interface.__new__(_GAWM_Interface)
    nn.Module.__init__(model)
    model.encoder = TinyEncoder()
    model._hidden_size = 16
    model._model_config = SimpleNamespace(hidden_size=16)
    model.train_encoder = False
    model.num_prefix_tokens = 3
    model.encoder_batch_size = 2
    model.normalized_pixels = False
    model.gawm_l_vision = True
    model.gawm_l_image_size = (16,16)
    model.feat_layers = (-4,-2,-1)
    return model


def test_dino_layer_order_keeps_cls_registers_and_views():
    encoder = backbone()
    pixels = torch.arange(6.).reshape(6,1,1,1).expand(-1,3,16,16)
    features = encoder._encode_patch_pixel_values(pixels, batch_size=1, time_steps=2, num_views=3)
    assert features.shape == (1,2,3,3,7,8)
    for t in range(2):
        for v in range(3):
            for layer, offset in enumerate((1,3,4)):
                torch.testing.assert_close(features[0,t,v,layer,0], torch.arange(8.)+t*3+v+offset)
    assert all(p.grad is None for p in encoder.parameters())


def test_each_camera_uses_same_adapter_without_cross_view_mixing():
    torch.manual_seed(8)
    pool = GAWMLVisualPooler(8, 8, 3, 4, 3, 16, 1, 2).eval()
    features = torch.randn(2,2,3,3,7,8)
    output = pool.official_tokens(features)
    changed = features.clone(); changed[:,:,1] += torch.randn_like(changed[:,:,1])
    second = pool.official_tokens(changed)
    torch.testing.assert_close(output[:,:,:1], second[:,:,:1], rtol=0, atol=0)
    torch.testing.assert_close(output[:,:,2:], second[:,:,2:], rtol=0, atol=0)
    assert not torch.equal(output[:,:,1], second[:,:,1])
    tokens, content = pool(features, True, torch.ones(2,3,dtype=torch.bool))
    torch.testing.assert_close(pool.remove_position(tokens), content)
    with pytest.raises(ValueError, match='every configured'):
        pool(features, view_valid_mask=torch.tensor([[True,False,True]]*2))


def vtt_cfg():
    return OmegaConf.create(dict(task_vectors=dict(format_version=1, split='train',
        vectors={'franka:pick object': list(range(8))})))


def test_vtt_checkpoint_is_self_contained_and_rejects_wrong_vocabulary():
    cfg = vtt_cfg()
    model = VTTConditioner(cfg,8,8,16).eval()
    checkpoint = copy.deepcopy(model.state_dict())
    deployment_cfg = OmegaConf.create(dict(task_names=list(cfg.task_names)))
    loaded = VTTConditioner(deployment_cfg,8,8,16).eval()
    with pytest.raises(RuntimeError, match='Load checkpoint'):
        loaded(['pick object'], robot_tag='franka', device='cpu')
    loaded.load_state_dict(checkpoint)
    torch.testing.assert_close(model(['PICK  object'],robot_tag='franka',device='cpu'),
                               loaded(['pick object'],robot_tag='franka',device='cpu'), atol=0, rtol=0)
    with pytest.raises(KeyError, match='No training VTT'):
        loaded(['unknown'], robot_tag='franka', device='cpu')
    wrong = VTTConditioner(OmegaConf.create(dict(task_names=['franka:other'])),8,8,16)
    with pytest.raises(RuntimeError, match='vocabulary'):
        wrong.load_state_dict(checkpoint, strict=False)
    wrong_vectors = copy.deepcopy(checkpoint); wrong_vectors['vectors'][0,0] += 1
    with pytest.raises(RuntimeError, match='differ'):
        model.load_state_dict(wrong_vectors)


def test_vtt_uses_final_minus_initial_cls_with_official_rgb_preprocessing():
    frames = [np.full((20,30,3), color, np.uint8) for color in [(240,0,20),(0,100,220)]]
    pixels = rgb_pixels(frames,(16,16))
    resized = np.stack([cv2.resize(im,(16,16),interpolation=cv2.INTER_LINEAR) for im in frames]).astype(np.float32)/255
    expected = (resized-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    np.testing.assert_array_equal(pixels.numpy(),expected.transpose(0,3,1,2))
    diff = episode_difference(TinyEncoder(),frames,(16,16),'cpu')
    expected_cls = TinyEncoder()(pixel_values=pixels).last_hidden_state[:,0]
    np.testing.assert_array_equal(diff, (expected_cls[1]-expected_cls[0]).numpy())


def tiny_config(views):
    bench = 'robotwin' if views == 3 else 'libero'
    folder = 'Robotwin' if views == 3 else 'LIBERO'
    cfg=OmegaConf.load(f'examples/{folder}/train_files/starvla_gawm_l_{bench}_vtt_c_12plus4.yaml')
    tag = 'aloha' if views == 3 else 'franka'
    cfg.framework.lang_cond.task_vectors=dict(format_version=1,split='train',vectors={f'{tag}:pick object':list(range(8))})
    cfg.framework.lang_cond.embed_dim=8
    cfg.framework.world_model.update(dict(visual_token_dim=8,visual_tokens_per_view=4,gawm_l_adapter_dim=16,
        gawm_l_adapter_depth=1,gawm_l_adapter_heads=2,feat_layers=[-4,-2,-1],gawm_l_image_size=[16,16],
        imagenet_normalized_inputs=False,residual_predictor_dim=8,residual_predictor_depth=1,
        residual_predictor_heads=2,residual_predictor_ffn=16,sync_latent_stats=False))
    cfg.framework.action_model.update(dict(action_hidden_dim=8,act_num_heads=2,act_num_layers=1,
        act_dim_feedforward=16,act_mlp_hidden_dim=16))
    return cfg, tag


@pytest.mark.parametrize('views',[2,3])
@pytest.mark.parametrize('bridge_norm',['none','fixed_layernorm'])
def test_gawm_forward_backward_reload_and_deployment(views,bridge_norm):
    cfg, tag=tiny_config(views)
    cfg.framework.world_model.gawm_l_bridge_norm=bridge_norm
    with patch('starVLA.model.framework.WM4A.GAWM.get_world_model',side_effect=lambda **kwargs: backbone()):
        model=GAWM(cfg)
        assert isinstance(model.world_model, VisualTokenLatentWorldModel)
        model.train()
        assert not model.backbone.encoder.training
        spec=cfg.framework.action_model.embodiment_heads[tag]
        frames=[Image.new('RGB',(16,16),(20*i,30,90)) for i in range(views)]
        sample=dict(image=frames,future_images=[frames,frames],robot_tag=tag,lang='pick object',
            state=np.zeros(spec.state_dim,np.float32), action=np.zeros((spec.action_horizon,spec.action_dim),np.float32),
            action_valid_mask=[True]*spec.action_horizon,view_valid_mask=[True]*views,
            future_frame_valid_mask=[True,True,False])
        if cfg.framework.world_model.future_time_offsets_s is not None:
            sample['future_time_offsets_s']=list(cfg.framework.world_model.future_time_offsets_s)
        result=model([sample]); assert torch.isfinite(result['action_loss'])
        result['action_loss'].backward()
        # GAWM's residual output starts at zero: the goal path becomes active
        # after the first optimizer update, without changing world-model init.
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        optimizer.step(); optimizer.zero_grad()
        model([sample])['action_loss'].backward()
        for module in [model.visual_token_pooler.fusion, model.visual_token_pooler.adapter,
                       model.task_embedding.projection, model.world_model, model.action_models[tag]]:
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()), type(module).__name__
        assert model.task_embedding.vectors.grad is None
        assert all(p.grad is None for p in model.backbone.parameters())
        model.eval()
        prediction=model.predict_action([sample])['normalized_actions']
        assert prediction.shape==(1,spec.action_horizon,spec.action_dim)
        restored=GAWM(cfg).eval(); restored.load_state_dict(model.state_dict())
        np.testing.assert_array_equal(prediction,restored.predict_action([sample])['normalized_actions'])
        with pytest.raises(ValueError,match='physical camera'):
            restored.predict_action([{**sample,'image':frames[:-1]}])


def test_published_configs_keep_world_model_and_action_contract():
    for folder,bench,base,views in [('Robotwin','robotwin','starvla_gawm_robotwin_3view_c_12plus4.yaml',3),
                                  ('LIBERO','libero','starvla_gawm_c_12plus4.yaml',2)]:
        cfg=OmegaConf.load(f'examples/{folder}/train_files/starvla_gawm_l_{bench}_vtt_c_12plus4.yaml')
        old=OmegaConf.load(f'examples/{folder}/train_files/{base}')
        assert cfg.framework.action_model==old.framework.action_model
        for field in ['n_future','ctx_len','visual_token_dim','residual_predictor_dim','residual_predictor_depth',
                      'residual_predictor_heads','residual_predictor_ffn','detach_wm_input','loss_latent_weight',
                      'latent_cosine_weight','future_time_offsets_s']:
            assert cfg.framework.world_model[field]==old.framework.world_model[field]
        assert cfg.framework.world_model.num_views==views
        assert cfg.framework.world_model.visual_tokens_per_view==64
        assert cfg.framework.world_model.feat_layers==[-12,-8,-4]
        if views==3: assert cfg.datasets.vla_data.cameras==['head_camera','left_camera','right_camera']
        else: assert cfg.datasets.vla_data.target_num_views==2


def test_vtt_preparation_means_all_training_episodes_and_saves_vocabulary(tmp_path):
    import json
    from scripts.prepare_gawm_vtt import prepare
    cfg, tag=tiny_config(2)
    del cfg.framework.lang_cond.task_vectors
    output=tmp_path/'vectors.json';cfg.framework.lang_cond.task_vectors_path=str(output)
    path=tmp_path/'input.yaml';OmegaConf.save(cfg,path)
    endpoints=[]
    for i in range(3):
        frames=[np.full((16,16,3),20,np.uint8),np.full((16,16,3),40+i*30,np.uint8)]
        endpoints.append(('franka:pick object',frames,dict(episode=i)))
    with patch('scripts.prepare_gawm_vtt.AutoModel.from_pretrained',return_value=TinyEncoder()), \
         patch('scripts.prepare_gawm_vtt.training_endpoints',return_value=iter(endpoints)):
        payload=prepare(str(path))
    expected=np.mean([episode_difference(TinyEncoder(),row[1],(16,16),'cpu') for row in endpoints],axis=0)
    np.testing.assert_allclose(payload['vectors']['franka:pick object'],expected,atol=1e-6)
    assert len(json.loads(output.read_text())['provenance']['franka:pick object'])==3
    prepared=OmegaConf.load(output.with_suffix('.prepared.yaml'))
    assert prepared.framework.lang_cond.task_names==['franka:pick object']


def test_access_tracked_config_persists_vtt_and_camera_contract(tmp_path):
    from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig
    cfg,_=tiny_config(2)
    tracked=AccessTrackedConfig(cfg)
    with patch('starVLA.model.framework.WM4A.GAWM.get_world_model',side_effect=lambda **kwargs: backbone()):
        model=GAWM(tracked)
    path=tmp_path/'config.yaml'
    tracked.save_accessed_config(path,use_original_values=False)
    saved=OmegaConf.load(path)
    assert saved.framework.lang_cond.task_names==['franka:pick object']
    assert saved.framework.world_model.visual_frontend=='gawm_l'
    assert saved.framework.world_model.camera_names==['agentview','eye_in_hand']


def test_libero_client_uses_server_size_and_opencv_pixels():
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient
    class Policy:
        def __init__(self,*args): self.payload=None
        def get_server_metadata(self):
            return dict(action_chunk_size=8,image_size=[32,16],image_resize_resample='opencv_linear')
        def predict_action(self,payload):
            self.payload=payload
            return {'data':{'actions':np.zeros((1,8,7))}}
    raw=np.random.default_rng(5).integers(0,256,(40,45,3),dtype=np.uint8)
    with patch('examples.LIBERO.eval_files.model2libero_interface.WebsocketClientPolicy',Policy):
        client=ModelClient()
        client.step(dict(image=[raw,raw],lang='pick object'),step=0)
    sent=client.client.payload['examples'][0]['image']
    for image in sent:
        np.testing.assert_array_equal(image,cv2.resize(raw,(32,16),interpolation=cv2.INTER_LINEAR))


def test_robotwin_client_sends_three_physical_cameras_in_checkpoint_order(monkeypatch):
    from examples.Robotwin.eval_files.gawm_hdf5_interface import ModelClient
    cameras=['head_camera','left_camera','right_camera']
    class Policy:
        def __init__(self,*args): self.payload=None
        def get_server_metadata(self):
            return dict(framework='GAWMOfficialHDF5',action_chunk_size=32,execute_horizon=16,
                state_dim=16,action_dim=14,state_representation='endpose',
                action_order='left_arm6,left_gripper,right_arm6,right_gripper',
                policy_image_channel_order='rgb',simulator_image_channel_order='rgb',
                swap_rb_before_normalization=False,ckpt_path='/tmp/model.pt',cameras=cameras)
        def predict_action(self,payload):
            self.payload=payload
            return {'ok':True,'data':{'actions':np.zeros((1,32,14))}}
    monkeypatch.setenv('ROBOTWIN_POLICY_IMAGE_CHANNEL_ORDER','rgb')
    images={name:dict(rgb=np.full((8,8,3),index,np.uint8)) for index,name in enumerate(cameras)}
    obs=dict(observation=images,endpose=dict(left_endpose=[0]*7,right_endpose=[0]*7,left_gripper=0.,right_gripper=0.))
    with patch('examples.Robotwin.eval_files.gawm_hdf5_interface.WebsocketClientPolicy',Policy):
        client=ModelClient(dict(port=1,policy_ckpt_path='/tmp/model.pt'))
        client.step(obs,'pick_object')
    sent=client.client.payload['examples'][0]['image']
    assert [int(im[0,0,0]) for im in sent]==[0,1,2]


def test_fixed_bridge_norm_bounds_scale_and_marks_checkpoint():
    torch.manual_seed(42)
    pool=GAWMLVisualPooler(8,8,2,4,3,16,1,2,bridge_norm='fixed_layernorm').eval()
    features=torch.randn(2,3,2,3,7,8)
    tokens,content=pool(features,True)
    torch.testing.assert_close(content.float().square().mean(-1),torch.ones_like(content[...,0]),atol=2e-3,rtol=0)
    with torch.no_grad():
        pool.bridge.weight.mul_(100);pool.bridge.bias.mul_(100)
    scaled_tokens,scaled=pool(features,True)
    torch.testing.assert_close(content,scaled,atol=2e-3,rtol=2e-3)
    torch.testing.assert_close(pool.remove_position(scaled_tokens),scaled)
    scaled.square().mean().backward()
    assert pool.bridge.weight.grad is not None and torch.isfinite(pool.bridge.weight.grad).all()
    legacy=GAWMLVisualPooler(8,8,2,4,3,16,1,2)
    legacy_state=legacy.state_dict()
    assert 'bridge_norm_version' not in legacy_state
    with pytest.raises(RuntimeError,match='bridge_norm_version'):pool.load_state_dict(legacy_state,strict=True)
    with pytest.raises(RuntimeError,match='bridge_norm_version'):legacy.load_state_dict(pool.state_dict(),strict=True)
    with pytest.raises(ValueError,match='normalization'):GAWMLVisualPooler(8,8,2,bridge_norm='typo')
