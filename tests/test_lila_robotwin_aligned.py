"""Recipe alignment must not silently alter legacy training or deployment."""
import copy
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from omegaconf import OmegaConf
import torch

from starVLA.dataloader.lerobot_datasets import FrameEpochSampler
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer
from starVLA.model.framework.WM4A.LiLaWAMTrain import LiLaWAMTrain
from starVLA.model.framework.WM4A.LiLaWAM import smooth_chunk
from starVLA.training.train_starvla import setup_optimizer_and_scheduler
from tests.test_lila_training import config, sample, TinyVision


def test_minmax_degenerate_dimension_and_continuous_grip_roundtrip():
    normalizer = Normalizer('min_max_safe', {'min':[1.,0.,4.], 'max':[1.,1.,4.+1e-7]})
    x = torch.tensor([[1.,.37,4.],[1.5,.82,4.5]])
    y = normalizer.forward(x)
    torch.testing.assert_close(y, torch.tensor([[-1.,-.26,-1.],[0.,.64,0.]]))
    torch.testing.assert_close(normalizer.inverse(y), x)


def test_frame_epoch_has_no_duplicates_and_shards_complete_updates():
    class Dataset:
        def __len__(self): return 273
        def set_epoch(self, epoch): self.epoch = epoch
    sampler = FrameEpochSampler(Dataset(),128,42)
    first = list(sampler)
    assert len(first) == len(set(first)) == 256
    # Eight ranks x two samples x eight microbatches per optimizer update.
    microbatches = np.array(first).reshape(-1,8,2)
    ranks = [microbatches[:,rank].flatten() for rank in range(8)]
    assert len(set(np.concatenate(ranks))) == 256
    sampler.set_epoch(1)
    assert list(sampler) != first
    sampler.set_epoch(0)
    assert list(sampler) == first


def test_frame_epoch_indexes_exact_child_and_does_not_resample_errors():
    class Child:
        def __init__(self, offset): self.offset=offset
        def __getitem__(self, index):
            if index==1: raise ValueError('bad frame')
            return self.offset+index
    mixture = LeRobotMixtureDataset.__new__(LeRobotMixtureDataset)
    mixture._frame_epoch=True; mixture._frame_ends=np.array([3,7])
    mixture.datasets=[Child(0),Child(100)]
    mixture._activate_dataset_cache=lambda child:None
    assert len(mixture)==7
    assert mixture[3]==100
    import pytest
    with pytest.raises(ValueError,match='bad frame'): mixture[4]


def test_packer_retains_rectangular_resolution_and_float32():
    child = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
    child.data_cfg = dict(packed_image_size=[320,240], image_resize_resample='opencv_linear',
                          packed_numeric_dtype='float32', target_num_views=3, future_obs_frames=True,
                          include_state=True)
    child._modality_keys = dict(video=['video.a','video.b','video.c'],action=['action.a'],
                               state=['state.a'], language=['lang'])
    child.tag='aloha'
    raw=np.random.default_rng(3).integers(0,256,(2,48,64,3),dtype=np.uint8)
    data={key:raw for key in child._modality_keys['video']}
    data.update({'action.a':np.full((32,14),.123456789,np.float32),
                 'state.a':np.zeros((1,14),np.float32),'lang':['test']})
    packed=child._pack_sample(data)
    np.testing.assert_array_equal(np.asarray(packed['image'][0]),cv2.resize(raw[0],(320,240)))
    np.testing.assert_array_equal(np.asarray(packed['future_images'][0][2]),cv2.resize(raw[1],(320,240)))
    assert packed['action'].dtype==np.float32
    assert packed['state'].dtype==np.float32


def test_clamped_tail_supervision_is_explicit_and_future_used():
    cfg=config();cfg['framework']['lila']['tail_supervision']='clamp'
    model=LiLaWAMTrain(cfg,vision_encoder=TinyVision())
    a=sample();a['future_frame_valid_mask']=[True,False]
    b=copy.deepcopy(a);b['action_valid_mask']=[True]*4;b['future_frame_valid_mask']=[True,True]
    torch.manual_seed(31);x=model([a])
    torch.manual_seed(31);y=model([b])
    torch.testing.assert_close(x['action_loss'],y['action_loss'],rtol=0,atol=0)
    assert x['latent_loss']>0


def test_no_clipping_and_physical_action_smoothing():
    cfg=config();cfg['framework']['lila'].update(normalized_continuous_action_clip=None,
                                                smooth_actions=True,inference_time_grid='linspace')
    model=LiLaWAMTrain(cfg,vision_encoder=TinyVision()).eval()
    generator=torch.Generator().manual_seed(23)
    expected=torch.randn(1,4,7,generator=generator).numpy()
    with patch.object(model.core,'forward',side_effect=lambda x,*a,**kw:(torch.zeros_like(x),None,None)):
        output=model.predict_action([sample()],generator=torch.Generator().manual_seed(23))['normalized_actions']
    np.testing.assert_array_equal(output,expected)
    physical=np.random.default_rng(7).normal(size=(2,32,14))
    np.testing.assert_array_equal(model.postprocess_actions(physical),np.stack([smooth_chunk(c) for c in physical]))


def test_scheduler_uses_40_epoch_horizon_not_stop_epoch():
    cfg=OmegaConf.load('examples/LiLaWAM/train_files/robotwin_3view_aligned_stage1.yaml')
    with patch('starVLA.training.train_starvla.build_param_lr_groups',return_value=[{'params':[torch.nn.Parameter(torch.ones(1))],'name':'test'}]):
        optimizer,scheduler=setup_optimizer_and_scheduler(None,cfg)
    step=cfg.trainer.max_train_steps
    actual=scheduler.lr_lambdas[0](step)*2e-4
    expected=5e-5+(2e-4-5e-5)*(1+np.cos(np.pi*12/40))/2
    assert abs(actual-expected)<1e-12
    assert actual>5e-5


def test_robotwin_client_obeys_native_order_resize_and_execution_horizon():
    from examples.Robotwin.eval_files.model2robotwin_interface import ModelClient
    class Policy:
        def __init__(self,*a): pass
        def get_server_metadata(self):
            return dict(action_chunk_size=32,action_execution_horizon=16,image_size=[320,240],
                image_resize_resample='opencv_linear',action_keys=['action.left_joints','action.left_gripper',
                'action.right_joints','action.right_gripper'])
        def predict_action(self,payload):
            self.payload=payload
            return {'data':{'actions':np.tile(np.arange(14),(1,32,1))}}
    with patch('examples.Robotwin.eval_files.model2robotwin_interface.WebsocketClientPolicy',Policy):
        client=ModelClient('unused')
    ex=sample('aloha')
    out=client.step(ex,0)
    np.testing.assert_array_equal(out,np.arange(14))
    assert client.execute_horizon==16
    assert client.client.payload['examples'][0]['image'][0].shape==(240,320,3)


def test_real_accelerate_accumulation_matches_full_batch_update():
    # Isolate Accelerator's process-global state from the rest of the suite.
    import subprocess
    import sys
    code = '''
from types import SimpleNamespace
import torch
from accelerate import Accelerator
from starVLA.training.train_starvla import VLATrainer
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(.1))
    def forward(self, batch):
        x,y=batch[0]['x'],batch[0]['y']
        return {'action_loss':((x*self.weight-y)**2).mean()}
accelerator=Accelerator(cpu=True,gradient_accumulation_steps=2)
model=Model(); reference=Model()
opt=torch.optim.SGD(model.parameters(),lr=.1)
refopt=torch.optim.SGD(reference.parameters(),lr=.1)
model,opt=accelerator.prepare(model,opt)
trainer=VLATrainer.__new__(VLATrainer)
trainer.model=model; trainer.optimizer=opt; trainer.accelerator=accelerator
trainer.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(opt.optimizer,lambda _:1.)
trainer.config=SimpleNamespace(framework=SimpleNamespace(name='Test'),trainer=SimpleNamespace(gradient_clipping=1.))
trainer.completed_steps=0
x=torch.tensor([1.,3.]); y=torch.tensor([2.,-2.])
for _ in range(2):
    reference([{'x':x,'y':y}])['action_loss'].backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
    refopt.step();refopt.zero_grad()
    for i in range(2): trainer._train_step([{'x':x[i:i+1],'y':y[i:i+1]}])
    torch.testing.assert_close(model.weight,reference.weight,rtol=0,atol=1e-7)
assert trainer.lr_scheduler.last_epoch==2
'''
    result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_accelerate_frame_sampler_resume_preserves_epoch_and_position():
    from accelerate.data_loader import prepare_data_loader, skip_first_batches
    from torch.utils.data import DataLoader, Dataset
    class Frames(Dataset):
        epoch=0
        def __len__(self): return 259
        def __getitem__(self,i): return i
        def set_epoch(self,epoch): self.epoch=epoch
    def loader(rank,epoch):
        ds=Frames()
        original=DataLoader(ds,batch_size=2,sampler=FrameEpochSampler(ds,128),drop_last=True)
        result=prepare_data_loader(original,num_processes=8,process_index=rank,even_batches=False)
        result.set_epoch(epoch)
        return result
    for rank in range(8):
        original=list(loader(rank,3))
        resumed=skip_first_batches(loader(rank,3),8)
        resumed.set_epoch(3)
        actual=list(resumed)
        assert len(actual)==8
        for a,b in zip(actual,original[8:]): torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert not torch.equal(torch.cat(original),torch.cat(list(loader(rank,2))))
