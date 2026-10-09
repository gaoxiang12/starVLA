"""C defaults must affect real sampling, optimizer updates and stage resumes."""
import copy

from omegaconf import OmegaConf
import pytest
import torch

from starVLA.training.recipe import (apply_training_recipe, resolve_training_budget,
    prepare_parameter_precision, reset_stage_optimizer_if_needed)
from starVLA.training.train_starvla import setup_optimizer_and_scheduler


def config():
    return OmegaConf.create({'framework': {'name': 'Test'}, 'datasets': {'vla_data': {
        'data_mix': 'test', 'dataset_py': 'lerobot_datasets'}},
        'trainer': {'learning_rate': {'base': 1e-5, 'action_models': 1e-4},
                    'max_train_steps': 40000, 'freeze_modules': ''}})


def test_default_replaces_old_recipe_but_keeps_model_and_cli_overrides():
    cfg = config()
    out = apply_training_recipe(cfg, OmegaConf.from_dotlist(['trainer.max_train_steps=60']))
    assert cfg.trainer.max_train_steps == 40000
    assert out.trainer.recipe == 'c'
    assert out.trainer.distributed_backend == 'ddp'
    assert out.trainer.learning_rate == {'base': 2e-4}
    assert out.trainer.optimizer.betas == [0.9, 0.99]
    assert out.trainer.max_train_steps == 60
    assert out.datasets.vla_data.sampling_mode == 'auto'
    assert out.framework == cfg.framework
    assert apply_training_recipe(out) == out
    legacy = apply_training_recipe(cfg, OmegaConf.from_dotlist(['trainer.recipe=legacy']))
    assert legacy.trainer.learning_rate == cfg.trainer.learning_rate
    assert legacy.trainer.max_train_steps == 40000


def test_epoch_budget_uses_real_frames_and_accumulated_global_batch():
    class Frames:
        def __len__(self): return 6120962
    cfg = apply_training_recipe(config())
    resolve_training_budget(cfg, Frames(), 128)
    assert cfg.trainer.steps_per_epoch == 47820
    assert cfg.trainer.stage1_steps == 573840
    assert cfg.trainer.max_train_steps == 765120
    assert cfg.trainer.lr_scheduler_total_steps == 1912800


def test_gawm_default_freezes_encoder_without_changing_architecture():
    cfg = config()
    cfg.framework = {'name':'GAWM','world_model':{'encoder_spec':'vitb16','train_encoder':True}}
    out = apply_training_recipe(cfg)
    assert out.framework.world_model.train_encoder is False
    assert out.framework.world_model.encoder_spec == 'vitb16'
    override = apply_training_recipe(cfg,OmegaConf.from_dotlist([
        'framework.world_model.train_encoder=true','trainer.learning_rate.backbone.encoder=1e-6']))
    assert override.framework.world_model.train_encoder is True
    assert override.trainer.learning_rate.backbone.encoder == 1e-6


def test_precision_preserves_frozen_weights_and_nonpersistent_buffers():
    model = torch.nn.Linear(2, 2)
    model.bias.requires_grad_(False)
    model.register_buffer('rope', torch.tensor([1.234567]), persistent=False)
    original = model.rope.clone()
    prepare_parameter_precision(model, apply_training_recipe(config()))
    assert model.weight.dtype == torch.float32
    prepare_parameter_precision(model, apply_training_recipe(
        config(), OmegaConf.from_dotlist(['trainer.parameter_dtype=bfloat16'])))
    assert model.weight.dtype == torch.bfloat16
    assert model.bias.dtype == torch.float32
    assert model.rope.dtype == torch.float32
    torch.testing.assert_close(model.rope, original, rtol=0, atol=0)


def test_default_precision_keeps_small_adamw_updates():
    model = torch.nn.LayerNorm(8)
    prepare_parameter_precision(model, apply_training_recipe(config()))
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
    model(torch.randn(4, 8)).square().mean().backward()
    opt.step()
    assert not torch.all(model.weight == 1)


def test_auto_sampling_uses_frames_and_preserves_robot_weights(monkeypatch):
    from types import SimpleNamespace
    import starVLA.dataloader.lerobot_datasets as loaders
    monkeypatch.setitem(loaders.DATASET_NAMED_MIXTURES,'test',[('a',1.,'test'),('b',1.,'test')])
    monkeypatch.setattr(loaders,'LeRobotMixtureDataset',lambda *args,**kwargs: kwargs)
    tags = {'a':'aloha','b':'aloha'}
    monkeypatch.setattr(loaders,'make_LeRobotSingleDataset',lambda root,name,*a,**kw: SimpleNamespace(tag=tags[name]))
    cfg = apply_training_recipe(config())
    cfg.datasets.vla_data.data_root_dir='/tmp/unused'
    loaders.get_vla_dataset(cfg.datasets.vla_data)
    assert cfg.datasets.vla_data.sampling_mode == 'frame_epoch'
    tags['b']='franka'
    cfg.datasets.vla_data.sampling_mode='auto'
    with pytest.raises(ValueError,match='explicit embodiment_sampling_weights'):
        loaders.get_vla_dataset(cfg.datasets.vla_data)
    cfg.datasets.vla_data.sampling_mode='auto'
    cfg.datasets.vla_data.embodiment_sampling_weights={'aloha':.7,'franka':.3}
    loaders.get_vla_dataset(cfg.datasets.vla_data)
    assert cfg.datasets.vla_data.sampling_mode == 'weighted_mixture'
    assert cfg.datasets.vla_data.embodiment_sampling_weights == {'aloha':.7,'franka':.3}


def test_two_stage_updates_match_fresh_adam_and_resume():
    class Frames:
        def __len__(self): return 16
    cfg = apply_training_recipe(config(), OmegaConf.from_dotlist([
        'trainer.stage_epochs=[1,1]', 'trainer.scheduler_epochs=4']))
    resolve_training_budget(cfg, Frames(), 4)
    model = torch.nn.Linear(2, 1)
    reference = copy.deepcopy(model)
    opt, schedule = setup_optimizer_and_scheduler(model, cfg)
    refopt = torch.optim.AdamW(reference.parameters(), lr=2e-4, betas=(.9,.99), weight_decay=.01)
    refsched = torch.optim.lr_scheduler.CosineAnnealingLR(refopt, T_max=16, eta_min=5e-5)
    resumed = None
    for step in range(8):
        reset_stage_optimizer_if_needed(opt, step, 4)
        if step == 4:
            refopt = torch.optim.AdamW(reference.parameters(), lr=4e-5, betas=(.9,.99), weight_decay=.01)
            refsched = torch.optim.lr_scheduler.CosineAnnealingLR(refopt, T_max=16, eta_min=1e-5)
        assert opt.param_groups[0]['lr'] == pytest.approx(refopt.param_groups[0]['lr'], abs=1e-15)
        for m,o,s in [(model,opt,schedule),(reference,refopt,refsched)]:
            m(torch.ones(3,2)).square().mean().backward();o.step();o.zero_grad();s.step()
        for p,q in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=1e-7)
        if step == 5:
            resumed = copy.deepcopy(model)
            ro,rs = setup_optimizer_and_scheduler(resumed,cfg)
            ro.load_state_dict(copy.deepcopy(opt.state_dict()))
            rs.load_state_dict(copy.deepcopy(schedule.state_dict()))
        elif step > 5:
            resumed(torch.ones(3,2)).square().mean().backward();ro.step();ro.zero_grad();rs.step()
            for p,q in zip(model.parameters(),resumed.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_real_trainer_full_state_resume_and_frame_sequence(tmp_path):
    """Run the real loop/Accelerate save+load in independent CPU processes."""
    import os
    import subprocess
    import sys
    source = r'''
import os, sys
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from unittest.mock import patch
from starVLA.training.recipe import apply_training_recipe,resolve_training_budget
from starVLA.training.train_starvla import VLATrainer,setup_optimizer_and_scheduler
from starVLA.dataloader.lerobot_datasets import FrameEpochSampler,collate_fn
root, steps, resume = Path(sys.argv[1]),int(sys.argv[2]),sys.argv[3]=='yes'
root.mkdir(parents=True,exist_ok=True)
cfg=apply_training_recipe(OmegaConf.create(dict(framework=dict(name='Test'),
    datasets=dict(vla_data=dict()),trainer=dict(freeze_modules=''),
    output_dir=str(root),seed=42)),OmegaConf.from_dotlist([
    'trainer.stage_epochs=[1,1]','trainer.scheduler_epochs=4',
    'trainer.expected_global_batch_size=4','trainer.gradient_accumulation_steps=2',
    'datasets.vla_data.per_device_batch_size=2','datasets.vla_data.sampling_mode=frame_epoch',
    'trainer.save_interval=2','trainer.logging_frequency=1',
    'trainer.collect_rng_states=true',
    f'trainer.max_train_steps={steps}',f'trainer.is_resume={str(resume).lower()}']))
class Frames(Dataset):
    epoch=0
    def __len__(self): return 16
    def set_epoch(self,e): self.epoch=e
    def __getitem__(self,i): return {'index':i,'x':torch.tensor([i/16,1.]),'y':torch.tensor([i%3/3])}
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Dropout(.1),torch.nn.Linear(4,1));self.trace=[]
    def forward(self,examples):
        self.trace.extend(x['index'] for x in examples)
        x=torch.stack([e['x'] for e in examples]);y=torch.stack([e['y'] for e in examples])
        return {'action_loss':(self.net(x)-y).square().mean()}
torch.manual_seed(42)
model=Model();ds=Frames()
loader=DataLoader(ds,batch_size=2,sampler=FrameEpochSampler(ds,4,42),collate_fn=collate_fn,
                  generator=torch.Generator().manual_seed(42))
resolve_training_budget(cfg,ds,4)
opt,sched=setup_optimizer_and_scheduler(model,cfg)
acc=Accelerator(cpu=True,gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=2,sync_with_dataloader=False),
                step_scheduler_with_optimizer=False)
trainer=VLATrainer(cfg,model,loader,opt,sched,acc)
# A single-process CPU run has no distributed process group. Only guard the
# repository's legacy rank/print helpers; model/optimizer/loader/IO are real.
with patch('torch.distributed.get_rank',return_value=0):
    trainer.prepare_training();trainer.train();trainer.close_dataloader()
torch.save({'model':acc.get_state_dict(trainer.model),'trace':model.trace,'lr':trainer.lr_scheduler.get_last_lr()},root/f'smoke_{steps}.pt')
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', ACCELERATE_USE_CPU='true',
               ACCELERATE_USE_DEEPSPEED='false', WANDB_MODE='disabled',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', STARVLA_DISABLE_TQDM='1')
    for directory, steps, resume in [('full',8,'no'),('resume',3,'no'),('resume',8,'yes')]:
        result = subprocess.run([sys.executable,'-c',source,str(tmp_path/directory),str(steps),resume],
                                env=env,capture_output=True,text=True,timeout=90)
        assert result.returncode == 0, result.stdout+result.stderr
    full = torch.load(tmp_path/'full/smoke_8.pt',weights_only=True)
    prefix = torch.load(tmp_path/'resume/smoke_3.pt',weights_only=True)
    suffix = torch.load(tmp_path/'resume/smoke_8.pt',weights_only=True)
    assert full['trace'] == prefix['trace']+suffix['trace']
    for key in full['model']:
        torch.testing.assert_close(full['model'][key],suffix['model'][key],rtol=0,atol=0)
    assert full['lr'] == suffix['lr']
    checkpoints = tmp_path/'resume/checkpoints'
    assert (checkpoints/'steps_4_training_state/complete.json').is_file()  # stage boundary retained
    assert (checkpoints/'steps_8_training_state/complete.json').is_file()
    assert not (checkpoints/'steps_2_training_state').exists()
