import json
from pathlib import Path

import pytest
import torch
from accelerate.data_loader import prepare_data_loader
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from starVLA.training.train_gawm_official import OfficialRecipeTrainer
from starVLA.training.train_starvla import VLATrainer, setup_optimizer_and_scheduler
from starVLA.dataloader.lerobot_datasets import FrameEpochSampler
from examples.LiLaWAM.train_official_robotwin import EpochSampler
from examples.LiLaWAM.run_gawm_C import validated_result

ROOT=Path(__file__).resolve().parents[1]


def test_C_inherits_actual_starvla_loop_and_update():
    assert OfficialRecipeTrainer.train is VLATrainer.train
    assert OfficialRecipeTrainer._train_step is VLATrainer._train_step


def test_C_accelerate_rank_assignment_matches_B_across_epochs():
    class Frames(Dataset):
        epoch=0
        def __len__(self): return 1027
        def __getitem__(self,i): return i
        def set_epoch(self,epoch): self.epoch=epoch
    for epoch in (0,1,7):
        for rank in range(8):
            ds=Frames()
            base=DataLoader(ds,batch_size=16,sampler=FrameEpochSampler(ds,128,42),drop_last=True)
            loader=prepare_data_loader(base,num_processes=8,process_index=rank,even_batches=False)
            loader.set_epoch(epoch)
            actual=torch.cat(list(loader)).tolist()
            expected=list(EpochSampler(len(ds),42+epoch,rank=rank,world_size=8,global_batch_size=128))
            assert actual==expected


def test_C_scheduler_matches_B_recipe_at_milestones():
    for stage in (1,2):
        cfg=OmegaConf.load(ROOT/f'examples/LiLaWAM/train_files/gawm_official_C_stage{stage}.yaml')
        cfg.trainer.freeze_modules=''
        model=torch.nn.Linear(1,1)
        optimizer,scheduler=setup_optimizer_and_scheduler(model,cfg)
        reference_optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.training.learning_rate)
        reference=torch.optim.lr_scheduler.CosineAnnealingLR(reference_optimizer,T_max=40*47820,eta_min=cfg.training.lr_min)
        for step in (0,1,117000,316000,cfg.trainer.max_train_steps):
            reference.last_epoch=step
            actual=cfg.training.learning_rate*scheduler.lr_lambdas[0](step)
            assert actual==pytest.approx(reference._get_closed_form_lr()[0],abs=1e-15)


def test_C_production_gate_rejects_partial_or_changed_evaluation(tmp_path):
    path=tmp_path/'summary.json'
    result={'state':'complete','trials':100,'sources_unchanged':True,'results':[{'task':'blocks_ranking_rgb'}]}
    path.write_text(json.dumps(result));assert validated_result(path)==result
    for key,value in [('state','incomplete'),('trials',99),('sources_unchanged',False)]:
        bad=dict(result);bad[key]=value;path.write_text(json.dumps(bad))
        with pytest.raises(ValueError): validated_result(path)
