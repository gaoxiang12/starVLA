"""Experiment C: B's exact model/data under the existing StarVLA trainer.

VLATrainer.train and VLATrainer._train_step are inherited without modification.
Only dataset/model boundary adapters, checkpoint IO and audit logging differ.
This standalone controlled experiment is not a unified pretraining mixture.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from starVLA.training.train_starvla import VLATrainer, build_accelerator, setup_optimizer_and_scheduler
from starVLA.dataloader.lerobot_datasets import FrameEpochSampler, collate_fn
from examples.LiLaWAM.gawm_official import GAWMOfficialDataset, GAWMOfficialWrapper
from examples.LiLaWAM.train_official_robotwin import (
    load_dataset_module, strict_collate, parameter_digest, worker_init,
    save_checkpoint, rng_state, restore_rng,
)
from examples.LiLaWAM.official_robotwin_data import write_json, digest, expected_episode_count


class OfficialFrameDataset(GAWMOfficialDataset):
    epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        example = super().__getitem__(index)
        example['robot_tag'] = 'aloha'
        example['frame_index'] = int(index)
        return example


class OfficialGAWMForStarVLA(GAWMOfficialWrapper):
    def forward(self, examples, **kwargs):
        loss, metrics = super().forward(strict_collate(examples))
        return dict(metrics, action_loss=loss)


def policy_core(model):
    return nn.ModuleDict({name: child for name, child in model.policy.named_children() if name != 'backbone'})


class OfficialRecipeTrainer(VLATrainer):
    """Use StarVLA's loop/update; export the same small, resumable files as B."""
    def _init_checkpointing(self):
        self.resume_training_state = None
        self.pending_rng = None
        path = self.config.trainer.get('pretrained_checkpoint')
        if path:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.core.load_state_dict(checkpoint['model_state_dict'], strict=True)
            if self.config.trainer.is_resume:
                if checkpoint['world_size'] != self.accelerator.num_processes:
                    raise ValueError('C resume must keep the same world size')
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.completed_steps = checkpoint['global_step']
                self.initial_hash = checkpoint['initial_random_model_sha256']
                self.pending_rng = {key: checkpoint[key] for key in ('world_size','rng','rng_by_rank','global_step')}

    def _adjust_lr_scheduler_for_resume(self):
        # Full scheduler state was restored above. Never fast-forward it again.
        pass

    def _save_checkpoint(self):
        states = [None] * self.accelerator.num_processes
        dist.all_gather_object(states, rng_state())
        output = Path(self.config.output_dir)
        if self.accelerator.is_main_process:
            epoch, step = divmod(self.completed_steps, self.steps_per_epoch)
            save_checkpoint(output/'latest.pt', self.core, self.optimizer, self.lr_scheduler,
                epoch, step, self.completed_steps, self.config, self.initial_hash, states)
            for milestone in self.config.experiment.milestone_steps:
                if self.completed_steps == milestone:
                    for src, dest in (('latest.pt', f'checkpoint_step_{milestone}.pt'), ('policy.pt', f'policy_step_{milestone}.pt')):
                        if not (output/dest).exists(): os.link(output/src, output/dest)
            write_json(output/'checkpoint_status.json', dict(global_step=self.completed_steps,
                checkpoint=str(output/'latest.pt'), world_size=self.accelerator.num_processes))
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        keys = sorted(metrics)
        values = torch.tensor([metrics[k] for k in keys], device='cuda', dtype=torch.float32)
        dist.all_reduce(values)
        values /= self.accelerator.num_processes
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite training metrics')
        if self.accelerator.is_main_process:
            record = dict(zip(keys, values.tolist()))
            epoch, step = divmod(self.completed_steps, self.steps_per_epoch)
            record.update(global_step=self.completed_steps, epoch=epoch+1, step_in_epoch=step,
                steps_per_epoch=self.steps_per_epoch, lr=self.optimizer.param_groups[0]['lr'],
                elapsed_seconds=time.monotonic()-self.started, world_size=self.accelerator.num_processes,
                global_batch_size=self.total_batch_size, status='training',
                trainer='starVLA.training.train_starvla.VLATrainer')
            self.metrics.write(json.dumps(record)+'\n')
            if self.completed_steps % 20 == 0 or self.completed_steps == self.start_step+1:
                write_json(Path(self.config.output_dir)/'status.json', record)
                print(json.dumps(record), flush=True)

    def _finalize_training(self):
        self._save_checkpoint()
        if self.accelerator.is_main_process:
            self.metrics.close()
            write_json(Path(self.config.output_dir)/'status.json', dict(status='completed',
                global_step=self.completed_steps, checkpoint=str(Path(self.config.output_dir)/'latest.pt')))


def run(args):
    cfg = OmegaConf.load(args.config)
    if args.output is not None:
        cfg.output_dir = str(args.output)
    if args.smoke_steps:
        cfg.trainer.max_train_steps = args.smoke_steps
    if args.resume:
        cfg.trainer.is_resume = True
        cfg.trainer.pretrained_checkpoint = str(args.resume)
    if args.init_from:
        cfg.trainer.pretrained_checkpoint = str(args.init_from)
        cfg.trainer.is_resume = False
    cfg.run_root_dir = str(Path(cfg.output_dir).parent)
    cfg.run_id = Path(cfg.output_dir).name
    if (cfg.trainer.gradient_accumulation_steps != 1 or cfg.datasets.vla_data.per_device_batch_size != 16
            or cfg.trainer.expected_global_batch_size != 128):
        raise ValueError('C must match B: eight ranks, local batch16, accumulation1')
    accelerator = build_accelerator(cfg)
    if accelerator.num_processes != 8:
        raise ValueError('C requires eight ranks to match B noise and batch assignment')
    rank = accelerator.process_index
    torch.set_num_threads(4)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    source = Path(cfg.official_source)
    module = load_dataset_module(source)
    dataset = OfficialFrameDataset(module.create_dataset(cfg, val=False), module)
    audit = json.loads(Path(cfg.dataset_audit).read_text())
    if (audit['status'] != 'passed' or audit['tasks'] != 50 or audit['frames'] != len(dataset)
            or len(dataset.all_episodes) != expected_episode_count(source/'utils/outlier_files 500-all.txt')
            or audit['outlier_report_sha256'] != digest(source/'utils/outlier_files 500-all.txt')):
        raise ValueError('Official dataset audit mismatch')
    loader = DataLoader(dataset, batch_size=16, sampler=FrameEpochSampler(dataset,128,seed=cfg.seed),
        drop_last=True, num_workers=2, pin_memory=True, collate_fn=collate_fn, worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(cfg.seed), prefetch_factor=2)
    model = OfficialGAWMForStarVLA(cfg, cfg.official_stats)
    core = policy_core(model)
    initial_hash = parameter_digest(core)
    baseline = json.loads(Path(cfg.baseline_provenance).read_text())
    if initial_hash != baseline['initial_random_model_sha256']:
        raise ValueError(f'Initial policy differs from B: {initial_hash}')
    optimizer, scheduler = setup_optimizer_and_scheduler(model, cfg)
    # Verify the generic optimizer groups contain B's exact trainable parameters in order.
    assert [id(p) for g in optimizer.param_groups for p in g['params']] == [id(p) for p in core.parameters()]
    output = Path(cfg.output_dir); output.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        OmegaConf.save(cfg, output/'config.yaml')
        write_json(output/'provenance.json', dict(initial_random_model_sha256=initial_hash,
            baseline_initialization_identical=True, dataset_frames=len(dataset), dataset_episodes=len(dataset.all_episodes),
            world_size=8, global_batch=128, local_batch=16, accumulation=1,
            training_loop='inherited VLATrainer.train', update='inherited VLATrainer._train_step',
            trainer_sha256=digest(Path(__file__).with_name('train_starvla.py')), adapter_sha256=digest(Path(__file__)),
            config_sha256=digest(args.config), stats_sha256=digest(Path(cfg.official_stats))))
    trainer = OfficialRecipeTrainer(cfg, model, loader, optimizer, scheduler, accelerator)
    trainer.raw, trainer.core, trainer.initial_hash = model, core, initial_hash
    trainer.steps_per_epoch = len(dataset)//128
    trainer.metrics = (output/'metrics.jsonl').open('a',buffering=1) if accelerator.is_main_process else None
    gradient_groups, batch_trace = {}, []
    last_examples = None
    if args.smoke_steps:
        for name, param in core.named_parameters():
            if param.requires_grad:
                def capture(grad, group=name.split('.')[0]):
                    gradient_groups[group] = max(gradient_groups.get(group,0.),float(grad.float().norm()))
                param.register_hook(capture)
        def capture_batch(_module, arguments):
            nonlocal last_examples
            last_examples = arguments[0]
            batch_trace.append([ex['frame_index'] for ex in last_examples])
        model.register_forward_pre_hook(capture_batch)
    try:
        trainer.prepare_training()
        model.train()
        if trainer.pending_rng:
            restore_rng(trainer.pending_rng,rank,8,cfg.seed)
        else:
            torch.cuda.manual_seed(cfg.seed+rank)
        trainer.start_step = trainer.completed_steps
        trainer.started = time.monotonic()
        trainer.train()
        if args.smoke_steps:
            hashes = [None]*8
            dist.all_gather_object(hashes,parameter_digest(core))
            assert len(set(hashes)) == 1, 'DDP replicas differ before reloading'
            for group in ('task_embedding','embodiment_embedding','visual_token_pooler','world_model','action_models'):
                assert gradient_groups.get(group,0.) > 0, group
            assert all(p.grad is None and not p.requires_grad for p in model.policy.backbone.encoder.parameters())
            trace = [None]*8; dist.all_gather_object(trace,batch_trace)
            model.eval()
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16): before=model(last_examples)['action_loss']
            saved=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
            core.load_state_dict(saved['model_state_dict'],strict=True)
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16): after=model(last_examples)['action_loss']
            assert torch.equal(before,after), 'Checkpoint reload changed inference loss'
            if accelerator.is_main_process:
                write_json(output/'smoke_result.json',dict(status='passed',updates=trainer.completed_steps-trainer.start_step,
                    initial_hash=initial_hash,final_hash=hashes[0],replicas_identical_before_reload=True,
                    checkpoint_reload_loss_difference=float((before-after).abs()),gradient_groups=gradient_groups,
                    frame_indices_by_rank=trace,peak_cuda_gb=torch.cuda.max_memory_allocated()/1e9))
    finally:
        trainer.close_dataloader()
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--smoke-steps',type=int,default=0)
    p.add_argument('--resume',type=Path)
    p.add_argument('--init-from',type=Path)
    a=p.parse_args()
    try: run(a)
    except Exception as exc:
        if int(os.environ.get('RANK',0)) == 0:
            out=a.output or Path(OmegaConf.load(a.config).output_dir)
            write_json(out/'status.json',dict(status='failed',error=repr(exc)))
        raise
