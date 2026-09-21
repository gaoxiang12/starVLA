"""Extend a completed LiLa run with Adam state and continuous cosine learning rate.

Source weights/optimizer remain untouched. Only the new run's scheduler and
Accelerate microstep metadata are rebased; the actual DeepSpeed update count is
the authoritative starting step.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf


def cosine_base(start_step, end_step, current_lr, min_lr):
    if not 0 <= start_step < end_step or not 0 < min_lr <= current_lr:
        raise ValueError('Invalid continuation steps or learning-rate limits')
    factor = .5 * (1 + math.cos(math.pi * start_step / end_step))
    return min_lr + (current_lr - min_lr) / factor


def prepare(source, target_config, run_id, end_step=100000, source_step=20000, min_lr=1e-5):
    source, target_config = Path(source).resolve(), Path(target_config).resolve()
    state = source / f'checkpoints/steps_{source_step}_training_state'
    tag = (state / 'latest').read_text().strip()
    metadata = torch.load(state/tag/'mp_rank_00_model_states.pt', map_location='cpu',
                          weights_only=False, mmap=True)
    scheduler = torch.load(state/'scheduler.bin', map_location='cpu', weights_only=False)
    random = torch.load(state/'random_states_0.pkl', map_location='cpu', weights_only=False)
    start = int(metadata['global_steps'])
    rates = scheduler['_last_lr']
    if len(rates) != 1 or scheduler['last_epoch'] != source_step:
        raise ValueError('Expected a single LR group and matching source scheduler step')
    cfg = OmegaConf.load(source/'config.full.yaml')
    for key in ('output_dir', 'config_yaml'):
        if key in cfg:
            del cfg[key]
    cfg.run_id = run_id
    cfg.trainer.max_train_steps = end_step
    cfg.trainer.is_resume = True
    cfg.trainer.pretrained_checkpoint = None
    cfg.trainer.sync_with_dataloader = False
    cfg.trainer.num_warmup_steps = 0
    cfg.trainer.save_interval = 5000
    cfg.trainer.eval_interval = 2000
    cfg.trainer.learning_rate.base = cosine_base(start, end_step, rates[0], min_lr)
    cfg.trainer.lr_scheduler_type = 'cosine_with_min_lr'
    cfg.trainer.scheduler_specific_kwargs = {'min_lr': min_lr}
    target = Path(cfg.run_root_dir) / run_id
    if target.exists() or target_config.exists():
        raise FileExistsError('Use a fresh run directory and config path')
    new_scheduler = copy.deepcopy(scheduler)
    new_scheduler.update(base_lrs=[cfg.trainer.learning_rate.base], last_epoch=start,
                         _step_count=start+1, _last_lr=rates)
    # DeepSpeed reconstructs its local microstep counter from zero. Checkpoint
    # gradients are not persisted, so Accelerate must start at that boundary.
    old_microstep = random['step']
    random['step'] = 0
    destination = target/f'checkpoints/steps_{start}_training_state'
    destination.mkdir(parents=True)
    for path in state.iterdir():
        if path.name not in ('scheduler.bin', 'random_states_0.pkl'):
            (destination/path.name).symlink_to(path, target_is_directory=path.is_dir())
    torch.save(new_scheduler, destination/'scheduler.bin')
    torch.save(random, destination/'random_states_0.pkl')
    (destination.parent/f'steps_{start}_pytorch_model.pt').symlink_to(
        source/f'checkpoints/steps_{source_step}_pytorch_model.pt')
    OmegaConf.save(cfg, target_config)
    report = dict(source_run=str(source), source_checkpoint_label=source_step,
        actual_start_step=start, target_step=end_step, additional_updates=end_step-start,
        source_accelerator_microstep=old_microstep, resumed_accelerator_microstep=0,
        original_scheduler=scheduler, resumed_scheduler=new_scheduler,
        config=str(target_config), current_lr=rates[0], final_lr=min_lr,
        schedule='cosine over cumulative updates, calibrated to match current LR at resume',
        preserved='source weights, Adam moments, DeepSpeed global step, Torch/NumPy/Python RNG state',
        caveat='New metadata rebases scheduler/microstep counters; data iterator restarts. Keep source run for symlink targets.')
    (target/'continuation_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--target-config', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_run, args.target_config, args.run_id), indent=2))
