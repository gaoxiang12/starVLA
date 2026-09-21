"""C's training policy, independent of model architecture and robot schema."""
import math
from pathlib import Path

from omegaconf import OmegaConf
import torch


def apply_training_recipe(cfg, overrides=None):
    """Experiment < recipe < training_overrides < command line.

    Managed recipe fields replace old fields, including old per-module LRs.
    Checkpoint readers never call this function; archived configs stay readable.
    """
    cli = OmegaConf.create({}) if overrides is None else overrides
    selected = OmegaConf.select(cli, 'trainer.recipe', default=None)
    if selected is None:
        selected = OmegaConf.select(cfg, 'trainer.recipe', default='c')
    if selected not in ('c', 'legacy'):
        raise ValueError('trainer.recipe must be c or legacy')
    cfg = OmegaConf.merge(cfg)
    if selected == 'c' and not cfg.get('recipe_resolved', False):
        profile = OmegaConf.load(Path(__file__).parents[1]/'config/training/c_recipe.yaml')
        # Fill general fields, then replace managed subtrees instead of retaining
        # old action/encoder learning rates inside learning_rate.
        cfg = OmegaConf.merge(profile, cfg)
        for key, value in profile.trainer.items():
            cfg.trainer[key] = value
        for key, value in profile.datasets.vla_data.items():
            cfg.datasets.vla_data[key] = value
        if str(OmegaConf.select(cfg, 'framework.name', default='')).startswith('GAWM'):
            # Encoder freezing is part of C's training policy; never apply the
            # new policy LR to an old config's trainable DINO by accident.
            if OmegaConf.select(cfg, 'framework.world_model') is not None:
                cfg.framework.world_model.train_encoder = False
        cfg = OmegaConf.merge(cfg, cfg.get('training_overrides', {}))
    cfg = OmegaConf.merge(cfg, cli)
    cfg.trainer.recipe = selected
    cfg.recipe_resolved = True
    return cfg


def resolve_training_budget(cfg, dataset, global_batch):
    """Record actual frame counts and translate C's epochs into optimizer steps."""
    if cfg.trainer.get('recipe') != 'c':
        return
    children = getattr(dataset, 'datasets', [dataset])
    frames = sum(len(child) for child in children)
    steps = frames // global_batch
    if not steps:
        raise ValueError('Dataset is smaller than one global optimizer batch')
    epochs = list(cfg.trainer.stage_epochs)
    if len(epochs) != 2 or any(int(x) != x or x <= 0 for x in epochs):
        raise ValueError('C stage_epochs must contain two positive integer epoch counts')
    cfg.trainer.frames_per_epoch = frames
    cfg.trainer.steps_per_epoch = steps
    cfg.trainer.stage1_steps = int(epochs[0]) * steps
    if cfg.trainer.max_train_steps is None:
        cfg.trainer.max_train_steps = int(sum(epochs)) * steps
    if cfg.trainer.lr_scheduler_total_steps is None:
        cfg.trainer.lr_scheduler_total_steps = int(cfg.trainer.scheduler_epochs) * steps
    if int(cfg.trainer.lr_scheduler_total_steps) < max(int(x)*steps for x in epochs):
        raise ValueError('Scheduler period must cover each training stage')
    if int(cfg.trainer.num_warmup_steps) != 0:
        raise ValueError('C has no warmup; use recipe=legacy for other schedules')


def prepare_parameter_precision(model, cfg):
    """Cast learned weights only; preserve frozen assets and FP32 buffers."""
    dtype = cfg.trainer.get('parameter_dtype')
    if dtype is None:
        return
    if dtype not in ('float32', 'bfloat16'):
        raise ValueError('parameter_dtype must be float32 or bfloat16')
    for param in model.parameters():
        if param.requires_grad and param.is_floating_point():
            param.data = param.data.to(getattr(torch, dtype))


def resume_contract(cfg):
    """Stable settings needed for a continuation, excluding run paths/stop step."""
    cfg = cfg.unwrap() if hasattr(cfg, 'unwrap') else cfg
    fields = ('seed', 'framework', 'datasets.vla_data', 'trainer.learning_rate',
              'trainer.optimizer', 'trainer.parameter_dtype', 'trainer.stage2_lr_scale',
              'trainer.distributed_backend', 'trainer.ddp_kwargs', 'trainer.gradient_accumulation_steps')
    result = {}
    for key in fields:
        value = OmegaConf.select(cfg, key)
        result[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return result


def c_lr_multiplier(step, *, stage1_steps, period, minimum_ratio, stage2_scale):
    stage2 = step >= stage1_steps
    within = step-stage1_steps if stage2 else step
    factor = minimum_ratio + (1-minimum_ratio)*(1+math.cos(math.pi*min(within, period)/period))/2
    return factor * (stage2_scale if stage2 else 1.0)


def reset_stage_optimizer_if_needed(optimizer, completed_steps, stage1_steps):
    """Keep the prepared optimizer object, but start stage 2 with fresh Adam state."""
    if completed_steps == stage1_steps:
        optimizer.state.clear()
        return True
    return False
