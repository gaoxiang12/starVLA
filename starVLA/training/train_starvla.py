# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA defaults to the C recipe on PyTorch + Accelerate/DDP. DeepSpeed is opt-in.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from functools import partial
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed, DataLoaderConfiguration, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args
from starVLA.training.recipe import (apply_training_recipe, resolve_training_budget,
    prepare_parameter_precision, c_lr_multiplier, reset_stage_optimizer_if_needed, resume_contract)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def build_accelerator(cfg) -> Accelerator:
    """Construct Accelerate after loading the configured accumulation factor."""
    gradient_accumulation_steps = int(
        getattr(cfg.trainer, "gradient_accumulation_steps", 1)
    )
    if gradient_accumulation_steps < 1:
        raise ValueError("trainer.gradient_accumulation_steps must be positive")

    # The trainer owns scheduler stepping below. Disabling Accelerate's
    # automatic coupling prevents AcceleratedScheduler from stepping once per
    # process.
    backend = getattr(cfg.trainer, 'distributed_backend',
                      'deepspeed' if getattr(cfg.trainer, 'recipe', None) == 'legacy' else 'ddp')
    if backend not in ('deepspeed', 'ddp'):
        raise ValueError('trainer.distributed_backend must be deepspeed or ddp')
    if backend == 'ddp':
        # Old accelerate launch YAMLs may export this flag. An explicit DDP
        # choice must not be silently replaced by the launcher's DeepSpeed.
        os.environ['ACCELERATE_USE_DEEPSPEED'] = 'false'
    return Accelerator(
        mixed_precision=getattr(cfg.trainer, 'mixed_precision', 'bf16'),
        deepspeed_plugin=DeepSpeedPlugin() if backend == 'deepspeed' else None,
        kwargs_handlers=([DistributedDataParallelKwargs(**dict(cfg.trainer.ddp_kwargs))]
            if backend == 'ddp' and getattr(cfg.trainer, 'ddp_kwargs', None) else []),
        dataloader_config=DataLoaderConfiguration(even_batches=(
            getattr(getattr(cfg, 'datasets', None), 'vla_data', {}).get('sampling_mode') not in ('frame_epoch', 'auto'))),
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=gradient_accumulation_steps,
            # DeepSpeed ZeRO-2 partitions gradients and rejects no_sync().
            # It performs accumulation internally, so keep synchronization
            # enabled for every micro-batch.
            sync_each_batch=True,
            # Step-based runs can span loader epochs. Opt out of flushing a
            # partial accumulation at epoch end when DeepSpeed owns boundaries.
            sync_with_dataloader=bool(getattr(cfg.trainer, "sync_with_dataloader", True)),
        ),
        step_scheduler_with_optimizer=False,
    )


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if (cfg.trainer.get('recipe') == 'c' and not cfg.trainer.is_resume
            and (output_dir / 'config.full.yaml').exists()):
        raise ValueError(f'Existing run at {output_dir}; use a new run_id or trainer.is_resume=true')

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    accelerator.wait_for_everyone()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=bool(cfg.trainer.optimizer.get('fused', False)),
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    if cfg.trainer.get('recipe') == 'c':
        minimum = float(cfg.trainer.scheduler_specific_kwargs.min_lr)
        base = float(cfg.trainer.learning_rate.base)
        if not 0 <= minimum <= base or not 0 < float(cfg.trainer.stage2_lr_scale) <= 1:
            raise ValueError('Invalid C minimum LR or stage2_lr_scale')
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, partial(
            c_lr_multiplier, stage1_steps=int(cfg.trainer.stage1_steps),
            period=int(cfg.trainer.lr_scheduler_total_steps), minimum_ratio=minimum/base,
            stage2_scale=float(cfg.trainer.stage2_lr_scale)))
        return optimizer, lr_scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.get('lr_scheduler_total_steps', cfg.trainer.max_train_steps),
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        expected_batch = cfg.trainer.get('expected_global_batch_size')
        if expected_batch is not None and self.total_batch_size != int(expected_batch):
            raise ValueError(f'Global batch is {self.total_batch_size}, expected {expected_batch}; '
                             'adjust per_device_batch_size or gradient_accumulation_steps')

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        resuming_c = self.config.trainer.get('recipe') == 'c' and self.config.trainer.is_resume
        if not resuming_c:
            self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader, self.lr_scheduler = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.lr_scheduler,
        )

        if self.resume_training_state:
            self._load_checkpoint(self.resume_training_state)
            self._repair_lr_scheduler_after_resume()

        if resuming_c:
            self._save_initial_configs()

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases (best-effort; must not block training)."""
        self._wandb_enabled = False
        if os.environ.get("WANDB_MODE") == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            self.accelerator.wait_for_everyone()
            return
        if self.accelerator.is_main_process:
            try:
                wandb.init(
                    name=self.config.run_id,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                )
                self._wandb_enabled = True
            except Exception as exc:
                logger.warning(f"W&B init failed; continuing without W&B: {exc}")
                self._wandb_enabled = False
        # Rendezvous after rank-0 W&B init. Otherwise a slow or failing init on
        # rank 0 lets the other ranks reach the first collective alone and
        # eventually hit an NCCL watchdog timeout.
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        self.resume_training_state = None

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if getattr(self.config.trainer, 'recipe', None) == 'c':
                complete = list(Path(self.checkpoint_dir).glob('steps_*_training_state/complete.json'))
                if not complete:
                    raise ValueError('No complete C training state available to resume')
                latest = max(complete, key=lambda p: int(p.parent.name.split('_')[1]))
                meta = json.loads(latest.read_text())
                if meta['world_size'] != self.accelerator.num_processes or meta['global_batch_size'] != self.total_batch_size:
                    raise ValueError('C resume must preserve world size and global batch size')
                if meta.get('contract') != resume_contract(self.config):
                    raise ValueError('C resume changed model, data or optimizer configuration; use a new run')
                for key in ('frames_per_epoch', 'steps_per_epoch', 'stage1_steps', 'lr_scheduler_total_steps'):
                    if meta.get(key) != self.config.trainer.get(key):
                        raise ValueError(f'C resume changed {key}; use a new run for changed training budgets')
                self.completed_steps = meta['step']
                resume_from_checkpoint = str(latest.parent)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                training_state = os.path.join(
                    self.checkpoint_dir, f"steps_{self.completed_steps}_training_state"
                )
                if os.path.isdir(training_state):
                    self.resume_training_state = training_state
                else:
                    if not getattr(self.config.trainer, 'allow_weights_only_resume', True):
                        raise ValueError('C resume requires full optimizer/scheduler/RNG state; '
                                         'use is_resume=false for a weights-only warm start')
                    self.model = self.load_pretrained_backbones(
                        self.model, self.resume_from_checkpoint, reload_modules=None
                    )
                    logger.warning(
                        "No full training state found for step %s; falling back to weights-only resume",
                        self.completed_steps,
                    )
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Advance the scheduler only for legacy weights-only resumes."""
        if self.completed_steps > 0 and not self.resume_training_state:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _repair_lr_scheduler_after_resume(self):
        """Rewind scheduler states saved with Accelerate's per-process stepping."""
        repair_scheduler = bool(
            getattr(self.config.trainer, "repair_lr_scheduler_on_resume", False)
        )
        if not repair_scheduler:
            return

        scheduler = getattr(self.lr_scheduler, "scheduler", self.lr_scheduler)
        scheduler.step(self.completed_steps)
        if hasattr(scheduler, "_step_count"):
            scheduler._step_count = self.completed_steps + 1
        logger.warning(
            "Repaired LR scheduler to external step %s; current LR: %s",
            self.completed_steps,
            scheduler.get_last_lr(),
        )

    def _save_checkpoint(self):
        """Save current training state."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()
        training_state_path = checkpoint_path + "_training_state"
        self.accelerator.save_state(training_state_path)
        self.accelerator.wait_for_everyone()
        if self.config.trainer.get('collect_rng_states', False):
            # DDP model/optimizer states are replicated, but RNG files are per
            # rank. Consolidate them before marking a node-local save complete.
            rank = self.accelerator.process_index
            local_rng = Path(training_state_path) / f'random_states_{rank}.pkl'
            gathered = [None] * self.accelerator.num_processes if rank == 0 else None
            if dist.is_initialized():
                dist.gather_object(local_rng.read_bytes(), gathered, dst=0)
            else:
                gathered = [local_rng.read_bytes()]
            if rank == 0:
                for saved_rank, payload in enumerate(gathered):
                    (Path(training_state_path) / f'random_states_{saved_rank}.pkl').write_bytes(payload)
            self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process and getattr(self.config.trainer, 'recipe', None) == 'c':
            meta = dict(step=self.completed_steps, world_size=self.accelerator.num_processes,
                        global_batch_size=self.total_batch_size, contract=resume_contract(self.config))
            for key in ('frames_per_epoch', 'steps_per_epoch', 'stage1_steps', 'lr_scheduler_total_steps'):
                meta[key] = self.config.trainer.get(key)
            marker = Path(training_state_path) / 'complete.json'
            temporary = marker.with_suffix('.tmp')
            temporary.write_text(json.dumps(meta, indent=2)+'\n')
            temporary.replace(marker)
            keep = int(self.config.trainer.keep_last_checkpoints)
            if keep < 1:
                raise ValueError('keep_last_checkpoints must be positive')
            markers = sorted(Path(self.checkpoint_dir).glob('steps_*_training_state/complete.json'),
                             key=lambda p: int(p.parent.name.split('_')[1]))
            milestones = set(self.config.trainer.milestone_steps) | {self.config.trainer.stage1_steps}
            for old in markers[:-keep]:
                step = int(old.parent.name.split('_')[1])
                if step not in milestones:
                    shutil.rmtree(old.parent)
                    for suffix in ('_pytorch_model.pt', '_model.safetensors'):
                        (Path(self.checkpoint_dir)/f'steps_{step}{suffix}').unlink(missing_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.print(f"✅ Full training state saved at {training_state_path}")

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if (self.config.trainer.get('recipe') == 'c'
                and self.completed_steps % self.config.trainer.logging_frequency == 0
                and dist.is_initialized()):
            # Different ranks can train different robot heads. Gather named
            # scalars so per-robot metrics aren't averaged with absent values.
            records = [None] * self.accelerator.num_processes
            dist.all_gather_object(records, metrics)
            metrics = {key: float(np.mean([r[key] for r in records if key in r]))
                       for key in set().union(*(r.keys() for r in records))}
            if any(not np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError('Non-finite metrics on a training rank')
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and self.accelerator.is_main_process:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            updates_per_epoch = self.config.trainer.get('steps_per_epoch') or (
                len(self.vla_train_dataloader) / self.accelerator.gradient_accumulation_steps)
            metrics["epoch"] = round(self.completed_steps / updates_per_epoch, 4)
            metrics['global_batch_size'] = self.total_batch_size
            metrics['anchor_draws'] = self.completed_steps * self.total_batch_size
            # Keep a local, dependency-free metric history even when W&B is
            # disabled or misconfigured. This is especially important for
            # staged runs whose checkpoint should be gated on loss quality.
            local_record = {"step": self.completed_steps, **metrics}
            non_finite = {
                key: value
                for key, value in local_record.items()
                if isinstance(value, (float, np.floating))
                and not np.isfinite(value)
            }
            if non_finite:
                raise FloatingPointError(
                    f"Non-finite metrics at step {self.completed_steps}: {non_finite}"
                )
            metrics_path = os.path.join(self.config.output_dir, "metrics.jsonl")
            with open(metrics_path, "a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(local_record, allow_nan=False) + "\n")
            if getattr(self, "_wandb_enabled", False):
                try:
                    wandb.log(metrics, step=self.completed_steps)
                except Exception as exc:
                    self._wandb_enabled = False
                    logger.warning(f"W&B log failed; disabling W&B: {exc}")
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        loader = self.vla_train_dataloader
        if self.config.datasets.vla_data.get('sampling_mode') == 'frame_epoch':
            microbatches_per_epoch = len(loader)
            accumulation = self.accelerator.gradient_accumulation_steps
            if microbatches_per_epoch % accumulation:
                raise ValueError('frame_epoch must contain complete optimizer updates on every rank')
            steps_per_epoch = microbatches_per_epoch // accumulation
            self.vla_epoch_count, step_in_epoch = divmod(self.completed_steps, steps_per_epoch)
            loader.set_epoch(self.vla_epoch_count)
            if step_in_epoch:
                loader = self.accelerator.skip_first_batches(loader, step_in_epoch * accumulation)
                loader.set_epoch(self.vla_epoch_count)
        self.vla_iter = iter(loader)

    def close_dataloader(self):
        """Stop prefetch workers while distributed/CUDA contexts are still alive.

        Leaving persistent workers to Python finalizers after NCCL teardown can
        strand a rank during short runs or at a two-stage training boundary.
        """
        iterator = getattr(self, 'vla_iter', None)
        close = getattr(iterator, 'close', None)
        if callable(close):
            close()
        loader = self.vla_train_dataloader
        base_loader = getattr(loader, 'base_dataloader', loader)
        workers = getattr(base_loader, '_iterator', None)
        shutdown = getattr(workers, '_shutdown_workers', None)
        if callable(shutdown):
            shutdown()
            base_loader._iterator = None
        self.vla_iter = None

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            if self.config.datasets.vla_data.get('sampling_mode') == 'frame_epoch':
                self.vla_epoch_count += 1
                self.vla_train_dataloader.set_epoch(self.vla_epoch_count)
                self.vla_iter = iter(self.vla_train_dataloader)
            else:
                self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                    self.vla_train_dataloader, self.vla_epoch_count
                )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=(not self.accelerator.is_local_main_process)
            or os.environ.get("STARVLA_DISABLE_TQDM", "0") == "1",
            mininterval=float(os.environ.get("STARVLA_TQDM_MININTERVAL", "5.0")),
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            if self.config.trainer.get('recipe') == 'c':
                reset_stage_optimizer_if_needed(self.optimizer, self.completed_steps,
                                                self.config.trainer.stage1_steps)
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # DeepSpeed receives every micro-batch, but optimizer-step
            # counters, evaluation, metrics, and checkpoints must advance only
            # at the configured accumulation boundary.
            if not self.accelerator.sync_gradients:
                continue

            progress_bar.update(1)
            self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.config.trainer.eval_interval > 0 and self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            milestone = (self.config.trainer.get('recipe') == 'c' and self.completed_steps in
                         set(self.config.trainer.milestone_steps) | {self.config.trainer.stage1_steps})
            if (self.completed_steps % self.config.trainer.save_interval == 0 or milestone) and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Evaluate without dropout; optionally use disjoint per-task episodes."""
        from starVLA.training.trainer_utils.action_validation import (
            HeldOutActionEvaluator,
            evaluate_action_batch,
        )
        step_metrics = {} if step_metrics is None else step_metrics
        held_out = int(self.config.datasets.vla_data.get("validation_episode_stride", 0)) > 0
        if held_out:
            # Dataset construction synchronizes metadata caches across ranks.
            # Every rank must participate, even though only rank 0 evaluates.
            if not hasattr(self, "action_evaluator"):
                self.action_evaluator = HeldOutActionEvaluator(self.config)
            if self.accelerator.is_main_process:
                scores = self.action_evaluator.evaluate(
                    self.accelerator.unwrap_model(self.model), self.completed_steps
                )
                step_metrics.update(scores)
                step_metrics["mse_score"] = scores["validation/mse_score"]
        else:
            if self.config.datasets.vla_data.get('sampling_mode') == 'frame_epoch':
                raise ValueError('Frame-epoch validation requires a separate held-out split; '
                                 'it must not consume training frames')
            examples = self._get_next_batch()
            scores = evaluate_action_batch(self.accelerator.unwrap_model(self.model), examples)
            if self.accelerator.is_main_process:
                step_metrics["mse_score"] = scores["mse_score"]
        self.accelerator.wait_for_everyone()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                if self.config.framework.name in {"GAWM", "GAWM-L", "GAWMObjectFusion", "GAWMCartesian", "GAWMCompactExpert"}:
                    output_dict = self.model.forward(batch_vla, optimizer_step=self.completed_steps)
                else:
                    output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            self.accelerator.backward(total_loss)

            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
            # AcceleratedOptimizer only clears on a synchronization boundary.
            # Clearing before backward would discard previous microbatch grads
            # precisely on the final microbatch of an accumulated update.
            self.optimizer.zero_grad()

        # Historical action_dit_loss includes auxiliary objectives in GAWM.
        # Keep the alias for dashboards; use total_loss and l1_action_loss to
        # distinguish the optimization objective from ACT prediction error.
        step_log = {"total_loss": total_loss.item(), "action_dit_loss": action_loss.item()}
        # Surface any auxiliary scalar losses the framework reports.
        for k in (
            "l1_action_loss",
            "continuous_action_l1",
            "gripper_action_l1",
            "gripper_action_accuracy",
            "gripper_position_accuracy",
            "first_action_l1",
            "valid_action_fraction",
            "latent_loss",
            "latent_cosine_loss",
            "delta_scale",
            "delta_target_rms",
            "delta_pred_rms",
            "delta_copy_mse",
            "delta_pred_mse",
            "delta_mean_baseline_mse",
            "delta_to_copy_ratio",
            "delta_direction_cosine",
            "visual_token_diversity_loss",
            "visual_token_variance_loss",
            "visual_token_mean_cosine",
            "visual_content_rms",
            "visual_tokens_rms",
            "latent_mse_over_delta_scale_sq",
        ):
            v = output_dict.get(k) if isinstance(output_dict, dict) else None
            if torch.is_tensor(v):
                step_log[k] = v.item()
        if isinstance(output_dict, dict):
            for k, v in output_dict.items():
                if (
                    k.startswith(("dino_", "temporal_", "predicted_latent_", "latent_batch_", "latent_loss_horizon_", "spatial_", "object_", "tcp_", "contact_", "cartesian_", "compact_"))
                    or k == "gripper_transition_l1"
                ) and torch.is_tensor(v):
                    step_log[k] = v.item()
        robot_tags = {
            str(example.get("robot_tag"))
            for example in batch_vla
            if example.get("robot_tag") is not None
        }
        if len(robot_tags) == 1:
            robot_tag = next(iter(robot_tags))
            for metric_name in (
                "total_loss",
                "action_dit_loss",
                "l1_action_loss",
                "continuous_action_l1",
                "gripper_action_l1",
                "gripper_action_accuracy",
                "gripper_position_accuracy",
                "first_action_l1",
                "valid_action_fraction",
                "latent_loss",
                "latent_cosine_loss",
                "delta_to_copy_ratio",
                "delta_direction_cosine",
                "visual_token_mean_cosine",
                "tcp_position_loss_m",
                "tcp_contact_error_mm",
                "gripper_transition_l1",
                "contact_fraction",
                "contact_objective_loss",
                "cartesian_target_error_mm",
                "cartesian_coarse_target_error_mm",
                "cartesian_actual_target_error_mm",
                "cartesian_rotation_error_deg",
                "cartesian_ik_residual_mm",
                "cartesian_ik_converged_fraction",
                "cartesian_geometry_loss",
                "cartesian_action_correction_l1",
                "compact_flow_velocity_loss",
                "compact_regression_loss",
                "compact_context_tokens",
            ):
                if metric_name in step_log:
                    step_log[f"{metric_name}/{robot_tag}"] = step_log[metric_name]
        return step_log

    def _finalize_training(self):
        """Training end processing."""
        if self.completed_steps % self.config.trainer.save_interval:
            self._save_checkpoint()
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    cfg = apply_training_recipe(cfg)
    accelerator = build_accelerator(cfg)
    global_batch = (cfg.datasets.vla_data.per_device_batch_size * accelerator.num_processes
                    * accelerator.gradient_accumulation_steps)
    expected = cfg.trainer.get('expected_global_batch_size')
    if expected is not None and global_batch != int(expected):
        raise ValueError(f'Global batch is {global_batch}, expected {expected}; '
                         'adjust per_device_batch_size or gradient_accumulation_steps')
    accelerator.print(accelerator.state)
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # Model construction initializes experiment-specific branches (for
    # example the predictable-innovation basis). Seed before construction so
    # the YAML seed governs those parameters, not only the later train loop.
    rank = dist.get_rank() if dist.is_initialized() else 0
    construction_seed = cfg.get('seed', 42) + (rank if cfg.trainer.recipe == 'legacy' else 0)
    set_seed(construction_seed)

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla = TrainerUtils.freeze_backbones(vla, cfg.trainer.get('freeze_modules', ''))
    prepare_parameter_precision(vla, cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    resolve_training_budget(cfg, vla_train_dataloader.dataset, global_batch)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    try:
        trainer.train()
    finally:
        trainer.close_dataloader()

    logger.info("... and that's all, folks!")
    accelerator.wait_for_everyone()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = apply_training_recipe(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    main(cfg)
