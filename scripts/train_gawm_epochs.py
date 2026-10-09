"""GAWM training with complete, shuffled LIBERO passes and epoch accounting."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from accelerate import skip_first_batches
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Sampler, Subset

from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.train_starvla import (
    VLATrainer, build_accelerator, setup_optimizer_and_scheduler,
)


class EpochPermutationSampler(Sampler):
    """Every frame once per epoch before Accelerate pads the final global batch."""

    def __init__(self, dataset, seed=42):
        self.size = len(dataset)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())


def exclude_trajectories(dataset, episode_ids):
    """Keep every indexed frame except explicitly listed damaged episodes."""
    excluded = set(episode_ids)
    if not excluded:
        return dataset
    existing = {int(episode) for episode, _ in dataset.all_steps}
    if excluded - existing:
        raise ValueError(f"Unknown excluded episodes: {excluded - existing}")
    keep = [index for index, (episode, _) in enumerate(dataset.all_steps) if int(episode) not in excluded]
    return Subset(dataset, keep)


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class EpochTrainer(VLATrainer):
    def _start_epoch(self, epoch, offset=0):
        self.vla_epoch_count = epoch
        self.epoch_sampler.set_epoch(epoch)
        self.vla_train_dataloader.set_epoch(epoch)
        loader = self.vla_train_dataloader
        if offset:
            loader = skip_first_batches(loader, num_batches=offset)
            loader.set_epoch(epoch)
        # Retain the temporary resume loader until its iterator is exhausted.
        self.active_epoch_loader = loader
        return iter(loader)

    def _create_data_iterators(self):
        epoch, offset = divmod(self.completed_steps, self.steps_per_epoch)
        self.vla_iter = self._start_epoch(epoch, offset=offset)

    def _get_next_batch(self):
        try:
            return next(self.vla_iter)
        except StopIteration:
            self.vla_iter = self._start_epoch(self.vla_epoch_count + 1)
            return next(self.vla_iter)

    def _save_checkpoint(self):
        super()._save_checkpoint()
        # Written only after all nodes finished writing their local shards.
        if self.accelerator.is_main_process:
            marker = Path(self.config.output_dir) / f"checkpoint_ready_{self.completed_steps}.json"
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps({"step": self.completed_steps}))
            temporary.replace(marker)

    def _finalize_training(self):
        if self.completed_steps % self.config.trainer.save_interval:
            self._save_checkpoint()
        super()._finalize_training()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume-step", type=int, default=0)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_yaml)
    cfg.run_id = args.run_id
    cfg.trainer.is_resume = bool(args.resume_step)
    cfg.output_dir = str(Path(cfg.run_root_dir) / args.run_id)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accelerator = build_accelerator(cfg)
    accelerator.dataloader_config.dispatch_batches = False
    accelerator.dataloader_config.even_batches = True
    if cfg.trainer.gradient_accumulation_steps != 1:
        raise ValueError("Epoch trainer requires accumulation=1 for exact epoch accounting")
    set_seed(cfg.seed)
    mixture = get_vla_dataset(
        cfg.datasets.vla_data, balance_dataset_weights=True,
        balance_trajectory_weights=True, seed=cfg.seed,
    )
    # Use real indexed frames, not the mixture's with-replacement __getitem__.
    exclusions_path = cfg.datasets.vla_data.get("episode_exclusions_file")
    exclusions = json.loads(Path(exclusions_path).read_text())["excluded_episodes"] if exclusions_path else {}
    unknown_datasets = set(exclusions) - {d.dataset_name for d in mixture.datasets}
    if unknown_datasets:
        raise ValueError(f"Unknown datasets in exclusions: {unknown_datasets}")
    epoch_datasets = [exclude_trajectories(d, exclusions.get(d.dataset_name, [])) for d in mixture.datasets]
    dataset = ConcatDataset(epoch_datasets)
    expected_frames = int(cfg.datasets.vla_data.expected_frames)
    if len(dataset) != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, found {len(dataset)}")
    sampler = EpochPermutationSampler(dataset, cfg.seed)
    batch = int(cfg.datasets.vla_data.per_device_batch_size)
    steps_per_epoch = math.ceil(len(dataset) / (batch * accelerator.num_processes))
    total_steps = steps_per_epoch * int(cfg.trainer.num_train_epochs)
    cfg.trainer.max_train_steps = args.smoke_steps or total_steps
    cfg.trainer.eval_interval = cfg.trainer.max_train_steps + 1
    cfg.trainer.num_warmup_steps = (
        0 if args.smoke_steps else max(1, round(total_steps * cfg.trainer.warmup_ratio))
    )
    if args.smoke_steps:
        cfg.trainer.logging_frequency = 1
        cfg.trainer.save_interval = args.smoke_steps
    loader = DataLoader(
        dataset, batch_size=batch, sampler=sampler, collate_fn=collate_fn,
        num_workers=cfg.datasets.vla_data.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    if accelerator.is_main_process:
        mixture.save_dataset_statistics(output / "dataset_statistics.json")
        manifest = {
            "datasets": [
                {"name": d.dataset_name, "original_frames": len(d), "frames": len(subset)}
                for d, subset in zip(mixture.datasets, epoch_datasets)
            ],
            "excluded_episodes": exclusions,
            "unique_frames_per_epoch": len(dataset), "epochs": cfg.trainer.num_train_epochs,
            "world_size": accelerator.num_processes, "per_device_batch": batch,
            "global_batch": batch * accelerator.num_processes,
            "steps_per_epoch": steps_per_epoch, "max_train_steps": cfg.trainer.max_train_steps,
            "padding_frames_per_epoch": steps_per_epoch * batch * accelerator.num_processes - len(dataset),
            "sampling": "global permutation without replacement; final batch padded",
            "initialization": "full_checkpoint" if args.resume_step else "random",
            "resume_step": args.resume_step, "validation_only": bool(args.smoke_steps),
        }
        (output / "training_plan.json").write_text(json.dumps(manifest, indent=2))
        print("TRAINING PLAN " + json.dumps(manifest), flush=True)
    accelerator.wait_for_everyone()
    model = build_framework(cfg)
    optimizer, scheduler = setup_optimizer_and_scheduler(model, cfg)
    trainer = EpochTrainer(cfg, model, loader, optimizer, scheduler, accelerator)
    trainer.epoch_sampler = sampler
    trainer.steps_per_epoch = steps_per_epoch
    trainer.prepare_training()
    if args.resume_step:
        assert trainer.resume_training_state, "Refusing weights-only resume"
        assert trainer.completed_steps == trainer.model.global_steps == args.resume_step
        scheduler_state = getattr(trainer.lr_scheduler, "scheduler", trainer.lr_scheduler)
        assert scheduler_state.last_epoch == args.resume_step
        state = trainer.model.optimizer.optimizer.state
        optimizer_steps = [int(s["step"].item()) for s in state.values() if "step" in s]
        assert optimizer_steps and all(step == args.resume_step for step in optimizer_steps)
        restored_hash = state_digest(accelerator.unwrap_model(trainer.model).state_dict())
        restored_hashes = [None] * accelerator.num_processes
        dist.all_gather_object(restored_hashes, restored_hash)
        assert len(set(restored_hashes)) == 1, "Restored model states differ across ranks"
        if accelerator.is_main_process:
            saved = torch.load(trainer.resume_from_checkpoint, map_location="cpu", weights_only=True)
            assert state_digest(saved) == restored_hash, "Restored weights differ from checkpoint"
            del saved
        accelerator.wait_for_everyone()
        resume_report = {
            "rank": accelerator.process_index, "step": trainer.completed_steps,
            "optimizer_steps": optimizer_steps, "scheduler_step": scheduler_state.last_epoch,
            "data_epoch": trainer.completed_steps // steps_per_epoch,
            "skip_batches": trainer.completed_steps % steps_per_epoch,
            "restored_model_hash": restored_hash,
        }
        (output / f"resume_rank_{accelerator.process_index}.json").write_text(json.dumps(resume_report))
        print("FULL RESUME VERIFIED " + json.dumps(resume_report), flush=True)
    assert len(trainer.vla_train_dataloader) == steps_per_epoch
    trainer.train()
    assert trainer.completed_steps == cfg.trainer.max_train_steps
    digest = state_digest(accelerator.unwrap_model(trainer.model).state_dict())
    hashes = [None] * accelerator.num_processes
    dist.all_gather_object(hashes, digest)
    assert len(set(hashes)) == 1, "Final GAWM states differ across ranks"
    if accelerator.is_main_process:
        (output / "training_complete.json").write_text(json.dumps({
            "completed_steps": trainer.completed_steps, "model_hash": hashes[0],
            "ranks": len(hashes), "replicas_identical": True,
        }, indent=2))
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
