"""Exercise the real LIBERO loader/trainer with a small, randomly initialized ACT.

Launch with Accelerate and scripts/multinode_accelerate.yaml. This validates
distributed infrastructure, not pretrained VLA quality or full-model capacity.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn

from starVLA.model.modules.action_model.ACT_ActionHeader import TurboStyleACTActionHead
from starVLA.model.modules.action_model.action_loss import masked_action_l1_loss
from starVLA.training.train_starvla import (
    VLATrainer, build_accelerator, prepare_data, setup_optimizer_and_scheduler,
)


class SmokeACT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.action_model = TurboStyleACTActionHead(
            token_dim=16, hidden_dim=64, action_dim=7, horizon=8,
            num_frames=1, num_visual_tokens=8, num_heads=4, num_layers=1,
            dim_feedforward=128, mlp_hidden_dim=64, dropout=0,
        )

    def actions(self, examples):
        arrays = np.stack([
            np.asarray(view.resize((32, 32)), dtype=np.float32)
            for example in examples for view in example["image"]
        ])
        param = next(self.parameters())
        images = torch.as_tensor(arrays, device=param.device, dtype=param.dtype)
        images = images.permute(0, 3, 1, 2) / 255.0
        features = self.encoder(images).flatten(2).transpose(1, 2)
        tokens = features.reshape(len(examples), 1, 8, 16)
        return self.action_model(tokens)

    def forward(self, examples):
        predictions = self.actions(examples).float()
        target = torch.as_tensor(
            np.stack([item["action"] for item in examples]), device=predictions.device,
        )
        valid = torch.as_tensor(
            np.stack([item["action_valid_mask"] for item in examples]),
            device=predictions.device,
        )
        loss = masked_action_l1_loss(predictions, target, valid)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite action loss")
        return {"action_loss": loss, "l1_action_loss": loss}

    @torch.no_grad()
    def predict_action(self, examples, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = self.actions(examples)
        return {"normalized_actions": result.float().cpu().numpy()}


def weight_hash(model, *, parameters_only=False):
    digest = hashlib.sha256()
    values = model.named_parameters() if parameters_only else model.state_dict().items()
    for name, value in sorted(values):
        digest.update(name.encode())
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-start-step", type=int, default=0)
    parser.add_argument("--data-root", default=".cache/multinode/data")
    parser.add_argument("--model", choices=["smoke", "gawm"], default="smoke")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error("run-id must be a directory name")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["STARVLA_DISABLE_TQDM"] = "1"
    cfg = OmegaConf.load(
        "examples/LIBERO/train_files/starvla_gawm_multinode.yaml" if args.model == "gawm"
        else "examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
    )
    cfg.run_id = args.run_id
    cfg.run_root_dir = ".cache/multinode/runs"
    cfg.output_dir = str(Path(cfg.run_root_dir) / cfg.run_id)
    if args.model == "smoke":
        cfg.framework = {"name": "SmokeACT", "description": "Random tiny CNN + native ACT"}
    data = cfg.datasets.vla_data
    data.data_root_dir = args.data_root
    if args.model == "smoke":
        data.data_mix = "libero_goal"
        data.per_device_batch_size = 2
    data.num_workers = 0
    data.action_valid_mask = True
    trainer_cfg = cfg.trainer
    trainer_cfg.max_train_steps = args.steps
    trainer_cfg.gradient_accumulation_steps = 2
    trainer_cfg.save_interval = 2
    trainer_cfg.eval_interval = 2
    trainer_cfg.logging_frequency = 1
    if args.model == "smoke":
        trainer_cfg.learning_rate = {"base": 0.001}
    trainer_cfg.lr_scheduler_type = "constant"
    trainer_cfg.scheduler_specific_kwargs = {}
    trainer_cfg.num_warmup_steps = 0
    trainer_cfg.freeze_modules = ""
    trainer_cfg.is_resume = args.resume
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accelerator = build_accelerator(cfg)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(42)
    # Verify a meaningful-sized NCCL payload before touching the dataset.
    payload = torch.full((4 * 1024 * 1024,), rank + 1., device=accelerator.device)
    torch.cuda.synchronize()
    start = time.monotonic()
    dist.all_reduce(payload)
    torch.cuda.synchronize()
    collective_seconds = time.monotonic() - start
    assert bool((payload == world * (world + 1) / 2).all())
    del payload
    print(f"NCCL PASS rank={rank}/{world} host={socket.gethostname()}", flush=True)
    loader = prepare_data(cfg, accelerator, output)
    if args.model == "gawm":
        from starVLA.model.framework.base_framework import build_framework
        model = build_framework(cfg)
    else:
        model = SmokeACT()
    optimizer, scheduler = setup_optimizer_and_scheduler(model, cfg)
    trainer = VLATrainer(cfg, model, loader, optimizer, scheduler, accelerator)
    trainer.prepare_training()
    start_step = trainer.completed_steps
    assert start_step == args.expected_start_step, (start_step, args.expected_start_step)
    assert trainer.model.global_steps == start_step
    optimizer_state = trainer.model.optimizer.optimizer.state
    optimizer_steps = [int(state["step"].item()) for state in optimizer_state.values() if "step" in state]
    if args.resume:
        assert trainer.resume_training_state, "Weights-only resume is insufficient"
        assert optimizer_steps and all(step == start_step for step in optimizer_steps), optimizer_steps
    initial_hash = weight_hash(accelerator.unwrap_model(trainer.model))
    unwrapped = accelerator.unwrap_model(trainer.model)
    components = (
        {"encoder": unwrapped.backbone.encoder, "world_model": unwrapped.world_model,
         "text_encoder": unwrapped.task_embedding, "action_heads": unwrapped.action_models}
        if args.model == "gawm" else {"encoder": unwrapped.encoder, "action_head": unwrapped.action_model}
    )
    initial_components = {name: weight_hash(module, parameters_only=True) for name, module in components.items()}
    scheduler = getattr(trainer.lr_scheduler, "scheduler", trainer.lr_scheduler)
    assert scheduler.last_epoch == start_step, (scheduler.last_epoch, start_step)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    trainer.train()
    elapsed = time.monotonic() - start
    assert trainer.completed_steps == args.steps
    assert trainer.model.global_steps == args.steps
    assert scheduler.last_epoch == args.steps
    final_components = {name: weight_hash(module, parameters_only=True) for name, module in components.items()}
    components_updated = {name: value != initial_components[name] for name, value in final_components.items()}
    assert all(components_updated.values()), components_updated
    final_hash = weight_hash(accelerator.unwrap_model(trainer.model))
    assert final_hash != initial_hash, "Optimizer did not update weights"
    hashes = [None] * world
    dist.all_gather_object(hashes, final_hash)
    assert len(set(hashes)) == 1, "Model replicas differ across ranks"
    report = {
        "rank": rank, "world_size": world, "hostname": socket.gethostname(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(accelerator.device),
        "start_step": start_step, "completed_steps": trainer.completed_steps,
        "deepspeed_global_steps": trainer.model.global_steps,
        "optimizer_steps_at_start": optimizer_steps,
        "resume_training_state": trainer.resume_training_state,
        "initial_hash": initial_hash, "final_hash": final_hash,
        "replicas_identical": True, "nccl_16mib_all_reduce_seconds": collective_seconds,
        "training_seconds": elapsed, "global_batch_size": trainer.total_batch_size,
        "framework": cfg.framework.name,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "components_updated": components_updated,
        "scheduler_last_epoch": scheduler.last_epoch,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    phase = "resume" if args.resume else "initial"
    (output / f"report_{phase}_rank{rank}.json").write_text(json.dumps(report, indent=2))
    print("TRAINING PASS " + json.dumps(report), flush=True)
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
