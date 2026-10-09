"""Reproducible LIBERO predictor adaptation in a fixed checkpoint feature space.

Cache once, then compare matched predictor-only and state-conditioned fits.
Episode partitions are disjoint for adaptation; the source checkpoint may have
seen all episodes. This is an offline diagnostic, not a rollout/generalization SR.
"""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from starVLA.model.framework.WM4A.GAWM import GAWM
from starVLA.model.modules.action_model.action_loss import masked_action_l1_loss


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_partition(all_steps, excluded, seed, counts):
    """Partition whole trajectories before choosing frames, without replacement."""
    rng = np.random.default_rng(seed)
    episodes = sorted({int(ep) for ep, _ in all_steps} - set(excluded))
    if len(episodes) < 3:
        raise ValueError("At least three undamaged episodes are required")
    episodes = rng.permutation(episodes)
    n_eval = max(1, int(len(episodes) * 0.15))
    groups = (episodes[2 * n_eval:], episodes[:n_eval], episodes[n_eval:2 * n_eval])
    result = {}
    for split, group, count in zip(("train", "val", "test"), groups, counts):
        allowed = set(map(int, group))
        candidates = [i for i, (ep, _) in enumerate(all_steps) if int(ep) in allowed]
        chosen = sorted(map(int, rng.choice(candidates, min(count, len(candidates)), replace=False)))
        result[split] = {"episodes": sorted(allowed), "indices": chosen}
    return result


def load_model(args, state_condition=False):
    config = OmegaConf.load(args.config)
    config.framework.world_model.train_encoder = False
    config.framework.world_model.freeze_visual_token_pooler = True
    config.framework.world_model.condition_world_model_on_state = state_condition
    config.framework.world_model.state_conditioning_mode = getattr(args, "state_conditioning", "goal")
    model = GAWM(config)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(model.remap_checkpoint_state_dict(weights), strict=False)
    allowed = {k for k in model.state_dict() if k.startswith("world_model_state_encoders.")}
    if set(missing) - allowed or unexpected:
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.requires_grad_(False).eval().to(args.device)
    return model


def cache(args):
    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_model(args)
    config = model.config
    mixture = get_vla_dataset(config.datasets.vla_data, seed=args.seed)
    exclusion_file = config.datasets.vla_data.get("episode_exclusions_file")
    exclusions = json.loads(Path(exclusion_file).read_text())["excluded_episodes"] if exclusion_file else {}
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_hash(args.checkpoint),
        "config": OmegaConf.to_container(config, resolve=True),
        "seed": args.seed, "datasets": {},
        "limitation": "Source checkpoint trained on these episodes; splits are held out only from adaptation.",
    }
    for dataset in mixture.datasets:
        name = dataset.dataset_name
        partitions = episode_partition(dataset.all_steps, exclusions.get(name, []), args.seed,
                                       (args.train_samples, args.eval_samples, args.eval_samples))
        manifest["datasets"][name] = partitions
        for split, partition in partitions.items():
            loader = DataLoader(Subset(dataset, partition["indices"]), batch_size=args.batch_size,
                                num_workers=args.workers, collate_fn=collate_fn, shuffle=False)
            chunks = {}
            with torch.inference_mode():
                for batch_index, examples in enumerate(loader):
                    model._validate_future_time_offsets(examples)
                    tag = model._resolve_batch_embodiment(examples)
                    if tag != "franka":
                        raise ValueError("This experiment currently supports LIBERO/franka only")
                    frames = [[ex["image"], *ex["future_images"]] for ex in examples]
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                        patches = model.backbone.encode_patch_frames(frames)
                    views = model._view_valid_mask_tensor(examples, patches.device)
                    latent = model.visual_token_pooler(patches.float(), view_valid_mask=views)
                    goal = model._condition_task_on_embodiment(
                        model._embed_task([ex["lang"] for ex in examples], latent.device), tag)
                    _, horizon, state_dim = model._action_runtime(tag)
                    mask = model._wm_loss_mask_tensor(examples, views, latent.device)
                    action_mask = model._action_valid_mask_tensor(examples, latent.device, horizon)
                    values = {
                        "latent": latent, "goal": goal,
                        "state": model._current_state_tensor(examples, latent.device, state_dim),
                        "mask": mask if mask is not None else torch.ones(latent.shape[:3], dtype=torch.bool, device=latent.device),
                        "action": torch.as_tensor(np.asarray([ex["action"] for ex in examples])),
                        "action_mask": action_mask if action_mask is not None else torch.ones(len(examples), horizon, dtype=torch.bool),
                    }
                    for key, value in values.items():
                        chunks.setdefault(key, []).append(value.detach().cpu())
                    if batch_index % 25 == 0:
                        print(f"cache {name} {split}: {min((batch_index + 1) * args.batch_size, len(loader.dataset))}/{len(loader.dataset)}", flush=True)
            torch.save({key: torch.cat(value) for key, value in chunks.items()}, output / f"{name}.{split}.pt")
            (output / "manifest.partial.json").write_text(json.dumps(manifest, indent=2))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_cache(directory, split, device):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    groups = []
    ranges = {}
    offset = 0
    for name in manifest["datasets"]:
        data = torch.load(Path(directory) / f"{name}.{split}.pt", weights_only=True)
        size = len(data["latent"])
        ranges[name] = (offset, offset + size)
        offset += size
        groups.append(data)
    return {key: torch.cat([data[key] for data in groups]).to(device) for key in groups[0]}, ranges


def prediction(model, batch):
    goal = model.condition_world_model(batch["goal"], batch["state"], "franka")
    result = model.world_model(batch["latent"], ctx_len=1, goal=goal,
                               update_stats=False, loss_mask=batch["mask"],
                               residual_correction=model.world_model_correction(
                                   batch["latent"][:, :1], batch["goal"], batch["state"], "franka"))
    head = model.action_models["franka"]
    tokens = torch.cat((batch["latent"][:, :1], result["pred_future_latent"]), dim=1)
    actions = head.predict_action(head.decode_action_queries(tokens, state=batch["state"]))
    return result, actions


@torch.no_grad()
def evaluate(model, data, ranges, batch_size):
    """Aggregate sums/counts, never average uneven batch means or ratios."""
    model.eval()
    per_dataset = {}
    for name, (start, end) in ranges.items():
        sums = torch.zeros(7, device=data["latent"].device, dtype=torch.float64)
        horizons = torch.zeros(2, 2, device=sums.device, dtype=torch.float64)
        for offset in range(start, end, batch_size):
            batch = {k: v[offset:min(offset + batch_size, end)] for k, v in data.items()}
            result, actions = prediction(model, batch)
            target = batch["latent"][:, 1:]
            mask = batch["mask"][:, 1:, :, None].float()
            error = (result["pred_future_latent"].float() - target).square() * mask
            residual = (target - batch["latent"][:, :1]) * mask
            count = mask.sum() * target.shape[-1]
            action_mask = batch["action_mask"][:, :, None].float()
            action_error = ((actions - batch["action"]).abs() * action_mask).sum()
            action_count = action_mask.sum() * actions.shape[-1]
            pred_delta = (result["pred_future_latent"] - batch["latent"][:, :1]) * mask
            valid = residual.flatten(2).square().sum(-1) > 1e-8
            cosine = torch.nn.functional.cosine_similarity(pred_delta.flatten(2), residual.flatten(2), dim=-1)
            sums += torch.stack((error.sum(), count, residual.square().sum(), action_error,
                                 action_count, (cosine * valid).sum(), valid.sum()))
            horizons[0] += error.sum((0, 2, 3))
            horizons[1] += mask.sum((0, 2, 3)) * target.shape[-1]
        per_dataset[name] = {"sums": sums.cpu().tolist(), "horizons": horizons.cpu().tolist()}

    def metrics(sums, horizons):
        error, count, copy_error, action_error, action_count, cosine, cosine_count = sums
        return {"latent_loss": error / max(count, 1), "copy_mse": copy_error / max(count, 1),
                "delta_to_copy_ratio": error / max(copy_error, 1e-8),
                "action_l1": action_error / max(action_count, 1),
                "direction_cosine": cosine / max(cosine_count, 1),
                "horizon_mse": (np.array(horizons[0]) / np.maximum(horizons[1], 1)).tolist()}

    total = np.array([v["sums"] for v in per_dataset.values()]).sum(0)
    horizons = np.array([v["horizons"] for v in per_dataset.values()]).sum(0)
    return {"pooled": metrics(total, horizons),
            "datasets": {name: metrics(v["sums"], v["horizons"]) for name, v in per_dataset.items()}}


def fit(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((Path(args.cache) / "manifest.json").read_text())
    if checkpoint_hash(args.checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Cached targets and model must come from the same checkpoint")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_model(args, state_condition=args.variant == "state")
    # Keep the cache's exact representation and time offsets for deployment.
    for section in ("world_model", "lang_cond", "action_model"):
        cached = manifest["config"]["framework"][section]
        actual = OmegaConf.to_container(model.config.framework[section], resolve=True)
        for key in set(cached) | set(actual):
            if key not in {"condition_world_model_on_state", "state_conditioning_mode"} and cached.get(key) != actual.get(key):
                raise ValueError(f"Cache configuration mismatch: {section}.{key}")
    train, _ = read_cache(args.cache, "train", args.device)
    val, val_ranges = read_cache(args.cache, "val", args.device)
    baseline = evaluate(model, val, val_ranges, args.batch_size)
    model.world_model.requires_grad_(not getattr(args, "adapter_only", False))
    model.world_model_state_encoders.requires_grad_(True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.1)
    best_score = baseline["pooled"]["latent_loss"]
    best_epoch = 0
    # Only mutable modules are copied; the 86M-parameter encoder stays frozen.
    def snapshot():
        return {name: copy.deepcopy(module.state_dict()) for name, module in
                (("world_model", model.world_model), ("world_model_state_encoders", model.world_model_state_encoders))}
    initial = snapshot()
    best = initial
    history = [{"epoch": 0, "validation": baseline}]
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.world_model.train()
        order = torch.randperm(len(train["latent"]), generator=generator)
        loss_sum = 0.0
        for indices in order.split(args.batch_size):
            batch = {key: value[indices.to(value.device)] for key, value in train.items()}
            result, actions = prediction(model, batch)
            action_loss = masked_action_l1_loss(actions, batch["action"], batch["action_mask"])
            loss = result["latent_loss"] + args.cosine_weight * result["latent_cosine_loss"] + args.action_weight * action_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        scheduler.step()
        validation = evaluate(model, val, val_ranges, args.batch_size)
        acceptable = bool(validation["pooled"]["action_l1"] <= baseline["pooled"]["action_l1"] * (1 + args.action_tolerance))
        if acceptable and validation["pooled"]["latent_loss"] < best_score:
            best_score = validation["pooled"]["latent_loss"]
            best_epoch, best = epoch, snapshot()
        row = {"epoch": epoch, "train_total_loss": loss_sum / len(order), "validation": validation,
               "action_guard_passed": acceptable, "best_epoch": best_epoch}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({"epoch": epoch, "best_epoch": best_epoch, **validation["pooled"]}), flush=True)
    for name, state in best.items():
        getattr(model, name).load_state_dict(state)
    del train, val
    torch.cuda.empty_cache()
    test, test_ranges = read_cache(args.cache, "test", args.device)
    final_metrics = evaluate(model, test, test_ranges, args.batch_size)
    for name, state in initial.items():
        getattr(model, name).load_state_dict(state)
    baseline_test = evaluate(model, test, test_ranges, args.batch_size)
    for name, state in best.items():
        getattr(model, name).load_state_dict(state)
    final_dir = output / "final_model"
    final_dir.mkdir()
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, final_dir / "pytorch_model.pt")
    OmegaConf.save(model.config, output / "config.yaml")
    # Use the statistics saved with the source model, not subset statistics.
    stats_path = Path(args.config).parent / "dataset_statistics.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Source normalization statistics required: {stats_path}")
    (output / "dataset_statistics.json").write_bytes(stats_path.read_bytes())
    report = {"arguments": vars(args), "source_sha256": manifest["checkpoint_sha256"],
              "best_epoch": best_epoch, "baseline_validation": baseline,
              "selected_validation": history[best_epoch]["validation"], "test": final_metrics,
              "baseline_test": baseline_test,
              "limitation": manifest["limitation"]}
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


@torch.no_grad()
def calibrate(args):
    """Fit a prediction amplitude on training data, preserving target units."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((Path(args.cache) / "manifest.json").read_text())
    if checkpoint_hash(args.checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Cached targets and model must come from the same checkpoint")
    model = load_model(args)
    train, _ = read_cache(args.cache, "train", args.device)
    dot, square = 0.0, 0.0
    for start in range(0, len(train["latent"]), args.batch_size):
        batch = {k: v[start:start + args.batch_size] for k, v in train.items()}
        predicted = model.world_model.regress_future(batch["latent"][:, :1], batch["goal"])
        delta = predicted - batch["latent"][:, :1]
        target_delta = batch["latent"][:, 1:] - batch["latent"][:, :1]
        mask = batch["mask"][:, 1:, :, None]
        dot += float((delta * target_delta * mask).double().sum())
        square += float((delta.square() * mask).double().sum())
    fitted_gain = max(0.0, min(2.0, dot / square)) if square > 0 else 1.0
    del train
    val, ranges = read_cache(args.cache, "val", args.device)
    initial_scale = model.world_model.delta_scale.clone()
    baseline = evaluate(model, val, ranges, args.batch_size)
    best_gain, best_score = 1.0, baseline["pooled"]["latent_loss"]
    history = []
    for fraction in (0.0, 0.25, 0.5, 1.0):
        gain = 1 + fraction * (fitted_gain - 1)
        model.world_model.delta_scale.copy_(initial_scale * gain)
        metrics = evaluate(model, val, ranges, args.batch_size)
        acceptable = bool(metrics["pooled"]["action_l1"] <= baseline["pooled"]["action_l1"] * (1 + args.action_tolerance))
        if acceptable and metrics["pooled"]["latent_loss"] < best_score:
            best_gain, best_score = gain, metrics["pooled"]["latent_loss"]
        history.append({"gain": gain, "validation": metrics, "action_guard_passed": acceptable})
    del val
    test, ranges = read_cache(args.cache, "test", args.device)
    model.world_model.delta_scale.copy_(initial_scale)
    baseline_test = evaluate(model, test, ranges, args.batch_size)
    model.world_model.delta_scale.copy_(initial_scale * best_gain)
    result = evaluate(model, test, ranges, args.batch_size)
    (output / "final_model").mkdir()
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, output / "final_model/pytorch_model.pt")
    OmegaConf.save(model.config, output / "config.yaml")
    (output / "dataset_statistics.json").write_bytes((Path(args.config).parent / "dataset_statistics.json").read_bytes())
    report = {"arguments": vars(args), "fitted_train_gain": fitted_gain, "selected_gain": best_gain,
              "baseline_validation": baseline, "validation_candidates": history,
              "baseline_test": baseline_test, "test": result,
              "source_sha256": manifest["checkpoint_sha256"], "limitation": manifest["limitation"]}
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"fitted_train_gain": fitted_gain, "selected_gain": best_gain, "test": result["pooled"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cache", "fit", "calibrate"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=512, help="Frames per dataset")
    parser.add_argument("--eval-samples", type=int, default=128, help="Frames per dataset per evaluation split")
    parser.add_argument("--cache")
    parser.add_argument("--variant", choices=("control", "state"), default="state")
    parser.add_argument("--adapter-only", action="store_true", help="Freeze the existing predictor; learn only state conditioning")
    parser.add_argument("--state-conditioning", choices=("goal", "residual"), default="goal")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--action-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    if args.batch_size < 1 or min(args.train_samples, args.eval_samples, args.epochs) < 1:
        parser.error("Batch size, sample counts and epochs must be positive")
    if args.mode != "cache" and not args.cache:
        parser.error("fit/calibrate require --cache")
    if args.adapter_only and (args.mode != "fit" or args.variant != "state"):
        parser.error("--adapter-only requires fit --variant state")
    torch.set_num_threads(2)
    {"cache": cache, "fit": fit, "calibrate": calibrate}[args.mode](args)


if __name__ == "__main__":
    main()
