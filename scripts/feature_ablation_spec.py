"""Make capacity-matched DINO layer ablations without changing GAWM training.

The input configuration is the archived ``input_config.yaml`` of the reference
run. Only layer selection and run/checkpoint bookkeeping may differ. In
particular, stopping earlier does not shorten C's scheduler or either stage.
This module does not launch training, copy assets, or modify a reference run.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from omegaconf import DictConfig, OmegaConf

from starVLA.training.recipe import apply_training_recipe, resolve_training_budget


VARIANTS = {
    "baseline": [-12, -8, -4],
    "late_layers": [-3, -2, -1],
    "last_layer": [-1, -1, -1],
    "wide_layers": [-24, -16, -8],
}

# Compare raw inputs as well as resolved recipes. Otherwise a change to an
# overridden (currently inactive) method field could escape the guard.
ALLOWED_CHANGES = frozenset({
    "framework.world_model.feat_layers",
    "framework.lang_cond.task_vectors_path",
    "run_id",
    "run_root_dir",
    "output_dir",
    "trainer.max_train_steps",
    "trainer.milestone_steps",
    "training_overrides.trainer.max_train_steps",
    "training_overrides.trainer.milestone_steps",
})


def _config(value):
    if isinstance(value, (str, Path)):
        return OmegaConf.load(value)
    if hasattr(value, "unwrap"):
        value = value.unwrap()
    if OmegaConf.is_config(value):
        return OmegaConf.create(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Mapping):
        return OmegaConf.create(value)
    raise TypeError("Expected a config path, DictConfig, or mapping")


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset(path):
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise FileNotFoundError(f"Required reference asset is missing: {file}")
    return {"path": str(file), "sha256": sha256_file(file)}


def _resolved(value):
    cfg = apply_training_recipe(_config(value))
    if cfg.trainer.recipe != "c":
        raise ValueError("Feature ablations require the reference C recipe")
    batch = _positive_int(cfg.trainer.expected_global_batch_size, "global batch")
    frames = _positive_int(cfg.datasets.vla_data.expected_frames, "expected_frames")
    resolve_training_budget(cfg, range(frames), batch)
    return cfg


def baseline_budget(base):
    """Return the reference stop step and nominal stages/scheduler in steps."""
    cfg = _resolved(base)
    trainer = cfg.trainer
    batch = int(cfg.datasets.vla_data.per_device_batch_size)
    accumulation = int(trainer.gradient_accumulation_steps)
    denominator = _positive_int(batch * accumulation, "per-rank effective batch")
    global_batch = int(trainer.expected_global_batch_size)
    if global_batch % denominator:
        raise ValueError("Reference global batch is not divisible by per-rank batch")
    return {
        "max_steps": int(trainer.max_train_steps),
        "steps_per_epoch": int(trainer.steps_per_epoch),
        "stage1_steps": int(trainer.stage1_steps),
        "stage_epochs": list(trainer.stage_epochs),
        "lr_scheduler_total_steps": int(trainer.lr_scheduler_total_steps),
        "global_batch": global_batch,
        "per_device_batch_size": batch,
        "gradient_accumulation_steps": accumulation,
        "world_size": global_batch // denominator,
    }


def _changes(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        changes = {}
        for key in sorted(before.keys() | after.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before:
                if isinstance(after[key], dict) and after[key]:
                    changes.update(_changes({}, after[key], path))
                else:
                    changes[path] = {"before": None, "after": after[key]}
            elif key not in after:
                if isinstance(before[key], dict) and before[key]:
                    changes.update(_changes(before[key], {}, path))
                else:
                    changes[path] = {"before": before[key], "after": None}
            else:
                changes.update(_changes(before[key], after[key], path))
        return changes
    if before != after:
        return {prefix: {"before": before, "after": after}}
    return {}


def _plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


def validate_config(base, candidate):
    """Reject any change outside the layer/run/checkpoint allowlist.

    Return raw input differences. Both raw and recipe-expanded configurations
    are checked, including the nominal C budget calculated from frame counts.
    A relocated VTT must already exist and match the reference bytes.
    """
    before, after = _config(base), _config(candidate)
    raw_changes = _changes(_plain(before), _plain(after))
    resolved_before, resolved_after = _resolved(before), _resolved(after)
    effective_changes = _changes(_plain(resolved_before), _plain(resolved_after))
    forbidden = (set(raw_changes) | set(effective_changes)) - ALLOWED_CHANGES
    if forbidden:
        raise ValueError("Feature ablation changed protected fields: " + ", ".join(sorted(forbidden)))

    wm = resolved_after.framework.world_model
    if (wm.get("visual_frontend") != "lila" or wm.get("future_objective") != "fixed_dino_patches"
            or wm.get("train_encoder") is not False or wm.get("detach_wm_input") is not False
            or wm.get("lila_bridge_norm") != "fixed_layernorm" or float(wm.latent_cosine_weight) != 0.):
        raise ValueError("Reference must retain frozen LiLa features and fixed DINO patch supervision")
    layers = list(wm.feat_layers)
    if len(layers) != 3 or layers not in VARIANTS.values():
        raise ValueError("feat_layers must be one of the four capacity-matched variants")
    if len(resolved_before.framework.world_model.feat_layers) != 3:
        raise ValueError("Reference must have exactly three fusion slots")
    if wm.get("encoder_spec") != "vitl16":
        raise ValueError("The four layer variants require the reference 24-layer ViT-L encoder")

    stop = _positive_int(resolved_after.trainer.max_train_steps, "max_steps")
    if stop > int(resolved_before.trainer.max_train_steps):
        raise ValueError("Ablation budget cannot exceed the complete reference budget")
    milestones = list(resolved_after.trainer.milestone_steps)
    if (any(isinstance(x, bool) or not isinstance(x, int) or not 0 < x <= stop for x in milestones)
            or milestones != sorted(set(milestones))):
        raise ValueError("milestone_steps must be sorted unique positive steps within the budget")

    vtt_key = "framework.lang_cond.task_vectors_path"
    if vtt_key in raw_changes:
        original = OmegaConf.select(resolved_before, vtt_key)
        relocated = OmegaConf.select(resolved_after, vtt_key)
        if not original or not relocated or sha256_file(original) != sha256_file(relocated):
            raise ValueError("Relocated VTT must have exactly the reference SHA-256")
    run_id = str(after.run_id)
    if Path(run_id).name != run_id or run_id in ("", ".", ".."):
        raise ValueError("run_id must be a named directory without path components")
    output = OmegaConf.select(after, "output_dir")
    if output is not None:
        expected_output = Path(after.run_root_dir).expanduser().resolve() / str(after.run_id)
        if Path(output).expanduser().resolve() != expected_output:
            raise ValueError("output_dir must match run_root_dir/run_id")
    return raw_changes


def make_config(base_path, run, variant, world_size, max_steps=None, eval_steps=None,
                *, task_vectors_path=None) -> DictConfig:
    """Build an archived-input-based arm, preserving batch, workers and policy.

    ``run`` is the intended training output directory. ``eval_steps`` specifies
    extra checkpoint milestones for the external closed-loop evaluator; no
    trainer-side evaluation method is added. Assets are relocated only when an
    already-copied, byte-identical VTT path is explicitly supplied.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant!r}")
    base = _config(base_path)
    budget = baseline_budget(base)
    if _positive_int(world_size, "world_size") != budget["world_size"]:
        raise ValueError(f"Reference batch requires {budget['world_size']} ranks; "
                         "per-device batch and accumulation must stay unchanged")
    stop = budget["max_steps"] if max_steps is None else _positive_int(max_steps, "max_steps")
    if stop > budget["max_steps"]:
        raise ValueError("Ablation budget cannot exceed the complete reference budget")
    if eval_steps is None:
        milestones = [int(x) for x in _resolved(base).trainer.milestone_steps if int(x) <= stop]
    else:
        milestones = [_positive_int(x, "eval step") for x in eval_steps]
        if any(x > stop for x in milestones):
            raise ValueError("Evaluation steps cannot exceed the ablation budget")
    milestones = sorted(set(milestones) | {stop})

    if Path(run).expanduser().name in ("", ".", ".."):
        raise ValueError("run must identify a named output directory")
    run = Path(run).expanduser().resolve()
    if run.name in ("", ".", ".."):
        raise ValueError("run must identify a named output directory")
    cfg = _config(base)
    cfg.run_id = run.name
    cfg.run_root_dir = str(run.parent)
    if "output_dir" in cfg:
        cfg.output_dir = str(run)
    cfg.framework.world_model.feat_layers = list(VARIANTS[variant])
    # Unresolved recipes replace managed trainer fields, so their override
    # subtree is authoritative. Resolved configs are updated directly too.
    OmegaConf.update(cfg, "training_overrides.trainer.max_train_steps", stop, force_add=True)
    OmegaConf.update(cfg, "training_overrides.trainer.milestone_steps", milestones, force_add=True)
    if cfg.get("recipe_resolved", False) or "max_train_steps" in cfg.trainer:
        cfg.trainer.max_train_steps = stop
    if cfg.get("recipe_resolved", False) or "milestone_steps" in cfg.trainer:
        cfg.trainer.milestone_steps = milestones
    if task_vectors_path is not None:
        cfg.framework.lang_cond.task_vectors_path = str(Path(task_vectors_path).expanduser().resolve())
    validate_config(base, cfg)
    return cfg


def asset_contract(config, *, baseline_dir=None):
    """Record schema and hashes of small frozen assets, without scanning data.

    No dataset or encoder weight file is rewritten. Full source/weight
    snapshots remain the controller's responsibility. ``baseline_dir`` can
    include the reference run's saved dataset statistics for LIBERO.
    """
    cfg = _resolved(config)
    data, wm, lang = cfg.datasets.vla_data, cfg.framework.world_model, cfg.framework.lang_cond
    encoder_config = Path(wm.vision_encoder_path).expanduser() / "config.json"
    spec = json.loads(encoder_config.read_text())
    if spec.get("num_hidden_layers") != 24 or spec.get("hidden_size") != 1024:
        raise ValueError("The four layer variants require the reference 24-layer ViT-L encoder")
    assets = {"encoder_config": _asset(encoder_config)}
    if lang.get("task_vectors_path"):
        assets["vtt"] = _asset(lang.task_vectors_path)
    if data.get("episode_exclusions_file"):
        assets["episode_exclusions"] = _asset(data.episode_exclusions_file)
    for key in ("normalization_statistics_path", "official_stats"):
        if data.get(key):
            assets[key] = _asset(data[key])
    if data.dataset_py == "robotwin_official_hdf5" and not data.get("official_stats"):
        assets["official_stats"] = _asset(Path(data.data_root_dir) / "training_metadata/stat-500-all.json")
    if baseline_dir is not None:
        statistics = Path(baseline_dir) / "dataset_statistics.json"
        if statistics.is_file():
            assets["saved_dataset_statistics"] = _asset(statistics)
    width, height = map(int, wm.lila_image_size)
    patch_size = int(spec["patch_size"])
    return {
        "assets": assets,
        "teacher": {"source": "frozen DINO last_hidden_state patch tokens", "feature_dim": spec["hidden_size"],
                    "patches_per_view": (width // patch_size) * (height // patch_size)},
        "interface": {"num_views": int(wm.num_views), "tokens_per_view": int(wm.visual_tokens_per_view),
                      "token_dim": int(wm.visual_token_dim), "feat_layers": list(wm.feat_layers),
                      "prefix_tokens": 1 + int(spec["num_register_tokens"]), "image_size": [width, height]},
        "data": _plain(cfg.datasets),
        "budget": baseline_budget(cfg),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-steps", type=int, nargs="+")
    parser.add_argument("--task-vectors-path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = make_config(args.baseline, args.run_dir, args.variant, args.world_size,
                      args.max_steps, args.eval_steps, task_vectors_path=args.task_vectors_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing review configuration: {output}")
    OmegaConf.save(cfg, output)
    print(json.dumps({"config": str(output.resolve()), "changes": validate_config(args.baseline, cfg),
                      "budget": baseline_budget(cfg)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
