"""Canonical GAWM-L configuration and migration of historical checkpoints.

Legacy spellings belong here; current recipes and model code use gawm_l.
This does not rename historical reference frameworks or checkpoint tensors.
"""
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict


_LEGACY_VISUAL_KEYS = {
    "lila_image_size": "gawm_l_image_size",
    "lila_adapter_dim": "gawm_l_adapter_dim",
    "lila_adapter_depth": "gawm_l_adapter_depth",
    "lila_adapter_heads": "gawm_l_adapter_heads",
    "lila_bridge_norm": "gawm_l_bridge_norm",
}


def migrate_gawm_config(config):
    """Normalize names in place, preserving paths and rejecting conflicts."""
    cfg = getattr(config, "_cfg", config)
    if cfg is None:
        return config
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
        config = cfg
    framework = cfg.get("framework")
    if framework is None or not str(framework.get("name", "GAWM")).startswith("GAWM"):
        return config
    world = framework.get("world_model")
    if world is None:
        return config
    with open_dict(world):
        for old, new in _LEGACY_VISUAL_KEYS.items():
            if old not in world:
                continue
            if new in world and world[new] != world[old]:
                raise ValueError(f"Conflicting legacy/current GAWM settings: {old} and {new}")
            world[new] = world.pop(old)
        if world.get("visual_frontend") == "lila":
            world.visual_frontend = "gawm_l"
    if framework.get("name") == "GAWM" and world.get("encoder_spec") == "vitl16":
        with open_dict(framework):
            framework.name = "GAWM-L"
    return config


def config_for_gawm_source(config, source):
    """Export canonical settings for a pinned source without editing its files."""
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    migrate_gawm_config(cfg)
    modules = Path(source) / "starVLA/model/modules"
    if (modules / "gawm_lila_vision.py").is_file() and not (modules / "gawm_l_vision.py").is_file():
        world = cfg.framework.world_model
        for old, new in _LEGACY_VISUAL_KEYS.items():
            if new in world:
                world[old] = world.pop(new)
        if world.get("visual_frontend") == "gawm_l":
            world.visual_frontend = "lila"
        if cfg.framework.name == "GAWM-L":
            cfg.framework.name = "GAWM"
    return cfg
