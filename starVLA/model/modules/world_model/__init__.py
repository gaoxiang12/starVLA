def get_world_model(config):
    """Factory for world model backends.

    GAWM is routed by ``framework.name`` and constructed from
    ``world_model.encoder_spec``. Other backends are routed by ``base_wm``.

    Every world-model wrapper exposes:
      - ``forward(**kwargs)`` → model outputs with hidden_states
      - ``build_inputs(images, instructions)`` → dict of tensors
      - ``generate(**kwargs)`` → generation (optional)
    """

    # Every WM4A framework provides a world_model section. GAWM embeds its
    # encoder weights in the unified checkpoint; generative backends still use
    # an external base_wm path.
    wm_cfg = config.framework.get("world_model", None)
    if wm_cfg is None:
        raise ValueError(
            "framework.world_model is required"
        )
    framework_name = str(config.framework.get("name", "")).strip().lower()
    wm_name = wm_cfg.get("base_wm", "")
    if framework_name == "gawm":
        from .GAWM import _GAWM_Interface

        return _GAWM_Interface(config)
    if not wm_name:
        raise ValueError(
            "framework.world_model.base_wm is required "
            "(e.g. facebook/dinov2-base or a DINOv3 .pth)"
        )

    if "cosmos-reason2" in wm_name.lower():
        from ..vlm.CosmosReason2 import _CosmosReason2_Interface

        return _CosmosReason2_Interface(config)
    elif "cosmos-predict2" in wm_name.lower() or "cosmos-predict2" in wm_name.lower():
        from .CosmoPredict2 import _CosmoPredict2_Interface

        return _CosmoPredict2_Interface(config)
    elif "wan2" in wm_name.lower() or "ti2v" in wm_name.lower():
        from .Wan2 import _Wan2_Interface

        return _Wan2_Interface(config)
    elif "taesd" in wm_name.lower():
        from .TAESD import _TAESD_Interface

        return _TAESD_Interface(config)
    elif "gawm" in wm_name.lower() or "dino" in wm_name.lower():
        from .GAWM import _GAWM_Interface

        return _GAWM_Interface(config)
    else:
        raise NotImplementedError(f"World model {wm_name} not implemented")
