def get_world_model(config):
    """Factory for world model backends.

    Routes to the correct world-model wrapper based on
    ``config.framework.world_model.base_wm``.

    Every world-model wrapper exposes:
      - ``forward(**kwargs)`` → model outputs with hidden_states
      - ``build_inputs(images, instructions)`` → dict of tensors
      - ``generate(**kwargs)`` → generation (optional)
    """

    # Every WM4A framework (LeWMOFT / Wan* / CosmoPredict2*) provides a
    # ``world_model`` section with an explicit ``base_wm``; the legacy fallback
    # to ``qwenvl.base_vlm`` was removed.
    wm_cfg = config.framework.get("world_model", None)
    if wm_cfg is None:
        raise ValueError(
            "framework.world_model is required "
            "(set framework.world_model.base_wm)"
        )
    wm_name = wm_cfg.get("base_wm", "")
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
    elif (
        "lewm" in wm_name.lower()
        or "le-wm" in wm_name.lower()
        or "vit" in wm_name.lower()
        or "dino" in wm_name.lower()
    ):
        # _LeWM_Interface now supports only raw DINOv3 checkpoints (*.pth with
        # 'dinov3' in the filename); it rejects anything else with a clear error.
        from .LeWM import _LeWM_Interface

        return _LeWM_Interface(config)
    else:
        raise NotImplementedError(f"World model {wm_name} not implemented")
