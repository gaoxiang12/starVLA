# Copyright 2026 starVLA / gemma-vla community.
# Licensed under the MIT License.
"""
Gemma4-GR00T Framework
Direct port of QwenGR00T to the Gemma 4 VLM backbone (`google/gemma-4-E2B-it`).

See `Gemma4PI.py` for the rationale: VLM swap is handled by the dispatcher in
`starVLA/model/modules/vlm/__init__.py`, so this file just re-registers under a new
framework name. Override forward / predict_action here only if Gemma 4 ever needs
GR00T-specific surgery.
"""
from typing import Optional

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("Gemma4GR00T")
class Gemma4_GR00T(Qwen_GR00T):
    """
    Gemma 4 + last-hidden-state cross-attention DiT (GR00T head).

    The QwenGR00T constructor already does:
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
    so wiring picks up Gemma 4 E2B's hidden_size=1536 automatically.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        backbone_hidden = self.qwen_vl_interface.model.config.hidden_size
        assert backbone_hidden in (1536, 2048, 2560), (
            f"[Gemma4GR00T] unexpected backbone hidden_size={backbone_hidden}; "
            f"check `framework.qwenvl.base_vlm` and DiT cross_attention_dim alignment."
        )
