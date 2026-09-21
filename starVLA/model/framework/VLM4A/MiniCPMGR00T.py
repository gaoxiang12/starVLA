# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
MiniCPM-GR00T Framework
Direct port of QwenGR00T to the MiniCPM-V 4.6 backbone (`openbmb/MiniCPM-V-4.6`).

See `MiniCPMPI.py` for the rationale: VLM swap is handled by the dispatcher in
`starVLA/model/modules/vlm/__init__.py`, so this file just re-registers under a new
framework name. Override forward / predict_action here only if MiniCPM-V needs
GR00T-specific surgery.
"""
from typing import Optional

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("MiniCPMGR00T")
class MiniCPM_GR00T(Qwen_GR00T):
    """
    MiniCPM-V 4.6 + last-hidden-state cross-attention DiT (GR00T head).

    QwenGR00T overwrites `cross_attention_dim` from
    `qwen_vl_interface.model.config.hidden_size`, which MiniCPM-V exposes as 1024.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        backbone_hidden = self.qwen_vl_interface.model.config.hidden_size
        assert backbone_hidden == 1024, (
            f"[MiniCPMGR00T] unexpected backbone hidden_size={backbone_hidden}; "
            "check `framework.qwenvl.base_vlm` and DiT cross_attention_dim alignment."
        )
