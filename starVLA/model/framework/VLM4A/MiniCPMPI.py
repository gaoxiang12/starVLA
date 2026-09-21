# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
MiniCPM-PI Framework
A direct port of QwenPI to the MiniCPM-V 4.6 backbone (`openbmb/MiniCPM-V-4.6`).

The actual VLM swap happens in `starVLA/model/modules/vlm/__init__.py::get_vlm_model`,
which routes any `framework.qwenvl.base_vlm` containing "minicpm-v" / "minicpmv" to
`_MiniCPM_VL_Interface`. Because that interface mirrors `_QWen3_VL_Interface`, the body
of this framework is identical to QwenPI — we simply re-register under a new name so
configs can ask for `framework.name=MiniCPMPI`.
"""
from typing import Optional

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("MiniCPMPI")
class MiniCPM_PI(Qwen_PI):
    """
    MiniCPM-V 4.6 + layer-wise flow-matching DiT action head.

    MiniCPM-V 4.6 exposes `text_config.hidden_size = 1024` and
    `text_config.num_hidden_layers = 24`; `Qwen_PI` reads those values from the
    wrapper at runtime and aligns the layer-wise DiT config automatically.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        backbone_hidden = self.qwen_vl_interface.model.config.hidden_size
        assert backbone_hidden == 1024, (
            f"[MiniCPMPI] unexpected backbone hidden_size={backbone_hidden}; "
            "check `framework.qwenvl.base_vlm` and DiT cross_attention_dim alignment."
        )
