# Copyright 2026 starVLA / gemma-vla community.
# Licensed under the MIT License.
"""
Gemma4-PI Framework
A direct port of QwenPI to the Gemma 4 VLM backbone (`google/gemma-4-E2B-it`).

The actual VLM swap happens in `starVLA/model/modules/vlm/__init__.py::get_vlm_model`,
which routes any `framework.qwenvl.base_vlm` containing "gemma-4" to
`_Gemma4_VL_Interface`. Because that interface mirrors `_QWen3_VL_Interface`, the body
of this framework is identical to QwenPI — we simply re-register under a new name so
configs can ask for `framework.name=Gemma4PI`.

If Gemma 4 ever needs framework-level surgery (extra projector, custom hidden-state
slicing, etc.), override `forward` / `predict_action` here rather than touching QwenPI.
"""
from typing import Optional

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("Gemma4PI")
class Gemma4_PI(Qwen_PI):
    """
    Gemma 4 + layer-wise flow-matching DiT action head.

    Notes:
        - DiT `cross_attention_dim` must equal Gemma 4 E2B's text_config.hidden_size = 1536.
          The constructor of `Qwen_PI` reads `qwen_vl_interface.model.config.hidden_size`,
          which `_Gemma4_VL_Interface` aligns to 1536 at load time, so the existing
          alignment logic in QwenPI works without modification.
        - PI consumes the last N hidden states for layer-wise cross-attention. Gemma 4 has
          35 text layers, so any reasonable DiT block count (≤35) fits.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        # Hard assertion against the documented Gemma4-E2B hidden size to fail loud
        # if anyone wires in a wrong checkpoint or a future variant.
        backbone_hidden = self.qwen_vl_interface.model.config.hidden_size
        assert backbone_hidden in (1536, 2048, 2560), (
            f"[Gemma4PI] unexpected backbone hidden_size={backbone_hidden}; "
            f"check `framework.qwenvl.base_vlm` and DiT cross_attention_dim alignment."
        )
