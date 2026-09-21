# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Haron Wan / CUHK Shenzhen] in [2026].

from typing import Optional

import torch
import torch.nn as nn
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

ROOT = Path(__file__).parents[1]
SEPARATOR = "-" * 20

PIXELS_PER_TOKEN = 32**2
"""Number of pixels per visual token."""


class _CosmosReason2_Interface(nn.Module):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        qwenvl_config = config.framework.get("qwenvl", {})
        model_name = qwenvl_config.get("base_vlm", "nvidia/Cosmos-Reason2-2B")

        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation
        )
        self.processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)
        self.config = config

        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )
        return outputs

    def generate(
        self,
        **kwargs,
    ):
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            messages.append(msg)

        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            # fps=4,
        )

        return inputs.to(self.model.device)
