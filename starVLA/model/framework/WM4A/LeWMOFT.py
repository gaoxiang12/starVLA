# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
LeWM-Flow Framework — LeWorldModel ViT encoder + flow-matching world model.

Uses the LeWM front-end (a pretrained ViT encoder) as the perception backbone
and a Wan-style flow-matching world model that both rolls out future latents
and flow-samples the action chunk. Actions come entirely from the world
model's flow head (the previous OFT / MLP regression head has been removed).

Architecture:
  ViT encoder → per-view latent concat[CLS, mean-pool] → [B, V, 2*hidden]
    → view fusion → [B, 1+Tf, hidden]
    → WanWorldModel flow predictor
        → future latents          (flow_latent_loss)
        → flow-sampled actions    (flow_action_loss)
    → optional state probe on [current, predicted future] latents (state_loss)

The ViT encoder is frozen by default; set ``world_model.train_encoder: true``
to keep the joint fine-tuning interface available.
"""

import hashlib
import os
import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.modules.world_model.wan_world_model import WanWorldModel
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class LeWMOFTDefaultConfig:
    """LeWM-OFT default parameters."""

    name: str = "LeWMOFT"

    # === World Model backbone (LeWM ViT encoder) ===
    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "WinKawaks/vit-tiny-patch16-224",
            "train_encoder": False,  # frozen ViT by default; flip for joint finetune
            "num_views": 2,          # camera views per frame (e.g. primary + wrist)
            # === flow-matching latent world model (ported le-wm WanPredictor) ===
            # When ``use_future_latent`` is on, a Wan-style predictor rolls out
            # future latents (flow_latent_loss) and the OFT head conditions on
            # ``[current latent, predicted future latents]`` before producing the
            # action chunk. Requires the dataloader to supply ``future_images``.
            "use_future_latent": True,
            "n_future": 2,            # number of future latents to predict
            "ctx_len": 1,             # clean context frames (current frame only)
            "predictor_dim": 384,
            "predictor_layers": 4,
            "predictor_heads": 6,
            "predictor_ffn": 1024,
            "flow_sample_steps": 10,
            "loss_latent_weight": 1.0,
            "loss_action_weight": 1.0,
            # === Optional state probe (align latents to future proprio) ===
            "use_state_probe": False,
            "state_dim": 8,
            "loss_state_weight": 0.5,
        }
    )

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "WinKawaks/vit-tiny-patch16-224",
        }
    )

    # === Action shape config (action_dim / horizon consumed by the flow model) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "MLP",
            "action_dim": 7,
            "action_hidden_dim": 384,
            "future_action_window_size": 8,
            "past_action_window_size": 0,
        }
    )

    # === Language / task conditioning (le-wm-style task embedding) ===
    # le-wm conditions its action head on a per-task one-hot. starVLA only
    # exposes language strings, so we hash each instruction into a fixed bucket
    # and look up a learnable embedding (an implicit task-id embedding that needs
    # no predefined task list). ``embed_dim: null`` -> world-model hidden size.
    lang_cond: dict = field(
        default_factory=lambda: {
            "num_buckets": 4096,
            "embed_dim": None,
        }
    )


@FRAMEWORK_REGISTRY.register("LeWMOFT")
class LeWM_OFT(baseframework):
    """World-Model-for-Action framework: LeWM ViT encoder + flow-matching world model."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(LeWMOFTDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)

        wm_hidden = self.backbone.model.config.hidden_size

        # `action_horizon` is the single source of truth for chunk length;
        # legacy aliases are normalised upstream by share_tools.apply_config_compat.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # === Language / task conditioning ===
        # Hash each instruction into a fixed bucket and look up a learnable task
        # embedding (an implicit one-hot / task-id embedding requiring no
        # predefined task list; stable across train/eval because identical
        # instruction strings hash identically). Feeds the world model as the
        # per-task ``goal`` conditioning.
        lang_cfg = self.config.framework.get("lang_cond", {}) or {}
        self.num_task_buckets = int(lang_cfg.get("num_buckets", 4096))
        _emb_dim = lang_cfg.get("embed_dim", None)
        self.task_emb_dim = int(_emb_dim) if _emb_dim else wm_hidden
        self.task_embedding = nn.Embedding(self.num_task_buckets, self.task_emb_dim)

        # === Flow-matching latent world model (le-wm WanPredictor) ===
        # Predicts future latents so the OFT head can "see the future" before
        # producing actions, and supplies a real latent-prediction loss.
        wm_cfg = self.config.framework.get("world_model", {}) or {}
        self.use_future_latent = bool(wm_cfg.get("use_future_latent", True))
        if self.use_future_latent:
            self.n_future = int(wm_cfg.get("n_future", 2))
            self.wm_ctx_len = int(wm_cfg.get("ctx_len", 1))
            raw_action_dim = int(self.config.framework.action_model.action_dim)
            assert (
                self.action_horizon % self.n_future == 0
            ), f"action_horizon ({self.action_horizon}) must be divisible by n_future ({self.n_future})"
            self.wm_segment_len = self.action_horizon // self.n_future
            self.loss_latent_weight = float(wm_cfg.get("loss_latent_weight", 1.0))
            self.loss_action_weight = float(wm_cfg.get("loss_action_weight", 1.0))

            self.world_model = WanWorldModel(
                latent_dim=wm_hidden,                       # per-frame latent C
                action_dim=self.wm_segment_len * raw_action_dim,  # macro-action
                goal_dim=self.task_emb_dim,                 # per-task conditioning
                dim=int(wm_cfg.get("predictor_dim", 384)),
                num_layers=int(wm_cfg.get("predictor_layers", 4)),
                num_heads=int(wm_cfg.get("predictor_heads", 6)),
                ffn_dim=int(wm_cfg.get("predictor_ffn", 1024)),
                ctx_len=self.wm_ctx_len,
                flow_sample_steps=int(wm_cfg.get("flow_sample_steps", 10)),
            )

            # === Multi-view fusion ===
            # encode_frames concatenates the V camera views per frame
            # (V * wm_hidden). Project back to wm_hidden so the world model and
            # action head keep their original width. Initialized to equal-weight
            # averaging so the fused latent initially reproduces the previous
            # mean-over-views behavior exactly -> smooth warm-start from a
            # mean-pool checkpoint; the model then learns per-view weighting.
            self.num_views = int(wm_cfg.get("num_views", 2))
            self.view_fuse = nn.Linear(self.num_views * wm_hidden, wm_hidden)
            with torch.no_grad():
                self.view_fuse.weight.zero_()
                eye = torch.eye(wm_hidden)
                for v in range(self.num_views):
                    self.view_fuse.weight[:, v * wm_hidden : (v + 1) * wm_hidden] = eye / self.num_views
                self.view_fuse.bias.zero_()

            # === Optional state probe (ground latents in physical state) ===
            # Decodes each latent frame back to the robot's proprioceptive
            # state so predicted future latents can be supervised against the
            # *future* physical state. Training-only; predict_action never uses
            # it. Requires the dataloader to pack aligned future ``state``.
            self.use_state_probe = bool(wm_cfg.get("use_state_probe", False))
            if self.use_state_probe:
                self.state_probe_dim = int(wm_cfg.get("state_dim", 8))
                self.loss_state_weight = float(wm_cfg.get("loss_state_weight", 0.5))
                self.state_probe = nn.Sequential(
                    nn.Linear(wm_hidden, wm_hidden),
                    nn.GELU(),
                    nn.Linear(wm_hidden, self.state_probe_dim),
                )
                self.state_loss_fn = nn.MSELoss()
        else:
            self.use_state_probe = False

        # Actions are produced entirely by the world model's flow head; there
        # is no longer an OFT / MLP action head to switch between at inference.
        logger.info("[LeWMOFT] action source = flow (world-model flow head)")

    def _hash_instruction(self, instruction: Optional[str]) -> int:
        """Map an instruction string to a stable embedding-table bucket.

        Uses md5 (not Python ``hash``) so the mapping is deterministic across
        processes/runs regardless of ``PYTHONHASHSEED``.
        """
        text = (instruction or "").strip().lower()
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.num_task_buckets

    def _embed_task(self, instructions: List[str], device: torch.device) -> torch.Tensor:
        ids = torch.tensor(
            [self._hash_instruction(s) for s in instructions],
            device=device,
            dtype=torch.long,
        )
        return self.task_embedding(ids)  # (B, task_emb_dim)

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        device = next(self.parameters()).device
        actions = torch.tensor(np.array(actions), device=device, dtype=torch.float32)
        actions_target = actions[:, -self.action_horizon :, :]

        # === Encode the current frame + future frames into a latent sequence ===
        # frames_per_example[b] = [current_views, future_views_1, ..., future_views_Tf]
        frames_per_example = []
        for example in examples:
            current = example["image"]
            future = example.get("future_images")
            if future is None:
                raise KeyError(
                    "LeWMOFT.use_future_latent=True requires 'future_images' in each "
                    "example (enable future-frame loading in the data config)."
                )
            frames_per_example.append([current] + list(future))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = self.backbone.encode_frames(frames_per_example)  # (B, 1+Tf, V*C)

        with torch.autocast("cuda", dtype=torch.float32):
            latent = latent.float()
            latent = self.view_fuse(latent)  # fuse camera views -> (B, 1+Tf, C)
            task_emb = self._embed_task(instructions, device=latent.device)

            # Macro-actions: split the per-step action chunk into n_future
            # segments, one per predicted future latent (drives frame i->i+1).
            B = latent.shape[0]
            macro_actions = actions_target.reshape(
                B, self.n_future, self.wm_segment_len * actions_target.shape[-1]
            )

            wm_out = self.world_model.flow_loss(
                latent, macro_actions, ctx_len=self.wm_ctx_len, goal=task_emb
            )
            pred_future_latent = wm_out["pred_future_latent"]  # (B, Tf, C)

            # State-probe input: [current real latent, predicted future latents].
            head_tokens = torch.cat([latent[:, : self.wm_ctx_len], pred_future_latent], dim=1)

            flow_latent_loss = wm_out["flow_latent_loss"]
            flow_action_loss = wm_out["flow_action_loss"]
            total_loss = (
                self.loss_latent_weight * flow_latent_loss
                + self.loss_action_weight * flow_action_loss
            )

            # === State probe: ground latents in (future) physical state ===
            # Decode [current real latent, predicted future latents] back to the
            # robot's proprioceptive state and supervise against the aligned
            # current+future states. This pushes the world model's imagined
            # future latents to encode where the arm will actually be.
            if self.use_state_probe:
                raw_states = [example.get("state") for example in examples]
                if any(s is None for s in raw_states):
                    raise KeyError(
                        "LeWMOFT.use_state_probe=True requires 'state' in each example "
                        "(set datasets.vla_data.include_state: true)."
                    )
                states = torch.tensor(
                    np.array(raw_states), device=head_tokens.device, dtype=torch.float32
                )  # (B, 1+Tf, D_state)
                T_head = head_tokens.shape[1]
                state_target = states[:, :T_head, : self.state_probe_dim]
                state_pred = self.state_probe(head_tokens.to(torch.float32))
                state_loss = self.state_loss_fn(state_pred, state_target)
                total_loss = total_loss + self.loss_state_weight * state_loss

        out = {
            "action_loss": total_loss,
            "flow_latent_loss": flow_latent_loss.detach(),
            "flow_action_loss": flow_action_loss.detach(),
        }
        if self.use_state_probe:
            out["state_loss"] = state_loss.detach()
        return out

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # === World-model path: imagine future latents + flow-sample actions ===
        frames_per_example = [[imgs] for imgs in batch_images]  # only current frame
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = self.backbone.encode_frames(frames_per_example)  # (B, 1, V*C)

        with torch.autocast("cuda", dtype=torch.float32):
            latent = latent.float()
            latent = self.view_fuse(latent)  # fuse camera views -> (B, 1, C)
            task_emb = self._embed_task(instructions, device=latent.device)
            _, pred_future_action = self.world_model.sample_future(
                latent, goal=task_emb, n_future=self.n_future, return_action=True
            )  # (B, Tf, wm_segment_len*action_dim)
            # Reshape the flow macro-actions into a per-step chunk:
            # (B, n_future, seg*A) -> (B, horizon, A).
            B = latent.shape[0]
            raw_action_dim = int(self.config.framework.action_model.action_dim)
            pred_actions = pred_future_action.reshape(B, self.action_horizon, raw_action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf
    from PIL import Image

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.name = "LeWMOFT"
    cfg.framework.qwenvl.base_vlm = "WinKawaks/vit-tiny-patch16-224"
    cfg.framework.world_model = {
        "base_wm": "WinKawaks/vit-tiny-patch16-224",
        "train_encoder": False,
    }

    model: LeWM_OFT = LeWM_OFT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "future_images": [[image, image], [image, image]],  # 2 future frames x 2 views
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."
    out = model([sample, sample2])
    print(out)
