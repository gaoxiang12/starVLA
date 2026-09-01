"""TAESD image encoder interface for spatial latent world models."""

from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _TAESD_Interface(nn.Module):
    """Expose deterministic TAESD encoder feature maps to GAWM."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        from diffusers import AutoencoderTiny

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get("base_wm", "madebyollin/taesd")
        self.config = config
        self.train_encoder = bool(wm_cfg.get("train_encoder", False))
        self.image_size = int(wm_cfg.get("encoder_image_size", 256))
        if self.image_size % 8 != 0:
            raise ValueError(
                f"TAESD encoder_image_size must be divisible by 8, got {self.image_size}"
            )

        logger.info(f"Loading TAESD vision encoder from {model_name}")
        vae = AutoencoderTiny.from_pretrained(model_name)
        self.encoder = vae.encoder
        self.feature_dim = int(vae.config.latent_channels)
        self.latent_grid_size = self.image_size // 8
        self.policy_hidden_size = int(wm_cfg.get("policy_hidden_dim", 1536))

        # TAESD emits a wide, shallow map (32x32x4 at 256px).  The action head
        # wants few, deep tokens, so fold each block of pixels into the channel
        # axis rather than projecting 4-d patches.
        self.tokens_per_view = int(
            wm_cfg.get("visual_tokens_per_view", wm_cfg.get("num_visual_tokens", 16))
        )
        tokens_per_side = int(round(self.tokens_per_view**0.5))
        if tokens_per_side**2 != self.tokens_per_view:
            raise ValueError(
                f"TAESD tokens_per_view must be a perfect square, got "
                f"{self.tokens_per_view}"
            )
        if self.latent_grid_size % tokens_per_side != 0:
            raise ValueError(
                f"TAESD latent grid {self.latent_grid_size} is not divisible by "
                f"{tokens_per_side} tokens per side"
            )
        self.tokens_per_side = tokens_per_side
        self.block_size = self.latent_grid_size // tokens_per_side
        self.patch_feature_dim = self.block_size**2 * self.feature_dim

        self.encoder.requires_grad_(self.train_encoder)
        self.encoder.train(self.train_encoder)
        self._model_config = SimpleNamespace(hidden_size=self.policy_hidden_size)

    @property
    def model(self):
        """Compatibility shim for framework code that reads model.config."""
        return SimpleNamespace(config=self._model_config)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_encoder:
            self.encoder.eval()
        return self

    def _to_pixel_values(self, flat: List, device: torch.device) -> torch.Tensor:
        """Resize RGB images and map pixel values from [0, 1] to [-1, 1]."""
        from torchvision.transforms import v2 as T

        transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize((self.image_size, self.image_size), antialias=True),
            ]
        )
        pixel_values = torch.stack([transform(image) for image in flat], dim=0)
        return pixel_values.to(device=device).mul_(2.0).sub_(1.0)

    @staticmethod
    def _flatten_frames(frames_per_example: List):
        flat = []
        num_frames = None
        num_views = None
        for frames in frames_per_example:
            if num_frames is None:
                num_frames = len(frames)
            elif len(frames) != num_frames:
                raise ValueError("all examples must contain the same number of frames")
            for frame in frames:
                views = frame if isinstance(frame, (list, tuple)) else [frame]
                if num_views is None:
                    num_views = len(views)
                elif len(views) != num_views:
                    raise ValueError("all frames must contain the same number of views")
                flat.extend(views)
        return flat, int(num_frames or 0), int(num_views or 0)

    def encode_feature_frames(self, frames_per_example: List) -> torch.Tensor:
        """Return TAESD maps with shape (B, T, V, C, H/8, W/8)."""
        flat, num_frames, num_views = self._flatten_frames(frames_per_example)
        if not flat:
            raise ValueError("frames_per_example must not be empty")
        device = next(self.encoder.parameters()).device
        pixel_values = self._to_pixel_values(flat, device)
        with torch.set_grad_enabled(self.train_encoder):
            features = self.encoder(pixel_values)

        batch_size = len(frames_per_example)
        return features.view(
            batch_size,
            num_frames,
            num_views,
            self.feature_dim,
            features.shape[-2],
            features.shape[-1],
        )

    def encode_patch_frames(self, frames_per_example: List) -> torch.Tensor:
        """Return space-to-depth tokens as (B, T, V, tokens_per_view, C*block^2)."""
        features = self.encode_feature_frames(frames_per_example)
        batch, num_frames, num_views = features.shape[:3]
        block, side = self.block_size, self.tokens_per_side
        tokens = features.view(
            batch, num_frames, num_views, self.feature_dim, side, block, side, block
        )
        tokens = tokens.permute(0, 1, 2, 4, 6, 3, 5, 7)
        return tokens.reshape(
            batch, num_frames, num_views, self.tokens_per_view, self.patch_feature_dim
        )

    def build_inputs(self, images: List, instructions: List, **kwargs):
        flat = []
        num_views = None
        for sample in images:
            views = sample if isinstance(sample, (list, tuple)) else [sample]
            num_views = len(views) if num_views is None else num_views
            if len(views) != num_views:
                raise ValueError("all samples must contain the same number of views")
            flat.extend(views)
        device = next(self.encoder.parameters()).device
        return {
            "pixel_values": self._to_pixel_values(flat, device),
            "n_views": num_views,
            "_is_wm_input": True,
        }

    def forward(self, **kwargs):
        kwargs.pop("_is_wm_input", False)
        num_views = int(kwargs.pop("n_views", 1))
        pixel_values = kwargs["pixel_values"]
        with torch.set_grad_enabled(self.train_encoder):
            features = self.encoder(pixel_values)
        batch_size = features.shape[0] // num_views
        latent = features.flatten(1).view(batch_size, num_views, -1)
        return SimpleNamespace(hidden_states=(latent,))
