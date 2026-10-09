# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""GAWM-L world-model components.

_GAWM_Interface extracts DINO image features. TokenResidualPredictor is the
shared future-token predictor. FixedDinoWorldModel supplies the current frozen
patch-teacher objective; CompactFixedDinoWorldModel adds the study options.
VisualTokenLatentWorldModel retains the legacy adapter-latent objective.
The outer framework assembles these components with the visual adapter and ACT.
"""

import math
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.gawm_config import migrate_gawm_config
from starVLA.training.trainer_utils import initialize_overwatch
from .temporal_regularization import temporal_curvature_loss

logger = initialize_overwatch(__name__)


# DINO feature encoder

class _GAWM_Interface(nn.Module):
    """World-model wrapper exposing a ViT encoder as a feature backbone."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        config = migrate_gawm_config(config)
        wm_cfg = config.framework.get("world_model", {})
        self.config = config
        self.train_encoder = bool(wm_cfg.get("train_encoder", False))
        self.gawm_l_vision = wm_cfg.get("visual_frontend", "grid") == "gawm_l"
        self.feat_layers = tuple(wm_cfg.get("feat_layers", [-12, -8, -4]))
        self.gawm_l_image_size = tuple(wm_cfg.get("gawm_l_image_size", [320, 240]))
        if self.gawm_l_vision and self.train_encoder:
            raise ValueError("GAWM-L visual alignment requires a frozen DINO encoder")
        encoder_spec = str(wm_cfg.get("encoder_spec", "vitb16")).strip().lower()
        model_name = wm_cfg.get("base_wm")
        hf_path = wm_cfg.get("vision_encoder_path")
        self.normalized_pixels = bool(wm_cfg.get("imagenet_normalized_inputs", False))
        self.encoder_batch_size = int(wm_cfg.get("encoder_batch_size", 0))

        from .dinov3_loader import build_dinov3, load_dinov3, spec_from_filename

        if hf_path:
            if model_name:
                raise ValueError("Choose either vision_encoder_path or base_wm")
            from starVLA.model.dinov3_assets import resolve_dinov3_path
            hf_path = resolve_dinov3_path(hf_path)
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(hf_path, local_files_only=True, torch_dtype=torch.bfloat16)
            if self.encoder.config.model_type != "dinov3_vit":
                raise ValueError("Expected a DINOv3 ViT checkpoint")
            from .dinov3_loader import SPECS
            spec = SPECS[encoder_spec]
            if (self.encoder.config.hidden_size, self.encoder.config.num_hidden_layers) != (spec['hidden'], spec['layers']):
                raise ValueError("HF checkpoint does not match encoder_spec")
            self.processor = None
            num_register = self.encoder.config.num_register_tokens
        elif model_name:
            model_path = Path(model_name).expanduser()
            if not model_path.is_absolute() and not model_path.exists():
                repo_relative = Path(__file__).resolve().parents[4] / model_path
                if repo_relative.exists():
                    model_path = repo_relative
            model_name = os.fspath(model_path)
            inferred_spec = spec_from_filename(model_name)
            if inferred_spec != encoder_spec:
                raise ValueError(
                    f"encoder_spec={encoder_spec!r} does not match "
                    f"DINO checkpoint architecture {inferred_spec!r}"
                )
            logger.info(
                f"Initializing DINOv3 {encoder_spec} from raw checkpoint {model_name}"
            )
            self.encoder, self.processor, num_register = load_dinov3(model_name)
        else:
            logger.info(
                f"Building DINOv3 {encoder_spec}; weights will come from the "
                "unified GAWM checkpoint"
            )
            self.encoder, self.processor, num_register = build_dinov3(encoder_spec)
        self.num_prefix_tokens = 1 + num_register

        vit_hidden = self.encoder.config.hidden_size
        # GAWM latent = concat(cls, mean-pool patches) -> 2 * hidden
        self._hidden_size = vit_hidden * 2

        # Frozen by default; keep joint fine-tuning interface available.
        self.encoder.requires_grad_(self.train_encoder)
        self.encoder.train(self.train_encoder)

        class _FakeConfig:
            pass

        self._model_config = _FakeConfig()
        self._model_config.hidden_size = self._hidden_size

    @property
    def model(self):
        """Compat shim: framework reads self.backbone.model.config.hidden_size."""

        class _ModelShim:
            pass

        shim = _ModelShim()
        shim.config = self._model_config
        return shim

    def _to_pixel_values(self, flat: List, device=None) -> torch.Tensor:
        """Preprocess a flat list of PIL images into a (N, C, H, W) tensor."""
        if self.normalized_pixels:
            values = torch.stack(list(flat))
            if values.ndim != 4 or values.shape[1] != 3 or not values.is_floating_point():
                raise ValueError("Expected ImageNet-normalized CHW float tensors")
            return values
        if self.gawm_l_vision:
            from starVLA.model.modules.gawm_l_vision import rgb_pixels
            return rgb_pixels(flat, self.gawm_l_image_size)
        if self.processor is not None:
            # Run resize/normalize on the encoder's device (GPU) when possible to
            # avoid a CPU-bound bottleneck (worsened by OMP_NUM_THREADS=1) when
            # encoding many views per forward pass.
            try:
                kwargs = {"images": flat, "return_tensors": "pt"}
                if device is not None and getattr(device, "type", None) == "cuda":
                    kwargs["device"] = device
                pixel_values = self.processor(**kwargs).pixel_values
            except TypeError:
                pixel_values = self.processor(images=flat, return_tensors="pt").pixel_values
        else:
            from torchvision.transforms import v2 as T

            tf = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize((224, 224), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            pixel_values = torch.stack([tf(im) for im in flat], dim=0)
        return pixel_values

    def build_inputs(self, images: List, instructions: List, **kwargs):
        """Preprocess multi-view images into ViT pixel values.

        ``images`` is a list of B examples; each example is a single PIL image
        or a list of V views. All views are flattened to (B * V, C, H, W).
        ``instructions`` is ignored (no text encoder in this front-end).
        """
        flat, n_views = [], None
        for sample in images:
            views = sample if isinstance(sample, (list, tuple)) else [sample]
            n_views = len(views) if n_views is None else n_views
            flat.extend(views)

        device = next(self.encoder.parameters()).device
        pixel_values = self._to_pixel_values(flat, device)

        return {
            "pixel_values": pixel_values.to(device),
            "n_views": n_views,
            "_is_wm_input": True,
        }

    def encode_frames(self, frames_per_example: List) -> torch.Tensor:
        """Encode a temporal sequence of multi-view frames into per-frame latents.

        ``frames_per_example`` is a list of B examples; each example is a list
        of T frames; each frame is a single image or a list of V views. All
        ``B * T * V`` views are encoded in a single ViT pass.

        Returns ``latent (B, T, V*2*hidden)`` where each frame's latent is the
        concatenation over its V views of ``concat([CLS, mean-pool(patches)])``.
        The framework projects this back to ``2*hidden`` via a learned
        view-fusion layer (preserving per-camera identity).
        """
        flat, T, V = [], None, None
        for frames in frames_per_example:
            T = len(frames) if T is None else T
            for frame in frames:
                views = frame if isinstance(frame, (list, tuple)) else [frame]
                V = len(views) if V is None else V
                flat.extend(views)

        device = next(self.encoder.parameters()).device
        pixel_values = self._to_pixel_values(flat, device).to(device)

        with torch.set_grad_enabled(self.train_encoder):
            out = self.encoder(pixel_values=pixel_values)
            hidden = out.last_hidden_state            # (B*T*V, prefix+N, D)
            cls = hidden[:, 0]                         # (B*T*V, D)
            pooled = hidden[:, self.num_prefix_tokens:].mean(dim=1)  # (B*T*V, D)
            vec = torch.cat([cls, pooled], dim=-1)     # (B*T*V, 2D)

        B = len(frames_per_example)
        # Concatenate views along the feature dim to preserve per-camera
        # identity (mean-pool would discard which camera saw what). The
        # framework fuses V*2D -> 2D via a learned projection.
        vec = vec.view(B, T, V, vec.shape[-1]).reshape(B, T, V * vec.shape[-1])  # (B, T, V*2D)
        return vec

    def encode_patch_frames(self, frames_per_example: List, *, return_teacher=False) -> torch.Tensor:
        """Encode frames into raw per-view patch tokens.

        ``frames_per_example`` has the same structure as ``encode_frames``.
        Returns ``patches (B, T, V, N, D)`` where ``N`` is the encoder patch
        grid length and ``D`` is the ViT/DINO hidden size.
        """
        flat, T, V = [], None, None
        for frames in frames_per_example:
            T = len(frames) if T is None else T
            for frame in frames:
                views = frame if isinstance(frame, (list, tuple)) else [frame]
                V = len(views) if V is None else V
                flat.extend(views)

        device = next(self.encoder.parameters()).device
        pixel_values = self._to_pixel_values(flat, device).to(device)

        return self._encode_patch_pixel_values(
            pixel_values,
            batch_size=len(frames_per_example),
            time_steps=T,
            num_views=V,
            return_teacher=return_teacher,
        )

    def encode_patch_image_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of raw image tensors into per-view patch tokens.

        This is the replay-friendly counterpart of :meth:`encode_patch_frames`.
        It avoids converting rollout images back to PIL during RL updates while
        preserving the exact HuggingFace DINOv3 image processor used at
        inference time.

        Args:
            images: Raw images shaped ``[B, T, V, H, W, C]`` or
                ``[B, T, V, C, H, W]``. ``uint8`` and floating tensors are
                accepted by the DINOv3 fast image processor.

        Returns:
            Patch tokens shaped ``[B, T, V, N, D]``.
        """
        if not isinstance(images, torch.Tensor) or images.ndim != 6:
            raise ValueError(
                "images must be a rank-6 tensor [B,T,V,H,W,C] or "
                f"[B,T,V,C,H,W], got {type(images).__name__} "
                f"with shape {getattr(images, 'shape', None)}"
            )

        batch_size, time_steps, num_views = images.shape[:3]
        if images.shape[-1] in (1, 3, 4):
            flat = images.reshape(-1, *images.shape[-3:])
        elif images.shape[3] in (1, 3, 4):
            flat = images.reshape(-1, *images.shape[-3:])
        else:
            raise ValueError(
                "Cannot infer image channel axis from shape "
                f"{tuple(images.shape)}; expected C in {{1,3,4}}."
            )

        device = next(self.encoder.parameters()).device
        pixel_values = self._to_pixel_values(flat, device).to(device)
        return self._encode_patch_pixel_values(
            pixel_values,
            batch_size=batch_size,
            time_steps=time_steps,
            num_views=num_views,
        )

    def _encode_patch_pixel_values(
        self,
        pixel_values: torch.Tensor,
        *,
        batch_size: int,
        time_steps: int,
        num_views: int,
        return_teacher: bool = False,
    ) -> torch.Tensor:
        """Run DINOv3 on already preprocessed pixels and restore B/T/V axes."""
        if self.normalized_pixels:
            pixel_values = pixel_values.to(dtype=next(self.encoder.parameters()).dtype)
        with torch.set_grad_enabled(self.train_encoder):
            chunk_size = self.encoder_batch_size or len(pixel_values)
            if self.gawm_l_vision:
                encoded, teachers = [], []
                for chunk in pixel_values.split(chunk_size):
                    outputs = self.encoder(pixel_values=chunk.to(next(self.encoder.parameters()).dtype),
                                           output_hidden_states=True, return_dict=True)
                    encoded.append(torch.stack([outputs.hidden_states[i] for i in self.feat_layers], dim=1))
                    if return_teacher:
                        teachers.append(outputs.last_hidden_state[:, self.num_prefix_tokens:].detach())
                features = torch.cat(encoded)
                features = features.reshape(batch_size, time_steps, num_views, *features.shape[1:])
                if return_teacher:
                    teacher = torch.cat(teachers)
                    return features, teacher.reshape(batch_size, time_steps, num_views, *teacher.shape[1:])
                return features
            patches = torch.cat([
                self.encoder(pixel_values=chunk).last_hidden_state[:, self.num_prefix_tokens:]
                for chunk in pixel_values.split(chunk_size)
            ])

        return patches.view(
            batch_size,
            time_steps,
            num_views,
            patches.shape[-2],
            patches.shape[-1],
        )

    def forward(self, **kwargs):
        """Encode views; return hidden states (B, V, 2*hidden) for the action head."""
        kwargs.pop("_is_wm_input", False)
        kwargs.pop("output_hidden_states", False)
        kwargs.pop("return_dict", True)
        n_views = int(kwargs.pop("n_views", 1))
        pixel_values = kwargs["pixel_values"]

        with torch.set_grad_enabled(self.train_encoder):
            out = self.encoder(pixel_values=pixel_values)
            hidden = out.last_hidden_state            # (BV, prefix+N, D)
            cls = hidden[:, 0]                         # (BV, D)
            pooled = hidden[:, self.num_prefix_tokens:].mean(dim=1)  # (BV, D)
            vec = torch.cat([cls, pooled], dim=-1)     # (BV, 2D)

        bv = vec.shape[0]
        b = bv // n_views
        latent = vec.view(b, n_views, vec.shape[-1])   # (B, V, 2D)

        class _WMOutput:
            def __init__(self, hidden_states_tuple):
                self.hidden_states = hidden_states_tuple

        return _WMOutput(hidden_states_tuple=(latent,))


# Shared future-token predictor

class _TokenResidualBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(hidden)
        hidden = hidden + self.attn(
            normalized, normalized, normalized, need_weights=False
        )[0]
        return hidden + self.mlp(self.norm2(hidden))


class TokenResidualPredictor(nn.Module):
    """Predict future visual-token residuals from one current frame and task."""

    def __init__(
        self,
        *,
        latent_dim: int,
        goal_dim: Optional[int],
        n_future: int,
        num_tokens: int,
        dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.n_future = int(n_future)
        self.num_tokens = int(num_tokens)
        self.anchor_proj = nn.Linear(latent_dim, dim)
        self.goal_proj = nn.Linear(goal_dim, dim) if goal_dim else None
        self.future_query = nn.Parameter(
            torch.randn(1, self.n_future, 1, dim) * 0.02
        )
        self.frame_embedding = nn.Parameter(
            torch.randn(1, 1 + self.n_future, 1, dim) * 0.02
        )
        self.token_embedding = nn.Parameter(
            torch.randn(1, 1, self.num_tokens, dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [_TokenResidualBlock(dim, num_heads, ffn_dim) for _ in range(int(depth))]
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        context: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, context_len, num_tokens, _ = context.shape
        if context_len != 1:
            raise ValueError(f"GAWM expects one context frame, got {context_len}")
        if num_tokens != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} context tokens, got {num_tokens}")

        current_tokens = self.anchor_proj(context)
        future_tokens = self.future_query.expand(batch_size, -1, num_tokens, -1)
        hidden = torch.cat([current_tokens, future_tokens], dim=1)
        hidden = hidden + self.frame_embedding + self.token_embedding
        if goal is not None and self.goal_proj is not None:
            hidden = hidden + self.goal_proj(goal.to(hidden.dtype)).view(
                batch_size, 1, 1, -1
            )
        hidden = hidden.reshape(batch_size, (1 + self.n_future) * num_tokens, -1)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden).view(
            batch_size, 1 + self.n_future, num_tokens, -1
        )
        return self.out(hidden[:, 1:])


# Fixed DINO patch supervision

class PatchFeatureDecoder(nn.Module):
    """Shared per-camera spatial readout; no current-image skip or teacher input."""
    def __init__(self, latent_dim, feature_dim, num_patches, num_views,
                 tokens_per_view, dim=384, depth=2, heads=6):
        super().__init__()
        self.num_views = num_views
        self.tokens_per_view = tokens_per_view
        self.query = nn.Parameter(torch.randn(1, num_patches, dim) * .02)
        self.memory = nn.Linear(latent_dim, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0.,
                                          batch_first=True, norm_first=True, activation='gelu')
        self.decoder = nn.TransformerDecoder(layer, depth, norm=nn.LayerNorm(dim))
        self.output = nn.Linear(dim, feature_dim)

    def forward(self, latent):
        b, t, k, d = latent.shape
        if k != self.num_views * self.tokens_per_view:
            raise ValueError('Decoder camera/token layout mismatch')
        memory = self.memory(latent.reshape(b*t*self.num_views, self.tokens_per_view, d))
        decoded = self.decoder(self.query.expand(memory.shape[0], -1, -1), memory)
        features = self.output(decoded)
        return features.reshape(b, t, self.num_views, *features.shape[-2:])


class FixedDinoWorldModel(nn.Module):
    """Future loss trains current adapter, predictor and decoder; teacher is detached.

    No adapter-space target or target-dependent EMA scale is used. Predictions
    use a fixed residual multiplier and non-affine LayerNorm at train AND test.
    """
    def __init__(self, *, latent_dim, goal_dim, n_future, num_tokens, num_views,
                 tokens_per_view, feature_dim, num_patches, dim=384, depth=4,
                 num_heads=6, ffn_dim=1024, decoder_dim=384, decoder_depth=2,
                 decoder_heads=6, reconstruction_weight=.1, smoothness_weight=0.,
                 time_offsets=(0., .2, .4), dense_smoothness_weight=0.,
                 temporal_reference_dt=.05):
        super().__init__()
        self.num_views, self.tokens_per_view = num_views, tokens_per_view
        self.num_tokens, self.n_future = num_tokens, n_future
        if n_future != 2:
            raise ValueError('Two fixed future horizons are required')
        self.reconstruction_weight = float(reconstruction_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.dense_smoothness_weight = float(dense_smoothness_weight)
        self.temporal_reference_dt = float(temporal_reference_dt)
        if not math.isfinite(self.temporal_reference_dt) or self.temporal_reference_dt <= 0:
            raise ValueError("temporal_reference_dt must be finite and positive")
        offsets = torch.tensor(time_offsets, dtype=torch.float32)
        if offsets.shape != (3,) or not torch.isfinite(offsets).all() or not torch.all(offsets[1:] > offsets[:-1]):
            raise ValueError('Three finite increasing temporal offsets are required')
        self.register_buffer('time_offsets', offsets)
        self.register_buffer('objective_version', torch.tensor(1, dtype=torch.uint8))
        self.residual_predictor = TokenResidualPredictor(latent_dim=latent_dim, goal_dim=goal_dim,
            n_future=n_future, num_tokens=num_tokens, dim=dim, depth=depth,
            num_heads=num_heads, ffn_dim=ffn_dim)
        self.feature_decoder = PatchFeatureDecoder(latent_dim, feature_dim, num_patches,
            num_views, tokens_per_view, decoder_dim, decoder_depth, decoder_heads)

    def regress_future(self, context, goal=None):
        if context.shape[1:] != (1, self.num_tokens, self.residual_predictor.out.out_features):
            raise ValueError('Expected a single current frame with the configured tokens')
        future = context + self.residual_predictor(context, goal=goal)
        return F.layer_norm(future.float(), (future.shape[-1],)).to(future.dtype)

    @staticmethod
    def cosine_error(prediction, target):
        return 1. - F.cosine_similarity(prediction.float(), target.detach().float(), dim=-1, eps=1e-6)

    @staticmethod
    def masked_mean(error, view_mask):
        weights = view_mask.to(error.dtype).unsqueeze(-1).expand_as(error)
        return (error * weights).sum() / weights.sum().clamp_min(1.)

    def forward(self, latent, *, ctx_len, goal=None,
                loss_mask=None, teacher_patches=None, temporal_latent=None, temporal_times=None, temporal_valid=None, temporal_event_weight=None):
        b, t, k, _ = latent.shape
        if (ctx_len, t, k) != (1, 3, self.num_tokens):
            raise ValueError('Expected current and two future latent frames')
        if teacher_patches is None or teacher_patches.shape[:3] != (b, t, self.num_views):
            raise ValueError('Frozen DINO patch teacher is required in training')
        teacher = teacher_patches.detach()
        if loss_mask is None:
            valid = torch.ones(b, t, self.num_views, device=latent.device, dtype=torch.bool)
        else:
            grouped = loss_mask.reshape(b, t, self.num_views, self.tokens_per_view)
            if not torch.equal(grouped, grouped[..., :1].expand_as(grouped)):
                raise ValueError('Patch supervision requires uniform validity within each camera')
            valid = grouped[..., 0].bool()
        future = self.regress_future(latent[:, :1], goal)
        decoded = self.feature_decoder(torch.cat([latent[:, :1], future], dim=1))
        if decoded.shape != teacher.shape:
            raise ValueError(f'Decoder/teacher shape mismatch: {decoded.shape} != {teacher.shape}')
        errors = self.cosine_error(decoded, teacher)
        future_loss = self.masked_mean(errors[:, 1:], valid[:, 1:])
        current_loss = self.masked_mean(errors[:, :1], valid[:, :1])
        # Coarse three-time-point smoothness only, not a claim of frame-level smoothness.
        sequence = torch.cat([latent[:, :1], future], dim=1).float()
        dt = (self.time_offsets[1:] - self.time_offsets[:-1]).to(sequence.device)
        velocities = (sequence[:, 1:] - sequence[:, :-1]) / dt.view(1, 2, 1, 1)
        bend = (velocities[:, 1] - velocities[:, 0]) * dt.mean()
        smooth_error = F.smooth_l1_loss(bend, torch.zeros_like(bend), reduction='none', beta=.1)
        smooth_error = smooth_error.reshape(b, self.num_views, self.tokens_per_view, -1).mean(-1)
        smooth_loss = self.masked_mean(smooth_error, valid.all(dim=1))
        aux_loss = self.reconstruction_weight * current_loss + self.smoothness_weight * smooth_loss
        dense_metrics = {}
        if self.dense_smoothness_weight > 0:
            if temporal_latent is None:
                raise ValueError("Adjacent temporal supervision is missing")
            repeated_goal = goal.repeat_interleave(3, dim=0) if goal is not None else None
            temporal_predictions = self.regress_future(
                temporal_latent.reshape(b*3, 1, self.num_tokens, -1), repeated_goal
            ).reshape(b, 3, self.n_future, self.num_tokens, -1)
            observed_smooth = temporal_curvature_loss(temporal_latent, temporal_times, temporal_valid, temporal_event_weight, reference_dt=self.temporal_reference_dt)
            predicted_smooth = temporal_curvature_loss(temporal_predictions, temporal_times, temporal_valid, temporal_event_weight, reference_dt=self.temporal_reference_dt)
            dense_loss = .5 * (observed_smooth + predicted_smooth)
            aux_loss = aux_loss + self.dense_smoothness_weight * dense_loss
            dense_metrics = dict(temporal_dense_loss=dense_loss,
                temporal_dense_current_loss=observed_smooth, temporal_dense_prediction_loss=predicted_smooth,
                temporal_dense_weighted_loss=self.dense_smoothness_weight*dense_loss,
                temporal_dense_valid_fraction=temporal_valid.float().mean(),
                temporal_dense_event_weight=temporal_event_weight.mean())
        result = dict(pred_future_latent=future, latent_loss=future_loss,
                      latent_cosine_loss=future_loss.new_zeros(()), auxiliary_loss=aux_loss,
                      dino_future_loss=future_loss, dino_current_loss=current_loss,
                      temporal_smoothness_loss=smooth_loss)
        result.update(dense_metrics)
        for h in range(2):
            result[f'latent_loss_horizon_{h+1}'] = self.masked_mean(errors[:, h+1:h+2], valid[:, h+1:h+2])
        with torch.no_grad():
            copy_error = self.cosine_error(teacher[:, :1].expand_as(teacher[:, 1:]), teacher[:, 1:])
            copy_loss = self.masked_mean(copy_error, valid[:, 1:])
            weights = valid[:, 1:, :, None, None].float()
            mean_teacher = (teacher[:, 1:].float()*weights).sum(0, keepdim=True)/weights.sum(0, keepdim=True).clamp_min(1.)
            mean_loss = self.masked_mean(self.cosine_error(mean_teacher.expand_as(teacher[:, 1:]), teacher[:, 1:]), valid[:, 1:])
            change = (latent[:, 1:].float()-latent[:, :-1].float()).square().mean(-1)
            change = change.reshape(b, 2, self.num_views, self.tokens_per_view)
            result.update(dino_copy_loss=copy_loss, dino_batch_mean_loss=mean_loss,
                dino_to_copy_ratio=future_loss.detach()/copy_loss.clamp_min(1e-6),
                predicted_latent_rms=future.float().square().mean().sqrt(),
                latent_batch_std=latent[:, 0].float().var(0, unbiased=False).mean().sqrt(),
                temporal_observed_delta_rms=self.masked_mean(change, valid[:, 1:] & valid[:, :-1]).sqrt())
        return result


# Optional compact-study extensions

def temporal_rope(value, positions, rotary_dim, theta=10000.):
    """Rotate Q/K [B,H,L,D] using real temporal coordinates, never query IDs."""
    if rotary_dim < 2 or rotary_dim % 2 or rotary_dim > value.shape[-1]:
        raise ValueError("RoPE dimension must be positive, even, and within head width")
    frequency = theta ** (-torch.arange(0, rotary_dim, 2, device=value.device, dtype=torch.float32) / rotary_dim)
    angles = positions.to(device=value.device, dtype=torch.float32)[:, None] * frequency[None]
    cos, sin = angles.cos().to(value.dtype)[None, None], angles.sin().to(value.dtype)[None, None]
    pairs = value[..., :rotary_dim].reshape(*value.shape[:-1], rotary_dim // 2, 2)
    first, second = pairs.unbind(-1)
    rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), -1).flatten(-2)
    return torch.cat((rotated, value[..., rotary_dim:]), -1)


def rope_residual_block(block, hidden, positions):
    normalized = block.norm1(hidden)
    attn = block.attn
    batch, length, dim = normalized.shape
    q, k, v = F.linear(normalized, attn.in_proj_weight, attn.in_proj_bias).chunk(3, -1)
    q, k, v = [x.reshape(batch, length, attn.num_heads, attn.head_dim).transpose(1, 2) for x in (q, k, v)]
    rotary_dim = max(2, (attn.head_dim // 2) // 2 * 2)
    q = temporal_rope(q, positions, rotary_dim)
    k = temporal_rope(k, positions, rotary_dim)
    attended = F.scaled_dot_product_attention(q, k, v, dropout_p=attn.dropout if block.training else 0., is_causal=False)
    attended = attended.transpose(1, 2).reshape(batch, length, dim)
    hidden = hidden + F.linear(attended, attn.out_proj.weight, attn.out_proj.bias)
    return hidden + block.mlp(block.norm2(hidden))


class CompactTokenResidualPredictor(TokenResidualPredictor):
    def configure(self, *, state_dim=0, use_rope=False, history_frames=1, time_offsets=(0,16,32), history_offset=16):
        self.use_rope = bool(use_rope)
        self.history_frames = int(history_frames)
        if self.history_frames not in (1, 2):
            raise ValueError("The study supports one or two observation frames")
        self.state_dim = int(state_dim)
        self.history_offset = float(history_offset)
        offsets = tuple(float(x) for x in time_offsets)
        if len(offsets) != 3 or offsets[0] != 0 or not 0 < offsets[1] < offsets[2]:
            raise ValueError("Expected current and two increasing future times")
        self.rope_times = offsets
        self.rope_time_scale = offsets[1]
        dim = self.anchor_proj.out_features
        # Adding a condition must not perturb initialization of shared modules.
        with torch.random.fork_rng(devices=[]):
            self.state_projection = (nn.Sequential(nn.Linear(state_dim, 128), nn.GELU(), nn.Linear(128, dim))
                                     if state_dim else None)
            if self.history_frames == 2:
                self.history_embedding = nn.Parameter(torch.randn(1, 1, 1, dim) * .02)
        return self

    def forward(self, context, goal=None, *, state=None, history=None):
        batch, frames, tokens, _ = context.shape
        if frames != 1 or tokens != self.num_tokens:
            raise ValueError("Expected one current latent frame")
        current = self.anchor_proj(context)
        future = self.future_query.expand(batch, -1, tokens, -1)
        hidden = torch.cat((current, future), 1) + self.frame_embedding + self.token_embedding
        times = list(self.rope_times)
        history_count = self.history_frames - 1
        if history_count:
            if history is None or history.shape != context.shape:
                raise ValueError("Two-frame prediction requires an explicit past latent frame")
            past = self.anchor_proj(history) + self.history_embedding + self.token_embedding
            hidden = torch.cat((past, hidden), 1)
            times.insert(0, -self.history_offset)
        if goal is not None and self.goal_proj is not None:
            hidden = hidden + self.goal_proj(goal.to(hidden.dtype))[:, None, None]
        hidden = hidden.reshape(batch, len(times) * tokens, -1)
        positions = torch.tensor(times, device=hidden.device).repeat_interleave(tokens) / self.rope_time_scale
        if self.state_projection is not None:
            if state is None or state.shape != (batch, self.state_dim):
                raise ValueError("World-model state condition is missing or malformed")
            state_token = self.state_projection(state.to(hidden.dtype))[:, None]
            hidden = torch.cat((hidden, state_token), 1)
            positions = torch.cat((positions, positions.new_zeros(1)))
        for block in self.blocks:
            hidden = rope_residual_block(block, hidden, positions) if self.use_rope else block(hidden)
        hidden = self.norm(hidden[:, :len(times)*tokens]).reshape(batch, len(times), tokens, -1)
        return self.out(hidden[:, history_count + 1:])


class CompactFixedDinoWorldModel(FixedDinoWorldModel):
    def __init__(self, *, compact_options, state_dim=0, **kwargs):
        super().__init__(**kwargs)
        self.compact_options = dict(compact_options)
        self.motion_weight = float(self.compact_options.get("motion_weight", 0.))
        if not math.isfinite(self.motion_weight) or not 0 <= self.motion_weight <= 2:
            raise ValueError("Motion weighting strength must be in [0,2]")
        # Keep every existing parameter name and initialization. Only opt-in
        # state/history parameters are added to the existing predictor.
        self.residual_predictor.__class__ = CompactTokenResidualPredictor
        self.residual_predictor.configure(state_dim=state_dim if self.compact_options.get("wm_state") else 0,
            use_rope=self.compact_options.get("temporal_rope", False),
            history_frames=self.compact_options.get("history_frames", 1),
            time_offsets=kwargs.get("time_offsets", (0.,16.,32.)), history_offset=16.)

    def regress_future(self, context, goal=None, *, state=None, history=None):
        future = context + self.residual_predictor(context, goal, state=state, history=history)
        return F.layer_norm(future.float(), (future.shape[-1],)).to(future.dtype)

    def forward(self, latent, *, ctx_len, goal=None, loss_mask=None,
                teacher_patches=None, temporal_latent=None, temporal_times=None,
                temporal_valid=None, temporal_event_weight=None, state=None, history=None,
                temporal_state=None, temporal_history=None):
        batch, frames, tokens, _ = latent.shape
        if (ctx_len, frames, tokens) != (1, 3, self.num_tokens):
            raise ValueError("Expected current and two future supervised frames")
        if teacher_patches is None or teacher_patches.shape[:3] != (batch, frames, self.num_views):
            raise ValueError("Frozen patch teacher is required")
        teacher = teacher_patches.detach()
        if loss_mask is None:
            valid = torch.ones(batch, frames, self.num_views, device=latent.device, dtype=torch.bool)
        else:
            grouped = loss_mask.reshape(batch, frames, self.num_views, self.tokens_per_view)
            if not torch.equal(grouped, grouped[..., :1].expand_as(grouped)):
                raise ValueError("Validity must be uniform within each camera")
            valid = grouped[..., 0].bool()
        future = self.regress_future(latent[:, :1], goal, state=state, history=history)
        decoded = self.feature_decoder(torch.cat((latent[:, :1], future), 1))
        if decoded.shape != teacher.shape:
            raise ValueError("Decoded/teacher feature shapes differ")
        errors = self.cosine_error(decoded, teacher)
        uniform_loss = self.masked_mean(errors[:, 1:], valid[:, 1:])
        future_loss = uniform_loss
        if self.motion_weight:
            change = self.cosine_error(teacher[:, :1].expand_as(teacher[:, 1:]), teacher[:, 1:]).clamp_min(0.)
            relative_change = change / change.mean(-1, keepdim=True).clamp_min(1e-4)
            # Detached bounded weights retain at least unit background weight.
            weights = (1. + self.motion_weight * relative_change.clamp(max=3.)).detach()
            weights = weights * valid[:, 1:, :, None].to(weights.dtype)
            future_loss = (errors[:, 1:] * weights).sum() / weights.sum().clamp_min(1.)
        current_loss = self.masked_mean(errors[:, :1], valid[:, :1])
        sequence = torch.cat((latent[:, :1], future), 1).float()
        dt = (self.time_offsets[1:] - self.time_offsets[:-1]).to(sequence.device)
        velocities = (sequence[:, 1:] - sequence[:, :-1]) / dt.view(1, 2, 1, 1)
        bend = (velocities[:, 1] - velocities[:, 0]) * dt.mean()
        smooth_error = F.smooth_l1_loss(bend, torch.zeros_like(bend), reduction="none", beta=.1)
        smooth_error = smooth_error.reshape(batch, self.num_views, self.tokens_per_view, -1).mean(-1)
        smooth_loss = self.masked_mean(smooth_error, valid.all(1))
        auxiliary = self.reconstruction_weight * current_loss + self.smoothness_weight * smooth_loss
        metrics = {}
        if self.dense_smoothness_weight:
            if temporal_latent is None:
                raise ValueError("Adjacent temporal supervision missing")
            repeated_goal = goal.repeat_interleave(3, 0) if goal is not None else None
            states = None
            if self.residual_predictor.state_dim:
                if temporal_state is None or temporal_state.shape != (batch, 3, self.residual_predictor.state_dim):
                    raise ValueError("Each temporal anchor requires its own observed state")
                states = temporal_state.flatten(0, 1)
            histories = None
            if self.residual_predictor.history_frames == 2:
                if temporal_history is None or temporal_history.shape != temporal_latent.shape:
                    raise ValueError("Each temporal anchor requires its own past observation")
                histories = temporal_history.reshape(batch*3, 1, tokens, -1)
            predicted = self.regress_future(temporal_latent.reshape(batch*3, 1, tokens, -1), repeated_goal,
                state=states, history=histories).reshape(batch, 3, self.n_future, tokens, -1)
            observed_smooth = temporal_curvature_loss(temporal_latent, temporal_times, temporal_valid,
                temporal_event_weight, reference_dt=self.temporal_reference_dt)
            predicted_smooth = temporal_curvature_loss(predicted, temporal_times, temporal_valid,
                temporal_event_weight, reference_dt=self.temporal_reference_dt)
            dense_loss = .5 * (observed_smooth + predicted_smooth)
            auxiliary = auxiliary + self.dense_smoothness_weight * dense_loss
            metrics.update(temporal_dense_loss=dense_loss, temporal_dense_current_loss=observed_smooth,
                temporal_dense_prediction_loss=predicted_smooth,
                temporal_dense_weighted_loss=self.dense_smoothness_weight*dense_loss,
                temporal_dense_valid_fraction=temporal_valid.float().mean(), temporal_dense_event_weight=temporal_event_weight.mean())
        result = dict(pred_future_latent=future, latent_loss=future_loss, latent_cosine_loss=future_loss.new_zeros(()),
            auxiliary_loss=auxiliary, dino_future_loss=future_loss, dino_future_uniform_loss=uniform_loss,
            dino_current_loss=current_loss, temporal_smoothness_loss=smooth_loss, **metrics)
        for horizon in range(2):
            result[f"latent_loss_horizon_{horizon+1}"] = self.masked_mean(errors[:, horizon+1:horizon+2], valid[:, horizon+1:horizon+2])
        with torch.no_grad():
            copy = self.cosine_error(teacher[:, :1].expand_as(teacher[:, 1:]), teacher[:, 1:])
            copy_loss = self.masked_mean(copy, valid[:, 1:])
            weights = valid[:, 1:, :, None, None].float()
            mean_teacher = (teacher[:, 1:].float()*weights).sum(0, keepdim=True)/weights.sum(0, keepdim=True).clamp_min(1.)
            mean_loss = self.masked_mean(self.cosine_error(mean_teacher.expand_as(teacher[:, 1:]), teacher[:, 1:]), valid[:, 1:])
            observed_change = (latent[:, 1:].float()-latent[:, :-1].float()).square().mean(-1).reshape(batch,2,self.num_views,self.tokens_per_view)
            result.update(dino_copy_loss=copy_loss, dino_batch_mean_loss=mean_loss,
                dino_to_copy_ratio=uniform_loss.detach()/copy_loss.clamp_min(1e-6),
                predicted_latent_rms=future.float().square().mean().sqrt(),
                latent_batch_std=latent[:,0].float().var(0,unbiased=False).mean().sqrt(),
                temporal_observed_delta_rms=self.masked_mean(observed_change,valid[:,1:] & valid[:,:-1]).sqrt())
        return result


# Legacy adapter-latent compatibility

class VisualTokenLatentWorldModel(nn.Module):
    """Predict all future visual-token latents in one deterministic pass."""

    def __init__(
        self,
        *,
        latent_dim: int,
        goal_dim: Optional[int],
        n_future: int,
        num_tokens: int,
        dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 1024,
        stats_momentum: float = 0.9,
        detach_input: bool = True,
    ) -> None:
        super().__init__()
        self.n_future = int(n_future)
        self.num_tokens = int(num_tokens)
        self.stats_momentum = float(stats_momentum)
        self.detach_input = bool(detach_input)
        self._stats_eps = 1e-4
        self.residual_predictor = TokenResidualPredictor(
            latent_dim=latent_dim,
            goal_dim=goal_dim,
            n_future=self.n_future,
            num_tokens=self.num_tokens,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
        )
        self.register_buffer("delta_scale", torch.ones(1))
        self.register_buffer("_delta_scale_ready", torch.zeros(1))

    def _apply(self, *args, **kwargs):
        # Keep EMA statistics in fp32 under module-wide bf16 conversion.
        delta_scale_fp32 = self.delta_scale.detach().float().clone()
        delta_scale_ready_fp32 = self._delta_scale_ready.detach().float().clone()
        module = super()._apply(*args, **kwargs)
        target_device = module.delta_scale.device
        module.delta_scale = delta_scale_fp32.to(device=target_device)
        module._delta_scale_ready = delta_scale_ready_fp32.to(device=target_device)
        return module

    @torch.no_grad()
    def _update_delta_scale(
        self,
        residual: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if getattr(self, "sync_stats", False) and torch.distributed.is_initialized():
            weights = torch.ones_like(residual[..., :1]) if mask is None else mask
            weights = weights.to(device=residual.device, dtype=torch.float32)
            moments = torch.stack(((residual.float().square() * weights).sum(),
                                   weights.sum() * residual.shape[-1]))
            torch.distributed.all_reduce(moments)
            if moments[1].item() == 0:
                return
            rms = (moments[0] / moments[1].clamp_min(1)).clamp_min(self._stats_eps).sqrt()
        elif mask is not None:
            weights = mask.to(device=residual.device, dtype=torch.float32)
            denominator = weights.sum() * residual.shape[-1]
            if denominator.item() == 0:
                return
            denominator = denominator.clamp_min(1.0)
            rms = ((residual.float().square() * weights).sum() / denominator).clamp_min(
                self._stats_eps
            ).sqrt()
        else:
            rms = residual.float().square().mean().clamp_min(self._stats_eps).sqrt()
        if float(self._delta_scale_ready) < 1.0:
            self.delta_scale.fill_(float(rms))
            self._delta_scale_ready.fill_(1.0)
        else:
            self.delta_scale.mul_(self.stats_momentum).add_(
                rms, alpha=1 - self.stats_momentum
            )

    def regress_future(
        self,
        context: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context.shape[1] != 1:
            raise ValueError(f"GAWM expects one context frame, got {context.shape[1]}")
        predicted_delta = self.residual_predictor(context, goal=goal)
        return context + predicted_delta * self.delta_scale.clamp_min(self._stats_eps)

    def forward(
        self,
        latent: torch.Tensor,
        *,
        ctx_len: int,
        goal: Optional[torch.Tensor] = None,
        update_stats: bool = True,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, total_frames, num_tokens = latent.shape[:3]
        if num_tokens != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} visual tokens, got {num_tokens}")
        if ctx_len != 1:
            raise ValueError(f"GAWM expects ctx_len=1, got {ctx_len}")
        if total_frames != 1 + self.n_future:
            raise ValueError(
                f"expected {1 + self.n_future} temporal frames, got {total_frames}"
            )
        if loss_mask is not None:
            if tuple(loss_mask.shape) != (batch_size, total_frames, num_tokens):
                raise ValueError(
                    "expected loss_mask shape "
                    f"{(batch_size, total_frames, num_tokens)}, got {tuple(loss_mask.shape)}"
                )
            if loss_mask.dtype not in {torch.bool, torch.float32}:
                raise ValueError(f"loss_mask must be bool or float32, got {loss_mask.dtype}")

        context = latent[:, :1]
        anchor = context
        if self.detach_input:
            context = context.detach()
            anchor = anchor.detach()
        future = latent[:, 1:]
        residual = (future - anchor).detach()
        residual_mask = loss_mask[:, 1:].unsqueeze(-1) if loss_mask is not None else None
        if self.training and update_stats:
            self._update_delta_scale(residual, mask=residual_mask)

        scale = self.delta_scale.clamp_min(self._stats_eps)
        predicted_residual = self.residual_predictor(context, goal=goal) * scale
        predicted_future = anchor + predicted_residual
        squared_error = (predicted_future.float() - future.detach().float()).square()
        if residual_mask is not None:
            weights = residual_mask.to(dtype=squared_error.dtype)
            latent_loss = (squared_error * weights).sum() / (
                weights.sum() * squared_error.shape[-1]
            ).clamp_min(1.0)
            pred_cosine_input = predicted_residual.float() * weights
            true_cosine_input = residual.float() * weights
        else:
            latent_loss = squared_error.mean()
            pred_cosine_input = predicted_residual.float()
            true_cosine_input = residual.float()
        direction_cosine = F.cosine_similarity(
            pred_cosine_input.flatten(2),
            true_cosine_input.flatten(2),
            dim=-1,
            eps=1e-8,
        )
        if residual_mask is None:
            direction_cosine = direction_cosine.mean()
            cosine_loss = 1.0 - direction_cosine
        else:
            # Completely masked horizons must not contribute a cosine penalty.
            valid_horizons = residual_mask.flatten(2).any(dim=-1).to(direction_cosine.dtype)
            count = valid_horizons.sum().clamp_min(1.0)
            cosine_loss = ((1.0 - direction_cosine) * valid_horizons).sum() / count
            direction_cosine = (direction_cosine * valid_horizons).sum() / count
        output = {
            "latent_loss": latent_loss,
            "latent_cosine_loss": cosine_loss,
            "pred_future_latent": predicted_future,
        }
        if residual_mask is not None:
            weights = residual_mask.to(dtype=squared_error.dtype)
            numerator = (squared_error * weights).sum(dim=(0, 2, 3))
            denominator = (
                weights.sum(dim=(0, 2, 3)) * squared_error.shape[-1]
            ).clamp_min(1.0)
            per_horizon_loss = numerator / denominator
        else:
            per_horizon_loss = squared_error.mean(dim=(0, 2, 3))
        for horizon_index, horizon_loss in enumerate(per_horizon_loss, start=1):
            output[f"latent_loss_horizon_{horizon_index}"] = horizon_loss

        with torch.no_grad():
            if residual_mask is not None:
                weights = residual_mask.to(dtype=torch.float32)
                denominator = (weights.sum() * residual.shape[-1]).clamp_min(1.0)

                def masked_mse(values: torch.Tensor) -> torch.Tensor:
                    return (values.float().square() * weights).sum() / denominator

                copy_mse = masked_mse(residual)
                pred_mse = masked_mse(predicted_residual - residual)
                mean_residual = (residual.float() * weights).sum(
                    dim=0, keepdim=True
                ) / weights.sum(dim=0, keepdim=True).clamp_min(1.0)
                mean_baseline_mse = masked_mse(residual - mean_residual)
                pred_rms = masked_mse(predicted_residual).sqrt()
            else:
                copy_mse = residual.float().square().mean()
                pred_mse = (predicted_residual.float() - residual.float()).square().mean()
                mean_residual = residual.float().mean(dim=0, keepdim=True)
                mean_baseline_mse = (residual.float() - mean_residual).square().mean()
                pred_rms = predicted_residual.float().square().mean().sqrt()
            output.update(
                {
                    "delta_scale": scale.detach().mean(),
                    "delta_target_rms": copy_mse.sqrt(),
                    "delta_pred_rms": pred_rms,
                    "delta_copy_mse": copy_mse,
                    "delta_pred_mse": pred_mse,
                    "delta_mean_baseline_mse": mean_baseline_mse,
                    "delta_to_copy_ratio": pred_mse / copy_mse.clamp_min(1e-8),
                    "delta_direction_cosine": direction_cosine,
                }
            )
        return output
