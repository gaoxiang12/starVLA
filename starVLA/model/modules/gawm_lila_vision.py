"""LiLa-WAM visual front-end and training-demonstration VTTs for GAWM.

The upstream fusion/adapter primitives are reused without an upstream runtime
checkout. GAWM's 384-wide world model is connected by a separate projection;
learned queries are not interpreted as spatial grid cells.
"""
import hashlib
import json
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from starVLA.model.modules.lila.primitives import MultiLayerConcatFusion, VisualFeatureAdapter


def rgb_pixels(images, size):
    """Official OpenCV linear resize + ImageNet normalization, no channel swap."""
    arrays = []
    for image in images:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            if image.ndim == 3 and image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("LiLa visual input must be uint8 RGB or explicitly normalized tensors")
        if (image.shape[1], image.shape[0]) != tuple(size):
            image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_LINEAR)
        arrays.append(image)
    values = np.stack(arrays).astype(np.float32) / 255.
    values = (values - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return torch.from_numpy(values.transpose(0, 3, 1, 2).copy())


def canonical_vtt_key(tag, language):
    text = " ".join(unicodedata.normalize("NFKC", language).strip().lower().split())
    if not tag or not text:
        raise ValueError("VTT requires an embodiment and a nonempty task")
    return f"{tag}:{text}"


class LiLaVisualPooler(nn.Module):
    """Shared official per-view adapter, followed by a GAWM interface bridge."""
    def __init__(self, patch_dim, token_dim, num_views, tokens_per_view=64,
                 num_layers=3, hidden_dim=768, depth=4, heads=8, bridge_norm="none"):
        super().__init__()
        self.num_views, self.tokens_per_view = num_views, tokens_per_view
        self.num_tokens = num_views * tokens_per_view
        self.fusion = MultiLayerConcatFusion(patch_dim, num_layers, patch_dim, 'linear', True)
        self.adapter = VisualFeatureAdapter(patch_dim, hidden_dim, tokens_per_view, heads, depth, dropout=0.)
        self.bridge = nn.Linear(hidden_dim, token_dim)
        if bridge_norm not in ("none", "fixed_layernorm"):
            raise ValueError(f"Unknown LiLa bridge normalization: {bridge_norm}")
        self.bridge_norm = bridge_norm
        if bridge_norm == "fixed_layernorm":
            # Distinguish the new interface in strict checkpoint loading even
            # though the normalization itself has no trainable parameters.
            self.register_buffer("bridge_norm_version", torch.tensor(1, dtype=torch.uint8))
        self.view_embedding = nn.Embedding(num_views, token_dim)
        nn.init.normal_(self.view_embedding.weight, std=.02)

    def position_tokens(self):
        # Query identity is already learned inside the adapter. Only camera
        # identity is added here; there is no invented 8x8 query geometry.
        return self.view_embedding.weight.repeat_interleave(self.tokens_per_view, 0)

    def official_tokens(self, features):
        if features.ndim != 6:
            raise ValueError("Expected features [B,T,V,L,N,D]")
        b, t, v, layers, n, d = features.shape
        if v != self.num_views or layers != self.fusion.num_layers:
            raise ValueError("LiLa camera/layer count mismatch")
        per_layer = [features[:, :, :, i].reshape(b*t*v, n, d) for i in range(layers)]
        return self.adapter(self.fusion(per_layer)).reshape(b, t, v, self.tokens_per_view, -1)

    def _mask(self, views, dtype):
        if views.ndim != 2 or views.shape[1] != self.num_views:
            raise ValueError("Invalid camera mask")
        # Both requested benchmarks have every physical camera. Reject missing
        # views instead of allowing zero keys into the unchanged world model.
        if not bool(views.all()):
            raise ValueError("LiLa-aligned GAWM requires every configured physical camera")
        return views.to(dtype).repeat_interleave(self.tokens_per_view, 1)[:, None, :, None]

    def remove_position(self, tokens, view_valid_mask=None):
        content = tokens - self.position_tokens().to(tokens.dtype)[None, None]
        if view_valid_mask is not None:
            content = content * self._mask(view_valid_mask, tokens.dtype)
        return content

    def forward(self, features, return_content=False, view_valid_mask=None):
        mask = self._mask(view_valid_mask, features.dtype) if view_valid_mask is not None else None
        adapted = self.official_tokens(features)
        content = self.bridge(adapted).flatten(2, 3)
        if self.bridge_norm == "fixed_layernorm":
            # Preserve official adapter internals, but bound the GAWM content
            # scale. FP32 statistics and no affine gain prevent scale drift.
            content = F.layer_norm(content.float(), (content.shape[-1],)).to(content.dtype)
        tokens = content + self.position_tokens().to(content.dtype)[None, None]
        if mask is not None:
            tokens, content = tokens * mask, content * mask
        return (tokens, content) if return_content else tokens


class VTTConditioner(nn.Module):
    """Frozen task vectors saved in checkpoints; unknown task IDs fail closed."""
    def __init__(self, cfg, feature_dim, output_dim, hidden_dim=768):
        super().__init__()
        payload = cfg.get('task_vectors')
        path = cfg.get('task_vectors_path')
        if payload is None and path and Path(path).is_file():
            payload = json.loads(Path(path).read_text())
        self.ready = payload is not None
        if self.ready:
            if payload.get('format_version') != 1 or payload.get('split') != 'train':
                raise ValueError("VTT asset must be format_version=1, split=train")
            self.names = sorted(payload['vectors'])
            array = np.asarray([payload['vectors'][key] for key in self.names], np.float32)
            if not self.names or array.shape != (len(self.names), feature_dim) or not np.isfinite(array).all():
                raise ValueError("Invalid VTT dimensions or nonfinite vectors")
            if cfg.get('task_names') is not None and list(cfg.task_names) != self.names:
                raise ValueError("VTT vocabulary differs from saved configuration")
            cfg.task_names = self.names
        else:
            self.names = list(cfg.get('task_names', []))
            if not self.names:
                raise FileNotFoundError(f"Prepare training VTTs first: {path}")
            array = np.zeros((len(self.names), feature_dim), np.float32)
        self.indices = {name: i for i, name in enumerate(self.names)}
        if len(self.indices) != len(self.names):
            raise ValueError("Duplicate VTT task keys")
        key_digest = hashlib.sha256(json.dumps(self.names, ensure_ascii=False).encode()).digest()
        self.register_buffer('key_fingerprint', torch.tensor(list(key_digest), dtype=torch.uint8))
        self.register_buffer('vectors', torch.from_numpy(array))
        self.projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
                                        nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.bridge = nn.Linear(hidden_dim, output_dim)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + 'key_fingerprint')
        vectors = state_dict.get(prefix + 'vectors')
        if saved is None or not torch.equal(saved.cpu(), self.key_fingerprint.cpu()):
            error_msgs.append('Checkpoint VTT vocabulary mismatch or absent fingerprint')
        elif vectors is None or vectors.shape != self.vectors.shape or not torch.isfinite(vectors).all():
            error_msgs.append('Checkpoint VTT vectors are missing or invalid')
        elif self.ready and not torch.equal(vectors.float().cpu(), self.vectors.float().cpu()):
            error_msgs.append('Checkpoint VTT vectors differ from configured training asset')
        else:
            self.ready = True
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, instructions, *, robot_tag, device):
        if not self.ready:
            raise RuntimeError('Load checkpoint VTT buffers before using a deployment-only configuration')
        keys = [canonical_vtt_key(robot_tag, text) for text in instructions]
        missing = set(keys) - self.indices.keys()
        if missing:
            raise KeyError(f'No training VTT for tasks: {sorted(missing)}')
        indices = torch.tensor([self.indices[key] for key in keys], device=self.vectors.device)
        vectors = self.vectors[indices].to(device=device, dtype=self.projection[0].weight.dtype)
        return self.bridge(self.projection(vectors))
