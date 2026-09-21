"""LiLa-WAM transformer primitives, vendored from teee000/LiLa-WAM.

Source: models/vla_model_fm.py (see UPSTREAM.txt). Changes: optional memory
padding masks, per-camera future-query offsets, and a safe bf16 attention path.
No upstream runtime import.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeMultiheadAttention(nn.MultiheadAttention):
    """Avoid native inference MHA on ZeRO-flattened bf16 parameter views.

    That fused path can require alignment not guaranteed for robot-specific
    odd-width I/O heads. The ordinary differentiable path uses the same
    parameters and SDPA without changing the process-global backend setting.
    """
    def forward(self, query, key, value, **kwargs):
        if query is key:
            key = key.view_as(key)
        return super().forward(query, key, value, **kwargs)


def safe_decoder_attention(layer, hidden_dim, num_heads, dropout):
    layer.self_attn = SafeMultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
    layer.multihead_attn = SafeMultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
    return layer


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Standard Transformer SinCos positional encoding.
    Returns: (1, length, embed_dim)
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embed dim must be divisible by 2")

    pos = torch.arange(length, dtype=torch.float32)
    grid = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (grid / (embed_dim // 2)))

    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.unsqueeze(0)


# --- Time Embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = emb.to(dtype=x.dtype)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class VisualFeatureAdapter(nn.Module):
    """Perceiver-style adapter: learnable queries cross-attend to vision tokens, output (B, num_queries, hidden_dim)."""
    def __init__(self, feat_dim, hidden_dim, num_queries=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries

        self.input_norm = nn.LayerNorm(feat_dim)
        self.feature_proj = nn.Linear(feat_dim, hidden_dim)

        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            safe_decoder_attention(decoder_layer, hidden_dim, num_heads, dropout), num_layers=num_layers)

    def forward(self, features):
        """
        features: (B, N, feat_dim)
        Returns: (B, num_queries, hidden_dim)
        """
        B = features.shape[0]
        memory = self.input_norm(features)
        memory = self.feature_proj(memory)
        tgt = self.query_embed.expand(B, -1, -1)
        out = self.transformer_decoder(tgt, memory)
        return out


class MultiLayerConcatFusion(nn.Module):
    """
    Multi-layer DINO feature fusion (concat mode):
      - Each layer is normalized by its own LayerNorm first (feature norms differ
        a lot across DINOv3 layers; shallow layers would be drowned out otherwise)
      - Concatenate token-wise along the feature dim -> (B, N, L*feat_dim)
      - Linear / MLP projection down to -> (B, N, out_dim)
    The output is then fed into a single VisualFeatureAdapter.
    """
    def __init__(self, feat_dim, num_layers, out_dim, proj_type="linear", pre_norm=True):
        super().__init__()
        self.num_layers = num_layers
        self.pre_norm = pre_norm
        if pre_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(num_layers)])

        in_dim = feat_dim * num_layers
        if proj_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
        elif proj_type == "linear":
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            raise ValueError(f"Unsupported concat proj_type: {proj_type}")

    def forward(self, feats_list):
        """feats_list: List[(B, N, feat_dim)] (all layers must share the same N)"""
        assert len(feats_list) == self.num_layers, \
            f"MultiLayerConcatFusion expects {self.num_layers} layers, got {len(feats_list)}"
        if self.pre_norm:
            feats_list = [ln(f) for ln, f in zip(self.layer_norms, feats_list)]
        x = torch.cat(feats_list, dim=-1)   # (B, N, L*feat_dim)
        return self.proj(x)                 # (B, N, out_dim)


class FutureFeatureDecoder(nn.Module):
    """
    Future-frame feature prediction decoder (Transformer):
      - Learnable query tokens as Q
      - cond tokens produced by the action head as KV (cross-attention)
      - Output projected to the target feature dim (= DINO hidden_size), supervised
        against future-frame patch features with a cosine loss

    Number of queries = number of future-frame patch tokens (dense prediction,
    one query per patch position).
    """
    def __init__(self, num_queries, hidden_dim, out_dim,
                 num_heads=4, num_layers=2, dropout=0.0):
        super().__init__()
        self.num_queries = num_queries
        self.out_dim = out_dim

        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)

        # Normalize cond tokens before they enter cross-attention
        self.kv_norm = nn.LayerNorm(hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            safe_decoder_attention(decoder_layer, hidden_dim, num_heads, dropout), num_layers=num_layers)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, cond_tokens, memory_mask=None, query_offset=None):
        """
        cond_tokens: (B, M, hidden_dim) — observation tokens evolved by the DiT
        Returns: (B, num_queries, out_dim) predicted future-frame patch features
        """
        B = cond_tokens.shape[0]
        memory = self.kv_norm(cond_tokens)
        tgt = self.query_embed.expand(B, -1, -1)
        if query_offset is not None:
            tgt = tgt + query_offset
        out = self.decoder(tgt, memory, memory_key_padding_mask=memory_mask)
        out = self.out_proj(self.out_norm(out))
        return out


def future_feature_cosine_loss(pred, target):
    """
    Per-token cosine loss = mean(1 - cos_sim).
    pred / target: (B, M, D). Computed in float32 for numerical stability
    under bf16 training.
    """
    assert pred.shape == target.shape, \
        f"future-feat pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)} " \
        f"(check patch count: num_queries should equal the future-frame DINO patch token count)"
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    cos = (pred * target).sum(dim=-1)        # (B, M)
    return (1.0 - cos).mean()


class DiTBlock(nn.Module):
    """
    DiT block with Self-Attention + MLP, conditioned via adaLN-zero on time embedding.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = SafeMultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

        # 6 params: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, emb_t, attn_mask=None, key_padding_mask=None):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = \
            self.adaLN_modulation(emb_t).chunk(6, dim=1)

        # Self-Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn1(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]

        # MLP
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x
