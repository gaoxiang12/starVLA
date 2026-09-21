"""Differentiable predicted-waypoint tokens read directly by action queries."""
import torch
from torch import nn


class SpatialGoalReadout(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.coordinates = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(),
                                         nn.Linear(hidden_dim, hidden_dim))
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        # Preserve a loaded policy exactly before the added readout learns.
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def pack(self, memory, logits, xy, view_mask):
        if memory.shape[:3] != logits.shape or xy.shape != (*logits.shape[:2], 2):
            raise ValueError('Inconsistent waypoint feature/logit/coordinate shapes')
        if view_mask.shape != logits.shape[:2]:
            raise ValueError('Invalid waypoint view mask')
        valid = view_mask.bool()
        # Missing views may carry padding NaNs. Remove them before softmax or
        # normalization; key padding alone cannot sanitize attention values.
        features = torch.where(valid[..., None, None], memory, 0.)
        scores = torch.where(valid[..., None], logits, 0.)
        coordinates = torch.where(valid[..., None], xy, 0.)
        pooled = (scores.softmax(-1)[..., None] * features).sum(-2)
        tokens = self.feature_norm(pooled) + self.coordinates(coordinates)
        return torch.where(valid[..., None], tokens, 0.), valid

    def forward(self, queries, tokens, valid):
        available = valid.any(-1)
        safe_valid = valid.clone()
        safe_valid[~available, 0] = True
        tokens = torch.where(valid[..., None], tokens, 0.)
        residual = self.attention(self.query_norm(queries), tokens, tokens,
                                  key_padding_mask=~safe_valid, need_weights=False)[0]
        return queries + residual * available[:, None, None]
