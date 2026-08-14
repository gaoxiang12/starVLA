"""TurboVLA-style ACT decoder for continuous action chunks.

The decoder keeps one learned query per action timestep. The queries attend to
visual memory and optional proprioceptive state tokens through a stack of
pre-norm Transformer decoder layers, then a small MLP predicts normalized
continuous actions. The architecture follows the Apache-2.0 TurboVLA reference
under ``thirdparty/TurboVLA/turbovla/models/action_head.py`` while remaining a
native StarVLA module with no third-party runtime dependency.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class ActionProjectionMLP(nn.Module):
    """Three-layer ReLU MLP matching TurboVLA's action projection."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            ]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            hidden = F.relu(layer(hidden))
        return self.layers[-1](hidden)


class StateTokenProjection(nn.Module):
    """Project current proprioception into ACT memory tokens."""

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_dim: int,
        state_hidden_dim: int,
        num_tokens: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.net = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, int(state_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(state_hidden_dim), self.num_tokens * self.hidden_dim),
        )
        self.position = nn.Parameter(
            torch.randn(1, self.num_tokens, self.hidden_dim) * 0.02
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(
                "ACT state must have shape [B,D] or [B,T,D] with "
                f"D={self.state_dim}, got {tuple(state.shape)}"
            )
        tokens = self.net(state).view(state.shape[0], self.num_tokens, self.hidden_dim)
        position = self.position.to(device=tokens.device, dtype=tokens.dtype)
        return self.output_norm(tokens + position)


class TurboStyleACTActionHead(nn.Module):
    """Decode an action chunk from LeWM visual and state memory.

    This follows TurboVLA's action-head form while retaining LeWM's explicit
    frame and spatial-token embeddings. LeWM needs those embeddings because its
    memory contains current and world-model-predicted future token grids.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int,
        action_dim: int,
        horizon: int,
        num_frames: int,
        num_visual_tokens: int,
        num_heads: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        mlp_hidden_dim: int = 512,
        dropout: float = 0.1,
        state_dim: int = 0,
        state_hidden_dim: int = 256,
        num_state_tokens: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.num_frames = int(num_frames)
        self.num_visual_tokens = int(num_visual_tokens)
        self.num_state_tokens = int(num_state_tokens) if state_dim > 0 else 0
        self.state_dim = int(state_dim)

        if min(
            self.hidden_dim,
            self.action_dim,
            self.horizon,
            self.num_frames,
            self.num_visual_tokens,
        ) < 1:
            raise ValueError("ACT dimensions, horizon, frames, and visual tokens must be positive")
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError(
                f"ACT hidden_dim={self.hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if self.state_dim > 0 and self.num_state_tokens < 1:
            raise ValueError("state-conditioned ACT requires at least one state token")

        self.visual_projection = nn.Linear(int(token_dim), self.hidden_dim)
        self.frame_embedding = nn.Embedding(self.num_frames, self.hidden_dim)
        self.token_embedding = nn.Embedding(self.num_visual_tokens, self.hidden_dim)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)

        self.state_projection: Optional[StateTokenProjection]
        if self.state_dim > 0:
            self.state_projection = StateTokenProjection(
                state_dim=self.state_dim,
                hidden_dim=self.hidden_dim,
                state_hidden_dim=int(state_hidden_dim),
                num_tokens=self.num_state_tokens,
                dropout=float(dropout),
            )
        else:
            self.state_projection = None

        self.action_queries = nn.Embedding(self.horizon, self.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(num_layers),
        )
        self.action_projection = ActionProjectionMLP(
            self.hidden_dim,
            int(mlp_hidden_dim),
            self.action_dim,
        )

    def build_memory(
        self,
        visual_tokens: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if visual_tokens.ndim != 4:
            raise ValueError(
                "ACT visual tokens must have shape [B,T,K,C], got "
                f"{tuple(visual_tokens.shape)}"
            )
        batch_size, num_frames, num_tokens, _ = visual_tokens.shape
        if num_frames > self.num_frames or num_tokens != self.num_visual_tokens:
            raise ValueError(
                "ACT visual memory expected "
                f"T<={self.num_frames}, K={self.num_visual_tokens}; "
                f"got T={num_frames}, K={num_tokens}"
            )

        memory = self.visual_projection(visual_tokens)
        frame_ids = torch.arange(num_frames, device=memory.device)
        token_ids = torch.arange(num_tokens, device=memory.device)
        memory = memory + self.frame_embedding(frame_ids).view(
            1, num_frames, 1, self.hidden_dim
        )
        memory = memory + self.token_embedding(token_ids).view(
            1, 1, num_tokens, self.hidden_dim
        )
        memory = memory.reshape(batch_size, num_frames * num_tokens, self.hidden_dim)
        memory = self.memory_norm(memory)

        if self.state_projection is not None:
            if state is None:
                raise ValueError("state-conditioned ACT requires current state")
            state = state.to(device=memory.device, dtype=memory.dtype)
            memory = torch.cat([memory, self.state_projection(state)], dim=1)
        return memory

    def decode_action_queries(
        self,
        visual_tokens: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.build_memory(visual_tokens, state=state)
        queries = self.action_queries.weight.to(dtype=memory.dtype)
        queries = queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        return self.decoder(tgt=queries, memory=memory)

    def predict_action(self, action_hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.action_projection(action_hidden))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.decode_action_queries(visual_tokens, state=state)
        return self.predict_action(hidden)
