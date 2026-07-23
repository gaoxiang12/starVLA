"""Training-only WALA-style transition supervision for visual action queries.

The teacher sees the current visual tokens and their ground-truth future
residuals.  It compresses that observed transition into a small token set and
learns to reconstruct the residuals.  A student resampler then predicts the
same transition tokens from the deployed policy's action-query hidden states.

None of these modules are used by policy inference.  This is intentional: the
validated LeWM-OFT control path remains unchanged while its action queries gain
future-dynamics supervision during fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _decoder_stack(hidden_dim: int, num_heads: int, depth: int) -> nn.TransformerDecoder:
    if hidden_dim % num_heads:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
        )
    layer = nn.TransformerDecoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=4 * hidden_dim,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=depth)


def token_cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean token-wise cosine distance in fp32."""

    prediction = F.normalize(prediction.float(), dim=-1, eps=1e-6)
    target = F.normalize(target.float(), dim=-1, eps=1e-6)
    return 1.0 - (prediction * target).sum(dim=-1).mean()


class FutureTransitionTokenizer(nn.Module):
    """Encode a ground-truth compact-DINO transition into latent-action tokens."""

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        num_visual_tokens: int,
        num_future: int,
        num_transition_tokens: int,
        depth: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_visual_tokens = int(num_visual_tokens)
        self.num_future = int(num_future)
        self.num_transition_tokens = int(num_transition_tokens)

        self.current_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.delta_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.visual_embedding = nn.Embedding(self.num_visual_tokens, self.hidden_dim)
        self.future_embedding = nn.Embedding(self.num_future, self.hidden_dim)
        self.current_type_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.delta_type_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.transition_queries = nn.Parameter(
            torch.randn(1, self.num_transition_tokens, self.hidden_dim) * 0.02
        )
        self.transition_embedding = nn.Parameter(
            torch.randn(1, self.num_transition_tokens, self.hidden_dim) * 0.02
        )
        self.encoder = _decoder_stack(self.hidden_dim, int(num_heads), int(depth))

    def forward(
        self,
        current: torch.Tensor,
        future_delta: torch.Tensor,
    ) -> torch.Tensor:
        expected_current = (self.num_visual_tokens, self.latent_dim)
        expected_future = (self.num_future, *expected_current)
        if current.ndim != 3 or tuple(current.shape[1:]) != expected_current:
            raise ValueError(
                f"expected current shape (B,{expected_current[0]},{expected_current[1]}), "
                f"got {tuple(current.shape)}"
            )
        if future_delta.ndim != 4 or tuple(future_delta.shape[1:]) != expected_future:
            raise ValueError(
                f"expected future_delta tail {expected_future}, got {tuple(future_delta.shape)}"
            )

        batch = current.shape[0]
        visual_ids = torch.arange(self.num_visual_tokens, device=current.device)
        future_ids = torch.arange(self.num_future, device=current.device)
        visual_position = self.visual_embedding(visual_ids).to(current.dtype)

        current_memory = self.current_projection(current)
        current_memory = (
            current_memory
            + visual_position[None]
            + self.current_type_embedding.to(current.dtype)
        )

        delta_memory = self.delta_projection(future_delta)
        delta_memory = delta_memory + visual_position[None, None]
        delta_memory = delta_memory + self.future_embedding(future_ids).to(
            current.dtype
        )[None, :, None]
        delta_memory = delta_memory + self.delta_type_embedding.to(current.dtype)
        delta_memory = delta_memory.flatten(1, 2)

        memory = torch.cat((current_memory, delta_memory), dim=1)
        queries = self.transition_queries.to(current.dtype).expand(batch, -1, -1)
        queries = queries + self.transition_embedding.to(current.dtype)
        return self.encoder(tgt=queries, memory=memory)


class FutureTransitionDecoder(nn.Module):
    """Decode normalized future visual-token residuals from transition tokens."""

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        num_visual_tokens: int,
        num_future: int,
        depth: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_visual_tokens = int(num_visual_tokens)
        self.num_future = int(num_future)

        self.current_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.visual_embedding = nn.Embedding(self.num_visual_tokens, self.hidden_dim)
        self.future_embedding = nn.Embedding(self.num_future, self.hidden_dim)
        self.query_embedding = nn.Parameter(
            torch.randn(1, self.num_future, self.num_visual_tokens, self.hidden_dim)
            * 0.02
        )
        self.decoder = _decoder_stack(self.hidden_dim, int(num_heads), int(depth))
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(
        self,
        current: torch.Tensor,
        transition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        expected_current = (self.num_visual_tokens, self.latent_dim)
        if current.ndim != 3 or tuple(current.shape[1:]) != expected_current:
            raise ValueError(
                f"expected current shape (B,{expected_current[0]},{expected_current[1]}), "
                f"got {tuple(current.shape)}"
            )
        if transition_tokens.ndim != 3 or transition_tokens.shape[0] != current.shape[0]:
            raise ValueError(
                "transition_tokens must have shape (B,Q,H), got "
                f"{tuple(transition_tokens.shape)}"
            )

        batch = current.shape[0]
        visual_ids = torch.arange(self.num_visual_tokens, device=current.device)
        future_ids = torch.arange(self.num_future, device=current.device)
        queries = self.current_projection(current)[:, None].expand(
            -1, self.num_future, -1, -1
        )
        queries = queries + self.visual_embedding(visual_ids).to(current.dtype)[
            None, None
        ]
        queries = queries + self.future_embedding(future_ids).to(current.dtype)[
            None, :, None
        ]
        queries = queries + self.query_embedding.to(current.dtype)
        hidden = self.decoder(
            tgt=queries.reshape(batch, self.num_future * self.num_visual_tokens, -1),
            memory=transition_tokens,
        )
        hidden = hidden.view(
            batch, self.num_future, self.num_visual_tokens, self.hidden_dim
        )
        return self.output(self.output_norm(hidden))


class ActionQueryTransitionResampler(nn.Module):
    """Predict transition tokens from LeWM-OFT's deployed action queries."""

    def __init__(
        self,
        *,
        action_hidden_dim: int,
        hidden_dim: int,
        num_action_queries: int,
        num_transition_tokens: int,
        depth: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.num_action_queries = int(num_action_queries)
        self.num_transition_tokens = int(num_transition_tokens)
        self.hidden_dim = int(hidden_dim)

        self.action_projection = nn.Linear(int(action_hidden_dim), self.hidden_dim)
        self.action_embedding = nn.Embedding(self.num_action_queries, self.hidden_dim)
        self.transition_queries = nn.Parameter(
            torch.randn(1, self.num_transition_tokens, self.hidden_dim) * 0.02
        )
        self.transition_embedding = nn.Parameter(
            torch.randn(1, self.num_transition_tokens, self.hidden_dim) * 0.02
        )
        self.resampler = _decoder_stack(self.hidden_dim, int(num_heads), int(depth))

    def forward(self, action_queries: torch.Tensor) -> torch.Tensor:
        if action_queries.ndim != 3 or action_queries.shape[1] != self.num_action_queries:
            raise ValueError(
                f"expected action_queries shape (B,{self.num_action_queries},H), "
                f"got {tuple(action_queries.shape)}"
            )
        batch = action_queries.shape[0]
        action_ids = torch.arange(self.num_action_queries, device=action_queries.device)
        memory = self.action_projection(action_queries)
        memory = memory + self.action_embedding(action_ids).to(memory.dtype)[None]
        queries = self.transition_queries.to(memory.dtype).expand(batch, -1, -1)
        queries = queries + self.transition_embedding.to(memory.dtype)
        return self.resampler(tgt=queries, memory=memory)


class WALAVisualTransitionAuxiliary(nn.Module):
    """Teacher autoencoder plus action-query student used by LeWM-OFT training."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_hidden_dim: int,
        hidden_dim: int,
        num_visual_tokens: int,
        num_future: int,
        num_action_queries: int,
        num_transition_tokens: int,
        encoder_depth: int,
        decoder_depth: int,
        resampler_depth: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        common = {
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "num_visual_tokens": num_visual_tokens,
            "num_future": num_future,
            "num_heads": num_heads,
        }
        self.teacher_encoder = FutureTransitionTokenizer(
            **common,
            num_transition_tokens=num_transition_tokens,
            depth=encoder_depth,
        )
        self.teacher_decoder = FutureTransitionDecoder(
            **common,
            depth=decoder_depth,
        )
        self.student_resampler = ActionQueryTransitionResampler(
            action_hidden_dim=action_hidden_dim,
            hidden_dim=hidden_dim,
            num_action_queries=num_action_queries,
            num_transition_tokens=num_transition_tokens,
            depth=resampler_depth,
            num_heads=num_heads,
        )

    def teacher_forward(
        self,
        current: torch.Tensor,
        future_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.teacher_encoder(current, future_delta)
        reconstruction = self.teacher_decoder(current, tokens)
        return tokens, reconstruction

    def student_forward(
        self,
        action_queries: torch.Tensor,
        current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.student_resampler(action_queries)
        reconstruction = self.teacher_decoder(current, tokens)
        return tokens, reconstruction
