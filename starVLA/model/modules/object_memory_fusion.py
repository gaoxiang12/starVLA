"""Fuse predicted object tokens inside every ACT decoder layer.

This candidate module is independent of the currently pinned GAWM experiment.
Its object-only path shares the ACT decoder and action projection with the full
path, allowing an auxiliary imitation objective to train object-to-action use.
"""
import torch
from torch import nn


class ObjectMemoryFusion(nn.Module):
    def __init__(self, object_dim, hidden_dim):
        super().__init__()
        self.project = nn.Linear(object_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.kind = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, action_head, visual_tokens, state, object_memory, *, objects_only=False):
        """Return action queries; the existing action head projects them to actions.

        `object_memory` contains only network-predicted tokens and view masks.
        `objects_only` omits visual/world-model memory but retains proprioception.
        It is an auxiliary training path, not an inference fallback or oracle.
        """
        tokens, valid = object_memory
        if tokens.ndim != 3 or valid.shape != tokens.shape[:2] or valid.dtype != torch.bool:
            raise ValueError('Object tokens require [B,K,D] and bool [B,K] validity')
        if tokens.shape[1] == 0 or state is None or state.ndim != 2 or state.shape[0] != tokens.shape[0]:
            raise ValueError('Nonempty object slots and one proprioceptive state per sample required')
        if action_head.state_projection is None:
            raise ValueError('Object fusion requires the existing ACT state projection')
        # Key padding cannot sanitize NaNs before a projection. Remove missing
        # view tokens before normalization, and mask them in every decoder layer.
        safe_tokens = torch.where(valid[..., None], tokens, 0.)
        objects = self.norm(self.project(safe_tokens)) + self.kind
        if objects_only:
            memory = action_head.state_projection(state.to(objects))
        else:
            memory = action_head.build_memory(visual_tokens, state=state)
        objects = objects.to(memory)
        valid = valid.to(device=memory.device)
        padding = torch.cat([torch.zeros(memory.shape[:2], device=memory.device, dtype=torch.bool),
                             ~valid], dim=1)
        memory = torch.cat([memory, objects], dim=1)
        queries = action_head.action_queries.weight.to(memory).unsqueeze(0).expand(memory.shape[0], -1, -1)
        return action_head.decoder(tgt=queries, memory=memory, memory_key_padding_mask=padding)
