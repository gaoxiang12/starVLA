"""Small OFT-inspired prefix transformer and direct residual-MLP action readout.

Current spatial patches and task tokens are updated by the same transformer
layers as action placeholders. Input tokens form a bidirectional prefix;
action placeholders use causal attention and are predicted in one forward pass.
No target actions, proprioception, or predicted future features enter this head.
"""
import torch
from torch import nn

from starVLA.model.modules.action_model.MLP_ActionHeader import L1RegressionActionHead


class OFTSpatialActionHead(nn.Module):
    def __init__(self, *, patch_dim=768, task_dim=384, hidden_dim=384, num_views=3,
                 grid_size=14, horizon=16, action_dim=14, depth=4, heads=6):
        super().__init__()
        if min(patch_dim, task_dim, hidden_dim, num_views, grid_size, horizon,
               action_dim, depth, heads) < 1 or hidden_dim % heads:
            raise ValueError('Invalid OFT spatial head dimensions')
        self.patch_dim, self.task_dim, self.hidden_dim = patch_dim, task_dim, hidden_dim
        self.num_views, self.grid_size = num_views, grid_size
        self.horizon, self.action_dim = horizon, action_dim
        self.patch_projection = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, hidden_dim))
        self.task_projection = nn.Linear(task_dim, hidden_dim)
        self.view_embedding = nn.Embedding(num_views, hidden_dim)
        self.spatial_embedding = nn.Embedding(grid_size**2, hidden_dim)
        # Same name/shape deliberately inherits pretrained ACT timestep queries.
        self.action_queries = nn.Embedding(horizon, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim*4,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(hidden_dim),
                                            enable_nested_tensor=False)
        self.readout = L1RegressionActionHead(input_dim=hidden_dim, hidden_dim=hidden_dim*2,
                                              action_dim=action_dim, NUM_ACTIONS_CHUNK=horizon)
        for embedding in (self.view_embedding, self.spatial_embedding, self.action_queries):
            nn.init.normal_(embedding.weight, std=.02)
        prefix = num_views*grid_size**2 + 1
        blocked = torch.zeros(prefix+horizon, prefix+horizon, dtype=torch.bool)
        blocked[:prefix, prefix:] = True
        blocked[prefix:, prefix:] = torch.ones(horizon, horizon, dtype=torch.bool).triu(1)
        self.register_buffer('attention_blocked', blocked, persistent=False)

    def forward(self, patches, task, view_valid=None, action_valid=None):
        expected = (self.num_views, self.grid_size**2, self.patch_dim)
        if patches.ndim != 4 or tuple(patches.shape[1:]) != expected:
            raise ValueError(f'Expected current [B,{expected}] patches, got {patches.shape}')
        batch = patches.shape[0]
        if task.shape != (batch, self.task_dim):
            raise ValueError('Invalid task embedding shape')
        if view_valid is None:
            view_valid = torch.ones(batch, self.num_views, device=patches.device, dtype=torch.bool)
        if view_valid.shape != (batch, self.num_views) or view_valid.dtype != torch.bool:
            raise ValueError('Invalid view mask')
        if not view_valid.any(-1).all():
            raise ValueError('Each example requires at least one valid view')
        if action_valid is None:
            action_valid = torch.ones(batch, self.horizon, device=patches.device, dtype=torch.bool)
        if action_valid.shape != (batch, self.horizon) or action_valid.dtype != torch.bool:
            raise ValueError('Invalid action mask')
        patches = torch.where(view_valid[:, :, None, None], patches, 0.)
        if not torch.isfinite(patches).all() or not torch.isfinite(task).all():
            raise ValueError('Nonfinite valid patches or task')
        visual = (self.patch_projection(patches) + self.view_embedding.weight[None, :, None]
                  + self.spatial_embedding.weight[None, None]).flatten(1, 2)
        visual_valid = view_valid[:, :, None].expand(-1, -1, self.grid_size**2).flatten(1, 2)
        tokens = torch.cat([visual, self.task_projection(task)[:, None],
                            self.action_queries.weight[None].expand(batch, -1, -1)], dim=1)
        valid = torch.cat([visual_valid, torch.ones(batch, 1, device=patches.device, dtype=torch.bool),
                           action_valid], dim=1)
        hidden = self.fusion(tokens, mask=self.attention_blocked, src_key_padding_mask=~valid)
        actions = self.readout(hidden[:, -self.horizon:])
        return torch.where(action_valid[..., None], actions, 0.)
