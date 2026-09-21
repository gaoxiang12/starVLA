"""Three-view LiLa: shared visual adapters/DiT, semantic robot I/O heads."""
import torch
from torch import nn
from .primitives import (
    SinusoidalPosEmb, get_1d_sincos_pos_embed, MultiLayerConcatFusion,
    VisualFeatureAdapter, DiTBlock, FutureFeatureDecoder,
)


class RobotIO(nn.Module):
    def __init__(self, action_dim, state_dim, hidden):
        super().__init__()
        self.action = nn.Linear(action_dim, hidden)
        self.state = nn.Linear(state_dim, hidden)
        self.output = nn.Linear(hidden, action_dim)
        self.identity = nn.Parameter(torch.randn(1, 1, hidden) * .02)


class MultiViewLiLa(nn.Module):
    def __init__(self, heads, feature_dim, num_layers, patch_count, *, hidden=768,
                 depth=12, num_heads=8, adapter_depth=4, queries=64,
                 future_depth=4, future_heads=4, num_views=3):
        super().__init__()
        self.num_views, self.queries = num_views, queries
        self.efficient_views = False
        self.heads = nn.ModuleDict({tag: RobotIO(s['action_dim'], s['state_dim'], hidden)
                                    for tag, s in heads.items()})
        self.time = nn.Sequential(SinusoidalPosEmb(hidden), nn.Linear(hidden, hidden * 2),
                                  nn.SiLU(), nn.Linear(hidden * 2, hidden))
        self.fusion = MultiLayerConcatFusion(feature_dim, num_layers, feature_dim,
                                             proj_type='linear', pre_norm=True)
        self.adapter = VisualFeatureAdapter(feature_dim, hidden, queries, num_heads,
                                             adapter_depth, dropout=0.)
        self.view = nn.Parameter(torch.randn(1, num_views, 1, hidden) * .02)
        self.registers = nn.Parameter(torch.randn(1, 2, hidden) * .02)
        self.task = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden),
                                  nn.GELU(), nn.Linear(hidden, hidden))
        self.types = nn.Parameter(torch.randn(1, 4, hidden) * .02)
        horizon = max(s['action_horizon'] for s in heads.values())
        self.register_buffer('action_position', get_1d_sincos_pos_embed(hidden, horizon))
        self.register_buffer('state_position', get_1d_sincos_pos_embed(hidden, 1))
        self.blocks = nn.ModuleList([DiTBlock(hidden, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.future = FutureFeatureDecoder(patch_count, hidden, feature_dim,
                                           future_heads, future_depth)

    def visual_tokens(self, features, view_valid):
        b, v, n, d = features[0].shape
        flattened = [x.reshape(b * v, n, d) for x in features]
        if self.efficient_views:
            indices = view_valid.flatten().nonzero().flatten()
            adapted = self.adapter(self.fusion([x[indices] for x in flattened]))
            visual = adapted.new_zeros(b * v, self.queries, adapted.shape[-1]).index_copy(0, indices, adapted)
        else:
            visual = self.adapter(self.fusion(flattened))
        visual = visual.reshape(b, v, self.queries, -1) + self.view
        visual = visual.masked_fill(~view_valid[:, :, None, None], 0)
        return visual.flatten(1, 2)

    def forward(self, noisy, t, state, visual, task, view_valid, tag, action_valid=None):
        io = self.heads[tag]
        b, h, _ = noisy.shape
        action = io.action(noisy) + self.action_position[:, :h] + self.types[:, 0:1]
        proprio = io.state(state) + self.state_position + self.types[:, 1:2]
        cond = torch.cat((proprio, self.registers.expand(b, -1, -1),
                          self.task(task)[:, None] + self.types[:, 2:3],
                          visual + self.types[:, 3:4]), dim=1)
        cond = cond + io.identity
        # Missing cameras are excluded as keys at every layer, not just zeroed.
        cond_mask = torch.cat((torch.zeros(b, 4, dtype=torch.bool, device=noisy.device),
                               (~view_valid).repeat_interleave(self.queries, dim=1)), dim=1)
        if self.efficient_views:
            keep = ~cond_mask.all(dim=0)
            cond, cond_mask = cond[:, keep], cond_mask[:, keep]
        action_mask = torch.zeros(b, h, dtype=torch.bool, device=noisy.device) if action_valid is None else ~action_valid
        mask = torch.cat((action_mask, cond_mask), dim=1)
        x = torch.cat((action, cond), dim=1)
        emb = self.time(t)
        for block in self.blocks:
            x = block(x, emb, key_padding_mask=mask)
        x = self.norm(x)
        return io.output(x[:, :h]), x[:, h:], cond_mask

    def predict_future(self, cond, mask, view_valid=None):
        # A shared dense decoder, one query per spatial patch for each camera.
        predictions = []
        empty_future = self.efficient_views and view_valid is not None and not view_valid.any()
        # Match the reference's zero gradients (and AdamW weight decay) even
        # when every future target is padding and the decoder can be skipped.
        zero = sum(p.reshape(-1)[0] * 0 for p in self.future.parameters()) if empty_future else 0
        for v in range(self.num_views):
            if self.efficient_views and view_valid is not None and not view_valid[:, v].any():
                predictions.append(cond.new_zeros(len(cond), self.future.num_queries, self.future.out_dim) + zero)
            else:
                predictions.append(self.future(cond, mask, self.view[:, v]))
        return torch.stack(predictions, dim=1)


def masked_mean(values, valid):
    """Zero contribution and zero gradient for padding, including empty targets."""
    valid = torch.broadcast_to(valid, values.shape)
    return torch.where(valid, values, 0.).sum() / valid.sum().clamp_min(1)
