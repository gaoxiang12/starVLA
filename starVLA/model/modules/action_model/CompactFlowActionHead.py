"""Small GR00T-style flow expert for full spatial memory and masked actions.

Reuses this repository's GR00T ActionEncoder, MLP and NVIDIA DiT blocks. Unlike
the generic GR00T wrapper, padded action positions are masked both in the loss
and as self-attention keys. This module does not require a VLM backbone.
"""
import torch
import torch.nn.functional as F
from torch import nn

from starVLA.model.modules.action_model.GR00T_ActionHeader import ActionEncoder, MLP
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


class CompactFlowActionHead(nn.Module):
    def __init__(self, *, context_dim=384, hidden_dim=384, action_dim=14,
                 state_dim=14, horizon=16, depth=6, heads=6, planning_tokens=8,
                 inference_steps=4):
        super().__init__()
        if min(context_dim, hidden_dim, action_dim, state_dim, horizon, depth,
               heads, planning_tokens, inference_steps) < 1 or hidden_dim % heads:
            raise ValueError('Invalid compact flow dimensions')
        if depth < 2:
            raise ValueError('Flow expert needs cross- and self-attention layers')
        self.action_dim, self.state_dim, self.horizon = action_dim, state_dim, horizon
        self.context_dim, self.hidden_dim = context_dim, hidden_dim
        self.inference_steps, self.timestep_buckets = inference_steps, 1000
        self.model = DiT(num_attention_heads=heads, attention_head_dim=hidden_dim // heads,
                         output_dim=hidden_dim, num_layers=depth, dropout=0.,
                         final_dropout=False, interleave_self_attention=True,
                         use_canonical_forward=True, cross_attention_dim=context_dim,
                         norm_type='ada_norm', positional_embeddings=None)
        self.action_encoder = ActionEncoder(action_dim, hidden_dim)
        self.state_encoder = MLP(state_dim, hidden_dim, hidden_dim)
        self.action_decoder = MLP(hidden_dim, hidden_dim, action_dim)
        self.planning_tokens = nn.Embedding(planning_tokens, hidden_dim)
        self.action_position = nn.Embedding(horizon, hidden_dim)
        nn.init.normal_(self.planning_tokens.weight, std=.02)
        nn.init.normal_(self.action_position.weight, std=.02)

    def _inputs(self, context, state, context_valid, action_valid):
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError('Expected [batch, spatial_tokens, context_dim] memory')
        if state.ndim == 3 and state.shape[1] == 1:
            state = state[:, 0]
        if state.shape != (context.shape[0], self.state_dim):
            raise ValueError('Expected one current proprioceptive state')
        if context_valid is None:
            context_valid = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
        if action_valid is None:
            action_valid = torch.ones((context.shape[0], self.horizon), dtype=torch.bool, device=context.device)
        if context_valid.shape != context.shape[:2] or context_valid.dtype != torch.bool:
            raise ValueError('Invalid spatial-memory mask')
        if action_valid.shape != (context.shape[0], self.horizon) or action_valid.dtype != torch.bool:
            raise ValueError('Invalid action mask')
        if not context_valid.any(-1).all():
            raise ValueError('Each example requires at least one valid context token')
        context = torch.where(context_valid[..., None], context, 0.)
        if not torch.isfinite(context).all() or not torch.isfinite(state).all():
            raise ValueError('Nonfinite valid context or state')
        return context, state, context_valid, action_valid

    def velocity(self, context, state, noisy_actions, time, *, context_valid=None, action_valid=None):
        context, state, context_valid, action_valid = self._inputs(context, state, context_valid, action_valid)
        if noisy_actions.shape != (context.shape[0], self.horizon, self.action_dim):
            raise ValueError('Invalid noised action shape')
        if time.shape != (context.shape[0],) or not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
            raise ValueError('Flow time must be finite in [0,1] for every example')
        noisy_actions = torch.where(action_valid[..., None], noisy_actions, 0.)
        if not torch.isfinite(noisy_actions).all():
            raise ValueError('Nonfinite valid action')
        buckets = (time * self.timestep_buckets).long()
        actions = self.action_encoder(noisy_actions, buckets) + self.action_position.weight[None]
        plans = self.planning_tokens.weight[None].expand(context.shape[0], -1, -1)
        hidden = torch.cat((self.state_encoder(state[:, None]), plans, actions), 1)
        prefix_valid = torch.ones(hidden.shape[0], 1 + plans.shape[1], device=context.device, dtype=torch.bool)
        sequence_valid = torch.cat((prefix_valid, action_valid), 1)
        time_embedding = self.model.timestep_encoder(buckets)
        # Match canonical GR00T cross/self alternation, with an action key mask.
        for index, block in enumerate(self.model.transformer_blocks):
            self_attention = index % 2 == 1
            hidden = block(hidden,
                           encoder_hidden_states=None if self_attention else context,
                           encoder_attention_mask=sequence_valid if self_attention else context_valid,
                           temb=time_embedding)
        shift, scale = self.model.proj_out_1(F.silu(time_embedding)).chunk(2, -1)
        hidden = self.model.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
        prediction = self.action_decoder(self.model.proj_out_2(hidden))[:, -self.horizon:]
        return torch.where(action_valid[..., None], prediction, 0.)

    def loss(self, context, state, actions, *, context_valid=None, action_valid=None,
             noise=None, time=None):
        context, state, context_valid, action_valid = self._inputs(context, state, context_valid, action_valid)
        expected = (context.shape[0], self.horizon, self.action_dim)
        if actions.shape != expected:
            raise ValueError('Invalid action target shape')
        actions = torch.where(action_valid[..., None], actions, 0.)
        if noise is None:
            noise = torch.randn_like(actions)
        if noise.shape != expected:
            raise ValueError('Invalid flow noise shape')
        noise = torch.where(action_valid[..., None], noise, 0.)
        if not torch.isfinite(actions).all() or not torch.isfinite(noise).all():
            raise ValueError('Nonfinite valid flow target or noise')
        if time is None:
            beta = torch.distributions.Beta(actions.new_tensor(1.5), actions.new_tensor(1.))
            time = (1 - beta.sample((actions.shape[0],)).clamp(max=.999) / .999)
        mix = time[:, None, None]
        noisy = (1 - mix) * noise + mix * actions
        prediction = self.velocity(context, state, noisy, time,
                                   context_valid=context_valid, action_valid=action_valid)
        target_velocity = actions - noise
        squared = (prediction - target_velocity).float().square()
        loss = (squared * action_valid[..., None]).sum() / (action_valid.sum() * self.action_dim).clamp_min(1)
        # Denoised estimate for auxiliary diagnostics; not an inference rollout.
        estimate = torch.where(action_valid[..., None], noisy + (1-mix)*prediction, 0.)
        return dict(loss=loss, estimated_actions=estimate)

    @torch.no_grad()
    def predict_action(self, context, state, *, context_valid=None, initial_noise=None):
        context, state, context_valid, valid = self._inputs(context, state, context_valid, None)
        expected = (context.shape[0], self.horizon, self.action_dim)
        actions = torch.randn(expected, device=context.device, dtype=context.dtype) if initial_noise is None else initial_noise.clone()
        if actions.shape != expected:
            raise ValueError('Invalid inference noise shape')
        for index in range(self.inference_steps):
            time = context.new_full((context.shape[0],), index / self.inference_steps)
            actions = actions + self.velocity(context, state, actions, time,
                context_valid=context_valid, action_valid=valid) / self.inference_steps
        return actions
