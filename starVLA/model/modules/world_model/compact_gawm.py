"""Opt-in small GAWM study arms; frozen-teacher supervision stays explicit."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .fixed_dino_world_model import FixedDinoWorldModel
from .visual_token_delta_world_model import TokenResidualPredictor
from .temporal_regularization import temporal_curvature_loss


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

    def forward(self, latent, *, ctx_len, goal=None, update_stats=True, loss_mask=None,
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
