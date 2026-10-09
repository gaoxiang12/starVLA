"""Compact future prediction supervised in a frozen DINO patch coordinate system."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .visual_token_delta_world_model import TokenResidualPredictor
from .temporal_regularization import temporal_curvature_loss


class PatchFeatureDecoder(nn.Module):
    """Shared per-camera spatial readout; no current-image skip or teacher input."""
    def __init__(self, latent_dim, feature_dim, num_patches, num_views,
                 tokens_per_view, dim=384, depth=2, heads=6):
        super().__init__()
        self.num_views = num_views
        self.tokens_per_view = tokens_per_view
        self.query = nn.Parameter(torch.randn(1, num_patches, dim) * .02)
        self.memory = nn.Linear(latent_dim, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0.,
                                          batch_first=True, norm_first=True, activation='gelu')
        self.decoder = nn.TransformerDecoder(layer, depth, norm=nn.LayerNorm(dim))
        self.output = nn.Linear(dim, feature_dim)

    def forward(self, latent):
        b, t, k, d = latent.shape
        if k != self.num_views * self.tokens_per_view:
            raise ValueError('Decoder camera/token layout mismatch')
        memory = self.memory(latent.reshape(b*t*self.num_views, self.tokens_per_view, d))
        decoded = self.decoder(self.query.expand(memory.shape[0], -1, -1), memory)
        features = self.output(decoded)
        return features.reshape(b, t, self.num_views, *features.shape[-2:])


class FixedDinoWorldModel(nn.Module):
    """Future loss trains current adapter, predictor and decoder; teacher is detached.

    No adapter-space target or target-dependent EMA scale is used. Predictions
    use a fixed residual multiplier and non-affine LayerNorm at train AND test.
    """
    def __init__(self, *, latent_dim, goal_dim, n_future, num_tokens, num_views,
                 tokens_per_view, feature_dim, num_patches, dim=384, depth=4,
                 num_heads=6, ffn_dim=1024, decoder_dim=384, decoder_depth=2,
                 decoder_heads=6, reconstruction_weight=.1, smoothness_weight=0.,
                 time_offsets=(0., .2, .4), dense_smoothness_weight=0.,
                 temporal_reference_dt=.05):
        super().__init__()
        self.num_views, self.tokens_per_view = num_views, tokens_per_view
        self.num_tokens, self.n_future = num_tokens, n_future
        if n_future != 2:
            raise ValueError('Two fixed future horizons are required')
        self.reconstruction_weight = float(reconstruction_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.dense_smoothness_weight = float(dense_smoothness_weight)
        self.temporal_reference_dt = float(temporal_reference_dt)
        if not math.isfinite(self.temporal_reference_dt) or self.temporal_reference_dt <= 0:
            raise ValueError("temporal_reference_dt must be finite and positive")
        offsets = torch.tensor(time_offsets, dtype=torch.float32)
        if offsets.shape != (3,) or not torch.isfinite(offsets).all() or not torch.all(offsets[1:] > offsets[:-1]):
            raise ValueError('Three finite increasing temporal offsets are required')
        self.register_buffer('time_offsets', offsets)
        self.register_buffer('objective_version', torch.tensor(1, dtype=torch.uint8))
        self.residual_predictor = TokenResidualPredictor(latent_dim=latent_dim, goal_dim=goal_dim,
            n_future=n_future, num_tokens=num_tokens, dim=dim, depth=depth,
            num_heads=num_heads, ffn_dim=ffn_dim)
        self.feature_decoder = PatchFeatureDecoder(latent_dim, feature_dim, num_patches,
            num_views, tokens_per_view, decoder_dim, decoder_depth, decoder_heads)

    def regress_future(self, context, goal=None):
        if context.shape[1:] != (1, self.num_tokens, self.residual_predictor.out.out_features):
            raise ValueError('Expected a single current frame with the configured tokens')
        future = context + self.residual_predictor(context, goal=goal)
        return F.layer_norm(future.float(), (future.shape[-1],)).to(future.dtype)

    @staticmethod
    def cosine_error(prediction, target):
        return 1. - F.cosine_similarity(prediction.float(), target.detach().float(), dim=-1, eps=1e-6)

    @staticmethod
    def masked_mean(error, view_mask):
        weights = view_mask.to(error.dtype).unsqueeze(-1).expand_as(error)
        return (error * weights).sum() / weights.sum().clamp_min(1.)

    def forward(self, latent, *, ctx_len, goal=None, update_stats=True,
                loss_mask=None, teacher_patches=None, temporal_latent=None, temporal_times=None, temporal_valid=None, temporal_event_weight=None):
        b, t, k, _ = latent.shape
        if (ctx_len, t, k) != (1, 3, self.num_tokens):
            raise ValueError('Expected current and two future latent frames')
        if teacher_patches is None or teacher_patches.shape[:3] != (b, t, self.num_views):
            raise ValueError('Frozen DINO patch teacher is required in training')
        teacher = teacher_patches.detach()
        if loss_mask is None:
            valid = torch.ones(b, t, self.num_views, device=latent.device, dtype=torch.bool)
        else:
            grouped = loss_mask.reshape(b, t, self.num_views, self.tokens_per_view)
            if not torch.equal(grouped, grouped[..., :1].expand_as(grouped)):
                raise ValueError('Patch supervision requires uniform validity within each camera')
            valid = grouped[..., 0].bool()
        future = self.regress_future(latent[:, :1], goal)
        decoded = self.feature_decoder(torch.cat([latent[:, :1], future], dim=1))
        if decoded.shape != teacher.shape:
            raise ValueError(f'Decoder/teacher shape mismatch: {decoded.shape} != {teacher.shape}')
        errors = self.cosine_error(decoded, teacher)
        future_loss = self.masked_mean(errors[:, 1:], valid[:, 1:])
        current_loss = self.masked_mean(errors[:, :1], valid[:, :1])
        # Coarse three-time-point smoothness only, not a claim of frame-level smoothness.
        sequence = torch.cat([latent[:, :1], future], dim=1).float()
        dt = (self.time_offsets[1:] - self.time_offsets[:-1]).to(sequence.device)
        velocities = (sequence[:, 1:] - sequence[:, :-1]) / dt.view(1, 2, 1, 1)
        bend = (velocities[:, 1] - velocities[:, 0]) * dt.mean()
        smooth_error = F.smooth_l1_loss(bend, torch.zeros_like(bend), reduction='none', beta=.1)
        smooth_error = smooth_error.reshape(b, self.num_views, self.tokens_per_view, -1).mean(-1)
        smooth_loss = self.masked_mean(smooth_error, valid.all(dim=1))
        aux_loss = self.reconstruction_weight * current_loss + self.smoothness_weight * smooth_loss
        dense_metrics = {}
        if self.dense_smoothness_weight > 0:
            if temporal_latent is None:
                raise ValueError("Adjacent temporal supervision is missing")
            repeated_goal = goal.repeat_interleave(3, dim=0) if goal is not None else None
            temporal_predictions = self.regress_future(
                temporal_latent.reshape(b*3, 1, self.num_tokens, -1), repeated_goal
            ).reshape(b, 3, self.n_future, self.num_tokens, -1)
            observed_smooth = temporal_curvature_loss(temporal_latent, temporal_times, temporal_valid, temporal_event_weight, reference_dt=self.temporal_reference_dt)
            predicted_smooth = temporal_curvature_loss(temporal_predictions, temporal_times, temporal_valid, temporal_event_weight, reference_dt=self.temporal_reference_dt)
            dense_loss = .5 * (observed_smooth + predicted_smooth)
            aux_loss = aux_loss + self.dense_smoothness_weight * dense_loss
            dense_metrics = dict(temporal_dense_loss=dense_loss,
                temporal_dense_current_loss=observed_smooth, temporal_dense_prediction_loss=predicted_smooth,
                temporal_dense_weighted_loss=self.dense_smoothness_weight*dense_loss,
                temporal_dense_valid_fraction=temporal_valid.float().mean(),
                temporal_dense_event_weight=temporal_event_weight.mean())
        result = dict(pred_future_latent=future, latent_loss=future_loss,
                      latent_cosine_loss=future_loss.new_zeros(()), auxiliary_loss=aux_loss,
                      dino_future_loss=future_loss, dino_current_loss=current_loss,
                      temporal_smoothness_loss=smooth_loss)
        result.update(dense_metrics)
        for h in range(2):
            result[f'latent_loss_horizon_{h+1}'] = self.masked_mean(errors[:, h+1:h+2], valid[:, h+1:h+2])
        with torch.no_grad():
            copy_error = self.cosine_error(teacher[:, :1].expand_as(teacher[:, 1:]), teacher[:, 1:])
            copy_loss = self.masked_mean(copy_error, valid[:, 1:])
            weights = valid[:, 1:, :, None, None].float()
            mean_teacher = (teacher[:, 1:].float()*weights).sum(0, keepdim=True)/weights.sum(0, keepdim=True).clamp_min(1.)
            mean_loss = self.masked_mean(self.cosine_error(mean_teacher.expand_as(teacher[:, 1:]), teacher[:, 1:]), valid[:, 1:])
            change = (latent[:, 1:].float()-latent[:, :-1].float()).square().mean(-1)
            change = change.reshape(b, 2, self.num_views, self.tokens_per_view)
            result.update(dino_copy_loss=copy_loss, dino_batch_mean_loss=mean_loss,
                dino_to_copy_ratio=future_loss.detach()/copy_loss.clamp_min(1e-6),
                predicted_latent_rms=future.float().square().mean().sqrt(),
                latent_batch_std=latent[:, 0].float().var(0, unbiased=False).mean().sqrt(),
                temporal_observed_delta_rms=self.masked_mean(change, valid[:, 1:] & valid[:, :-1]).sqrt())
        return result
