# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
WanWorldModel — flow-matching latent world model for the LeWM-OFT framework.

Wraps the ported le-wm ``WanPredictor`` (Wan-style temporal transformer) and
``FlowMatchScheduler`` so the action framework can:

  * ``flow_loss(latent, action, ctx_len, goal)`` — teacher-forced training pass
    that jointly denoises the *future* latent frames and their driving actions
    in a single predictor call, returning ``flow_latent_loss`` (future-latent
    prediction), ``flow_action_loss`` and the one-shot clean estimate
    ``pred_future_latent`` that the OFT head consumes.

  * ``sample_future(ctx_latent, goal, n_future)`` — inference rollout: jointly
    flow-samples the future latents + actions from noise, conditioned on the
    clean current latent (and the per-task ``goal`` vector). Used by
    ``predict_action`` so the OFT head still "sees" an imagined future when no
    ground-truth future frame is available.

The teacher-forced ``flow_loss`` math is distilled from
``thirdparty/le-wm-dev.libero/jepa.py:JEPA.flow_loss`` (the side-car / state /
idm losses are dropped — only the latent + action flow terms remain).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_scheduler import FlowMatchScheduler
from .wan_predictor import WanPredictor


def _modulate(x, shift, scale):
    """AdaLN-zero modulation (le-wm convention)."""
    return x * (1 + scale) + shift


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (ported from le-wm module.py).

    Penalizes deviation of a projected feature distribution from an isotropic
    unit Gaussian via the Epps-Pulley statistic over random 1-D projections.
    Applied here to the delta-head predictions to keep them from collapsing to
    the batch mean (the observed failure of the whitened MLP head).
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = int(num_proj)
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        # proj: (T, B, D) -- B is the sample dim for the empirical CF estimate.
        proj = proj.float()
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class _AdaLNBlock(nn.Module):
    """Transformer block with AdaLN-zero action conditioning (le-wm style)."""

    def __init__(self, dim, num_heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1 * attn_out
        h = _modulate(self.norm2(x), shift2, scale2)
        x = x + gate2 * self.mlp(h)
        return x


class TransformerDeltaHead(nn.Module):
    """Transformer predictor for future-latent residuals.

    Mirrors le-wm's ``ARPredictor`` (AdaLN-zero conditioned transformer) but in
    a parallel query form: an anchor token plus one learnable query per future
    step attend over each other. By default it is action-free: it predicts the
    latent relation from current latent (+goal) only, so the OFT head cannot see
    teacher-forced future actions through the predicted latent. Predicts the
    per-step residual ``future - anchor`` in a *globally* scaled space (single
    scalar, NO per-dim whitening) so the loss stays aligned with absolute-latent
    error.
    """

    def __init__(self, *, latent_dim, action_dim, goal_dim, n_future,
                 dim=384, depth=4, num_heads=6, ffn_dim=1024,
                 condition_on_action=False):
        super().__init__()
        self.n_future = int(n_future)
        self.latent_dim = int(latent_dim)
        self.condition_on_action = bool(condition_on_action)
        self.anchor_proj = nn.Linear(latent_dim, dim)
        self.act_proj = nn.Linear(action_dim, dim) if self.condition_on_action else None
        self.goal_proj = nn.Linear(goal_dim, dim) if goal_dim else None
        self.query = nn.Parameter(torch.randn(1, self.n_future, dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 1 + self.n_future, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [_AdaLNBlock(dim, num_heads, ffn_dim) for _ in range(int(depth))]
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, latent_dim)
        # Zero-init output -> initial residual prediction is ~0, i.e. the head
        # starts as a copy of the anchor (a safe warm-start when swapped in for
        # the flow-sampled future latent mid-training).
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, anchor, actions=None, goal=None):
        # anchor: (B,1,C); optional actions: (B,Tf,A); goal: (B,goal_dim) or None.
        B = anchor.shape[0]
        anchor_tok = self.anchor_proj(anchor)          # (B,1,dim)
        query_tok = self.query.expand(B, -1, -1)       # (B,Tf,dim)
        x = torch.cat([anchor_tok, query_tok], dim=1) + self.pos
        if self.condition_on_action:
            if actions is None:
                raise ValueError("TransformerDeltaHead requires actions when condition_on_action=True")
            act_c = self.act_proj(actions)             # (B,Tf,dim)
        else:
            act_c = x.new_zeros(B, self.n_future, x.shape[-1])
        if self.goal_proj is not None and goal is not None:
            g = self.goal_proj(goal).unsqueeze(1)      # (B,1,dim)
            anchor_c = g
            act_c = act_c + g
        else:
            anchor_c = x.new_zeros(B, 1, x.shape[-1])
        c = torch.cat([anchor_c, act_c], dim=1)        # (B,1+Tf,dim)
        for blk in self.blocks:
            x = blk(x, c)
        x = self.norm(x)
        return self.out(x[:, 1:])                       # (B,Tf,C) scaled residual


class WanWorldModel(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        goal_dim: Optional[int] = None,
        dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 1024,
        ctx_len: int = 1,
        flow_sample_steps: int = 20,
        num_train_timesteps: int = 1000,
        scheduler_kwargs: Optional[dict] = None,
        action_scheduler_kwargs: Optional[dict] = None,
        noisy_cond_prob: float = 0.5,
        gradient_checkpointing: bool = False,
        whiten_latent: bool = True,
        predict_residual: bool = True,
        stats_momentum: float = 0.99,
        delta_head_futures: int = 0,
        delta_head_hidden: int = 1024,
        delta_head_inference: bool = False,
        delta_head_type: str = "mlp",
        delta_head_dim: int = 384,
        delta_head_depth: int = 4,
        delta_head_heads: int = 6,
        delta_head_ffn: int = 1024,
        delta_head_sigreg_weight: float = 0.0,
        delta_head_condition_on_action: bool = False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.ctx_len = int(ctx_len)
        self.flow_sample_steps = int(flow_sample_steps)
        self._train_num_timesteps = int(num_train_timesteps)

        # === Latent flow-space normalization (fix for ill-conditioned target) ==
        # The raw fused latent has tiny per-dim cross-sample std (~0.13) while
        # the flow target mixes in unit-variance N(0,1) noise -> SNR ~1/60, so
        # flow_latent_loss plateaus and sample_future diverges. We track running
        # per-dim statistics and either:
        #   * whiten the context/current latent to unit variance (whiten_latent), and
        #   * flow-match the *whitened residual* (future - current) instead of the
        #     near-static absolute future latent (predict_residual).
        # Stats are running EMA buffers updated only during training (flow_loss)
        # and used (frozen) in both flow_loss and sample_future, so train/eval
        # normalization is identical.
        self.whiten_latent = bool(whiten_latent)
        self.predict_residual = bool(predict_residual)
        self.use_latent_stats = self.whiten_latent or self.predict_residual
        self.stats_momentum = float(stats_momentum)
        self._stats_eps = 1e-4
        self.register_buffer("lat_mean", torch.zeros(self.latent_dim))
        self.register_buffer("lat_std", torch.ones(self.latent_dim))
        self.register_buffer("resid_mean", torch.zeros(self.latent_dim))
        self.register_buffer("resid_std", torch.ones(self.latent_dim))
        self.register_buffer("_stats_ready", torch.zeros(1))

        self.predictor = WanPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            gradient_checkpointing=gradient_checkpointing,
        )

        # === Direct delta-regression auxiliary head ===
        # Predicts future-latent residuals deterministically from the current
        # latent (+goal) by default. This keeps the OFT training path aligned
        # with deployment: future latents must be available before actions are
        # generated, so clean future actions cannot be a delta-head input. An
        # action-conditioned mode remains available only for explicit ablations.
        self.delta_head_futures = int(delta_head_futures)
        self.delta_head_inference = bool(delta_head_inference)
        self.delta_head_type = str(delta_head_type).lower()
        self.delta_head_sigreg_weight = float(delta_head_sigreg_weight)
        self.delta_head_condition_on_action = bool(delta_head_condition_on_action)
        self._delta_goal_dim = int(goal_dim) if goal_dim else 0

        # Global (scalar) EMA scale of the residual (future - current). The
        # transformer delta head regresses residuals normalized by this single
        # scalar -- unlike per-dim whitening it preserves each dim's relative
        # magnitude, so the MSE stays aligned with absolute-latent error (which
        # the ridge oracle exploits) while staying O(1) for bf16 stability.
        self.register_buffer("delta_scale", torch.ones(1))
        self.register_buffer("_delta_scale_ready", torch.zeros(1))
        self.sigreg = None
        if self.delta_head_futures > 0:
            if self.delta_head_type == "transformer":
                self.delta_head = TransformerDeltaHead(
                    latent_dim=self.latent_dim,
                    action_dim=self.action_dim,
                    goal_dim=self._delta_goal_dim,
                    n_future=self.delta_head_futures,
                    dim=delta_head_dim,
                    depth=delta_head_depth,
                    num_heads=delta_head_heads,
                    ffn_dim=delta_head_ffn,
                    condition_on_action=self.delta_head_condition_on_action,
                )
            else:
                in_dim = self.latent_dim + self._delta_goal_dim
                if self.delta_head_condition_on_action:
                    in_dim += self.delta_head_futures * self.action_dim
                self.delta_head = nn.Sequential(
                    nn.Linear(in_dim, delta_head_hidden),
                    nn.GELU(),
                    nn.Linear(delta_head_hidden, delta_head_hidden),
                    nn.GELU(),
                    nn.Linear(delta_head_hidden, self.delta_head_futures * self.latent_dim),
                )
            if self.delta_head_sigreg_weight > 0:
                self.sigreg = SIGReg()
        else:
            self.delta_head = None

        # === Schedulers (lingbot-va recipe) ===
        # Latent and action use SEPARATE schedulers with very different SNR
        # shifts (reference: snr_shift=5.0 for latent i.e. high-noise emphasis,
        # action_snr_shift=0.05 i.e. low-noise emphasis), sigma_min=0 and
        # extra_one_step=True so the schedule spans the full [0, 1] sigma range.
        if scheduler_kwargs is None:
            scheduler_kwargs = {"shift": 5.0, "sigma_min": 0.0, "extra_one_step": True}
        if action_scheduler_kwargs is None:
            action_scheduler_kwargs = {"shift": 0.05, "sigma_min": 0.0, "extra_one_step": True}
        self.scheduler = FlowMatchScheduler(**scheduler_kwargs)
        self.action_scheduler = FlowMatchScheduler(**action_scheduler_kwargs)
        # Build the BSMNTW training weights over the discrete training schedule
        # (matches le-wm train.py setup).
        self.scheduler.set_timesteps(self._train_num_timesteps, training=True)
        self.action_scheduler.set_timesteps(self._train_num_timesteps, training=True)

        # Noisy-context augmentation (lingbot-va noisy_cond_prob=0.5): with this
        # probability the clean context frames are themselves noised at a
        # moderate sigma and the context timesteps are exposed to the predictor,
        # so imperfect / drifted context at rollout time stays in-distribution.
        self.noisy_cond_prob = float(noisy_cond_prob)

    # ------------------------------------------------------------------ #
    # Latent flow-space (de)normalization
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_latent_stats(self, latent, residual):
        """Running per-dim EMA of the latent mean/std and residual mean/std.

        ``latent`` : (B, T, C) all frames; ``residual`` : (B, Tf, C) future-current.
        First call seeds the buffers directly from the batch, subsequent calls
        blend with ``stats_momentum`` (BatchNorm-style running statistics).
        """
        f = latent.reshape(-1, self.latent_dim).float()
        r = residual.reshape(-1, self.latent_dim).float()
        f_mean, f_std = f.mean(0), f.std(0).clamp_min(self._stats_eps)
        r_mean, r_std = r.mean(0), r.std(0).clamp_min(self._stats_eps)
        if float(self._stats_ready) < 1.0:
            self.lat_mean.copy_(f_mean)
            self.lat_std.copy_(f_std)
            self.resid_mean.copy_(r_mean)
            self.resid_std.copy_(r_std)
            self._stats_ready.fill_(1.0)
        else:
            m = self.stats_momentum
            self.lat_mean.mul_(m).add_(f_mean, alpha=1 - m)
            self.lat_std.mul_(m).add_(f_std, alpha=1 - m)
            self.resid_mean.mul_(m).add_(r_mean, alpha=1 - m)
            self.resid_std.mul_(m).add_(r_std, alpha=1 - m)

    @torch.no_grad()
    def _update_delta_scale(self, residual):
        """Running EMA of a single global RMS scale of the residual."""
        rms = residual.float().pow(2).mean().clamp_min(self._stats_eps).sqrt()
        if float(self._delta_scale_ready) < 1.0:
            self.delta_scale.fill_(float(rms))
            self._delta_scale_ready.fill_(1.0)
        else:
            m = self.stats_momentum
            self.delta_scale.mul_(m).add_(rms, alpha=1 - m)

    def _whiten_latent(self, x):
        """Whiten an absolute latent to ~unit per-dim variance (context path)."""
        if not self.whiten_latent:
            return x
        return (x - self.lat_mean) / self.lat_std

    def _encode_target(self, fut, anchor):
        """Map absolute future latent -> unit-scale flow-space target.

        residual: (future - current - mu_r) / sigma_r ; else whitened absolute.
        """
        if self.predict_residual:
            return (fut - anchor - self.resid_mean) / self.resid_std
        if self.whiten_latent:
            return (fut - self.lat_mean) / self.lat_std
        return fut

    def _decode_target(self, z, anchor):
        """Inverse of ``_encode_target``: flow-space sample -> absolute future."""
        if self.predict_residual:
            return anchor + (z * self.resid_std + self.resid_mean)
        if self.whiten_latent:
            return z * self.lat_std + self.lat_mean
        return z

    # ------------------------------------------------------------------ #
    # Direct delta regression (deterministic, no flow sampling)
    # ------------------------------------------------------------------ #
    def _delta_head_flow_pred(self, anchor, actions=None, goal=None):
        """Delta-head prediction in the unit-scale flow space.

        anchor: (B,1,C) last clean context frame (absolute space);
        optional actions: (B,Tf,A) future macro-actions. Returns (B,Tf,C).
        """
        B = anchor.shape[0]
        parts = [self._whiten_latent(anchor[:, 0])]
        if self.delta_head_condition_on_action:
            if actions is None:
                raise ValueError("delta head requires actions when delta_head_condition_on_action=True")
            parts.append(actions.reshape(B, -1).to(anchor.dtype))
        if self._delta_goal_dim:
            g = goal if goal is not None else anchor.new_zeros(B, self._delta_goal_dim)
            parts.append(g.to(anchor.dtype))
        z = self.delta_head(torch.cat(parts, dim=-1))
        return z.view(B, self.delta_head_futures, self.latent_dim)

    def regress_future(self, ctx_latent, actions=None, goal=None):
        """Deterministic future latents from the delta head (absolute space).

        ctx_latent: (B,ctx,C); optional actions: (B,Tf,A). Returns (B,Tf,C).
        """
        assert self.delta_head is not None, "delta head disabled (delta_head_futures=0)"
        anchor = ctx_latent[:, -1:]
        if self.delta_head_type == "transformer":
            pred_delta = self.delta_head(anchor, actions, goal=goal)
            return anchor + pred_delta * self.delta_scale.clamp_min(self._stats_eps)
        z = self._delta_head_flow_pred(anchor, actions, goal=goal)
        return self._decode_target(z, anchor)

    # ------------------------------------------------------------------ #
    # Training: teacher-forced flow-matching loss
    # ------------------------------------------------------------------ #
    def flow_loss(self, latent, action, ctx_len=None, goal=None):
        """Single joint pass denoising future latents + future actions.

        Args:
            latent: (B, T, C) per-frame latent; frames ``[0, ctx_len)`` are the
                clean context, frames ``[ctx_len, T)`` are noised targets.
            action: (B, Ta, A) per-(macro)transition action; the action driving
                each future frame is a noised target.
            ctx_len: number of clean context frames (default ``self.ctx_len``).
            goal: (B, goal_dim) optional per-task conditioning (the task emb).
        Returns dict: flow_latent_loss, flow_action_loss, pred_future_latent.
        """
        ctx_len = self.ctx_len if ctx_len is None else ctx_len
        B, T = latent.shape[0], latent.shape[1]
        Tf = T - ctx_len
        device = latent.device
        assert Tf > 0, "need at least one future frame for flow loss"

        anchor = latent[:, ctx_len - 1 : ctx_len]  # (B,1,C) last clean context frame
        fut = latent[:, ctx_len:]  # (B,Tf,C) absolute future targets

        # Running per-dim latent/residual statistics (train-time only).
        if self.use_latent_stats and self.training:
            self._update_latent_stats(latent, fut - anchor)

        # Context + target mapped into the unit-scale flow space.
        ctx_flow = self._whiten_latent(latent[:, :ctx_len])
        tgt = self._encode_target(fut, anchor)  # (B,Tf,C) whitened residual / absolute

        ts_lat = self.scheduler.sample_timesteps(B * Tf, device).view(B, Tf)
        act_start = max(ctx_len - 1, 0)
        Ta = action.shape[1]
        Taf = Ta - act_start
        assert Taf > 0, "need at least one action target for flow loss"

        ts_act = self.action_scheduler.sample_timesteps(B * Taf, device).view(B, Taf)
        t_lat = torch.zeros(B, T, device=device)
        t_act = torch.zeros(B, Ta, device=device)
        t_lat[:, ctx_len:] = ts_lat
        t_act[:, act_start:] = ts_act

        # Noisy-context augmentation: noise the context frames at a moderate
        # sigma (upper index half of the schedule = sigma in (0, shifted(0.5)],
        # same as lingbot-va sample_timestep_id(min_bd=0.5)) and tell the
        # predictor via t_lat so it learns to use imperfect context.
        if self.training and self.noisy_cond_prob > 0 and torch.rand(()).item() < self.noisy_cond_prob:
            n_sched = len(self.scheduler.timesteps)
            cond_idx = torch.randint(n_sched // 2, n_sched, (B * ctx_len,))
            ts_cond = self.scheduler.timesteps[cond_idx].to(device).view(B, ctx_len)
            ctx_flow = self.scheduler.add_noise(ctx_flow, torch.randn_like(ctx_flow), ts_cond)
            t_lat[:, :ctx_len] = ts_cond

        noise_lat = torch.randn_like(tgt)
        noise_act = torch.randn_like(action)
        tgt_noisy = self.scheduler.add_noise(tgt, noise_lat, ts_lat)
        lat_noisy = torch.cat([ctx_flow, tgt_noisy], dim=1)  # (B,T,C)
        act_noisy = action.clone()
        act_noisy[:, act_start:] = self.action_scheduler.add_noise(
            action[:, act_start:], noise_act[:, act_start:], ts_act
        )

        v_lat, v_act = self.predictor(lat_noisy, act_noisy, t_lat, t_act, goal=goal)

        v_lat_f = v_lat[:, ctx_len:]
        v_act_f = v_act[:, act_start:]
        tgt_lat = noise_lat - tgt
        tgt_act = noise_act[:, act_start:] - action[:, act_start:]

        w_lat = self.scheduler.training_weight(ts_lat.reshape(-1)).view(B, Tf)
        w_act = self.action_scheduler.training_weight(ts_act.reshape(-1)).view(B, Taf)
        lat_err = (v_lat_f - tgt_lat).pow(2).mean(dim=-1)
        act_err = (v_act_f - tgt_act).pow(2).mean(dim=-1)
        flow_latent_loss = (lat_err * w_lat).mean()
        flow_action_loss = (act_err * w_act).mean()

        # one-shot clean estimate in flow space (z0 = z_t - sigma * v), decoded
        # back to the absolute future latent the OFT head consumes.
        sigma = self.scheduler.sigma_for(ts_lat).to(tgt).view(B, Tf, 1)
        z0 = tgt_noisy - sigma * v_lat_f
        pred_future_latent = self._decode_target(z0, anchor)

        out = {
            "flow_latent_loss": flow_latent_loss,
            "flow_action_loss": flow_action_loss,
            "pred_future_latent": pred_future_latent,
        }

        # Auxiliary delta head: regress future latents from the current latent
        # (+goal) by default. Optional action conditioning is kept only for
        # explicit ablations; using clean future actions here leaks GT actions
        # into the OFT head through delta_future_latent.
        if self.delta_head is not None:
            act_f = None
            if self.delta_head_condition_on_action:
                act_f = action[:, act_start : act_start + self.delta_head_futures]
            if self.delta_head_type == "transformer":
                # Raw residual target, normalized by a single global scalar (no
                # per-dim whitening) so the loss weights dims by true magnitude.
                resid = (fut[:, : self.delta_head_futures] - anchor).detach()
                if self.training:
                    self._update_delta_scale(resid)
                scale = self.delta_scale.clamp_min(self._stats_eps)
                tgt_delta = resid / scale
                pred_delta = self.delta_head(anchor, act_f, goal=goal)
                out["delta_latent_loss"] = (pred_delta - tgt_delta).pow(2).mean()
                out["delta_future_latent"] = anchor + pred_delta * scale
                if self.sigreg is not None:
                    out["delta_sigreg_loss"] = self.sigreg(pred_delta.transpose(0, 1))
            else:
                z_delta = self._delta_head_flow_pred(anchor, act_f, goal=goal)
                out["delta_latent_loss"] = (z_delta - tgt[:, : self.delta_head_futures]).pow(2).mean()
                out["delta_future_latent"] = self._decode_target(z_delta, anchor)

        return out

    # ------------------------------------------------------------------ #
    # Inference: joint flow-sampling rollout of future latents + actions
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_future(self, ctx_latent, goal=None, n_future=2, num_steps=None, return_action=False):
        """Roll out ``n_future`` future latents from the clean context.

        Jointly denoises the future latent frames and their (unknown) driving
        actions from noise in one flow-sampling loop, conditioned on the clean
        context latent + ``goal``. Returns ``(B, n_future, C)``.

        When ``return_action`` is True, also returns the flow-sampled future
        (macro-)actions ``(B, n_future, A)`` so the caller can execute them
        directly instead of the OFT head output (flow-vs-OFT A/B).

        The shared scheduler's discrete schedule is mutated for the short
        inference loop and restored to the training schedule before returning,
        so interleaved train/eval calls do not corrupt the training weights.
        """
        num_steps = self.flow_sample_steps if num_steps is None else num_steps
        B, ctx_len, C = ctx_latent.shape
        A = self.predictor.action_dim
        device = ctx_latent.device
        dtype = ctx_latent.dtype
        T = ctx_len + n_future
        act_start = max(ctx_len - 1, 0)
        Ta = act_start + n_future

        anchor = ctx_latent[:, ctx_len - 1 : ctx_len]  # (B,1,C)
        ctx_flow = self._whiten_latent(ctx_latent)  # context in flow space (fixed)

        self.scheduler.set_timesteps(num_steps)
        self.action_scheduler.set_timesteps(num_steps)
        try:
            # Future latents live in the unit-scale flow space (whitened residual
            # / absolute); sample them from N(0,1) noise which now matches scale.
            z = torch.randn(B, n_future, C, device=device, dtype=dtype)
            act = torch.zeros(B, Ta, A, device=device, dtype=dtype)
            act[:, act_start:] = torch.randn(B, n_future, A, device=device, dtype=dtype)

            # Joint denoising loop; latent and action follow their OWN sigma
            # schedules (different SNR shifts), stepping in lockstep.
            for tt_lat, tt_act in zip(self.scheduler.timesteps, self.action_scheduler.timesteps):
                lat = torch.cat([ctx_flow, z], dim=1)
                t_lat = torch.zeros(B, T, device=device)
                t_act = torch.zeros(B, Ta, device=device)
                t_lat[:, ctx_len:] = tt_lat
                t_act[:, act_start:] = tt_act
                v_lat, v_act = self.predictor(lat, act, t_lat, t_act, goal=goal)
                z = self.scheduler.step(v_lat[:, ctx_len:], tt_lat, z)
                act_future = self.action_scheduler.step(
                    v_act[:, act_start:], tt_act, act[:, act_start:]
                )
                act = torch.cat([act[:, :act_start], act_future], dim=1)
        finally:
            # Restore the training schedules / BSMNTW weights.
            self.scheduler.set_timesteps(self._train_num_timesteps, training=True)
            self.action_scheduler.set_timesteps(self._train_num_timesteps, training=True)

        future_latent = self._decode_target(z, anchor)  # -> absolute future latent
        # Optional: replace the flow-sampled latent with the deterministic
        # delta-head prediction. By default the delta head is action-free; if an
        # ablation explicitly enables action conditioning, use the flow-sampled
        # future actions rather than teacher-forced actions at rollout.
        if (
            self.delta_head is not None
            and self.delta_head_inference
            and n_future == self.delta_head_futures
        ):
            act_cond = act[:, act_start:] if self.delta_head_condition_on_action else None
            future_latent = self.regress_future(ctx_latent, act_cond, goal=goal)
        if return_action:
            return future_latent, act[:, act_start:]
        return future_latent
