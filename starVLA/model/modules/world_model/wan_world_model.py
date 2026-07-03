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
        flow_sample_steps: int = 10,
        num_train_timesteps: int = 1000,
        scheduler_kwargs: Optional[dict] = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.ctx_len = int(ctx_len)
        self.flow_sample_steps = int(flow_sample_steps)
        self._train_num_timesteps = int(num_train_timesteps)

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

        self.scheduler = FlowMatchScheduler(**(scheduler_kwargs or {}))
        # Build the BSMNTW training weights over the discrete training schedule
        # (matches le-wm train.py setup).
        self.scheduler.set_timesteps(self._train_num_timesteps, training=True)

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

        ts_lat = self.scheduler.sample_timesteps(B * Tf, device).view(B, Tf)
        act_start = max(ctx_len - 1, 0)
        Ta = action.shape[1]
        Taf = Ta - act_start
        assert Taf > 0, "need at least one action target for flow loss"

        ts_act = self.scheduler.sample_timesteps(B * Taf, device).view(B, Taf)
        t_lat = torch.zeros(B, T, device=device)
        t_act = torch.zeros(B, Ta, device=device)
        t_lat[:, ctx_len:] = ts_lat
        t_act[:, act_start:] = ts_act

        noise_lat = torch.randn_like(latent)
        noise_act = torch.randn_like(action)
        lat_noisy = latent.clone()
        act_noisy = action.clone()
        lat_noisy[:, ctx_len:] = self.scheduler.add_noise(
            latent[:, ctx_len:], noise_lat[:, ctx_len:], ts_lat
        )
        act_noisy[:, act_start:] = self.scheduler.add_noise(
            action[:, act_start:], noise_act[:, act_start:], ts_act
        )

        v_lat, v_act = self.predictor(lat_noisy, act_noisy, t_lat, t_act, goal=goal)

        v_lat_f = v_lat[:, ctx_len:]
        v_act_f = v_act[:, act_start:]
        tgt_lat = noise_lat[:, ctx_len:] - latent[:, ctx_len:]
        tgt_act = noise_act[:, act_start:] - action[:, act_start:]

        w_lat = self.scheduler.training_weight(ts_lat.reshape(-1)).view(B, Tf)
        w_act = self.scheduler.training_weight(ts_act.reshape(-1)).view(B, Taf)
        lat_err = (v_lat_f - tgt_lat).pow(2).mean(dim=-1)
        act_err = (v_act_f - tgt_act).pow(2).mean(dim=-1)
        flow_latent_loss = (lat_err * w_lat).mean()
        flow_action_loss = (act_err * w_act).mean()

        # one-shot clean estimate of the future latent: x0 = x_t - sigma * v
        sigma = self.scheduler.sigma_for(ts_lat).to(latent).view(B, Tf, 1)
        pred_future_latent = lat_noisy[:, ctx_len:] - sigma * v_lat_f

        return {
            "flow_latent_loss": flow_latent_loss,
            "flow_action_loss": flow_action_loss,
            "pred_future_latent": pred_future_latent,
        }

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

        self.scheduler.set_timesteps(num_steps)
        try:
            lat = torch.cat(
                [ctx_latent, torch.randn(B, n_future, C, device=device, dtype=dtype)], dim=1
            )
            act = torch.zeros(B, Ta, A, device=device, dtype=dtype)
            act[:, act_start:] = torch.randn(B, n_future, A, device=device, dtype=dtype)

            for tt in self.scheduler.timesteps:
                t_lat = torch.zeros(B, T, device=device)
                t_act = torch.zeros(B, Ta, device=device)
                t_lat[:, ctx_len:] = tt
                t_act[:, act_start:] = tt
                v_lat, v_act = self.predictor(lat, act, t_lat, t_act, goal=goal)
                lat_future = self.scheduler.step(v_lat[:, ctx_len:], tt, lat[:, ctx_len:])
                act_future = self.scheduler.step(v_act[:, act_start:], tt, act[:, act_start:])
                lat = torch.cat([lat[:, :ctx_len], lat_future], dim=1)
                act = torch.cat([act[:, :act_start], act_future], dim=1)
        finally:
            # Restore the training schedule / BSMNTW weights.
            self.scheduler.set_timesteps(self._train_num_timesteps, training=True)

        if return_action:
            return lat[:, ctx_len:], act[:, act_start:]
        return lat[:, ctx_len:]
