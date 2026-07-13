"""Token-level Wan-style flow world model.

This variant keeps multiple visual tokens per frame instead of collapsing each
frame to one latent vector. The causal mask is still frame-level: all visual
tokens from a frame share the same causal id, so tokens within a frame can
attend to each other, future frames cannot leak backward, and action tokens see
only the context frames available at their transition.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_scheduler import FlowMatchScheduler
from .wan_world_model import SIGReg
from .wan_predictor import WanBlock, WanRotaryPosEmbed, WanTimeEmbedding


class TokenWanPredictor(nn.Module):
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
        num_tokens: int,
        token_grid_shape: Tuple[int, int, int],
        freq_dim: int = 256,
        eps: float = 1e-6,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.num_tokens = int(num_tokens)
        self.token_grid_shape = tuple(int(v) for v in token_grid_shape)
        if self.num_tokens != math.prod(self.token_grid_shape):
            raise ValueError(
                f"num_tokens={self.num_tokens} does not match token_grid_shape={self.token_grid_shape}"
            )

        self.latent_in = nn.Linear(latent_dim, dim)
        self.token_embedding = nn.Embedding(self.num_tokens, dim)
        self.action_in = nn.Linear(action_dim, dim)
        self.goal_dim = goal_dim
        self.goal_in = nn.Linear(goal_dim, dim) if goal_dim is not None else None

        self.rope = WanRotaryPosEmbed(self.head_dim)
        self.time_embed = WanTimeEmbedding(dim, freq_dim=freq_dim)
        self.blocks = nn.ModuleList(
            [WanBlock(dim, ffn_dim, num_heads, eps=eps, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm_out = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self.latent_out = nn.Linear(dim, latent_dim)
        self.action_out = nn.Linear(dim, action_dim)

        nn.init.zeros_(self.latent_out.weight)
        nn.init.zeros_(self.latent_out.bias)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def _build_ids(
        self, B: int, T: int, K: int, Ta: int, device: torch.device, action_frame_offset: int = 0
    ):
        t_idx = torch.arange(T, device=device)
        token_idx = torch.arange(K, device=device)
        if K != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} visual tokens, got {K}")
        _, grid_h, grid_w = self.token_grid_shape
        tokens_per_view = grid_h * grid_w
        view_idx = token_idx // tokens_per_view
        local_idx = token_idx % tokens_per_view
        row_idx = local_idx // grid_w
        col_idx = local_idx % grid_w

        lat_frame = (2 * t_idx).repeat_interleave(K)
        lat_f = t_idx.repeat_interleave(K).float()
        # Separate views along the height coordinate while preserving the
        # within-view 2-D grid for RoPE.
        lat_h = (view_idx * (grid_h + 1) + row_idx).repeat(T).float()
        lat_w = col_idx.repeat(T).float()

        a_t = torch.arange(Ta, device=device) + int(action_frame_offset)
        act_frame = 2 * a_t + 1
        act_f = a_t.float()
        act_h = torch.zeros(Ta, device=device)
        act_w = torch.zeros(Ta, device=device)

        rope_f = torch.cat([lat_f, act_f])
        rope_h = torch.cat([lat_h, act_h])
        rope_w = torch.cat([lat_w, act_w])
        grid_ids = torch.stack([rope_f, rope_h, rope_w], dim=0)[None].expand(B, -1, -1)
        frame_ids = torch.cat([lat_frame, act_frame])
        return grid_ids, frame_ids

    @staticmethod
    def _block_causal_mask(frame_ids: torch.Tensor):
        q = frame_ids[:, None]
        k = frame_ids[None, :]
        return k <= q

    def forward(self, latent, action, t_latent, t_action, goal=None, action_frame_offset: int = 0):
        """Predict flow velocity for visual-token latents and macro-actions.

        Args:
            latent: (B, T, K, C)
            action: (B, Ta, A)
            t_latent: (B, T)
            t_action: (B, Ta)
        """
        B, T, K = latent.shape[:3]
        Ta = action.shape[1]
        device = latent.device

        token_ids = torch.arange(K, device=device)
        lat_tok = self.latent_in(latent)
        lat_tok = lat_tok + self.token_embedding(token_ids).view(1, 1, K, self.dim)
        lat_tok = lat_tok.reshape(B, T * K, self.dim)
        act_tok = self.action_in(action)
        if goal is not None and self.goal_in is not None:
            goal_tok = self.goal_in(goal.to(lat_tok.dtype))[:, None]
            lat_tok = lat_tok + goal_tok
            act_tok = act_tok + goal_tok

        x = torch.cat([lat_tok, act_tok], dim=1)
        grid_ids, frame_ids = self._build_ids(B, T, K, Ta, device, action_frame_offset=action_frame_offset)
        rotary_emb = self.rope(grid_ids)
        attn_mask = self._block_causal_mask(frame_ids.to(device))

        tok_t = torch.cat([t_latent.repeat_interleave(K, dim=1), t_action], dim=1)
        temb, mod = self.time_embed(tok_t)

        for block in self.blocks:
            x = block(x, mod, rotary_emb, attn_mask)

        table = self.scale_shift_table[None] + temb[:, :, None, :]
        shift, scale = table.unbind(2)
        x = (self.norm_out(x.float()) * (1 + scale) + shift).type_as(x)

        lat_x, act_x = torch.split(x, [T * K, Ta], dim=1)
        v_latent = self.latent_out(lat_x).view(B, T, K, self.latent_dim)
        v_action = self.action_out(act_x)
        return v_latent, v_action


class _TokenDeltaBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TokenTransformerDeltaHead(nn.Module):
    """Predict future residuals for spatially identified visual tokens.

    The head sees only the current token set and the task embedding. Future
    actions are deliberately excluded so the OFT conditioning path is the same
    during teacher-forced training and deployment.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        goal_dim: Optional[int],
        n_future: int,
        num_tokens: int,
        dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.n_future = int(n_future)
        self.num_tokens = int(num_tokens)
        self.anchor_proj = nn.Linear(latent_dim, dim)
        self.goal_proj = nn.Linear(goal_dim, dim) if goal_dim else None
        self.future_query = nn.Parameter(torch.randn(1, self.n_future, 1, dim) * 0.02)
        self.frame_embedding = nn.Parameter(torch.randn(1, 1 + self.n_future, 1, dim) * 0.02)
        self.token_embedding = nn.Parameter(torch.randn(1, 1, self.num_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [_TokenDeltaBlock(dim, num_heads, ffn_dim) for _ in range(int(depth))]
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, anchor: torch.Tensor, goal: Optional[torch.Tensor] = None) -> torch.Tensor:
        # anchor: (B, 1, K, C)
        B, _, K, _ = anchor.shape
        if K != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} anchor tokens, got {K}")
        anchor_tok = self.anchor_proj(anchor)
        future_tok = self.future_query.expand(B, -1, K, -1)
        x = torch.cat([anchor_tok, future_tok], dim=1)
        x = x + self.frame_embedding + self.token_embedding
        if goal is not None and self.goal_proj is not None:
            x = x + self.goal_proj(goal.to(x.dtype)).view(B, 1, 1, -1)
        x = x.reshape(B, (1 + self.n_future) * K, -1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).view(B, 1 + self.n_future, K, -1)
        return self.out(x[:, 1:])


class TokenWanWorldModel(nn.Module):
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
        num_tokens: int,
        token_grid_shape: Tuple[int, int, int],
        ctx_len: int = 1,
        flow_sample_steps: int = 10,
        num_train_timesteps: int = 1000,
        scheduler_kwargs: Optional[dict] = None,
        action_scheduler_kwargs: Optional[dict] = None,
        gradient_checkpointing: bool = False,
        delta_head_futures: int = 0,
        delta_head_inference: bool = False,
        delta_head_dim: int = 384,
        delta_head_depth: int = 4,
        delta_head_heads: int = 6,
        delta_head_ffn: int = 1024,
        delta_head_sigreg_weight: float = 0.0,
        stats_momentum: float = 0.99,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.ctx_len = int(ctx_len)
        self.num_tokens = int(num_tokens)
        self.token_grid_shape = tuple(int(v) for v in token_grid_shape)
        self.flow_sample_steps = int(flow_sample_steps)
        self._train_num_timesteps = int(num_train_timesteps)
        self.predictor = TokenWanPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_tokens=self.num_tokens,
            token_grid_shape=self.token_grid_shape,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.scheduler = FlowMatchScheduler(**(scheduler_kwargs or {}))
        self.action_scheduler = FlowMatchScheduler(**(action_scheduler_kwargs or {}))
        self.scheduler.set_timesteps(self._train_num_timesteps, training=True)
        self.action_scheduler.set_timesteps(self._train_num_timesteps, training=True)

        self.delta_head_futures = int(delta_head_futures)
        self.delta_head_inference = bool(delta_head_inference)
        self.stats_momentum = float(stats_momentum)
        self._stats_eps = 1e-4
        self.register_buffer("delta_scale", torch.ones(1))
        self.register_buffer("_delta_scale_ready", torch.zeros(1))
        if self.delta_head_futures > 0:
            self.delta_head = TokenTransformerDeltaHead(
                latent_dim=self.latent_dim,
                goal_dim=goal_dim,
                n_future=self.delta_head_futures,
                num_tokens=self.num_tokens,
                dim=delta_head_dim,
                depth=delta_head_depth,
                num_heads=delta_head_heads,
                ffn_dim=delta_head_ffn,
            )
            self.sigreg = SIGReg() if delta_head_sigreg_weight > 0 else None
        else:
            self.delta_head = None
            self.sigreg = None

    @torch.no_grad()
    def _update_delta_scale(self, residual: torch.Tensor) -> None:
        rms = residual.float().pow(2).mean().clamp_min(self._stats_eps).sqrt()
        if float(self._delta_scale_ready) < 1.0:
            self.delta_scale.fill_(float(rms))
            self._delta_scale_ready.fill_(1.0)
        else:
            self.delta_scale.mul_(self.stats_momentum).add_(
                rms, alpha=1 - self.stats_momentum
            )

    def regress_future(self, ctx_latent, goal=None):
        """Deterministically predict absolute future visual tokens."""
        if self.delta_head is None:
            raise RuntimeError("token delta head is disabled")
        anchor = ctx_latent[:, -1:]
        pred_delta = self.delta_head(anchor, goal=goal)
        return anchor + pred_delta * self.delta_scale.clamp_min(self._stats_eps)

    def flow_loss(self, latent, action, ctx_len=None, goal=None):
        ctx_len = self.ctx_len if ctx_len is None else ctx_len
        B, T, K = latent.shape[:3]
        Tf = T - ctx_len
        device = latent.device
        assert Tf > 0, "need at least one future frame for flow loss"

        ts_lat = self.scheduler.sample_timesteps(B * Tf, device).view(B, Tf)
        Ta = action.shape[1]
        Taf = Ta
        assert Taf > 0, "need at least one action target for flow loss"
        ts_act = self.action_scheduler.sample_timesteps(B * Taf, device).view(B, Taf)

        t_lat = torch.zeros(B, T, device=device)
        t_act = torch.zeros(B, Ta, device=device)
        t_lat[:, ctx_len:] = ts_lat
        t_act[:, :] = ts_act

        noise_lat = torch.randn_like(latent)
        noise_act = torch.randn_like(action)
        lat_noisy = latent.clone()
        act_noisy = action.clone()
        lat_noisy[:, ctx_len:] = self.scheduler.add_noise(
            latent[:, ctx_len:], noise_lat[:, ctx_len:], ts_lat
        )
        act_noisy[:, :] = self.action_scheduler.add_noise(action, noise_act, ts_act)

        action_frame_offset = max(ctx_len - 1, 0)
        v_lat, v_act = self.predictor(
            lat_noisy, act_noisy, t_lat, t_act, goal=goal, action_frame_offset=action_frame_offset
        )
        v_lat_f = v_lat[:, ctx_len:]
        v_act_f = v_act
        tgt_lat = noise_lat[:, ctx_len:] - latent[:, ctx_len:]
        tgt_act = noise_act - action

        w_lat = self.scheduler.training_weight(ts_lat.reshape(-1)).view(B, Tf)
        w_act = self.action_scheduler.training_weight(ts_act.reshape(-1)).view(B, Taf)
        lat_err = (v_lat_f - tgt_lat).pow(2).mean(dim=(-1, -2))
        act_err = (v_act_f - tgt_act).pow(2).mean(dim=-1)
        flow_latent_loss = (lat_err * w_lat).mean()
        flow_action_loss = (act_err * w_act).mean()

        sigma = self.scheduler.sigma_for(ts_lat).to(latent).view(B, Tf, 1, 1)
        pred_future_latent = lat_noisy[:, ctx_len:] - sigma * v_lat_f
        out = {
            "flow_latent_loss": flow_latent_loss,
            "flow_action_loss": flow_action_loss,
            "pred_future_latent": pred_future_latent,
        }

        if self.delta_head is not None:
            if self.delta_head_futures > Tf:
                raise ValueError(
                    f"delta_head_futures={self.delta_head_futures} exceeds available future frames={Tf}"
                )
            anchor = latent[:, ctx_len - 1 : ctx_len]
            residual = (
                latent[:, ctx_len : ctx_len + self.delta_head_futures] - anchor
            ).detach()
            if self.training:
                self._update_delta_scale(residual)
            scale = self.delta_scale.clamp_min(self._stats_eps)
            target_delta = residual / scale
            pred_delta = self.delta_head(anchor, goal=goal)
            out["delta_latent_loss"] = (pred_delta - target_delta).pow(2).mean()
            pred_residual = pred_delta * scale
            out["delta_future_latent"] = anchor + pred_residual

            with torch.no_grad():
                copy_mse = residual.float().pow(2).mean()
                pred_mse = (pred_residual.float() - residual.float()).pow(2).mean()
                mean_residual = residual.float().mean(dim=0, keepdim=True)
                mean_baseline_mse = (
                    residual.float() - mean_residual
                ).pow(2).mean()
                direction_cosine = F.cosine_similarity(
                    pred_residual.float().flatten(2),
                    residual.float().flatten(2),
                    dim=-1,
                    eps=1e-8,
                ).mean()
                out.update(
                    {
                        "delta_scale": scale.detach().mean(),
                        "delta_target_rms": residual.float().pow(2).mean().sqrt(),
                        "delta_pred_rms": pred_residual.float().pow(2).mean().sqrt(),
                        "delta_copy_mse": copy_mse,
                        "delta_pred_mse": pred_mse,
                        "delta_mean_baseline_mse": mean_baseline_mse,
                        "delta_to_copy_ratio": pred_mse / copy_mse.clamp_min(1e-8),
                        "delta_direction_cosine": direction_cosine,
                    }
                )
            if self.sigreg is not None:
                tf, channels = pred_delta.shape[1], pred_delta.shape[-1]
                sigreg_input = pred_delta.permute(1, 0, 2, 3).reshape(
                    tf, B * K, channels
                )
                out["delta_sigreg_loss"] = self.sigreg(sigreg_input)
        return out

    @torch.no_grad()
    def sample_future(self, ctx_latent, goal=None, n_future=2, num_steps=None, return_action=False):
        num_steps = self.flow_sample_steps if num_steps is None else num_steps
        B, ctx_len, K, C = ctx_latent.shape
        A = self.predictor.action_dim
        device = ctx_latent.device
        dtype = ctx_latent.dtype
        T = ctx_len + n_future
        action_frame_offset = max(ctx_len - 1, 0)
        Ta = n_future

        self.scheduler.set_timesteps(num_steps)
        self.action_scheduler.set_timesteps(num_steps)
        try:
            lat = torch.cat(
                [ctx_latent, torch.randn(B, n_future, K, C, device=device, dtype=dtype)], dim=1
            )
            act = torch.randn(B, Ta, A, device=device, dtype=dtype)

            for tt_lat, tt_act in zip(
                self.scheduler.timesteps, self.action_scheduler.timesteps
            ):
                t_lat = torch.zeros(B, T, device=device)
                t_act = torch.zeros(B, Ta, device=device)
                t_lat[:, ctx_len:] = tt_lat
                t_act[:, :] = tt_act
                v_lat, v_act = self.predictor(
                    lat, act, t_lat, t_act, goal=goal, action_frame_offset=action_frame_offset
                )
                lat_future = self.scheduler.step(
                    v_lat[:, ctx_len:], tt_lat, lat[:, ctx_len:]
                )
                act_future = self.action_scheduler.step(v_act, tt_act, act)
                lat = torch.cat([lat[:, :ctx_len], lat_future], dim=1)
                act = act_future
        finally:
            self.scheduler.set_timesteps(self._train_num_timesteps, training=True)
            self.action_scheduler.set_timesteps(self._train_num_timesteps, training=True)

        future_latent = lat[:, ctx_len:]
        if (
            self.delta_head is not None
            and self.delta_head_inference
            and n_future == self.delta_head_futures
        ):
            future_latent = self.regress_future(ctx_latent, goal=goal)

        if return_action:
            return future_latent, act
        return future_latent
