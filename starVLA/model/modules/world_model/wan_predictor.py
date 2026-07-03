"""Lightweight Wan-style latent world model (ported / distilled from
thirdparty/zs-va/wan_va/modules/model.py:WanTransformer3DModel).

Differences vs. the original:
  * No FlexAttention / KV-cache / flash-attn — plain torch SDPA with an
    explicit block-causal attention mask.
  * No text cross-attention (single self-attention stream).
  * Operates on a *unified per-frame latent vector* (B, T, C) instead of VAE
    video latents or a spatial patch grid, and on a per-frame action vector.
    Each frame is a single token; positions use temporal-only RoPE.
  * Trained from scratch (no pretrained Wan weights), small by default.

Sequence layout (single stream):
  [latent tokens of all T frames] ++ [action tokens of all Ta transitions]
  - latent frame t -> mask frame id 2*t
  - action transition t -> mask frame id 2*t + 1
  Block-causal mask keeps kv_frame_id <= q_frame_id, so:
    * latent[t] attends to action[<t] (causal, no leak of the action that
      produces frame t+1),
    * action[t] attends to latent[<=t] (obs up to t decides action t).

The model predicts a flow-matching velocity for both latent and action
tokens; the FlowMatchScheduler defines the noise schedule / target.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

try:
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps

    _HAS_DIFFUSERS = True
except Exception:  # pragma: no cover - fallback if diffusers missing
    _HAS_DIFFUSERS = False


__all__ = ["WanPredictor"]


# --------------------------------------------------------------------------- #
# Time embedding
# --------------------------------------------------------------------------- #
class _SinusoidalTimeEmbedding(nn.Module):
    """Fallback sinusoidal timestep embedding (used if diffusers absent)."""

    def __init__(self, freq_dim, dim):
        super().__init__()
        self.freq_dim = freq_dim
        self.linear_1 = nn.Linear(freq_dim, dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim)

    def _timestep_embedding(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t):
        return self.linear_2(self.act(self.linear_1(self._timestep_embedding(t))))


class WanTimeEmbedding(nn.Module):
    """Maps per-token diffusion timesteps to (temb, 6-way modulation)."""

    def __init__(self, dim, freq_dim=256):
        super().__init__()
        if _HAS_DIFFUSERS:
            self.timesteps_proj = Timesteps(
                num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
            )
            self.time_embedder = TimestepEmbedding(in_channels=freq_dim, time_embed_dim=dim)
            self.act_fn = nn.SiLU()
            self.time_proj = nn.Linear(dim, dim * 6)
            self._diffusers = True
        else:
            self.embed = _SinusoidalTimeEmbedding(freq_dim, dim)
            self.act_fn = nn.SiLU()
            self.time_proj = nn.Linear(dim, dim * 6)
            self._diffusers = False

    def forward(self, timesteps):
        # timesteps: (B, L)
        B, L = timesteps.shape
        flat = timesteps.reshape(-1)
        if self._diffusers:
            proj = self.timesteps_proj(flat)
            w_dtype = self.time_embedder.linear_1.weight.dtype
            if proj.dtype != w_dtype:
                proj = proj.to(w_dtype)
            temb = self.time_embedder(proj)
        else:
            temb = self.embed(flat)
        mod = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), mod.reshape(B, L, 6, -1)


# --------------------------------------------------------------------------- #
# Rotary position embedding (3D: frame / height / width)
# --------------------------------------------------------------------------- #
class WanRotaryPosEmbed(nn.Module):
    def __init__(self, attention_head_dim, theta=10000.0):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.theta = theta
        self.half_dim = attention_head_dim // 2  # target freq length per token
        self.f_dim = attention_head_dim - 2 * (attention_head_dim // 3)
        self.h_dim = attention_head_dim // 3
        self.w_dim = attention_head_dim // 3
        self.register_buffer(
            "f_freqs_base",
            1.0 / (theta ** (torch.arange(0, self.f_dim, 2)[: self.f_dim // 2].double() / self.f_dim)),
            persistent=False,
        )
        self.register_buffer(
            "h_freqs_base",
            1.0 / (theta ** (torch.arange(0, self.h_dim, 2)[: self.h_dim // 2].double() / self.h_dim)),
            persistent=False,
        )
        self.register_buffer(
            "w_freqs_base",
            1.0 / (theta ** (torch.arange(0, self.w_dim, 2)[: self.w_dim // 2].double() / self.w_dim)),
            persistent=False,
        )

    def forward(self, grid_ids):
        # grid_ids: (B, 3, L) with rows (frame, h, w)
        with torch.no_grad():
            dev = grid_ids.device
            f = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(dev)
            h = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(dev)
            w = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(dev)
            freqs = torch.cat([f, h, w], dim=-1).float()  # (B, L, sum/2)
            if freqs.shape[-1] < self.half_dim:
                # pad missing components with phase 0 (no rotation)
                freqs = F.pad(freqs, (0, self.half_dim - freqs.shape[-1]))
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex
        return freqs_cis


def _apply_rotary(x, freqs):
    # x: (B, L, heads, head_dim) ; freqs: (B, L, head_dim/2) complex
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_c * freqs.unsqueeze(2)).flatten(3)
    return x_out.to(x.dtype)


# --------------------------------------------------------------------------- #
# Self-attention block (AdaLN-zero, SDPA)
# --------------------------------------------------------------------------- #
class WanSelfAttention(nn.Module):
    def __init__(self, dim, heads, eps=1e-5, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(dropout)
        self.norm_q = nn.RMSNorm(dim, eps=eps)
        self.norm_k = nn.RMSNorm(dim, eps=eps)

    def project(self, x, rotary_emb):
        """Project x -> (q, k, v), each (B, heads, L, head_dim), RoPE applied."""
        q = self.norm_q(self.to_q(x)).unflatten(2, (self.heads, self.head_dim))
        k = self.norm_k(self.to_k(x)).unflatten(2, (self.heads, self.head_dim))
        v = self.to_v(x).unflatten(2, (self.heads, self.head_dim))
        if rotary_emb is not None:
            q = _apply_rotary(q, rotary_emb)
            k = _apply_rotary(k, rotary_emb)
        # (B, heads, L, head_dim)
        return tuple(t.transpose(1, 2) for t in (q, k, v))

    def combine(self, out):
        """(B, heads, L, head_dim) attention output -> (B, L, dim)."""
        B, H, L, D = out.shape
        out = out.transpose(1, 2).reshape(B, L, H * D)
        return self.drop(self.to_out(out))

    def forward(self, x, rotary_emb, attn_mask):
        q, k, v = self.project(x, rotary_emb)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.combine(out)


class WanBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads, eps=1e-6, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.attn = WanSelfAttention(dim, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, mod, rotary_emb, attn_mask):
        # mod: (B, L, 6, dim)
        table = self.scale_shift_table[None] + mod.float()  # (B, L, 6, dim)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = table.unbind(2)
        h = (self.norm1(x.float()) * (1 + scale_a) + shift_a).type_as(x)
        x = (x.float() + self.attn(h, rotary_emb, attn_mask).float() * gate_a).type_as(x)
        h = (self.norm2(x.float()) * (1 + scale_f) + shift_f).type_as(x)
        x = (x.float() + self.ffn(h).float() * gate_f).type_as(x)
        return x

    def prime(self, x, mod, rotary_emb, attn_mask):
        """Context-only block forward that also returns this layer's K/V.

        Used by the cached flow-denoise path: the context tokens never attend
        to the (later) noised token under the block-causal mask, so their
        hidden states / K/V here are identical to those in the full forward.
        """
        table = self.scale_shift_table[None] + mod.float()
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = table.unbind(2)
        h = (self.norm1(x.float()) * (1 + scale_a) + shift_a).type_as(x)
        q, k, v = self.attn.project(h, rotary_emb)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = (x.float() + self.attn.combine(out).float() * gate_a).type_as(x)
        h = (self.norm2(x.float()) * (1 + scale_f) + shift_f).type_as(x)
        x = (x.float() + self.ffn(h).float() * gate_f).type_as(x)
        return x, k, v

    def step(self, x_new, mod_new, rotary_new, k_ctx, v_ctx):
        """Single noised token attends to cached context K/V plus its own.

        Attention is permutation-invariant over keys, so concatenating the new
        token's K/V after the context K/V yields the same result as the full
        forward (the noised token attends to every token, unmasked).
        """
        table = self.scale_shift_table[None] + mod_new.float()
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = table.unbind(2)
        h = (self.norm1(x_new.float()) * (1 + scale_a) + shift_a).type_as(x_new)
        q, k, v = self.attn.project(h, rotary_new)
        k_all = torch.cat([k_ctx, k], dim=2)
        v_all = torch.cat([v_ctx, v], dim=2)
        out = F.scaled_dot_product_attention(q, k_all, v_all)
        x_new = (x_new.float() + self.attn.combine(out).float() * gate_a).type_as(x_new)
        h = (self.norm2(x_new.float()) * (1 + scale_f) + shift_f).type_as(x_new)
        x_new = (x_new.float() + self.ffn(h).float() * gate_f).type_as(x_new)
        return x_new


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class WanPredictor(nn.Module):
    """Lightweight Wan-style latent + action flow-matching world model.

    Args:
        latent_dim: dim C of the per-frame unified latent vector.
        action_dim: dimension of the per-frame (effective) action vector.
        dim: transformer hidden size (inner_dim).
        num_layers / num_heads / ffn_dim: backbone size.
    """

    def __init__(
        self,
        *,
        latent_dim,
        action_dim,
        dim=384,
        num_layers=6,
        num_heads=6,
        ffn_dim=1024,
        freq_dim=256,
        eps=1e-6,
        dropout=0.0,
        goal_dim=None,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.tokens_per_frame = 1  # unified per-frame latent vector

        self.latent_in = nn.Linear(latent_dim, dim)
        self.action_in = nn.Linear(action_dim, dim)
        # Optional per-episode goal conditioning: encode goal vector and
        # broadcast-add it to every latent/action token (see forward).
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

        # Zero-init the velocity heads for a calm start (predict ~0 flow).
        nn.init.zeros_(self.latent_out.weight)
        nn.init.zeros_(self.latent_out.bias)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    # -- ids / mask --------------------------------------------------------- #
    def _build_ids(self, B, T, Ta, device):
        # latent tokens: one per frame, temporal-only position (h = w = 0)
        t_idx = torch.arange(T, device=device)
        lat_frame = 2 * t_idx  # mask frame id for latent
        # action tokens
        a_t = torch.arange(Ta, device=device)
        act_frame = 2 * a_t + 1
        # grid ids for rope: only the frame axis is used (h = w = 0)
        grid_f = torch.cat([t_idx, a_t]).float()
        grid_h = torch.zeros(T + Ta, device=device)
        grid_w = torch.zeros(T + Ta, device=device)
        grid_ids = torch.stack([grid_f, grid_h, grid_w], dim=0)[None].expand(B, -1, -1)
        frame_ids = torch.cat([lat_frame, act_frame])  # (L,)
        return grid_ids, frame_ids

    @staticmethod
    def _block_causal_mask(frame_ids):
        # True where attention is allowed: kv_frame <= q_frame
        q = frame_ids[:, None]
        k = frame_ids[None, :]
        return (k <= q)  # (L, L) bool

    # -- forward ------------------------------------------------------------ #
    def forward(self, latent, action, t_latent, t_action, goal=None):
        """Predict flow-matching velocity for latent + action tokens.

        Args:
            latent: (B, T, C) — per-frame unified latent vectors (context
                frames clean, future frames noised externally).
            action: (B, Ta, A) — per-transition action (noised externally).
            t_latent: (B, T) — diffusion timestep per latent frame.
            t_action: (B, Ta) — diffusion timestep per action transition.
            goal: (B, goal_dim) — optional per-episode goal vector, encoded and
                broadcast-added to every token. Ignored if goal_in is None.
        Returns:
            v_latent: (B, T, C)
            v_action: (B, Ta, A)
        """
        B, T = latent.shape[0], latent.shape[1]
        Ta = action.shape[1]
        device = latent.device

        lat_tok = self.latent_in(latent)                    # (B, T, dim)
        act_tok = self.action_in(action)                    # (B, Ta, dim)
        if goal is not None and self.goal_in is not None:
            g = self.goal_in(goal.to(lat_tok.dtype))[:, None]  # (B, 1, dim)
            lat_tok = lat_tok + g
            act_tok = act_tok + g
        x = torch.cat([lat_tok, act_tok], dim=1)            # (B, L, dim)

        grid_ids, frame_ids = self._build_ids(B, T, Ta, device)
        rotary_emb = self.rope(grid_ids)                    # (B, L, head_dim/2)
        attn_mask = self._block_causal_mask(frame_ids.to(device))  # (L, L)

        # per-token diffusion time: one latent token per frame
        tok_t = torch.cat([t_latent, t_action], dim=1)      # (B, L)
        temb, mod = self.time_embed(tok_t)                  # temb:(B,L,dim) mod:(B,L,6,dim)

        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(block, x, mod, rotary_emb, attn_mask, use_reentrant=False)
            else:
                x = block(x, mod, rotary_emb, attn_mask)

        # final AdaLN
        table = self.scale_shift_table[None] + temb[:, :, None, :]  # (B,L,2,dim)
        shift, scale = table.unbind(2)
        x = (self.norm_out(x.float()) * (1 + scale) + shift).type_as(x)

        lat_x, act_x = torch.split(x, [T, Ta], dim=1)
        v_latent = self.latent_out(lat_x)                   # (B, T, C)
        v_action = self.action_out(act_x)                   # (B, Ta, A)
        return v_latent, v_action

    # -- cached flow-denoise (inference only) ------------------------------- #
    @torch.no_grad()
    def init_denoise_cache(self, ctx_latent, ctx_action, goal=None):
        """Prime a per-layer context K/V cache for cached latent denoising.

        During ``sample_next_latent`` the clean context (latent history +
        actions) is fixed across all flow-denoise steps, and under the
        block-causal mask the context never attends to the noised future
        latent. Its per-layer K/V is therefore identical at every step and is
        computed once here. ``denoise_step_cached`` then forwards only the
        single noised token against this cache, cutting the per-step cost from
        a (2*HS+1)-token forward to a 1-token forward.

        Returns an opaque cache dict consumed by ``denoise_step_cached``.
        """
        B, HS = ctx_latent.shape[0], ctx_latent.shape[1]
        Ta = ctx_action.shape[1]
        device = ctx_latent.device

        lat_tok = self.latent_in(ctx_latent)
        act_tok = self.action_in(ctx_action)
        g = None
        if goal is not None and self.goal_in is not None:
            g = self.goal_in(goal.to(lat_tok.dtype))[:, None]  # (B, 1, dim)
            lat_tok = lat_tok + g
            act_tok = act_tok + g
        x = torch.cat([lat_tok, act_tok], dim=1)            # (B, HS+Ta, dim)

        # Same ids as the full forward's context block (latent grid 0..HS-1,
        # action grid 0..Ta-1); the noised latent sits at grid frame HS.
        grid_ids, frame_ids = self._build_ids(B, HS, Ta, device)
        rotary_ctx = self.rope(grid_ids)
        attn_mask = self._block_causal_mask(frame_ids.to(device))
        tok_t = torch.zeros(B, HS + Ta, device=device)      # context time = 0
        _, mod = self.time_embed(tok_t)

        kv = []
        for block in self.blocks:
            x, k, v = block.prime(x, mod, rotary_ctx, attn_mask)
            kv.append((k, v))

        grid_noisy = torch.zeros(B, 3, 1, device=device)
        grid_noisy[:, 0, 0] = float(HS)                     # noised latent frame
        rotary_noisy = self.rope(grid_noisy)
        return {"kv": kv, "g": g, "rotary_noisy": rotary_noisy, "HS": HS}

    @torch.no_grad()
    def denoise_step_cached(self, cache, x_noisy, t):
        """One cached flow step: velocity for the single noised latent token.

        Args:
            cache: dict returned by ``init_denoise_cache``.
            x_noisy: (B, 1, C) current noised latent.
            t: scalar diffusion timestep for this step.
        Returns:
            v_latent: (B, 1, C) predicted flow velocity (matches the full
                forward's ``v_latent[:, -1:]`` to floating-point tolerance).
        """
        B = x_noisy.shape[0]
        tok = self.latent_in(x_noisy)                       # (B, 1, dim)
        if cache["g"] is not None:
            tok = tok + cache["g"]
        t_vec = torch.full((B, 1), float(t), device=x_noisy.device, dtype=torch.float32)
        temb, mod = self.time_embed(t_vec)                  # temb:(B,1,dim)

        x = tok
        for block, (k_ctx, v_ctx) in zip(self.blocks, cache["kv"]):
            x = block.step(x, mod, cache["rotary_noisy"], k_ctx, v_ctx)

        table = self.scale_shift_table[None] + temb[:, :, None, :]  # (B,1,2,dim)
        shift, scale = table.unbind(2)
        x = (self.norm_out(x.float()) * (1 + scale) + shift).type_as(x)
        return self.latent_out(x)                           # (B, 1, C)


if __name__ == "__main__":
    # quick shape self-test
    torch.manual_seed(0)
    B, T, C, A = 2, 4, 384, 10
    model = WanPredictor(
        latent_dim=C, action_dim=A,
        dim=384, num_layers=4, num_heads=6, ffn_dim=1024,
    )
    z = torch.randn(B, T, C)
    a = torch.randn(B, T, A)
    tl = torch.randint(0, 1000, (B, T)).float()
    ta = torch.randint(0, 1000, (B, T)).float()
    vz, va = model(z, a, tl, ta)
    print("v_latent", tuple(vz.shape), "v_action", tuple(va.shape))
    assert vz.shape == z.shape and va.shape == a.shape
    print("OK params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
