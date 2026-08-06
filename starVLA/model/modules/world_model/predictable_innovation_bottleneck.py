"""Action-free bottleneck for predictable local visual dynamics.

The mature world model predicts normalized *cumulative* future deltas.  This
module freezes that prediction and represents only its remaining error.  The
representation is deliberately smaller and more dynamic than a per-token
cumulative residual:

1. cumulative +4/+8 errors are converted to local increments;
2. fixed ``(horizon, spatial-token, channel)`` means are removed;
3. a shared spatial basis and a shared channel basis encode each increment;
4. the causal predictor sees visual history, the frozen prediction and task,
   but never future targets, actions, or proprioceptive state;
5. decoded local innovations are cumulatively summed back into +4/+8 deltas.

With 32 visual tokens, four transition modes and rank 64, the deployed code is
``H x 4 x 64`` instead of ``H x 32 x 64``.  Both bases have orthonormal rows,
so the oracle is a genuine low-rank projection rather than an unconstrained
autoencoder.  A zero-initialized predictor preserves the warm-start model
exactly at step zero.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualAttentionBlock(nn.Module):
    """Small pre-norm self-attention block used by the causal predictor."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(hidden)
        hidden = hidden + self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        return hidden + self.mlp(self.norm2(hidden))


class _CausalInnovationPredictor(nn.Module):
    """Predict compact local-increment codes without future/action inputs."""

    def __init__(
        self,
        *,
        latent_dim: int,
        goal_dim: Optional[int],
        n_future: int,
        num_tokens: int,
        transition_tokens: int,
        context_len: int,
        rank: int,
        dim: int,
        depth: int,
        num_heads: int,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(
                f"predictor dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.latent_dim = int(latent_dim)
        self.n_future = int(n_future)
        self.num_tokens = int(num_tokens)
        self.transition_tokens = int(transition_tokens)
        self.context_len = int(context_len)
        self.rank = int(rank)

        self.context_projection = nn.Linear(self.latent_dim, dim)
        self.base_projection = nn.Linear(self.latent_dim, dim)
        self.goal_projection = nn.Linear(goal_dim, dim) if goal_dim else None

        self.context_embedding = nn.Parameter(
            torch.randn(1, self.context_len, self.num_tokens, dim) * 0.02
        )
        self.base_horizon_embedding = nn.Parameter(
            torch.randn(1, self.n_future, 1, dim) * 0.02
        )
        self.spatial_token_embedding = nn.Parameter(
            torch.randn(1, 1, self.num_tokens, dim) * 0.02
        )
        self.query_horizon_embedding = nn.Parameter(
            torch.randn(1, self.n_future, 1, dim) * 0.02
        )
        self.transition_mode_embedding = nn.Parameter(
            torch.randn(1, 1, self.transition_tokens, dim) * 0.02
        )
        self.context_type_embedding = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.base_type_embedding = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.query_type_embedding = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.transition_queries = nn.Parameter(
            torch.randn(1, self.n_future, self.transition_tokens, dim) * 0.02
        )

        self.blocks = nn.ModuleList(
            [
                _ResidualAttentionBlock(dim, num_heads, ffn_dim)
                for _ in range(int(depth))
            ]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, self.rank)

        # Exact warm-start compatibility: no correction before optimization.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _validate_inputs(
        self,
        context: torch.Tensor,
        base_prediction: torch.Tensor,
        goal: Optional[torch.Tensor],
    ) -> None:
        if context.ndim != 4:
            raise ValueError(
                "context must have shape (B, context_len, num_tokens, latent_dim), "
                f"got {tuple(context.shape)}"
            )
        batch = context.shape[0]
        expected_context = (
            batch,
            self.context_len,
            self.num_tokens,
            self.latent_dim,
        )
        if tuple(context.shape) != expected_context:
            raise ValueError(
                f"expected context shape {expected_context}, got {tuple(context.shape)}"
            )

        expected_base = (
            batch,
            self.n_future,
            self.num_tokens,
            self.latent_dim,
        )
        if base_prediction.ndim != 4 or tuple(base_prediction.shape) != expected_base:
            raise ValueError(
                f"expected base_prediction shape {expected_base}, "
                f"got {tuple(base_prediction.shape)}"
            )

        if self.goal_projection is not None:
            if goal is None:
                raise ValueError("task-conditioned innovation predictor requires goal")
            expected_goal = (batch, self.goal_projection.in_features)
            if goal.ndim != 2 or tuple(goal.shape) != expected_goal:
                raise ValueError(
                    f"expected goal shape {expected_goal}, got {tuple(goal.shape)}"
                )

    def forward(
        self,
        context: torch.Tensor,
        base_prediction: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``(B, H, M, R)`` causal local-dynamics codes."""

        self._validate_inputs(context, base_prediction, goal)
        batch = context.shape[0]
        base_prediction = base_prediction.detach()

        context_hidden = self.context_projection(context)
        context_hidden = (
            context_hidden
            + self.context_embedding.to(context_hidden.dtype)
            + self.spatial_token_embedding.to(context_hidden.dtype)
            + self.context_type_embedding.to(context_hidden.dtype)
        )

        base_hidden = self.base_projection(base_prediction.to(context_hidden.dtype))
        base_hidden = (
            base_hidden
            + self.base_horizon_embedding.to(base_hidden.dtype)
            + self.spatial_token_embedding.to(base_hidden.dtype)
            + self.base_type_embedding.to(base_hidden.dtype)
        )

        queries = self.transition_queries.to(context_hidden.dtype).expand(batch, -1, -1, -1)
        queries = (
            queries
            + self.query_horizon_embedding.to(queries.dtype)
            + self.transition_mode_embedding.to(queries.dtype)
            + self.query_type_embedding.to(queries.dtype)
        )

        hidden = torch.cat(
            (
                context_hidden.flatten(1, 2),
                base_hidden.flatten(1, 2),
                queries.flatten(1, 2),
            ),
            dim=1,
        )
        if self.goal_projection is not None:
            task = self.goal_projection(goal.to(hidden.dtype)).unsqueeze(1)
            hidden = hidden + task

        for block in self.blocks:
            hidden = block(hidden)

        query_count = self.n_future * self.transition_tokens
        query_hidden = self.output_norm(hidden[:, -query_count:]).view(
            batch, self.n_future, self.transition_tokens, -1
        )
        return self.output(query_hidden)


class _OrthonormalRows(nn.Module):
    """Differentiable row-orthonormal projection basis."""

    def __init__(self, input_dim: int, rank: int) -> None:
        super().__init__()
        if rank < 1 or rank > input_dim:
            raise ValueError(f"rank must be in [1, {input_dim}], got {rank}")
        self.input_dim = int(input_dim)
        self.rank = int(rank)
        raw = torch.randn(self.input_dim, self.rank)
        with torch.no_grad():
            raw.copy_(torch.linalg.qr(raw, mode="reduced").Q)
        self.raw_basis = nn.Parameter(raw)

    def forward(self) -> torch.Tensor:
        # Projection and decoding consume this fp32 result even under bf16.
        q, r = torch.linalg.qr(self.raw_basis.float(), mode="reduced")
        diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
        signs = torch.where(
            diagonal < 0,
            -torch.ones_like(diagonal),
            torch.ones_like(diagonal),
        )
        return (q * signs.detach().unsqueeze(-2)).transpose(-2, -1)


class PredictableInnovationBottleneck(nn.Module):
    """Learn compact, non-collapsed, action-free local visual dynamics."""

    def __init__(
        self,
        *,
        latent_dim: int,
        goal_dim: Optional[int],
        n_future: int,
        num_tokens: int,
        context_len: int = 1,
        rank: int = 64,
        transition_tokens: int = 4,
        predictor_dim: int = 384,
        predictor_depth: int = 4,
        predictor_heads: int = 6,
        predictor_ffn_dim: int = 1024,
        min_code_std: float = 0.25,
        require_fixed_mean: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("n_future", n_future),
            ("num_tokens", num_tokens),
            ("context_len", context_len),
            ("rank", rank),
            ("transition_tokens", transition_tokens),
            ("predictor_dim", predictor_dim),
            ("predictor_depth", predictor_depth),
            ("predictor_heads", predictor_heads),
            ("predictor_ffn_dim", predictor_ffn_dim),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if int(transition_tokens) > int(num_tokens):
            raise ValueError("transition_tokens cannot exceed num_tokens")
        if min_code_std < 0:
            raise ValueError(f"min_code_std must be non-negative, got {min_code_std}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.latent_dim = int(latent_dim)
        self.n_future = int(n_future)
        self.num_tokens = int(num_tokens)
        self.context_len = int(context_len)
        self.rank = int(rank)
        self.transition_tokens = int(transition_tokens)
        self.min_code_std = float(min_code_std)
        self.require_fixed_mean = bool(require_fixed_mean)
        self.eps = float(eps)

        # Shared coordinates make +4 and +4->+8 increments comparable.
        self.basis = _OrthonormalRows(self.latent_dim, self.rank)
        self.spatial_basis = _OrthonormalRows(
            self.num_tokens, self.transition_tokens
        )
        self.predictor = _CausalInnovationPredictor(
            latent_dim=self.latent_dim,
            goal_dim=goal_dim,
            n_future=self.n_future,
            num_tokens=self.num_tokens,
            transition_tokens=self.transition_tokens,
            context_len=self.context_len,
            rank=self.rank,
            dim=int(predictor_dim),
            depth=int(predictor_depth),
            num_heads=int(predictor_heads),
            ffn_dim=int(predictor_ffn_dim),
        )

        # The raw-coordinate mean is independent of rotations of either learned
        # basis. It is calibrated once from frozen training data, then used
        # unchanged by both training and held-out evaluation. The target never
        # depends on co-batch samples or per-rank batch composition.
        self.register_buffer(
            "fixed_local_error_mean",
            torch.zeros(1, self.n_future, self.num_tokens, self.latent_dim),
        )
        self.register_buffer("fixed_local_error_count", torch.zeros(1))

    def _apply(self, *args, **kwargs):
        mean_fp32 = self.fixed_local_error_mean.detach().float().clone()
        count_fp32 = self.fixed_local_error_count.detach().float().clone()
        module = super()._apply(*args, **kwargs)
        device = module.fixed_local_error_mean.device
        module.fixed_local_error_mean = mean_fp32.to(device=device)
        module.fixed_local_error_count = count_fp32.to(device=device)
        return module

    @torch.no_grad()
    def set_fixed_error_mean(
        self, mean: torch.Tensor, *, sample_count: int | float
    ) -> None:
        expected = tuple(self.fixed_local_error_mean.shape)
        if mean.ndim == 3:
            mean = mean.unsqueeze(0)
        if tuple(mean.shape) != expected:
            raise ValueError(
                f"expected fixed local-error mean shape {expected}, got {tuple(mean.shape)}"
            )
        if float(sample_count) <= 0:
            raise ValueError("fixed local-error mean sample_count must be positive")
        if not torch.isfinite(mean).all():
            raise ValueError("fixed local-error mean contains non-finite values")
        self.fixed_local_error_mean.copy_(
            mean.detach().to(
                device=self.fixed_local_error_mean.device, dtype=torch.float32
            )
        )
        self.fixed_local_error_count.fill_(float(sample_count))

    def _validate_target(
        self, target_delta: torch.Tensor, base_prediction: torch.Tensor
    ) -> None:
        if target_delta.ndim != 4 or target_delta.shape != base_prediction.shape:
            raise ValueError(
                "target_delta must match base_prediction shape "
                f"{tuple(base_prediction.shape)}, got {tuple(target_delta.shape)}"
            )

    @staticmethod
    def _to_local(cumulative: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (cumulative[:, :1], cumulative[:, 1:] - cumulative[:, :-1]),
            dim=1,
        )

    @staticmethod
    def _to_cumulative(local: torch.Tensor) -> torch.Tensor:
        return local.cumsum(dim=1)

    @staticmethod
    def _project(
        values: torch.Tensor,
        spatial_basis: torch.Tensor,
        channel_basis: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "bhkd,mk,rd->bhmr", values, spatial_basis, channel_basis
        )

    @staticmethod
    def _decode(
        codes: torch.Tensor,
        spatial_basis: torch.Tensor,
        channel_basis: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "bhmr,mk,rd->bhkd", codes, spatial_basis, channel_basis
        )

    def _center_local_error(self, local_error: torch.Tensor) -> torch.Tensor:
        """Remove fixed token means while retaining sample-specific dynamics."""

        local_error = local_error.float()
        if self.require_fixed_mean and float(self.fixed_local_error_count.item()) <= 0:
            raise RuntimeError(
                "predictable innovation requires a precomputed fixed local-error mean"
            )
        return local_error - self.fixed_local_error_mean

    @torch.no_grad()
    def _distribution_diagnostics(
        self, code: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Report variation after removing every transition-mode identity."""

        code = code.float()
        batch, horizons, modes, rank = code.shape
        mode_centered = code - code.mean(dim=0, keepdim=True)
        flattened = mode_centered.permute(1, 0, 2, 3).reshape(
            horizons, batch * modes, rank
        )
        std = flattened.square().mean(dim=1).clamp_min(0.0).sqrt()
        denominator = max(flattened.shape[1] - 1, 1)
        covariance = torch.matmul(flattened.transpose(-1, -2), flattened)
        covariance = covariance / denominator
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        eigenvalue_sum = eigenvalues.sum(dim=-1, keepdim=True)
        probabilities = eigenvalues / eigenvalue_sum.clamp_min(self.eps)
        entropy = -(
            probabilities * probabilities.clamp_min(self.eps).log()
        ).sum(dim=-1)
        effective_rank = torch.where(
            eigenvalue_sum.squeeze(-1) > self.eps,
            entropy.exp(),
            torch.zeros_like(entropy),
        ).mean()
        return {
            "std": std.mean(),
            "std_min": std.min(),
            "std_median": std.median(),
            "effective_rank": effective_rank,
        }

    def _code_statistics(
        self,
        target_code: torch.Tensor,
        predicted_code: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Remove each (horizon, transition-mode, rank) mean across the batch
        # before variance/covariance calculations.  A learned mode embedding
        # therefore cannot satisfy anti-collapse by emitting a fixed bias.
        code = target_code.float()
        batch, horizons, modes, rank = code.shape
        centered = code - code.mean(dim=0, keepdim=True)
        flattened = centered.permute(1, 0, 2, 3).reshape(
            horizons, batch * modes, rank
        )
        variance = flattened.square().mean(dim=1)
        std = (variance + self.eps).sqrt()
        variance_loss = F.relu(self.min_code_std - std).mean()

        normalized = flattened / std.detach().unsqueeze(1).clamp_min(self.eps)
        denominator = max(flattened.shape[1] - 1, 1)
        correlation = torch.matmul(normalized.transpose(-1, -2), normalized)
        correlation = correlation / denominator
        identity_mask = torch.eye(rank, device=code.device, dtype=torch.bool)
        off_diagonal = correlation.masked_fill(identity_mask.unsqueeze(0), 0.0)
        covariance_loss = off_diagonal.square().sum(dim=(-2, -1)).div(rank).mean()

        target_diagnostics = self._distribution_diagnostics(target_code)
        predicted_diagnostics = self._distribution_diagnostics(predicted_code)
        return {
            "innovation_variance_loss": variance_loss,
            "innovation_covariance_loss": covariance_loss,
            "innovation_effective_rank": target_diagnostics["effective_rank"],
            "innovation_target_std": target_diagnostics["std"],
            "innovation_target_std_min": target_diagnostics["std_min"],
            "innovation_target_std_median": target_diagnostics["std_median"],
            "innovation_target_code_std": target_diagnostics["std"],
            "innovation_pred_code_std": predicted_diagnostics["std"],
            "innovation_predicted_std": predicted_diagnostics["std"],
            "innovation_predicted_std_min": predicted_diagnostics["std_min"],
            "innovation_predicted_std_median": predicted_diagnostics["std_median"],
            "innovation_predicted_effective_rank": predicted_diagnostics[
                "effective_rank"
            ],
        }

    def _orthogonality_statistics(
        self,
        channel_basis: torch.Tensor,
        spatial_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        errors = []
        for basis in (channel_basis.float(), spatial_basis.float()):
            gram = basis @ basis.transpose(-1, -2)
            identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
            errors.append(gram - identity)
        loss = torch.stack([error.square().mean() for error in errors]).mean()
        maximum = torch.stack([error.abs().max() for error in errors]).max().detach()
        return {
            "innovation_orthogonality_loss": loss,
            "innovation_orthogonality_error": maximum,
            "innovation_channel_orthogonality_error": errors[0].abs().max().detach(),
            "innovation_spatial_orthogonality_error": errors[1].abs().max().detach(),
        }

    def _predict(
        self,
        context: torch.Tensor,
        base_prediction: torch.Tensor,
        goal: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        base_prediction = base_prediction.detach()
        predicted_code = self.predictor(context, base_prediction, goal=goal)
        channel_basis = self.basis()
        spatial_basis = self.spatial_basis()
        decoded_local = self._decode(
            predicted_code.float(), spatial_basis, channel_basis
        )
        decoded_cumulative = self._to_cumulative(decoded_local)
        final_prediction = (
            base_prediction.float() + decoded_cumulative
        ).to(base_prediction.dtype)
        return {
            "final_prediction": final_prediction,
            "base_prediction": base_prediction,
            "predicted_code": predicted_code,
            "decoded_local_innovation": decoded_local,
            "decoded_innovation": decoded_cumulative,
            "channel_basis": channel_basis,
            "spatial_basis": spatial_basis,
        }

    def inference(
        self,
        context: torch.Tensor,
        base_prediction: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return final cumulative deltas without accepting future targets."""

        return self._predict(context, base_prediction, goal)["final_prediction"]

    def forward(
        self,
        context: torch.Tensor,
        base_prediction: torch.Tensor,
        target_delta: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        # Complete the deployed causal path before reading any future target.
        output = self._predict(context, base_prediction, goal)
        channel_basis = output.pop("channel_basis")
        spatial_basis = output.pop("spatial_basis")
        output.update(self._orthogonality_statistics(channel_basis, spatial_basis))
        if target_delta is None:
            return output

        base_prediction = output["base_prediction"]
        self._validate_target(target_delta, base_prediction)
        target = target_delta.detach().float()
        base_fp32 = base_prediction.float()
        cumulative_error = target - base_fp32
        local_error = self._to_local(cumulative_error)
        dynamic_local_error = self._center_local_error(local_error)
        dynamic_cumulative_error = self._to_cumulative(dynamic_local_error)
        fixed_mean_cumulative = self._to_cumulative(self.fixed_local_error_mean)

        target_code = self._project(
            dynamic_local_error, spatial_basis, channel_basis
        )
        oracle_local = self._decode(target_code, spatial_basis, channel_basis)
        oracle_cumulative = self._to_cumulative(oracle_local)
        oracle_prediction = base_fp32 + oracle_cumulative

        predicted_code = output["predicted_code"].float()
        code_error = predicted_code - target_code
        target_energy = target_code.square().mean(dim=(0, 2, 3))
        code_nmse_per_horizon = code_error.square().mean(dim=(0, 2, 3))
        code_loss = (
            code_nmse_per_horizon
            / target_energy.detach().clamp_min(self.eps)
        ).mean()
        with torch.no_grad():
            code_cosine = F.cosine_similarity(
                predicted_code,
                target_code,
                dim=-1,
                eps=self.eps,
            ).mean()

        base_mse = cumulative_error.square().mean()
        mean_only_mse = (
            cumulative_error - fixed_mean_cumulative
        ).square().mean()
        dynamic_base_mse = dynamic_cumulative_error.square().mean()
        dynamic_capture_residual = dynamic_cumulative_error - oracle_cumulative
        dynamic_oracle_mse = dynamic_capture_residual.square().mean()
        capture_loss = (
            dynamic_oracle_mse
            / dynamic_base_mse.detach().clamp_min(self.eps)
        )
        raw_oracle_residual = cumulative_error - oracle_cumulative
        oracle_mse = raw_oracle_residual.square().mean()
        final_mse = (output["final_prediction"].float() - target).square().mean()
        oracle_headroom = base_mse - oracle_mse
        improvement = base_mse - final_mse

        output.update(
            {
                "oracle_prediction": oracle_prediction,
                "target_code": target_code,
                "oracle_local_innovation": oracle_local,
                "oracle_innovation": oracle_cumulative,
                "innovation_code_loss": code_loss,
                "innovation_code_nmse": code_loss.detach(),
                "innovation_code_cosine": code_cosine,
                "innovation_code_cosine_loss": (1.0 - code_cosine).detach(),
                "innovation_capture_loss": capture_loss,
                "innovation_final_mse": final_mse,
                "innovation_base_mse": base_mse,
                "innovation_oracle_mse": oracle_mse,
                "innovation_dynamic_base_mse": dynamic_base_mse.detach(),
                "innovation_dynamic_oracle_mse": dynamic_oracle_mse.detach(),
                "innovation_mean_only_mse": mean_only_mse.detach(),
                "innovation_mean_only_improvement": (
                    base_mse - mean_only_mse
                ).detach(),
                "innovation_final_raw_loss": final_mse,
                "innovation_base_raw_loss": base_mse,
                "innovation_oracle_raw_loss": oracle_mse,
                "innovation_improvement": improvement.detach(),
                "innovation_oracle_headroom": oracle_headroom.detach(),
                "innovation_realized_headroom_fraction": (
                    improvement / oracle_headroom.clamp_min(self.eps)
                ).detach(),
                "innovation_explained_fraction": (1.0 - capture_loss).detach(),
                "innovation_static_error_rms": self.fixed_local_error_mean.square()
                .mean()
                .sqrt()
                .detach(),
                "innovation_dynamic_error_rms": dynamic_local_error.square()
                .mean()
                .sqrt()
                .detach(),
                "innovation_fixed_mean_count": self.fixed_local_error_count.detach(),
                "innovation_code_dimensions": torch.as_tensor(
                    self.n_future * self.transition_tokens * self.rank,
                    device=target.device,
                    dtype=torch.float32,
                ),
                "innovation_transition_tokens": torch.as_tensor(
                    self.transition_tokens,
                    device=target.device,
                    dtype=torch.float32,
                ),
            }
        )
        output.update(self._code_statistics(target_code, predicted_code))

        final_per_horizon = (
            output["final_prediction"].float() - target
        ).square().mean(dim=(0, 2, 3))
        base_per_horizon = cumulative_error.square().mean(dim=(0, 2, 3))
        oracle_per_horizon = raw_oracle_residual.square().mean(dim=(0, 2, 3))
        for index in range(self.n_future):
            suffix = index + 1
            output[f"innovation_final_mse_horizon_{suffix}"] = final_per_horizon[index]
            output[f"innovation_base_mse_horizon_{suffix}"] = base_per_horizon[index]
            output[f"innovation_oracle_mse_horizon_{suffix}"] = oracle_per_horizon[index]

        return output


__all__ = ["PredictableInnovationBottleneck"]
