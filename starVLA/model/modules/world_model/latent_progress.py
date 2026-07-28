"""Latent-space task progress estimation for visual-action policies."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentGoalPredictor(nn.Module):
    """Predict deployable terminal visual tokens from start tokens and a task."""

    def __init__(
        self,
        *,
        latent_dim: int,
        task_dim: int,
        hidden_dim: int,
        num_tokens: int,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_tokens = int(num_tokens)
        self.start_norm = nn.LayerNorm(self.latent_dim)
        self.start_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.task_proj = nn.Linear(task_dim, hidden_dim)
        self.token_embedding = nn.Parameter(
            torch.randn(self.num_tokens, hidden_dim) * 0.02
        )
        self.predictor = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

    def forward(
        self, start_latent: torch.Tensor, task_embedding: torch.Tensor
    ) -> torch.Tensor:
        if start_latent.ndim != 3:
            raise ValueError(
                "start_latent must have shape (B, K, C), "
                f"got {tuple(start_latent.shape)}"
            )
        batch_size, num_tokens, latent_dim = start_latent.shape
        if num_tokens != self.num_tokens or latent_dim != self.latent_dim:
            raise ValueError(
                f"expected start latent (B, {self.num_tokens}, {self.latent_dim}), "
                f"got {tuple(start_latent.shape)}"
            )
        if task_embedding.ndim != 2 or task_embedding.shape[0] != batch_size:
            raise ValueError(
                "task_embedding must have shape (B, D), "
                f"got {tuple(task_embedding.shape)}"
            )

        hidden = self.start_proj(self.start_norm(start_latent))
        hidden = hidden + self.task_proj(task_embedding).unsqueeze(1)
        hidden = hidden + self.token_embedding.to(hidden.dtype).unsqueeze(0)
        return self.predictor(hidden)


class LatentProgressChecker(nn.Module):
    """Compare start, current, and goal tokens and return scalar progress."""

    def __init__(self, *, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
        )
        # Per-token relations retain the fixed spatial token correspondence.
        self.relation_encoder = nn.Sequential(
            nn.Linear(5 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        # Four explicit geometric features make the straight-line latent
        # projection available as a prior while the MLP learns curved progress.
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim + 4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    @staticmethod
    def _validate_triplet(
        start_latent: torch.Tensor,
        current_latent: torch.Tensor,
        goal_latent: torch.Tensor,
    ) -> None:
        if (
            start_latent.ndim != 3
            or current_latent.shape != start_latent.shape
            or goal_latent.shape != start_latent.shape
        ):
            raise ValueError(
                "start/current/goal latents must share shape (B, K, C), got "
                f"{tuple(start_latent.shape)}, {tuple(current_latent.shape)}, "
                f"{tuple(goal_latent.shape)}"
            )

    def forward(
        self,
        start_latent: torch.Tensor,
        current_latent: torch.Tensor,
        goal_latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_triplet(start_latent, current_latent, goal_latent)
        if start_latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"expected latent dim {self.latent_dim}, got {start_latent.shape[-1]}"
            )

        start = self.shared_projection(start_latent)
        current = self.shared_projection(current_latent)
        goal = self.shared_projection(goal_latent)
        start_to_current = current - start
        current_to_goal = goal - current
        start_to_goal = goal - start

        relations = torch.cat(
            [
                start_to_current,
                current_to_goal,
                start_to_goal,
                current * start,
                current * goal,
            ],
            dim=-1,
        )
        relation_summary = self.relation_encoder(relations).mean(dim=1)

        # Geometry is an explicit, non-learned prior. Compute it in fp32 and
        # stop its gradients: when a freshly initialized goal predictor emits a
        # goal close to the start, differentiating through ||goal-start||^-2
        # can amplify an otherwise ordinary progress loss into NaN gradients.
        flat_sc = start_to_current.flatten(1).float()
        flat_sg = start_to_goal.flatten(1).float()
        denominator = flat_sg.square().sum(dim=-1).clamp_min(1e-4)
        geometric_progress = (flat_sc * flat_sg).sum(dim=-1) / denominator
        start_distance = (flat_sc.square().mean(dim=-1) + 1e-6).sqrt()
        goal_distance = (
            current_to_goal.flatten(1).float().square().mean(dim=-1) + 1e-6
        ).sqrt()
        distance_ratio = start_distance / (
            start_distance + goal_distance
        ).clamp_min(1e-6)
        goal_cosine = F.cosine_similarity(
            current.flatten(1), goal.flatten(1), dim=-1, eps=1e-6
        )
        start_cosine = F.cosine_similarity(
            current.flatten(1), start.flatten(1), dim=-1, eps=1e-6
        )
        geometric_features = torch.stack(
            [
                geometric_progress.clamp(-1.0, 2.0),
                distance_ratio.clamp(0.0, 1.0),
                goal_cosine.float().clamp(-1.0, 1.0),
                start_cosine.float().clamp(-1.0, 1.0),
            ],
            dim=-1,
        ).detach().to(relation_summary.dtype)

        progress_logit = self.output(
            torch.cat([relation_summary, geometric_features], dim=-1)
        ).squeeze(-1)
        return {
            "progress": progress_logit.sigmoid(),
            "progress_logit": progress_logit,
            "geometric_progress": geometric_progress.clamp(0.0, 1.0),
        }


class ProgressActionConditioner(nn.Module):
    """Turn explicit scalar progress into an action-query residual."""

    def __init__(
        self,
        *,
        action_hidden_dim: int,
        chunk_len: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.action_hidden_dim = int(action_hidden_dim)
        self.chunk_len = int(chunk_len)
        self.conditioner = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.chunk_len * self.action_hidden_dim),
        )
        # Loading a policy that predates progress conditioning starts from the
        # exact same action queries; gradients can open this branch immediately.
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def forward(
        self, action_queries: torch.Tensor, progress: torch.Tensor
    ) -> torch.Tensor:
        if action_queries.ndim != 3:
            raise ValueError(
                f"action_queries must have shape (B, Q, H), got {tuple(action_queries.shape)}"
            )
        batch_size, chunk_len, hidden_dim = action_queries.shape
        if chunk_len != self.chunk_len or hidden_dim != self.action_hidden_dim:
            raise ValueError(
                f"expected action queries (B, {self.chunk_len}, "
                f"{self.action_hidden_dim}), got {tuple(action_queries.shape)}"
            )
        if progress.shape != (batch_size,):
            raise ValueError(
                f"progress must have shape {(batch_size,)}, got {tuple(progress.shape)}"
            )
        progress = progress.to(device=action_queries.device, dtype=action_queries.dtype)
        features = torch.stack(
            [progress, progress.square(), 1.0 - progress], dim=-1
        )
        residual = self.conditioner(features).view(
            batch_size, self.chunk_len, self.action_hidden_dim
        )
        return action_queries + residual


def latent_goal_loss(
    predicted_goal: torch.Tensor, target_goal: torch.Tensor
) -> torch.Tensor:
    """Scale-robust terminal-latent regression loss."""
    if predicted_goal.shape != target_goal.shape:
        raise ValueError(
            f"goal shapes must match, got {tuple(predicted_goal.shape)} and "
            f"{tuple(target_goal.shape)}"
        )
    predicted = F.layer_norm(predicted_goal.float(), predicted_goal.shape[-1:])
    target = F.layer_norm(target_goal.detach().float(), target_goal.shape[-1:])
    l1 = F.smooth_l1_loss(predicted, target)
    cosine = 1.0 - F.cosine_similarity(
        predicted.flatten(1), target.flatten(1), dim=-1, eps=1e-6
    ).mean()
    return l1 + 0.1 * cosine


def progress_ranking_loss(
    progress: torch.Tensor,
    target: torch.Tensor,
    episode_ids: Sequence[object],
    *,
    margin: float = 0.02,
) -> torch.Tensor:
    """Pairwise monotonic ranking loss for samples from the same episode."""
    if progress.ndim != 1 or target.shape != progress.shape:
        raise ValueError("progress and target must have matching shape (B,)")
    if len(episode_ids) != progress.shape[0]:
        raise ValueError("episode_ids length must equal the batch size")

    terms = []
    for i in range(progress.shape[0]):
        for j in range(progress.shape[0]):
            if episode_ids[i] != episode_ids[j] or target[j] <= target[i]:
                continue
            terms.append(F.relu(progress[i] - progress[j] + margin))
    if not terms:
        return progress.new_zeros(())
    return torch.stack(terms).mean()
