"""Action-loss helpers shared by heterogeneous robot embodiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _action_mask(
    valid_mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return a broadcastable float mask for ``[B, H, D]`` actions."""

    if reference.ndim != 3:
        raise ValueError(
            f"reference actions must have shape [B,H,D], got {tuple(reference.shape)}"
        )
    batch_size, horizon, action_dim = reference.shape
    if valid_mask is None:
        return reference.new_ones((batch_size, horizon, 1))
    mask = torch.as_tensor(valid_mask, device=reference.device)
    if mask.shape == (batch_size, horizon):
        mask = mask.unsqueeze(-1)
    if mask.shape not in {
        (batch_size, horizon, 1),
        (batch_size, horizon, action_dim),
    }:
        raise ValueError(
            "action valid mask must have shape [B,H], [B,H,1], or [B,H,D]; "
            f"got {tuple(mask.shape)} for actions {tuple(reference.shape)}"
        )
    return mask.to(dtype=reference.dtype)


def masked_action_l1_loss(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean absolute error over valid action elements only."""

    if pred_actions.shape != target_actions.shape or pred_actions.ndim != 3:
        raise ValueError(
            "pred_actions and target_actions must have matching [B,H,D] shapes, "
            f"got {tuple(pred_actions.shape)} and {tuple(target_actions.shape)}"
        )
    mask = _action_mask(valid_mask, pred_actions)
    denominator = mask.sum()
    if mask.shape[-1] == 1:
        denominator = denominator * pred_actions.shape[-1]
    if not bool(denominator.detach().gt(0)):
        raise ValueError("action valid mask contains no valid target elements")
    return ((pred_actions - target_actions).abs() * mask).sum() / denominator


def action_l1_diagnostics(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    gripper_indices: Sequence[int] = (),
) -> dict[str, torch.Tensor]:
    """Return comparable continuous/gripper/first-step action diagnostics."""

    action_dim = pred_actions.shape[-1]
    gripper_indices = tuple(sorted({int(index) for index in gripper_indices}))
    if any(index < 0 or index >= action_dim for index in gripper_indices):
        raise ValueError(
            f"gripper indices {gripper_indices} are invalid for action_dim={action_dim}"
        )
    continuous_indices = tuple(
        index for index in range(action_dim) if index not in gripper_indices
    )
    mask = _action_mask(valid_mask, pred_actions)
    metrics = {
        "first_action_l1": masked_action_l1_loss(
            pred_actions[:, :1], target_actions[:, :1], mask[:, :1]
        ),
        "valid_action_fraction": mask.mean(),
    }
    if continuous_indices:
        continuous_mask = (
            mask if mask.shape[-1] == 1 else mask[..., continuous_indices]
        )
        metrics["continuous_action_l1"] = masked_action_l1_loss(
            pred_actions[..., continuous_indices],
            target_actions[..., continuous_indices],
            continuous_mask,
        )
    if gripper_indices:
        gripper_mask = (
            mask if mask.shape[-1] == 1 else mask[..., gripper_indices]
        )
        pred_gripper = pred_actions[..., gripper_indices]
        target_gripper = target_actions[..., gripper_indices]
        metrics["gripper_action_l1"] = masked_action_l1_loss(
            pred_gripper, target_gripper, gripper_mask
        )
        correct = ((pred_gripper >= 0.5) == (target_gripper >= 0.5)).to(
            pred_actions.dtype
        )
        denominator = gripper_mask.sum()
        if gripper_mask.shape[-1] == 1:
            denominator = denominator * len(gripper_indices)
        metrics["gripper_action_accuracy"] = (
            correct * gripper_mask
        ).sum() / denominator
    return metrics
