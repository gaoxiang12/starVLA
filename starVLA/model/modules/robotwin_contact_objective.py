"""Geometric and closing-transition supervision for continuous RoboTwin joints."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer
from starVLA.model.modules.robotwin_kinematics import SerialTCPKinematics


class RoboTwinContactObjective(nn.Module):
    def __init__(self, urdf_path, statistics_path, *, position_weight=.1,
                 gripper_weight=.02, contact_boost=4., transition_radius=3):
        super().__init__()
        stats = json.loads(Path(statistics_path).read_text())['aloha']['action']
        self._normalizer = Normalizer('q99', {key: stats[key][:12] for key in ('q01', 'q99')}, q99_clip=None)
        if any(len(stats[key]) != 14 for key in ('q01', 'q99')):
            raise ValueError('Contact objective requires 14D continuous Aloha actions')
        # Deliberately not registered as model buffers: DeepSpeed may cast the
        # trainable model to BF16, but URDF geometry and normalization constants
        # must retain FP32 precision. These frozen helpers have no parameters.
        self._chains = [SerialTCPKinematics(urdf_path, f'{side}_link6',
                                           [f'{side}_joint{i}' for i in range(1, 7)])
                        for side in ('fl', 'fr')]
        self.position_weight = float(position_weight)
        self.gripper_weight = float(gripper_weight)
        self.contact_boost = float(contact_boost)
        self.transition_radius = int(transition_radius)
        if min(self.position_weight, self.gripper_weight, self.contact_boost, self.transition_radius) < 0:
            raise ValueError('Contact objective weights/radius must be nonnegative')
        self._geometry_device = None

    def _prepare(self, device):
        if self._geometry_device != device:
            for chain in self._chains:
                chain.to(device=device, dtype=torch.float32)
            self._normalizer.statistics = {key: value.to(device=device, dtype=torch.float32)
                                           for key, value in self._normalizer.statistics.items()}
            self._geometry_device = device

    def forward(self, prediction, target, state, valid_mask=None):
        if prediction.shape != target.shape or prediction.shape[-1] != 14 or state.shape != (prediction.shape[0], 14):
            raise ValueError('Contact objective expects [B,H,14] actions and [B,14] state')
        self._prepare(prediction.device)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            valid = (torch.ones(prediction.shape[:2], device=prediction.device, dtype=torch.bool)
                     if valid_mask is None else valid_mask.to(device=prediction.device, dtype=torch.bool))
            if valid.shape != prediction.shape[:2]:
                raise ValueError('Invalid contact action mask shape')
            # Padded values must be removed before trigonometry, not multiplied
            # by zero afterwards (0 * NaN would still poison the loss).
            pred = torch.where(valid[..., None], prediction.float(), 0.)
            truth = torch.where(valid[..., None], target.float().detach(), 0.)
            initial_grip = state.float().detach()[:, None, 12:14]
            previous = torch.cat([initial_grip, truth[:, :-1, 12:14]], dim=1)
            previous_valid = torch.cat([torch.ones_like(valid[:, :1]), valid[:, :-1]], dim=1)
            closing = ((truth[..., 12:14]-previous) < -.001) & valid[..., None] & previous_valid[..., None]
            contact = F.max_pool1d(closing.transpose(1, 2).float(),
                                  kernel_size=2*self.transition_radius+1, stride=1,
                                  padding=self.transition_radius).transpose(1, 2).bool()
            contact &= valid[..., None]
            joints_pred = self._normalizer.inverse(pred[..., :12])
            joints_truth = self._normalizer.inverse(truth[..., :12])
            distances = torch.stack([torch.linalg.vector_norm(
                chain(joints_pred[..., arm*6:(arm+1)*6])-chain(joints_truth[..., arm*6:(arm+1)*6]), dim=-1)
                for arm, chain in enumerate(self._chains)], dim=-1)
            weights = valid[..., None].float() * (1+self.contact_boost*contact.float())
            position_loss = (distances*weights).sum()/weights.sum().clamp_min(1)
            grip_error = (pred[..., 12:14]-truth[..., 12:14]).abs()
            gripper_loss = (grip_error*contact).sum()/contact.sum().clamp_min(1)
            loss = self.position_weight*position_loss + self.gripper_weight*gripper_loss
            metrics = dict(tcp_position_loss_m=position_loss.detach(),
                           tcp_contact_error_mm=(1000*(distances*contact).sum()/contact.sum().clamp_min(1)).detach(),
                           gripper_transition_l1=gripper_loss.detach(),
                           contact_fraction=(contact.sum()/(2*valid.sum()).clamp_min(1)).detach(),
                           contact_objective_loss=loss.detach())
        return loss, metrics
