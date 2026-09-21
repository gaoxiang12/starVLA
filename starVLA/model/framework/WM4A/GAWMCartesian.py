"""Opt-in GAWM Cartesian correction decoder with direct geometric supervision."""
import numpy as np
import torch

from starVLA.model.framework.WM4A.GAWM import GAWM
from starVLA.model.modules.action_model.CartesianResidualACT import CartesianResidualACT, cartesian_context
from starVLA.model.modules.robotwin_pose_kinematics import rotation_log
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register('GAWMCartesian')
class GAWMCartesian(GAWM):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cartesian_cfg = cfg.framework.cartesian_action
        self.action_models['aloha'] = CartesianResidualACT.from_existing(self.action_models['aloha'],
            self.cartesian_cfg, cfg.datasets.vla_data.normalization_statistics_path)

    def _call_context(self, examples, inference):
        if not examples or any(ex.get('robot_tag') != 'aloha' for ex in examples):
            raise ValueError('GAWMCartesian currently accepts only explicit Aloha examples')
        return dict(owner=self, head=self.action_models['aloha'], inference=inference)

    def forward(self, examples=None, **kwargs):
        context = self._call_context(examples, False)
        token = cartesian_context.set(context)
        try:
            output = super().forward(examples, **kwargs)
            head = self.action_models['aloha']
            prediction = context['corrected_actions']
            target = torch.as_tensor(np.asarray([ex['action'] for ex in examples]),
                                     dtype=torch.float32, device=prediction.device)
            valid = self._action_valid_mask_tensor(examples, prediction.device, prediction.shape[1])
            if valid is None:
                valid = torch.ones(prediction.shape[:2], device=prediction.device, dtype=torch.bool)
            with torch.autocast(device_type=prediction.device.type, enabled=False):
                target = torch.where(valid[..., None], target, 0.)
                joints = head._normalizer.inverse(target[..., :12])
                positions, rotations, coarse_positions, actual_positions = [], [], [], []
                for arm, chain in enumerate(head._chains):
                    truth = chain.pose(joints[..., arm*6:(arm+1)*6])
                    position, rotation = context['target_poses'][arm]
                    positions.append(torch.linalg.vector_norm(position-truth[..., :3, 3], dim=-1))
                    coarse_positions.append(torch.linalg.vector_norm(
                        context['proposal_poses'][arm][..., :3, 3]-truth[..., :3, 3], dim=-1))
                    actual_positions.append(torch.linalg.vector_norm(
                        context['actual_poses'][arm][..., :3, 3]-truth[..., :3, 3], dim=-1))
                    rotations.append(torch.linalg.vector_norm(rotation_log(
                        rotation @ truth[..., :3, :3].transpose(-1, -2)), dim=-1))
                positions, rotations = torch.stack(positions, -1), torch.stack(rotations, -1)
                mask = valid[..., None].expand_as(positions)
                denominator = mask.sum().clamp_min(1)
                position_loss = (positions*mask).sum()/denominator
                rotation_loss = (rotations*mask).sum()/denominator
                residual = (context['ik_position_residual_m']*mask).sum()/denominator
                # Centimeter/radian units make the intended precision explicit.
                geometry_loss = float(self.cartesian_cfg.get('position_weight', .1))*position_loss/.01
                geometry_loss += float(self.cartesian_cfg.get('rotation_weight', .02))*rotation_loss/.1
                geometry_loss += float(self.cartesian_cfg.get('ik_residual_weight', .1))*residual/.01
            output['action_loss'] = output['action_loss']+geometry_loss
            output.update(cartesian_target_error_mm=(1000*position_loss).detach(),
                cartesian_coarse_target_error_mm=(1000*(torch.stack(coarse_positions,-1)*mask).sum()/denominator).detach(),
                cartesian_actual_target_error_mm=(1000*(torch.stack(actual_positions,-1)*mask).sum()/denominator).detach(),
                cartesian_rotation_error_deg=(180/torch.pi*rotation_loss).detach(),
                cartesian_ik_residual_mm=(1000*residual).detach(),
                cartesian_ik_converged_fraction=(context['ik_converged']*mask).sum().detach()/denominator,
                cartesian_geometry_loss=geometry_loss.detach(),
                cartesian_action_correction_l1=(prediction-context['coarse_actions']).abs().mean().detach())
            return output
        finally:
            context.clear()
            cartesian_context.reset(token)

    def predict_action(self, examples, **kwargs):
        context = self._call_context(examples, True)
        token = cartesian_context.set(context)
        try:
            result = super().predict_action(examples, **kwargs)
            result['cartesian_ik_converged'] = context['ik_converged'].detach().cpu().numpy()
            result['cartesian_ik_position_residual_m'] = context['ik_position_residual_m'].detach().cpu().numpy()
            return result
        finally:
            context.clear()
            cartesian_context.reset(token)
