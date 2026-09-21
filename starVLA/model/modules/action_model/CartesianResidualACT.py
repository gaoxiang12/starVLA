"""ACT with an explicit Cartesian pose correction on its executed action path."""
from contextvars import ContextVar
import json
from pathlib import Path

import torch

from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer
from starVLA.model.modules.action_model.ACT_ActionHeader import TurboStyleACTActionHead, ActionProjectionMLP
from starVLA.model.modules.robotwin_pose_kinematics import SerialPoseKinematics, rotation_exp


cartesian_context = ContextVar('gawm_cartesian_action_call', default=None)


class CartesianResidualACT(TurboStyleACTActionHead):
    """Keep a pretrained joint proposal; learn pose corrections and project by IK.

    The joint projection weights stay frozen, but its input gradient is retained:
    the shared decoder features remain trainable. Learned pose/gripper corrections
    directly change returned actions. Geometry remains full precision.
    """
    def __init__(self, *, geometry, statistics_path, **kwargs):
        super().__init__(**kwargs)
        if self.action_dim != 14 or self.state_dim != 14 or self.gripper_indices != (12, 13):
            raise ValueError('Cartesian residual decoder currently requires Aloha model-order 14D actions')
        stats = json.loads(Path(statistics_path).read_text())['aloha']['action']
        self._normalizer = Normalizer('q99', {k:stats[k][:12] for k in ('q01', 'q99')}, q99_clip=None)
        # Frozen geometry is not registered under a BF16-cast DeepSpeed model.
        self._chains = [SerialPoseKinematics(geometry['urdf_path'], f'{prefix}_link6',
            [f'{prefix}_joint{i}' for i in range(1, 7)]) for prefix in ('fl', 'fr')]
        self._geometry_device = None
        self.translation_scale_m = float(geometry.get('translation_scale_m', .08))
        self.rotation_scale_rad = float(geometry.get('rotation_scale_rad', .35))
        self.ik_iterations = int(geometry.get('ik_iterations', 12))
        if not 0 < self.translation_scale_m <= .15 or not 0 < self.rotation_scale_rad <= 1:
            raise ValueError('Invalid local correction extent')
        self.pose_projection = ActionProjectionMLP(self.hidden_dim, kwargs['mlp_hidden_dim'], 14)
        torch.nn.init.zeros_(self.pose_projection.layers[-1].weight)
        torch.nn.init.zeros_(self.pose_projection.layers[-1].bias)
        self.action_projection.requires_grad_(False)

    @classmethod
    def from_existing(cls, original, geometry, statistics_path):
        layer = original.decoder.layers[0]
        result = cls(geometry=geometry, statistics_path=statistics_path,
            token_dim=original.visual_projection.in_features, hidden_dim=original.hidden_dim,
            action_dim=original.action_dim, horizon=original.horizon, num_frames=original.num_frames,
            num_visual_tokens=original.num_visual_tokens, num_heads=layer.self_attn.num_heads,
            num_layers=len(original.decoder.layers), dim_feedforward=layer.linear1.out_features,
            mlp_hidden_dim=original.action_projection.layers[0].out_features, dropout=layer.dropout.p,
            state_dim=original.state_dim, state_hidden_dim=original.state_projection.net[1].out_features,
            num_state_tokens=original.num_state_tokens, output_activation=original.output_activation,
            gripper_indices=original.gripper_indices)
        incompatible = result.load_state_dict(original.state_dict(), strict=False)
        assert not incompatible.unexpected_keys
        assert all(k.startswith('pose_projection.') for k in incompatible.missing_keys)
        return result

    def prepare_geometry(self, device):
        if self._geometry_device != device:
            # predict_action may be the first call and run under inference_mode.
            # Cached geometry must remain ordinary tensors for later backward.
            with torch.inference_mode(False):
                for chain in self._chains:
                    chain.to(device=device, dtype=torch.float32)
                    for name, value in chain.named_buffers():
                        if torch.is_inference(value):
                            setattr(chain, name, value.clone())
                self._normalizer.statistics = {k:v.to(device=device, dtype=torch.float32).clone()
                                               for k,v in self._normalizer.statistics.items()}
            self._geometry_device = device

    def predict_action(self, action_hidden):
        context = cartesian_context.get()
        if context is None or context.get('head') is not self:
            raise RuntimeError('Cartesian action decoder requires an isolated framework call context')
        # Freezing projection weights does not freeze their trainable inputs.
        # Detaching here drops a real output dependency and produces a biased
        # shared-decoder gradient that can rapidly move the joint proposal.
        coarse = super().predict_action(action_hidden)
        correction = self.pose_projection(action_hidden)
        self.prepare_geometry(action_hidden.device)
        with torch.autocast(device_type=action_hidden.device.type, enabled=False):
            coarse, correction = coarse.float(), correction.float()
            seed = self._normalizer.inverse(coarse[..., :12])
            solved, proposals, targets, actual, flags, residuals = [], [], [], [], [], []
            for arm, chain in enumerate(self._chains):
                q = seed[..., arm*6:(arm+1)*6]
                proposal = chain.pose(q)
                proposals.append(proposal)
                delta = correction[..., arm*6:(arm+1)*6]
                target_position = proposal[..., :3, 3]+self.translation_scale_m*delta[..., :3].tanh()
                target_rotation = rotation_exp(self.rotation_scale_rad*delta[..., 3:].tanh()) @ proposal[..., :3, :3]
                result = chain.solve_local(q, target_position, target_rotation, iterations=self.ik_iterations)
                solved.append(result['joints'])
                targets.append((target_position, target_rotation))
                actual.append(chain.pose(result['joints']))
                flags.append(result['converged'])
                residuals.append(result['position_error_m'])
            joints = self._normalizer.forward(torch.cat(solved, -1))
            grippers = (coarse[..., 12:]+2*correction[..., 12:].tanh()).clamp(-1, 1)
            converged = torch.stack(flags, -1)
            if context['inference']:
                # A numerically unresolved pose must not issue a closing command.
                grippers = torch.where(converged, grippers, torch.ones_like(grippers))
            output = torch.cat((joints, grippers), -1)
            context.update(proposal_poses=proposals, target_poses=targets, actual_poses=actual, ik_converged=converged,
                           ik_position_residual_m=torch.stack(residuals, -1), coarse_actions=coarse,
                           corrected_actions=output)
        return output
