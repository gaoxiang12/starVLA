"""Differentiable TCP poses and bounded local IK for explicit Cartesian actions.

Geometry is expressed in the URDF root frame. The optional world transform must
be applied by the caller to both positions and orientations, never just XYZ.
"""
import math
import xml.etree.ElementTree as ET

import torch

from starVLA.model.modules.robotwin_kinematics import SerialTCPKinematics


def skew(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


def rotation_exp(vector):
    """Axis-angle exponential, including a finite gradient at zero rotation."""
    angle = torch.linalg.vector_norm(vector, dim=-1)
    cross = skew(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return (identity + torch.sinc(angle/math.pi)[..., None, None]*cross
            + (.5*torch.sinc(angle/(2*math.pi)).square())[..., None, None]*(cross @ cross))


def rotation_log(matrix):
    """Principal axis angle; select the stable quaternion component near pi."""
    r00, r01, r02 = matrix[..., 0, :].unbind(-1)
    r10, r11, r12 = matrix[..., 1, :].unbind(-1)
    r20, r21, r22 = matrix[..., 2, :].unbind(-1)
    squared = torch.stack((1+r00+r11+r22, 1+r00-r11-r22,
                           1-r00+r11-r22, 1-r00-r11+r22), -1)
    magnitude = squared.clamp_min(1e-12).sqrt()
    candidates = torch.stack((
        torch.stack((squared[..., 0], r21-r12, r02-r20, r10-r01), -1),
        torch.stack((r21-r12, squared[..., 1], r10+r01, r02+r20), -1),
        torch.stack((r02-r20, r10+r01, squared[..., 2], r12+r21), -1),
        torch.stack((r10-r01, r20+r02, r21+r12, squared[..., 3]), -1)), -2)
    candidates = candidates/(2*magnitude.clamp_min(.1)[..., :, None])
    index = magnitude.argmax(-1)
    quaternion = candidates.gather(-2, index[..., None, None].expand(*index.shape, 1, 4)).squeeze(-2)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    quaternion = quaternion/torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)
    scalar, vector = quaternion[..., :1], quaternion[..., 1:]
    sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    scale = 2*torch.atan2(sine, scalar)/sine.clamp_min(1e-12)
    scale = torch.where(sine < 1e-6, 2/scalar.clamp_min(1e-12), scale)
    return scale*vector


class SerialPoseKinematics(SerialTCPKinematics):
    """TCP transform and geometric spatial Jacobian in declared joint order."""
    def __init__(self, urdf_path, target_link, joint_names, tool_offset=(.12, 0., 0.),
                 tool_rotation=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))):
        super().__init__(urdf_path, target_link, joint_names, tool_offset=tool_offset)
        # FK here ends at the URDF CHILD LINK, not SAPIEN joint6.global_pose.
        # The latter requires diag(1,-1,-1) in Robot._trans_endpose; applying
        # that correction again to link FK creates an erroneous 180-degree roll.
        self.register_buffer('tool_rotation', torch.tensor(tool_rotation, dtype=torch.float64))
        joints = {j.get('name'):j for j in ET.parse(urdf_path).getroot().findall('joint')}
        bounds = []
        for name in joint_names:
            joint = joints[name]
            limit = joint.find('limit')
            if joint.get('type') == 'continuous':
                bounds.append((-float('inf'), float('inf')))
            elif limit is None:
                raise ValueError(f'Missing joint limits: {name}')
            else:
                bounds.append((float(limit.get('lower')), float(limit.get('upper'))))
        self.register_buffer('joint_limits', torch.tensor(bounds, dtype=torch.float64))

    def pose_and_jacobian(self, joints):
        if joints.shape[-1] != len(self.joint_names) or joints.dtype not in (torch.float32, torch.float64):
            raise ValueError('Pose geometry requires the correct joint dimension and FP32/FP64 inputs')
        transform = torch.eye(4, device=joints.device, dtype=joints.dtype).expand(*joints.shape[:-1], 4, 4)
        origins, axes, kinds = {}, {}, {}
        for index, kind, origin, axis in zip(self.indices, self.kinds, self.origins, self.axes):
            transform = transform @ origin.to(joints)
            if index < 0:
                continue
            axis = axis.to(joints)
            origins[index] = transform[..., :3, 3]
            axes[index] = (transform[..., :3, :3] @ axis[..., None]).squeeze(-1)
            kinds[index] = kind
            motion = torch.eye(4, device=joints.device, dtype=joints.dtype).expand_as(transform).clone()
            coordinate = joints[..., index]
            if kind == 'prismatic':
                motion[..., :3, 3] = coordinate[..., None]*axis
            else:
                motion[..., :3, :3] = rotation_exp(coordinate[..., None]*axis)
            transform = transform @ motion
        tool = torch.eye(4, device=joints.device, dtype=joints.dtype)
        tool[:3, :3] = self.tool_rotation.to(joints)
        tool[:3, 3] = self.tool_offset.to(joints)
        transform = transform @ tool
        columns = []
        for index in range(len(self.joint_names)):
            if kinds[index] == 'prismatic':
                linear, angular = axes[index], torch.zeros_like(axes[index])
            else:
                angular = axes[index]
                linear = torch.linalg.cross(angular, transform[..., :3, 3]-origins[index], dim=-1)
            columns.append(torch.cat((linear, angular), -1))
        return transform, torch.stack(columns, -1)

    def pose(self, joints):
        return self.pose_and_jacobian(joints)[0]

    def solve_local(self, seed, target_position, target_rotation, *, iterations=12,
                    damping=.005, orientation_weight=.1, max_joint_step=.2,
                    position_tolerance=.002, rotation_tolerance=.02):
        """Bounded differentiable DLS; report residuals instead of claiming reachability.

        This is geometric local IK, not a collision planner or dynamic controller.
        The caller must handle nonconvergence explicitly before executing actions.
        """
        if target_position.shape != (*seed.shape[:-1], 3) or target_rotation.shape != (*seed.shape[:-1], 3, 3):
            raise ValueError('IK batch shapes do not match')
        if iterations < 1 or min(damping, orientation_weight, max_joint_step,
                                 position_tolerance, rotation_tolerance) <= 0:
            raise ValueError('Invalid IK settings')
        if not all(torch.isfinite(x).all() for x in (seed, target_position, target_rotation)):
            raise ValueError('IK inputs must be finite')
        bounds = self.joint_limits.to(seed)
        q = seed.clamp(bounds[:, 0], bounds[:, 1])
        weights = seed.new_tensor([1, 1, 1, orientation_weight, orientation_weight, orientation_weight])
        identity = torch.eye(6, dtype=seed.dtype, device=seed.device)
        for _ in range(iterations):
            actual, jacobian = self.pose_and_jacobian(q)
            error = torch.cat((target_position-actual[..., :3, 3],
                rotation_log(target_rotation @ actual[..., :3, :3].transpose(-1, -2))), -1)*weights
            jacobian = jacobian*weights[:, None]
            step = (jacobian.transpose(-1, -2) @ torch.linalg.solve(
                jacobian @ jacobian.transpose(-1, -2)+damping**2*identity, error[..., None])).squeeze(-1)
            step = step/(step.abs().amax(-1, keepdim=True)/max_joint_step).clamp_min(1)
            q = (q+step).clamp(bounds[:, 0], bounds[:, 1])
        actual = self.pose(q)
        position_error = torch.linalg.vector_norm(target_position-actual[..., :3, 3], dim=-1)
        rotation_error = torch.linalg.vector_norm(rotation_log(
            target_rotation @ actual[..., :3, :3].transpose(-1, -2)), dim=-1)
        return dict(joints=q, position_error_m=position_error, rotation_error_rad=rotation_error,
                    converged=(position_error <= position_tolerance) & (rotation_error <= rotation_tolerance))
