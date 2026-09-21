"""Differentiable serial-chain TCP positions from the evaluator's URDF."""
import math
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn


def origin_matrix(element):
    transform = np.eye(4)
    if element is None:
        return transform
    transform[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    roll, pitch, yaw = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    cx, cy, cz = math.cos(roll), math.cos(pitch), math.cos(yaw)
    sx, sy, sz = math.sin(roll), math.sin(pitch), math.sin(yaw)
    transform[:3, :3] = [[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                          [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                          [-sy, cy*sx, cy*cx]]
    return transform


class SerialTCPKinematics(nn.Module):
    def __init__(self, urdf_path, target_link, joint_names, tool_offset=(.12, 0., 0.)):
        super().__init__()
        root = ET.parse(urdf_path).getroot()
        parents = {joint.find('child').get('link'): joint for joint in root.findall('joint')}
        chain, seen = [], set()
        link = target_link
        if link not in {item.get('name') for item in root.findall('link')}:
            raise ValueError(f'Unknown link: {link}')
        while link in parents:
            if link in seen:
                raise ValueError('Cycle in URDF')
            seen.add(link)
            joint = parents[link]
            chain.append(joint)
            link = joint.find('parent').get('link')
        chain.reverse()
        active = [joint.get('name') for joint in chain if joint.get('type') != 'fixed']
        if set(active) != set(joint_names) or len(active) != len(joint_names):
            raise ValueError(f'Joint specification does not match chain: {active}')
        self.joint_names = tuple(joint_names)
        self.indices, self.kinds = [], []
        origins, axes = [], []
        for joint in chain:
            kind = joint.get('type')
            if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
                raise ValueError(f'Unsupported joint type: {kind}')
            if joint.find('mimic') is not None:
                raise ValueError('Mimic joints require an explicit mapping')
            self.kinds.append(kind)
            self.indices.append(-1 if kind == 'fixed' else self.joint_names.index(joint.get('name')))
            origins.append(origin_matrix(joint.find('origin')))
            axis_element = joint.find('axis')
            axis = np.fromstring(axis_element.get('xyz', '1 0 0') if axis_element is not None else '1 0 0', sep=' ')
            if kind != 'fixed' and np.linalg.norm(axis) == 0:
                raise ValueError('Zero rotation/translation axis')
            axes.append(axis / max(np.linalg.norm(axis), 1e-12))
        self.register_buffer('origins', torch.tensor(np.asarray(origins), dtype=torch.float64))
        self.register_buffer('axes', torch.tensor(np.asarray(axes), dtype=torch.float64))
        self.register_buffer('tool_offset', torch.tensor(tool_offset, dtype=torch.float64))

    def forward(self, joints):
        if joints.shape[-1] != len(self.joint_names):
            raise ValueError('Incorrect number of joint coordinates')
        transform = torch.eye(4, device=joints.device, dtype=joints.dtype).expand(*joints.shape[:-1], 4, 4)
        for index, kind, origin, axis in zip(self.indices, self.kinds, self.origins, self.axes):
            transform = transform @ origin.to(joints)
            if index < 0:
                continue
            axis = axis.to(joints)
            motion = torch.eye(4, device=joints.device, dtype=joints.dtype).expand_as(transform).clone()
            angle = joints[..., index]
            if kind == 'prismatic':
                motion[..., :3, 3] = angle[..., None] * axis
            else:
                x, y, z = axis.unbind()
                zero = torch.zeros_like(x)
                skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
                motion[..., :3, :3] = (torch.eye(3, device=joints.device, dtype=joints.dtype)
                    + angle.sin()[..., None, None]*skew
                    + (1-angle.cos())[..., None, None]*(skew @ skew))
            transform = transform @ motion
        return transform[..., :3, 3] + (transform[..., :3, :3] @ self.tool_offset.to(joints)[..., None]).squeeze(-1)
