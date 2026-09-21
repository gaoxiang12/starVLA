"""Independently compare SAPIEN joint-frame and URDF link-frame TCP rotations."""
import copy
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import sapien
import torch

from starVLA.model.modules.robotwin_pose_kinematics import SerialPoseKinematics


def main():
    urdf = Path('/data/gaoxiang/Code/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf')
    original = ET.parse(urdf).getroot()
    parents = {j.find('child').get('link'):j for j in original.findall('joint')}
    names, links = set(), set()
    for target in ('fl_link6', 'fr_link6'):
        link = target
        links.add(link)
        while link in parents:
            joint = parents[link]
            names.add(joint.get('name'))
            link = joint.find('parent').get('link')
            links.add(link)
    minimal = ET.Element('robot', name='pose_frame_validation')
    for name in sorted(links):
        link = ET.SubElement(minimal, 'link', name=name)
        inertia = ET.SubElement(link, 'inertial')
        ET.SubElement(inertia, 'mass', value='1')
        ET.SubElement(inertia, 'inertia', ixx='.01', ixy='0', ixz='0', iyy='.01', iyz='0', izz='.01')
    for joint in original.findall('joint'):
        if joint.get('name') in names:
            minimal.append(copy.deepcopy(joint))
    differences, fk_differences = [], []
    with tempfile.TemporaryDirectory(prefix='aloha_pose_frame_') as directory:
        path = Path(directory)/'robot.urdf'
        ET.ElementTree(minimal).write(path)
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(str(path))
        joints = {j.name:j for j in robot.get_active_joints()}
        link_map = {link.name:link for link in robot.get_links()}
        joint_names = list(joints)
        qs = np.random.default_rng(42).uniform(-1, 1, (32, len(joint_names)))
        for q in qs:
            robot.set_qpos(q)
            for side in ('fl', 'fr'):
                joint_matrix = joints[f'{side}_joint6'].global_pose.to_transformation_matrix()
                link_matrix = link_map[f'{side}_link6'].pose.to_transformation_matrix()
                api_rotation = joint_matrix[:3, :3] @ np.diag([1., -1., -1.])
                differences.append(float(np.abs(api_rotation-link_matrix[:3, :3]).max()))
                chain = SerialPoseKinematics(urdf, f'{side}_link6', [f'{side}_joint{i}' for i in range(1,7)])
                ordered = q[[joint_names.index(f'{side}_joint{i}') for i in range(1,7)]]
                actual = chain.pose(torch.tensor(ordered, dtype=torch.float64)).numpy()
                fk_differences.append(float(np.abs(actual[:3, :3]-api_rotation).max()))
    report = dict(samples=32, arms=2, max_joint_api_vs_child_link_rotation_difference=float(max(differences)),
        max_fk_default_tool_rotation_vs_api_difference=float(max(fk_differences)),
        passed=max(differences)<2e-6 and max(fk_differences)<2e-6,
        conclusion='For this SAPIEN Aloha asset, joint6.global_pose rotation times diag(1,-1,-1) '
                   'equals link6.pose rotation. URDF child-link FK therefore uses IDENTITY tool rotation. '
                   'Applying that diagonal again to child-link FK introduces a 180-degree roll error.')
    path = Path(__file__).resolve().parents[3]/'playground/Checkpoints/gawm_cartesian_model_smoke_r3_20260909/independent_pose_frame_validation.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
