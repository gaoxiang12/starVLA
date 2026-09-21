"""Validate differentiable TCP positions against collisionless SAPIEN CPU FK."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import sapien
import torch

from starVLA.model.modules.robotwin_kinematics import SerialTCPKinematics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original = ET.parse(args.urdf).getroot()
    parents = {j.find('child').get('link'): j for j in original.findall('joint')}
    names, links = set(), set()
    for target in ('fl_link6', 'fr_link6'):
        link = target
        links.add(link)
        while link in parents:
            joint = parents[link]
            names.add(joint.get('name'))
            link = joint.find('parent').get('link')
            links.add(link)
    minimal = ET.Element('robot', name='aloha_fk_validation')
    for name in sorted(links):
        link = ET.SubElement(minimal, 'link', name=name)
        inertia = ET.SubElement(link, 'inertial')
        ET.SubElement(inertia, 'mass', value='1')
        ET.SubElement(inertia, 'inertia', ixx='.01', ixy='0', ixz='0', iyy='.01', iyz='0', izz='.01')
    for joint in original.findall('joint'):
        if joint.get('name') in names:
            minimal.append(copy.deepcopy(joint))
    with tempfile.TemporaryDirectory(prefix='robotwin_fk_') as directory:
        path = Path(directory)/'minimal.urdf'
        ET.ElementTree(minimal).write(path)
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(str(path))
        joint_names = [j.name for j in robot.get_active_joints()]
        link_map = {link.name: link for link in robot.get_links()}
        configurations = np.random.default_rng(42).uniform(-1, 1, (32, len(joint_names)))
        configurations[0] = 0
        reference = []
        for q in configurations:
            robot.set_qpos(q)
            row = []
            for side in ('fl', 'fr'):
                pose = link_map[f'{side}_link6'].pose.to_transformation_matrix()
                row.append((pose[:3, 3]+pose[:3, :3]@np.array([.12, 0., 0.])).tolist())
            reference.append(row)
    predictions = []
    for side in ('fl', 'fr'):
        ordered = [f'{side}_joint{i}' for i in range(1, 7)]
        chain = SerialTCPKinematics(args.urdf, f'{side}_link6', ordered)
        q = configurations[:, [joint_names.index(name) for name in ordered]]
        predictions.append(chain(torch.tensor(q, dtype=torch.float64)).numpy())
    predicted = np.stack(predictions, axis=1)
    error = np.linalg.norm(predicted-np.asarray(reference), axis=-1)
    report = dict(urdf=str(args.urdf.resolve()), urdf_sha256=hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
                  samples=32, arms=2, max_position_error_m=float(error.max()), mean_position_error_m=float(error.mean()),
                  passed=bool(error.max() < 1e-6),
                  note='Independent SAPIEN articulation transforms, no renderer, no physics stepping, identical joint origins/axes. '
                       'Base-frame TCP with +0.12m local-X offset; compares geometric positions, not drive tracking dynamics.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    if not report['passed']:
        raise RuntimeError('Kinematics does not match evaluator')


if __name__ == '__main__':
    main()
