"""Compare TCP pose convention and local IK with recorded expert correction states."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from starVLA.model.modules.robotwin_pose_kinematics import SerialPoseKinematics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = Path(__file__).resolve().parents[3]
    asset = root.parent/'RoboTwin/assets/embodiments/aloha-agilex'
    config = yaml.safe_load((asset/'config.yml').read_text())
    root_pose = np.asarray(config['robot_pose'][0])
    world_rotation = Rotation.from_quat(root_pose[[4, 5, 6, 3]]).as_matrix()
    chains = [SerialPoseKinematics(asset/config['urdf_path'], f'{s}_link6',
              [f'{s}_joint{i}' for i in range(1, 7)]) for s in ('fl', 'fr')]
    paths = []
    for campaign in ('gawm_pregrasp_correction_pilot_20260909', 'gawm_pregrasp_correction_train20_20260909'):
        paths.extend(sorted((root/'playground/Checkpoints'/campaign).glob('source_*/result.json')))
    records = []
    for path in paths:
        result = json.loads(path.read_text())
        label_path = Path(result['raw_output'])/'pregrasp_labels.json'
        document = json.loads(label_path.read_text())
        rows, names = document['rows'], document['actual_joint_names']
        q = np.asarray([r['actual_articulation_qpos'] for r in rows])
        indices = np.unique(np.linspace(0, len(rows)-1, min(len(rows), 20), dtype=int))
        position_errors, rotation_errors = [], []
        for arm, side in enumerate(('fl', 'fr')):
            ordered = q[indices][:, [names.index(f'{side}_joint{i}') for i in range(1, 7)]]
            predicted = chains[arm].pose(torch.tensor(ordered, dtype=torch.float64)).numpy()
            xyz = predicted[:, :3, 3] @ world_rotation.T+root_pose[:3]
            rotation = world_rotation @ predicted[:, :3, :3]
            truth = np.asarray([rows[i]['actual_tcp_poses'][arm] for i in indices])
            truth_rotation = Rotation.from_quat(truth[:, [4, 5, 6, 3]]).as_matrix()
            position_errors.extend(np.linalg.norm(xyz-truth[:, :3], axis=-1))
            rotation_errors.extend(Rotation.from_matrix(rotation @ truth_rotation.transpose(0, 2, 1)).magnitude())
        arm = ('left', 'right').index(result['arm'])
        side = ('fl', 'fr')[arm]
        seed = q[0, [names.index(f'{side}_joint{i}') for i in range(1, 7)]]
        goal = np.asarray(rows[0]['expert_goal_ee_pose'])
        goal_rotation = Rotation.from_quat(goal[[4, 5, 6, 3]]).as_matrix()
        goal_position = (goal[:3]+goal_rotation @ np.array([.12, 0, 0])-root_pose[:3]) @ world_rotation
        goal_rotation = world_rotation.T @ goal_rotation
        solution = chains[arm].solve_local(torch.tensor(seed, dtype=torch.float64),
            torch.tensor(goal_position, dtype=torch.float64), torch.tensor(goal_rotation, dtype=torch.float64))
        records.append(dict(source_episode=result['source']['source_episode'], offset_xy_mm=result['offset_xy_mm'],
            yaw_deg=result['yaw_deg'], checked_pose_pairs=len(indices)*2,
            max_fk_position_difference_m=float(max(position_errors)),
            max_fk_rotation_difference_rad=float(max(rotation_errors)),
            ik_converged=bool(solution['converged']), ik_position_error_mm=float(solution['position_error_m']*1000),
            ik_orientation_error_deg=float(solution['rotation_error_rad']*180/torch.pi),
            source=str(label_path), sha256=hashlib.sha256(label_path.read_bytes()).hexdigest()))
    assert records
    report = dict(state='complete', records=records,
        all_pose_conventions_match=all(r['max_fk_position_difference_m']<2e-6
                                      and r['max_fk_rotation_difference_rad']<2e-6 for r in records),
        all_local_targets_converged=all(r['ik_converged'] for r in records),
        geometry_source_sha256=hashlib.sha256((root/'starVLA/model/modules/robotwin_pose_kinematics.py').read_bytes()).hexdigest(),
        note='FK vs recorded SAPIEN measured TCP pose; local IK from actual perturbed qpos to known expert pregrasp. '
             'Verifies geometric pose convention/reachability, not learned perception or physical execution of IK actions.')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(cases=len(records), all_pose_conventions_match=report['all_pose_conventions_match'],
        all_local_targets_converged=report['all_local_targets_converged'],
        max_position_difference_m=max(r['max_fk_position_difference_m'] for r in records),
        max_rotation_difference_rad=max(r['max_fk_rotation_difference_rad'] for r in records),
        max_ik_position_error_mm=max(r['ik_position_error_mm'] for r in records))))
    assert report['all_pose_conventions_match'] and report['all_local_targets_converged']


if __name__ == '__main__':
    main()
