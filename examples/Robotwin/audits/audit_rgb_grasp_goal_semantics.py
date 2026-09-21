"""Compare measured first-close TCP labels with commanded FK on train scenes."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import torch
import yaml

from examples.Robotwin.audits.build_rgb_focus_labels import tcp_positions, project
from starVLA.model.modules.robotwin_pose_kinematics import SerialPoseKinematics


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stats(values):
    values = np.asarray(values)
    return dict(count=len(values), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), maximum=float(values.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchors', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--spatial-labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    manifest = json.loads(args.anchors.read_text())
    split = json.loads(args.split.read_text())
    assert manifest['dataset'] == split['dataset']
    assert sorted(r['episode_index'] for r in manifest['episodes']) == sorted(split['train_episode_ids'])
    assert not set(split['train_episode_ids']) & set(split['validation_episode_ids'])
    config_path = Path('/data/gaoxiang/Code/RoboTwin/assets/embodiments/aloha-agilex/config.yml')
    config = yaml.safe_load(config_path.read_text())
    assert len(config['robot_pose']) == 1 and config['gripper_bias'] == .12
    root = np.asarray(config['robot_pose'][0])
    world_rotation = Rotation.from_quat(root[[4, 5, 6, 3]]).as_matrix()
    urdf = (config_path.parent / config['urdf_path']).resolve()
    chains = [SerialPoseKinematics(urdf, f'{prefix}_link6',
              [f'{prefix}_joint{i}' for i in range(1, 7)]) for prefix in ('fl', 'fr')]
    rows = []
    for source in manifest['episodes']:
        episode, close = source['episode_index'], source['first_close_frame']
        side = source['active_arm']
        arm = ('left', 'right').index(side)
        assert digest(source['raw_path']) == source['raw_sha256']
        assert digest(source['parquet_path']) == source['parquet_sha256']
        label_path = args.spatial_labels / f'episode_{episode:06d}.npz'
        with h5py.File(source['raw_path']) as raw, np.load(label_path, allow_pickle=False) as labels:
            actions = np.asarray(raw['joint_action/vector'])
            parquet = np.asarray(pq.read_table(source['parquet_path'], columns=['action'])['action'].to_pylist())
            np.testing.assert_allclose(actions, parquet, atol=1e-6, rtol=0)
            np.testing.assert_array_equal(np.argwhere(actions[:, [6, 13]] < .2)[0], [close, arm])
            # Every legal original training anchor must target this same first close.
            legal = source['target_end_exclusive'] - manifest['maximum_target_offset']
            assert legal > 0
            np.testing.assert_array_equal(labels['target_step'][:legal], np.full(legal, close))
            np.testing.assert_array_equal(labels['arm'][:legal], np.full(legal, arm))
            np.testing.assert_array_equal(labels['kind'][:legal], np.zeros(legal, dtype=int))
            measured_ee = np.asarray(raw[f'endpose/{side}_endpose'][close])
            measured_tcp = tcp_positions(measured_ee[None])[0]
            joint_command = np.asarray(raw[f'joint_action/{side}_arm'][close])
            with torch.no_grad():
                command_pose = chains[arm].pose(torch.tensor(joint_command, dtype=torch.float64)).numpy()
            command_xyz = world_rotation @ command_pose[:3, 3] + root[:3]
            command_rotation = world_rotation @ command_pose[:3, :3]
            measured_rotation = Rotation.from_quat(measured_ee[[4, 5, 6, 3]]).as_matrix()
            delta_mm = (measured_tcp - command_xyz) * 1000
            # Reproduce the existing projected supervision at ALL legal anchors.
            max_projection_difference = 0.
            for view, camera in enumerate(('head_camera', 'left_camera', 'right_camera')):
                group = raw[f'observation/{camera}']
                xy, valid = project(np.broadcast_to(measured_tcp, (legal, 3)),
                                    np.asarray(group['intrinsic_cv'][:legal]),
                                    np.asarray(group['extrinsic_cv'][:legal]))
                np.testing.assert_array_equal(valid, labels['valid'][:legal, view])
                np.testing.assert_allclose(xy, labels['xy'][:legal, view], atol=1e-6, rtol=0)
                max_projection_difference = max(max_projection_difference, float(np.abs(xy-labels['xy'][:legal, view]).max()))
        rows.append(dict(episode_index=episode, first_close_frame=close, arm=arm,
                         legal_anchors=legal, measured_first_close_tcp_world_m=measured_tcp.tolist(),
                         commanded_first_close_tcp_world_m=command_xyz.tolist(),
                         measured_minus_command_xyz_mm=delta_mm.tolist(),
                         measured_to_command_xy_mm=float(np.linalg.norm(delta_mm[:2])),
                         measured_to_command_3d_mm=float(np.linalg.norm(delta_mm)),
                         measured_to_command_rotation_deg=float(np.rad2deg(Rotation.from_matrix(measured_rotation @ command_rotation.T).magnitude())),
                         projection_max_normalized_difference=max_projection_difference,
                         source_raw_sha256=source['raw_sha256'], source_parquet_sha256=source['parquet_sha256'],
                         spatial_labels_sha256=digest(label_path)))
        if len(rows) % 100 == 0:
            print(f'Audited {len(rows)} train episodes', flush=True)
    report = dict(state='first_close_goal_semantics_audited', episodes=len(rows),
                  legal_anchors=sum(r['legal_anchors'] for r in rows),
                  measured_to_command_xy_mm=stats([r['measured_to_command_xy_mm'] for r in rows]),
                  measured_to_command_3d_mm=stats([r['measured_to_command_3d_mm'] for r in rows]),
                  measured_to_command_abs_z_mm=stats([abs(r['measured_minus_command_xyz_mm'][2]) for r in rows]),
                  measured_to_command_rotation_deg=stats([r['measured_to_command_rotation_deg'] for r in rows]),
                  source_sha256={str(p.resolve()): digest(p) for p in (args.anchors, args.split, config_path, urdf, Path(__file__))},
                  records=rows,
                  limitations=[
                      'Existing 2D labels project measured first-close TCP; current Cartesian action loss targets commanded FK.',
                      'Their difference can include physical tracking, contact and recording timing; this audit does not isolate cause.',
                      'Original HDF lacks block poses/contact traces, so neither first-close point is independently certified as the ideal grasp pose.',
                      'Train-source episodes only; this is label feasibility evidence, not learned-policy evaluation or a new training run.',
                      'No 3D labels have been wired into the current model by this audit.',
                  ])
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'source_sha256')}), flush=True)


if __name__ == '__main__':
    main()
