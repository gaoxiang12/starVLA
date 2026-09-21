"""Quantify commanded TCP geometry at the first recorded closure; no rollout replay."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from starVLA.model.modules.robotwin_kinematics import SerialTCPKinematics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--trace-root', type=Path, required=True)
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--robot-config', type=Path, default=Path(
        '/data/gaoxiang/Code/RoboTwin/assets/embodiments/aloha-agilex/config.yml'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    metrics = [json.loads(line) for line in args.metrics.read_text().splitlines()]
    outcome, = [row for row in metrics if row['trial'] == args.trial]
    assert outcome['task'] == 'blocks_ranking_rgb'
    first_close = outcome['first_close_step']
    # The call at step160 records all16 executed actions, through step175.
    if first_close is None or not 0 <= first_close < 176:
        raise ValueError('This audit requires the closure inside the densely recorded prefix')
    config = yaml.safe_load(args.robot_config.read_text())
    assert len(config['robot_pose']) == 1, 'Only shared-base dual-arm embodiment supported'
    pose = np.asarray(config['robot_pose'][0], dtype=float)
    q = pose[3:]
    rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    urdf = (args.robot_config.parent / config['urdf_path']).resolve()
    assert config['gripper_bias'] == .12
    np.testing.assert_allclose(np.asarray(config['global_trans_matrix']) @
        np.asarray(config['delta_matrix']) @ np.array([.12, 0, 0]), [.12, 0, 0], atol=1e-12)
    chains = [SerialTCPKinematics(urdf, f'{side}_link6', [f'{side}_joint{i}' for i in range(1, 7)])
              for side in ('fl', 'fr')]
    inputs, records, continuity = [], [], []
    previous_action = None
    # Include all dense calls through 160, not selected action checkpoints.
    for step in range(0, 161, 16):
        path = args.trace_root / f'trial_{args.trial:03d}' / f'step_{step:04d}.npz'
        with np.load(path, allow_pickle=False) as trace:
            meta = json.loads(str(trace['metadata']))
            assert meta['state_order'] == 'model: left6, right6, left_gripper, right_gripper'
            assert meta['server_metadata']['action_specs']['aloha']['action_spec_id'] == 'aloha_dual_joint_contgrip_next_recorded_14'
            action = trace['actions'][0].copy()
            state = trace['state'].copy()
        assert action.shape == (16, 14) and np.isfinite(action).all()
        if previous_action is not None:
            np.testing.assert_allclose(state, previous_action, atol=1e-6, rtol=0)
            continuity.append(dict(step=step, max_state_previous_command_difference=float(
                np.abs(state-previous_action).max())))
        previous_action = action[-1].copy()
        inputs.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        positions = [chain(torch.as_tensor(action[:, arm*6:(arm+1)*6], dtype=torch.float64)).numpy()
                     @ rotation.T + pose[:3] for arm, chain in enumerate(chains)]
        for index in range(16):
            records.append(dict(step=step+index, grippers=action[index, 12:14].tolist(),
                commanded_tcp_world_m=[positions[0][index].tolist(), positions[1][index].tolist()]))
    detected = next(row for row in records if min(row['grippers']) < .2)
    assert detected['step'] == first_close, 'Saved actions disagree with recorded first closure'
    arm = int(np.argmin(detected['grippers']))
    blocks = np.asarray(outcome['initial_block_positions_m'])
    command = np.asarray(detected['commanded_tcp_world_m'][arm])
    deltas = command - blocks
    report = dict(trial=args.trial, outcome=outcome, first_close_step=first_close,
        closing_arm=('left', 'right')[arm], first_close_tcp_world_m=command.tolist(),
        delta_from_initial_block_centers_mm=(deltas*1000).tolist(),
        horizontal_distance_from_initial_blocks_mm=(np.linalg.norm(deltas[:, :2], axis=1)*1000).tolist(),
        tool_offset_m=[.12, 0, 0], robot_pose=pose.tolist(), trace_inputs=inputs,
        next_observation_drive_target_checks=continuity,
        source_hashes={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.robot_config, urdf, Path('starVLA/model/modules/robotwin_kinematics.py'),
                      Path('examples/Robotwin/eval_files/model2robotwin_interface.py'))},
        records=records,
        note='FK of saved unnormalized absolute joint commands, not measured qpos/TCP, contacts, '
             'tracking error, or expert grasp targets. Reference objects are INITIAL centers, not '
             'contemporaneous poses. The known fixed dual-arm base transform comes from evaluator '
             'robot config. Color identification is only for diagnosis; no values feed back to policy.')
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    window_start, window_end = max(0, first_close - 56), min(175, first_close + 24)
    selected = [r for r in records if window_start <= r['step'] <= window_end]
    xyz = np.asarray([r['commanded_tcp_world_m'][arm] for r in selected])
    steps = np.asarray([r['step'] for r in selected])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(xyz[:, 0], xyz[:, 1], color='black',
                 label=f'Commanded TCP, steps {window_start}-{window_end}')
    for i, color in enumerate(('red', 'green', 'blue')):
        axes[0].scatter(*blocks[i, :2], c=color, marker='s', s=55, label=f'Initial {color} center')
    axes[0].scatter(*command[:2], c='black', marker='x', s=80, label=f'First close {first_close}')
    axes[0].set(xlabel='World X (m)', ylabel='World Y (m)', title='Joint-command FK vs initial block centers')
    axes[0].axis('equal'); axes[0].legend(fontsize=7)
    axes[1].plot(steps, np.linalg.norm(xyz[:, :2]-blocks[0, :2], axis=1)*1000, label='XY distance to initial red')
    axes[1].plot(steps, (xyz[:, 2]-blocks[0, 2])*1000, label='Z above initial red center')
    axes[1].axvline(first_close, c='black', linestyle='--', label='First close <0.2')
    axes[1].set(xlabel='Action step', ylabel='Distance (mm)', title='Commanded position near closure')
    axes[1].legend(fontsize=8)
    fig.savefig(args.output/'command_geometry.png', dpi=150)
    plt.close(fig)
    print(json.dumps({k:report[k] for k in ('first_close_step', 'closing_arm',
        'first_close_tcp_world_m', 'delta_from_initial_block_centers_mm',
        'horizontal_distance_from_initial_blocks_mm')}, indent=2))


if __name__ == '__main__':
    main()
