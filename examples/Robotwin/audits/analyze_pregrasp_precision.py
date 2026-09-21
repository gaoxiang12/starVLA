"""Audit recorded physical grasp geometry without changing policy or scoring."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution_mm(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'count': 0}
    return dict(count=len(values), minimum=float(values.min()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), maximum=float(values.max()),
                fraction_le_10mm=float(np.mean(values <= 10)),
                fraction_10_to_20mm=float(np.mean((values > 10) & (values <= 20))),
                fraction_20_to_30mm=float(np.mean((values > 20) & (values <= 30))),
                fraction_gt_30mm=float(np.mean(values > 30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--baseline-case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    case = json.loads((args.case/'result.json').read_text())
    baseline_case = json.loads((args.baseline_case/'result.json').read_text())
    assert case['seed'] == baseline_case['seed'] and case['mode'] == baseline_case['mode']
    assert case['policy'] == baseline_case['policy']
    for key in ('initial_block_positions_m', 'initial_block_quaternions', 'block_half_extents'):
        np.testing.assert_array_equal(case[key], baseline_case[key])
    assert case['initial_scene_audit']['initial_rgb_sha256'] == baseline_case['initial_scene_audit']['initial_rgb_sha256']
    actions = json.loads((args.case/'action_trace.json').read_text())
    baseline = json.loads((args.baseline_case/'action_trace.json').read_text())
    poses = np.load(args.case/'pregrasp_physics_poses.npz', allow_pickle=False)
    scoring = np.load(args.case/'physics_scoring_trace.npz', allow_pickle=False)['samples']
    times = poses['sim_s']
    np.testing.assert_allclose(times[1:], scoring[:, 0], atol=1e-9, rtol=0)
    np.testing.assert_array_equal(poses['block_entity_ids'], case['block_entity_ids'])
    tcp = poses['tcp_world_matrix'][..., :3, 3]
    blocks = poses['block_world_matrix'][..., :3, 3]
    contacts = scoring[:, 4:16].reshape(-1, 3, 2, 2).astype(bool)
    action_times = np.asarray([a['sim_s'] for a in actions])
    indices = np.searchsorted(times, action_times)
    np.testing.assert_allclose(times[indices], action_times, atol=1e-9, rtol=0)
    measured = np.asarray([a['actual_tcp_poses'] for a in actions])[..., :3]
    tcp_check = float(np.abs(tcp[indices]-measured).max())
    assert tcp_check < 1e-6, 'Recorded TCP convention does not match simulator API'
    requested = np.asarray([a['requested_robot_action'] for a in actions])
    same_execution = case['execute_horizon'] == baseline_case['execute_horizon']
    if same_execution:
        baseline_requested = np.asarray([a['requested_robot_action'] for a in baseline[:len(actions)]])
        np.testing.assert_array_equal(requested, baseline_requested)
        np.testing.assert_allclose(action_times, [a['sim_s'] for a in baseline[:len(actions)]], atol=1e-9, rtol=0)

    config_path = Path('/data/gaoxiang/Code/RoboTwin/assets/embodiments/aloha-agilex/config.yml')
    config = yaml.safe_load(config_path.read_text())
    assert len(config['robot_pose']) == 1 and config['gripper_bias'] == .12
    root_pose = np.asarray(config['robot_pose'][0])
    root_rotation = Rotation.from_quat(root_pose[[4, 5, 6, 3]]).as_matrix()
    urdf = (config_path.parent/config['urdf_path']).resolve()
    event = case['attempts'][0] if case['attempts'] else None
    arm = event['arm'] if event is not None else ('left', 'right').index(case['initial_target_side'])
    prefix = ('fl', 'fr')[arm]
    chain = SerialTCPKinematics(urdf, f'{prefix}_link6', [f'{prefix}_joint{i}' for i in range(1, 7)])
    joint_slice = slice(0, 6) if arm == 0 else slice(7, 13)
    command_tcp = chain(torch.tensor(requested[:, joint_slice], dtype=torch.float64)).numpy()
    command_tcp = command_tcp @ root_rotation.T + root_pose[:3]
    # Validate command FK against independently recorded physical joint states.
    names = list(poses['active_joint_names'][0])
    joint_indices = [names.index(f'{prefix}_joint{i}') for i in range(1, 7)]
    actual_q = poses['articulation_qpos'][indices, 0][:, joint_indices]
    physical_fk = chain(torch.tensor(actual_q, dtype=torch.float64)).numpy() @ root_rotation.T + root_pose[:3]
    fk_check = float(np.abs(physical_fk-tcp[indices, arm]).max())
    assert fk_check < 2e-6, 'Command FK world transform does not match measured physical TCP'

    delta = tcp[:, arm]-blocks[:, 0]
    xy_mm = np.linalg.norm(delta[:, :2], axis=-1)*1000
    # This is the intermediate drive target, NOT physical finger aperture.
    drive = (poses['finger_drive_targets'][:, arm, 0]+.01)/.055
    close_index = int(np.searchsorted(times, event['sim_s'])) if event is not None else None
    if event is not None:
        opened = np.flatnonzero(drive[:close_index+1] >= .95)
        if not len(opened):
            raise ValueError('First closure has no preceding open state')
        begin = int(opened[-1]+1)
        closed = np.flatnonzero((np.arange(len(times)) >= close_index) & (drive < .05))
        end = int(closed[0]) if len(closed) else close_index
        window = np.arange(begin, end+1)
    else:
        begin, end = 0, len(times)-1
        window = np.asarray([], dtype=int)

    def geometry(i):
        return dict(sim_s=float(times[i]), horizontal_distance_mm=float(xy_mm[i]),
                    tcp_minus_current_red_mm=(delta[i]*1000).tolist(),
                    red_displacement_from_initial_mm=((blocks[i, 0]-blocks[0, 0])*1000).tolist(),
                    normalized_finger_drive_target=float(drive[i]),
                    actual_finger_joint_positions_m=[float(poses['articulation_qpos'][i, 0, names.index(f'{prefix}_joint{j}')])
                                                     for j in (7, 8)],
                    tcp_rotation_world=poses['tcp_world_matrix'][i, arm, :3, :3].tolist())

    thresholds = {}
    for threshold in ((.95, .8, .5, .2, .05) if event is not None else ()):
        candidates = window[drive[window] < threshold]
        thresholds[str(threshold)] = geometry(int(candidates[0])) if len(candidates) else None
    contact_events = {}
    for block, color in enumerate(('red', 'green', 'blue')):
        contact_events[color] = {}
        for name, present in [('any_finger', contacts[:, block, arm].any(-1)),
                              ('both_fingers', contacts[:, block, arm].all(-1))]:
            ticks = np.flatnonzero(present)
            contact_events[color][name] = dict(physics_ticks=len(ticks),
                first=geometry(int(ticks[0]+1)) if len(ticks) else None)

    # Compare a requested target with the actual state AFTER that action finishes.
    # During TOPP interpolation, distance to the final target is not tracking error.
    finished = np.flatnonzero((action_times[:-1] >= times[begin]) & (action_times[:-1] <= times[end]))
    endpoint_error_mm = np.linalg.norm(command_tcp[:-1]-tcp[indices[1:], arm], axis=-1)*1000
    action_records = [dict(step=int(i), sim_s=float(action_times[i]),
        command_tcp_world_m=command_tcp[i].tolist(),
        commanded_xy_to_current_red_mm=float(np.linalg.norm(command_tcp[i, :2]-blocks[indices[i], 0, :2])*1000),
        actual_xy_to_current_red_mm=float(xy_mm[indices[i]]),
        action_end_tcp_to_requested_target_mm=float(endpoint_error_mm[i])) for i in finished]
    report = dict(seed=case['seed'], arm=('left', 'right')[arm],
        execute_horizon=case['execute_horizon'], termination=case['termination'],
        scored_result=case['result'], closing_command_observed=event is not None,
        arm_selection='first_closing_arm' if event is not None else 'target_side_for_no_closure_audit_only',
        cube_edge_mm=(np.asarray(case['block_half_extents'][0])*2000).tolist(),
        first_close_command_issue=geometry(close_index) if event is not None else None,
        final_geometry=geometry(len(times)-1), drive_threshold_events=thresholds,
        closing_window=dict(start_sim_s=float(times[begin]), end_sim_s=float(times[end]),
                            xy_distance_mm=distribution_mm(xy_mm[window])) if event is not None else None,
        action_end_tcp_target_error_mm=distribution_mm(endpoint_error_mm[finished]),
        target_error_scope='closing_window' if event is not None else 'entire_recorded_rollout_no_closure',
        closure_action_records=action_records, contacts=contact_events,
        validation=dict(baseline_initial_scene_exact=True,
                        baseline_prefix_actions_exact=True if same_execution else None,
                        baseline_prefix_sim_times_equal=True if same_execution else None,
                        tcp_api_max_difference_m=tcp_check, physical_qpos_fk_max_difference_m=fk_check),
        source_sha256={str(p.resolve()):digest(p) for p in [args.case/'result.json',
            args.case/'action_trace.json', args.case/'pregrasp_physics_poses.npz',
            args.case/'physics_scoring_trace.npz', args.baseline_case/'action_trace.json', config_path, urdf,
            Path(__file__)]},
        limitations=['Post-hoc selected failure prefix; not an unbiased success estimate.',
            'TCP-to-current-cube-center distance is geometric offset, not distance to an independently labeled optimal grasp.',
            'Drive thresholds locate the physical drive ramp; actual finger joint positions are recorded separately.',
            'Timing-window fractions describe correlated physics ticks, not independent episodes.',
            'Action-end TCP target distance can include residual motion; it is not a full low-level controller identification.'])
    (args.output/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    visible = (times >= max(0, times[begin]-3)) & (times <= times[end]+3)
    axes[0].plot(times[visible], xy_mm[visible], label='Actual TCP to current red center (XY)')
    selected = (action_times >= times[visible][0]) & (action_times <= times[visible][-1])
    cmd_xy = np.linalg.norm(command_tcp[:, :2]-blocks[indices, 0, :2], axis=-1)*1000
    axes[0].plot(action_times[selected], cmd_xy[selected], '.--', label='Requested target FK to current red (XY)')
    for mm in (10, 20, 30):
        axes[0].axhline(mm, color='gray', linestyle=':', linewidth=.7)
    axes[0].set(ylabel='Horizontal distance (mm)', title=f'Seed {case["seed"]}, {report["arm"]} arm, cube edge {report["cube_edge_mm"][0]:.1f} mm')
    axes[0].legend(fontsize=8)
    axes[1].plot(times[visible], drive[visible], label='Normalized intermediate finger drive target')
    axes[1].set(xlabel='Simulation time (seconds)', ylabel='Drive target')
    axes[1].legend(fontsize=8)
    if event is not None:
        for ax in axes:
            ax.axvspan(times[begin], times[end], alpha=.12, color='red')
            ax.axvline(times[close_index], linestyle='--', color='black', linewidth=.8)
    else:
        axes[0].set_title(axes[0].get_title()+'; no closing command within budget')
    fig.savefig(args.output/'precision.png', dpi=160)
    plt.close(fig)
    print(json.dumps({key:report[key] for key in ('seed', 'cube_edge_mm', 'closing_window',
                     'action_end_tcp_target_error_mm', 'validation')}, indent=2))


if __name__ == '__main__':
    main()
