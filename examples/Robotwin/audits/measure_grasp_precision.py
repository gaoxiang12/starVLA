"""Per-episode passive grasp alignment metrics; simulator truth never feeds policy."""
import argparse
import json
from pathlib import Path

import numpy as np


def bin_mm(value):
    if value is None:
        return 'no_closure'
    if not np.isfinite(value) or value < 0:
        raise ValueError('Alignment distance must be finite and nonnegative')
    if value <= 10:
        return 'le_10mm'
    if value <= 20:
        return '10_to_20mm'
    if value <= 30:
        return '20_to_30mm'
    return 'gt_30mm'


def measure(directory, reference=None, require_same_actions=False):
    directory = Path(directory)
    case = json.loads((directory/'result.json').read_text())
    assert case['state'] == 'complete' and case['mode'] == 'full'
    assert case['target'] == 'red' and not case['training_eligible']
    actions = json.loads((directory/'action_trace.json').read_text())
    poses = np.load(directory/'pregrasp_physics_poses.npz', allow_pickle=False)
    scoring = np.load(directory/'physics_scoring_trace.npz', allow_pickle=False)['samples']
    times = poses['sim_s']
    assert np.all(np.diff(times) > 0)
    np.testing.assert_allclose(times[1:], scoring[:,0], atol=1e-9, rtol=0)
    np.testing.assert_array_equal(poses['block_entity_ids'], case['block_entity_ids'])
    tcp = poses['tcp_world_matrix'][..., :3,3]
    blocks = poses['block_world_matrix'][..., :3,3]
    assert np.isfinite(tcp).all() and np.isfinite(blocks).all()
    indices = np.searchsorted(times, [a['sim_s'] for a in actions])
    np.testing.assert_allclose(times[indices], [a['sim_s'] for a in actions], atol=1e-9, rtol=0)
    np.testing.assert_allclose(tcp[indices], np.asarray([a['actual_tcp_poses'] for a in actions])[..., :3], atol=1e-6, rtol=0)
    scene_match = action_match = None
    if reference is not None:
        reference = Path(reference)
        original = json.loads((reference/'result.json').read_text())
        assert case['seed'] == original['seed'] and case['mode'] == original['mode']
        for key in ('initial_block_positions_m','initial_block_quaternions','block_half_extents'):
            np.testing.assert_array_equal(case[key], original[key])
        assert case['initial_scene_audit']['initial_rgb_sha256'] == original['initial_scene_audit']['initial_rgb_sha256']
        scene_match = True
        if require_same_actions:
            assert case['policy'] == original['policy'] and case['execute_horizon'] == original['execute_horizon']
            previous = json.loads((reference/'action_trace.json').read_text())
            np.testing.assert_array_equal([a['requested_robot_action'] for a in actions], [a['requested_robot_action'] for a in previous])
            np.testing.assert_allclose([a['sim_s'] for a in actions], [a['sim_s'] for a in previous], atol=1e-9, rtol=0)
            assert case['result'] == original['result'] and case['termination'] == original['termination']
            action_match = True
    elif require_same_actions:
        raise ValueError('Exact replay check requires a reference case')
    first = case['attempts'][0] if case['attempts'] else None
    events = {}
    before = window = None
    if first is not None:
        arm = int(first['arm'])
        close_index = int(np.searchsorted(times, first['sim_s']))
        drive = (poses['finger_drive_targets'][:,arm,0]+.01)/.055
        opened = np.flatnonzero(drive[:close_index+1] >= .95)
        assert len(opened), 'Missing preclosure open observation'
        before_index = int(opened[-1])
        closed = np.flatnonzero((np.arange(len(times)) >= close_index) & (drive < .05))
        end = int(closed[0]) if len(closed) else close_index
        assert before_index < end
        delta = (tcp[:,arm]-blocks[:,0])*1000
        xy = np.linalg.norm(delta[:,:2], axis=-1)

        def geometry(index):
            return dict(sim_s=float(times[index]), horizontal_offset_mm=float(xy[index]),
                tcp_minus_red_xyz_mm=delta[index].tolist(),
                red_displacement_from_initial_mm=((blocks[index,0]-blocks[0,0])*1000).tolist(),
                drive_target=float(drive[index]),
                tcp_rotation_world=poses['tcp_world_matrix'][index,arm,:3,:3].tolist())

        before = geometry(before_index)
        for threshold in (.95,.8,.5,.2,.05):
            eligible = np.flatnonzero((np.arange(len(times)) > before_index) &
                                     (np.arange(len(times)) <= end) & (drive < threshold))
            events[str(threshold)] = geometry(int(eligible[0])) if len(eligible) else None
        values = xy[before_index+1:end+1]
        window = dict(minimum_mm=float(values.min()), median_mm=float(np.median(values)),
            maximum_mm=float(values.max()), start_sim_s=float(times[before_index+1]),
            end_sim_s=float(times[end]), physics_ticks=len(values), reached_drive_below_05=bool(len(closed)))
    return dict(seed=case['seed'], source_case=str(directory.resolve()),
        checkpoint=case['policy'], execute_horizon=case['execute_horizon'], scored_result=case['result'],
        termination=case['termination'], closing_command_observed=first is not None,
        first_closing_arm=int(first['arm']) if first is not None else None,
        cube_edge_mm=(np.asarray(case['block_half_extents'][0])*2000).tolist(),
        preclosure=before, preclosure_bin=bin_mm(before['horizontal_offset_mm'] if before else None),
        drive_threshold_events=events, closing_window=window,
        reference_initial_scene_exact=scene_match, reference_actions_and_result_exact=action_match,
        note='One first-attempt alignment sample per episode; no-closure episodes remain failures. '
             'TCP-to-red-center XY is geometric offset, not an independently labeled optimal grasp error. '
             'Closure timing is defined by the intermediate drive ramp; physical finger aperture is recorded separately.')


def summarize(records, planned=20):
    assert len({r['seed'] for r in records}) == len(records) <= planned
    bins = {key:0 for key in ('le_10mm','10_to_20mm','20_to_30mm','gt_30mm','no_closure')}
    for row in records:
        bins[row['preclosure_bin']] += 1
    return dict(planned_episodes=planned, completed_episodes=len(records),
        pending_episodes=planned-len(records), preclosure_episode_bins=bins,
        successes=sum(r['scored_result']['success'] for r in records),
        first_attempt_successes=sum(r['scored_result']['first_attempt_success'] for r in records),
        denominator='All planned development episodes, including no-closure failures; unfinished cases stay pending.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--require-same-actions', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = measure(args.case, args.reference, args.require_same_actions)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
