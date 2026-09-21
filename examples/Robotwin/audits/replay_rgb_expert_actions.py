"""Test recorded expert actions through the unmodified RoboTwin evaluation controller.

This is an oracle diagnostic, never a learned-policy benchmark result.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import h5py
import numpy as np
from PIL import Image

from examples.Robotwin.audits.build_rgb_focus_labels import tcp_positions
from examples.Robotwin.audits.smoke_rgb_recovery_teacher import (
    _load_robotwin_evaluator, load_task_config, save,
)
from examples.Robotwin.audits.verify_rgb_recovery_sources import verify_source
from ranking_outcome_diagnostics import rgb_arrangement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin-root', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--episodes', type=int, nargs='+', default=[3, 1, 2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    records = {r['source_episode']: r for r in json.loads(args.sources.read_text())['records']}
    assert set(args.episodes).issubset(records) and len(set(args.episodes)) == len(args.episodes)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(state='starting', pid=os.getpid(), cases=[],
                  note='Oracle next-recorded command sequence through stock take_action(qpos), '
                       'not a learned policy. No expert replanning, no custom control timing, '
                       'no changes to the original datasets or benchmark results.')
    env, video, active = None, None, None
    env_open = False

    def status():
        save(output / 'status.json', dict(report, active_case=active,
             time=time.strftime('%Y-%m-%d %H:%M:%S')))

    def stop(signum, frame):
        raise RuntimeError(f'Signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    status()
    try:
        os.chdir(args.robotwin_root.resolve())
        evaluator = _load_robotwin_evaluator(args.robotwin_root.resolve())
        from test_render import Sapien_TEST
        Sapien_TEST()
        for episode in args.episodes:
            record = records[episode]
            case = output / f'episode_{episode:06d}'
            case.mkdir()
            active = dict(source_episode=episode, scene_seed=record['scene_seed'], state='initializing')
            report['state'] = 'running'
            status()
            config, camera = load_task_config(evaluator, case)
            env = evaluator.class_decorator('blocks_ranking_rgb')
            env.setup_demo(now_ep_num=0, seed=record['scene_seed'], is_test=True, **config)
            env_open = True
            active['source_verification'] = verify_source(env, record, case)
            with h5py.File(record['source_hdf5']) as data:
                actions = np.asarray(data['joint_action/vector'], dtype=np.float32)
                reference_tcp = np.stack([tcp_positions(data[f'endpose/{side}_endpose'][:])
                                          for side in ('left', 'right')], axis=1)
            assert actions.shape == (record['frames'], 14)
            assert reference_tcp.shape == (len(actions), 2, 3)
            current_tcp = np.asarray([env.robot.get_left_tcp_pose()[:3], env.robot.get_right_tcp_pose()[:3]])
            np.testing.assert_allclose(current_tcp, reference_tcp[0], atol=1e-6, rtol=0)
            active['initial_tcp_reference_max_difference_m'] = float(np.abs(current_tcp-reference_tcp[0]).max())
            initial_positions = np.asarray([getattr(env, f'block{i}').get_pose().p for i in (1, 2, 3)])
            maximum_lift = np.zeros(3)
            rows = []
            video = subprocess.Popen(['ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo',
                '-pixel_format', 'rgb24', '-video_size', f"{camera['w']}x{camera['h']}",
                '-framerate', '10', '-i', '-', '-pix_fmt', 'yuv420p', '-vcodec', 'libx264',
                '-threads', '2', '-crf', '23', str(case / 'oracle_replay.mp4')], stdin=subprocess.PIPE)
            active.update(state='replaying', source_frames=len(actions), total_next_recorded_actions=len(actions)-1)
            for target_index in range(1, len(actions)):
                observation = env.get_obs()
                video.stdin.write(observation['observation']['head_camera']['rgb'].tobytes())
                # Source RoboTwin ordering already matches take_action:
                # [left6, left_gripper, right6, right_gripper].
                env.take_action(actions[target_index], action_type='qpos')
                tcp = np.asarray([env.robot.get_left_tcp_pose()[:3], env.robot.get_right_tcp_pose()[:3]])
                positions = np.asarray([getattr(env, f'block{i}').get_pose().p for i in (1, 2, 3)])
                maximum_lift = np.maximum(maximum_lift, positions[:, 2]-initial_positions[:, 2])
                row = dict(target_record_index=target_index, policy_action_steps=int(env.take_action_cnt),
                           tcp_error_vs_recorded_mm=(1000*np.linalg.norm(tcp-reference_tcp[target_index], axis=-1)).tolist(),
                           actual_tcp_positions_m=tcp.tolist(), recorded_tcp_positions_m=reference_tcp[target_index].tolist(),
                           block_positions_m=positions.tolist(), success=bool(env.eval_success))
                rows.append(row)
                active.update(actions_executed=int(env.take_action_cnt), max_lift_m=maximum_lift.tolist())
                if target_index % 16 == 0:
                    status()
                if env.eval_success:
                    break
            final_observation = env.get_obs()
            video.stdin.write(final_observation['observation']['head_camera']['rgb'].tobytes())
            video.stdin.close()
            assert video.wait(timeout=30) == 0
            video = None
            Image.fromarray(final_observation['observation']['head_camera']['rgb']).save(case / 'final.png')
            errors = np.asarray([r['tcp_error_vs_recorded_mm'] for r in rows])
            active.update(state='complete', oracle_success=bool(env.eval_success),
                          check_success=bool(env.check_success()),
                          final_arrangement=rgb_arrangement(
                              [getattr(env, f'block{i}').get_pose().p for i in (1, 2, 3)],
                              env.is_left_gripper_open(), env.is_right_gripper_open()),
                          mean_tcp_error_vs_recorded_mm=errors.mean(axis=0).tolist(),
                          p90_tcp_error_vs_recorded_mm=np.quantile(errors, .9, axis=0).tolist(),
                          comparison_scope='Same command record index, not equal physical elapsed time')
            save(case / 'steps.json', rows)
            save(case / 'result.json', active)
            report['cases'].append(active)
            env.close_env(clear_cache=True)
            env_open, env = False, None
            status()
        active = None
        report.update(state='complete', oracle_successes=sum(r['oracle_success'] for r in report['cases']))
        status()
    except BaseException as exc:
        report.update(state='failed', error=repr(exc))
        status()
        raise
    finally:
        if video is not None:
            video.stdin.close()
            video.wait(timeout=30)
        if env_open and env is not None:
            env.close_env(clear_cache=True)


if __name__ == '__main__':
    main()
