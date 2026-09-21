"""Replay a recorded empty grasp, then test online expert recovery without reset.

Diagnostic only: the replay seed is reserved for evaluation and all artifacts
are excluded from training. Expert-assisted success is not policy success.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'examples/Robotwin/eval_files'))
from robotwin_eval_runner import _load_robotwin_evaluator
from ranking_outcome_diagnostics import rgb_arrangement

TO_MODEL = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
TO_ROBOT = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
VIEWS = ('head_camera', 'left_camera', 'right_camera')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_task_config(evaluator, output):
    """Use the same clean Aloha scene as the standard evaluator."""
    config = yaml.safe_load(Path('task_config/demo_clean.yml').read_text())
    embodiments = yaml.safe_load(Path('task_config/_embodiment_config.yml').read_text())
    cameras = yaml.safe_load(Path('task_config/_camera_config.yml').read_text())
    assert config['embodiment'] == ['aloha-agilex']
    robot_file = embodiments['aloha-agilex']['file_path']
    camera = cameras[config['camera']['head_camera_type']]
    config.update(task_name='blocks_ranking_rgb', task_config='demo_clean',
                  left_robot_file=robot_file, right_robot_file=robot_file,
                  dual_arm_embodied=True,
                  left_embodiment_config=evaluator.get_embodiment_config(robot_file),
                  right_embodiment_config=evaluator.get_embodiment_config(robot_file),
                  head_camera_h=camera['h'], head_camera_w=camera['w'],
                  eval_mode=True, save_data=False, eval_video_log=False,
                  eval_video_save_dir=None, need_plan=True, render_freq=0,
                  save_path=str(output / 'unused_upstream_data'))
    return config, camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin-root', type=Path, required=True)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--handoff-step', type=int, default=112)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.handoff_step > 0 and args.handoff_step % 16 == 0
    out, trace_root = args.output.resolve(), args.trace_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_files = [trace_root / f'step_{step:04d}.npz' for step in range(0, args.handoff_step+1, 16)]
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    save(out / 'manifest.json', dict(seed=args.seed, handoff_step=args.handoff_step,
         source_hashes=source_hashes, pid=os.getpid(), training_eligible=False,
         note='Reserved evaluation scene. Diagnostic replay and expert recovery, never model benchmark success.'))
    env, video = None, None
    report = dict(state='starting', seed=args.seed, training_eligible=False,
                  replay_checks=[], recovery_frames=0, expert_assisted_success=None)
    phase = 'replay'
    recorded = []

    def status():
        save(out / 'status.json', dict(report, phase=phase, pid=os.getpid(),
             time=time.strftime('%Y-%m-%d %H:%M:%S')))

    def stop(signum, frame):
        raise RuntimeError(f'Signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        os.chdir(args.robotwin_root.resolve())
        evaluator = _load_robotwin_evaluator(args.robotwin_root.resolve())
        from test_render import Sapien_TEST
        from envs.utils import ArmTag
        Sapien_TEST()
        config, camera = load_task_config(evaluator, out)
        env = evaluator.class_decorator('blocks_ranking_rgb')
        env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **config)
        initial_positions = np.asarray([getattr(env, f'block{i}').get_pose().p for i in (1, 2, 3)])
        report['initial_block_positions_m'] = initial_positions.tolist()
        report['state'] = 'running'
        first_close_arm = None
        maximum_lift = np.zeros(3)
        for start in range(0, args.handoff_step+1, 16):
            with np.load(trace_root / f'step_{start:04d}.npz', allow_pickle=False) as trace:
                observation = env.get_obs()
                state = np.asarray(observation['joint_action']['vector'])[TO_MODEL]
                state_error = float(np.abs(state-trace['state']).max())
                images = np.asarray([observation['observation'][view]['rgb'] for view in VIEWS])
                assert images.shape == trace['native_images'].shape
                pixel_mae = np.abs(images.astype(float)-trace['native_images'].astype(float)).mean(axis=(1, 2, 3))
                check = dict(step=start, state_max_error=state_error, per_view_pixel_mae=pixel_mae.tolist())
                report['replay_checks'].append(check)
                status()
                # Drive targets should reproduce exactly; small render/physics
                # differences are reported, never silently treated as exact replay.
                assert state_error < 1e-5, check
                assert float(pixel_mae.max()) < (1.0 if start == 0 else 5.0), check
                if start == 0:
                    Image.fromarray(images[0]).save(out / 'initial.png')
                if start == args.handoff_step:
                    Image.fromarray(images[0]).save(out / 'handoff.png')
                    break
                actions = np.asarray(trace['actions'])
                assert actions.shape == (1, 16, 14) and np.isfinite(actions).all()
                for action in actions[0]:
                    if first_close_arm is None and action[12:14].min() < .2:
                        first_close_arm = int(action[12:14].argmin())
                    env.get_obs()
                    env.take_action(action[TO_ROBOT], action_type='qpos')
                    positions = np.asarray([getattr(env, f'block{i}').get_pose().p for i in (1, 2, 3)])
                    maximum_lift = np.maximum(maximum_lift, positions[:, 2]-initial_positions[:, 2])
        positions = np.asarray([getattr(env, f'block{i}').get_pose().p for i in (1, 2, 3)])
        tcp = np.asarray([env.robot.get_left_tcp_pose()[:3], env.robot.get_right_tcp_pose()[:3]])
        distances = np.linalg.norm(tcp[:, None, :]-positions[None, :, :], axis=-1)
        report.update(first_close_arm=first_close_arm, prefix_max_lift_m=maximum_lift.tolist(),
                      position_reference='gripper_tcp',
                      handoff_block_positions_m=positions.tolist(), tcp_to_block_distances_m=distances.tolist())
        assert first_close_arm is not None
        assert maximum_lift.max() < .01, 'Prefix is not an empty grasp under the declared height criterion'
        assert np.abs(positions[:, 2]-initial_positions[:, 2]).max() < .01
        assert distances.min() > .06, 'A block may still be held or in contact with the gripper'
        assert not env.check_success(), 'Recovery is unnecessary: scene already successful'

        video = subprocess.Popen(['ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', f"{camera['w']}x{camera['h']}",
            '-framerate', '10', '-i', '-', '-pix_fmt', 'yuv420p', '-vcodec', 'libx264',
            '-threads', '2', '-crf', '23', str(out / 'expert_recovery.mp4')], stdin=subprocess.PIPE)

        def capture():
            observation = env.get_obs()
            state = np.asarray(observation['joint_action']['vector']).copy()
            recorded.append(dict(phase=phase, state=state.tolist()))
            video.stdin.write(np.asarray(observation['observation']['head_camera']['rgb'], dtype=np.uint8).tobytes())
            report['recovery_frames'] = len(recorded)
            if len(recorded) % 20 == 0:
                status()

        env._take_picture = capture
        env.set_path_lst(dict(need_plan=True, left_joint_path=[], right_joint_path=[]))
        assert env.plan_success
        phase = 'open_empty_grippers'
        capture()
        env.together_open_gripper()
        phase = 'withdraw_after_empty_grasp'
        active_arm = ArmTag('left' if first_close_arm == 0 else 'right')
        env.move(env.move_by_displacement(arm_tag=active_arm, z=.07))
        assert env.plan_success, 'Recovery withdrawal planning failed'
        phase = 'return_to_origin'
        env.move(env.back_to_origin(arm_tag=ArmTag('left')), env.back_to_origin(arm_tag=ArmTag('right')))
        assert env.plan_success, 'Return-to-origin planning failed'
        phase = 'expert_rgb_replan'
        env.play_once()
        capture()
        final_positions = [getattr(env, f'block{i}').get_pose().p for i in (1, 2, 3)]
        report.update(state='complete', plan_success=bool(env.plan_success),
                      expert_assisted_success=bool(env.check_success()),
                      final_arrangement=rgb_arrangement(final_positions, env.is_left_gripper_open(),
                                                        env.is_right_gripper_open()))
        Image.fromarray(env.get_obs()['observation']['head_camera']['rgb']).save(out / 'final.png')
        save(out / 'recorded_drive_targets.json', recorded)
        status()
    except BaseException as exc:
        report.update(state='failed', error=repr(exc))
        status()
        raise
    finally:
        if video is not None:
            video.stdin.close()
            video.wait(timeout=30)
        if env is not None:
            env.close_env(clear_cache=True)


if __name__ == '__main__':
    main()
