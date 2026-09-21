"""Collect expert-only recovery clips after policy empty grasps on verified train scenes."""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import signal
import time
import weakref

import cv2
import h5py
import numpy as np
import torch
from PIL import Image

from examples.Robotwin.audits.smoke_rgb_recovery_teacher import (
    VIEWS, _load_robotwin_evaluator, load_task_config, save,
)
from examples.Robotwin.audits.verify_rgb_recovery_sources import digest, verify_source
from examples.Robotwin.audits.recovery_frame_capture import record_frame_phases
from model2robotwin_interface import get_model, reset_model, eval as policy_step
from policy_input_trace import trace_policy_call
from ranking_outcome_diagnostics import rgb_arrangement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin-root', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--max-prefix-steps', type=int, default=160)
    parser.add_argument('--allow-tabletop-handoff', action='store_true',
                        help='Also query the expert at unlifted table states; these are not certified empty grasps')
    args = parser.parse_args()
    output, raw_root = args.output.resolve(), args.raw_root.resolve()
    checkpoint = args.checkpoint.resolve()
    sources = json.loads(args.sources.read_text())
    split = json.loads(Path(sources['split']).read_text())
    assert digest(sources['split']) == sources['split_sha256']
    assert digest(sources['seed_file']) == sources['seed_file_sha256']
    seeds = [int(x) for x in Path(sources['seed_file']).read_text().split()]
    for record in sources['records']:
        assert record['source_episode'] in split['train_episode_ids']
        assert record['source_episode'] not in split['validation_episode_ids']
        assert record['scene_seed'] == seeds[record['source_episode']]
    assert checkpoint == Path(sources['policy_checkpoint']).resolve()
    raw_root.mkdir(parents=True, exist_ok=False)
    report = dict(state='starting', pid=os.getpid(), cases=[], checkpoint=str(checkpoint),
                  checkpoint_sha256=digest(checkpoint), raw_root=str(raw_root),
                  note='Pilot collection on verified training scenes. Policy prefixes are excluded from BC targets. '
                       'Raw clips require conversion and loader auditing before training. No benchmark success is recorded.')
    active_case, env, model = None, None, None
    env_is_open = False
    phase = 'starting'

    def status():
        save(output / 'collector_status.json', dict(report, phase=phase,
             active_case=active_case, time=time.strftime('%Y-%m-%d %H:%M:%S')))

    def stop(signum, frame):
        raise RuntimeError(f'Signal {signum}')

    def release_environment():
        nonlocal env
        reference = weakref.ref(env)
        env = None
        gc.collect()
        torch.cuda.empty_cache()
        active_case['environment_python_reference_released'] = reference() is None

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    status()
    try:
        os.chdir(args.robotwin_root.resolve())
        evaluator = _load_robotwin_evaluator(args.robotwin_root.resolve())
        from test_render import Sapien_TEST
        from envs.utils import ArmTag
        Sapien_TEST()
        model = get_model(dict(policy_ckpt_path=str(checkpoint), host='127.0.0.1', port=args.port,
                               unnorm_key='aloha', action_mode='abs', execute_horizon=16))
        assert Path(model.client.get_server_metadata()['ckpt_path']).resolve() == checkpoint
        assert model.execute_horizon == model.action_chunk_size == 16
        report['state'] = 'running'
        for record in sources['records']:
            case = output / 'cases' / f"source_{record['source_episode']:06d}"
            case.mkdir(parents=True, exist_ok=False)
            active_case = dict(source_episode=record['source_episode'], scene_seed=record['scene_seed'],
                               state='starting', raw_training_ready=False)
            phase = 'verify_initial_scene'
            status()
            config, _ = load_task_config(evaluator, case)
            env = evaluator.class_decorator('blocks_ranking_rgb')
            env.setup_demo(now_ep_num=0, seed=record['scene_seed'], is_test=True, **config)
            env_is_open = True
            active_case['source_verification'] = verify_source(env, record, case)
            env.set_instruction('blocks ranking rgb')
            reset_model(model)
            initial = np.asarray([getattr(env, f'block{i}').get_pose().p for i in (1, 2, 3)])
            active_case['initial_block_positions_m'] = initial.tolist()
            maximum_lift = np.zeros(3)
            first_close_step, first_close_arm = None, None
            handoff = False
            phase = 'policy_prefix'
            for step in range(args.max_prefix_steps):
                observation = env.get_obs()
                trace_path = case / 'policy_prefix_traces' / f'step_{step:04d}.npz'
                with trace_policy_call(model.client, trace_path) if step % 16 == 0 else nullcontext():
                    policy_step(env, model, observation)
                grippers = model.raw_actions[step % 16, 12:14]
                if first_close_step is None and grippers.min() < .2:
                    first_close_step, first_close_arm = step, int(grippers.argmin())
                positions = np.asarray([getattr(env, f'block{i}').get_pose().p for i in (1, 2, 3)])
                maximum_lift = np.maximum(maximum_lift, positions[:, 2]-initial[:, 2])
                active_case.update(prefix_steps=step+1, first_close_step=first_close_step,
                                   first_close_arm=first_close_arm, prefix_max_lift_m=maximum_lift.tolist())
                if env.eval_success or maximum_lift.max() >= .01:
                    active_case['state'] = 'skipped_nonempty_grasp'
                    break
                if (step+1) % 16:
                    continue
                status()
                if first_close_step is None or step-first_close_step < 16:
                    continue
                tcp = np.asarray([env.robot.get_left_tcp_pose()[:3], env.robot.get_right_tcp_pose()[:3]])
                distances = np.linalg.norm(tcp[:, None]-positions[None], axis=-1)
                height_error = float(np.abs(positions[:, 2]-initial[:, 2]).max())
                xy_displacement = float(np.linalg.norm(positions[:, :2]-initial[:, :2], axis=-1).max())
                in_workspace = bool((np.abs(positions[:, 0]) <= .35).all()
                                    and (positions[:, 1] >= -.3).all() and (positions[:, 1] <= .15).all())
                active_case.setdefault('handoff_checks', []).append(dict(
                    step=step+1, closed_steps=step-first_close_step,
                    position_reference='gripper_tcp',
                    min_tcp_distance_m=float(distances.min()), max_height_difference_m=height_error,
                    max_xy_displacement_m=xy_displacement, objects_in_workspace=in_workspace,
                    block_positions_m=positions.tolist(), tcp_positions_m=tcp.tolist()))
                clear_empty = distances.min() > .06 and height_error < .01 and xy_displacement < .02
                tabletop = (args.allow_tabletop_handoff and height_error < .01 and in_workspace
                            and grippers[first_close_arm] < .2 and not env.check_success())
                status()
                if clear_empty or tabletop:
                    handoff = True
                    active_case.update(state='expert_handoff',
                                       handoff_type='confirmed_empty_grasp' if clear_empty else 'tabletop_continuation',
                                       tcp_to_block_distances_m=distances.tolist(),
                                       handoff_block_positions_m=positions.tolist())
                    break
            if not handoff:
                if active_case['state'] != 'skipped_nonempty_grasp':
                    active_case['state'] = 'skipped_no_confirmed_empty_grasp'
                env.close_env(clear_cache=True)
                env_is_open = False
                release_environment()
                save(case / 'result.json', active_case)
                report['cases'].append(active_case)
                status()
                continue

            observation = env.get_obs()
            handoff_state = np.asarray(observation['joint_action']['vector']).copy()
            handoff_rgb = observation['observation']['head_camera']['rgb'].copy()
            Image.fromarray(handoff_rgb).save(case / 'handoff.png')
            raw_case = raw_root / f"source_{record['source_episode']:06d}" / 'demo_clean'
            raw_case.mkdir(parents=True, exist_ok=False)
            env.save_dir, env.ep_num, env.FRAME_IDX, env.save_data = str(raw_case), 0, 0, True
            frame_phases = []
            def on_capture(count):
                active_case['recovery_frames'] = count
                if count % 20 == 0:
                    status()

            env.set_path_lst(dict(need_plan=True, left_joint_path=[], right_joint_path=[]))
            assert env.plan_success
            with record_frame_phases(env, frame_phases, lambda: phase, on_capture):
                phase = 'handoff'
                env._take_picture()
                phase = 'open_empty_grippers'
                env.together_open_gripper()
                phase = 'withdraw_after_empty_grasp'
                arm = ArmTag('left' if first_close_arm == 0 else 'right')
                env.move(env.move_by_displacement(arm_tag=arm, z=.07))
                if env.plan_success:
                    phase = 'return_to_origin'
                    env.move(env.back_to_origin(arm_tag=ArmTag('left')), env.back_to_origin(arm_tag=ArmTag('right')))
                info = None
                if env.plan_success:
                    phase = 'expert_rgb_replan'
                    info = env.play_once()
                env._take_picture()
            success = bool(env.plan_success and env.check_success())
            positions = [getattr(env, f'block{i}').get_pose().p for i in (1, 2, 3)]
            active_case.update(plan_success=bool(env.plan_success), expert_assisted_success=success,
                               final_arrangement=rgb_arrangement(positions, env.is_left_gripper_open(),
                                                                 env.is_right_gripper_open()))
            Image.fromarray(env.get_obs()['observation']['head_camera']['rgb']).save(case / 'final.png')
            save(raw_case / 'frame_phases.json', frame_phases)
            provenance = dict(source=record,
                 policy_checkpoint=report['checkpoint'], policy_checkpoint_sha256=report['checkpoint_sha256'],
                 policy_prefix_dir=str(case / 'policy_prefix_traces'), policy_prefix_is_training_target=False,
                 initial_handoff_state_order='robotwin: left6,left_gripper,right6,right_gripper',
                 save_freq=config['save_freq'], physics_timestep_s=float(env.scene.get_timestep()),
                 state_semantics='joint drive targets', action_semantics='next recorded target after conversion',
                 image_storage='Legacy upstream cv2 JPEG encoding of native RGB; use established BGR video correction after conversion',
                 result=active_case)
            save(raw_case / 'recovery_provenance.json', provenance)
            env.close_env(clear_cache=True)
            env_is_open = False
            if success:
                phase = 'merge_and_verify_raw_clip'
                status()
                env.merge_pkl_to_hdf5_video()
                hdf5 = raw_case / 'data/episode0.hdf5'
                with h5py.File(hdf5) as data:
                    values = data['joint_action/vector'][:]
                    assert values.shape == (len(frame_phases), 14) and len(values) > 16
                    assert np.isfinite(values).all()
                    np.testing.assert_allclose(values[0], handoff_state, atol=1e-6, rtol=0)
                    for view in VIEWS:
                        assert len(data[f'observation/{view}/rgb']) == len(values)
                        for field in ('intrinsic_cv', 'extrinsic_cv', 'cam2world_gl'):
                            geometry = data[f'observation/{view}/{field}'][:]
                            assert len(geometry) == len(values) and np.isfinite(geometry).all()
                    decoded = cv2.imdecode(np.frombuffer(data['observation/head_camera/rgb'][0], dtype=np.uint8), cv2.IMREAD_COLOR)
                    assert decoded is not None
                    active_case['handoff_jpeg_pixel_mae'] = float(np.abs(decoded.astype(float)-handoff_rgb.astype(float)).mean())
                    assert active_case['handoff_jpeg_pixel_mae'] < 3.0
                save(raw_case / 'scene_info.json', {'episode_0': info})
                (raw_case / 'seed.txt').write_text(str(record['scene_seed'])+'\n')
                active_case.update(state='raw_clip_verified', hdf5=str(hdf5),
                                   hdf5_sha256=digest(hdf5), recovery_frames=len(frame_phases))
                env.remove_data_cache()
            else:
                active_case.update(state='expert_recovery_failed', retained_cache=env.folder_path['cache'])
            release_environment()
            save(raw_case / 'recovery_provenance.json', provenance)
            save(case / 'result.json', active_case)
            report['cases'].append(active_case)
            status()
        active_case = None
        report.update(state='complete', accepted_raw_clips=sum(r['state']=='raw_clip_verified' for r in report['cases']))
        status()
    except BaseException as exc:
        report.update(state='failed', error=repr(exc))
        status()
        raise
    finally:
        if model is not None:
            model.client.close()
        if env_is_open and env is not None and hasattr(env, 'scene'):
            env.close_env(clear_cache=True)


if __name__ == '__main__':
    main()
