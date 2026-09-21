"""One red-block lift-and-hold diagnostic; scoring truth never enters the policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples/Robotwin/eval_files')]
from examples.Robotwin.audits.grasp_lift_scoring import AttemptCounter, LiftHoldScore
from examples.Robotwin.audits.smoke_rgb_recovery_teacher import load_task_config
from robotwin_eval_runner import _load_robotwin_evaluator


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


class BudgetReached(Exception):
    pass


class ObservedScene:
    """Forward every scene call unchanged, observing immediately after physics step."""
    def __init__(self, scene, callback):
        self._scene, self._callback = scene, callback

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def step(self):
        result = self._scene.step()
        self._callback()
        return result


def bottom_z(actor):
    transform = actor.get_pose().to_transformation_matrix()
    return float(transform[2, 3] - np.abs(transform[2, :3]) @ np.asarray(actor.config['extents']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--mode', choices=['full', 'near'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--port', type=int, default=6820)
    parser.add_argument('--execute-horizon', type=int, default=16)
    parser.add_argument('--expert-smoke', action='store_true')
    parser.add_argument('--max-actions', type=int, default=600)
    parser.add_argument('--max-sim-seconds', type=float, default=120.)
    args = parser.parse_args()
    if not args.expert_smoke and args.checkpoint is None:
        parser.error('Policy evaluation requires a checkpoint')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    robotwin = ROOT.parent / 'RoboTwin'
    os.chdir(robotwin)
    evaluator = _load_robotwin_evaluator(robotwin)
    from test_render import Sapien_TEST
    from envs.utils import ArmTag
    from model2robotwin_interface import get_model, reset_model, eval as policy_step
    import transforms3d.quaternions as tq
    import transforms3d.axangles as ta

    Sapien_TEST()
    env, original_scene, model, container, stream = None, None, None, None, None
    frames, physics, actions = 0, [], []
    score, attempts = LiftHoldScore(), AttemptCounter()
    report = dict(state='starting', mode=args.mode, seed=args.seed, target='red',
                  policy='expert_positive_control' if args.expert_smoke else str(args.checkpoint.resolve()),
                  training_eligible=False, execute_horizon=args.execute_horizon,
                  scoring=dict(clearance_m=.05, continuous_hold_sim_s=1.,
                      contact='both moving finger links of same arm; contact separation <= 1 mm',
                      max_attempts=2, max_actions=args.max_actions, max_sim_seconds=args.max_sim_seconds),
                  note='Dedicated grasp diagnostic, not the original ranking benchmark. Existing policy keeps its original task ID; red is the first target in that task. Near initialization uses simulator geometry; policy observations do not.')

    def status():
        save(out / 'status.json', dict(report, pid=os.getpid(), frames=frames,
             score=score.result(), attempts=attempts.events, time=time.strftime('%Y-%m-%d %H:%M:%S')))

    def frame(observation):
        nonlocal container, stream, frames
        rgb = np.concatenate([observation['observation'][v]['rgb']
                              for v in ('head_camera', 'left_camera', 'right_camera')], axis=1)
        if container is None:
            container = av.open(str(out / 'rollout.mp4'), 'w')
            stream = container.add_stream('libx264', rate=20)
            stream.width, stream.height = rgb.shape[1], rgb.shape[0]
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '20', 'preset': 'veryfast', 'threads': '2'}
        for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
            container.mux(packet)
        frames += 1

    try:
        config, _ = load_task_config(evaluator, out)
        env = evaluator.class_decorator('blocks_ranking_rgb')
        env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **config)
        blocks = [env.block1, env.block2, env.block3]
        initial = np.asarray([b.get_pose().p for b in blocks])
        reference_bottom = np.asarray([bottom_z(b) for b in blocks])
        report.update(initial_block_positions_m=initial.tolist(),
                      initial_block_quaternions=[b.get_pose().q.tolist() for b in blocks],
                      block_half_extents=[list(b.config['extents']) for b in blocks])
        from examples.Robotwin.audits.audit_rgb_scene_and_color import colors
        initial_rgb = env.get_obs()['observation']['head_camera']['rgb']
        # Existing scene audit used uncorrected converted videos. Reverse only this
        # scoring-side image to compare in that same convention; policy stays RGB.
        current_colors = colors(initial_rgb[..., ::-1])
        audit_path = ROOT/'examples/Robotwin/audits/rgb_scene_and_color_20260907.json'
        scene_audit = json.loads(audit_path.read_text())
        if set(current_colors) != {'red', 'green', 'blue'}:
            raise RuntimeError('Initial scene overlap audit could not identify all three blocks')
        matches, distances = [], []
        for row in scene_audit['episodes_detail']:
            reference = row.get('initial', {})
            if set(reference) != set(current_colors):
                continue
            distance = max(float(np.linalg.norm(np.asarray(current_colors[c]['xy'])-reference[c]['xy']))
                           for c in current_colors)
            distances.append((distance, row['episode']))
            if distance <= 1.5 and all(.8 <= current_colors[c]['area']/reference[c]['area'] <= 1.25
                                       for c in current_colors):
                matches.append(row['episode'])
        report['initial_scene_audit'] = dict(legacy_video_order_colors=current_colors,
            reference_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            initial_rgb_sha256=hashlib.sha256(initial_rgb.tobytes()).hexdigest(),
            compared_reference_scenes=len(distances), matched_original_episodes=matches,
            nearest_reference_distance_px=min(distances)[0], nearest_reference_episode=min(distances)[1])
        if matches:
            raise RuntimeError(f'New seed duplicates an original dataset scene: {matches}')
        arm = ArmTag('left' if initial[0, 0] < 0 else 'right')
        report['initial_target_side'] = str(arm)
        if args.mode == 'near':
            pre, _ = env.choose_grasp_pose(env.block1, arm_tag=arm, pre_dis=.06, target_dis=0.)
            if pre is None:
                raise RuntimeError('Near setup has no pregrasp pose')
            rng = np.random.default_rng(args.seed + 70000000)
            perturbation = rng.uniform([-.015, -.015, -.1745329252], [.015, .015, .1745329252])
            pre = np.asarray(pre, dtype=float)
            pre[:2] += perturbation[:2]
            pre[3:] = tq.qmult(tq.mat2quat(ta.axangle2mat([0, 0, 1], perturbation[2])), pre[3:])
            env.move(env.move_to_pose(arm_tag=arm, target_pose=pre.tolist()))
            displacement = np.linalg.norm(np.asarray([b.get_pose().p for b in blocks]) - initial, axis=1)
            if not env.plan_success or displacement.max() > .002:
                raise RuntimeError('Near initialization failed or disturbed a block by more than 2 mm')
            report['near_setup'] = dict(requested_ee_pose=pre.tolist(), perturbation_xy_yaw=perturbation.tolist(),
                actual_ee_pose=env.get_arm_pose(str(arm)), block_displacement_m=displacement.tolist())

        original_scene = env.scene
        finger_ids = [[int(joint.child_link.entity.per_scene_id) for joint, _, _ in group]
                      for group in (env.robot.left_gripper, env.robot.right_gripper)]
        if any(len(group) != 2 for group in finger_ids):
            raise RuntimeError('Scorer is specific to two moving fingers per Aloha arm')
        block_ids = [int(b.actor.per_scene_id) for b in blocks]
        assert len(set(block_ids)) == 3, 'Do not identify all three box actors by their shared name'
        dt = float(original_scene.get_timestep())
        report.update(physics_dt_s=dt, block_entity_ids=block_ids, finger_entity_ids=finger_ids)
        finger_lookup = {entity: (arm_i, finger_i) for arm_i, group in enumerate(finger_ids)
                         for finger_i, entity in enumerate(group)}
        block_lookup = {entity: i for i, entity in enumerate(block_ids)}

        def contact_snapshot():
            contacts = np.zeros((3, 2, 2), dtype=bool)
            for contact in original_scene.get_contacts():
                ids = [int(body.entity.per_scene_id) for body in contact.bodies]
                for b_id, finger_id in (ids, ids[::-1]):
                    if b_id in block_lookup and finger_id in finger_lookup:
                        a, f = finger_lookup[finger_id]
                        if any(point.separation <= .001 for point in contact.points):
                            contacts[block_lookup[b_id], a, f] = True
            return contacts

        if contact_snapshot().any():
            raise RuntimeError('Scored initial state must not already touch any block')

        def after_physics_step():
            clearances = np.asarray([bottom_z(b) for b in blocks]) - reference_bottom
            contacts = contact_snapshot()
            score.update(dt, clearances, contacts, attempts.arm_attempts)
            physics.append([score.elapsed_s, *clearances.tolist(), *contacts.ravel().astype(int).tolist(),
                            *attempts.arm_attempts.tolist()])
            if not args.expert_smoke and not score.success and score.elapsed_s >= args.max_sim_seconds:
                raise BudgetReached('simulation_time_budget')

        env.scene = ObservedScene(original_scene, after_physics_step)
        env.check_success = lambda: score.success
        env.eval_success = False
        env.take_action_cnt = 0
        env.step_lim = args.max_actions
        env.set_instruction('blocks ranking rgb')
        report['state'] = 'running'
        frame(env.get_obs())
        status()
        if args.expert_smoke:
            selected = 0 if str(arm) == 'left' else 1
            command = np.ones(2)
            command[selected] = 0.
            attempts.command(command, 0, 0.)
            # Capture expert motion at its existing picture callbacks; never policy score.
            env._take_picture = lambda: frame(env.get_obs())
            env.move(env.grasp_actor(env.block1, arm_tag=arm, pre_grasp_dis=.09))
            env.move(env.move_by_displacement(arm_tag=arm, z=.09))
            for tick in range(int(2. / dt)):
                env.scene.step()
                if tick % 10 == 0:
                    frame(env.get_obs())
                if score.success:
                    break
            if not env.plan_success:
                raise RuntimeError('Expert positive-control planning failed')
            report['termination'] = 'expert_control_complete'
        else:
            model = get_model(dict(policy_ckpt_path=str(args.checkpoint.resolve()), host='127.0.0.1',
                                  port=args.port, unnorm_key='aloha', action_mode='abs',
                                  execute_horizon=args.execute_horizon))
            meta = model.client.get_server_metadata()
            assert Path(meta['ckpt_path']).resolve() == args.checkpoint.resolve()
            report['server_metadata'] = meta
            reset_model(model)
            take_action = env.take_action

            def budgeted_action(action, action_type='qpos'):
                if action_type != 'qpos':
                    raise ValueError('This baseline expects the unchanged joint-action policy')
                action = np.asarray(action)
                if not attempts.command(action[[6, 13]], env.take_action_cnt, score.elapsed_s):
                    raise BudgetReached('third_grasp_attempt_refused')
                actions.append(dict(action_step=env.take_action_cnt, sim_s=score.elapsed_s,
                    requested_robot_action=action.tolist(),
                    actual_tcp_poses=[env.robot.get_left_tcp_pose(), env.robot.get_right_tcp_pose()],
                    actual_articulation_qpos=[env.robot.left_entity.get_qpos().tolist(), env.robot.right_entity.get_qpos().tolist()],
                    block_positions_m=[b.get_pose().p.tolist() for b in blocks]))
                take_action(action, action_type=action_type)

            env.take_action = budgeted_action
            report['termination'] = 'action_budget'
            try:
                for step in range(args.max_actions):
                    if score.elapsed_s >= args.max_sim_seconds:
                        report['termination'] = 'simulation_time_budget'
                        break
                    policy_step(env, model, env.get_obs())
                    frame(env.get_obs())
                    if step % 16 == 0:
                        status()
                    if score.success:
                        report['termination'] = 'success'
                        break
            except BudgetReached as exc:
                report['termination'] = str(exc)
        report.update(state='complete', result=score.result(), attempts=attempts.events,
                      actions=env.take_action_cnt, frames=frames,
                      final_block_positions_m=[b.get_pose().p.tolist() for b in blocks])
        save(out / 'result.json', report)
        status()
    except BaseException as exc:
        report.update(state='failed', error=repr(exc))
        status()
        raise
    finally:
        if physics:
            np.savez_compressed(out / 'physics_scoring_trace.npz', samples=np.asarray(physics),
                columns=np.asarray(['sim_s', 'red_clearance', 'green_clearance', 'blue_clearance'] +
                    [f'contact_b{b}_a{a}_f{f}' for b, a, f in np.ndindex(3, 2, 2)] + ['left_attempt', 'right_attempt']))
        save(out / 'action_trace.json', actions)
        if container is not None:
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        if model is not None:
            model.client.close()
        if env is not None:
            if original_scene is not None:
                env.scene = original_scene
            env.close_env(clear_cache=True)


if __name__ == '__main__':
    main()
