"""Use the standard evaluator and add observational block-lift diagnostics."""
import json
from contextlib import nullcontext
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model2robotwin_interface as interface
from policy_input_trace import trace_policy_call, should_trace_policy_call
from ranking_outcome_diagnostics import rgb_arrangement
from robotwin_eval_runner import main

OUT = Path(os.environ['ROBOTWIN_RANKING_METRICS_PATH'])
original_eval = interface.eval


def record_eval(env, model, observation):
    step = env.take_action_cnt
    if step == 0:
        model._ranking_probe = dict(
            initial_z=np.array([getattr(env, f'block{i}').get_pose().p[2] for i in (1, 2, 3)]),
            initial_positions=[getattr(env, f'block{i}').get_pose().p.tolist() for i in (1, 2, 3)],
            first_block_lift_steps=[None, None, None],
            max_lift=np.zeros(3), first_close_step=None, first_arm=None,
            first_attempt_ended=False, first_attempt_lifted=False, first_lift_step=None)
    trace_root = os.environ.get('ROBOTWIN_POLICY_TRACE_DIR')
    execute_horizon = getattr(model, 'execute_horizon', model.action_chunk_size)
    capture = trace_root and should_trace_policy_call(int(env.test_num), step, execute_horizon)
    # Separate attempts using the retry wrapper's unique metrics filename.
    trace_path = (Path(trace_root) / OUT.stem / f'trial_{int(env.test_num):03d}'
                  / f'step_{step:04d}.npz') if capture else None
    with trace_policy_call(model.client, trace_path) if capture else nullcontext():
        original_eval(env, model, observation)
    probe = model._ranking_probe
    z = np.array([getattr(env, f'block{i}').get_pose().p[2] for i in (1, 2, 3)])
    lift = z - probe['initial_z']
    probe['max_lift'] = np.maximum(probe['max_lift'], lift)
    for index in range(3):
        if lift[index] > .04 and probe['first_block_lift_steps'][index] is None:
            probe['first_block_lift_steps'][index] = step
    grippers = np.asarray(model.raw_actions[step % execute_horizon])[12:14]
    if probe['first_close_step'] is None and grippers.min() < .2:
        probe['first_close_step'] = step
        probe['first_arm'] = int(grippers.argmin())
    if lift.max() > .04:
        if probe['first_lift_step'] is None:
            probe['first_lift_step'] = step
        if probe['first_close_step'] is not None and not probe['first_attempt_ended']:
            probe['first_attempt_lifted'] = True
    if probe['first_arm'] is not None and grippers[probe['first_arm']] > .8:
        probe['first_attempt_ended'] = True
    if env.eval_success or env.take_action_cnt >= env.step_lim:
        row = dict(task=env.task_name, trial=int(env.test_num), success=bool(env.eval_success),
                   action_steps=int(env.take_action_cnt), max_lift_m=probe['max_lift'].tolist(),
                   blocks_lifted=int((probe['max_lift'] > .04).sum()),
                   any_block_lifted=bool((probe['max_lift'] > .04).any()),
                   first_close_step=probe['first_close_step'], first_lift_step=probe['first_lift_step'],
                   first_attempt_lifted=probe['first_attempt_lifted'],
                   initial_block_positions_m=probe['initial_positions'],
                   first_block_lift_steps=probe['first_block_lift_steps'])
        if env.task_name == 'blocks_ranking_rgb':
            row['rgb_arrangement'] = rgb_arrangement(
                [getattr(env, f'block{i}').get_pose().p for i in (1, 2, 3)],
                env.is_left_gripper_open(), env.is_right_gripper_open())
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open('a') as handle:
            handle.write(json.dumps(row) + '\n')


interface.eval = record_eval
if __name__ == '__main__':
    main()
