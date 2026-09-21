"""Record a diagnostic rollout without changing policy or success predicates."""
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'examples/Robotwin/eval_files'))
sys.path.insert(0, str(ROOT))
import model2robotwin_interface as interface
from examples.Robotwin.eval_files.robotwin_eval_runner import main

OUT = Path(os.environ['RANKING_TRACE_DIR'])
OUT.mkdir(parents=True, exist_ok=False)
original_eval = interface.eval


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def positions(env):
    return [getattr(env, f'block{i}').get_pose().p.tolist() for i in (1, 2, 3)]


def trace_eval(env, model, observation):
    step = env.take_action_cnt
    if step % 80 == 0:
        for camera in ('head_camera', 'left_camera', 'right_camera'):
            Image.fromarray(observation['observation'][camera]['rgb']).save(OUT / f'{step:04d}_{camera}.png')
    before = positions(env)
    original_eval(env, model, observation)
    after = positions(env)
    action = model.raw_actions[step % model.action_chunk_size]
    row = dict(step=step, blocks_before=before, blocks_after=after,
               state=observation['joint_action']['vector'], endpose=observation.get('endpose', {}),
               model_order_action=action, left_open=env.is_left_gripper_open(),
               right_open=env.is_right_gripper_open(), success=bool(env.eval_success))
    with (OUT / 'trace.jsonl').open('a') as handle:
        handle.write(json.dumps(native(row)) + '\n')
    if env.eval_success or env.take_action_cnt >= env.step_lim:
        final = env.get_obs()
        Image.fromarray(final['observation']['head_camera']['rgb']).save(OUT / 'final_head_camera.png')
        (OUT / 'outcome.json').write_text(json.dumps(dict(success=bool(env.eval_success),
            steps=env.take_action_cnt, blocks=after, left_open=env.is_left_gripper_open(),
            right_open=env.is_right_gripper_open()), indent=2) + '\n')


interface.eval = trace_eval
if __name__ == '__main__':
    main()
