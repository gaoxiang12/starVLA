"""Reuse the joint-action client, replacing only the raw feedback input."""
import numpy as np
from examples.Robotwin.eval_files.model2robotwin_interface import get_model as joint_model, reset_model
from starVLA.robotwin_feedback import FEEDBACK, feedback_keys
from starVLA.task_language import resolve_task_language


def get_model(usr_args):
    if usr_args.get('action_mode', 'abs') != 'abs':
        raise ValueError('Endpose feedback ablation retains absolute joint actions')
    model = joint_model(usr_args)
    metadata = model.client.get_server_metadata()
    matches = [v for v in FEEDBACK if metadata.get('state_keys') == feedback_keys(v)]
    if len(matches) != 1:
        raise ValueError(f'Unexpected server feedback keys: {metadata.get("state_keys")}')
    if metadata['action_specs']['aloha']['action_dim'] != 14:
        raise ValueError('Feedback ablation must output 14 joint/gripper actions')
    model.feedback_variant = matches[0]
    return model


def eval(TASK_ENV, model, observation):
    instruction = resolve_task_language(TASK_ENV.get_instruction(),
        getattr(TASK_ENV, 'task_name', TASK_ENV.__class__.__name__), model.task_language_mode)
    example = dict(lang=str(instruction),
        image=[observation['observation'][v]['rgb'] for v in ('head_camera', 'left_camera', 'right_camera')],
        state=FEEDBACK[model.feedback_variant](observation))
    action = model.step(example, step=TASK_ENV.take_action_cnt)
    if np.asarray(action).shape != (14,):
        raise ValueError('Expected native RoboTwin joint action, not endpose output')
    TASK_ENV.take_action(action)
