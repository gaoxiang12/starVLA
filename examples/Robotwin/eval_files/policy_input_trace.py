"""Sparse, replayable policy calls; observations only, never simulator labels."""
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np


TRACE_SCHEDULE = dict(first_trials=3, dense_through_action_step=160,
                      later_every_policy_calls=8)


def should_trace_policy_call(trial, action_step, execute_horizon):
    """Retain dense initial grasp coverage plus sparse later pick/place calls."""
    if execute_horizon <= 0:
        raise ValueError('A positive execute horizon is required')
    return (0 <= trial < TRACE_SCHEDULE['first_trials'] and action_step >= 0
            and action_step % execute_horizon == 0
            and (action_step <= TRACE_SCHEDULE['dense_through_action_step']
                 or (action_step // execute_horizon) % TRACE_SCHEDULE['later_every_policy_calls'] == 0))


@contextmanager
def trace_policy_call(client, path):
    """Intercept one evaluation step without changing its request or response."""
    original = client.predict_action

    def predict(payload):
        response = original(payload)
        example, = payload['examples']
        arrays = {
            key: np.asarray(example[key])
            for key in ('image', 'native_images', 'image_history', 'state')
            if key in example
        }
        arrays['actions'] = np.asarray(response['data']['actions'])
        arrays['metadata'] = np.asarray(json.dumps({
            'lang': example['lang'],
            'episode_start': bool(example.get('episode_start', False)),
            'unnorm_key': payload.get('unnorm_key'),
            'request_options': {key: value for key, value in payload.items()
                                if key not in ('examples', 'unnorm_key')},
            'server_metadata': getattr(client, '_server_metadata', {}),
            'view_order': ['head_camera', 'left_camera', 'right_camera'],
            'state_order': 'model: left6, right6, left_gripper, right_gripper',
            'state_semantics': 'unnormalized joint drive targets, not measured qpos',
            'image_semantics': 'exact policy input channel order',
            'capture_schedule': TRACE_SCHEDULE,
        }))
        if any(value.dtype.hasobject for value in arrays.values()):
            raise TypeError('Policy traces must be loadable without pickle')
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as handle:
            np.savez_compressed(handle, **arrays)
        return response

    client.predict_action = predict
    try:
        yield
    finally:
        client.predict_action = original
