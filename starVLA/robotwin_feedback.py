"""NumPy-only feedback contracts usable in both training and simulator envs."""
import numpy as np


def measured_endpose(observation):
    """Native world-frame EE pose (not TCP), wxyz quaternion, continuous grips."""
    values = observation['endpose']
    parts = []
    for side in ('left', 'right'):
        pose = np.asarray(values[f'{side}_endpose'], dtype=np.float32)
        if pose.shape[-1:] != (7,) or not np.isfinite(pose).all():
            raise ValueError(f'Invalid {side} endpose')
        norm = np.linalg.norm(pose[..., 3:7], axis=-1)
        if not np.allclose(norm, 1., atol=1e-3, rtol=0):
            raise ValueError(f'Invalid {side} wxyz quaternion')
        grip = np.asarray(values[f'{side}_gripper'], dtype=np.float32)
        if grip.shape == pose.shape[:-1]:
            grip = grip[..., None]
        if grip.shape != (*pose.shape[:-1], 1):
            raise ValueError(f'Invalid {side} gripper shape')
        if not np.isfinite(grip).all() or np.any((grip < -1e-5) | (grip > 1+1e-5)):
            raise ValueError(f'Invalid {side} gripper value')
        parts.extend((pose, grip))
    return np.concatenate(parts, axis=-1)


def padded_command(observation):
    """Width-matched control: [L6,0,Lgrip,R6,0,Rgrip], raw env joint order."""
    command = np.asarray(observation['joint_action']['vector'], dtype=np.float32)
    if command.shape[-1:] != (14,) or not np.isfinite(command).all():
        raise ValueError('Invalid RoboTwin joint command')
    zero = np.zeros_like(command[..., :1])
    return np.concatenate((command[..., :6], zero, command[..., 6:7],
                           command[..., 7:13], zero, command[..., 13:14]), axis=-1)


FEEDBACK = {'endpose': measured_endpose, 'command': padded_command}


def feedback_keys(variant):
    if variant not in FEEDBACK:
        raise ValueError(f'Unknown feedback variant: {variant}')
    return [f'state.{variant}_left', 'state.left_gripper',
            f'state.{variant}_right', 'state.right_gripper']
