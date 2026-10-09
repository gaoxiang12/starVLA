"""RoboTwin qpos adapter for the public official-HDF5 GAWM checkpoint."""
from collections import deque
from pathlib import Path
import os
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.Robotwin.eval_files.lila_wam_interface import endpose_state


class ModelClient:
    def __init__(self, args):
        self.client = WebsocketClientPolicy(args.get('host', '127.0.0.1'), int(args['port']))
        meta = self.client.get_server_metadata()
        expected = dict(framework='GAWMOfficialHDF5', action_chunk_size=32, execute_horizon=16,
                        state_dim=16, action_dim=14, state_representation='endpose',
                        action_order='left_arm6,left_gripper,right_arm6,right_gripper')
        if any(meta.get(k) != v for k,v in expected.items()):
            raise ValueError(f'Policy observation/action contract mismatch: {meta}')
        order = os.environ.get('ROBOTWIN_POLICY_IMAGE_CHANNEL_ORDER')
        if order not in ('rgb', 'bgr') or meta.get('policy_image_channel_order') != order:
            raise ValueError('Policy image channel order differs from requested protocol')
        if meta.get('simulator_image_channel_order') != 'rgb' or meta.get('swap_rb_before_normalization') != (order == 'bgr'):
            raise ValueError('Invalid simulator-to-policy color conversion metadata')
        if Path(meta['ckpt_path']).resolve() != Path(args['policy_ckpt_path']).resolve():
            raise ValueError('Connected server checkpoint differs from requested checkpoint')
        self.cameras = list(meta.get('cameras', [meta.get('camera', 'head_camera')]))
        allowed = [['head_camera'], ['head_camera', 'front_camera'],
                   ['head_camera', 'left_camera', 'right_camera']]
        if self.cameras not in allowed:
            raise ValueError(f'Unsupported physical camera order: {self.cameras}')
        self.actions = deque()
        self.episode_start = True

    def reset(self):
        self.actions.clear()
        self.episode_start = True

    def step(self, observation, task_name):
        if not self.actions:
            example = dict(task_name=task_name, state=endpose_state(observation),
                image=[observation['observation'][camera]['rgb'] for camera in self.cameras],
                reset_history=self.episode_start)
            response = self.client.predict_action({'examples':[example]})
            if not response.get('ok'):
                raise RuntimeError(f'Policy inference failed: {response}')
            chunk = np.asarray(response['data']['actions'])
            if chunk.shape != (1,32,14) or not np.isfinite(chunk).all():
                raise ValueError('Invalid predicted action chunk')
            self.actions.extend(chunk[0,:16])
            self.episode_start = False
        return self.actions.popleft()


def get_model(usr_args):
    return ModelClient(usr_args)


def reset_model(model):
    model.reset()


def eval(TASK_ENV, model, observation):
    task = getattr(TASK_ENV, 'task_name', TASK_ENV.__class__.__name__)
    TASK_ENV.take_action(model.step(observation, task), action_type='qpos')
