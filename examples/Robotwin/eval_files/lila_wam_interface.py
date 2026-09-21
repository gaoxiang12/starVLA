"""RoboTwin adapter preserving LiLa-WAM's official observation/action order."""
from collections import deque
from pathlib import Path
import numpy as np
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def capture_cuda_rng():
    import torch
    return torch.cuda.get_rng_state().cpu().numpy()


def restore_cuda_rng(state):
    import torch
    state = np.asarray(state)
    if state.dtype != np.uint8 or state.ndim != 1:
        raise ValueError("Invalid CUDA RNG state returned by policy server")
    torch.cuda.set_rng_state(torch.from_numpy(state.copy()))


def endpose_state(observation):
    pose = observation["endpose"]
    state = np.concatenate((pose["left_endpose"], [pose["left_gripper"]],
                            pose["right_endpose"], [pose["right_gripper"]])).astype(np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("LiLa-WAM requires two 7-D endposes and two gripper values")
    return state


class ModelClient:
    def __init__(self, args):
        self.client = WebsocketClientPolicy(args.get("host", "127.0.0.1"), int(args.get("port", 5794)))
        meta = self.client.get_server_metadata()
        if meta.get("framework") not in {"LiLaWAM", "GAWM_Experiment_B"}:
            raise ValueError("Connected server is not LiLaWAM")
        if meta.get('framework') == 'GAWM_Experiment_B' and (
            meta.get('state_dim') != 16 or meta.get('state_representation') != 'endpose'
            or meta.get('action_order') != 'left_arm6,left_gripper,right_arm6,right_gripper'
            or meta.get('execute_horizon') != 16 or meta.get('action_chunk_size') != 32):
            raise ValueError('GAWM experiment B observation/action contract mismatch')
        if meta.get("policy_rng") != "simulator_cuda":
            raise ValueError("LiLa-WAM server must preserve the simulator's CUDA RNG stream")
        if Path(meta["ckpt_path"]).resolve() != Path(args["policy_ckpt_path"]).resolve():
            raise ValueError("Connected server checkpoint differs from requested checkpoint")
        self.horizon = int(meta["execute_horizon"])
        self.chunk_size = int(meta["action_chunk_size"])
        self.actions = deque()

    def reset(self):
        self.actions.clear()

    def step(self, observation, task_name):
        if not self.actions:
            example = dict(task_name=task_name, state=endpose_state(observation),
                           image=[observation["observation"]["head_camera"]["rgb"]])
            # Upstream inference shares the simulator process and its seeded
            # CUDA generator. Transfer its state in both directions so model
            # noise and any subsequent simulator RNG draws keep that behavior.
            response = self.client.predict_action({"examples": [example], "cuda_rng_state": capture_cuda_rng()})
            if not response.get("ok"):
                raise RuntimeError(f"LiLa-WAM inference failed: {response}")
            chunk = np.asarray(response["data"]["actions"])
            if chunk.shape != (1, self.chunk_size, 14) or not np.isfinite(chunk).all():
                raise ValueError(f"Invalid LiLa-WAM action chunk: {chunk.shape}")
            restore_cuda_rng(response["data"]["cuda_rng_state"])
            self.actions.extend(chunk[0, :self.horizon])
        return self.actions.popleft()


def get_model(usr_args):
    return ModelClient(usr_args)


def reset_model(model):
    model.reset()


def eval(TASK_ENV, model, observation):
    task = getattr(TASK_ENV, "task_name", TASK_ENV.__class__.__name__)
    TASK_ENV.take_action(model.step(observation, task), action_type="qpos")
