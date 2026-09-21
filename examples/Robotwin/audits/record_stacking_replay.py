"""Record one explicitly selected policy rollout, without adding benchmark scores."""
import json
import os
from pathlib import Path
import sys

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'examples/Robotwin/eval_files'), str(ROOT)]
import model2robotwin_interface as interface
import robotwin_eval_runner as runner

OUT = Path(os.environ['STACK_REPLAY_DIR'])
OUT.mkdir(parents=True, exist_ok=False)
SEED = int(os.environ['STACK_REPLAY_SEED'])
TASK = os.environ['STACK_REPLAY_TASK']
original_eval = interface.eval
original_load = runner._load_robotwin_evaluator
container = None
stream = None
frames = 0
actual_seed = None


class RecordingComplete(Exception):
    pass


def write_frame(observation):
    global container, stream, frames
    rgb = np.concatenate([observation['observation'][camera]['rgb']
                          for camera in ('head_camera', 'left_camera', 'right_camera')], axis=1)
    if container is None:
        container = av.open(str(OUT / 'recording.mp4'), 'w')
        stream = container.add_stream('libx264', rate=20)
        stream.width, stream.height = rgb.shape[1], rgb.shape[0]
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'veryfast'}
    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
        container.mux(packet)
    frames += 1


def record_eval(env, model, observation):
    if actual_seed != SEED:
        raise RuntimeError(f'Selected scene was skipped: requested {SEED}, actual {actual_seed}')
    write_frame(observation)
    original_eval(env, model, observation)
    result = dict(task=TASK, seed=actual_seed, actions=env.take_action_cnt,
                  limit=env.step_lim, frames=frames, success=bool(env.eval_success))
    (OUT / 'progress.json').write_text(json.dumps(result) + '\n')
    if env.eval_success or env.take_action_cnt >= env.step_lim:
        write_frame(env.get_obs())
        result.update(frames=frames, complete=True, playback_fps=20,
                      sampling='one frame per policy action, not physical real time',
                      views=['head', 'left_wrist', 'right_wrist'],
                      checkpoint=os.environ['ROBOTWIN_POLICY_CKPT_PATH'],
                      instruction=env.get_instruction(),
                      note='Fresh simulation of selected original evaluation seed; not an archived original video or new benchmark estimate.')
        (OUT / 'metadata.json').write_text(json.dumps(result, indent=2) + '\n')
        env.close_env()
        model.client.close()
        raise RecordingComplete


def load_with_seed(root):
    module = original_load(root)
    original_policy = module.eval_policy
    original_class = module.class_decorator

    def selected_policy(*args, **kwargs):
        args = list(args)
        args[4] = SEED
        return original_policy(*args, **kwargs)

    def tracked_class(task):
        env = original_class(task)
        setup = env.setup_demo

        def tracked_setup(*args, **kwargs):
            global actual_seed
            actual_seed = kwargs['seed']
            return setup(*args, **kwargs)

        env.setup_demo = tracked_setup
        return env

    module.eval_policy = selected_policy
    module.class_decorator = tracked_class
    return module


interface.eval = record_eval
runner._load_robotwin_evaluator = load_with_seed
if __name__ == '__main__':
    try:
        runner.main()
    except RecordingComplete:
        print('Selected scene recording complete; no benchmark score written.', flush=True)
    finally:
        if container is not None:
            for packet in stream.encode():
                container.mux(packet)
            container.close()
