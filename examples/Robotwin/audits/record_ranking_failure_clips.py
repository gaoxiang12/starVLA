"""Record a continuous prefix of a known failed rollout, without scoring it."""
import json
import os
from pathlib import Path
import sys

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'examples/Robotwin/eval_files'))
sys.path.insert(0, str(ROOT))
import model2robotwin_interface as interface
from examples.Robotwin.eval_files.robotwin_eval_runner import main

OUT = Path(os.environ['RANKING_VIDEO_DIR'])
OUT.mkdir(parents=True, exist_ok=False)
TASK = os.environ['RANKING_VIDEO_TASK']
LIMIT = int(os.environ.get('RANKING_VIDEO_STEPS', '320'))
original_eval = interface.eval
container = None
stream = None
frames = 0


class ClipComplete(Exception):
    pass


def write_frame(observation):
    global container, stream, frames
    rgb = np.concatenate([observation['observation'][c]['rgb']
                          for c in ('head_camera', 'left_camera', 'right_camera')], axis=1)
    if container is None:
        container = av.open(str(OUT / 'recording.mp4'), mode='w')
        stream = container.add_stream('libx264', rate=10)
        stream.width, stream.height = rgb.shape[1], rgb.shape[0]
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'veryfast'}
    frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
    for packet in stream.encode(frame):
        container.mux(packet)
    frames += 1


def record_eval(env, model, observation):
    step = env.take_action_cnt
    write_frame(observation)
    original_eval(env, model, observation)
    (OUT / 'progress.json').write_text(json.dumps(dict(step=env.take_action_cnt,
         limit=LIMIT, frames=frames, success=bool(env.eval_success))) + '\n')
    if env.eval_success or env.take_action_cnt >= LIMIT:
        write_frame(env.get_obs())
        metadata = dict(task=TASK, frames=frames, actions=env.take_action_cnt, playback_fps=10,
                        sampling='one frame per policy action, not physical real time',
                        views=['head', 'left_wrist', 'right_wrist'],
                        clip_prefix_only=True, formal_seed=100000,
                        formal_seed_result='failure in original unmodified 1200-step evaluation',
                        success_during_prefix=bool(env.eval_success))
        (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        env.close_env()
        model.client.close()
        raise ClipComplete


interface.eval = record_eval
if __name__ == '__main__':
    try:
        main()
    except ClipComplete:
        print('Diagnostic video prefix complete; no benchmark score was written.', flush=True)
    finally:
        if container is not None:
            for packet in stream.encode():
                container.mux(packet)
            container.close()
