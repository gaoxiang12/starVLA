"""Held-out, progress-stratified action probe against a current-command baseline."""
import json
from pathlib import Path
import av
import cv2
import numpy as np
import pyarrow.parquet as pq
from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.Robotwin.eval_files.model2robotwin_interface import ROBOTWIN_TO_MODEL_JOINT_ORDER

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'examples/Robotwin/audits/ranking_failure_20260907'
CHECKPOINT = ROOT / 'playground/Checkpoints/gawm_s_robotwin_continuous_next_49tasks_40k_20260905/final_model/pytorch_model.pt'
PROC = PolicyNormProcessor(str(CHECKPOINT), unnorm_key='aloha')


def normalize(actions):
    data = {}
    cursor = 0
    for key in PROC.action_keys:
        width = PROC._action_key_dims[key]
        data[key] = actions[..., cursor:cursor + width].copy()
        cursor += width
    result = PROC.transform.apply(data)
    return np.concatenate([result[k].numpy() for k in PROC.action_keys], axis=-1)


def evaluate(task, port):
    client = WebsocketClientPolicy('127.0.0.1', port)
    root = Path('/data/gaoxiang/RoboTwinGenerated/Clean') / task
    rows = []
    try:
        for episode in range(0, 160, 20):
            raw = pq.read_table(root / f'data/chunk-000/episode_{episode:06d}.parquet',
                                columns=['action']).column('action').to_pylist()
            actions = np.asarray(raw, dtype=np.float32)[:, ROBOTWIN_TO_MODEL_JOINT_ORDER]
            anchors = [int((i + .5) / 8 * (len(actions) - 1)) for i in range(8)]
            images = {t: [] for t in anchors}
            for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
                with av.open(str(root / f'videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4')) as video:
                    for i, frame in enumerate(video.decode(video=0)):
                        if i in images:
                            images[i].append(cv2.resize(frame.to_ndarray(format='rgb24'), (224, 224), interpolation=cv2.INTER_AREA))
                        if i >= max(anchors):
                            break
            for start in (0, 4):
                ts = anchors[start:start + 4]
                examples = [dict(image=images[t], state=actions[t], lang=task.replace('_', ' '), episode_start=True) for t in ts]
                response = client.predict_action(dict(examples=examples, unnorm_key='aloha', do_sample=False))
                pred = np.asarray(response['data']['actions'])
                assert pred.shape == (4, 16, 14) and np.isfinite(pred).all()
                for j, t in enumerate(ts):
                    target = actions[t + 1:t + 17]
                    n = len(target)
                    hold = np.repeat(actions[t:t + 1], n, axis=0)
                    pn, tn, hn = normalize(pred[j, :n]), normalize(target), normalize(hold)
                    rows.append(dict(episode=episode, anchor=t, progress_bin=start + j,
                        normalized_l1=float(np.abs(pn - tn).mean()),
                        hold_normalized_l1=float(np.abs(hn - tn).mean()),
                        joint_l1_rad=float(np.abs(pred[j, :n, :12] - target[:, :12]).mean()),
                        hold_joint_l1_rad=float(np.abs(hold[:, :12] - target[:, :12]).mean()),
                        grip_l1=float(np.abs(pred[j, :n, 12:] - target[:, 12:]).mean()),
                        hold_grip_l1=float(np.abs(hold[:, 12:] - target[:, 12:]).mean())))
            print(task, 'episode', episode, 'complete', flush=True)
        metrics = [k for k in rows[0] if k not in ('episode', 'anchor', 'progress_bin')]
        summary = {k: float(np.mean([r[k] for r in rows])) for k in metrics}
        by_bin = [{k: float(np.mean([r[k] for r in rows if r['progress_bin'] == i])) for k in metrics} for i in range(8)]
        payload = dict(task=task, episodes=list(range(0, 160, 20)), samples=len(rows),
                       checkpoint=str(CHECKPOINT), summary=summary, by_progress_bin=by_bin, rows=rows)
        (OUT / f'{task}_heldout_probe.json').write_text(json.dumps(payload, indent=2) + '\n')
        print(task, summary, flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    for task, port in [('blocks_ranking_rgb', 5801), ('blocks_ranking_size', 5802)]:
        evaluate(task, port)
