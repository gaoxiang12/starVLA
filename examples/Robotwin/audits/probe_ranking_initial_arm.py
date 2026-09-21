"""Test initial arm choice on all 50 held-out episodes per ranking task."""
import json
from pathlib import Path
import av
import cv2
import numpy as np
import pyarrow.parquet as pq
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.Robotwin.eval_files.model2robotwin_interface import ROBOTWIN_TO_MODEL_JOINT_ORDER

OUT = Path(__file__).resolve().parent / 'ranking_failure_20260907'


def arm_energy(chunk, initial):
    return [float(np.linalg.norm(chunk[:, s:s + 6] - initial[s:s + 6], axis=1).max()) for s in (0, 6)]


def main(task, port):
    root = Path('/data/gaoxiang/RoboTwinGenerated/Clean') / task
    rows = []
    client = WebsocketClientPolicy('127.0.0.1', port)
    try:
        for start in range(0, 1000, 100):
            examples, references, ep_ids = [], [], []
            for episode in range(start, start + 100, 20):
                actions = np.asarray(pq.read_table(root / f'data/chunk-000/episode_{episode:06d}.parquet',
                    columns=['action']).column('action').to_pylist(), dtype=np.float32)[:, ROBOTWIN_TO_MODEL_JOINT_ORDER]
                images = []
                for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
                    with av.open(str(root / f'videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4')) as video:
                        frame = next(video.decode(video=0))
                        images.append(cv2.resize(frame.to_ndarray(format='rgb24'), (224, 224), interpolation=cv2.INTER_AREA))
                examples.append(dict(image=images, state=actions[0], lang=task.replace('_', ' '), episode_start=True))
                references.append(actions)
                ep_ids.append(episode)
            response = client.predict_action(dict(examples=examples, unnorm_key='aloha', do_sample=False))
            pred = np.asarray(response['data']['actions'])
            for episode, p, reference in zip(ep_ids, pred, references):
                expected_energy = arm_energy(reference[1:17], reference[0])
                predicted_energy = arm_energy(p, reference[0])
                expected, predicted = int(np.argmax(expected_energy)), int(np.argmax(predicted_energy))
                rows.append(dict(episode=episode, expected_arm=expected, predicted_arm=predicted,
                    expected_motion=expected_energy, predicted_motion=predicted_energy,
                    correct=expected == predicted, joint_l1_rad=float(np.abs(p[:, :12] - reference[1:17, :12]).mean())))
            print(task, 'processed', len(rows), flush=True)
        summary = dict(task=task, count=len(rows), correct=sum(r['correct'] for r in rows),
            expected_left=sum(r['expected_arm'] == 0 for r in rows), predicted_left=sum(r['predicted_arm'] == 0 for r in rows),
            joint_l1_rad=float(np.mean([r['joint_l1_rad'] for r in rows])), rows=rows,
            method='argmax arm joint displacement norm in first 16 targets, all 50 held-out episodes at t=0; arm choice is not object selection or grasp success')
        (OUT / f'{task}_initial_arm.json').write_text(json.dumps(summary, indent=2) + '\n')
        print({k: v for k, v in summary.items() if k != 'rows'}, flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main('blocks_ranking_rgb', 5801)
    main('blocks_ranking_size', 5802)
