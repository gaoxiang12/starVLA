"""Locate expert first-lift prefix candidates, never certify grasp from TCP alone."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main():
    split_path = ROOT/'examples/Robotwin/audits/rgb_scene_safe_validation_20260907.json'
    split = json.loads(split_path.read_text())
    raw = Path('/data/gaoxiang/RoboTwinGenerated_raw/Clean/blocks_ranking_rgb/demo_clean/data')
    records, excluded = [], []
    for episode in split['train_episode_ids']:
        path = raw/f'episode{episode}.hdf5'
        if not path.exists():
            excluded.append(dict(episode=episode, reason='raw_recording_unavailable'))
            continue
        with h5py.File(path) as h:
            grips = np.stack([np.asarray(h[f'endpose/{a}_gripper']) for a in ('left', 'right')], axis=-1)
            closings = np.argwhere(grips < .2)
            if not len(closings):
                excluded.append(dict(episode=episode, reason='no_closing_command'))
                continue
            close, arm = map(int, closings[0])
            name = ('left', 'right')[arm]
            poses = np.asarray(h[f'endpose/{name}_endpose'], dtype=float)
            q = poses[:, 3:]/np.linalg.norm(poses[:, 3:], axis=1, keepdims=True)
            # Recorded endpose is EE; current Aloha TCP has +0.12 local-x offset.
            tcp_z = poses[:, 2] + .12 * 2 * (q[:, 1]*q[:, 3]-q[:, 0]*q[:, 2])
            releases = np.flatnonzero(grips[close+1:, arm] > .8)
            release = close+1+int(releases[0]) if len(releases) else len(grips)
            raised = np.flatnonzero(tcp_z[close:release]-tcp_z[close] >= .06)
            if not len(raised):
                excluded.append(dict(episode=episode, reason='no_6cm_tcp_rise_before_release'))
                continue
            endpoint = close+int(raised[0])
            records.append(dict(episode=episode, raw_path=str(path), source_frames=len(grips),
                first_close_frame=close, active_arm=name, first_release_frame=release,
                candidate_end_exclusive=endpoint+1, candidate_frames=endpoint+1,
                tcp_rise_m=float(tcp_z[endpoint]-tcp_z[close]),
                certified_object_lift=False, certified_hold_seconds=None))
    assert not set(r['episode'] for r in records) & set(split['validation_episode_ids'])
    report = dict(state='candidate_boundaries_only', training_enabled=False,
                  split=str(split_path), split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                  candidates=records, excluded=excluded,
                  summary=dict(candidates=len(records), excluded=len(excluded),
                    mean_source_frames=float(np.mean([r['source_frames'] for r in records])),
                    mean_candidate_frames=float(np.mean([r['candidate_frames'] for r in records]))),
                  limitations=['TCP rise is not object-lift or contact ground truth.',
                    'Raw recordings do not provide exact physical timestamps for every saved frame; no one-second hold claim is made.',
                    'Candidate clips require inspection or teacher phase capture before conversion/training.',
                    'Any conversion must truncate video, action, future targets together; +1 actions and +6/+12 futures need new tail masks.'])
    out=ROOT/'playground/Checkpoints/gawm_grasp_lift_preflight_20260909/prefix_candidates.json'
    with out.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
