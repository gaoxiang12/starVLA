"""Audit-aligned TCP waypoint supervision; labels never enter inference inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def gripper_events(actions):
    events = []
    for arm, column in enumerate((6, 13)):
        grip = actions[:, column]
        closed = grip < .2
        opened = grip > .8
        for t in np.flatnonzero(closed[1:] & ~closed[:-1]) + 1:
            events.append((int(t), arm, 0))
        for t in np.flatnonzero(opened[1:] & ~opened[:-1]) + 1:
            events.append((int(t), arm, 1))
    return sorted(events)


def tcp_positions(poses):
    # Robot._trans_endpose(is_endpose=False) subtracts 0.12 along local X.
    # Convert recorded EE back to TCP in world coordinates; quaternion is wxyz.
    q = poses[:, 3:7]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q.T
    x_axis = np.stack([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)], axis=-1)
    return poses[:, :3] + .12 * x_axis


def project(points, intrinsic, extrinsic):
    camera = np.einsum('nij,nj->ni', extrinsic[:, :, :3], points) + extrinsic[:, :, 3]
    pixel = np.einsum('nij,nj->ni', intrinsic, camera)
    uv = pixel[:, :2] / np.maximum(pixel[:, 2:3], 1e-8)
    # Pixel centres in [0,1], consistent with align_corners=False crops.
    xy = (uv + .5) / np.array([320., 240.])
    valid = (camera[:, 2] > .01) & np.isfinite(xy).all(-1) & ((xy > 0) & (xy < 1)).all(-1)
    return np.clip(xy, 0, 1).astype(np.float32), valid


def build(args):
    import pyarrow.parquet as pq

    args.output.mkdir(parents=True, exist_ok=True)
    episodes = [json.loads(line) for line in
                (args.dataset / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
    episode_ids = [int(row['episode_index']) for row in episodes]
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError('Expected nonempty, unique episode IDs in metadata')
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    rows = []
    for record in episodes:
        episode = int(record['episode_index'])
        path = info['data_path'].format(episode_chunk=episode // info['chunks_size'], episode_index=episode)
        table = pq.read_table(args.dataset / path, columns=['action'])
        action = np.array(table['action'].to_pylist(), dtype=np.float32)
        n = len(action)
        if n != int(record['length']):
            raise ValueError(f'Episode {episode} metadata/action length mismatch')
        events = gripper_events(action)
        event_anchors = set(range(min(16, n-1)))
        for t, _, _ in events:
            event_anchors.update(range(max(0,t-20), min(n-1,t+13)))
        xy = np.zeros((n,3,2), np.float32)
        valid = np.zeros((n,3), bool)
        arm_label = np.full(n, -1, np.int64)
        kind_label = np.full(n, -1, np.int64)
        target_steps = np.full(n, -1, np.int64)
        raw_path = args.raw / f'episode{episode}.hdf5'
        aligned = False
        if raw_path.exists():
            with h5py.File(raw_path) as raw:
                source = np.asarray(raw['joint_action/vector'], dtype=np.float32)
                if source.shape != action.shape or not np.allclose(source, action, atol=1e-6, rtol=0):
                    raise ValueError(f'Raw/LeRobot mismatch: {episode}; refusing geometrical labels')
                aligned = True
                tcp = [tcp_positions(np.asarray(raw[f'endpose/{side}_endpose'])) for side in ('left','right')]
                for t in range(n):
                    upcoming = [e for e in events if e[0] >= t]
                    if upcoming:
                        target, arm, kind = upcoming[0]
                        # Beyond 120 recorded samples, do not force a distant waypoint.
                        if target-t <= 120:
                            arm_label[t], kind_label[t], target_steps[t] = arm, kind, target
                has = target_steps >= 0
                points = np.zeros((n,3))
                for arm in (0,1):
                    sel = has & (arm_label == arm)
                    points[sel] = tcp[arm][target_steps[sel]]
                for v, camera in enumerate(('head_camera','left_camera','right_camera')):
                    g = raw[f'observation/{camera}']
                    xy[:,v], valid[:,v] = project(points, np.asarray(g['intrinsic_cv']), np.asarray(g['extrinsic_cv']))
                    valid[:,v] &= has
        np.savez_compressed(args.output / f'episode_{episode:06d}.npz',
                            xy=xy, valid=valid, arm=arm_label, kind=kind_label,
                            target_step=target_steps, event_anchors=np.array(sorted(event_anchors),np.int64))
        rows.append(dict(episode=episode, frames=n, raw_aligned=aligned,
                         valid_view_labels=int(valid.sum()), labeled_frames=int(valid.any(-1).sum()),
                         events=events, action_sha256=hashlib.sha256(action.tobytes()).hexdigest()))
        if episode % 100 == 0:
            print('labels',episode,flush=True)
    manifest = dict(dataset=str(args.dataset.resolve()), raw=str(args.raw.resolve()),
                    label='next_gripper_event_world_TCP_projected_in_current_camera',
                    tool_offset_m=.12, image_wh=[320,240],
                    visibility='positive depth and in-frame only; not an occlusion test',
                    total_episodes=len(rows), aligned_raw_episodes=sum(r['raw_aligned'] for r in rows),
                    labeled_frames=sum(r['labeled_frames'] for r in rows), episodes=rows)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print({k:v for k,v in manifest.items() if k!='episodes'},flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('/data/gaoxiang/RoboTwinGenerated/Clean/blocks_ranking_rgb'))
    parser.add_argument('--raw', type=Path, default=Path('/data/gaoxiang/RoboTwinGenerated_raw/Clean/blocks_ranking_rgb/demo_clean/data'))
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args())
