"""Read-only schema/numeric audit of generated LeRobot RoboTwin data."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pyarrow.parquet as pq


def audit_task(task):
    info = json.loads((task/'meta/info.json').read_text())
    episodes = [json.loads(line) for line in (task/'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
    expected = {e['episode_index']: e['length'] for e in episodes}
    files = sorted(task.glob('data/*/*.parquet'))
    cameras = [k for k in info['features'] if k.startswith('observation.images.')]
    errors, schemas, same, frames, stable, hashes = [], Counter(), 0, 0, True, {}
    mins = np.full(14, np.inf); maxs = np.full(14, -np.inf)
    continuous_gripper_frames = 0
    ids = set()
    for path in files:
        try:
            before = path.stat()
            ep = int(path.stem.split('_')[-1]); ids.add(ep)
            pf = pq.ParquetFile(path)
            names = pf.schema_arrow.names
            schemas[tuple(names)] += 1
            assert all(c in names for c in cameras), 'Missing camera column'
            t = pf.read(columns=['observation.state', 'action', 'frame_index', 'episode_index', 'timestamp'], use_threads=False)
            n = t.num_rows; assert n == expected[ep], 'Episode length mismatch'
            state = np.asarray(t['observation.state'].to_pylist(), dtype=np.float32)
            action = np.asarray(t['action'].to_pylist(), dtype=np.float32)
            assert state.shape == action.shape == (n, 14), 'Expected 14D state/action'
            assert np.isfinite(state).all() and np.isfinite(action).all(), 'Nonfinite state/action'
            assert np.array_equal(t['frame_index'].to_numpy(), np.arange(n)), 'Frame indices not consecutive'
            assert (t['episode_index'].to_numpy() == ep).all(), 'Episode index mismatch'
            timestamps = t['timestamp'].to_numpy()
            assert np.isfinite(timestamps).all() and (np.diff(timestamps)>0).all(), 'Invalid timestamps'
            same += int(np.array_equal(state, action)); frames += n
            mins = np.minimum(mins, action.min(0)); maxs = np.maximum(maxs, action.max(0))
            grippers = action[:, [6, 13]]
            continuous_gripper_frames += int(((grippers > .001) & (grippers < .999)).any(1).sum())
            digest = hashlib.sha256(action.tobytes()).hexdigest()
            hashes.setdefault(digest, []).append(ep)
            after = path.stat(); stable &= (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        except Exception as exc:
            errors.append({'file':str(path),'error':repr(exc)})
    if ids != set(expected): errors.append({'episode_ids_missing':sorted(set(expected)-ids),'extra':sorted(ids-set(expected))})
    if len(files) != info['total_episodes'] or frames != info['total_frames']: errors.append({'metadata_count_mismatch':True})
    decoded = 0
    for ep in sorted({min(expected), sorted(expected)[len(expected)//2], max(expected)}):
        path = task/info['data_path'].format(episode_chunk=ep//info['chunks_size'], episode_index=ep)
        try:
            t = pq.read_table(path, columns=cameras, use_threads=False)
            for i in sorted({0, len(t)//2, len(t)-1}):
                for camera in cameras:
                    rec = t[camera][i].as_py()
                    assert rec['bytes'], 'Missing inline bytes'
                    im = Image.open(io.BytesIO(rec['bytes'])).convert('RGB'); im.load()
                    assert im.size == (320,240), f'Unexpected image size {im.size}'
                    decoded += 1
        except Exception as exc: errors.append({'file':str(path),'image_error':repr(exc)})
    return dict(task=task.name,episodes=len(files),frames=frames,expected_frames=info['total_frames'],cameras=cameras,
                state_equals_action_episodes=same,numeric_check_passed=not errors,source_stable=stable,
                action_min=mins.tolist(),action_max=maxs.tolist(),continuous_gripper_frames=continuous_gripper_frames,
                duplicate_action_sequences=[v for v in hashes.values() if len(v)>1],sample_images_decoded=decoded,
                schemas=[{'columns':list(k),'episodes':v} for k,v in schemas.items()],errors=errors)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    started=time.time();tasks=sorted(d for d in args.root.iterdir() if (d/'meta/info.json').is_file());rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(audit_task,tasks):
            rows.append(row);print(row['task'],row['episodes'],row['frames'],'errors',len(row['errors']),flush=True)
    result=dict(root=str(args.root.resolve()),status='passed_schema_numeric_and_sample_images' if all(r['numeric_check_passed'] and r['source_stable'] for r in rows) else 'issues_found',
                tasks=len(rows),episodes=sum(r['episodes'] for r in rows),frames=sum(r['frames'] for r in rows),
                state_equals_action_episodes=sum(r['state_equals_action_episodes'] for r in rows),
                sample_images_decoded=sum(r['sample_images_decoded'] for r in rows),elapsed_seconds=time.time()-started,
                coverage='Every parquet numeric frame/schema checked; 3 episodes x 3 frames x all cameras decoded per task. No claim of full image decode or measured state availability.',results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))


if __name__=='__main__':main()
