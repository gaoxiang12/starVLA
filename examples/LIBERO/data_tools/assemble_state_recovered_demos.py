#!/usr/bin/env python3
"""Assemble validated state-render artifacts into separate LeRobot datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd


def signature(actions):
    canonical = np.round(np.asarray(actions, dtype=np.float32), 6)
    return hashlib.blake2b(f'{len(actions)}x7:'.encode() + canonical.astype('<f4').tobytes(), digest_size=20).hexdigest()


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--existing-audit', type=Path, required=True)
    parser.add_argument('--modality', type=Path, default=Path('examples/LIBERO/train_files/modality.json'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    expected = {(r['source_file'], r['source_demo']) for r in json.loads((args.inputs / 'manifest.json').read_text()) if r['needs_recovery']}
    records = {}
    for path in args.artifacts.glob('*/result.json'):
        record = json.loads(path.read_text())
        key = record['source_file'], record['source_demo']
        if key in records:
            raise ValueError(f'Duplicate source: {key}')
        records[key] = (path.parent, record)
    if set(records) != expected:
        raise ValueError(f'Incomplete artifacts: missing {len(expected - set(records))}, unexpected {len(set(records) - expected)}')
    existing = {r['action_signature'] for r in json.loads(args.existing_audit.read_text()) if not r['errors']}
    seen = set()
    report = {'datasets': {}, 'rejected': [], 'new_episodes': 0, 'new_frames': 0}
    for suite in sorted({key[0].split('/')[0] for key in records}):
        selected = [(path, r) for key, (path, r) in sorted(records.items()) if key[0].startswith(suite + '/')]
        name = suite + '_state_recovered_20260926_lerobot'
        target = args.output_root / name
        if target.exists():
            raise FileExistsError(target)
        meta = target / 'meta'
        meta.mkdir(parents=True)
        info = json.loads((args.inputs / (suite + '_template.json')).read_text())
        episodes = []
        provenance = []
        tasks = {}
        global_index = 0
        for artifact, record in selected:
            if not record['success']:
                report['rejected'].append(record)
                continue
            if not record.get('terminal_success') or not record.get('videos_fully_decoded'):
                raise ValueError('Unverified episode')
            with np.load(artifact / 'trajectory.npz') as trajectory:
                actions = trajectory['action']
                states = trajectory['state']
                source_indices = trajectory['source_frame_index'].tolist()
            n = len(actions)
            sig = signature(actions)
            if sig != record['action_signature'] or sig in existing or sig in seen:
                raise ValueError(f'Action mismatch or duplicate: {record}')
            if states.shape != (n, 8) or actions.shape != (n, 7) or not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError('Invalid episode arrays')
            seen.add(sig)
            index = len(episodes)
            chunk = index // 1000
            task_id = tasks.setdefault(record['language'], len(tasks))
            data_path = target / info['data_path'].format(episode_chunk=chunk, episode_index=index)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'observation.state': list(states), 'action': list(actions), 'timestamp': np.arange(n, dtype=np.float32) / 20, 'frame_index': np.arange(n, dtype=np.int64), 'episode_index': np.full(n, index, dtype=np.int64), 'index': np.arange(global_index, global_index + n, dtype=np.int64), 'task_index': np.full(n, task_id, dtype=np.int64)}).to_parquet(data_path, index=False)
            for short, key in [('image', 'observation.images.image'), ('wrist_image', 'observation.images.wrist_image')]:
                video_path = target / info['video_path'].format(episode_chunk=chunk, episode_index=index, video_key=key)
                video_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(artifact / (short + '.mp4'), video_path)
            episodes.append({'episode_index': index, 'tasks': [record['language']], 'length': n})
            provenance.append({**record, 'episode_index': index, 'source_frame_indices': source_indices})
            global_index += n
        info.update(total_episodes=len(episodes), total_frames=global_index, total_tasks=len(tasks), total_videos=2 * len(episodes), total_chunks=(len(episodes) + 999) // 1000, chunks_size=1000, fps=20, splits={'train': f'0:{len(episodes)}'})
        for feat in info['features'].values():
            if feat.get('dtype') == 'video':
                for key in ('info', 'video_info'):
                    if key in feat:
                        feat[key].update({'video.codec': 'h264', 'video.pix_fmt': 'yuv420p', 'video.fps': 20})
        (meta / 'info.json').write_text(json.dumps(info, indent=2) + '\n')
        shutil.copy2(args.modality, meta / 'modality.json')
        jsonl(meta / 'tasks.jsonl', [{'task_index': index, 'task': task} for task, index in tasks.items()])
        jsonl(meta / 'episodes.jsonl', episodes)
        jsonl(target / 'recovery_manifest.jsonl', provenance)
        report['datasets'][name] = {'episodes': len(episodes), 'frames': global_index, 'unique_language_labels': len(tasks)}
        report['new_episodes'] += len(episodes)
        report['new_frames'] += global_index
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
