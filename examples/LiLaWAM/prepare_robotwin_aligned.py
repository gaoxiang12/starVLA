"""Audit the existing 50-task three-view export, without copying its images.

Recompute native-order numeric statistics on ALL local training episodes.
Reuse the author's task VTT assets explicitly, rather than silently recomputing
them on a different camera/data distribution. No policy weights are imported.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq
from PIL import Image

from examples.LiLaWAM.prepare import stats, write_json
from examples.Robotwin.eval_files.summarize_robotwin_eval import ALL_TASKS
from starVLA.task_language import resolve_task_language


def prepare(config, source):
    cfg = OmegaConf.load(config)
    data, model = cfg.datasets.vla_data, cfg.framework.lila
    output = Path(model.task_vectors_path).parent
    output.mkdir(parents=True, exist_ok=True)
    audit = dict(status='running', datasets=[], episodes=0, frames=0,
                 image_audit='first and last frame of first episode, all three views, each task/split',
                 numeric_audit='every frame of every episode',
                 policy_weights_loaded=False, vtt_source=str(source),
                 differences=['different trajectories', 'three cameras', '14D command state, not endpose',
                              'FP32 CPU normalization before bf16 cast, rather than bf16 GPU normalization'])
    write_json(output/'preparation_audit.json', audit)
    action_parts, state_parts = [], []
    for split in ('Clean', 'Randomized'):
        for task in ALL_TASKS:
            root = Path(data.data_root_dir)/'RoboTwin'/split/task
            meta = root/'meta'
            info = json.loads((meta/'info.json').read_text())
            modalities = json.loads((meta/'modality.json').read_text())
            expected_video = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
            if list(modalities['video']) != expected_video:
                raise ValueError(f'{root}: camera ordering differs')
            # The new DataConfig reads native-order arm/gripper slices.
            for modality, column in [('action', 'action'), ('state', 'observation.state')]:
                for key, start, end in [('left_joints',0,6),('left_gripper',6,7),
                                        ('right_joints',7,13),('right_gripper',13,14)]:
                    m = modalities[modality][key]
                    if (m.get('original_key'), m['start'], m['end']) != (column,start,end):
                        raise ValueError(f'{root}: unexpected {modality}/{key} mapping')
            episodes = [json.loads(s) for s in (meta/'episodes.jsonl').read_text().splitlines()]
            if len(episodes) != info['total_episodes']:
                raise ValueError(f'{root}: episode count mismatch')
            count = 0
            hashes = hashlib.sha256()
            for row in episodes:
                ep = int(row['episode_index'])
                path = root/info['data_path'].format(episode_chunk=ep//info['chunks_size'], episode_index=ep)
                table = pq.read_table(path, columns=['action','observation.state','frame_index'])
                action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
                state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
                if action.shape != (row['length'],14) or state.shape != action.shape:
                    raise ValueError(f'{path}: numeric shape mismatch')
                if not np.isfinite(action).all() or not np.isfinite(state).all():
                    raise ValueError(f'{path}: nonfinite state/actions')
                if not np.array_equal(table['frame_index'].to_numpy(), np.arange(len(action))):
                    raise ValueError(f'{path}: nonsequential frames')
                for array in (action, state):
                    if ((array[:,[6,13]] < -1e-5) | (array[:,[6,13]] > 1+1e-5)).any():
                        raise ValueError(f'{path}: invalid continuous gripper')
                    hashes.update(array.tobytes())
                action_parts.append(action); state_parts.append(state); count += len(action)
                if ep == int(episodes[0]['episode_index']):
                    columns = [modalities['video'][v]['original_key'] for v in expected_video]
                    images = pq.read_table(path, columns=columns)
                    for column in columns:
                        for index in (0, len(action)-1):
                            value = images[column][index].as_py()
                            with Image.open(io.BytesIO(value['bytes'])) as image:
                                image.convert('RGB').load()
            if count != info['total_frames']:
                raise ValueError(f'{root}: frame count mismatch')
            # Rehydrate missing standard metadata from the existing raw cache.
            # The actual training bounds below are recomputed, not this cache.
            stats_path = meta/'stats.json'
            if not stats_path.exists():
                cached = json.loads((meta/'stats_gr00t.json').read_text())
                raw = cached.get('statistics', cached)
                if not all(key in raw for key in ('action','observation.state')):
                    raise ValueError(f'{root}: invalid statistics cache')
                write_json(stats_path, raw)
            audit['datasets'].append(dict(path=str(root), episodes=len(episodes), frames=count,
                numeric_sha256=hashes.hexdigest(), metadata_sha256={name:hashlib.sha256((meta/name).read_bytes()).hexdigest()
                for name in ('info.json','modality.json','episodes.jsonl','tasks.jsonl','stats.json','stats_gr00t.json')}))
            audit['episodes'] += len(episodes); audit['frames'] += count
            write_json(output/'preparation_audit.json', audit)
            print(f'{split}/{task}: {len(episodes)} episodes, {count} frames', flush=True)
    if (audit['frames'],audit['episodes']) != (cfg.alignment.expected_frames,cfg.alignment.expected_episodes):
        raise ValueError('Dataset inventory changed; recompute epoch/step budget before training')
    statistics = {'aloha': {'action':stats(np.concatenate(action_parts)),
                            'state':stats(np.concatenate(state_parts))}}
    write_json(data.normalization_statistics_path, statistics)
    vectors, vtt_hashes = {}, {}
    for task in ALL_TASKS:
        path = Path(source)/task/'task_cond.npy'
        value = np.load(path, allow_pickle=False)
        if value.shape != (1024,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid DINO-L VTT: {path}')
        language = resolve_task_language('',task,'dataset_name')
        vectors[f'aloha:{language}'] = value.tolist()
        vtt_hashes[task] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(model.task_vectors_path, dict(format_version=1,vectors=vectors,source=str(source),sha256=vtt_hashes))
    audit.update(status='prepared', steps_per_epoch=audit['frames']//128,
                 global_batch_size=128, task_count=len(vectors), vtt_sha256=vtt_hashes)
    write_json(output/'preparation_audit.json', audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('datasets','vtt_sha256')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='examples/LiLaWAM/train_files/robotwin_3view_aligned_stage1.yaml')
    parser.add_argument('--task-cond-dir', default='/data/gaoxiang/ckpts/lila_starvla/lila_robotwin_official_ddp8_20260916/upstream_source/data-500-taskcond')
    args = parser.parse_args()
    prepare(args.config, args.task_cond_dir)
