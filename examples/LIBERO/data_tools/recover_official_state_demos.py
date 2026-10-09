#!/usr/bin/env python3
"""Render missing official demonstrations from recorded pre-action states.

Consumes audited HDF5 input packs with actions, states, source model XML, and
provenance. Writes isolated episode artifacts; no training dataset is modified.
Uses source states, never benchmark evaluation initial states. Unlike action
replay, this preserves the recorded trajectory despite simulator drift.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import av
import h5py
import imageio
import numpy as np
import robosuite
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle


def keep_indices(actions):
    previous = None
    kept = []
    for i, action in enumerate(actions):
        if np.linalg.norm(action[:-1]) < 1e-4 and (previous is None or action[-1] == previous[-1]):
            continue
        kept.append(i)
        previous = action
    return np.asarray(kept, dtype=np.int64)


def signature(actions):
    canonical = np.round(np.asarray(actions, dtype=np.float32), 6)
    return hashlib.blake2b(f'{len(actions)}x7:'.encode() + canonical.astype('<f4').tobytes(), digest_size=20).hexdigest()


def source_xml(value):
    tree = ET.fromstring(value)
    for elem in tree.iter():
        name = elem.get('file')
        if name and '/assets/' in name:
            prefix = Path(robosuite.__file__).parent / 'models/assets' if '/robosuite/' in name else Path(get_libero_path('assets'))
            target = prefix / name.split('/assets/', 1)[1]
            if not target.is_file():
                raise FileNotFoundError(target)
            elem.set('file', str(target))
    return ET.tostring(tree, encoding='unicode')


def write_video(path, frames):
    imageio.mimwrite(path, frames, fps=20, codec='libx264', format='FFMPEG', pixelformat='yuv420p', macro_block_size=1, output_params=['-crf', '18', '-threads', '1'])
    with av.open(str(path)) as container:
        decoded = 0
        for frame in container.decode(video=0):
            if (frame.width, frame.height) != (256, 256):
                raise ValueError('Wrong video dimensions')
            decoded += 1
    if decoded != len(frames):
        raise ValueError(f'Video frame count mismatch: {decoded}/{len(frames)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--controls-only', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = json.loads((args.inputs / 'manifest.json').read_text())
    records = [r for r in records if bool(r.get('control')) == args.controls_only]
    records = records[args.shard::args.shards]
    if args.retry_errors:
        retry_records = []
        for record in records:
            key = hashlib.sha256((record['source_file'] + ':' + record['source_demo']).encode()).hexdigest()[:20]
            result_path = args.output / key / 'result.json'
            if result_path.exists() and json.loads(result_path.read_text()).get('error'):
                retry_records.append(record)
        records = retry_records
    if args.limit:
        records = records[:args.limit]
    env = None
    current_task = None
    suites = {}
    handles = {}
    started = time.time()
    for ordinal, record in enumerate(records):
        key = hashlib.sha256((record['source_file'] + ':' + record['source_demo']).encode()).hexdigest()[:20]
        out = args.output / key
        out.mkdir(exist_ok=True)
        manifest_path = out / 'result.json'
        if manifest_path.exists():
            previous_result = json.loads(manifest_path.read_text())
            if not args.retry_errors or not previous_result.get('error'):
                continue
            (out / 'first_attempt.json').write_text(json.dumps(previous_result, indent=2) + '\n')
        result = {**record, 'method': 'official_pre_action_state_render', 'source_revision': 'f13aa24a3da8c43c7225569f28c562979fa0e35a', 'seed': 7, 'uses_fixed_evaluation_initial_states': False, 'success': False}
        try:
            suite_name, source = record['source_file'].split('/')
            if suite_name not in suites:
                suite = benchmark.get_benchmark_dict()[suite_name]()
                suites[suite_name] = {suite.get_task(i).name: suite.get_task(i) for i in range(suite.n_tasks)}
                handles[suite_name] = h5py.File(args.inputs / (suite_name + '.hdf5'), 'r')
            task_name = source.removesuffix('_demo.hdf5')
            task = suites[suite_name][task_name]
            if task.language.strip().lower() != record['language'].strip().lower():
                raise ValueError('Task language mismatch')
            demo = handles[suite_name][Path(source).stem][record['source_demo']]
            bddl_path = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
            bddl_text = bddl_path.read_text()
            xml_text = str(demo.attrs['model_file'])
            if 'new_salad_dressing_1' in bddl_text and 'salad_dressing_1_main' in xml_text and 'new_salad_dressing_1_main' not in xml_text:
                compatibility_dir = args.output.parent / 'source_bddl_compatibility'
                compatibility_dir.mkdir(exist_ok=True)
                bddl_path = compatibility_dir / task.bddl_file
                bddl_text = bddl_text.replace('new_salad_dressing', 'salad_dressing')
                temp_bddl = bddl_path.with_suffix(f'.{args.shard}.tmp')
                temp_bddl.write_text(bddl_text)
                temp_bddl.replace(bddl_path)
                result['source_object_compatibility'] = 'Use original salad_dressing object type from recorded model XML instead of later new_salad_dressing replacement'
                result['bddl_sha256'] = hashlib.sha256(bddl_text.encode()).hexdigest()
            if current_task != (suite_name, task_name):
                if env is not None:
                    env.close()
                env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256, render_gpu_device_id=args.gpu)
                env.seed(7)
                env.reset()
                current_task = (suite_name, task_name)
            demo = handles[suite_name][Path(source).stem][record['source_demo']]
            actions = demo['actions'][:]
            source_states = demo['states'][:]
            if not np.isfinite(actions).all() or not np.isfinite(source_states).all() or len(actions) != len(source_states):
                raise ValueError('Invalid source state/action data')
            env.reset_from_xml_string(source_xml(demo.attrs['model_file']))
            env.regenerate_obs_from_state(source_states[-1])
            terminal_success = bool(env.check_success())
            if not terminal_success:
                _, _, terminal_success, _ = env.step(actions[-1])
                terminal_success = bool(terminal_success)
            result['terminal_success'] = terminal_success
            if not terminal_success:
                result['rejection'] = 'official terminal state failed current environment success predicate'
            else:
                kept = keep_indices(actions)
                output_actions = actions[kept].astype(np.float32)
                output_actions[:, -1] = (1 - output_actions[:, -1]) / 2
                if signature(output_actions) != record['action_signature']:
                    raise ValueError('Action signature mismatch')
                states = []
                main_images = []
                wrist_images = []
                for idx in kept:
                    obs = env.regenerate_obs_from_state(source_states[idx])
                    state = np.r_[obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']].astype(np.float32)
                    if state.shape != (8,) or not np.isfinite(state).all():
                        raise ValueError('Invalid rendered proprioception')
                    states.append(state)
                    main_images.append(np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]))
                    wrist_images.append(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1]))
                np.savez_compressed(out / 'trajectory.npz', state=np.stack(states), action=output_actions, source_frame_index=kept)
                write_video(out / 'image.mp4', main_images)
                write_video(out / 'wrist_image.mp4', wrist_images)
                result.update(success=True, frames=len(kept), resolution=[256, 256], videos_fully_decoded=True)
        except Exception as exc:
            result['error'] = repr(exc)
            # Recreate the environment on the next record after any partial failure.
            current_task = None
        temp = out / 'result.json.tmp'
        temp.write_text(json.dumps(result, indent=2) + '\n')
        temp.replace(manifest_path)
        print(json.dumps({'shard': args.shard, 'completed': ordinal + 1, 'assigned': len(records), 'source': record['source_file'], 'demo': record['source_demo'], 'success': result['success'], 'error': result.get('error'), 'rejection': result.get('rejection'), 'elapsed_seconds': round(time.time() - started)}), flush=True)
    if env is not None:
        env.close()
    for handle in handles.values():
        handle.close()


if __name__ == '__main__':
    main()
