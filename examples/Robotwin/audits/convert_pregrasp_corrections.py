"""Convert certified correction pilots, retaining real timestamps and pose labels."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from examples.Robotwin.audits.verify_rgb_recovery_sources import digest
from examples.Robotwin.data_preparation import convert_extracted
from starVLA.task_language import canonical_task_text

ROOT = Path(__file__).resolve().parents[3]
PHASES = ('align_pregrasp', 'descend_aligned', 'close', 'lift', 'hold')


def keep_latest_at_each_time(times):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise ValueError('Expected finite nondecreasing physical timestamps')
    # Action-boundary callbacks can capture the same physical instant twice.
    # Keep the latest phase at that instant, applying ONE mapping to every field.
    return np.flatnonzero(np.r_[np.diff(times) > 0, True])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert str(root).startswith('/data/gaoxiang/') and not str(root).startswith(str(ROOT))
    output = root/'RoboTwinGenerated/Clean/blocks_ranking_rgb'
    raw_index, labels = root/'raw_index', root/'labels'
    for path in (output, raw_index, labels, output.with_name('.blocks_ranking_rgb.tmp-convert')):
        assert not path.exists(), f'Refusing existing output {path}'
    status = json.loads((args.campaign/'status.json').read_text())
    assert status['state'] == 'complete' and all(r['exitcode'] == 0 for r in status['completed'])
    results = [json.loads((args.campaign/r['case']/'result.json').read_text()) for r in status['completed']]
    split_path = ROOT/'examples/Robotwin/audits/rgb_scene_safe_validation_20260907.json'
    split = json.loads(split_path.read_text())
    (raw_index/'data').mkdir(parents=True)
    (raw_index/'instructions').mkdir()
    labels.mkdir()
    records = []
    for episode, result in enumerate(results):
        assert result['state'] == 'raw_verified' and result['result']['first_attempt_success']
        assert result['source']['source_episode'] in split['train_episode_ids']
        assert result['split_sha256'] == digest(split_path)
        raw = Path(result['raw_output'])
        assert digest(result['hdf5']) == result['hdf5_sha256']
        assert digest(raw/'pregrasp_labels.json') == result['labels_sha256']
        document = json.loads((raw/'pregrasp_labels.json').read_text())
        rows = document['rows']
        times = np.asarray([r['sim_s'] for r in rows])
        keep = keep_latest_at_each_time(times)
        assert np.all(np.diff(times[keep]) > 0)
        # Validate before dropping any duplicate timestamps; no arbitrary filtering.
        assert len(rows) == result['frames'] and all(r['frame'] == i for i, r in enumerate(rows))
        assert set(r['phase'] for r in rows) == set(PHASES)
        with h5py.File(result['hdf5'], 'r') as source, h5py.File(raw_index/'data'/f'episode{episode}.hdf5', 'x') as dest:
            def copy(name, obj):
                if isinstance(obj, h5py.Group):
                    target = dest.require_group(name)
                else:
                    values = obj[()]
                    if obj.ndim and obj.shape[0] == len(rows):
                        values = values[keep]
                    elif name.startswith(('joint_action/', 'endpose/', 'observation/')):
                        raise ValueError(f'Unexpected non-frame-aligned raw field {name}: {obj.shape}')
                    target = dest.create_dataset(name, data=values, dtype=obj.dtype)
                for key, value in obj.attrs.items():
                    target.attrs[key] = value
            source.visititems(copy)
            assert dest['joint_action/vector'].shape == (len(keep), 14)
        selected = [rows[i] for i in keep]
        arm = ('left', 'right').index(result['arm'])
        np.savez_compressed(labels/f'episode_{episode:06d}.npz',
            raw_frame_index=keep, sim_s=times[keep],
            phase=np.asarray([PHASES.index(r['phase']) for r in selected], dtype=np.int64),
            active_arm=np.full(len(keep), arm, dtype=np.int64),
            measured_tcp_poses=np.asarray([r['actual_tcp_poses'] for r in selected], dtype=np.float32),
            measured_articulation_qpos=np.asarray([r['actual_articulation_qpos'] for r in selected], dtype=np.float32),
            actual_joint_names=np.asarray(document['actual_joint_names']),
            block_poses=np.asarray([r['block_poses'] for r in selected], dtype=np.float32),
            contacts=np.asarray([r['contacts'] for r in selected], dtype=bool),
            phase_goal_ee_pose=np.asarray([r['expert_goal_ee_pose'] for r in selected], dtype=np.float32),
            phase_goal_tcp_world_m=np.asarray([r['expert_goal_tcp_world_m'] for r in selected], dtype=np.float32),
            phase_goal_minus_measured_tcp_m=np.asarray([r['actual_tcp_to_phase_goal_delta_m'] for r in selected], dtype=np.float32),
            expert_grasp_ee_pose=np.repeat(np.asarray(result['expert_grasp_ee_pose'])[None],len(keep),axis=0),
            expert_pregrasp_ee_pose=np.repeat(np.asarray(result['expert_pregrasp_ee_pose'])[None],len(keep),axis=0))
        (raw_index/'instructions'/f'episode{episode}.json').write_text(json.dumps(
            {'seen': [canonical_task_text('blocks_ranking_rgb')]})+'\n')
        records.append(dict(episode_index=episode, source_episode=result['source']['source_episode'],
            scene_seed=result['source']['scene_seed'], arm=result['arm'], source_frames=len(rows),
            frames=len(keep), dropped_same_physics_time_frames=len(rows)-len(keep),
            raw_frame_mapping=keep.tolist(), source_hdf5=result['hdf5'], source_hdf5_sha256=result['hdf5_sha256'],
            labels_sha256=digest(labels/f'episode_{episode:06d}.npz'),
            physical_duration_s=float(times[keep][-1]-times[keep][0])))
    convert_extracted(raw_index, output, ROOT/'examples/Robotwin/train_files/modality.json',
                      task_name='blocks_ranking_rgb')
    audit_path = output/'meta/audit/conversion_audit.json'
    audit = json.loads(audit_path.read_text())
    audit.pop('control_hz', None)
    audit.update(state_semantics='joint drive targets, not measured qpos',
        action_semantics='same-record absolute drive targets; continuous_next loader applies +1',
        video_storage_fps=30, physics_dt_s=results[0]['physics_dt_s'], save_freq_physics_steps=15,
        physical_timestamps=str(labels),
        time_note='Video timestamps index video storage, not physical time. Exact sim_s is in label sidecars. '
                  'Zero-time boundary duplicates removed consistently from all modalities; no invented hold frames.',
        training_enabled=False, loader_audit_complete=False,
        normalization_note='Retain original Aloha normalization for model warm start; pilot statistics are descriptive only.')
    audit_path.write_text(json.dumps(audit, indent=2)+'\n')
    manifest = dict(state='converted_not_training_ready', dataset=str(output), labels=str(labels),
        raw_index=str(raw_index), phase_names=list(PHASES), episodes=records,
        original_split=str(split_path), original_split_sha256=digest(split_path),
        train_episode_ids=list(range(len(records))), validation_episode_ids=[], excluded_episode_ids=[],
        training_enabled=False, loader_audit_complete=False,
        note=f'{len(records)}-scene train-source correction set, not a trained model or validation set. '
             'Pose, phase, object and contact ground truth are supervision/audit only.')
    (root/'conversion_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.campaign/'converted_dataset.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'episodes'}, indent=2))


if __name__ == '__main__':
    main()
