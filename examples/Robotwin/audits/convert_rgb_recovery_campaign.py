"""Convert a completed recovery campaign while retaining every episode's origin."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from examples.Robotwin.audits.build_rgb_focus_labels import build
from examples.Robotwin.audits.verify_rgb_recovery_sources import digest
from examples.Robotwin.data_preparation import convert_extracted
from starVLA.task_language import canonical_task_text

ROOT = Path(__file__).resolve().parents[3]


def add_recovery_priority_anchors(plan, labels):
    """Mark expert correction starts without changing actions or dropping later frames."""
    rows = []
    correction_phases = {'handoff', 'open_empty_grippers', 'withdraw_after_empty_grasp', 'return_to_origin'}
    for record in plan['episodes']:
        phases = json.loads(Path(record['frame_phases']).read_text())
        assert len(phases) == record['frames']
        assert set(phases) <= correction_phases | {'expert_rgb_replan'}
        anchors = np.asarray([i for i, phase in enumerate(phases[:-1])
                              if phase in correction_phases], dtype=np.int64)
        path = labels / f"episode_{record['episode_index']:06d}.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        assert 'priority_anchors' not in arrays
        assert len(arrays['xy']) == len(phases)
        np.savez_compressed(path, **arrays, priority_anchors=anchors)
        rows.append(dict(episode_index=record['episode_index'], source_episode=record['source_episode'],
                         frames=len(phases), priority_anchors=len(anchors),
                         frame_phases_sha256=digest(record['frame_phases'])))
    (labels / 'priority_sampling.json').write_text(json.dumps(dict(
        correction_phases=sorted(correction_phases), episodes=rows,
        note='Candidate train anchor subset only; disabled unless priority_sampling_probability > 0. '
             'All ordinary anchors and spatial labels are retained.'), indent=2)+'\n')


def build_plan(campaign):
    manifest = json.loads((campaign / 'manifest.json').read_text())
    assert digest(manifest['sources']) == manifest['source_manifest_sha256']
    sources = json.loads(Path(manifest['sources']).read_text())
    assert digest(sources['split']) == sources['split_sha256']
    assert digest(sources['seed_file']) == sources['seed_file_sha256']
    split = json.loads(Path(sources['split']).read_text())
    seeds = [int(s) for s in Path(sources['seed_file']).read_text().split()]
    eligible = {r['source_episode']: r for r in sources['records']}
    collected = json.loads((campaign / 'collector_status.json').read_text())
    assert collected['state'] == 'complete' and collected['active_case'] is None
    assert len(collected['cases']) == len(eligible)
    seen, hashes, action_hashes, accepted, rejected = set(), set(), set(), [], []
    for case in sorted(collected['cases'], key=lambda r: r['source_episode']):
        episode = case['source_episode']
        assert episode in eligible and episode not in seen
        seen.add(episode)
        assert episode in split['train_episode_ids'] and episode not in split['validation_episode_ids']
        assert case['scene_seed'] == seeds[episode] == eligible[episode]['scene_seed']
        if case['state'] != 'raw_clip_verified':
            rejected.append(dict(source_episode=episode, scene_seed=case['scene_seed'], reason=case['state']))
            continue
        result_path = campaign / 'cases' / f'source_{episode:06d}' / 'result.json'
        result = json.loads(result_path.read_text())
        for key in ('source_episode', 'scene_seed', 'state', 'hdf5', 'hdf5_sha256', 'recovery_frames'):
            assert result[key] == case[key], f'Case/status mismatch: {episode}/{key}'
        assert result['expert_assisted_success'] and result['final_arrangement']['all_success_predicates']
        raw = Path(result['hdf5'])
        raw_hash = digest(raw)
        assert raw_hash == result['hdf5_sha256'] and raw_hash not in hashes
        hashes.add(raw_hash)
        provenance = json.loads((raw.parent.parent / 'recovery_provenance.json').read_text())
        assert provenance['policy_prefix_is_training_target'] is False
        assert provenance['source']['source_episode'] == episode
        assert provenance['policy_checkpoint_sha256'] == manifest['checkpoint_sha256']
        assert abs(provenance['physics_timestep_s']-.004) < 1e-9 and provenance['save_freq'] == 15
        phases = json.loads((raw.parent.parent / 'frame_phases.json').read_text())
        assert phases[0] == 'handoff' and len(phases) == result['recovery_frames']
        with h5py.File(raw) as data:
            commands = np.asarray(data['joint_action/vector'], dtype=np.float32)
            assert commands.shape == (len(phases), 14) and len(commands) > 16
            assert np.isfinite(commands).all()
            action_hash = hashlib.sha256(commands.tobytes()).hexdigest()
            assert action_hash not in action_hashes, 'Duplicate expert command sequence'
            action_hashes.add(action_hash)
            for camera in ('head_camera', 'left_camera', 'right_camera'):
                assert len(data[f'observation/{camera}/rgb']) == len(commands)
                for field in ('intrinsic_cv', 'extrinsic_cv', 'cam2world_gl'):
                    matrix = data[f'observation/{camera}/{field}'][:]
                    assert len(matrix) == len(commands) and np.isfinite(matrix).all()
            for side in ('left', 'right'):
                poses = data[f'endpose/{side}_endpose'][:]
                assert poses.shape == (len(commands), 7) and np.isfinite(poses).all()
                assert (np.linalg.norm(poses[:, 3:], axis=-1) > .99).all()
        accepted.append(dict(episode_index=len(accepted), source_episode=episode,
            scene_seed=case['scene_seed'], source_hdf5=str(raw.resolve()), source_hdf5_sha256=raw_hash,
            action_sha256=action_hash, frames=len(commands), case_result=str(result_path.resolve()),
            physics_timestep_s=provenance['physics_timestep_s'], save_freq=provenance['save_freq'],
            frame_phases=str((raw.parent.parent / 'frame_phases.json').resolve())))
    assert len(accepted) == collected['accepted_raw_clips'] and accepted
    return dict(state='raw_plan_verified', campaign=str(campaign.resolve()),
        source_manifest=manifest['sources'], source_manifest_sha256=manifest['source_manifest_sha256'],
        original_split=sources['split'], original_split_sha256=sources['split_sha256'],
        policy_checkpoint=manifest['checkpoint'], policy_checkpoint_sha256=manifest['checkpoint_sha256'],
        episodes=accepted, rejected=rejected, frames=sum(r['frames'] for r in accepted),
        train_episode_ids=list(range(len(accepted))), validation_episode_ids=[], excluded_episode_ids=[],
        training_enabled=False, loader_audit_complete=False,
        note='Train-only supplement. Numbered raw symlinks preserve conversion/label alignment; original validation is separate. Videos still require full conversion/loader audit.')


def convert_plan(plan, output, labels, raw_index):
    for path in (output, labels, raw_index, output.with_name(f'.{output.name}.tmp-convert'),
                 output.with_name(f'.{output.name}.previous')):
        assert not path.exists(), f'Refusing existing conversion destination: {path}'
    (raw_index / 'data').mkdir(parents=True)
    (raw_index / 'instructions').mkdir()
    for record in plan['episodes']:
        episode = record['episode_index']
        # Hash again immediately before constructing the conversion source index.
        assert digest(record['source_hdf5']) == record['source_hdf5_sha256']
        (raw_index / 'data' / f'episode{episode}.hdf5').symlink_to(record['source_hdf5'])
        (raw_index / 'instructions' / f'episode{episode}.json').write_text(
            json.dumps({'seen': [canonical_task_text('blocks_ranking_rgb')]}) + '\n')
    (raw_index / 'source_mapping.json').write_text(json.dumps(plan, indent=2) + '\n')
    convert_extracted(raw_index, output, ROOT / 'examples/Robotwin/train_files/modality.json',
                      task_name='blocks_ranking_rgb')
    build(SimpleNamespace(dataset=output, raw=raw_index / 'data', output=labels))
    add_recovery_priority_anchors(plan, labels)
    audit_path = output / 'meta/audit/conversion_audit.json'
    audit = json.loads(audit_path.read_text())
    audit.pop('control_hz')
    audit.update(state_semantics='joint drive targets, not measured qpos',
        action_semantics='same-record absolute drive targets; continuous_next loader applies +1',
        video_storage_fps=30, physics_timestep_s=.004, save_freq_physics_steps=15,
        time_note='Storage FPS does not establish physical control rate; phase boundaries can duplicate records.')
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    plan = dict(plan, state='converted_labels_ready', dataset=str(output.resolve()),
                labels=str(labels.resolve()), raw_index=str(raw_index.resolve()))
    (output / 'meta/audit/recovery_source_mapping.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--plan-output', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--raw-index', type=Path)
    args = parser.parse_args()
    assert not args.plan_output.exists()
    plan = build_plan(args.campaign.resolve())
    destinations = (args.output, args.labels, args.raw_index)
    if any(p is not None for p in destinations):
        assert all(p is not None for p in destinations), 'Conversion needs output, labels, and raw-index together'
        plan = convert_plan(plan, *(p.resolve() for p in destinations))
    args.plan_output.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({k: v for k, v in plan.items() if k not in ('episodes', 'rejected')}), flush=True)


if __name__ == '__main__':
    main()
