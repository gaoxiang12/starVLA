"""Recovery conversion must preserve origins, episode boundaries, and split isolation."""
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import pytest

from examples.Robotwin.audits.convert_rgb_recovery_campaign import build_plan, convert_plan


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_campaign(tmp_path):
    campaign = tmp_path / 'campaign'
    split = tmp_path / 'split.json'
    write(split, dict(train_episode_ids=[7, 19], validation_episode_ids=[4], excluded_episode_ids=[]))
    seeds = tmp_path / 'seeds.txt'
    seeds.write_text('\n'.join(str(200+i) for i in range(20)))
    sources = tmp_path / 'sources.json'
    write(sources, dict(split=str(split), split_sha256=sha(split), seed_file=str(seeds),
        seed_file_sha256=sha(seeds), records=[dict(source_episode=e, scene_seed=200+e) for e in (7, 19)]))
    write(campaign / 'manifest.json', dict(sources=str(sources), source_manifest_sha256=sha(sources),
                                         checkpoint='fake-test-only.pt', checkpoint_sha256='test-digest'))
    cases = []
    commands_by_source = {}
    for episode in (19, 7):  # Intentionally collect in a different order from final local IDs.
        folder = tmp_path / 'raw' / str(episode)
        raw = folder / 'data/episode0.hdf5'
        raw.parent.mkdir(parents=True)
        n = 21 if episode == 7 else 23
        commands = np.zeros((n, 14), dtype=np.float32)
        commands[:, 0] = np.linspace(episode/100, episode/100+.1, n)
        commands[:, [6, 13]] = 1
        commands[5:15, 6] = 0
        commands_by_source[episode] = commands
        buffer = io.BytesIO()
        Image.new('RGB', (320, 240), (episode, 30, 40)).save(buffer, format='JPEG')
        with h5py.File(raw, 'w') as data:
            data['joint_action/vector'] = commands
            for side in ('left', 'right'):
                poses = np.zeros((n, 7), dtype=np.float32)
                poses[:, 0] = episode / 100
                poses[:, 2:4] = 1
                data[f'endpose/{side}_endpose'] = poses
            for camera in ('head_camera', 'left_camera', 'right_camera'):
                data[f'observation/{camera}/rgb'] = np.asarray([buffer.getvalue()] * n)
                intrinsic = np.array([[100,0,160], [0,100,120], [0,0,1]], dtype=np.float32)
                data[f'observation/{camera}/intrinsic_cv'] = np.repeat(intrinsic[None], n, 0)
                data[f'observation/{camera}/extrinsic_cv'] = np.repeat(np.eye(4, dtype=np.float32)[None,:3], n, 0)
                data[f'observation/{camera}/cam2world_gl'] = np.repeat(np.eye(4, dtype=np.float32)[None], n, 0)
        write(folder / 'frame_phases.json', ['handoff'] + ['expert_rgb_replan'] * (n-1))
        write(folder / 'recovery_provenance.json', dict(policy_prefix_is_training_target=False,
            source=dict(source_episode=episode), policy_checkpoint_sha256='test-digest',
            physics_timestep_s=float(np.float32(.004)), save_freq=15))
        result = dict(source_episode=episode, scene_seed=200+episode, state='raw_clip_verified',
            hdf5=str(raw), hdf5_sha256=sha(raw), recovery_frames=n, expert_assisted_success=True,
            final_arrangement=dict(all_success_predicates=True))
        write(campaign / f'cases/source_{episode:06d}/result.json', result)
        cases.append(result)
    write(campaign / 'collector_status.json', dict(state='complete', active_case=None,
        cases=cases, accepted_raw_clips=2))
    return campaign, commands_by_source


def test_two_sources_keep_actions_and_labels_with_local_reindexing(tmp_path):
    campaign, commands = fixture_campaign(tmp_path)
    plan = build_plan(campaign)
    assert [(r['episode_index'], r['source_episode']) for r in plan['episodes']] == [(0, 7), (1, 19)]
    output, labels, index = (tmp_path / name for name in ('converted', 'labels', 'raw_index'))
    converted = convert_plan(plan, output, labels, index)
    assert converted['train_episode_ids'] == [0, 1] and converted['validation_episode_ids'] == []
    for episode, source in ((0, 7), (1, 19)):
        table = pq.read_table(output / f'data/chunk-000/episode_{episode:06d}.parquet')
        np.testing.assert_array_equal(np.asarray(table['action'].to_pylist(), np.float32), commands[source])
        assert set(table['episode_index'].to_pylist()) == {episode}
        assert len(table) == len(commands[source])
        with np.load(labels / f'episode_{episode:06d}.npz') as spatial:
            assert len(spatial['xy']) == len(table)
            np.testing.assert_array_equal(spatial['priority_anchors'], [0])
            assert spatial['target_step'][0] == 5
            assert spatial['valid'][0].all()
            expected_x = (100 * (source/100+.12) + 160 + .5) / 320
            np.testing.assert_allclose(spatial['xy'][0, :, 0], expected_x, atol=1e-6, rtol=0)
        assert (index / f'data/episode{episode}.hdf5').resolve() == Path(plan['episodes'][episode]['source_hdf5'])


@pytest.mark.parametrize('failure', ['validation_leak', 'duplicate_source', 'changed_raw', 'unfinished', 'case_mismatch'])
def test_untrusted_or_incomplete_campaign_rejected(tmp_path, failure):
    campaign, _ = fixture_campaign(tmp_path)
    manifest = json.loads((campaign / 'manifest.json').read_text())
    status_path = campaign / 'collector_status.json'
    status = json.loads(status_path.read_text())
    if failure == 'validation_leak':
        sources_path = Path(manifest['sources'])
        sources = json.loads(sources_path.read_text())
        split_path = Path(sources['split'])
        write(split_path, dict(train_episode_ids=[19], validation_episode_ids=[4,7], excluded_episode_ids=[]))
        sources['split_sha256'] = sha(split_path)
        write(sources_path, sources)
        manifest['source_manifest_sha256'] = sha(sources_path)
        write(campaign / 'manifest.json', manifest)
    elif failure == 'duplicate_source':
        status['cases'][1] = status['cases'][0]
        write(status_path, status)
    elif failure == 'changed_raw':
        with h5py.File(status['cases'][0]['hdf5'], 'r+') as raw:
            raw['joint_action/vector'][0, 0] = 99
    elif failure == 'unfinished':
        status['state'] = 'running'
        write(status_path, status)
    else:
        path = campaign / 'cases/source_000007/result.json'
        result = json.loads(path.read_text())
        result['recovery_frames'] += 1
        write(path, result)
    with pytest.raises(AssertionError):
        build_plan(campaign)
