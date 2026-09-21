"""Select spread-out existing training episode IDs for independent recovery collection."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and args.count > 0
    source = json.loads(args.template.read_text())
    assert digest(source['split']) == source['split_sha256']
    assert digest(source['seed_file']) == source['seed_file_sha256']
    split = json.loads(Path(source['split']).read_text())
    scenes = json.loads(Path(source['scene_audit']).read_text())
    assert not scenes['errors']
    points = {r['episode']: np.asarray([r['initial'][c]['xy'] for c in ('red', 'green', 'blue')])
              for r in scenes['episodes_detail'] if set(r['initial']) == {'red', 'green', 'blue'}}
    seeds = [int(s) for s in Path(source['seed_file']).read_text().split()]
    prior = {r['source_episode'] for r in source['records']}
    raw_dir = Path(source['records'][0]['source_hdf5']).parent
    dataset = Path(split['dataset'])
    validation = sorted(split['validation_episode_ids'])
    assert set(validation).issubset(points)
    candidates = sorted(e for e in split['train_episode_ids'] if e < len(seeds) and e not in prior
                        and e in points and (raw_dir / f'episode{e}.hdf5').is_file())
    assert len(candidates) >= args.count
    # Uniform coverage over available source episode indices, no selection on model outcomes.
    selected = [candidates[i] for i in np.linspace(0, len(candidates)-1, args.count, dtype=int)]
    records = []
    for episode in selected:
        distances = [float(np.linalg.norm(points[episode]-points[v], axis=-1).max()) for v in validation]
        nearest = int(np.argmin(distances))
        assert distances[nearest] > 1.5
        for previous in records:
            assert np.linalg.norm(points[episode]-points[previous['source_episode']], axis=-1).max() > 1.5
        raw_path = raw_dir / f'episode{episode}.hdf5'
        parquet = dataset / f'data/chunk-000/episode_{episode:06d}.parquet'
        action = np.asarray(pq.read_table(parquet, columns=['action'])['action'].to_pylist())
        with h5py.File(raw_path) as raw:
            original = raw['joint_action/vector'][:]
        assert original.shape == action.shape and original.shape[1] == 14
        assert np.isfinite(original).all() and np.isfinite(action).all()
        np.testing.assert_allclose(original, action, atol=1e-6, rtol=0)
        records.append(dict(source_episode=episode, scene_seed=seeds[episode], source_hdf5=str(raw_path),
                            source_hdf5_sha256=digest(raw_path), source_parquet=str(parquet),
                            frames=len(action), raw_parquet_action_max_difference=float(np.abs(original-action).max()),
                            nearest_validation_episode=validation[nearest],
                            nearest_validation_max_block_center_distance_px=distances[nearest],
                            requires_runtime_initial_scene_match=True))
        print('verified source', episode, flush=True)
    source.update(state='offline_sources_verified_runtime_pending', records=records,
                  selection='Evenly spaced over eligible original training episode IDs; excludes initial pilot sources. No selection using policy outcomes.',
                  scene_audit_sha256=digest(source['scene_audit']),
                  note='Recovery collection candidates only; each actual scene must still pass runtime verify_source. No evaluation data, no recovery training yet.')
    args.output.write_text(json.dumps(source, indent=2) + '\n')
    print('prepared', selected, flush=True)


if __name__ == '__main__':
    main()
