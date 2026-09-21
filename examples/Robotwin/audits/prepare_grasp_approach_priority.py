"""Prepare an isolated approach-priority ablation from audited train phases."""
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[3]
    audit = root / 'playground/Checkpoints/gawm_grasp_precision_training_20260909'
    converted = Path('/data/gaoxiang/RoboTwinPregraspCorrections_20260909/train20_converted')
    output = converted.parent / 'approach_priority_labels_v1'
    assert not output.exists()
    conversion = json.loads((converted / 'conversion_manifest.json').read_text())
    spatial_manifest = json.loads((converted / 'spatial_labels/manifest.json').read_text())
    split = json.loads((audit / 'split.json').read_text())
    assert conversion['phase_names'] == ['align_pregrasp', 'descend_aligned', 'close', 'lift', 'hold']
    assert len(conversion['episodes']) == 20
    prepared = []
    for row in conversion['episodes']:
        episode = row['episode_index']
        assert row['source_episode'] in split['train_episode_ids']
        assert row['source_episode'] not in split['validation_episode_ids']
        phase_path = converted / 'labels' / f'episode_{episode:06d}.npz'
        assert digest(phase_path) == row['labels_sha256']
        spatial_path = converted / 'spatial_labels' / phase_path.name
        with np.load(phase_path, allow_pickle=False) as phases, np.load(spatial_path, allow_pickle=False) as source:
            labels = {key: source[key].copy() for key in source.files}
            assert len(labels['xy']) == len(phases['phase']) == row['frames']
            priority = np.flatnonzero(np.isin(phases['phase'], [0, 1])).astype(np.int64)
            assert len(priority) > 0 and priority.max() < row['frames'] - 16
            assert np.all(labels['kind'][priority] == 0)
            assert np.all(labels['arm'][priority] == ('left', 'right').index(row['arm']))
            assert labels['valid'][priority].any(axis=1).all()
            assert 'priority_anchors' not in labels
            labels['priority_anchors'] = priority
        prepared.append((phase_path.name, labels, dict(episode=episode,
            source_episode=row['source_episode'], scene_seed=row['scene_seed'],
            priority_anchors=priority.tolist(), phase_labels_sha256=digest(phase_path),
            original_spatial_labels_sha256=digest(spatial_path))))
    # Validate every source before creating new artifacts. Existing arrays remain identical.
    output.mkdir()
    records = []
    for name, arrays, record in prepared:
        np.savez_compressed(output / name, **arrays)
        with np.load(output / name, allow_pickle=False) as reread:
            for key, value in arrays.items():
                np.testing.assert_array_equal(value, reread[key])
        record['output_sha256'] = digest(output / name)
        records.append(record)
    spatial_manifest.update(priority_semantics='Current observation in align_pregrasp or descend_aligned; train sources only',
                            priority_records=records,
                            source_manifest_sha256=digest(converted / 'spatial_labels/manifest.json'))
    (output / 'manifest.json').write_text(json.dumps(spatial_manifest, indent=2) + '\n')
    correction_key = 'RoboTwinPregraspCorrections_20260909/train20_converted/RoboTwinGenerated/Clean/blocks_ranking_rgb'
    configs = []
    for variant, base in [('joint', 'joint_train1000.yaml'), ('cartesian', 'cartesian_train1000_r2.yaml')]:
        config = OmegaConf.load(audit / base)
        config.run_id = f'gawm_grasp_precision_{variant}_approach_priority_1000_20260909'
        options = config.datasets.vla_data.dataset_options[correction_key]
        options.spatial_supervision_dir = str(output)
        options.priority_sampling_probability = .7
        target = audit / f'{variant}_approach_priority_train1000.yaml'
        assert not target.exists()
        OmegaConf.save(config, target)
        configs.append(dict(variant=variant, path=str(target), sha256=digest(target),
                            base_path=str(audit / base), base_sha256=digest(audit / base)))
    report = dict(state='prepared_not_trained', configs=configs, label_directory=str(output),
                  correction_priority_probability=.7, retained_uniform_probability=.3,
                  dataset_weights=[.8, .2], original_sampling_unchanged=True,
                  normalization_unchanged=True, initial_checkpoint_unchanged=True,
                  records=records, script_sha256=digest(Path(__file__)),
                  note='Independent sampling ablation. Existing runs are unchanged; no new training process started.')
    with (audit / 'approach_priority_preparation.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
