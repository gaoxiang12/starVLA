"""Check the actual recovery experiment mixture and its planned sampling exposure."""
import argparse
import collections
import copy
import itertools
import json
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from examples.Robotwin.audits.verify_rgb_recovery_sources import digest
from starVLA.dataloader.lerobot_datasets import EmbodimentBatchSampler, get_vla_dataset

ROOT = Path(__file__).resolve().parents[3]
ORDER = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    cfg = OmegaConf.load(args.config)
    plan = json.loads(args.plan.read_text())
    assert plan['state'] == 'converted_labels_ready' and plan['episodes']
    assert digest(plan['original_split']) == plan['original_split_sha256']
    split = json.loads(Path(plan['original_split']).read_text())
    records = {r['episode_index']: r for r in plan['episodes']}
    assert len(records) == len(plan['episodes'])
    assert set(records) == set(plan['train_episode_ids']) and not plan['validation_episode_ids']
    assert all(r['source_episode'] in split['train_episode_ids'] and
               r['source_episode'] not in split['validation_episode_ids'] for r in records.values())
    data_cfg = cfg.datasets.vla_data
    data_cfg.num_workers = 0
    mix = get_vla_dataset(data_cfg, mode='train')
    assert len(mix.datasets) == 2
    original, recovery = mix.datasets
    assert set(original.trajectory_ids) == set(split['train_episode_ids'])
    assert set(recovery.trajectory_ids) == set(records)
    assert Path(recovery.dataset_path).resolve() == Path(plan['dataset']).resolve()
    assert Path(recovery.data_cfg.spatial_supervision_dir).resolve() == Path(plan['labels']).resolve()
    np.testing.assert_allclose(mix.dataset_sampling_weights, [.9, .1], atol=1e-12, rtol=0)
    checkpoint = Path(cfg.trainer.pretrained_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    assert digest(checkpoint) == plan['policy_checkpoint_sha256']
    norm = PolicyNormProcessor(str(checkpoint), unnorm_key='aloha')
    phase_lookup, range_rows, read_rows = {}, [], []
    for index, child in enumerate(mix.datasets):
        assert child.action_spec_id == 'aloha_dual_joint_contgrip_next_recorded_14'
        assert child.data_cfg.normalization_statistics_path == data_cfg.normalization_statistics_path
        child.transforms.train()
        for sample_index in range(4):
            ex = mix[(index, sample_index)]
            assert ex['action'].shape == (16, 14) and ex['lang'] == 'blocks ranking rgb'
            assert np.isfinite(ex['action']).all() and ex['action_valid_mask'].any()
    for episode, record in records.items():
        assert digest(record['source_hdf5']) == record['source_hdf5_sha256']
        phases = json.loads(Path(record['frame_phases']).read_text())
        assert len(phases) == record['frames']
        phase_lookup[episode] = phases
        with h5py.File(record['source_hdf5']) as raw:
            commands = np.asarray(raw['joint_action/vector'], np.float32)[:, ORDER]
        fields = {}
        for keys in (norm.action_keys, norm.state_keys):
            for key, start, stop in zip(keys, (0, 6, 12, 13), (6, 12, 13, 14)):
                fields[key] = commands[:, start:stop].copy()
        transformed = norm.transform(fields)
        normalized = np.concatenate([np.asarray(transformed[k]) for k in norm.action_keys], -1)
        assert np.isfinite(normalized).all()
        restored = norm.unapply_actions(normalized)
        np.testing.assert_allclose(restored, commands, atol=1e-6, rtol=0)
        assert (normalized[:, 12:] >= -1e-6).all() and (normalized[:, 12:] <= 1+1e-6).all()
        range_rows.append(dict(episode=episode, source_episode=record['source_episode'],
            frames=len(commands), normalized_min=normalized.min(0).tolist(),
            normalized_max=normalized.max(0).tolist(),
            joint_fraction_beyond_1_001=float((np.abs(normalized[:, :12]) > 1.001).mean()),
            normalization_roundtrip_max_error=float(np.abs(restored-commands).max())))
        n = record['frames']
        for anchor in (0, n//2, n-2):
            sample = recovery._pack_sample(recovery.transforms(recovery.get_step_data(episode, anchor)))
            sample = recovery._attach_action_validity(sample, episode, anchor)
            sample = recovery._attach_future_frame_validity(sample, episode, anchor)
            sample = recovery._attach_spatial_supervision(sample, episode, anchor)
            np.testing.assert_array_equal(sample['action_valid_mask'], anchor+np.arange(1, 17) < n)
            np.testing.assert_array_equal(sample['future_frame_valid_mask'], anchor+np.array([0, 6, 12]) < n)
            assert sample['action'].shape == (16, 14) and sample['state'].shape == (1, 14)
            assert np.isfinite(sample['action']).all() and np.isfinite(sample['state']).all()
            assert sample['lang'] == 'blocks ranking rgb' and len(sample['native_images']) == 3
            read_rows.append(dict(episode=episode, anchor=anchor,
                                 valid_actions=int(sample['action_valid_mask'].sum())))
    sampler = EmbodimentBatchSampler(mix, batch_size=data_cfg.per_device_batch_size, seed=cfg.seed,
                                     embodiment_weights=data_cfg.embodiment_sampling_weights)
    microbatches = int(cfg.trainer.max_train_steps * cfg.trainer.gradient_accumulation_steps)
    expected = microbatches * int(data_cfg.per_device_batch_size)
    counts = collections.Counter()
    phases = collections.Counter()
    per_episode = collections.Counter()
    anchors = collections.defaultdict(set)
    for batch in itertools.islice(sampler, microbatches):
        for dataset_index, sample_index in batch:
            child, episode, anchor = mix.sample_step(sample_index, dataset_index=dataset_index)
            counts[dataset_index] += 1
            if child is recovery:
                phases[phase_lookup[int(episode)][int(anchor)]] += 1
                per_episode[int(episode)] += 1
                anchors[int(episode)].add(int(anchor))
    assert sum(counts.values()) == expected and counts[1] > 0
    validation_cfg = copy.deepcopy(data_cfg)
    validation_cfg.episode_split = 'validation'
    validation = get_vla_dataset(validation_cfg, mode='validation')
    assert len(validation.datasets) == 1
    assert set(validation.datasets[0].trajectory_ids) == set(split['validation_episode_ids'])
    correction = sum(value for key, value in phases.items() if key != 'expert_rgb_replan')
    report = dict(state='complete', config=str(args.config.resolve()), config_sha256=digest(args.config),
        conversion_plan=str(args.plan.resolve()), conversion_plan_sha256=digest(args.plan),
        source_checkpoint_sha256=plan['policy_checkpoint_sha256'],
        training_episode_counts=[len(original.trajectory_ids), len(recovery.trajectory_ids)],
        actual_dataset_sampling_counts=dict(counts), planned_samples=expected,
        recovery_phase_sample_counts=dict(phases), correction_samples=correction,
        correction_fraction_of_total=correction/expected, recovery_samples_by_episode=dict(per_episode),
        unique_recovery_anchors_by_episode={key: len(value) for key, value in anchors.items()},
        priority_probability=float(recovery.data_cfg.get('priority_sampling_probability', 0)),
        event_probability=float(recovery.data_cfg.get('event_sampling_probability', 0)),
        range_checks=range_rows, train_read_checks=read_rows,
        validation_episode_ids=sorted(int(i) for i in validation.datasets[0].trajectory_ids),
        training_started=False,
        note='Actual CPU loader and prospective single-process epoch0 sampling. '
             'No optimizer or model inference. Joint tanh_linear_tail is unbounded; '
             'range statistics are reported, not arbitrarily clipped. Final training decision remains separate.')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(state='complete', training_episode_counts=report['training_episode_counts'],
        planned_samples=expected, correction_samples=correction, correction_fraction=correction/expected)), flush=True)


if __name__ == '__main__':
    main()
