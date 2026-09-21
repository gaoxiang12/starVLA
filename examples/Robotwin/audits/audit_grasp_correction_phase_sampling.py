"""Measure correction-phase exposure using the configured shared sampler."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--baseline-config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    config = OmegaConf.load(args.config)
    mixture = get_vla_dataset(config.datasets.vla_data, mode='train')
    original, correction = mixture.datasets
    np.testing.assert_allclose(mixture.dataset_sampling_weights, [.8, .2])
    for child in mixture.datasets:
        assert child.data_cfg.event_sampling_probability == 0
    assert original.data_cfg.priority_sampling_probability == 0
    priority_probability = float(correction.data_cfg.priority_sampling_probability)
    assert 0 <= priority_probability < 1
    baseline = None
    if args.baseline_config:
        baseline_config = OmegaConf.load(args.baseline_config)
        baseline = get_vla_dataset(baseline_config.datasets.vla_data, mode='train')
        assert baseline.datasets[1].data_cfg.priority_sampling_probability == 0
    root = Path('/data/gaoxiang/RoboTwinPregraspCorrections_20260909/train20_converted')
    conversion = json.loads((root / 'conversion_manifest.json').read_text())
    names = conversion['phase_names']
    assert names == ['align_pregrasp', 'descend_aligned', 'close', 'lift', 'hold']
    assert set(correction.trajectory_ids) == set(range(20))
    expected = np.zeros(len(names))
    phases, rows = {}, []
    for episode, length, weight in zip(correction.trajectory_ids, correction.trajectory_lengths,
                                       mixture.trajectory_sampling_weights[1]):
        p = root / 'labels' / f'episode_{episode:06d}.npz'
        with np.load(p, allow_pickle=False) as data:
            phases[int(episode)] = data['phase'].copy()
        assert len(phases[int(episode)]) == length
        eligible = int(length) - int(correction.minimum_action_offset)
        counts = np.bincount(phases[int(episode)][:eligible], minlength=len(names))
        distribution = counts / eligible
        priority_counts = None
        if priority_probability:
            priority = correction._spatial_episode(int(episode))['priority_anchors']
            assert len(priority) and len(priority) == len(np.unique(priority))
            assert np.all((priority >= 0) & (priority < eligible))
            priority_counts = np.bincount(phases[int(episode)][priority], minlength=len(names))
            assert priority_counts[2:].sum() == 0
            distribution = (1-priority_probability)*distribution + priority_probability*priority_counts/len(priority)
        expected += float(weight) * distribution
        rows.append(dict(episode=int(episode), eligible_anchors=eligible,
                         phase_counts=counts.tolist(),
                         priority_phase_counts=priority_counts.tolist() if priority_counts is not None else None,
                         labels_sha256=digest(p)))
    draws = 10000
    observed = np.zeros(len(names), dtype=int)
    original_count = 0
    for index in range(draws):
        child, episode, anchor = mixture.sample_step(index)
        if baseline is not None:
            old_child, old_episode, old_anchor = baseline.sample_step(index)
            assert (child is original) == (old_child is baseline.datasets[0])
            assert episode == old_episode
            if child is original:
                assert anchor == old_anchor
        if child is original:
            original_count += 1
        else:
            assert child is correction
            observed[phases[int(episode)][anchor]] += 1
    assert original_count + int(observed.sum()) == draws
    assert abs(observed.sum() / draws - .2) < .02
    np.testing.assert_allclose(observed / observed.sum(), expected, atol=.025, rtol=0)
    report = dict(state='configured_phase_exposure_measured', sample_draws=draws,
                  original_approach_draws=original_count, correction_draws=int(observed.sum()),
                  correction_priority_probability=priority_probability,
                  baseline_original_draws_exact=baseline is not None,
                  phases={name: dict(actual_draws=int(observed[i]),
                                     fraction_of_all_draws=float(observed[i] / draws),
                                     expected_fraction_of_all_draws=float(.2 * expected[i]),
                                     expected_fraction_within_correction=float(expected[i]))
                          for i, name in enumerate(names)},
                  records=rows,
                  source_sha256={str(p.resolve()): digest(p) for p in
                      (args.config, root / 'conversion_manifest.json', Path(__file__),
                       Path('starVLA/dataloader/gr00t_lerobot/datasets.py'),
                       *([args.baseline_config] if args.baseline_config else []))},
                  limitations=[
                      'Fractions count current-observation anchors, not the number of target actions or their gradient contribution.',
                      'A 16-action window can cross phase boundaries; this audit does not assign every target to its anchor phase.',
                      'Original 80% data also trains approach behavior; the small correction-alignment fraction is not total approach exposure.',
                      'Deterministic sampler replay is independent of running workers; no training configuration was changed.',
                  ])
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'source_sha256')}), flush=True)


if __name__ == '__main__':
    main()
