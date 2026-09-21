"""Exercise the actual weighted loader and certify all original prefix bounds."""
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from examples.Robotwin.audits.prepare_grasp_precision_training import ROOT, OUT, digest
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def main():
    assert not (OUT/'mixture_audit.json').exists()
    config = OmegaConf.load(OUT/'cartesian_smoke.yaml')
    cfg = config.datasets.vla_data
    mixture = get_vla_dataset(cfg, mode='train')
    original, correction = mixture.datasets
    split = json.loads((OUT/'split.json').read_text())
    assert set(original.trajectory_ids) == set(split['train_episode_ids'])
    assert set(correction.trajectory_ids) == set(range(20))
    assert not hasattr(correction, 'training_anchor_limits')
    np.testing.assert_allclose(mixture.dataset_sampling_weights, [.8, .2], atol=1e-12)
    indexed_counts = {}
    for episode, step in original.all_steps:
        assert 0 <= step < original.training_anchor_limits[int(episode)]
        assert step+16 < original.training_target_ends[int(episode)]
        indexed_counts[int(episode)] = indexed_counts.get(int(episode), 0)+1
    assert indexed_counts == original.training_anchor_limits
    counts, seen, max_anchors = [0, 0], [set(), set()], {}
    for i in range(10000):
        child, episode, step = mixture.sample_step(i)
        index = 0 if child is original else 1
        counts[index] += 1
        seen[index].add(int(episode))
        if index == 0:
            assert step+16 < child.training_target_ends[int(episode)]
            max_anchors[int(episode)] = max(max_anchors.get(int(episode), -1), int(step))
    assert .18 < counts[1]/sum(counts) < .22
    assert seen == [set(split['train_episode_ids']), set(range(20))]
    sample_reports = []
    for index, child in enumerate(mixture.datasets):
        child.transforms.eval()
        assert child.action_spec_id == 'aloha_dual_joint_contgrip_next_recorded_14'
        assert child.data_cfg.normalization_statistics_path == cfg.normalization_statistics_path
        assert child.control_hz is None and child.future_time_offsets_s is None
        # Directly exercise boundary samples as well as mixture packing.
        records, actions = [], []
        for ep, length in list(zip(child.trajectory_ids, child.trajectory_lengths))[:20]:
            end = original.training_anchor_limits[int(ep)] if index == 0 else int(length)-1
            for anchor in sorted({0, end//2, end-1}):
                sample = child._pack_sample(child.transforms(child.get_step_data(ep, anchor)))
                sample = child._attach_action_validity(sample, ep, anchor)
                sample = child._attach_future_frame_validity(sample, ep, anchor)
                sample = child._attach_spatial_supervision(sample, ep, anchor)
                assert sample['action'].shape == (16, 14) and sample['state'].shape == (1, 14)
                assert sample['lang'] == 'blocks ranking rgb'
                assert len(sample['native_images']) == 3
                assert sample['spatial_target_xy'].shape == (3, 2)
                assert np.isfinite(sample['action']).all() and np.isfinite(sample['state']).all()
                np.testing.assert_array_equal(sample['action_valid_mask'], anchor+np.arange(1,17) < length)
                np.testing.assert_array_equal(sample['future_frame_valid_mask'], anchor+np.asarray([0,6,12]) < length)
                if index == 0:
                    assert sample['action_valid_mask'].all() and sample['future_frame_valid_mask'].all()
                actions.append(sample['action'][sample['action_valid_mask']])
                records.append(dict(episode=int(ep), anchor=anchor,
                    valid_actions=int(sample['action_valid_mask'].sum())))
        for i in range(4):
            assert mixture[(index, i)]['robot_tag'] == 'aloha'
        values = np.concatenate(actions)
        sample_reports.append(dict(dataset=str(child.dataset_path), boundary_checks=records,
            normalization_min=values.min(0).tolist(), normalization_max=values.max(0).tolist(),
            continuous_abs_p99=np.quantile(abs(values[:, :12]), .99, axis=0).tolist(),
            gripper_closed_fraction=(values[:,12:] < -.6).mean(0).tolist()))
    cfg.episode_split = 'validation'
    validation = get_vla_dataset(cfg, mode='validation')
    assert len(validation.datasets) == 1
    assert set(validation.datasets[0].trajectory_ids) == set(split['validation_episode_ids'])
    assert not hasattr(validation.datasets[0], 'training_anchor_limits')
    report = dict(state='shared_mixture_verified', samples=sample_reports,
        random_draw_counts_10000=counts, random_seen_episodes=[len(s) for s in seen],
        original_legal_anchor_count=len(original.all_steps),
        validation_episode_ids=sorted(map(int, validation.datasets[0].trajectory_ids)),
        normalization_sha256=digest(cfg.normalization_statistics_path),
        configured_training_weights=[.8,.2], training_started=False,
        purpose='Training smoke acceptance. No learned policy grasp or closure precision result.')
    (OUT/'mixture_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'samples'}), flush=True)


if __name__ == '__main__':
    main()
