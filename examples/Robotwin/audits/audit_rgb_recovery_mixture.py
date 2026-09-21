"""Verify real original/recovery mixing with unchanged held-out scenes and scales."""
import copy
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from transformers import DINOv3ViTImageProcessorFast

from examples.Robotwin.audits.convert_rgb_recovery_pilot import ROOT, CAMP
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def main():
    output = CAMP / 'mixture_audit.json'
    assert not output.exists()
    run = ROOT / 'playground/Checkpoints/gawm_rgb_focus_local_v2_5k_20260907'
    cfg = OmegaConf.load(ROOT / 'examples/Robotwin/train_files/starvla_gawm_rgb_refine_baseline.yaml').datasets.vla_data
    cfg.data_mix = 'robotwin_ranking_rgb_recovery_pilot_continuous_next_wm'
    cfg.normalization_statistics_path = str(run / 'dataset_statistics.json')
    recovery_name = 'RoboTwinRecoveryPilot_20260908/RoboTwinGenerated/Clean/blocks_ranking_rgb'
    cfg.dataset_options = {recovery_name: dict(splits=['train'], episode_split_manifest=None,
        validation_episode_stride=0, spatial_supervision_dir=str(CAMP / 'converted_labels'))}
    cfg.num_workers = 0
    mix = get_vla_dataset(cfg, mode='train')
    assert len(mix.datasets) == 2
    split = json.loads(Path(cfg.episode_split_manifest).read_text())
    assert set(mix.datasets[0].trajectory_ids) == set(split['train_episode_ids'])
    assert set(mix.datasets[1].trajectory_ids) == {0}
    np.testing.assert_allclose(mix.dataset_sampling_weights, [.9, .1], atol=1e-12, rtol=0)
    metadata = []
    transform_checks = []
    for index, child in enumerate(mix.datasets):
        child.transforms.train()
        assert child.data_cfg.normalization_statistics_path == cfg.normalization_statistics_path
        assert child.action_spec_id == 'aloha_dual_joint_contgrip_next_recorded_14'
        metadata.append(dict(dataset=str(child.dataset_path), episode_count=len(child.trajectory_ids),
                             labels=child.data_cfg.spatial_supervision_dir,
                             action_spec_id=child.action_spec_id,
                             transforms=[type(t).__name__ for t in child.transforms.transforms]))
        for sample_index in range(4):
            ex = mix[(index, sample_index)]
            assert ex['lang'] == 'blocks ranking rgb' and ex['action'].shape == (16, 14)
            assert ex['action_valid_mask'].any() and np.isfinite(ex['action']).all()
            assert ex['spatial_target_xy'].shape == (3, 2)
            assert len(ex['native_images']) == 3
        episode = int(child.trajectory_ids[0])
        for anchor in (0, 36, 128):
            raw = child.get_step_data(episode, anchor)
            transformed = child.transforms(copy.deepcopy(raw))
            for key in child.modality_keys['video']:
                np.testing.assert_array_equal(raw[key], transformed[key])
            transform_checks.append(dict(dataset=index, episode=episode, anchor=anchor,
                                         all_current_future_video_arrays_unchanged=True))
    # Exercise actual weighted dataset/episode/step sampling, without decoding 2000 videos.
    counts = [0, 0]
    for i in range(2000):
        child, episode, anchor = mix.sample_step(i)
        index = next(j for j, d in enumerate(mix.datasets) if child is d)
        counts[index] += 1
        assert episode in child.trajectory_ids and anchor >= 0
    assert .07 < counts[1] / sum(counts) < .13
    original_cfg = copy.deepcopy(cfg)
    original_cfg.data_mix = 'robotwin_ranking_rgb_continuous_next_wm'
    original_cfg.dataset_options = {}
    original = get_vla_dataset(original_cfg, mode='train')
    # Normalized original-data examples must remain identical when adding a supplement.
    for anchor in (0, 36, 128):
        left, right = mix.datasets[0], original.datasets[0]
        episode = int(left.trajectory_ids[0])
        a = left._pack_sample(left.transforms(left.get_step_data(episode, anchor)))
        b = right._pack_sample(right.transforms(right.get_step_data(episode, anchor)))
        np.testing.assert_array_equal(a['action'], b['action'])
        np.testing.assert_array_equal(a['state'], b['state'])
    cfg.episode_split = 'validation'
    validation = get_vla_dataset(cfg, mode='validation')
    assert len(validation.datasets) == 1
    assert set(validation.datasets[0].trajectory_ids) == set(split['validation_episode_ids'])
    assert len(validation.datasets[0].trajectory_ids) == 20
    processor = DINOv3ViTImageProcessorFast()
    assert not processor.do_center_crop and processor.size == {'height': 224, 'width': 224}
    report = dict(state='complete', training_datasets=metadata, actual_sampling_counts_2000=counts,
                  configured_dataset_weights=[.9, .1],
                  validation_episode_ids=sorted(int(i) for i in validation.datasets[0].trajectory_ids),
                  normalization_statistics_path=cfg.normalization_statistics_path,
                  original_normalized_targets_unchanged=True, transform_checks=transform_checks,
                  processor_center_crop=False, processor_resize_hw=[224, 224],
                  training_started=False,
                  note='Pilot mixture plumbing check only. 90/10 is a fixture weight, not a selected final training recipe. No model inference or performance result.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
