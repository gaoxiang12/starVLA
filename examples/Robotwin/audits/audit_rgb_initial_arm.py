"""Check the first moving arm on real held-out RGB demonstration starts."""
import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = args.run / 'checkpoints' / f'steps_{args.step}_pytorch_model.pt'
    cfg = OmegaConf.load(args.run / 'config.full.yaml')
    cfg.datasets.vla_data.episode_split = 'validation'
    cfg.datasets.vla_data.episode_split_manifest = str(args.split_manifest.resolve())
    cfg.datasets.vla_data.normalization_statistics_path = str(args.run / 'dataset_statistics.json')
    mixture = get_vla_dataset(cfg.datasets.vla_data, mode='validation')
    dataset, = mixture.datasets
    dataset.transforms.eval()
    expected = set(json.loads(args.split_manifest.read_text())['validation_episode_ids'])
    assert set(dataset.trajectory_ids) == expected
    processor = PolicyNormProcessor(str(checkpoint), unnorm_key='aloha')
    model = build_framework(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.cuda().eval()
    rows = []
    with torch.inference_mode():
        for episode in sorted(expected):
            raw = dataset.get_step_data(episode, 0)
            state = np.concatenate([np.asarray(raw[key])[-1].reshape(-1).copy()
                                    for key in processor.state_keys])
            sample = dataset._pack_sample(dataset.transforms(raw))
            inputs = {key: value for key, value in sample.items()
                      if 'future' not in key and not key.startswith('spatial_')
                      and key not in ('action', 'action_valid_mask')}
            response = model.predict_action([inputs])
            predicted = processor.unapply_actions(np.asarray(response['normalized_actions'])[0])
            target = processor.unapply_actions(np.asarray(sample['action']))
            motion = lambda actions: [float(np.abs(actions[:, :6]-state[:6]).mean()),
                                      float(np.abs(actions[:, 6:12]-state[6:12]).mean())]
            pred_motion, target_motion = motion(predicted), motion(target)
            target_arm, pred_arm = int(np.argmax(target_motion)), int(np.argmax(pred_motion))
            informative = max(target_motion) > .02 and min(target_motion) < .2 * max(target_motion)
            row = dict(episode=int(episode), first_frame=True,
                       target_motion_left_right=target_motion, predicted_motion_left_right=pred_motion,
                       informative=informative, target_arm=target_arm, predicted_arm=pred_arm,
                       correct=target_arm == pred_arm, action_l1_radians_and_grip=float(np.abs(predicted-target).mean()))
            if 'spatial_predicted_xy' in response:
                row['predicted_head_xy_pixels'] = (np.asarray(response['spatial_predicted_xy'])[0, 0]*[320, 240]).tolist()
            rows.append(row)
    selected = [row for row in rows if row['informative']]
    report = dict(checkpoint=str(checkpoint.resolve()), split=str(args.split_manifest.resolve()),
                  rows=rows, informative=len(selected), correct=sum(row['correct'] for row in selected),
                  by_target_arm={str(arm): dict(count=sum(row['target_arm']==arm for row in selected),
                      correct=sum(row['correct'] and row['target_arm']==arm for row in selected)) for arm in (0, 1)},
                  note='Real demonstration starts with no color edits; first 16 recorded action targets. '
                       'Arm identity inferred from target joint displacement, excluding ambiguous starts. '
                       'This tests initial arm choice, not grasp success; 20 episodes contain 10 scene pairs.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
