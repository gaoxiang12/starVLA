"""Measure geometric TCP error at held-out gripper-closing events."""
import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.modules.robotwin_kinematics import SerialTCPKinematics


def summarize(rows):
    values = np.asarray([r['tcp_error_mm'] for r in rows])
    return dict(count=len(rows), median_mm=float(np.median(values)), p90_mm=float(np.quantile(values, .9)),
                mean_mm=float(values.mean()), above_10mm=float((values > 10).mean()),
                above_20mm=float((values > 20).mean()),
                predicted_closed_fraction=float(np.mean([r['predicted_gripper'] < .2 for r in rows])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-gripper-trajectories', action='store_true',
                        help='Save the complete predicted/target command chunk for timing diagnosis')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = args.run/'checkpoints'/f'steps_{args.step}_pytorch_model.pt'
    cfg = OmegaConf.load(args.run/'config.full.yaml')
    cfg.datasets.vla_data.episode_split = 'validation'
    cfg.datasets.vla_data.episode_split_manifest = str(args.split_manifest.resolve())
    cfg.datasets.vla_data.normalization_statistics_path = str(args.run/'dataset_statistics.json')
    dataset, = get_vla_dataset(cfg.datasets.vla_data, mode='validation').datasets
    dataset.transforms.eval()
    expected = set(json.loads(args.split_manifest.read_text())['validation_episode_ids'])
    assert set(dataset.trajectory_ids) == expected
    processor = PolicyNormProcessor(str(checkpoint), unnorm_key='aloha')
    model = build_framework(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.cuda().eval()
    chains = [SerialTCPKinematics(args.urdf, f'{side}_link6', [f'{side}_joint{i}' for i in range(1, 7)])
              for side in ('fl', 'fr')]
    examples, records = [], []
    for episode in sorted(expected):
        table = dataset.get_trajectory_data(episode)
        actions = np.asarray(table['action'].tolist())
        for arm, column in enumerate((6, 13)):
            closed = actions[:, column] < .2
            for event in np.flatnonzero(closed[1:] & ~closed[:-1])+1:
                for lookahead in (1, 8, 16):
                    anchor = int(event)-lookahead
                    if anchor < 0:
                        continue
                    raw = dataset.get_step_data(episode, anchor)
                    selected_target = np.concatenate([np.asarray(raw[key])[lookahead-1].reshape(-1).copy()
                                                       for key in processor.action_keys])
                    raw_target = actions[event][[0,1,2,3,4,5,7,8,9,10,11,12,6,13]]
                    np.testing.assert_allclose(selected_target, raw_target, atol=1e-6, rtol=0)
                    if args.include_gripper_trajectories:
                        trajectory = np.concatenate([np.asarray(raw[key]).reshape(len(raw[key]), -1)
                                                     for key in processor.action_keys], axis=-1).copy()
                    sample = dataset._pack_sample(dataset.transforms(raw))
                    examples.append({key: value for key, value in sample.items()
                                     if 'future' not in key and not key.startswith('spatial_')
                                     and key not in ('action', 'action_valid_mask')})
                    target = processor.unapply_actions(np.asarray(sample['action']))[lookahead-1]
                    # Packing currently quantizes normalized targets to float16.
                    # Validate temporal alignment before quantization, then report
                    # its geometric contribution separately from model error.
                    records.append(dict(episode=int(episode), event=int(event), arm=arm,
                                        anchor=anchor, lookahead=lookahead, target=raw_target,
                                        quantized_target=target))
                    if args.include_gripper_trajectories:
                        records[-1]['target_gripper_trajectory'] = trajectory[:, 12+arm].tolist()
                        records[-1]['action_valid_mask'] = np.asarray(
                            sample.get('action_valid_mask', np.ones(len(trajectory), dtype=bool)),
                            dtype=bool).reshape(-1).tolist()
    with torch.inference_mode():
        for start in range(0, len(examples), 4):
            response = model.predict_action(examples[start:start+4])
            for row, normalized in zip(records[start:start+4], response['normalized_actions']):
                action_chunk = processor.unapply_actions(np.asarray(normalized))
                action = action_chunk[row['lookahead']-1]
                target = row.pop('target')
                quantized_target = row.pop('quantized_target')
                arm = row['arm']
                if args.include_gripper_trajectories:
                    row['predicted_gripper_trajectory'] = action_chunk[:, 12+arm].tolist()
                q = slice(arm*6, (arm+1)*6)
                positions = chains[arm](torch.tensor(np.stack([action[q], target[q], quantized_target[q]]), dtype=torch.float64)).numpy()
                delta = positions[0]-positions[1]
                row.update(tcp_error_mm=float(np.linalg.norm(delta)*1000),
                           horizontal_error_mm=float(np.linalg.norm(delta[:2])*1000),
                           vertical_error_mm=float(abs(delta[2])*1000),
                           label_quantization_error_mm=float(np.linalg.norm(positions[2]-positions[1])*1000),
                           joint_l1_rad=float(np.abs(action[q]-target[q]).mean()),
                           predicted_gripper=float(action[12+arm]), target_gripper=float(target[12+arm]))
    report = dict(checkpoint=str(checkpoint.resolve()), split=str(args.split_manifest.resolve()),
                  urdf=str(args.urdf.resolve()), rows=records,
                  by_lookahead={str(h): summarize([r for r in records if r['lookahead']==h]) for h in (1,8,16)},
                  by_arm={str(a): summarize([r for r in records if r['arm']==a]) for a in (0,1)},
                  note='Teacher-observation joint-target FK, not measured closed-loop TCP or contact success. '
                       'Exact gripper <0.2 transition and next-recorded target index are checked. '
                       '20 validation episodes represent 10 paired initial layouts.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
