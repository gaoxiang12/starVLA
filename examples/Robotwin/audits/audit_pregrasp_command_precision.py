"""Compare deployed command FK on fixed held-out approach observations; no rollouts."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.seeded_episode_policy import SeededEpisodePolicy
from examples.Robotwin.audits.run_grasp_lift_development import ROOT, digest, save
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.modules.robotwin_pose_kinematics import SerialPoseKinematics, rotation_log


def distribution(values):
    x = np.asarray(values)
    return dict(count=len(x), median=float(np.median(x)), p90=float(np.quantile(x, .9)),
                maximum=float(x.max()), fraction_le_10mm=float(np.mean(x <= 10)),
                fraction_gt_20mm=float(np.mean(x > 20)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'train'], default='validation')
    parser.add_argument('--episode-count', type=int, default=20)
    parser.add_argument('--before-close', type=int, nargs='+', default=[48, 32, 16, 4])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--inference-steps', type=int, default=None,
                        help='Offline flow solver diagnostic; never changes saved configuration or live servers')
    args = parser.parse_args()
    assert args.episode_count > 0 and len(set(args.before_close)) == len(args.before_close)
    assert all(x > 0 for x in args.before_close)
    assert args.inference_steps is None or args.inference_steps > 0
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    audit = ROOT/'playground/Checkpoints/gawm_grasp_precision_training_20260909'
    split_path = audit/'split.json'
    split = json.loads(split_path.read_text())
    config_path = ROOT/'playground/Checkpoints/gawm_grasp_precision_joint_1000_20260909/config.full.yaml'
    cfg = OmegaConf.load(config_path).datasets.vla_data
    cfg.episode_split = args.split
    mixture = get_vla_dataset(cfg, mode=args.split)
    assert len(mixture.datasets) == (1 if args.split == 'validation' else 2)
    child = mixture.datasets[0]
    child.transforms.eval()
    episodes = sorted(int(x) for x in child.trajectory_ids)
    assert episodes == sorted(split[f'{args.split}_episode_ids'])
    other_split = 'train' if args.split == 'validation' else 'validation'
    assert not set(episodes) & set(split[f'{other_split}_episode_ids'])
    assert args.episode_count <= len(episodes)
    episodes = [episodes[i] for i in np.linspace(0, len(episodes)-1, args.episode_count, dtype=int)]
    anchors_path = audit/'training_anchors.json'
    anchors = json.loads(anchors_path.read_text())
    legal_ends = {r['episode_index']:r['target_end_exclusive']-anchors['maximum_target_offset'] for r in anchors['episodes']}
    samples, sources = [], {}
    for episode in episodes:
        path = Path(split['dataset'])/f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet'
        sources[str(path)] = digest(path)
        raw = np.asarray(pq.read_table(path, columns=['action'])['action'].to_pylist(), dtype=np.float32)
        close, arm = np.argwhere(raw[:, [6, 13]] < .2)[0]
        # Parquet simulator order L6,Lg,R6,Rg -> deployed order L6,R6,Lg,Rg.
        actions = raw[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]
        for before in args.before_close:
            anchor = int(close)-before
            if anchor < 0:
                continue
            if args.split == 'train':
                assert anchor < legal_ends[episode], 'Train fit audit must use legal original training anchors'
            sample = child._pack_sample(child.transforms(child.get_step_data(episode, anchor)))
            assert sample['action'].shape == (16, 14)
            payload = dict(image=sample['image'], native_images=sample['native_images'],
                           lang=sample['lang'], state=actions[anchor:anchor+1], episode_start=True)
            images = [np.ascontiguousarray(x) for x in payload['image']+payload['native_images']]
            image_sha = [hashlib.sha256(x.tobytes()).hexdigest() for x in images]
            samples.append(dict(episode=episode, anchor=anchor, before_first_closed_command=before,
                first_closed_frame=int(close), active_arm=int(arm), payload=payload,
                target=actions[anchor+1:anchor+17], normalized_target=sample['action'],
                normalized_state=sample['state'], image_sha256=image_sha,
                target_count_through_first_closed_command=min(16, before)))
    assert len(samples) == len(args.before_close)*len(episodes), 'Unexpected missing planned approach anchor'
    urdf = Path('/data/gaoxiang/Code/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf')
    chains = [SerialPoseKinematics(urdf, f'{prefix}_link6',
              [f'{prefix}_joint{i}' for i in range(1, 7)]) for prefix in ('fl', 'fr')]
    # Robot root is translated and yaw-rotated only, so XY/3D norms are world invariant.
    manifest = dict(state='observations_frozen', episodes=episodes, observation_count=len(samples),
        anchors_before_first_closed_command=args.before_close, policy_seed=20260909,
        split=args.split, device=args.device,
        inference_steps_override=args.inference_steps,
        source_sha256=dict(sources, **{str(p):digest(p) for p in (split_path, anchors_path, config_path, urdf, Path(__file__))}),
        observations=[{k:v for k,v in s.items() if k not in ('payload','target','normalized_target','normalized_state')}
                      for s in samples],
        checkpoints=[dict(path=str(p.resolve()), sha256=digest(p)) for p in args.checkpoint])
    save(out/'manifest.json', manifest)
    reports = []
    for checkpoint in args.checkpoint:
        wrapper = PolicyServerWrapper(str(checkpoint.resolve()), device=args.device, use_bf16=False, unnorm_key='aloha')
        model = wrapper._framework
        if args.inference_steps is not None:
            assert getattr(model, 'expert_mode', None) == 'flow'
            model.action_models['aloha'].inference_steps = args.inference_steps
        assert wrapper.metadata['action_specs']['aloha']['action_horizon'] == 16
        proc = wrapper._get_processor('aloha')
        seeded = SeededEpisodePolicy(wrapper, 20260909)
        rows = []
        for sample in samples:
            # Verify the exact data ordering, next-frame offset, and normalization.
            n = sample['target_count_through_first_closed_command']
            np.testing.assert_allclose(proc.unapply_actions(sample['normalized_target'])[:n],
                                       sample['target'][:n], atol=.005, rtol=0)
            np.testing.assert_array_equal(proc.apply_state(sample['payload']['state']).astype(sample['normalized_state'].dtype),
                                          sample['normalized_state'])
            prediction = seeded.predict_action(examples=[sample['payload']])['actions'][0]
            assert prediction.shape == (16, 14) and np.isfinite(prediction).all()
            arm = sample['active_arm']
            section = slice(arm*6, arm*6+6)
            with torch.inference_mode():
                estimated = chains[arm].pose(torch.tensor(prediction[:n, section], dtype=torch.float64))
                target = chains[arm].pose(torch.tensor(sample['target'][:n, section], dtype=torch.float64))
                delta = (estimated[:, :3, 3]-target[:, :3, 3])*1000
                angle = torch.linalg.vector_norm(rotation_log(estimated[:, :3, :3] @ target[:, :3, :3].transpose(-1,-2)),dim=-1)*180/np.pi
            xy = torch.linalg.vector_norm(delta[:, :2],dim=-1).numpy()
            xyz = torch.linalg.vector_norm(delta,dim=-1).numpy()
            rows.append(dict(episode=sample['episode'], anchor=sample['anchor'], active_arm=arm,
                before_first_closed_command=sample['before_first_closed_command'], scored_command_count=n,
                command_xy_error_mm=xy.tolist(), command_3d_error_mm=xyz.tolist(),
                command_rotation_error_deg=angle.tolist(),
                predicted_gripper=prediction[:n,12+arm].tolist(), target_gripper=sample['target'][:n,12+arm].tolist(),
                first_command_xy_error_mm=float(xy[0]), first_command_3d_error_mm=float(xyz[0]),
                final_scored_command_xy_error_mm=float(xy[-1]), mean_chunk_xy_error_mm=float(xy.mean())))
            save(out/'status.json', dict(state='running', checkpoint=str(checkpoint), observations_complete=len(rows), observations=len(samples)))
        report = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest(checkpoint),
            framework=type(model).__name__, mode=getattr(model,'expert_mode',None),
            inference_steps=getattr(model.action_models['aloha'], 'inference_steps', None),
            first_command_xy=distribution([r['first_command_xy_error_mm'] for r in rows]),
            first_command_3d=distribution([r['first_command_3d_error_mm'] for r in rows]),
            chunk_mean_xy=distribution([r['mean_chunk_xy_error_mm'] for r in rows]),
            by_anchor={str(before):distribution([r['first_command_xy_error_mm'] for r in rows if r['before_first_closed_command']==before])
                       for before in args.before_close}, records=rows)
        reports.append(report)
        save(out/f'model_{len(reports)-1}.json', report)
        del seeded, model, wrapper
        gc.collect()
        torch.cuda.empty_cache()
    save(out/'report.json', dict(state='complete', reports=reports, manifest_sha256=digest(out/'manifest.json'),
        limitations=[
            f'{args.split} expert observations, not policy rollout states or physical grasp outcomes.',
            'FK compares commanded poses to expert commanded poses, not object-center alignment or tracking error.',
            'Four correlated observations per validation episode; counts are not independent trials.',
            'Each observation resets policy history and the same fixed Torch noise schedule; context length is one.',
            'Closed-command reference is first grip <0.2; late observations may already be closing.',
            'Only commands through first closed reference are scored, so late-anchor chunks contain fewer commands.',
            'Raw expert commands are ground truth; inverse normalized FP16 labels are checked within 0.005 rad.',
        ]))
    save(out/'status.json', dict(state='complete', models=len(reports), observations_per_model=len(samples)))


if __name__ == '__main__':
    main()
