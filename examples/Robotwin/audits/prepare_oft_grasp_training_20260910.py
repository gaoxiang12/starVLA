"""Audit complete first-grasp windows and prepare data/architecture comparisons."""
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from examples.Robotwin.audits.prepare_grasp_precision_training import ROOT, digest
from examples.Robotwin.audits.run_grasp_lift_development import save

OUT = ROOT/'playground/Checkpoints/gawm_oft_grasp_training_20260910'
SHARED = Path('/data/gaoxiang/ckpts/gawm_oft_grasp_20260910')
OLD = ROOT/'playground/Checkpoints/gawm_grasp_precision_training_20260909'


def first_grasp_window(gripper, close, offset=16):
    gripper = np.asarray(gripper)
    if gripper.ndim != 1 or not np.isfinite(gripper).all():
        raise ValueError('Finite one-dimensional gripper trace required')
    closed = np.flatnonzero(gripper < .2)
    if not len(closed) or int(closed[0]) != close or close < offset:
        raise ValueError('First-close boundary mismatch or too little approach context')
    reopen = np.flatnonzero((np.arange(len(gripper)) > close) & (gripper > .8))
    if not len(reopen):
        raise ValueError('No first reopen boundary; do not silently use the full trajectory')
    end = int(reopen[0])
    if end-close <= offset:
        raise ValueError('Insufficient closed-stage context')
    return end, end-offset


def main():
    OUT.mkdir(exist_ok=False)
    old_path = OLD/'training_anchors.json'
    old = json.loads(old_path.read_text())
    split_path = OLD/'split.json'
    split = json.loads(split_path.read_text())
    rows, old_closed, new_closed, rises = [], [], [], []
    for row in old['episodes']:
        raw_path, parquet_path = Path(row['raw_path']), Path(row['parquet_path'])
        assert digest(raw_path) == row['raw_sha256']
        assert digest(parquet_path) == row['parquet_sha256']
        with h5py.File(raw_path) as f:
            actions = np.asarray(f['joint_action/vector'])
            grip = np.asarray(f[f"endpose/{row['active_arm']}_gripper"])
            poses = np.asarray(f[f"endpose/{row['active_arm']}_endpose"])
        parquet_actions = np.asarray(pq.read_table(parquet_path, columns=['action'])['action'].to_pylist())
        np.testing.assert_allclose(actions, parquet_actions, atol=1e-6, rtol=0)
        close = row['first_close_frame']
        end, limit = first_grasp_window(grip, close)
        arm_index = 6 if row['active_arm'] == 'left' else 13
        np.testing.assert_allclose(grip, actions[:, arm_index], atol=1e-6, rtol=0)
        rise = float((poses[close:end, 2]-poses[close, 2]).max())
        assert rise > .05, 'No observed TCP lift in the first closed interval'
        previous_limit = row['target_end_exclusive']-16
        old_closed.append(float(np.mean(grip[1:previous_limit+1] < .2)))
        new_closed.append(float(np.mean(grip[1:limit+1] < .2)))
        rises.append(rise)
        rows.append(dict(row, target_end_exclusive=end, first_reopen_above_08=end,
                         previous_target_end_exclusive=row['target_end_exclusive'],
                         current_observation_closed_anchors=int((grip[:limit] < .2).sum()),
                         legal_anchors=limit, observed_tcp_lift_during_closed_m=rise))
    assert len(rows) == 475
    assert {r['episode_index'] for r in rows} == set(split['train_episode_ids'])
    assert not set(split['validation_episode_ids']) & set(split['train_episode_ids'])
    assert all(v == 0 for v in old_closed)
    bounds = OUT/'first_grasp_anchors.json'
    save(bounds, dict(format_version=1, dataset=old['dataset'], maximum_target_offset=16,
        episodes=rows, parent_manifest=str(old_path), parent_sha256=digest(old_path),
        semantics='All current/action/future frames precede first active-arm reopen >0.8. '
                  'Contains approach, closure and observed TCP lift; no claim of certified block contact/hold labels.'))
    base = OmegaConf.load(OLD/'joint_train1000.yaml')
    base.run_root_dir = str(SHARED)
    base.datasets.vla_data.training_anchor_manifest = str(bounds)
    base.trainer.is_resume = False
    data_only = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    data_only.run_id = 'data_only1000'
    configs = {'data_only1000': data_only}
    for variant in ('act', 'oft'):
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        wm = cfg.framework.world_model
        for key in ('loss_latent_weight', 'latent_cosine_weight', 'visual_token_diversity_weight',
                    'visual_token_variance_weight'):
            wm[key] = 0.
        cfg.framework.spatial_focus.heatmap_loss_weight = 0.
        cfg.framework.spatial_focus.coordinate_loss_weight = 0.
        cfg.trainer.learning_rate = dict(base=1e-5, action_models=1e-4,
                                         backbone=dict(encoder=1e-5), spatial_focus=1e-4)
        if variant == 'oft':
            cfg.framework.name = 'GAWMOFT'
            cfg.framework.spatial_focus.enabled = False
            cfg.framework.oft_action = dict(hidden_dim=384, depth=4, heads=6, grid_size=14)
            cfg.trainer.learning_rate.pop('spatial_focus')
        for stage, steps in [('warmup500', 500), ('joint4500', 4500)]:
            current = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            name = f'{variant}_{stage}'
            current.run_id = name
            current.trainer.max_train_steps = steps
            current.trainer.num_warmup_steps = 50 if stage == 'warmup500' else 200
            current.trainer.logging_frequency = 20
            current.trainer.eval_interval = 250 if stage == 'warmup500' else 500
            current.trainer.save_interval = current.trainer.eval_interval
            current.trainer.validation_samples_per_task = 32
            if stage == 'warmup500':
                frozen = ['backbone','world_model','visual_token_pooler','task_embedding',
                          'embodiment_embedding','action_models.franka','action_models.oxe_bridge']
                if variant == 'act':
                    frozen.append('spatial_focus')
            else:
                current.trainer.pretrained_checkpoint = str(SHARED/f'{variant}_warmup500/final_model/pytorch_model.pt')
                frozen = ['action_models.franka','action_models.oxe_bridge']
                if variant == 'oft':
                    frozen += ['world_model','visual_token_pooler']
            current.trainer.freeze_modules = ','.join(frozen)
            configs[name] = current
    smoke = OmegaConf.create(OmegaConf.to_container(configs['oft_joint4500'], resolve=True))
    smoke.run_id = 'oft_smoke20'
    smoke.trainer.pretrained_checkpoint = base.trainer.pretrained_checkpoint
    smoke.trainer.max_train_steps = 20
    smoke.trainer.num_warmup_steps = 2
    smoke.trainer.eval_interval = smoke.trainer.save_interval = 20
    smoke.trainer.logging_frequency = 1
    smoke.trainer.validation_samples_per_task = 8
    configs['oft_smoke20'] = smoke
    records = []
    for name, cfg in configs.items():
        p = OUT/f'{name}.yaml'
        OmegaConf.save(cfg, p)
        records.append(dict(name=name, path=str(p), sha256=digest(p), steps=int(cfg.trainer.max_train_steps)))
    save(OUT/'preparation.json', dict(state='raw_verified_complete_first_grasp_windows',
        source_split=str(split_path), source_split_sha256=digest(split_path), train_episodes=len(rows),
        validation_episodes=len(split['validation_episode_ids']), original_legal_anchors=sum(r['previous_target_end_exclusive']-16 for r in rows),
        new_legal_anchors=sum(r['legal_anchors'] for r in rows),
        original_first_action_closed_fraction=float(np.mean(old_closed)),
        new_first_action_closed_fraction_uniform_episode=float(np.mean(new_closed)),
        minimum_observed_tcp_lift_m=min(rises), median_observed_tcp_lift_m=float(np.median(rises)),
        correction_data_unchanged=True, mixture_weights=[.8,.2], normalization_unchanged=True,
        anchor_manifest=str(bounds), anchor_sha256=digest(bounds), configs=records,
        design='Data-only 1000 vs historical matched 1000; ACT vs OFT-style 500 head-only +4500 joint steps, '
               'same new windows/data/normalization/optimizer schedule. OFT current-RGB/no-state/direct L1 pathway differs as a bundle.',
        final_test_scenes_used=False))
    print(str(OUT), flush=True)


if __name__ == '__main__':
    main()
