"""Pin pre-closure source windows and matched Cartesian/joint training configs."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'playground/Checkpoints/gawm_grasp_precision_training_20260909'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(name, value):
    with (OUT/name).open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')


def main():
    OUT.mkdir(exist_ok=True)
    candidate_path = ROOT/'playground/Checkpoints/gawm_grasp_lift_preflight_20260909/prefix_candidates.json'
    candidates = json.loads(candidate_path.read_text())
    split_path = Path(candidates['split'])
    assert digest(split_path) == candidates['split_sha256']
    split = json.loads(split_path.read_text())
    dataset = Path(split['dataset'])
    rows = []
    for r in candidates['candidates']:
        episode = r['episode']
        parquet = dataset/f'data/chunk-000/episode_{episode:06d}.parquet'
        actions = np.asarray(pq.read_table(parquet, columns=['action'])['action'].to_pylist())
        with h5py.File(r['raw_path']) as raw:
            commands = np.asarray(raw['joint_action/vector'])
            grips = np.stack([np.asarray(raw[f'endpose/{arm}_gripper']) for arm in ('left', 'right')], -1)
        assert commands.shape == actions.shape and np.isfinite(actions).all()
        np.testing.assert_allclose(commands, actions, atol=1e-6, rtol=0)
        close, arm = map(int, np.argwhere(grips < .2)[0])
        np.testing.assert_allclose(grips, actions[:, [6, 13]], atol=1e-6, rtol=0)
        assert close == r['first_close_frame'] and close >= 16
        rows.append(dict(episode_index=episode, target_end_exclusive=close+1,
            first_close_frame=close, active_arm=('left', 'right')[arm],
            source_frames=len(actions), raw_path=r['raw_path'], raw_sha256=digest(r['raw_path']),
            parquet_path=str(parquet), parquet_sha256=digest(parquet)))
    selected = sorted(r['episode_index'] for r in rows)
    assert len(selected) == 475 and set(selected).issubset(split['train_episode_ids'])
    ids = {json.loads(line)['episode_index'] for line in (dataset/'meta/episodes.jsonl').read_text().splitlines()}
    validation = split['validation_episode_ids']
    write('split.json', dict(dataset=str(dataset), train_episode_ids=selected,
        validation_episode_ids=validation, excluded_episode_ids=sorted(ids-set(selected)-set(validation)),
        parent_split=str(split_path), parent_sha256=digest(split_path),
        reason='Only original training scenes with locally verified raw first-close boundaries. Validation unchanged.'))
    write('training_anchors.json', dict(format_version=1, dataset=str(dataset),
        maximum_target_offset=16, episodes=rows, source_candidate_sha256=digest(candidate_path),
        semantics='Every current/action/future frame is at or before the first closed command. No certified original-data lift or hold claim.'))
    cfg = OmegaConf.load(ROOT/'playground/Checkpoints/gawm_cartesian_model_smoke_r4_20260909/config.yaml')
    for key in ('config_yaml', 'output_dir'):
        cfg.pop(key, None)
    data = cfg.datasets.vla_data
    data.data_root_dir = '/data/gaoxiang'
    data.data_mix = 'robotwin_rgb_grasp_precision'
    data.balance_trajectory_weights = True
    data.validation_episode_stride = 20
    data.episode_split_manifest = str(OUT/'split.json')
    data.training_anchor_manifest = str(OUT/'training_anchors.json')
    data.spatial_supervision_dir = str(ROOT/'playground/Checkpoints/gawm_rgb_focus_20260907/labels')
    data.event_sampling_probability = 0
    data.priority_sampling_probability = 0
    data.num_workers = 2
    correction = 'RoboTwinPregraspCorrections_20260909/train20_converted/RoboTwinGenerated/Clean/blocks_ranking_rgb'
    data.dataset_options = {correction: dict(splits=['train'], episode_split_manifest=None,
        validation_episode_stride=0, training_anchor_manifest=None,
        spatial_supervision_dir='/data/gaoxiang/RoboTwinPregraspCorrections_20260909/train20_converted/spatial_labels')}
    cfg.trainer.pretrained_checkpoint = 'playground/Checkpoints/gawm_rgb_focus_local_v2_5k_20260907/final_model/pytorch_model.pt'
    cfg.trainer.logging_frequency = 1
    cfg.trainer.max_train_steps = 20
    cfg.trainer.num_warmup_steps = 2
    cfg.trainer.eval_interval = 20
    cfg.trainer.save_interval = 20
    cfg.trainer.validation_samples_per_task = 8
    for variant in ('cartesian', 'joint'):
        current = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        current.run_id = f'gawm_grasp_precision_{variant}_smoke20_20260909'
        if variant == 'joint':
            current.framework.name = 'GAWM'
            current.framework.pop('cartesian_action')
        path = OUT/f'{variant}_smoke.yaml'
        assert not path.exists()
        OmegaConf.save(current, path)
    write('preparation.json', dict(state='source_windows_verified', original_train_episodes=len(rows),
        legal_original_anchors=sum(r['target_end_exclusive']-16 for r in rows),
        mean_original_prefix_frames=float(np.mean([r['target_end_exclusive'] for r in rows])),
        correction_train_episodes=20, mixture_weights=[.8,.2], uniform_episodes_within_dataset=True,
        validation_episodes=validation, training_started=False,
        limitation='New 3D phase/contact sidecars are audited provenance; current loss uses FK of demonstrated next commands, not these sidecars.'))
    print(str(OUT), flush=True)


if __name__ == '__main__':
    main()
