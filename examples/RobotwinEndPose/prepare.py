"""Prepare an isolated feedback ablation; never modify original demonstrations."""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from starVLA.robotwin_feedback import FEEDBACK, feedback_keys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'playground/Checkpoints/gawm_endpose_20260911'
DATASET = Path('/data/gaoxiang/RoboTwinEndPose/Clean/blocks_ranking_rgb')
RUNS = Path('/data/gaoxiang/ckpts/gawm_endpose_20260911')
BASE = ROOT/'playground/Checkpoints/qwen_gawm_ranking_20260910'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def statistics(values):
    x = np.asarray(values, dtype=np.float64)
    return {k: v.tolist() for k, v in dict(mean=x.mean(0), std=x.std(0), min=x.min(0),
        max=x.max(0), q01=np.quantile(x, .01, axis=0), q99=np.quantile(x, .99, axis=0)).items()}


def prepare_data(resume=False):
    OUT.mkdir(parents=True, exist_ok=True)
    source_split = ROOT/'playground/Checkpoints/gawm_grasp_precision_training_20260909/split.json'
    split = json.loads(source_split.read_text())
    source = Path(split['dataset'])
    assert len(split['validation_episode_ids']) == 20
    unavailable_validation = [ep for ep in split['validation_episode_ids'] if not Path(
        f'/data/gaoxiang/RoboTwinGenerated_raw/Clean/blocks_ranking_rgb/demo_clean/data/episode{ep}.hdf5').is_file()]
    # Availability-only restriction, shared by both arms; never draw replacements from test scenes.
    split['validation_episode_ids'] = [ep for ep in split['validation_episode_ids'] if ep not in unavailable_validation]
    assert split['validation_episode_ids']
    selected = sorted(split['train_episode_ids']+split['validation_episode_ids'])
    train = set(split['train_episode_ids'])
    assert len(train) == 475
    assert not (OUT/'data_manifest.json').exists(), 'Already prepared; do not overwrite'
    DATASET.mkdir(parents=True, exist_ok=resume)
    (DATASET/'meta').mkdir(exist_ok=resume)
    info = json.loads((source/'meta/info.json').read_text())
    modalities = json.loads((source/'meta/modality.json').read_text())
    stats = json.loads((source/'meta/stats.json').read_text())
    gr00t = json.loads((source/'meta/stats_gr00t.json').read_text())
    collected = {v: [] for v in FEEDBACK}
    rows = []
    lengths = {}
    for ep in selected:
        chunk = ep//info['chunks_size']
        relative = Path(info['data_path'].format(episode_chunk=chunk, episode_index=ep))
        original = source/relative
        table = pq.read_table(original)
        commands = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        raw = Path(f'/data/gaoxiang/RoboTwinGenerated_raw/Clean/blocks_ranking_rgb/demo_clean/data/episode{ep}.hdf5')
        with h5py.File(raw) as f:
            native = np.asarray(f['joint_action/vector'], dtype=np.float32)
            poses = {k: np.asarray(f['endpose'][k]) for k in
                     ('left_endpose','right_endpose','left_gripper','right_gripper')}
        np.testing.assert_array_equal(commands, native)
        np.testing.assert_array_equal(np.asarray(table['observation.state'].to_pylist(), np.float32), native)
        np.testing.assert_array_equal(np.asarray(table['frame_index']), np.arange(len(native)))
        for i, side in ((6,'left'), (13,'right')):
            np.testing.assert_allclose(poses[f'{side}_gripper'].reshape(-1), native[:,i], atol=1e-6, rtol=0)
        obs = dict(endpose=poses, joint_action={'vector': native})
        hashes = {}
        for variant, pack in FEEDBACK.items():
            values = pack(obs)
            assert values.shape == (len(native),16)
            hashes[variant] = hashlib.sha256(values.tobytes()).hexdigest()
            table = table.append_column(f'observation.{variant}_feedback',
                pa.array(values.tolist(), type=pa.list_(pa.float32())))
            if ep in train:
                collected[variant].append(values)
        destination = DATASET/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert resume
            assert pq.read_table(destination).equals(table), f'Incomplete output changed: {destination}'
        else:
            pq.write_table(table, destination)
        for key, feature in info['features'].items():
            if feature['dtype'] != 'video':
                continue
            relative_video = Path(info['video_path'].format(episode_chunk=chunk, episode_index=ep, video_key=key))
            assert (source/relative_video).is_file()
            target = DATASET/relative_video
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                assert resume and target.resolve() == (source/relative_video).resolve()
            else:
                target.symlink_to(source/relative_video)
        lengths[ep] = len(native)
        rows.append(dict(episode=ep, frames=len(native), raw=str(raw), source_parquet=str(original),
            parquet_sha256=digest(original), output_sha256=digest(destination), feedback_sha256=hashes))
        if len(rows)%50 == 0:
            print(f'Converted and aligned {len(rows)}/{len(selected)} episodes', flush=True)
    for variant in FEEDBACK:
        original_key = f'observation.{variant}_feedback'
        info['features'][original_key] = dict(dtype='float32', shape=[16], names=None)
        stats[original_key] = statistics(np.concatenate(collected[variant]))
        gr00t['statistics'][original_key] = stats[original_key]
        for side, start in (('left',0), ('right',8)):
            modalities['state'][f'{variant}_{side}'] = dict(start=start, end=start+7,
                original_key=original_key, dtype='float32', rotation_type=None, absolute=True)
    # Grippers retain their original current-command columns and continuous transform.
    info.update(total_episodes=len(selected), total_frames=sum(lengths.values()),
        total_videos=len(selected)*3, splits={'train': f'0:{len(selected)}'})
    for name, value in [('info.json',info), ('modality.json',modalities), ('stats.json',stats), ('stats_gr00t.json',gr00t)]:
        save(DATASET/'meta'/name, value)
    episodes = [json.loads(line) for line in (source/'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
    (DATASET/'meta/episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in episodes if row['episode_index'] in lengths))
    (DATASET/'meta/tasks.jsonl').write_text((source/'meta/tasks.jsonl').read_text())
    # Keep canonical language metadata, including its source episode audit.
    if (source/'meta/task_language').exists():
        (DATASET/'meta/task_language').symlink_to(source/'meta/task_language', target_is_directory=True)
    split.update(dataset=str(DATASET), excluded_episode_ids=[])
    save(OUT/'split.json', split)
    # Spatial labels are unchanged and referenced, not regenerated from test scenes.
    old_labels = Path(OmegaConf.load(BASE/'gawm_joint.yaml').datasets.vla_data.spatial_supervision_dir)
    labels = OUT/'spatial_labels'; labels.mkdir()
    manifest = json.loads((old_labels/'manifest.json').read_text()); manifest['dataset'] = str(DATASET)
    save(labels/'manifest.json', manifest)
    for ep in selected:
        source_label = old_labels/f'episode_{ep:06d}.npz'
        if source_label.is_file():
            (labels/source_label.name).symlink_to(source_label)
    reference = OmegaConf.load(BASE/'gawm_joint.yaml').datasets.vla_data.normalization_statistics_path
    norm = json.loads(Path(reference).read_text())
    for variant in FEEDBACK:
        value = copy.deepcopy(norm)
        value['aloha']['state'] = stats[f'observation.{variant}_feedback']
        save(OUT/f'{variant}_statistics.json', value)
    report = dict(state='aligned_data_prepared', dataset=str(DATASET), source_dataset=str(source),
        source_split_sha256=digest(source_split), train_episodes=len(train), validation_episodes=len(split['validation_episode_ids']),
        unavailable_validation_episode_ids=unavailable_validation,
        validation_change_reason='Ten incoming-source episodes have no local raw HDF5 endpose. Both arms use the same remaining validation episodes; training and closed-loop scenes unchanged.',
        train_frames=sum(lengths[e] for e in train), state_statistics_train_only=True,
        action_statistics_source=str(reference), action_statistics_sha256=digest(reference),
        unchanged_action_contract='aloha_dual_joint_contgrip_next_recorded_14',
        endpose_contract='Native RoboTwin world EE position in meters + wxyz quaternion, not TCP; grips continuous',
        command_control='[L6,0,Lgrip,R6,0,Rgrip]; same width and fresh state modules',
        video_storage='Selected source videos symlinked without reencoding', episodes=rows)
    save(OUT/'data_manifest.json', report)
    prepare_cache()


def prepare_cache():
    # The shared loader filters an existing cache by split, but builds a missing
    # cache from its current split. Seed all selected episodes before either arm
    # opens the train subset so validation steps cannot disappear from the cache.
    episodes = [json.loads(line) for line in (DATASET/'meta/episodes.jsonl').read_text().splitlines()]
    steps = [(row['episode_index'], step) for row in episodes for step in range(row['length'])]
    path = DATASET/'meta/steps_data_index.pkl'
    if path.exists():
        old = path.with_suffix('.train_only_cache.bak')
        if not old.exists(): path.rename(old)
    with path.with_suffix('.tmp').open('wb') as stream:
        pickle.dump(dict(steps=steps,num_trajectories=len(episodes),total_steps=len(steps),
                         delete_pause_frame=False,source='Unfiltered feedback subset metadata'),stream)
    path.with_suffix('.tmp').replace(path)


def prepare_configs():
    assert json.loads((OUT/'data_manifest.json').read_text())['state'] == 'aligned_data_prepared'
    base = OmegaConf.to_container(OmegaConf.load(BASE/'gawm_joint.yaml'), resolve=True)
    for variant in FEEDBACK:
        for stage, steps in [('smoke',20), ('warmup',500), ('joint',4500)]:
            cfg = copy.deepcopy(base)
            cfg.update(run_id=f'{variant}_{stage}', run_root_dir=str(RUNS))
            data = cfg['datasets']['vla_data']
            data.update(data_mix=f'robotwin_ranking_rgb_feedback_{variant}',
                episode_split_manifest=str(OUT/'split.json'),
                normalization_statistics_path=str(OUT/f'{variant}_statistics.json'),
                spatial_supervision_dir=str(OUT/'spatial_labels'))
            spec = cfg['framework']['action_model']['embodiment_heads']['aloha']
            spec.update(state_dim=16, state_spec_id=('aloha_native_world_ee_xyz_wxyz_contgrip_16'
                if variant == 'endpose' else 'aloha_joint_command_padded_contgrip_16'))
            # Crop selection detaches coordinates. With zero auxiliary weights,
            # a fresh spatial state query has no effective training signal.
            # Restore the existing locator objective identically in both arms.
            cfg['framework']['spatial_focus'].update(heatmap_loss_weight=.002, coordinate_loss_weight=.02)
            t = cfg['trainer']
            t.update(pretrained_checkpoint=str(RUNS/('initial/pytorch_model.pt' if stage != 'joint'
                else f'{variant}_warmup/final_model/pytorch_model.pt')),
                reload_modules=None, max_train_steps=steps, num_warmup_steps=2 if stage=='smoke' else 50 if stage=='warmup' else 200)
            if stage == 'warmup':
                t['freeze_modules'] = ('backbone,world_model,visual_token_pooler,task_embedding,'
                    'embodiment_embedding,action_models.franka,action_models.oxe_bridge')
                # Spatial query also consumes changed state: warm it up in both arms.
            if stage == 'smoke':
                t.update(eval_interval=20, validation_samples_per_task=8, logging_frequency=1)
            OmegaConf.save(OmegaConf.create(cfg), OUT/f'{variant}_{stage}.yaml')
    protocol = json.loads((BASE/'protocol.json').read_text())
    protocol.update(variants=list(FEEDBACK), feedback_only=True, output_action_dim=14,
        execute_horizon=16, state_dim=16, initial_checkpoint_shared=True,
        common_spatial_supervision=dict(heatmap_loss_weight=.002,coordinate_loss_weight=.02),
        comparison_boundary='Feedback-only contrast between these two arms; both re-enable spatial supervision versus prior L1-only runs')
    save(OUT/'protocol.json', protocol)


def prepare_initial():
    import torch
    from starVLA.model.framework.base_framework import build_framework
    torch.set_num_threads(4); torch.manual_seed(42)
    cfg = OmegaConf.load(OUT/'command_smoke.yaml')
    model = build_framework(cfg)
    source = Path(OmegaConf.load(BASE/'gawm_warmup.yaml').trainer.pretrained_checkpoint)
    if not source.is_absolute():
        source = ROOT/source
    state = model.remap_checkpoint_state_dict(torch.load(source, map_location='cpu', weights_only=True, mmap=True))
    target = model.state_dict()
    reset_prefixes = ('action_models.aloha.state_projection.', 'spatial_focus.query.')
    reset = [k for k in target if k.startswith(reset_prefixes)]
    assert reset and set(state) == set(target)
    unexpected = [k for k in target if state[k].shape != target[k].shape and k not in reset]
    assert not unexpected, unexpected
    merged = {k: target[k] if k in reset else state[k] for k in target}
    model.load_state_dict(merged, strict=True)
    initial = RUNS/'initial'; initial.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), initial/'pytorch_model.pt')
    save(OUT/'initialization.json', dict(state='shared_initialization_verified', source=str(source),
        source_sha256=digest(source), checkpoint=str(initial/'pytorch_model.pt'),
        checkpoint_sha256=digest(initial/'pytorch_model.pt'), reset_keys=reset,
        inherited_keys=len(target)-len(reset), seed=42,
        note='Both arms use exactly the same tensors. Only state projection and state/task spatial query are reset.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['data','configs','initial','cache'], required=True)
    p.add_argument('--resume-incomplete', action='store_true')
    args = p.parse_args()
    if args.stage == 'data':
        prepare_data(resume=args.resume_incomplete)
    else:
        {'configs': prepare_configs, 'initial': prepare_initial, 'cache':prepare_cache}[args.stage]()
