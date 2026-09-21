"""Audit every converted recovery episode against its original recorded commands/images."""
import argparse
import json
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from PIL import Image

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from examples.Robotwin.audits.convert_rgb_recovery_campaign import build_plan
from examples.Robotwin.audits.verify_rgb_recovery_sources import digest
from starVLA.dataloader.lerobot_datasets import DATASET_NAMED_MIXTURES, get_vla_dataset

ROOT = Path(__file__).resolve().parents[3]
PERMUTATION = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot-campaign', type=Path,
                        help='For the existing single-clip pilot whose conversion predates the generic plan.')
    args = parser.parse_args()
    assert not args.output.exists()
    converted = json.loads(args.plan.read_text())
    assert converted['state'] == 'converted_labels_ready'
    plan = build_plan(args.pilot_campaign) if args.pilot_campaign else converted
    assert not plan['validation_episode_ids']
    assert digest(plan['original_split']) == plan['original_split_sha256']
    split = json.loads(Path(plan['original_split']).read_text())
    records = plan['episodes']
    assert [r['episode_index'] for r in records] == plan['train_episode_ids'] == list(range(len(records)))
    assert len({r['source_episode'] for r in records}) == len(records)
    assert all(r['source_episode'] in split['train_episode_ids'] and
               r['source_episode'] not in split['validation_episode_ids'] for r in records)
    dataset = Path(converted['dataset'])
    info = json.loads((dataset / 'meta/info.json').read_text())
    metadata = [json.loads(s) for s in (dataset / 'meta/episodes.jsonl').read_text().splitlines()]
    assert len(records) == info['total_episodes'] == len(metadata)
    assert {r['episode_index']: r['frames'] for r in records} == {
        r['episode_index']: r['length'] for r in metadata}
    assert sum(r['frames'] for r in records) == info['total_frames'] == plan['frames']
    run = ROOT / 'playground/Checkpoints/gawm_rgb_focus_local_v2_5k_20260907'
    cfg = OmegaConf.load(run / 'config.full.yaml').datasets.vla_data
    # Process-local registration exercises the public loader without enabling a training mixture.
    cfg.data_mix = '_recovery_campaign_loader_audit'
    DATASET_NAMED_MIXTURES[cfg.data_mix] = [(dataset.name, 1.0, 'robotwin_continuous_next_wm')]
    cfg.data_root_dir = str(dataset.parent)
    cfg.dataset_options = {}
    cfg.episode_split_manifest = None
    cfg.validation_episode_stride = 0
    cfg.episode_split = 'train'
    cfg.spatial_supervision_dir = converted['labels']
    cfg.normalization_statistics_path = str(run / 'dataset_statistics.json')
    cfg.num_workers = 0
    cfg.event_sampling_probability = 0
    mix = get_vla_dataset(cfg, mode='validation')
    child, = mix.datasets
    child.transforms.eval()
    assert set(child.trajectory_ids) == set(plan['train_episode_ids'])
    norm = PolicyNormProcessor(str(run / 'final_model/pytorch_model.pt'), unnorm_key='aloha')
    results = []
    for record in records:
        episode, n = record['episode_index'], record['frames']
        assert digest(record['source_hdf5']) == record['source_hdf5_sha256']
        phases = json.loads(Path(record['frame_phases']).read_text())
        assert len(phases) == n
        with np.load(Path(converted['labels']) / f'episode_{episode:06d}.npz', allow_pickle=False) as labels:
            priority_verified = 'priority_anchors' in labels
            if priority_verified:
                expected_priority = [i for i, phase in enumerate(phases[:-1]) if phase in
                    {'handoff', 'open_empty_grippers', 'withdraw_after_empty_grasp', 'return_to_origin'}]
                np.testing.assert_array_equal(labels['priority_anchors'], expected_priority)
        boundaries = [i for i in range(1, n) if phases[i] != phases[i-1]]
        anchors = sorted(set([0, 1, n-17, n-13, n-7, n-2, n-1] +
                             np.linspace(0, n-1, 8, dtype=int).tolist() +
                             [j for i in boundaries for j in (i-1, i)]))
        fields = dict(episode_index=episode, episode_chunk=episode // info['chunks_size'])
        table = pq.read_table(dataset / info['data_path'].format(**fields))
        assert len(table) == n
        assert set(table['episode_index'].to_pylist()) == {episode}
        assert table['frame_index'].to_pylist() == list(range(n))
        with h5py.File(record['source_hdf5']) as raw:
            commands = raw['joint_action/vector'][:].astype(np.float32)
            assert commands.shape == (n, 14) and np.isfinite(commands).all()
            for key in ('action', 'observation.state'):
                np.testing.assert_allclose(np.stack(table[key].to_pylist()), commands, atol=1e-6, rtol=0)
            videos = []
            for video_key in child.modality_keys['video']:
                raw_key = child.lerobot_modality_meta.video[video_key.removeprefix('video.')].original_key
                video = dataset / info['video_path'].format(**fields, video_key=raw_key)
                with av.open(str(video)) as container:
                    count = sum(1 for _ in container.decode(video=0))
                assert count == n
                videos.append(dict(path=str(video), decoded_frames=count))
            assert len(videos) == 3
            rows = []
            for anchor in anchors:
                data = child.get_step_data(episode, anchor)
                expected = commands[np.minimum(anchor+np.arange(1, 17), n-1)][:, PERMUTATION]
                valid = anchor+np.arange(1, 17) < n
                selected = np.concatenate([np.asarray(data[k]).reshape(16, -1) for k in norm.action_keys], -1)
                np.testing.assert_allclose(selected[valid], expected[valid], atol=1e-6, rtol=0)
                sample = child._pack_sample(child.transforms(data))
                sample = child._attach_action_validity(sample, episode, anchor)
                sample = child._attach_future_frame_validity(sample, episode, anchor)
                sample = child._attach_spatial_supervision(sample, episode, anchor)
                np.testing.assert_array_equal(sample['action_valid_mask'], valid)
                np.testing.assert_array_equal(child.delta_indices[child.modality_keys['video'][0]], [0, 6, 12])
                np.testing.assert_array_equal(sample['future_frame_valid_mask'], anchor+np.array([0, 6, 12]) < n)
                assert sample['lang'] == 'blocks ranking rgb'
                assert sample['action'].shape == (16, 14) and sample['state'].shape == (1, 14)
                assert np.isfinite(sample['action']).all() and np.isfinite(sample['state']).all()
                assert sample['spatial_target_xy'].shape == (3, 2)
                error = np.abs(norm.unapply_actions(np.asarray(sample['action'], np.float32))[valid]-expected[valid])
                pixels = []
                for view, camera in enumerate(('head_camera', 'left_camera', 'right_camera')):
                    def image_at(index):
                        return cv2.imdecode(np.frombuffer(raw[f'observation/{camera}/rgb'][index], np.uint8), cv2.IMREAD_COLOR)
                    truth = image_at(anchor)
                    native = np.asarray(sample['native_images'][view])
                    assert native.shape == truth.shape == (240, 320, 3)
                    mae = float(np.abs(native.astype(float)-truth).mean())
                    assert mae < 4, (episode, anchor, camera, mae)
                    pixels.append(mae)
                    for offset, image in [(0, sample['image'][view])] + [
                            (offset, sample['future_images'][h][view]) for h, offset in enumerate((6, 12))]:
                        target = np.asarray(Image.fromarray(image_at(min(anchor+offset, n-1))).resize((224, 224)))
                        assert np.abs(np.asarray(image).astype(float)-target).mean() < 4
                rows.append(dict(anchor=anchor, phase=phases[anchor], valid_actions=int(valid.sum()),
                    max_unnormalized_quantization_error=float(error.max()) if error.size else None,
                    native_rgb_mae=pixels))
        results.append(dict(episode_index=episode, source_episode=record['source_episode'],
                            frames=n, phase_boundaries=boundaries, videos=videos, anchors=rows,
                            priority_anchors_verified=priority_verified))
    for i in range(4):
        example = mix[(0, i)]
        assert example['action'].shape == (16, 14) and example['lang'] == 'blocks ranking rgb'
    report = dict(state='complete', conversion_plan=str(args.plan.resolve()), dataset=str(dataset),
        episodes=results, frames=plan['frames'], original_split=plan['original_split'],
        normalization_statistics_path=cfg.normalization_statistics_path, training_enabled=False,
        note='Train-only data audit. All videos fully decoded; raw/Parquet commands checked in full; '
             'loader alignment/RGB/masks checked at phase boundaries, distributed anchors and episode tails. '
             'This does not certify a future training mixture or policy performance.')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(state='complete', episodes=len(results), frames=plan['frames'],
                         audited_anchors=sum(len(r['anchors']) for r in results))), flush=True)


if __name__ == '__main__':
    main()
