"""Validate correction clips through the shared loader, including phase boundaries."""
import argparse
import json
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from starVLA.dataloader.lerobot_datasets import get_vla_dataset

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    manifest = json.loads(args.manifest.read_text())
    dataset = Path(manifest['dataset'])
    run = ROOT/'playground/Checkpoints/gawm_rgb_focus_local_v2_5k_20260907'
    cfg = OmegaConf.load(run/'config.full.yaml').datasets.vla_data
    cfg.data_root_dir = str(dataset.parents[2])
    cfg.episode_split_manifest = None
    cfg.validation_episode_stride = 0
    cfg.episode_split = 'train'
    cfg.spatial_supervision_dir = None
    cfg.normalization_statistics_path = str(run/'dataset_statistics.json')
    cfg.num_workers = 0
    cfg.event_sampling_probability = 0
    mixture = get_vla_dataset(cfg, mode='validation')
    child, = mixture.datasets
    child.transforms.eval()
    assert set(child.trajectory_ids) == set(manifest['train_episode_ids'])
    norm = PolicyNormProcessor(str(run/'final_model/pytorch_model.pt'), unnorm_key='aloha')
    order = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
    records = []
    for record in manifest['episodes']:
        episode, n = record['episode_index'], record['frames']
        labels = np.load(Path(manifest['labels'])/f'episode_{episode:06d}.npz', allow_pickle=False)
        assert len(labels['sim_s']) == n and np.all(np.diff(labels['sim_s']) > 0)
        np.testing.assert_array_equal(labels['raw_frame_index'], record['raw_frame_mapping'])
        phase_changes = np.flatnonzero(np.diff(labels['phase']) != 0)+1
        anchors = sorted({0, 1, n-17, n-13, n-7, n-2, n-1, *phase_changes.tolist(),
                          *(phase_changes-1).tolist()})
        videos = []
        for video in sorted(dataset.glob(f'videos/*/*/episode_{episode:06d}.mp4')):
            with av.open(str(video)) as container:
                count = sum(1 for _ in container.decode(video=0))
            assert count == n
            videos.append(dict(path=str(video), decoded_frames=count))
        assert len(videos) == 3
        checks = []
        with h5py.File(Path(manifest['raw_index'])/'data'/f'episode{episode}.hdf5') as raw:
            commands = np.asarray(raw['joint_action/vector'], dtype=np.float32)
            assert commands.shape == (n, 14)
            for anchor in anchors:
                data = child.get_step_data(episode, anchor)
                selected = np.concatenate([np.asarray(data[k]).reshape(16, -1).copy() for k in norm.action_keys], axis=-1)
                valid = anchor+np.arange(1, 17) < n
                expected = commands[np.minimum(anchor+np.arange(1, 17), n-1)][:, order]
                np.testing.assert_allclose(selected[valid], expected[valid], atol=1e-6, rtol=0)
                sample = child._pack_sample(child.transforms(data))
                sample = child._attach_action_validity(sample, episode, anchor)
                sample = child._attach_future_frame_validity(sample, episode, anchor)
                np.testing.assert_array_equal(sample['action_valid_mask'], valid)
                np.testing.assert_array_equal(sample['future_frame_valid_mask'], anchor+np.asarray([0, 6, 12]) < n)
                assert sample['action'].shape == (16, 14) and sample['state'].shape == (1, 14)
                assert sample['lang'] == 'blocks ranking rgb'
                assert np.isfinite(sample['action']).all() and np.isfinite(sample['state']).all()
                reconstructed = norm.unapply_actions(np.asarray(sample['action'], dtype=np.float32))
                error = np.abs(reconstructed[valid]-expected[valid])
                image_errors = []
                for v, view in enumerate(('head_camera', 'left_camera', 'right_camera')):
                    def decode(i):
                        return cv2.imdecode(np.frombuffer(raw[f'observation/{view}/rgb'][i], np.uint8), cv2.IMREAD_COLOR)
                    truth = decode(anchor)
                    native = np.asarray(sample['native_images'][v])
                    assert native.shape == truth.shape == (240, 320, 3)
                    mae = float(np.abs(native.astype(float)-truth).mean())
                    assert mae < 4, (episode, anchor, view, mae)
                    image_errors.append(mae)
                    for h, offset in enumerate((6, 12)):
                        future = np.asarray(Image.fromarray(decode(min(anchor+offset, n-1))).resize((224, 224)))
                        assert np.abs(np.asarray(sample['future_images'][h][v]).astype(float)-future).mean() < 4
                checks.append(dict(anchor=anchor, phase=int(labels['phase'][anchor]),
                    sim_s=float(labels['sim_s'][anchor]), valid_actions=int(valid.sum()),
                    future_mask=sample['future_frame_valid_mask'].tolist(), native_rgb_mae=image_errors,
                    max_action_quantization_error=float(error.max()) if error.size else None))
        records.append(dict(episode_index=episode, videos=videos, checks=checks,
            phase_boundaries=phase_changes.tolist(), frames=n))
    for i in range(4):
        sample = mixture[(0, i)]
        assert sample['action'].shape == (16, 14) and sample['lang'] == 'blocks ranking rgb'
    report = dict(state='shared_bc_loader_verified', records=records,
        dataset=str(dataset), normalization_statistics_path=cfg.normalization_statistics_path,
        new_pose_supervision_loader_verified=False, training_enabled=False,
        note='Actual shared BC loader and transforms verified on all supplied train-only clips. '
             'Pose labels remain sidecars and are not consumed by an existing model. '
             'No training, no validation or learned grasp success claim.')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(state=report['state'], episodes=len(records),
        checked_anchors=sum(len(r['checks']) for r in records), frames=sum(r['frames'] for r in records))))


if __name__ == '__main__':
    main()
