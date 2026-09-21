"""Exercise real recovery-data transforms, timing, masks, and RGB decoding."""
import json
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from examples.Robotwin.audits.convert_rgb_recovery_pilot import ROOT, CAMP, OUTPUT
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def main():
    destination = CAMP / 'loader_audit.json'
    assert not destination.exists()
    run = ROOT / 'playground/Checkpoints/gawm_rgb_focus_local_v2_5k_20260907'
    cfg = OmegaConf.load(run / 'config.full.yaml').datasets.vla_data
    cfg.data_root_dir = str(OUTPUT.parents[2])
    cfg.episode_split_manifest = None
    cfg.validation_episode_stride = 0
    cfg.episode_split = 'train'
    cfg.spatial_supervision_dir = str(CAMP / 'converted_labels')
    cfg.normalization_statistics_path = str(run / 'dataset_statistics.json')
    cfg.num_workers = 0
    cfg.event_sampling_probability = 0
    # This is a deterministic audit of train-only data, not a validation score.
    mix = get_vla_dataset(cfg, mode='validation')
    child, = mix.datasets
    child.transforms.eval()
    assert set(child.trajectory_ids) == {0}
    norm = PolicyNormProcessor(str(run / 'final_model/pytorch_model.pt'), unnorm_key='aloha')
    result = json.loads((CAMP / 'cases/source_000003/result.json').read_text())
    permutation = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
    anchors = [0, 1, 10, 22, 36, 72, 128, 256, 400, 518, 522, 528, 533, 534]
    rows, videos = [], []
    with h5py.File(result['hdf5']) as raw:
        commands = raw['joint_action/vector'][:].astype(np.float32)
        n = len(commands)
        for video in sorted(OUTPUT.glob('videos/*/*/*.mp4')):
            with av.open(str(video)) as container:
                count = sum(1 for _ in container.decode(video=0))
            assert count == n
            videos.append(dict(path=str(video), decoded_frames=count))
        assert len(videos) == 3
        for anchor in anchors:
            data = child.get_step_data(0, anchor)
            selected = np.concatenate([np.asarray(data[k]).reshape(16, -1).copy()
                                       for k in norm.action_keys], axis=-1)
            valid = anchor + np.arange(1, 17) < n
            expected = commands[np.minimum(anchor+np.arange(1, 17), n-1)][:, permutation]
            np.testing.assert_allclose(selected[valid], expected[valid], atol=1e-6, rtol=0)
            sample = child._pack_sample(child.transforms(data))
            sample = child._attach_action_validity(sample, 0, anchor)
            sample = child._attach_future_frame_validity(sample, 0, anchor)
            sample = child._attach_spatial_supervision(sample, 0, anchor)
            np.testing.assert_array_equal(sample['action_valid_mask'], valid)
            offsets = np.asarray(child.delta_indices[child.modality_keys['video'][0]])
            np.testing.assert_array_equal(offsets, [0, 6, 12])
            np.testing.assert_array_equal(sample['future_frame_valid_mask'], anchor+offsets < n)
            assert sample['lang'] == 'blocks ranking rgb'
            assert sample['action'].shape == (16, 14) and sample['state'].shape == (1, 14)
            assert np.isfinite(sample['action']).all() and np.isfinite(sample['state']).all()
            unnormalized = norm.unapply_actions(np.asarray(sample['action'], dtype=np.float32))
            error = np.abs(unnormalized[valid]-expected[valid])
            pixel_errors = []
            for v, camera in enumerate(('head_camera', 'left_camera', 'right_camera')):
                def image_at(index):
                    return cv2.imdecode(np.frombuffer(raw[f'observation/{camera}/rgb'][index], np.uint8), cv2.IMREAD_COLOR)
                native = np.asarray(sample['native_images'][v])
                truth = image_at(anchor)
                assert native.shape == truth.shape == (240, 320, 3)
                mae = float(np.abs(native.astype(float)-truth).mean())
                assert mae < 4, (anchor, camera, mae)
                pixel_errors.append(mae)
                resized = np.asarray(Image.fromarray(truth).resize((224, 224)))
                assert np.abs(np.asarray(sample['image'][v]).astype(float)-resized).mean() < 4
                for h, offset in enumerate((6, 12)):
                    future_truth = np.asarray(Image.fromarray(image_at(min(anchor+offset, n-1))).resize((224, 224)))
                    assert np.abs(np.asarray(sample['future_images'][h][v]).astype(float)-future_truth).mean() < 4
            rows.append(dict(anchor=anchor, valid_actions=int(valid.sum()),
                             future_frame_valid_mask=sample['future_frame_valid_mask'].tolist(),
                             max_unnormalized_quantization_error=float(error.max()) if error.size else None,
                             native_rgb_mae=pixel_errors))
        # Also exercise the public mixture sampling path.
        for i in range(4):
            sample = mix[(0, i)]
            assert sample['action'].shape == (16, 14) and sample['lang'] == 'blocks ranking rgb'
    report = dict(state='complete', dataset=str(OUTPUT), frames=n, videos=videos, anchors=rows,
                  normalization_statistics_path=cfg.normalization_statistics_path,
                  split='Train-only supplement; local episode 0 maps to original train episode 3. Original validation remains separate.',
                  training_started=False,
                  note='Raw next-recorded alignment checked before quantization. No model inference or success-rate claim.')
    destination.write_text(json.dumps(report, indent=2) + '\n')
    provenance_path = CAMP / 'converted_dataset.json'
    provenance = json.loads(provenance_path.read_text())
    provenance.update(loader_audit_complete=True, loader_audit=str(destination))
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
