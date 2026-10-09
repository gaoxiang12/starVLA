import json

import cv2
import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf

from starVLA.dataloader.robotwin_official_hdf5 import RoboTwinOfficialDataset


@pytest.fixture
def dataset(tmp_path):
    relative = 'data/pick_bottle/demo_clean/data/episode0.hdf5'
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    length = 40
    with h5py.File(path, 'w') as f:
        f['joint_action/vector'] = np.arange(length, dtype=np.float32)[:, None] + np.arange(14)[None, :]
        for key, dim, offset in [('left_endpose', 7, 0), ('left_gripper', 1, 7),
                                 ('right_endpose', 7, 8), ('right_gripper', 1, 15)]:
            f['endpose/' + key] = np.tile(np.arange(offset, offset+dim), (length, 1))
        rgb = f.create_dataset('observation/head_camera/rgb', (length,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
        for i in range(length):
            im = np.full((12, 16, 3), (200, 40, i), dtype=np.uint8)  # Simulator RGB, encoded directly upstream
            rgb[i] = cv2.imencode('.jpg', im)[1]
    report = dict(status='passed', task='pick_bottle', episodes=1, frames=length,
                  records=[dict(path=relative, frames=length)])
    (tmp_path / 'audits').mkdir()
    (tmp_path / 'audits/pick_bottle.json').write_text(json.dumps(report))
    (tmp_path / 'dataset_audit.json').write_text(json.dumps(dict(
        status='passed', tasks=1, episodes=1, frames=length, task_summaries=[report])))
    stats = {k: dict(min=[0]*d, max=[100]*d) for k, d in [('action', 14), ('state', 16)]}
    stats_path = tmp_path / 'stats.json'
    stats_path.write_text(json.dumps({'robotwin2': stats}))
    return RoboTwinOfficialDataset(OmegaConf.create(dict(data_root_dir=str(tmp_path),
        official_stats=str(stats_path), expected_frames=length)))


def test_native_state_action_order_and_alignment(dataset):
    sample = dataset[3]
    np.testing.assert_allclose(sample['state'], 2*np.arange(16)/100-1, atol=1e-7)
    expected = 2*(np.arange(3, 35)[:, None]+np.arange(14)[None, :])/100-1
    np.testing.assert_allclose(sample['action'], expected, atol=1e-7)
    assert sample['lang'] == 'pick bottle'
    assert sample['action_valid_mask'].all()
    assert sample['future_frame_valid_mask'].all()


def test_rgb_preprocessing_and_recorded_offsets(dataset):
    sample = dataset[0]
    pixels = [sample['image'][0], *(x[0] for x in sample['future_images'])]
    assert all(p.shape == (3, 240, 320) for p in pixels)
    restored = [(p.numpy()[:, 0, 0]*np.array([.229,.224,.225]) + np.array([.485,.456,.406]))*255 for p in pixels]
    for rgb, offset in zip(restored, [0,16,32]):
        np.testing.assert_allclose(rgb, [200,40,offset], atol=3)


def test_tail_masks_and_no_cross_episode_supervision(dataset):
    sample = dataset[38]
    assert sample['action_valid_mask'].sum() == 2
    assert sample['future_frame_valid_mask'].tolist() == [True, False, False]
    np.testing.assert_array_equal(sample['action'][2:], np.repeat(sample['action'][1:2],30,axis=0))
    for i in [-1, len(dataset)]:
        with pytest.raises(IndexError):
            dataset[i]


def test_audit_length_mismatch_fails(dataset):
    path = dataset.root / dataset.records[0][0]
    with h5py.File(path,'r+') as f:
        del f['joint_action/vector']
        f['joint_action/vector'] = np.zeros((39,14))
    with pytest.raises(ValueError, match='audit'):
        dataset[0]


def test_statistics_capture_training_semantics(dataset, tmp_path):
    path = tmp_path / 'dataset_statistics.json'
    dataset.save_dataset_statistics(path)
    saved = json.loads(path.read_text())['aloha']
    assert saved['num_transitions'] == 40
    assert saved['tail_supervision'] == 'masked'
    assert saved['action']['mask'] == [True]*14
    assert saved['future_recorded_offsets'] == [0,16,32]
    # Official min/max is not a clipping operation.
    assert (dataset.normalize(np.full(14,200), 'action') == 3).all()


def test_three_physical_views_preserve_order_and_future_offsets(dataset):
    path = dataset.root / dataset.records[0][0]
    with h5py.File(path, 'r+') as f:
        for camera, base in [('left_camera', (20, 180)), ('right_camera', (50, 80))]:
            stream = f.create_dataset(f'observation/{camera}/rgb', (40,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
            for i in range(40):
                rgb = np.full((12, 16, 3), (*base, i), dtype=np.uint8)
                stream[i] = cv2.imencode('.jpg', rgb)[1]
    dataset.cameras = ['head_camera', 'left_camera', 'right_camera']
    sample = dataset[0]
    assert sample['view_valid_mask'] == [True, True, True]
    for views, offset in zip([sample['image'], *sample['future_images']], [0, 16, 32]):
        assert len(views) == 3
        for pixels, rg in zip(views, [(200, 40), (20, 180), (50, 80)]):
            restored = (pixels.numpy()[:, 0, 0]*np.array([.229,.224,.225]) + np.array([.485,.456,.406]))*255
            np.testing.assert_allclose(restored, [*rg, offset], atol=3)


def test_missing_physical_views_are_not_padded_or_duplicated(dataset):
    dataset.cameras = ['head_camera', 'left_camera', 'right_camera']
    with pytest.raises(ValueError, match='Missing required cameras'):
        dataset[0]


def test_camera_frame_mismatch_fails(dataset):
    path = dataset.root / dataset.records[0][0]
    with h5py.File(path, 'r+') as f:
        f['observation/left_camera/rgb'] = np.zeros((2,), dtype='S2')
    dataset.cameras = ['head_camera', 'left_camera']
    with pytest.raises(ValueError, match='Camera frame count'):
        dataset[0]


def test_adjacent_frames_masks_events_and_main_sample_unchanged(dataset):
    baseline = dataset[3]
    dataset.temporal_neighbors = True
    sample = dataset[3]
    for key in ('state', 'action', 'action_valid_mask', 'future_frame_valid_mask'):
        np.testing.assert_array_equal(sample[key], baseline[key])
    for key in ('image', 'future_images'):
        def flat(x):
            for item in x:
                if isinstance(item, list): yield from flat(item)
                else: yield item.numpy()
        for a, b in zip(flat(sample[key]), flat(baseline[key])):
            np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(sample['temporal_neighbor_times'], [-1, 0, 1])
    assert sample['temporal_neighbor_valid'] and sample['temporal_event_weight'] == 1.
    for pixels, offset in zip(sample['temporal_neighbor_images'], [2, 4]):
        rgb = (pixels[0].numpy()[:, 0, 0]*np.array([.229,.224,.225]) + np.array([.485,.456,.406]))*255
        np.testing.assert_allclose(rgb, [200,40,offset], atol=3)
    assert not dataset[0]['temporal_neighbor_valid']
    assert not dataset[39]['temporal_neighbor_valid']
    # Either arm crossing the midpoint downweights the event; other channels do not.
    path = dataset.root / dataset.records[0][0]
    with h5py.File(path, 'r+') as f:
        f['joint_action/vector'][2:4, 13] = 40.
        f['joint_action/vector'][4:6, 13] = 60.
    assert dataset[3]['temporal_event_weight'] == .25
    assert dataset[15]['temporal_event_weight'] == 1.
