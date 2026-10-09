"""Audited RoboTwin HDF5 frames for the public GAWM trainer.

One index is one recorded frame. Read-only HDF5 handles are opened inside each
sample, never shared between DataLoader processes. No upstream source checkout
or task embedding assets are required.
"""
import bisect
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from starVLA.task_language import canonical_task_text


class RoboTwinOfficialDataset(Dataset):
    tag = 'aloha'
    action_spec_id = 'robotwin_official_absolute_dual_joint_native_14'
    state_spec_id = 'robotwin_official_endpose_native_16'

    def __init__(self, data_cfg):
        self.root = Path(data_cfg.data_root_dir)
        self.epoch = 0
        self.temporal_neighbors = bool(data_cfg.get("temporal_neighbors", False))
        self.history_offset = int(data_cfg.get("history_recorded_offset", 0))
        self.temporal_state_metadata = bool(data_cfg.get("temporal_state_metadata", False))
        if self.history_offset not in (0, 16):
            raise ValueError("History must match the 16-step deployment observation interval")
        self.horizon = int(data_cfg.get('action_horizon', 32))
        self.offsets = np.asarray(data_cfg.get('future_recorded_offsets', [0, 16, 32]), dtype=np.int64)
        self.image_size = tuple(data_cfg.get('obs_image_size', [320, 240]))
        self.cameras = list(data_cfg.get('cameras', ['head_camera']))
        if (not self.cameras or len(set(self.cameras)) != len(self.cameras)
                or any(not isinstance(c, str) or not c or '/' in c for c in self.cameras)):
            raise ValueError('Expected unique, nonempty camera names')
        self.image_channel_order = data_cfg.get('image_channel_order', 'rgb')
        if self.image_channel_order not in ('rgb', 'bgr'):
            raise ValueError('image_channel_order must be rgb or bgr')
        if self.horizon < 1 or self.offsets.shape != (3,) or self.offsets[0] != 0 or np.any(np.diff(self.offsets) <= 0):
            raise ValueError('Expected a positive horizon and three increasing frame offsets starting at zero')
        audit_path = self.root / 'dataset_audit.json'
        audit = json.loads(audit_path.read_text())
        if audit['status'] != 'passed':
            raise ValueError('RoboTwin dataset audit has not passed')
        self.records, lengths, audit_hashes = [], [], {}
        for summary in sorted(audit['task_summaries'], key=lambda x: x['task']):
            task = summary['task']
            path = self.root / 'audits' / f'{task}.json'
            report = json.loads(path.read_text())
            audit_hashes[task] = hashlib.sha256(path.read_bytes()).hexdigest()
            records = report['records']
            if (report.get('status') != 'passed' or report['task'] != task
                    or len(records) != int(summary['episodes'])
                    or sum(int(r['frames']) for r in records) != int(summary['frames'])):
                raise ValueError(f'Inconsistent audit for {task}')
            # Fail before DDP startup if a requested physical view is absent.
            first_path = self.root / records[0]['path'] if records else None
            if first_path is not None and first_path.is_file():
                with h5py.File(first_path, 'r') as handle:
                    self._camera_streams(handle, int(records[0]['frames']), str(first_path))
            for record in records:
                relative = Path(record['path'])
                frames = int(record['frames'])
                if relative.is_absolute() or '..' in relative.parts or frames < 1:
                    raise ValueError(f'Invalid audit record: {record}')
                if not (self.root / relative).is_file():
                    raise FileNotFoundError(self.root / relative)
                self.records.append((str(relative), task, frames))
                lengths.append(frames)
        self.ends = np.cumsum(lengths, dtype=np.int64)
        if (len(set(r[0] for r in self.records)) != len(self.records)
                or len(self.records) != audit['episodes']
                or len(audit['task_summaries']) != audit['tasks']
                or not len(self.ends) or int(self.ends[-1]) != audit['frames']):
            raise ValueError('Dataset index does not match the root audit')
        expected = data_cfg.get('expected_frames')
        if expected is not None and len(self) != int(expected):
            raise ValueError(f'Dataset has {len(self)} frames, expected {expected}')
        stats_path = Path(data_cfg.get('official_stats', str(self.root / 'training_metadata/stat-500-all.json')))
        self.stats = json.loads(stats_path.read_text())['robotwin2']
        self.limits = {}
        for key, dim in [('action', 14), ('state', 16)]:
            low = np.asarray(self.stats[key]['min'], dtype=np.float32)
            high = np.asarray(self.stats[key]['max'], dtype=np.float32)
            if low.shape != (dim,) or high.shape != (dim,) or not np.isfinite([low, high]).all() or (high < low).any():
                raise ValueError(f'Invalid official {key} statistics')
            span = high - low
            self.limits[key] = low, np.where(span < 1e-6, 1., span)
        self.provenance = dict(
            dataset_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            task_audit_sha256=audit_hashes,
            official_stats_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            num_trajectories=len(self.records), num_transitions=len(self),
            normalization='official_min_max_no_clipping',
            action_spec_id=self.action_spec_id, state_spec_id=self.state_spec_id,
            future_recorded_offsets=self.offsets.tolist(), image_size=list(self.image_size),
            camera=self.cameras[0] if len(self.cameras) == 1 else None,
            cameras=self.cameras, image_channel_order=self.image_channel_order,
            jpeg_encoding_convention='opencv_imencode_on_simulator_rgb',
            tail_supervision='masked', task_language_mode='dataset_name',
        )

        if self.temporal_neighbors:
            self.provenance.update(temporal_neighbors=True, temporal_coordinate='recorded_frames',
                temporal_reference_dt=1., temporal_neighbor_offsets=[-1, 0, 1],
                temporal_event_proxy='either_arm_command_midpoint_crossing_within_two_frames',
                temporal_event_weight=.25)
        if self.history_offset:
            self.provenance.update(history_recorded_offset=self.history_offset, history_boundary="repeat_first_observation")

    def _temporal_metadata(self, actions, anchor, length):
        # Command midpoint crossings are an external gripper-event proxy, not
        # learned from latents. Both arms use official min/max coordinates.
        window = np.asarray(actions[max(0, anchor-2):min(length, anchor+3)], dtype=np.float32)
        grippers = self.normalize(window, 'action')[:, [6, 13]] > 0.
        event = bool(np.any(grippers[1:] != grippers[:-1]))
        return dict(temporal_neighbor_times=np.array([-1., 0., 1.], np.float32),
                    temporal_neighbor_valid=bool(0 < anchor < length-1),
                    temporal_event_weight=.25 if event else 1.)

    def __len__(self):
        return int(self.ends[-1])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def normalize(self, value, key):
        low, span = self.limits[key]
        value = np.asarray(value, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f'Non-finite RoboTwin {key}')
        return (2 * (value - low) / span - 1).astype(np.float32)

    def _camera_streams(self, handle, length, relative):
        missing = [c for c in self.cameras if f'observation/{c}/rgb' not in handle]
        if missing:
            available = list(handle.get('observation', {}).keys())
            raise ValueError(f'Missing required cameras {missing} in {relative}; available={available}')
        streams = [handle[f'observation/{c}/rgb'] for c in self.cameras]
        if any(len(stream) != length for stream in streams):
            raise ValueError(f'Camera frame count differs from audit: {relative}')
        return streams

    def _image(self, encoded):
        buffer = np.frombuffer(encoded, dtype=np.uint8) if isinstance(encoded, (bytes, np.bytes_)) else np.asarray(encoded, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Could not decode RoboTwin camera JPEG')
        # RoboTwin writes simulator RGB directly with cv2.imencode. Decoding
        # with OpenCV therefore restores physical RGB already (unlike a
        # conventionally encoded JPEG). Historical checkpoints trained with
        # an extra R/B swap can explicitly request image_channel_order=bgr.
        if self.image_channel_order == 'bgr':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.
        image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return torch.from_numpy(image.transpose(2, 0, 1).copy())

    def __getitem__(self, index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = bisect.bisect_right(self.ends, index)
        relative, task, length = self.records[episode]
        anchor = index - (int(self.ends[episode - 1]) if episode else 0)
        frame_indices = anchor + self.offsets
        stop = min(anchor + self.horizon, length)
        # Closing on every read also makes fork/persistent workers safe and
        # bounds file descriptors regardless of the number of trajectories.
        with h5py.File(self.root / relative, 'r') as handle:
            actions = handle['joint_action/vector']
            cameras = self._camera_streams(handle, length, relative)
            if actions.shape != (length, 14):
                raise ValueError(f'HDF5 length/schema differs from audit: {relative}')
            action = np.asarray(actions[anchor:stop], dtype=np.float32)
            action = np.concatenate([action, np.repeat(action[-1:], self.horizon-len(action), axis=0)])
            state = np.concatenate([np.asarray(handle[f'endpose/{key}'][anchor]).reshape(-1)
                                    for key in ('left_endpose', 'left_gripper', 'right_endpose', 'right_gripper')])
            if state.shape != (16,):
                raise ValueError(f'Expected 16 state dimensions: {relative}')
            images = [[self._image(rgb[min(int(i), length - 1)]) for rgb in cameras]
                      for i in frame_indices]
            temporal = {}
            if self.history_offset:
                anchors = (anchor-1, anchor, anchor+1) if self.temporal_neighbors else (anchor,)
                histories = [[self._image(rgb[max(0, min(i-self.history_offset, length-1))]) for rgb in cameras]
                             for i in anchors]
                temporal["temporal_history_images" if self.temporal_neighbors else "history_images"] = histories
            if self.temporal_neighbors:
                temporal.update(self._temporal_metadata(actions, anchor, length))
                temporal['temporal_neighbor_images'] = [
                    [self._image(rgb[max(0, min(i, length-1))]) for rgb in cameras]
                    for i in (anchor-1, anchor+1)]
                if self.temporal_state_metadata:
                    temporal["temporal_neighbor_states"] = np.stack([
                        self.normalize(np.concatenate([np.asarray(handle[f'endpose/{key}'][max(0,min(i,length-1))]).reshape(-1)
                            for key in ('left_endpose','left_gripper','right_endpose','right_gripper')]), 'state')
                        for i in (anchor-1,anchor,anchor+1)])
        return dict(
            **temporal,
            image=images[0], future_images=images[1:],
            state=self.normalize(state, 'state'), action=self.normalize(action, 'action'),
            action_valid_mask=(anchor + np.arange(self.horizon) < length),
            future_frame_valid_mask=frame_indices < length, view_valid_mask=[True] * len(self.cameras),
            lang=canonical_task_text(task), robot_tag=self.tag,
            action_spec_id=self.action_spec_id, state_spec_id=self.state_spec_id,
        )

    def save_dataset_statistics(self, path):
        statistics = {key: {**self.stats[key], 'mask': [True] * dim,
                            'normalization_modes': ['min_max'] * dim}
                      for key, dim in [('action', 14), ('state', 16)]}
        statistics.update(self.provenance)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({self.tag: statistics}, indent=2) + '\n')
