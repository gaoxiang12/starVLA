"""Build GAWM-L VTTs from the exact GAWM training episodes (read-only data).

PYTHONPATH=. .venv/bin/python scripts/prepare_gawm_vtt.py --config <yaml> --device cpu
Writes the configured task_vectors_path and an adjacent prepared config. No
validation/evaluation episodes are added, no normalization statistics replaced.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from omegaconf import OmegaConf
import torch
from transformers import AutoModel
from starVLA.model.gawm_config import migrate_gawm_config

from starVLA.model.modules.gawm_l_vision import rgb_pixels, canonical_vtt_key
from starVLA.task_language import canonical_task_text, configured_task_language_mode, resolve_task_language


def training_endpoints(cfg):
    """Yield canonical key, two primary-camera RGB images, and provenance."""
    data = cfg.datasets.vla_data
    if data.get('episode_split', 'train') != 'train':
        raise ValueError('VTT preparation accepts only the training split')
    if data.dataset_py == 'robotwin_official_hdf5':
        from starVLA.dataloader.robotwin_official_hdf5 import RoboTwinOfficialDataset
        dataset = RoboTwinOfficialDataset(data)
        camera = dataset.cameras[0]
        for relative, task, length in dataset.records:
            if length < 2:
                raise ValueError(f'Episode has no transition: {relative}')
            with h5py.File(dataset.root / relative, 'r') as handle:
                dataset._camera_streams(handle, length, relative)
                frames = []
                for idx in (0, length-1):
                    image = cv2.imdecode(np.frombuffer(handle[f'observation/{camera}/rgb'][idx], np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f'Invalid image: {relative}/{idx}')
                    if dataset.image_channel_order == 'bgr':
                        raise ValueError('GAWM-L VTT requires physical RGB training images')
                    frames.append(image)
            yield canonical_vtt_key(dataset.tag, canonical_task_text(task)), frames, dict(path=relative, frames=length, camera=camera)
    elif data.dataset_py == 'lerobot_datasets':
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset
        mixture = get_vla_dataset(data, seed=int(cfg.get('seed', 42)))
        for child in mixture.datasets:
            # all_steps reflects episode splits, corrupt-episode exclusions and
            # training-anchor restrictions used by the actual training loader.
            selected = {int(ep) for ep, _ in child.all_steps}
            lengths = dict(zip(map(int, child.trajectory_ids), map(int, child.trajectory_lengths)))
            if len(child.modality_keys['video']) != int(cfg.framework.world_model.num_views):
                raise ValueError(f'Physical camera count mismatch: {child.dataset_name}')
            mode = configured_task_language_mode(data, child.tag)
            for ep in sorted(selected):
                length = lengths[ep]
                if length < 2:
                    raise ValueError(f'Episode has no transition: {child.dataset_name}/{ep}')
                child.curr_traj_data = child.get_trajectory_data(ep)
                language = child.get_language(ep, child.modality_keys['language'][0], 0)[0]
                language = resolve_task_language(language, child.dataset_name, mode)
                camera = child.modality_keys['video'][0]
                frames = [child.get_video(ep, camera, idx)[0] for idx in (0, length-1)]
                if data.get('video_channel_order', 'rgb') == 'bgr':
                    frames = [np.ascontiguousarray(frame[..., ::-1]) for frame in frames]
                yield canonical_vtt_key(child.tag, language), frames, dict(dataset=child.dataset_name, episode=ep, frames=length, camera=camera)
    else:
        raise ValueError(f'Unsupported training loader: {data.dataset_py}')


@torch.no_grad()
def episode_difference(encoder, frames, size, device):
    pixels = rgb_pixels(frames, size).to(device=device, dtype=next(encoder.parameters()).dtype)
    cls = encoder(pixel_values=pixels, return_dict=True).last_hidden_state[:, 0].float()
    return (cls[1] - cls[0]).cpu().numpy()


def prepare(config, device='cpu'):
    cfg = migrate_gawm_config(OmegaConf.load(config))
    if cfg.framework.lang_cond.type != 'vtt' or cfg.framework.world_model.visual_frontend != 'gawm_l':
        raise ValueError('Use a GAWM-L vision + VTT config')
    output = Path(cfg.framework.lang_cond.task_vectors_path)
    if output.exists():
        raise FileExistsError(f'Choose a new task_vectors_path; refusing to overwrite {output}')
    wm = cfg.framework.world_model
    encoder = AutoModel.from_pretrained(wm.vision_encoder_path, local_files_only=True,
            torch_dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32).to(device).eval()
    encoder.requires_grad_(False)
    sums, counts, provenance = {}, defaultdict(int), defaultdict(list)
    for key, frames, record in training_endpoints(cfg):
        diff = episode_difference(encoder, frames, tuple(wm.gawm_l_image_size), device)
        if not np.isfinite(diff).all():
            raise ValueError(f'Nonfinite VTT feature: {record}')
        sums[key] = sums.get(key, np.zeros_like(diff, dtype=np.float64)) + diff
        counts[key] += 1
        record['endpoint_sha256'] = hashlib.sha256(b''.join(np.asarray(f).tobytes() for f in frames)).hexdigest()
        provenance[key].append(record)
        if sum(counts.values()) % 100 == 0:
            print(f'Processed {sum(counts.values())} training episodes / {len(counts)} tasks', flush=True)
    if not sums:
        raise ValueError('No training episodes')
    vectors = {key: (sums[key]/counts[key]).astype(np.float32).tolist() for key in sorted(sums)}
    payload = dict(format_version=1, split='train', vectors=vectors, provenance=dict(provenance),
        feature='mean(last_frame_CLS-first_frame_CLS), primary camera, last_hidden_state',
        encoder_path=str(wm.vision_encoder_path), image_size=list(wm.gawm_l_image_size),
        camera_names=list(wm.camera_names), config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(output)
    cfg.framework.lang_cond.task_names = sorted(vectors)
    prepared = output.with_suffix('.prepared.yaml')
    OmegaConf.save(cfg, prepared)
    print(f'Wrote {len(vectors)} VTTs from {sum(counts.values())} episodes: {output}', flush=True)
    print(f'Training config: {prepared}', flush=True)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    prepare(args.config, args.device)
