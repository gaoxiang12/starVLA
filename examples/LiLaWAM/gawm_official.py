"""Experiment B: native GAWM objective inside the LiLa reproduction training loop.

Reuse the frozen upstream HDF5 reader, frame inventory, interpolation and clamping.
The only extra observation is GAWM's intermediate future at recorded offset +16.
"""
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.utils.data import Dataset

from starVLA.task_language import resolve_task_language
from starVLA.model.framework.WM4A.LiLaWAM import preprocess_image, smooth_chunk


class GAWMOfficialDataset(Dataset):
    def __init__(self, upstream, dataset_module):
        self.upstream = upstream
        self.normalize_image = dataset_module._normalize_image
        self.all_episodes = upstream.all_episodes

    def __len__(self):
        return len(self.upstream)

    def __getitem__(self, idx):
        # Deliberately fail on read errors; never silently replace the sampled frame.
        ds = self.upstream
        global_idx = ds.valid_indices[idx]
        ep = ds.episode_metadata[np.searchsorted(ds._ep_end_bounds, global_idx, side='right')]
        anchor = global_idx - ep['global_start']
        indices, masks = ds._get_query_indices(anchor, ep['length'])
        indices['head_camera'] = [min(anchor + dt, ep['length'] - 1) for dt in (0, 16, 32)]
        data = ds._load_hdf5_data(ep['hdf5_path'], indices)
        pixels = torch.from_numpy(np.stack([self.normalize_image(im) for im in data['frame']['head_camera']])).float()
        result = dict(state=data['state'], action_sequence=data['action_sequence'],
                      pixel_values=pixels, task_name=ep['task_name'], **masks)
        for key in ('state', 'action_sequence', 'pixel_values'):
            if not torch.isfinite(result[key]).all():
                raise ValueError(f'Nonfinite {key} in {ep["hdf5_path"]} at {anchor}')
        return result


class OfficialNormalizer:
    def __init__(self, path):
        self.stats = json.loads(Path(path).read_text())['robotwin2']

    def limits(self, x, key):
        lo = torch.tensor(self.stats[key]['min'], device=x.device, dtype=x.dtype)
        hi = torch.tensor(self.stats[key]['max'], device=x.device, dtype=x.dtype)
        span = hi - lo
        span[span < 1e-6] = 1
        return lo, span

    def normalize(self, x, key):
        lo, span = self.limits(x, key)
        return 2 * (x - lo) / span - 1

    def denormalize(self, x):
        lo, span = self.limits(x, 'action')
        return (x + 1) / 2 * span + lo


def prepare_precision(policy, device):
    policy.to(device=device)
    # DINO's nonpersistent RoPE frequency buffers must retain their loaded fp32.
    for name, child in policy.named_children():
        if name != 'backbone':
            child.to(dtype=torch.bfloat16)
    return policy


class GAWMOfficialWrapper(nn.Module):
    def __init__(self, cfg, stats):
        super().__init__()
        from starVLA.model.framework.WM4A.GAWM import GAWM
        self.cfg = cfg
        # Avoid the inference-only Transformer fastpath's mixed-dtype restrictions.
        torch.backends.mha.set_fastpath_enabled(False)
        self.policy = prepare_precision(GAWM(cfg), 'cuda')
        self.normalizer = OfficialNormalizer(stats)

    def train(self, mode=True):
        super().train(mode)
        self.policy.backbone.encoder.eval()
        return self

    def forward(self, batch):
        # Match upstream conversion to bf16 BEFORE normalization, including tail repeats.
        states = self.normalizer.normalize(batch['state'].to('cuda', torch.bfloat16), 'state').float().cpu().numpy()
        actions = self.normalizer.normalize(batch['action_sequence'].to('cuda', torch.bfloat16), 'action').float().cpu().numpy()
        examples = []
        for i, task in enumerate(batch['task_name']):
            frames = batch['pixel_values'][i]
            examples.append(dict(lang=resolve_task_language(None, task, 'dataset_name'),
                robot_tag='aloha', state=states[i], action=actions[i], image=[frames[0]],
                future_images=[[frames[1]], [frames[2]]]))
        metrics = self.policy(examples)
        return metrics['action_loss'], metrics


def make_model(cfg, source, stats):
    wrapper = GAWMOfficialWrapper(cfg, stats)
    # Like LiLa, checkpoints/Adam cover the learned policy; DINO is a separate asset.
    core = nn.ModuleDict({name: child for name, child in wrapper.policy.named_children() if name != 'backbone'})
    return wrapper, core, wrapper.policy.backbone.encoder


class GAWMOfficialPolicy:
    """Same physical-action/endpose WebSocket contract as the LiLa evaluation."""
    def __init__(self, config):
        settings = config.framework.official_robotwin
        self.cfg = OmegaConf.load(settings.config_path)
        self.wrapper, core, _ = make_model(self.cfg, Path(settings.source_root), Path(settings.norm_stats_path))
        checkpoint = torch.load(settings.checkpoint_path, map_location='cpu', weights_only=True)
        core.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.wrapper.eval()
        self.metadata = dict(framework='GAWM_Experiment_B', ckpt_path=str(settings.checkpoint_path),
            action_chunk_size=32, execute_horizon=16, state_dim=16, action_dim=14,
            state_representation='endpose', action_order='left_arm6,left_gripper,right_arm6,right_gripper',
            action_mode='absolute_qpos', camera='head_camera', image_size=[320, 240],
            normalization='official_min_max', smooth_actions=True, policy_rng='simulator_cuda')

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        if len(examples) != 1:
            raise ValueError('Expected one observation per request')
        example = examples[0]
        task = example['task_name']
        if not task or Path(task).name != task:
            raise ValueError('Expected an explicit task ID')
        state = np.asarray(example['state'], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError('Expected finite 16D endpose state')
        rng = kwargs.get('cuda_rng_state')
        if rng is not None:
            rng = np.asarray(rng)
            if rng.dtype != np.uint8 or rng.ndim != 1:
                raise ValueError('Invalid simulator CUDA RNG state')
            torch.cuda.set_rng_state(torch.from_numpy(rng.copy()))
        state = self.wrapper.normalizer.normalize(torch.as_tensor(state, device='cuda', dtype=torch.bfloat16), 'state')
        pixels = torch.from_numpy(preprocess_image(example['image'][0], self.cfg.dataset.image_size))
        result = self.wrapper.policy.predict_action([dict(image=[pixels], state=state.float().cpu().numpy(),
            lang=resolve_task_language(None, task, 'dataset_name'), robot_tag='aloha')])
        normalized = torch.as_tensor(result['normalized_actions'], device='cuda', dtype=torch.bfloat16)
        actions = self.wrapper.normalizer.denormalize(normalized)[0].float().cpu().numpy()
        if self.cfg.inference.smooth_actions:
            actions = smooth_chunk(actions)
        if not np.isfinite(actions).all():
            raise ValueError('Nonfinite policy actions')
        result['actions'] = actions[None]
        if rng is not None:
            result['cuda_rng_state'] = torch.cuda.get_rng_state().cpu().numpy()
        return result
