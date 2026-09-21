"""Native StarVLA training/deployment for three-camera LiLa-WAM.

Consumes shared LeRobot normalized actions/state. The official one-camera
checkpoint adapter remains registered independently as ``LiLaWAM``.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoModel

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.lila.core import MultiViewLiLa, masked_mean
from starVLA.model.modules.lila.image_processing import resize_rgb
from starVLA.model.modules.action_model.action_loss import action_l1_diagnostics, masked_action_l1_loss
from starVLA.task_language import configured_task_language_mode, resolve_task_language


def plain_config(config):
    if hasattr(config, 'unwrap'):
        # Keep framework and deployment settings in the trainer's accessed-only
        # config.yaml, as well as in config.full.yaml.
        def visit(node):
            if hasattr(node, 'unwrap'):
                if OmegaConf.is_list(node.unwrap()):
                    for i in range(len(node)): visit(node[i])
                else:
                    for _, value in node.items(): visit(value)
        visit(config.framework)
        visit(config.datasets)
        config = config.unwrap()
    return OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)


def task_key(tag, language, data_config):
    mode = configured_task_language_mode(data_config, tag)
    # The shared loader already resolves dataset_name tasks to canonical text.
    language = resolve_task_language(language, language, mode)
    if not language.strip():
        raise ValueError('LiLa VTT requires a nonempty canonical task')
    return f'{tag}:{language}'


@FRAMEWORK_REGISTRY.register('LiLaWAMTrain')
class LiLaWAMTrain(baseframework):
    expects_normalized_state = True
    # Shared LeRobot _pack_sample uses Pillow's RGB BICUBIC default.
    image_resize_resample = 'bicubic'
    normalized_continuous_action_clip = 1.0

    def __init__(self, config, *, vision_encoder=None):
        super().__init__()
        self.settings = cfg = plain_config(config)
        model = cfg['framework']['lila']
        self.embodiment_head_specs = cfg['framework']['action_model']['embodiment_heads']
        self.data_config = cfg['datasets']['vla_data']
        self.image_size = tuple(model.get('image_size', [224, 224]))  # W,H
        self.image_resize_resample = model.get('image_resize_resample', 'bicubic')
        self.normalized_continuous_action_clip = model.get('normalized_continuous_action_clip', 1.0)
        self.tail_supervision = model.get('tail_supervision', 'mask')
        if self.tail_supervision not in ('mask', 'clamp'):
            raise ValueError('tail_supervision must be mask or clamp')
        self.smooth_actions = bool(model.get('smooth_actions', False))
        self.inference_time_grid = model.get('inference_time_grid', 'legacy')
        if self.inference_time_grid not in ('legacy', 'linspace'):
            raise ValueError('Unsupported inference_time_grid')
        self.action_execution_horizon = model.get('action_execution_horizon')
        self.num_views = int(model.get('num_views', 3))
        if self.num_views != 3:
            raise ValueError('This training integration requires three camera slots')
        self.encoder = vision_encoder if vision_encoder is not None else AutoModel.from_pretrained(
            model['vision_encoder_path'], local_files_only=True, torch_dtype=torch.float32)
        self.encoder.requires_grad_(False).eval()
        dim = self.encoder.config.hidden_size
        patch = self.encoder.config.patch_size
        if any(x % patch for x in self.image_size):
            raise ValueError('Image dimensions must be divisible by the encoder patch size')
        self.patch_count = (self.image_size[0] // patch) * (self.image_size[1] // patch)
        self.layers = list(model.get('feat_layers', [-12, -8, -4]))
        self.target_layer = int(model.get('future_target_layer', -4))
        self.encoder_batch_size = int(model.get('encoder_batch_size', 6))
        self.efficient_views = bool(model.get('efficient_views', False))
        self.future_weight = float(model.get('lambda_future_feat', .5))
        self.inference_steps = int(model.get('num_inference_steps', 10))
        if self.inference_steps < 1 or self.encoder_batch_size < 1:
            raise ValueError('Inference steps and encoder batch size must be positive')
        self.core = MultiViewLiLa(self.embodiment_head_specs, dim, len(self.layers), self.patch_count,
            hidden=model.get('hidden_dim', 768), depth=model.get('depth', 12),
            num_heads=model.get('num_heads', 8), adapter_depth=model.get('adapter_depth', 4),
            queries=model.get('queries_per_view', 64), future_depth=model.get('future_depth', 4),
            future_heads=model.get('future_heads', 4), num_views=self.num_views)
        self.core.efficient_views = self.efficient_views
        path = model.get('task_vectors_path')
        payload = json.loads(Path(path).read_text()) if path else model.get('task_vectors')
        if not payload or payload.get('format_version') != 1:
            raise ValueError('Prepare a version-1 training-only VTT file before training')
        vectors = payload['vectors']
        self.task_names = sorted(vectors)
        self.task_indices = {key: i for i, key in enumerate(self.task_names)}
        array = np.asarray([vectors[key] for key in self.task_names], dtype=np.float32)
        if array.shape != (len(self.task_names), dim) or not np.isfinite(array).all():
            raise ValueError('VTT dimension/nonfinite values do not match the visual encoder')
        self.register_buffer('task_vectors', torch.from_numpy(array))
        fingerprint = hashlib.sha256(json.dumps(self.task_names, ensure_ascii=False).encode()).digest()
        self.register_buffer('task_key_fingerprint', torch.tensor(list(fingerprint), dtype=torch.uint8))
        parameter_dtype = model.get('parameter_dtype', 'float32')
        if parameter_dtype not in ('float32', 'bfloat16'):
            raise ValueError('Unsupported LiLa parameter_dtype')
        self.to(dtype=getattr(torch, parameter_dtype))

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # A same-size but reordered/different VTT vocabulary must never silently
        # attach checkpoint vectors to a different set of task descriptions.
        saved = state_dict.get('task_key_fingerprint')
        if saved is None or not torch.equal(saved.cpu(), self.task_key_fingerprint.cpu()):
            raise ValueError('Checkpoint VTT task vocabulary differs from its configuration')
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Frozen targets stay deterministic under trainer.train().
        return self

    def _route(self, examples):
        tags = {x.get('robot_tag') for x in examples}
        if len(tags) != 1 or next(iter(tags)) not in self.embodiment_head_specs:
            raise ValueError('LiLa requires homogeneous batches with a registered robot_tag')
        tag = next(iter(tags))
        spec = self.embodiment_head_specs[tag]
        for ex in examples:
            for key in ('action_spec_id', 'state_spec_id'):
                if key in ex and ex[key] != spec[key]:
                    raise ValueError(f'{tag} {key} mismatch: {ex[key]} != {spec[key]}')
        return tag, spec

    def _pixels(self, frames, valid):
        images = []
        for row, mask in zip(frames, valid.cpu().tolist()):
            if len(row) > self.num_views:
                raise ValueError('Too many camera images')
            for v in range(self.num_views):
                if not mask[v]:
                    a = np.zeros((*self.image_size[::-1], 3), dtype=np.uint8)
                else:
                    if v >= len(row):
                        raise ValueError('Valid camera slot has no image')
                    im = row[v] if isinstance(row[v], Image.Image) else Image.fromarray(np.asarray(row[v]))
                    a = np.asarray(resize_rgb(im, self.image_size, self.image_resize_resample))
                images.append(a.copy())
        device = next(self.core.parameters()).device
        x = torch.from_numpy(np.stack(images)).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        return (x - mean) / std

    @torch.no_grad()
    def _encode(self, pixels, layers, valid=None):
        total = len(pixels)
        indices = None
        if self.efficient_views and valid is not None:
            indices = valid.flatten().nonzero().flatten()
            # One shape-probe frame handles an all-invalid future batch; its
            # features are discarded, and no target contributes to the loss.
            pixels = pixels[indices] if len(indices) else pixels[:1]
        unique_layers = list(dict.fromkeys(layers))
        output = [[] for _ in unique_layers]
        for chunk in pixels.split(self.encoder_batch_size):
            hidden = self.encoder(pixel_values=chunk.to(next(self.encoder.parameters()).dtype),
                                  output_hidden_states=True).hidden_states
            for dst, layer in zip(output, unique_layers):
                dst.append(hidden[layer])
        encoded = {}
        for layer, pieces in zip(unique_layers, output):
            value = torch.cat(pieces)
            if indices is not None:
                dense = value.new_zeros(total, *value.shape[1:])
                if len(indices): dense.index_copy_(0, indices, value)
                value = dense
            encoded[layer] = value
        return [encoded[layer] for layer in layers]

    def _inputs(self, examples):
        tag, spec = self._route(examples)
        device, dtype = self.task_vectors.device, next(self.core.parameters()).dtype
        views = []
        for ex in examples:
            count = len(ex['image'])
            # A three-slot deployment must explicitly mask a missing LIBERO view.
            default = [v < min(count, spec['num_observed_views']) for v in range(3)]
            mask = ex.get('view_valid_mask', default)
            if len(mask) != 3 or not any(mask):
                raise ValueError('view_valid_mask must contain three flags and a real camera')
            views.append(mask)
        valid = torch.as_tensor(views, device=device, dtype=torch.bool)
        raw_state = []
        for ex in examples:
            s = np.asarray(ex['state'], dtype=np.float32)
            if s.ndim == 1:
                s = s[None]
            if s.shape != (1, spec['state_dim']) or not np.isfinite(s).all():
                raise ValueError(f'Expected current normalized state [1,{spec["state_dim"]}]')
            raw_state.append(s)
        state = torch.as_tensor(np.stack(raw_state), device=device, dtype=dtype)
        ids = [self.task_indices[task_key(tag, ex['lang'], self.data_config)] for ex in examples]
        task = self.task_vectors[ids].to(dtype)
        pixels = self._pixels([ex['image'] for ex in examples], valid)
        encoded = self._encode(pixels, self.layers + [self.target_layer], valid)
        features = [f.reshape(len(examples), 3, *f.shape[1:]).to(dtype) for f in encoded[:-1]]
        current_patches = encoded[-1][:, -self.patch_count:].reshape(len(examples), 3, self.patch_count, -1)
        visual = self.core.visual_tokens(features, valid)
        return tag, spec, state, task, valid, visual, current_patches

    def forward(self, examples, **kwargs):
        tag, spec, state, task, valid, visual, current_patches = self._inputs(examples)
        device, dtype = state.device, state.dtype
        action = torch.as_tensor(np.asarray([x['action'] for x in examples]), device=device, dtype=dtype)
        if action.shape != (len(examples), spec['action_horizon'], spec['action_dim']):
            raise ValueError('Action dimensions/horizon do not match semantic head')
        mask = torch.as_tensor(np.asarray([x['action_valid_mask'] for x in examples]), device=device, dtype=torch.bool)
        if mask.shape != action.shape[:2] or not mask.any():
            raise ValueError('Action mask has the wrong shape or no valid targets')
        if self.tail_supervision == 'clamp':
            # Dedicated recipe reproduction: the shared loader repeats absolute
            # terminal commands. Default masked training remains unchanged.
            mask = torch.ones_like(mask)
        if not torch.isfinite(action[mask]).all():
            raise ValueError('Nonfinite valid actions')
        # Padding must not influence valid velocities through bidirectional attention.
        action = action.masked_fill(~mask[..., None], 0)
        noise = torch.randn_like(action).masked_fill(~mask[..., None], 0)
        t = torch.randn(len(examples), device=device, dtype=dtype).sigmoid()
        xt = (1 - t[:, None, None]) * noise + t[:, None, None] * action
        velocity, cond, cond_mask = self.core(xt, t, state, visual, task, valid, tag, mask)
        flow = masked_mean((velocity.float() - (action - noise).float()).square(), mask[..., None])
        future_loss = flow * 0
        future_metrics = {}
        if self.future_weight:
            temporal = torch.as_tensor(np.asarray([x['future_frame_valid_mask'] for x in examples]),
                                       device=device, dtype=torch.bool)
            if temporal.shape != (len(examples), 2):
                raise ValueError('LiLa expects current + one future frame validity flag')
            if self.tail_supervision == 'clamp':
                temporal = torch.ones_like(temporal)
            future_valid = valid & temporal[:, 1:2]
            future_frames = []
            for ex in examples:
                if len(ex['future_images']) != 1:
                    raise ValueError('LiLa expects exactly one future image set')
                future_frames.append(ex['future_images'][0])
            pixels = self._pixels(future_frames, future_valid)
            target = self._encode(pixels, [self.target_layer], future_valid)[0][:, -self.patch_count:]
            target = target.reshape(len(examples), 3, self.patch_count, -1)
            pred = self.core.predict_future(cond, cond_mask, future_valid)
            cosine = 1 - F.cosine_similarity(pred.float(), target.float(), dim=-1)
            future_loss = masked_mean(cosine, future_valid[..., None])
            with torch.no_grad():
                predicted_delta = pred.float() - current_patches.float()
                target_delta = target.float() - current_patches.float()
                predicted_rms = masked_mean(predicted_delta.square(), future_valid[..., None, None]).sqrt()
                target_rms = masked_mean(target_delta.square(), future_valid[..., None, None]).sqrt()
                direction = F.cosine_similarity(predicted_delta, target_delta, dim=-1)
                direction_valid = future_valid[..., None] & (target_delta.norm(dim=-1) > 1e-6)
                future_metrics = {'delta_to_copy_ratio': predicted_rms / target_rms.clamp_min(1e-6),
                                  'delta_direction_cosine': masked_mean(direction, direction_valid)}
        # One-step reconstruction is a diagnostic, not the training objective.
        l1 = masked_mean((velocity.float() - (action-noise).float()).abs(), mask[..., None])
        total = flow + self.future_weight * future_loss
        estimated = (xt + (1-t[:, None, None]) * velocity).detach().float()
        diagnostics = action_l1_diagnostics(estimated, action.float(), mask,
                                            gripper_indices=spec.get('gripper_indices', ()))
        diagnostics['l1_action_loss'] = masked_action_l1_loss(estimated, action.float(), mask)
        return {'action_loss': total, 'flow_matching_loss': flow.detach(),
                'latent_loss': future_loss.detach(), 'latent_cosine_loss': future_loss.detach(),
                'flow_velocity_l1': l1.detach(), f'flow_matching_loss/{tag}': flow.detach(),
                **diagnostics, **future_metrics}

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        tag, spec, state, task, valid, visual, _ = self._inputs(examples)
        generator = kwargs.get('generator')
        x = torch.randn(len(examples), spec['action_horizon'], spec['action_dim'],
                        device=state.device, dtype=state.dtype, generator=generator)
        times = torch.linspace(0, 1, self.inference_steps + 1, device=x.device, dtype=x.dtype)
        for i in range(self.inference_steps):
            t = (times[i].expand(len(examples)) if self.inference_time_grid == 'linspace'
                 else x.new_full((len(examples),), i / self.inference_steps))
            velocity, _, _ = self.core(x, t, state, visual, task, valid, tag)
            x = (x + velocity * (times[i + 1] - times[i]) if self.inference_time_grid == 'linspace'
                 else x + velocity / self.inference_steps)
        # Match the q99_clip=1 training target domain after integration.
        # Leave grippers to the shared binary/unit_interval inverse transform.
        continuous = torch.ones(spec['action_dim'], dtype=torch.bool, device=x.device)
        continuous[list(spec.get('gripper_indices', ()))] = False
        if self.normalized_continuous_action_clip is not None:
            x = torch.where(continuous, x.clamp(-self.normalized_continuous_action_clip,
                                              self.normalized_continuous_action_clip), x)
        return {'normalized_actions': x.float().cpu().numpy()}

    def postprocess_actions(self, actions):
        # Smooth in physical coordinates, after the shared inverse transform.
        if not self.smooth_actions:
            return actions
        from starVLA.model.framework.WM4A.LiLaWAM import smooth_chunk
        return np.stack([smooth_chunk(chunk) for chunk in actions])
