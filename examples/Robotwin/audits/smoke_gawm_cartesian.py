"""Load real GAWM weights and verify Cartesian forward/backward without training."""
import argparse
import json
from pathlib import Path
import time
import types

import numpy as np
from omegaconf import OmegaConf
import torch

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.WM4A.GAWM import GAWM
from starVLA.model.modules.action_model.ACT_ActionHeader import TurboStyleACTActionHead

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    run = ROOT/'playground/Checkpoints/gawm_rgb_focus_local_v2_5k_20260907'
    cfg = OmegaConf.load(run/'config.full.yaml')
    cfg.framework.name = 'GAWMCartesian'
    cfg.framework.cartesian_action = dict(
        urdf_path=str(ROOT.parent/'RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf'),
        translation_scale_m=.08, rotation_scale_rad=.35, ik_iterations=12,
        position_weight=.1, rotation_weight=.02, ik_residual_weight=.1)
    cfg.datasets.vla_data.data_root_dir = '/data/gaoxiang/RoboTwinPregraspCorrections_20260909/pilot_converted'
    cfg.datasets.vla_data.episode_split_manifest = None
    cfg.datasets.vla_data.validation_episode_stride = 0
    cfg.datasets.vla_data.episode_split = 'train'
    cfg.datasets.vla_data.spatial_supervision_dir = None
    cfg.datasets.vla_data.normalization_statistics_path = str(run/'dataset_statistics.json')
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.event_sampling_probability = 0
    # This is a model-path smoke; legacy crop-coordinate labels are not supplied.
    cfg.framework.spatial_focus.teacher_warmup_steps = 0
    cfg.framework.spatial_focus.teacher_end_steps = 0
    OmegaConf.save(cfg, out/'config.yaml')
    mixture = get_vla_dataset(cfg.datasets.vla_data, mode='validation')
    child, = mixture.datasets
    child.transforms.eval()
    examples = []
    for episode, anchor in [(0, 2), (1, 4)]:
        sample = child._pack_sample(child.transforms(child.get_step_data(episode, anchor)))
        sample = child._attach_action_validity(sample, episode, anchor)
        sample = child._attach_future_frame_validity(sample, episode, anchor)
        # No crop-coordinate labels exist for this pilot yet. Explicit masks
        # disable that auxiliary supervision while preserving predicted crops.
        sample['spatial_target_xy'] = np.zeros((3, 2), dtype=np.float32)
        sample['spatial_target_valid'] = np.zeros(3, dtype=bool)
        # The indexed/public loader also supplies these standard schema fields.
        sample['robot_tag'] = 'aloha'
        sample['action_spec_id'] = child.action_spec_id
        sample['state_spec_id'] = child.state_spec_id
        examples.append(sample)
    model = build_framework(cfg)
    loaded = model.load_state_dict(torch.load(run/'final_model/pytorch_model.pt', map_location='cpu', weights_only=True), strict=False)
    assert not loaded.unexpected_keys
    assert loaded.missing_keys and all(k.startswith('action_models.aloha.pose_projection.') for k in loaded.missing_keys)
    model.cuda().float().eval()
    head = model.action_models['aloha']
    inputs = [{k:v for k,v in ex.items() if k not in ('action', 'action_valid_mask')
               and 'future' not in k and not k.startswith(('spatial_', 'pregrasp_'))} for ex in examples]
    start = time.monotonic()
    with torch.inference_mode():
        corrected = model.predict_action(inputs)
        # Same weights, same backbone, ordinary ACT projection for reference.
        head.predict_action = types.MethodType(TurboStyleACTActionHead.predict_action, head)
        try:
            baseline = GAWM.predict_action(model, inputs)['normalized_actions']
        finally:
            del head.predict_action
    warm_delta = np.abs(corrected['normalized_actions']-baseline)
    assert np.isfinite(corrected['normalized_actions']).all()
    model.zero_grad(set_to_none=True)
    losses = model(examples=examples, optimizer_step=0)
    assert all(torch.isfinite(value).all() for value in losses.values() if torch.is_tensor(value))
    losses['action_loss'].backward()
    gradients = {name:float(parameter.grad.float().norm()) for name, parameter in head.pose_projection.named_parameters()
                 if parameter.grad is not None}
    assert gradients and all(np.isfinite(value) for value in gradients.values())
    assert gradients['layers.2.weight'] > 0 and gradients['layers.2.bias'] > 0
    assert all(p.grad is None for p in head.action_projection.parameters())
    report = dict(state='forward_backward_verified', checkpoint=str(run/'final_model/pytorch_model.pt'),
        newly_initialized_keys=loaded.missing_keys, examples=2, optimizer_steps=0, weights_saved=False,
        parameter_count=sum(p.numel() for p in model.parameters()),
        added_pose_head_parameters=sum(p.numel() for p in head.pose_projection.parameters()),
        zero_correction_action_difference_mean=float(warm_delta.mean()),
        zero_correction_action_difference_max=float(warm_delta.max()),
        zero_correction_ik_converged_fraction=float(corrected['cartesian_ik_converged'].mean()),
        zero_correction_ik_residual_max_m=float(corrected['cartesian_ik_position_residual_m'].max()),
        losses={k:float(v.detach()) for k,v in losses.items() if torch.is_tensor(v) and v.numel()==1},
        pose_head_gradient_norms=gradients, elapsed_seconds=time.monotonic()-start,
        peak_cuda_memory_gb=torch.cuda.max_memory_allocated()/1e9,
        note='Real pretrained model and measured correction clips; full forward/backward only. '
             'No optimizer steps, checkpoint publication, rollout, or grasp-improvement claim. '
             'Pose supervision here targets expert next-command FK; phase-goal sidecars are not yet consumed.')
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
