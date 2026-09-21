"""Verify saved real-trainer weights and post-update Cartesian gradient flow."""
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from examples.Robotwin.audits.prepare_grasp_precision_training import ROOT, OUT, digest
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def main():
    report_path = OUT/'trained_smoke_audit.json'
    assert not report_path.exists()
    original_path = ROOT/'playground/Checkpoints/gawm_rgb_focus_local_v2_5k_20260907/final_model/pytorch_model.pt'
    original = torch.load(original_path, map_location='cpu', weights_only=True)
    reports = {}
    for variant in ('joint', 'cartesian'):
        status = json.loads((OUT/f'{variant}_smoke_status.json').read_text())
        assert status['state'] == 'trainer_completed_checkpoint_audit_pending'
        path = Path(status['final_checkpoint'])
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert all(torch.isfinite(v).all() for v in state.values() if torch.is_floating_point(v))
        rows = [json.loads(line) for line in (path.parents[1]/'metrics.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == list(range(1,21))
        keys = ['action_dit_loss', 'l1_action_loss', 'continuous_action_l1', 'gripper_action_l1',
                'latent_loss', 'delta_to_copy_ratio', 'delta_direction_cosine']
        assert all(np.isfinite([row[key] for key in keys]).all() for row in rows)
        projection = [k for k in original if k.startswith('action_models.aloha.action_projection.')]
        changes = {k:float((state[k].float()-original[k].to(state[k].dtype).float()).abs().max()) for k in projection}
        if variant == 'cartesian':
            assert all(value == 0 for value in changes.values()), 'Frozen joint projection changed'
        else:
            assert any(value > 0 for value in changes.values()), 'Control joint head never updated'
        reports[variant] = dict(checkpoint=str(path), checkpoint_sha256=digest(path), steps=20,
            weight_dtypes=sorted(set(str(v.dtype) for v in state.values() if torch.is_floating_point(v))),
            joint_projection_max_changes_vs_quantized_source=changes,
            last_training_metrics={key:rows[-1][key] for key in keys},
            saved_training_state=str(path.parents[1]/'checkpoints/steps_20_training_state'))
    del original
    cfg = OmegaConf.load(OUT/'cartesian_smoke.yaml')
    torch.manual_seed(42)
    model = build_framework(cfg)
    model.load_state_dict(state, strict=True)
    model.cuda().float().eval()
    head = model.action_models['aloha']
    assert head.pose_projection.layers[-1].weight.abs().sum() > 0
    mixture = get_vla_dataset(cfg.datasets.vla_data, mode='validation')
    examples = [mixture[(dataset_index, 0)] for dataset_index in (0, 1)]
    inputs = [{k:v for k,v in ex.items() if k not in ('action', 'action_valid_mask')
        and 'future' not in k and not k.startswith(('spatial_', 'pregrasp_'))} for ex in examples]
    with torch.inference_mode():
        predicted = model.predict_action(inputs)
    assert np.isfinite(predicted['normalized_actions']).all()
    assert predicted['normalized_actions'].shape == (2,16,14)
    model.zero_grad(set_to_none=True)
    losses = model(examples=examples, optimizer_step=20)
    losses['action_loss'].backward()
    gradients = {name:float(p.grad.float().norm()) for name,p in head.pose_projection.named_parameters() if p.grad is not None}
    assert len(gradients) == 6 and all(np.isfinite(v) and v > 0 for v in gradients.values())
    assert all(p.grad is None for p in head.action_projection.parameters())
    reports['cartesian'].update(post_update_pose_head_gradient_norms=gradients,
        reload_inference_ik_converged_fraction=float(predicted['cartesian_ik_converged'].mean()),
        post_update_probe_losses={k:float(v.detach()) for k,v in losses.items() if torch.is_tensor(v) and v.numel()==1})
    report = dict(state='real_trainer_smokes_verified', variants=reports,
        total_smoke_optimizer_steps=40, further_optimizer_steps_in_this_audit=0,
        primary_rollout_precision_evaluated=False,
        note='Both real BF16/DeepSpeed trainers completed 20 updates and held-out action validation. Saved weights reload; early Cartesian layers receive finite nonzero gradients. Teacher-state action errors are not closed-loop pregrasp errors.')
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
