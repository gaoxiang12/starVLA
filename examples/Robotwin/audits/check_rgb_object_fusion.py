"""Real-input compatibility, gradient, and deployment-ablation preflight."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from examples.Robotwin.audits.spatial_ablation import apply_spatial_ablation
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new preflight output')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    cfg = OmegaConf.load(args.config)
    model = build_framework(cfg)
    source = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    source = model.remap_checkpoint_state_dict(source)
    missing, unexpected = model.load_state_dict(source, strict=False)
    allowed = ('spatial_focus.object_readout.', 'spatial_focus.early_fusion.')
    assert not unexpected and missing and all(k.startswith(allowed) for k in missing)
    target = model.state_dict()
    assert all(torch.equal(target[k], v) for k, v in source.items())
    retained = len(source)
    added_parameters = sum(p.numel() for name, p in model.named_parameters() if name in missing)
    del source, target
    cfg.datasets.vla_data.episode_split = 'train'
    child, = get_vla_dataset(cfg.datasets.vla_data, mode='train').datasets
    child.transforms.eval()
    pairs = sorted(zip(map(int, child.trajectory_ids), map(int, child.trajectory_lengths)))
    examples, records = [], []
    for (ep, length), anchor in zip(pairs[:2], [0, -5]):
        anchor = anchor if anchor >= 0 else length+anchor
        index = next(i for i, pair in enumerate(child.all_steps)
                     if int(pair[0]) == ep and int(pair[1]) == anchor)
        sample = child[index]
        examples.append(sample)
        action_valid = np.asarray(sample.get('action_valid_mask', np.ones(len(sample['action']), dtype=bool)))
        if anchor == length-5:
            assert action_valid.sum() < len(sample['action']), 'Tail example must exercise action padding'
        records.append(dict(episode=ep, anchor=anchor,
            native_head_sha256=hashlib.sha256(np.asarray(sample['native_images'][0]).tobytes()).hexdigest(),
            valid_actions=int(action_valid.sum())))
    model.cuda().eval()
    output = model(examples)
    assert 'object_action_l1' in output
    assert all(torch.isfinite(v).all() for v in output.values() if torch.is_tensor(v))
    parameters = dict(early_fusion=model.spatial_focus.early_fusion.project.weight,
        shared_decoder=model.action_models['aloha'].decoder.layers[0].multihead_attn.in_proj_weight,
        object_heatmap=model.spatial_focus.object_readout.heatmaps.weight)
    gradients = torch.autograd.grad(output['action_loss'], list(parameters.values()))
    norms = {name:float(grad.norm()) for name, grad in zip(parameters, gradients)}
    assert all(np.isfinite(v) and v > 0 for v in norms.values())
    metrics = {name:float(value) for name, value in output.items() if value.numel() == 1}
    del output, gradients
    predict_examples = [{k:v for k,v in sample.items() if k not in ('action', 'action_valid_mask')
                        and not k.startswith('future') and not k.startswith('spatial_')} for sample in examples]
    original_refine = model.spatial_focus.refine_queries
    predictions = {}
    # The detector labels must not enter the deployed inference path.
    with patch('starVLA.dataloader.rgb_object_supervision_core.candidates_with_core',
               side_effect=AssertionError('Training label helper reached inference')):
        for mode in ('full', 'no_objects', 'full'):
            model.spatial_focus.refine_queries = original_refine
            apply_spatial_ablation(model, mode)
            action = np.asarray(model.predict_action(predict_examples)['normalized_actions'])
            assert np.isfinite(action).all() and action.shape == (2, 16, 14)
            if mode in predictions:
                np.testing.assert_array_equal(action, predictions[mode])
            else:
                predictions[mode] = action
    difference = float(np.abs(predictions['full']-predictions['no_objects']).max())
    assert difference > 0
    report = dict(state='complete', source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        old_tensors_retained_exact=retained, missing_new_keys=missing,
        added_parameters=added_parameters, total_parameters=sum(p.numel() for p in model.parameters()),
        inputs=records, metrics=metrics, loss_gradient_norms=norms,
        no_objects_max_normalized_action_difference=difference, restored_full_exact=True,
        training_labels_unreachable_in_inference=True,
        note='Real GPU forward/backward and model-level inference preflight, not optimizer training '
             'or task evaluation. New memory intentionally changes initialization actions; source '
             'weights themselves remain exact. Tail action padding retained and masked.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('missing_new_keys', 'metrics')}, indent=2))


if __name__ == '__main__':
    main()
