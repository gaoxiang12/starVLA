"""Decompose action and auxiliary gradients from the same training forward pass."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def group(name):
    for prefix, label in [('backbone.encoder.', 'vision_encoder'),
                          ('visual_token_pooler.', 'visual_pooler'),
                          ('action_models.aloha.', 'action_head'),
                          ('spatial_focus.', 'spatial_focus')]:
        if name.startswith(prefix):
            return label
    return 'remaining_parameters'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    cfg = OmegaConf.load(args.run/'config.full.yaml')
    cfg.datasets.vla_data.episode_split = 'validation'
    cfg.datasets.vla_data.episode_split_manifest = str(args.split_manifest.resolve())
    cfg.datasets.vla_data.normalization_statistics_path = str(args.run/'dataset_statistics.json')
    data = get_vla_dataset(cfg.datasets.vla_data, mode='validation')
    for child in data.datasets:
        child.transforms.eval()
    model = build_framework(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.cuda().train()
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params = [p for name, p in named]
    module = importlib.import_module('starVLA.model.framework.WM4A.GAWM')
    original = module.masked_action_l1_loss
    captured = []

    def capture(*items, **kwargs):
        loss = original(*items, **kwargs)
        captured.append(loss)
        return loss

    module.masked_action_l1_loss = capture
    rows = []
    try:
        for batch_index in range(4):
            torch.manual_seed(42+batch_index)
            examples = [data[(0, i)] for i in range(2*batch_index, 2*batch_index+2)]
            captured.clear()
            output = model(examples=examples, optimizer_step=args.step)
            assert len(captured) == 1
            action = captured[0]
            auxiliary = output['action_loss'] - action
            ga = torch.autograd.grad(action, params, retain_graph=True, allow_unused=True)
            gx = torch.autograd.grad(auxiliary, params, allow_unused=True)
            sums = {}
            for (name, param), a, x in zip(named, ga, gx):
                label = group(name)
                totals = sums.setdefault(label, torch.zeros(3, device='cuda'))
                if a is not None:
                    totals[0] += a.float().square().sum()
                if x is not None:
                    totals[1] += x.float().square().sum()
                if a is not None and x is not None:
                    totals[2] += (a.float()*x.float()).sum()
            groups = {}
            for label, totals in sums.items():
                assert torch.isfinite(totals).all()
                aa, xx, ax = totals.cpu().tolist()
                groups[label] = dict(action_gradient_l2=aa**.5, auxiliary_gradient_l2=xx**.5,
                    auxiliary_to_action_norm_ratio=(xx/aa)**.5 if aa > 0 else None,
                    cosine=ax/(aa*xx)**.5 if aa > 0 and xx > 0 else None)
            rows.append(dict(batch=batch_index, samples=2, action_loss=float(action.detach()),
                             auxiliary_loss=float(auxiliary.detach()), gradients=groups))
            del ga, gx, output, action, auxiliary
    finally:
        module.masked_action_l1_loss = original
    report = dict(checkpoint=str(args.checkpoint.resolve()), split_manifest=str(args.split_manifest.resolve()),
                  optimizer_step=args.step, samples=8, batches=rows,
                  note='No optimizer step or checkpoint mutation. Same-forward gradient decomposition; model training mode on held-out fixed images. FP32 parameters, encoder internal BF16 autocast. These are pre-clipping gradient norms on four small batches, not actual Adam updates or a task-success metric.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
