"""Measure whether action loss reaches the waypoint selector on real inputs."""
import argparse
import importlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--branch', choices=('goal', 'objects'), default='goal')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = OmegaConf.load(args.run / 'config.full.yaml')
    cfg.datasets.vla_data.episode_split = 'validation'
    cfg.datasets.vla_data.normalization_statistics_path = str(args.run / 'dataset_statistics.json')
    data = get_vla_dataset(cfg.datasets.vla_data, mode='validation')
    for child in data.datasets:
        child.transforms.eval()
    model = build_framework(cfg)
    checkpoint = args.run / 'final_model/pytorch_model.pt'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.cuda().eval()
    focus = model.spatial_focus
    if args.branch == 'goal':
        assert focus.goal_readout is not None
        names = ('query', 'key', 'offset', 'goal_readout')
    else:
        assert focus.object_readout is not None
        names = tuple('object_readout.' + key for key in
                      ('encoder', 'heatmaps', 'visibility', 'project', 'identity', 'readout'))
    def group(name):
        return next((key for key in names if name.startswith(key + '.')), None)
    parameters = [(name, param) for name, param in focus.named_parameters()
                  if group(name) is not None and param.requires_grad]
    module = importlib.import_module('starVLA.model.framework.WM4A.GAWM')
    original_loss, original_refine = module.masked_action_l1_loss, focus.refine_queries
    captured = {}

    def capture(*positional, **kwargs):
        value = original_loss(*positional, **kwargs)
        captured['action'] = value
        captured['prediction'] = positional[0].detach().cpu()
        return value

    rows = []
    module.masked_action_l1_loss = capture
    try:
        for start in (0, 2, 4, 6):
            examples = [data[(0, i)] for i in range(start, start + 2)]
            row = dict(sample_indices=list(range(start, start + 2)), modes={})
            predictions = {}
            ablated = 'no_' + args.branch
            for mode in ('full', ablated):
                focus.refine_queries = (original_refine if mode == 'full' else lambda q, memories:
                                        original_refine(q, {k: v for k, v in memories.items() if k != args.branch}))
                output = model(examples=examples, optimizer_step=5000)
                grads = torch.autograd.grad(captured['action'], [p for _, p in parameters], allow_unused=True)
                squares = dict.fromkeys(names, 0.)
                for (name, _), grad in zip(parameters, grads):
                    if grad is not None:
                        assert torch.isfinite(grad).all()
                        squares[group(name)] += float(grad.float().square().sum())
                row['modes'][mode] = dict(action_l1=float(output['l1_action_loss']),
                                         gradient_norms={key: value**.5 for key, value in squares.items()})
                if args.branch == 'objects':
                    diagnostics = {key: float(value) for key, value in output.items() if key.startswith('object_')}
                    assert diagnostics and all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())
                    row['modes'][mode]['object_supervision'] = diagnostics
                predictions[mode] = captured['prediction']
                del output, grads
                captured.clear()
            for key in (('query', 'key', 'offset') if args.branch == 'goal' else names):
                assert row['modes'][ablated]['gradient_norms'][key] == 0
                assert row['modes']['full']['gradient_norms'][key] > 0
            row['max_normalized_action_change'] = float((predictions['full'] - predictions[ablated]).abs().max())
            assert row['max_normalized_action_change'] > 0
            rows.append(row)
    finally:
        module.masked_action_l1_loss = original_loss
        focus.refine_queries = original_refine
    args.output.write_text(json.dumps(dict(checkpoint=str(checkpoint.resolve()), branch=args.branch, batches=rows,
         passed=True, note='Action L1 gradients only; no spatial or world-model auxiliary loss gradients. '
         'Eight real validation inputs in four batches, no optimizer updates. Not success-rate evidence.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
