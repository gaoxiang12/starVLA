"""Verify exact checkpoint migration and real-input initial policy preservation."""
import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--branch', choices=('goal', 'objects'), default='goal')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = OmegaConf.load(args.config)
    cfg.datasets.vla_data.episode_split = 'validation'
    data = get_vla_dataset(cfg.datasets.vla_data, mode='validation')
    for child in data.datasets:
        child.transforms.eval()
    model = build_framework(cfg)
    source = torch.load(cfg.trainer.pretrained_checkpoint, map_location='cpu', weights_only=True)
    result = model.load_state_dict(source, strict=False)
    attribute = 'goal_readout' if args.branch == 'goal' else 'object_readout'
    assert result.missing_keys and all(key.startswith(f'spatial_focus.{attribute}.')
                                      for key in result.missing_keys)
    assert not result.unexpected_keys
    loaded = model.state_dict()
    assert all(torch.equal(loaded[key], value) for key, value in source.items())
    preserved = len(source)
    model.load_state_dict(loaded, strict=True)
    added = sum(p.numel() for p in getattr(model.spatial_focus, attribute).parameters())
    del loaded
    del source
    model.cuda().eval()
    focus = model.spatial_focus
    original = focus.refine_queries
    predictions = {}
    examples = [data[(0, i)] for i in range(8)]
    inputs = [{key: value for key, value in example.items()
               if not key.startswith('spatial_') and 'future' not in key
               and key not in ('action', 'action_valid_mask')} for example in examples]
    try:
        for mode in ('full', 'ablated'):
            focus.refine_queries = (original if mode == 'full' else lambda queries, memories:
                                    original(queries, {k: v for k, v in memories.items() if k != args.branch}))
            predictions[mode] = np.concatenate([model.predict_action(inputs[i:i+4])['normalized_actions']
                                               for i in range(0, 8, 4)])
    finally:
        focus.refine_queries = original
    assert np.isfinite(predictions['full']).all()
    difference = float(np.abs(predictions['full'] - predictions['ablated']).max())
    assert difference == 0., difference
    args.output.write_text(json.dumps(dict(checkpoint=str(cfg.trainer.pretrained_checkpoint),
         branch=args.branch, added_parameters=added, original_tensors_preserved=preserved,
         new_keys=result.missing_keys, samples=8,
         initial_max_action_change=difference, passed=True,
         note='Exact real-input full/branch-ablated equality before training; all existing checkpoint tensors preserved.'),
         indent=2) + '\n')


if __name__ == '__main__':
    main()
