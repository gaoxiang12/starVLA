"""Inspect real action tensors using saved training dtypes and deployment dtypes."""
import argparse
from collections import Counter
import importlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = OmegaConf.load(args.run / 'config.full.yaml')
    cfg.datasets.vla_data.episode_split = 'validation'
    cfg.datasets.vla_data.normalization_statistics_path = str(args.run / 'dataset_statistics.json')
    dataset = get_vla_dataset(cfg.datasets.vla_data, mode='validation')
    for child in dataset.datasets:
        child.transforms.eval()
    examples = [dataset[(0, i)] for i in range(4)]
    model = build_framework(cfg).cuda().eval()
    checkpoint = args.run / 'final_model/pytorch_model.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(saved, strict=True)
    # Restore each tensor's SAVED dtype, including float32 running-stat buffers.
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if name in saved:
            tensor.data = saved[name].to(device='cuda').clone()
    dtypes = dict(Counter(str(value.dtype) for value in saved.values()))
    del saved
    module = importlib.import_module('starVLA.model.framework.WM4A.GAWM')
    original = module.masked_action_l1_loss
    captured = []

    def capture(prediction, target, valid=None):
        captured.append(dict(prediction_dtype=str(prediction.dtype), target_dtype=str(target.dtype),
                             prediction=prediction.detach().float().cpu().numpy(),
                             loss=float(original(prediction, target, valid))))
        return original(prediction, target, valid)

    module.masked_action_l1_loss = capture
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in (0, 2):
                model(examples=examples[i:i+2], optimizer_step=5000)
    finally:
        module.masked_action_l1_loss = original
    train_predictions = np.concatenate([row.pop('prediction') for row in captured])
    assert np.isfinite(train_predictions).all()
    inputs = [{k: v for k, v in ex.items() if not k.startswith('spatial_') and 'future' not in k
               and k not in ('action', 'action_valid_mask')} for ex in examples]
    saved_dtype_inference = np.concatenate([model.predict_action(inputs[i:i+2])['normalized_actions']
                                           for i in (0, 2)])
    model.float()
    deployed = np.concatenate([model.predict_action(inputs[i:i+2])['normalized_actions'] for i in (0, 2)])
    assert np.isfinite(deployed).all()
    difference = np.abs(train_predictions - deployed)
    comparisons = {}
    for name, first, second in [('forward_vs_predict_same_dtype', train_predictions, saved_dtype_inference),
                                 ('saved_vs_fp32_predict', saved_dtype_inference, deployed)]:
        delta = np.abs(first - second)
        comparisons[name] = dict(mean=float(delta.mean()), maximum=float(delta.max()),
                                joint_maximum=float(delta[..., :12].max()),
                                gripper_maximum=float(delta[..., 12:14].max()))
    args.output.write_text(json.dumps(dict(checkpoint=str(checkpoint.resolve()), torch_version=torch.__version__,
         saved_tensor_dtypes=dtypes, training_forward_batches=captured, deployment_action_dtype=str(deployed.dtype),
         samples=4, mean_action_difference=float(difference.mean()), max_action_difference=float(difference.max()),
         isolated_comparisons=comparisons,
         note='Saved parameter/buffer dtypes restored exactly. Training forward uses the trainer outer BF16 '
              'autocast and model FP32 islands, with dropout/stat updates disabled for comparison. '
              'Deployment uses the same weight values expanded to FP32. No optimizer or live process changes; '
              'this checks tensor precision, not optimizer master weights or closed-loop success.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
