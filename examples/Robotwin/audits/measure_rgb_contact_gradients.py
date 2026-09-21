"""Calibrate added contact supervision on real near-closing training forwards."""
import argparse
import importlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from examples.Robotwin.audits.measure_rgb_loss_gradients import group
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--contact-audit', type=Path, required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = OmegaConf.load(args.run/'config.full.yaml')
    cfg.datasets.vla_data.episode_split = 'validation'
    cfg.datasets.vla_data.episode_split_manifest = str(args.split_manifest.resolve())
    cfg.datasets.vla_data.normalization_statistics_path = str(args.run/'dataset_statistics.json')
    cfg.framework.contact_objective = dict(enabled=True, robot_tag='aloha', urdf_path=str(args.urdf.resolve()),
                                           position_weight=.1, gripper_weight=.02, contact_boost=4., transition_radius=3)
    dataset, = get_vla_dataset(cfg.datasets.vla_data, mode='validation').datasets
    dataset.transforms.eval()
    records = json.loads(args.contact_audit.read_text())['rows']
    selected = [row for row in records if row['lookahead']==8][:8]
    assert len(selected) == 8
    examples = []
    for row in selected:
        episode, anchor = row['episode'], row['anchor']
        sample = dataset._pack_sample(dataset.transforms(dataset.get_step_data(episode, anchor)))
        sample = dataset._attach_action_validity(sample, episode, anchor)
        sample = dataset._attach_future_frame_validity(sample, episode, anchor)
        examples.append(dataset._attach_spatial_supervision(sample, episode, anchor))
    model = build_framework(cfg)
    checkpoint = args.run/'checkpoints'/f'steps_{args.step}_pytorch_model.pt'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.cuda().train()
    parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    module = importlib.import_module('starVLA.model.framework.WM4A.GAWM')
    original_action = module.masked_action_l1_loss
    original_contact = model.contact_objective.forward
    captured = {}

    def action(*positional, **kwargs):
        captured['action'] = original_action(*positional, **kwargs)
        return captured['action']

    def contact(*positional, **kwargs):
        result = original_contact(*positional, **kwargs)
        captured['contact'] = result[0]
        return result

    module.masked_action_l1_loss = action
    model.contact_objective.forward = contact
    rows = []
    try:
        for start in range(0, 8, 2):
            torch.manual_seed(42+start)
            output = model(examples=examples[start:start+2], optimizer_step=5000)
            gradients = [torch.autograd.grad(captured[key], [p for _, p in parameters],
                         retain_graph=key=='action', allow_unused=True) for key in ('action', 'contact')]
            groups = {}
            for (name, _), a, c in zip(parameters, *gradients):
                totals = groups.setdefault(group(name), torch.zeros(3, device='cuda'))
                if a is not None:
                    totals[0] += a.float().square().sum()
                if c is not None:
                    totals[1] += c.float().square().sum()
                if a is not None and c is not None:
                    totals[2] += (a.float()*c.float()).sum()
            summary = {}
            for name, totals in groups.items():
                assert torch.isfinite(totals).all()
                aa, cc, ac = totals.tolist()
                summary[name] = dict(action_norm=aa**.5, contact_norm=cc**.5,
                                     contact_to_action=(cc/aa)**.5 if aa else None,
                                     cosine=ac/(aa*cc)**.5 if aa and cc else None)
            rows.append(dict(samples=selected[start:start+2], gradients=summary,
                             metrics={key: float(value.detach()) for key, value in output.items()
                                      if key in ('l1_action_loss', 'tcp_position_loss_m', 'tcp_contact_error_mm',
                                                 'gripper_transition_l1', 'contact_objective_loss', 'contact_fraction')}))
            del gradients, output
            captured.clear()
    finally:
        module.masked_action_l1_loss = original_action
        model.contact_objective.forward = original_contact
    report = dict(checkpoint=str(checkpoint.resolve()), configuration=OmegaConf.to_container(cfg.framework.contact_objective),
                  batches=rows, note='Four paired near-contact forwards, FP32 parameters and geometry, internal DINO BF16 autocast. '
                                     'Pre-clipping gradients, no optimizer step. Small diagnostic sample; not success-rate evidence.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
