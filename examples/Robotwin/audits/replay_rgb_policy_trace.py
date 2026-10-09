"""Reproduce a saved live GAWM request and compare spatial-memory interventions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from examples.Robotwin.audits.spatial_ablation import apply_spatial_ablation
from starVLA.model.framework.WM4A.GAWM import GAWM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--ablations', nargs='+', choices=('no_focus', 'no_objects', 'no_dense', 'no_local', 'no_goal'),
                        default=['no_focus'], help='Independent interventions, each reset to the full model')
    parser.add_argument('--server-metadata', type=Path,
                        help='Explicit handshake metadata for traces made before metadata capture was enabled')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.trace, allow_pickle=False) as trace:
        metadata = json.loads(trace['metadata'].item())
        example = {key: trace[key].copy() for key in ('image', 'native_images', 'image_history', 'state')
                   if key in trace}
        observed = trace['actions'].copy()
    # NPZ stacks equal-sized camera arrays; restore the list nesting expected
    # by the live wrapper's image-to-PIL conversion.
    for key in ('image', 'native_images'):
        if key in example:
            example[key] = list(example[key])
    if 'image_history' in example:
        example['image_history'] = [list(frame) for frame in example['image_history']]
    example.update(lang=metadata['lang'], episode_start=metadata['episode_start'])
    server = (json.loads(args.server_metadata.read_text()) if args.server_metadata
              else metadata['server_metadata'])
    checkpoint = args.checkpoint or Path(server['ckpt_path'])
    if args.device == 'cpu':
        import torch
        torch.set_num_threads(4)
    wrapper = PolicyServerWrapper(str(checkpoint), device=args.device,
                                  use_bf16=False, unnorm_key=metadata['unnorm_key'])
    model = wrapper._framework
    if not isinstance(model, GAWM):
        raise ValueError('This replay audit covers stateless GAWM requests only')
    original_refine = model.spatial_focus.refine_queries
    original_predict = model.predict_action
    captured = {}

    def predict(*positional, **kwargs):
        result = original_predict(*positional, **kwargs)
        captured.clear()
        captured.update(result)
        return result

    model.predict_action = predict
    reports = {}
    for mode in dict.fromkeys(['full', *args.ablations]):
        model.spatial_focus.refine_queries = original_refine
        apply_spatial_ablation(model, mode)
        response = wrapper.predict_action(examples=[example], unnorm_key=metadata['unnorm_key'],
                                          **metadata['request_options'])
        actions = np.asarray(response['actions'])
        reports[mode] = dict(actions=actions.tolist(),
                             mean_abs_change_from_recorded=float(np.abs(actions-observed).mean()),
                             max_abs_change_from_recorded=float(np.abs(actions-observed).max()),
                             predicted_xy=np.asarray(captured['spatial_predicted_xy']).tolist())
    recorded_mode = server.get('spatial_ablation', 'full')
    report = dict(trace=str(args.trace.resolve()), checkpoint=str(checkpoint.resolve()), device=args.device,
                  trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  recorded_mode=recorded_mode, scores=reports,
                  note='Same saved live input, no simulator labels. Action units are unnormalized model order.')
    if recorded_mode not in reports:
        raise ValueError(f'Unsupported recorded intervention {recorded_mode}')
    report['replay_matches'] = reports[recorded_mode]['max_abs_change_from_recorded'] <= 1e-5
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if not report['replay_matches']:
        raise RuntimeError('Live replay differs by more than 1e-5; inspect report before using this audit')


if __name__ == '__main__':
    main()
