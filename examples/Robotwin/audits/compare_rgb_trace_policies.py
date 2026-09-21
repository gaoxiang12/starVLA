"""Compare stateless RGB policies on each other's exact saved observations."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.model.framework.WM4A.GAWM import GAWM


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', action='append', required=True, metavar='NAME=NPZ')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.1)
    inputs = {}
    for item in args.trace:
        name, raw_path = item.split('=', 1)
        assert name and name not in inputs
        path = Path(raw_path).resolve()
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(z['metadata'].item())
            example = {k: z[k].copy() for k in
                       ('image', 'native_images', 'image_history', 'state') if k in z}
            actions = z['actions'].copy()
        for k in ('image', 'native_images'):
            if k in example:
                example[k] = list(example[k])
        if 'image_history' in example:
            example['image_history'] = [list(frame) for frame in example['image_history']]
        example.update(lang=meta['lang'], episode_start=meta['episode_start'])
        assert meta['server_metadata']['spatial_ablation'] == 'full'
        assert meta['state_order'] == 'model: left6, right6, left_gripper, right_gripper'
        inputs[name] = dict(path=path, meta=meta, example=example, actions=actions)
    assert len(inputs) >= 2
    matrix, checkpoints, replay = {}, {}, {}
    for name, source in inputs.items():
        checkpoint = Path(source['meta']['server_metadata']['ckpt_path'])
        wrapper = PolicyServerWrapper(str(checkpoint), device=args.device, use_bf16=False,
                                      unnorm_key=source['meta']['unnorm_key'])
        assert isinstance(wrapper._framework, GAWM)
        checkpoints[name] = dict(path=str(checkpoint), sha256=digest(checkpoint))
        matrix[name] = {}
        for observation_name, observation in inputs.items():
            meta = observation['meta']
            response = wrapper.predict_action(examples=[copy.deepcopy(observation['example'])],
                unnorm_key=meta['unnorm_key'], **meta['request_options'])
            actions = np.asarray(response['actions'])
            assert actions.shape == (1, 16, 14) and np.isfinite(actions).all()
            grip = actions[0, :, 12:14]
            closed = np.flatnonzero((grip < .2).any(axis=1))
            matrix[name][observation_name] = dict(actions=actions.tolist(),
                gripper_min=grip.min(axis=0).tolist(),
                first_close_chunk_index=int(closed[0]) if len(closed) else None)
            if name == observation_name:
                error = float(np.abs(actions-observation['actions']).max())
                replay[name] = dict(max_abs_action_error=error, matches=error <= 1e-5)
        del wrapper
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    verified = all(row['matches'] for row in replay.values())
    report = dict(state='complete' if verified else 'unverified_replay', device=args.device,
        inputs={name: dict(path=str(row['path']), sha256=digest(row['path']))
                for name, row in inputs.items()}, checkpoints=checkpoints,
        self_replay=replay, policy_by_observation=matrix,
        note='Each matrix column uses one identical saved input across policies. Self-replays '
             'must match the recorded GPU actions within1e-5 before interpreting the matrix. '
             'This does not intervene in the simulator or establish task success; image/state '
             'differences between columns are not isolated from each other.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(state=report['state'], self_replay=replay,
        matrix={p: {i: {k: v for k, v in row.items() if k != 'actions'}
                    for i, row in columns.items()} for p, columns in matrix.items()}), indent=2))
    if not verified:
        raise RuntimeError('Self-replay failed; do not interpret cross-policy results')


if __name__ == '__main__':
    main()
