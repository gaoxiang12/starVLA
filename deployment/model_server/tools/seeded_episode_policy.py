"""Opt-in, episode-reset Torch RNG for reproducible stochastic policy evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class SeededEpisodePolicy:
    def __init__(self, policy, seed: int, log_path=None):
        if not isinstance(seed, int) or seed < 0:
            raise ValueError('Policy seed must be a nonnegative integer')
        self.policy, self.seed = policy, seed
        self.query_index = None
        self.episode_index = -1
        self.cuda_devices = sorted({p.device.index for p in policy._framework.parameters() if p.device.type == 'cuda'})
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.open('x').close()

    @property
    def rng_metadata(self):
        return dict(base_seed=self.seed, scheme='sha256_base_seed_colon_query_index_low63',
                    reset='episode_start=True restarts the same query-indexed noise schedule in every scene',
                    scope='Torch CPU and model CUDA generators; external Torch RNG state is restored',
                    batch_size=1)

    def predict_action(self, examples, **kwargs):
        if not isinstance(examples, list) or len(examples) != 1:
            raise ValueError('Seeded episode evaluation requires a single-example request')
        if 'episode_start' not in examples[0] or not isinstance(examples[0]['episode_start'], (bool, np.bool_)):
            raise ValueError('Seeded episode evaluation requires an explicit episode_start boolean')
        start = bool(examples[0]['episode_start'])
        if start:
            self.query_index = 0
            self.episode_index += 1
        if self.query_index is None:
            raise ValueError('First request must mark episode_start=True')
        digest = hashlib.sha256(f'{self.seed}:{self.query_index}'.encode()).digest()
        query_seed = int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1)
        with torch.random.fork_rng(devices=self.cuda_devices):
            torch.random.default_generator.manual_seed(query_seed)
            for device in self.cuda_devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(query_seed)
            result = self.policy.predict_action(examples=examples, **kwargs)
        actions = np.ascontiguousarray(result['actions'])
        if self.log_path is not None:
            record = dict(episode_index=self.episode_index, query_index=self.query_index,
                          episode_start=start, query_seed=query_seed,
                          action_shape=list(actions.shape), action_dtype=str(actions.dtype),
                          action_sha256=hashlib.sha256(actions.tobytes()).hexdigest())
            with self.log_path.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        self.query_index += 1
        return result
