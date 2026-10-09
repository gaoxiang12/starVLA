"""Repartition a replicated C/DDP checkpoint without changing training progress.

Model, optimizer and scheduler files are copied byte-for-byte. A new checkpoint
records the changed batch partition and deterministic fresh RNG streams, with
the original completion metadata retained. This is not a bitwise continuation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
from omegaconf import OmegaConf
import torch

from starVLA.training.recipe import apply_training_recipe, resume_contract, resolve_training_budget


def validate_repartition(metadata, cfg, local_sizes):
    world = sum(local_sizes)
    if not local_sizes or any(n < 1 for n in local_sizes):
        raise ValueError('All nodes need at least one rank')
    if cfg.trainer.recipe != 'c' or cfg.trainer.distributed_backend != 'ddp':
        raise ValueError('Only replicated C/DDP states support repartitioning')
    if cfg.datasets.vla_data.sampling_mode != 'frame_epoch':
        raise ValueError('Repartitioning requires deterministic frame_epoch sampling')
    batch = world * cfg.datasets.vla_data.per_device_batch_size * cfg.trainer.gradient_accumulation_steps
    if batch != metadata['global_batch_size'] or batch != cfg.trainer.expected_global_batch_size:
        raise ValueError('Repartitioning must preserve the global optimizer batch')
    old = copy.deepcopy(metadata['contract'])
    new = resume_contract(cfg)
    old_batch = old['datasets.vla_data'].pop('per_device_batch_size')
    new['datasets.vla_data'].pop('per_device_batch_size')
    if old != new:
        raise ValueError('Only per_device_batch_size may change in the resume contract')
    if old_batch * metadata['world_size'] * cfg.trainer.gradient_accumulation_steps != batch:
        raise ValueError('Source batch metadata is inconsistent')
    for key in ('frames_per_epoch', 'steps_per_epoch', 'stage1_steps', 'lr_scheduler_total_steps'):
        if cfg.trainer[key] != metadata[key]:
            raise ValueError(f'Repartitioning cannot change {key}')
    return world


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def fresh_rng(seed, local_size, accelerator_step, cuda_state_factory=None):
    if cuda_state_factory is None:
        cuda_state_factory = lambda seed: torch.Generator(device='cuda:0').manual_seed(seed).get_state()
    cuda_state = cuda_state_factory(seed)
    return dict(step=accelerator_step, random_state=random.Random(seed).getstate(),
                numpy_random_seed=np.random.RandomState(seed).get_state(),
                torch_manual_seed=torch.Generator().manual_seed(seed).get_state(),
                torch_cuda_manual_seed=[cuda_state.clone() for _ in range(local_size)])


def migrate(source, destination, cfg, local_sizes, cuda_state_factory=None):
    source, destination = Path(source), Path(destination)
    metadata = json.loads((source / 'complete.json').read_text())
    world = validate_repartition(metadata, cfg, local_sizes)
    required = ['model.safetensors', 'optimizer.bin', 'scheduler.bin']
    if not all((source / name).is_file() for name in required):
        raise ValueError('Source lacks complete model/optimizer/scheduler state')
    rng_paths = [source / f'random_states_{rank}.pkl' for rank in range(metadata['world_size'])]
    if not all(path.is_file() for path in rng_paths):
        raise ValueError('Source lacks per-rank RNG state')
    # These are our own locally produced checkpoints, not external pickle files.
    accelerator_steps = {torch.load(path, map_location='cpu', weights_only=False)['step'] for path in rng_paths}
    if len(accelerator_steps) != 1:
        raise ValueError('Source rank step counters differ')
    accelerator_step = accelerator_steps.pop()
    stats = source / 'dataset_statistics.json'
    if not stats.exists():
        stats = source.parent.parent / 'dataset_statistics.json'
    if not stats.exists():
        raise FileNotFoundError(stats)
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in required:
        shutil.copy2(source / name, destination / name)
        hashes[name] = digest(source / name)
        if digest(destination / name) != hashes[name]:
            raise IOError(f'Checkpoint copy hash differs: {name}')
    shutil.copy2(stats, destination / 'dataset_statistics.json')
    shutil.copy2(source / 'complete.json', destination / 'source_complete.json')
    seeds = []
    for local_size in local_sizes:
        for _ in range(local_size):
            rank = len(seeds)
            seed = (int(cfg.seed) + int(metadata['step']) * world + rank) % (2**32)
            seeds.append(seed)
            torch.save(fresh_rng(seed, local_size, accelerator_step, cuda_state_factory),
                       destination / f'random_states_{rank}.pkl')
    provenance = dict(source=str(source.resolve()), step=metadata['step'],
                      old_world_size=metadata['world_size'], new_world_size=world,
                      old_per_device_batch=metadata['contract']['datasets.vla_data']['per_device_batch_size'],
                      new_per_device_batch=cfg.datasets.vla_data.per_device_batch_size,
                      global_batch_size=metadata['global_batch_size'], local_sizes=local_sizes,
                      rng_policy='fresh deterministic per-rank streams; not bitwise continuation',
                      rank_seeds=seeds, unchanged_file_sha256=hashes)
    (destination / 'migration.json').write_text(json.dumps(provenance, indent=2) + '\n')
    metadata.update(world_size=world, contract=resume_contract(cfg), repartition=provenance)
    (destination / 'complete.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local-sizes', required=True)
    args = parser.parse_args()
    cfg = apply_training_recipe(OmegaConf.load(args.config))
    resolve_training_budget(cfg, range(cfg.datasets.vla_data.expected_frames), cfg.trainer.expected_global_batch_size)
    sizes = [int(x) for x in args.local_sizes.split(',')]
    print(json.dumps(migrate(args.source, args.destination, cfg, sizes), indent=2))


if __name__ == '__main__':
    main()
