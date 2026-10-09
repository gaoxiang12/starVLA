import copy
import json

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.data import BatchSampler
from accelerate.data_loader import BatchSamplerShard

from scripts.repartition_c_checkpoint import migrate, validate_repartition, fresh_rng
from scripts.run_gawm_c_cluster import resolve_gpu_map
from starVLA.training.recipe import apply_training_recipe, resolve_training_budget, resume_contract


def config(batch=4):
    cfg = apply_training_recipe(OmegaConf.load('examples/Robotwin/train_files/starvla_gawm_official_c_12plus4.yaml'))
    cfg.datasets.vla_data.per_device_batch_size = batch
    resolve_training_budget(cfg, range(cfg.datasets.vla_data.expected_frames), 128)
    return cfg


def metadata():
    cfg = config(8)
    return dict(step=1200, world_size=16, global_batch_size=128, contract=resume_contract(cfg),
                **{k: cfg.trainer[k] for k in ['frames_per_epoch','steps_per_epoch','stage1_steps','lr_scheduler_total_steps']})


def test_gpu_map_validates_each_host_and_unique_devices():
    assert resolve_gpu_map(['a','b'], '0,1', '{"a":"1,2,3,4","b":"0,1"}') == {'a':'1,2,3,4','b':'0,1'}
    with pytest.raises(ValueError):
        resolve_gpu_map(['a','b'], '0', '{"a":"0"}')
    with pytest.raises(ValueError):
        resolve_gpu_map(['a'], '0,0')
    with pytest.raises(ValueError, match='order larger nodes first'):
        resolve_gpu_map(['a','b'], '0', '{"a":"0,1","b":"0,1,2,3"}')


def test_only_batch_partition_may_change():
    cfg = config()
    meta = metadata()
    assert validate_repartition(meta, cfg, [8,8,8,4,4]) == 32
    for path, value in [('trainer.learning_rate.base', .001),
                        ('datasets.vla_data.future_recorded_offsets', [0,8,16]),
                        ('trainer.stage1_steps', 100),
                        ('datasets.vla_data.per_device_batch_size', 8)]:
        changed = copy.deepcopy(cfg)
        OmegaConf.update(changed, path, value)
        with pytest.raises(ValueError):
            validate_repartition(meta, changed, [8,8,8,4,4])
    assert meta == metadata()  # validation must not mutate the original contract


def test_repartition_keeps_weights_optimizer_scheduler_and_step(tmp_path):
    src = tmp_path / 'source'; src.mkdir()
    (src/'complete.json').write_text(json.dumps(metadata()))
    (src/'dataset_statistics.json').write_text('{"aloha":{}}')
    for name in ['model.safetensors','optimizer.bin','scheduler.bin']:
        (src/name).write_bytes(('unchanged '+name).encode())
    for rank in range(16):
        torch.save({'step':1200}, src/f'random_states_{rank}.pkl')
    target = tmp_path/'migrated'
    report = migrate(src,target,config(),[8,8,8,4,4],lambda seed: torch.tensor([seed]))
    assert report['step'] == 1200 and report['new_world_size'] == 32
    for name in report['unchanged_file_sha256']:
        assert (src/name).read_bytes() == (target/name).read_bytes()
    assert json.loads((src/'complete.json').read_text())['world_size'] == 16
    assert json.loads((target/'complete.json').read_text())['world_size'] == 32
    assert len(list(target.glob('random_states_*.pkl'))) == 32
    for rank,size in [(0,8),(23,8),(24,4),(31,4)]:
        state=torch.load(target/f'random_states_{rank}.pkl',weights_only=False)
        assert state['step']==1200 and len(state['torch_cuda_manual_seed'])==size
    assert len(set(report['rank_seeds']))==32


def test_fresh_rng_is_repeatable_with_distinct_rank_streams():
    factory=lambda seed: torch.tensor([seed])
    a=fresh_rng(123,4,1200,factory);b=fresh_rng(123,4,1200,factory)
    assert torch.equal(a['torch_manual_seed'],b['torch_manual_seed'])
    np.testing.assert_array_equal(a['numpy_random_seed'][1],b['numpy_random_seed'][1])
    assert not torch.equal(a['torch_manual_seed'],fresh_rng(124,4,1200,factory)['torch_manual_seed'])


def test_global_frame_batches_remain_identical_after_repartition():
    # Same deterministic frame permutation, resumed at the same optimizer step.
    permutation=torch.randperm(1024,generator=torch.Generator().manual_seed(42)).tolist()
    def global_batches(world,batch):
        ranks=[list(BatchSamplerShard(BatchSampler(permutation,batch,drop_last=True),
                   num_processes=world,process_index=r,even_batches=False)) for r in range(world)]
        return [[item for rank in ranks for item in rank[step]] for step in range(3,8)]
    assert global_batches(16,8)==global_batches(32,4)
