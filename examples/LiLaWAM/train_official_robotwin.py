"""Single-view official LiLa-WAM recipe with auditable stages and resumable checkpoints.

Uses the same upstream model, loss, normalization, and HDF5 loader as the
StarVLA LiLaWAM inference adapter. This is not the three-camera LiLaWAMTrain variant.
"""
import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from examples.LiLaWAM.official_robotwin_data import digest, write_json, expected_episode_count
from starVLA.model.framework.WM4A.LiLaWAM import upstream_models


class EpochSampler(Sampler):
    def __init__(self, size, seed, offset=0, rank=0, world_size=1, global_batch_size=None):
        self.size, self.seed, self.offset = size, seed, offset
        self.rank, self.world_size, self.global_batch_size = rank, world_size, global_batch_size
        if world_size > 1 and (not global_batch_size or global_batch_size % world_size or offset % global_batch_size):
            raise ValueError('Distributed sampler requires complete global batches and aligned resume offset')

    def __iter__(self):
        indices = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed))
        if self.world_size > 1:
            usable = self.size // self.global_batch_size * self.global_batch_size
            batches = indices[self.offset:usable].reshape(-1, self.world_size, self.global_batch_size // self.world_size)
            return iter(batches[:, self.rank, :].reshape(-1).tolist())
        return iter(indices[self.offset:].tolist())

    def __len__(self):
        if self.world_size > 1:
            return (self.size // self.global_batch_size * self.global_batch_size - self.offset) // self.world_size
        return self.size - self.offset


def rng_state():
    return {'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state(),
            'numpy':np.random.get_state(), 'python':random.getstate()}


def restore_rng(ckpt, rank, world_size, seed):
    previous_world = ckpt.get('world_size', 1)
    if previous_world == world_size and 'rng_by_rank' in ckpt:
        state = ckpt['rng_by_rank'][rank]
    elif rank == 0:
        state = ckpt['rng']
    else:
        # A change in world size changes noise assignment. Preserve all optimizer
        # and data positions, but create independent reproducible streams per rank.
        rank_seed = seed + ckpt['global_step'] * 1009 + rank
        random.seed(rank_seed); np.random.seed(rank_seed % 2**32)
        torch.manual_seed(rank_seed); torch.cuda.manual_seed(rank_seed)
        return
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def parameter_digest(action):
    h = hashlib.sha256()
    for key, value in action.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_dataset_module(source):
    spec = importlib.util.spec_from_file_location('_lila_official_training_dataset', source / 'dataloader/dataset.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_collate(batch):
    if any(x is None for x in batch):
        raise ValueError('Official dataset returned a failed sample; refusing to silently drop it')
    return {k: torch.stack([x[k] for x in batch]) if isinstance(batch[0][k], torch.Tensor) else [x[k] for x in batch]
            for k in batch[0]}


def make_model(cfg, source, stats):
    if cfg.get('experiment', {}).get('model_kind') == 'gawm':
        from examples.LiLaWAM.gawm_official import make_model as make_gawm
        return make_gawm(cfg, source, stats)
    upstream = upstream_models(source)
    dtype = torch.bfloat16
    vision, hidden, registers, patch = upstream.ModelFactory.create_vision_encoder(
        cfg.model.vision_encoder.checkpoint_path, dtype, 'cuda')
    action = upstream.ModelFactory.create_action_model(
        cfg, hidden, len(cfg.model.vision_encoder.feat_layers), task_cond_dim=hidden, patch_size=patch)
    # Matches upstream train.py, including bf16 trainable parameters/Adam states.
    action.to('cuda', dtype=dtype)
    t = cfg.training
    training = {k: t[k] for k in ('time_mu', 'time_sigma', 'use_vel_weight', 'vel_weight_alpha', 'vel_weight_sigma')}
    training.update(use_future_feat=True, lambda_future_feat=t.lambda_future_feat)
    wrapper = upstream.VLAWrapper(vision, action, cfg.training.time_sampler,
        list(cfg.model.vision_encoder.feat_layers), cfg.model.vision_encoder.include_cls_register,
        registers, 'cuda', dtype, str(stats), training, cfg.model.future_feat.target_layer)
    return wrapper, action, vision


def worker_init(worker_id):
    import cv2
    cv2.setNumThreads(1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(path, action, optimizer, scheduler, epoch, step, global_step, cfg, initial_hash, rank_rngs=None):
    payload = {'model_state_dict': action.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'scheduler_state_dict': scheduler.state_dict(), 'epoch': epoch, 'step_in_epoch': step,
               'global_step': global_step, 'config': OmegaConf.to_container(cfg, resolve=True),
               'initial_random_model_sha256': initial_hash,
               'rng': rank_rngs[0] if rank_rngs else rng_state(),
               'world_size':len(rank_rngs) if rank_rngs else 1}
    if rank_rngs:
        payload['rng_by_rank'] = rank_rngs
    temp = path.with_suffix('.partial')
    torch.save(payload, temp)
    temp.replace(path)
    # A weights-only file stays compatible with the existing safe inference loader.
    policy = path.parent / 'policy.pt'
    policy_temp = policy.with_suffix('.partial')
    torch.save({'model_state_dict':action.state_dict(), 'epoch':epoch, 'global_step':global_step}, policy_temp)
    policy_temp.replace(policy)


def run(args):
    rank, world_size = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        # StarVLA's imported utilities may initialize Accelerate PartialState.
        if not dist.is_initialized():
            dist.init_process_group('nccl', device_id=torch.device('cuda', local_rank))
        if dist.get_world_size() != world_size or dist.get_rank() != rank:
            raise ValueError('Existing process group does not match torchrun')
    primary = rank == 0
    args.output.mkdir(parents=True, exist_ok=True)
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    cfg = OmegaConf.load(args.config)
    if cfg.common.state_dim != 16 or cfg.common.action_dim != 14 or list(cfg.dataset.camera_names) != ['head_camera']:
        raise ValueError('This reproduction requires the official single-camera 16D-state / 14D-action recipe')
    if args.init_from and args.resume:
        raise ValueError('init_from and resume are mutually exclusive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    if primary:
        OmegaConf.save(cfg, args.output / 'config.yaml')
    dataset_module = load_dataset_module(args.source)
    dataset = dataset_module.create_dataset(cfg, val=False)
    is_gawm = cfg.get('experiment', {}).get('model_kind') == 'gawm'
    if is_gawm:
        from examples.LiLaWAM.gawm_official import GAWMOfficialDataset
        dataset = GAWMOfficialDataset(dataset, dataset_module)
    if not args.smoke_steps:
        audit = json.loads(args.audit.read_text())
        report = args.source / 'utils/outlier_files 500-all.txt'
        if (audit.get('status') != 'passed' or audit['episodes'] != expected_episode_count(report)
                or audit['tasks'] != 50 or audit.get('outlier_report_sha256') != digest(report)):
            raise ValueError('Full data audit is required before production training')
        if len(dataset) != audit['frames'] or len(dataset.all_episodes) != audit['episodes']:
            raise ValueError('Dataset no longer matches audit')
    batch = int(cfg.training.batch_size)
    if batch % world_size:
        raise ValueError('Global batch must be divisible by world size')
    local_batch = batch // world_size
    if batch != 128 and not args.smoke_steps:
        raise ValueError('Official effective batch must be 128')
    steps_per_epoch = len(dataset) // batch
    if steps_per_epoch == 0:
        raise ValueError('Dataset smaller than batch')
    wrapper, action, vision = make_model(cfg, args.source, args.stats)
    initial_hash = parameter_digest(action)
    optimizer = torch.optim.AdamW(action.parameters(), lr=cfg.training.learning_rate,
        betas=tuple(cfg.training.betas), weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=int(cfg.training.epochs) * steps_per_epoch, eta_min=cfg.training.lr_min)
    start_epoch = start_step = global_step = 0
    resume_world_size = 1
    if args.init_from or args.resume:
        ckpt = torch.load(args.init_from or args.resume, map_location='cpu', weights_only=False)
        action.load_state_dict(ckpt['model_state_dict'], strict=True)
        if args.resume:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch, start_step, global_step = ckpt['epoch'], ckpt['step_in_epoch'], ckpt['global_step']
            initial_hash = ckpt['initial_random_model_sha256']
            resume_world_size = ckpt.get('world_size', 1)
            restore_rng(ckpt, rank, world_size, args.seed)
            if ckpt.get('config', {}).get('training', {}).get('batch_size', batch) != batch:
                raise ValueError('Resume must preserve global batch size')
        del ckpt
    if not args.resume and world_size > 1:
        torch.cuda.manual_seed(args.seed + rank)
    train_model = wrapper
    if world_size > 1:
        train_model = DistributedDataParallel(wrapper, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True)
    provenance = {'seed': args.seed, 'initial_random_model_sha256': initial_hash,
        'trainable_parameters': sum(p.numel() for p in action.parameters() if p.requires_grad),
        'frozen_parameters': sum(p.numel() for p in vision.parameters()),
        'policy_initialization': str(args.init_from or args.resume) if (args.init_from or args.resume) else 'random; no trained policy checkpoint loaded',
        'dataset_frames': len(dataset), 'dataset_episodes': len(dataset.all_episodes),
        'batch_size': batch, 'steps_per_epoch': steps_per_epoch, 'stop_after_epochs': args.stop_epochs,
        'world_size':world_size, 'per_device_batch_size':local_batch,
        'resume_world_size':resume_world_size, 'resumed_global_step':global_step,
        'resumed_step_in_epoch':start_step, 'resumed_lr':optimizer.param_groups[0]['lr'],
        'loaded_model_sha256':parameter_digest(action) if primary and (args.resume or args.init_from) else None,
        'rng_note':'World-size changes retain rank-0 RNG and seed other ranks independently; no bitwise equivalence claim.',
        'scheduler_horizon_epochs': cfg.training.epochs, 'parameter_dtype': 'bfloat16 (upstream default)',
        'deterministic_algorithms':args.deterministic,
        'training_runner_sha256':digest(Path(__file__)),
        'environment':{'python':sys.version, 'torch':torch.__version__, 'cuda':torch.version.cuda,
                       'gpu':torch.cuda.get_device_name()},
        'source_sha256': {str(p.relative_to(args.source)): digest(p) for folder in ('models','dataloader')
                          for p in (args.source / folder).glob('*.py')},
        'stats_sha256': digest(args.stats), 'config_sha256': digest(args.config)}
    if is_gawm:
        provenance['experiment'] = OmegaConf.to_container(cfg.experiment, resolve=True)
        provenance['gawm_source_sha256'] = {
            str(p): digest(p) for p in [Path(__file__).with_name('gawm_official.py'),
                *Path('starVLA/model/framework/WM4A').glob('GAWM.py'),
                *Path('starVLA/model/modules/world_model').glob('*.py'),
                Path('starVLA/model/modules/action_model/ACT_ActionHeader.py'),
                Path('starVLA/task_language.py')]}
    if primary:
        provenance_path = args.output / 'provenance.json'
        if provenance_path.exists():
            provenance_path.replace(args.output / f'provenance_before_resume_{global_step}_{time.time_ns()}.json')
        write_json(provenance_path, provenance)
    if any(p.requires_grad for p in vision.parameters()):
        raise ValueError('DINO must remain frozen')
    metrics = (args.output / 'metrics.jsonl').open('a', buffering=1) if primary else None
    started, first_step, gradient_groups = time.monotonic(), global_step, {}
    probe_name = ('action_models.aloha.action_projection.layers.2.weight' if is_gawm else 'output_proj.weight')
    probe_parameter = dict(action.named_parameters())[probe_name]
    probe = probe_parameter.detach().clone()
    last_saved = None
    for epoch in range(start_epoch, args.stop_epochs):
        offset = start_step * batch if epoch == start_epoch else 0
        sampler = EpochSampler(len(dataset), args.seed + epoch, offset, rank, world_size, batch)
        workers = args.workers_per_rank if args.workers_per_rank is not None else int(cfg.system.num_workers)
        loader = DataLoader(dataset, batch_size=local_batch, sampler=sampler, drop_last=True,
            num_workers=workers, pin_memory=True, collate_fn=strict_collate,
            worker_init_fn=worker_init, generator=torch.Generator().manual_seed(args.seed + epoch),
            **({'prefetch_factor': 2} if workers else {}))
        train_model.train()
        vision.eval()
        for step, batch_data in enumerate(loader, offset // batch):
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, info = train_model(batch_data)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(action.parameters(), cfg.training.grad_clip_norm, error_if_nonfinite=True)
            if args.smoke_steps:
                for name, p in action.named_parameters():
                    group = name.split('.')[0]
                    if p.grad is not None:
                        gradient_groups[group] = max(gradient_groups.get(group, 0.), float(p.grad.float().norm()))
            optimizer.step()
            scheduler.step()
            global_step += 1
            scalars = {k:float(v) for k,v in info.items() if isinstance(v, (float, int)) or (torch.is_tensor(v) and v.numel() == 1)}
            scalars['loss'] = float(loss)
            if world_size > 1:
                values = torch.tensor(list(scalars.values()), device='cuda', dtype=torch.float32)
                dist.all_reduce(values)
                scalars = dict(zip(scalars, (values/world_size).tolist()))
            record = {'epoch': epoch + 1, 'step_in_epoch': step + 1, 'steps_per_epoch': steps_per_epoch,
                'global_step': global_step, 'loss': float(loss), 'grad_norm': float(grad_norm),
                'lr': optimizer.param_groups[0]['lr'], 'step_seconds': time.monotonic() - tick,
                'elapsed_seconds': time.monotonic() - started,
                'resumed_global_step':first_step, 'world_size':world_size, 'global_batch_size':batch,
                'per_device_batch_size':local_batch, **scalars}
            if primary:
                metrics.write(json.dumps(record) + '\n')
            if primary and (global_step % 20 == 0 or global_step == first_step + 1):
                status = dict(record, status='training', gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
                write_json(args.output / 'status.json', status)
                print(json.dumps(status), flush=True)
            smoke_done = args.smoke_steps and global_step - first_step >= args.smoke_steps
            end_epoch = step + 1 == steps_per_epoch
            if global_step % args.save_steps == 0 or end_epoch or smoke_done:
                saved_epoch, saved_step = (epoch + 1, 0) if end_epoch else (epoch, step + 1)
                last_saved = args.output / 'latest.pt'
                states = [None] * world_size
                if world_size > 1:
                    dist.all_gather_object(states, rng_state())
                else:
                    states[0] = rng_state()
                if primary:
                    save_checkpoint(last_saved, action, optimizer, scheduler, saved_epoch, saved_step, global_step, cfg, initial_hash, states)
                    if end_epoch:
                        epoch_path = args.output / f'checkpoint_epoch_{epoch+1}.pt'
                        if not epoch_path.exists():
                            os.link(last_saved, epoch_path)
                    if global_step in cfg.get('experiment', {}).get('milestone_steps', []):
                        for current, target in ((last_saved, args.output / f'checkpoint_step_{global_step}.pt'),
                            (args.output / 'policy.pt', args.output / f'policy_step_{global_step}.pt')):
                            if not target.exists():
                                os.link(current, target)
                    write_json(args.output / 'checkpoint_status.json', {'checkpoint':str(last_saved), 'global_step':global_step, 'world_size':world_size})
                if world_size > 1:
                    dist.barrier()
            if smoke_done:
                if torch.equal(probe, probe_parameter.detach()):
                    raise ValueError('Action output weights did not update')
                learning_groups = (('task_embedding', 'embodiment_embedding', 'visual_token_pooler', 'world_model', 'action_models')
                    if is_gawm else ('concat_fusion', 'dino_adapters', 'blocks', 'future_feat_decoder'))
                for group in learning_groups:
                    if gradient_groups.get(group, 0.) <= 0:
                        raise ValueError(f'Missing learning signal for {group}')
                if any(p.grad is not None for p in vision.parameters()):
                    raise ValueError('Frozen DINO accumulated gradients')
                # Load the actual serialized result strictly, then compare predictions under the same RNG.
                wrapper.eval()
                rng = torch.cuda.get_rng_state()
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    before, _ = wrapper(batch_data)
                saved = torch.load(last_saved, map_location='cpu', weights_only=False)
                action.load_state_dict(saved['model_state_dict'], strict=True)
                del saved
                torch.cuda.set_rng_state(rng)
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    after, _ = wrapper(batch_data)
                if not torch.equal(before, after):
                    raise ValueError('Checkpoint reload changed fixed-RNG loss')
                replicas_identical = True
                if world_size > 1:
                    hashes = [None] * world_size
                    dist.all_gather_object(hashes, parameter_digest(action))
                    replicas_identical = len(set(hashes)) == 1
                    if not replicas_identical:
                        raise ValueError('DDP model replicas diverged')
                if primary:
                    write_json(args.output / 'smoke_result.json', {'status':'passed', 'updates':args.smoke_steps,
                    'finite_gradients':True, 'frozen_encoder':True, 'nonzero_gradient_groups':gradient_groups,
                    'checkpoint_reload_loss_difference':float((before-after).abs()),
                    'peak_cuda_gb':torch.cuda.max_memory_allocated()/1e9, 'last_record':record,
                    'world_size':world_size,'replicas_identical':replicas_identical})
                return
    if primary:
        metrics.close()
        write_json(args.output / 'status.json', {'status':'completed', 'global_step':global_step, 'checkpoint':str(last_saved)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('/data/gaoxiang/Code/LiLa-WAM'))
    p.add_argument('--stats', type=Path, default=Path('/data/gaoxiang/Code/LiLa-WAM/utils/stat-500-all.json'))
    p.add_argument('--audit', type=Path, default=Path('/data/gaoxiang/LiLaWAM_RoboTwin_Official/dataset_audit.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stop-epochs', type=int, default=12)
    p.add_argument('--save-steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--init-from', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--deterministic', action='store_true', help='Enable deterministic CUDA kernels for resume verification')
    p.add_argument('--workers-per-rank', type=int)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        run(args)
    except Exception as exc:
        if int(os.environ.get('RANK', 0)) == 0:
            write_json(args.output / 'status.json', {'status':'failed', 'error':repr(exc)})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
