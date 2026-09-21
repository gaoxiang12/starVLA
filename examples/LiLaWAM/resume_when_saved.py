"""Resume on a free GPU after a complete checkpoint; stop source only after progress.

This one-shot handoff preserves model/optimizer/scheduler state. It never stops
the source if the replacement fails to resume. Process birth times guard signals.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
from omegaconf import OmegaConf
from examples.LiLaWAM.prepare import write_json

ROOT = Path(__file__).resolve().parents[2]


def tail(path, size=262144):
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read().decode(errors='replace')


def owned_supervisor(identity):
    process = psutil.Process(identity['pid'])
    if abs(process.create_time() - identity['birth']) > .01:
        raise RuntimeError('Supervisor PID was reused')
    if process.uids().real != os.getuid():
        raise RuntimeError('Supervisor belongs to another user')
    command = process.cmdline()
    if 'examples.LiLaWAM.run_training' not in command or identity['config'] not in command:
        raise RuntimeError('Unexpected supervisor command')
    return process


def validate_checkpoint(state_dir, step, accumulation):
    import torch
    tag = (state_dir / 'latest').read_text().strip()
    model = torch.load(state_dir / tag / 'mp_rank_00_model_states.pt',
                       map_location='cpu', weights_only=False, mmap=True)
    scheduler = torch.load(state_dir / 'scheduler.bin', map_location='cpu', weights_only=False)
    random = torch.load(state_dir / 'random_states_0.pkl', map_location='cpu', weights_only=False)
    if model['global_steps'] != step or scheduler['last_epoch'] != step:
        raise ValueError('Checkpoint optimizer/scheduler step does not match filename')
    if random['step'] % accumulation:
        raise ValueError('Saved Accelerate counter is not aligned to the new accumulation boundary')
    optim = list((state_dir / tag).glob('*optim_states.pt'))
    if len(optim) != 1 or not optim[0].stat().st_size:
        raise ValueError('Expected one complete single-GPU optimizer state')
    return dict(optimizer_step=model['global_steps'], scheduler_step=scheduler['last_epoch'],
                accelerator_microstep=random['step'], target_accumulation=accumulation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--target-config', required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--gpus', default='1,3')
    parser.add_argument('--port', type=int, default=29875)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    source = Path(args.source_run).resolve()
    config = Path(args.target_config).resolve()
    cfg = OmegaConf.load(config)
    target = Path(cfg.run_root_dir) / cfg.run_id
    if source == target or not cfg.trainer.is_resume:
        raise ValueError('Use a new run directory and is_resume: true')
    target.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, '-m', 'examples.LiLaWAM.resume_when_saved',
                   *[arg for arg in sys.argv[1:] if arg != '--detach']]
        with (target / 'handoff.log').open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(target / 'handoff_launch.json', dict(pid=child.pid,
                   birth=psutil.Process(child.pid).create_time(), command=command))
        print(f'Handoff PID {child.pid}; status: {target / "handoff_status.json"}')
        return

    identity = json.loads((source / 'launch.json').read_text())
    source_cfg = OmegaConf.load(identity['config'])
    # Only compute batching and run identity may change during this handoff.
    old, new = [OmegaConf.to_container(c, resolve=True) for c in (source_cfg, cfg)]
    for c in (old, new):
        for key in ('run_id', 'run_root_dir'):
            c.pop(key, None)
        for key in ('encoder_batch_size', 'efficient_views'):
            c['framework']['lila'].pop(key, None)
        c['datasets']['vla_data'].pop('per_device_batch_size')
        c['trainer'].pop('gradient_accumulation_steps')
        c['trainer'].pop('is_resume')
    if old != new:
        raise ValueError('Handoff changes settings beyond compute batching')
    def batch(c):
        return c.datasets.vla_data.per_device_batch_size * c.trainer.gradient_accumulation_steps
    if batch(cfg) != batch(source_cfg):
        raise ValueError('Effective batch must stay fixed')
    owned_supervisor(identity)
    files = [config, Path(cfg.framework.lila.task_vectors_path),
             Path(cfg.datasets.vla_data.normalization_statistics_path),
             ROOT / 'starVLA/model/framework/WM4A/LiLaWAMTrain.py',
             ROOT / 'starVLA/training/train_starvla.py',
             ROOT / 'starVLA/dataloader/lerobot_datasets.py',
             ROOT / 'examples/LiLaWAM/train_files/data_registry/data_config.py',
             *ROOT.glob('starVLA/model/modules/lila/*.py')]
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    report = dict(state='waiting_for_checkpoint', pid=os.getpid(),
                  source=str(source), target=str(target), checkpoint_step=args.step,
                  source_identity=identity, input_hashes=hashes)
    status_path = target / 'handoff_status.json'
    def status(state):
        report.update(state=state, time=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(status_path, report)
    status('waiting_for_checkpoint')
    deadline = time.monotonic() + 8 * 3600
    state_dir = source / f'checkpoints/steps_{args.step}_training_state'
    weights = source / f'checkpoints/steps_{args.step}_pytorch_model.pt'
    saved = False
    try:
        while time.monotonic() < deadline:
            owned_supervisor(identity)
            src_status = json.loads((source / 'run_status.json').read_text())
            if src_status['state'] != 'training':
                raise RuntimeError('Source is no longer training; leaving it untouched')
            report['source_step'] = src_status.get('last_observed_step', 0)
            saved = saved or f'Full training state saved at {state_dir}' in tail(source / 'trainer.log')
            if saved and weights.is_file() and (state_dir / 'latest').is_file():
                gpu = None
                for candidate in args.gpus.split(','):
                    free = int(subprocess.check_output(['nvidia-smi', '-i', candidate,
                        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
                    if free >= 24576:
                        gpu = candidate
                        break
                if gpu is not None:
                    break
                status('waiting_for_free_gpu')
            else:
                status('waiting_for_checkpoint')
            time.sleep(20)
        else:
            raise TimeoutError('No complete checkpoint/free GPU within 8 hours')
        for filename, digest in hashes.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Input changed while waiting: {filename}')
        report['checkpoint_counters'] = validate_checkpoint(
            state_dir, args.step, int(cfg.trainer.gradient_accumulation_steps))
        checkpoint_dir = target / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=False)
        for path in (weights, state_dir):
            (checkpoint_dir / path.name).symlink_to(path, target_is_directory=path.is_dir())
        report.update(gpu=gpu, resume_state=str(state_dir))
        status('starting_replacement')
        subprocess.run([sys.executable, '-m', 'examples.LiLaWAM.run_training',
            '--config', str(config), '--gpu', gpu, '--port', str(args.port), '--detach'],
            cwd=ROOT, check=True)
        deadline = time.monotonic() + 15 * 60
        while time.monotonic() < deadline:
            target_status = target / 'run_status.json'
            if target_status.exists():
                replacement = json.loads(target_status.read_text())
                if replacement['state'] == 'failed':
                    raise RuntimeError('Replacement failed; source kept running')
                resumed = f'Resumed from checkpoint: {checkpoint_dir / state_dir.name}' in tail(target / 'trainer.log')
                if resumed and replacement.get('last_observed_step', 0) > args.step:
                    owned_supervisor(json.loads((target / 'launch.json').read_text()))
                    old_process = owned_supervisor(identity)
                    old_process.send_signal(signal.SIGTERM)
                    report['source_stop_requested'] = True
                    for _ in range(45):
                        if not old_process.is_running() or old_process.status() == psutil.STATUS_ZOMBIE:
                            break
                        time.sleep(1)
                    else:
                        raise RuntimeError('Replacement running, but source supervisor has not stopped')
                    report['replacement_step'] = replacement['last_observed_step']
                    status('handoff_complete')
                    return
            time.sleep(10)
        raise TimeoutError('Replacement progress not confirmed; source kept running')
    except BaseException as error:
        report['error'] = repr(error)
        status('failed_after_source_stop' if report.get('source_stop_requested') else 'failed_source_not_stopped')
        raise


if __name__ == '__main__':
    main()
