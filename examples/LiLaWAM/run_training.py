"""Launch the standard StarVLA trainer with durable logs and a status file."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import psutil
from omegaconf import OmegaConf
from examples.LiLaWAM.prepare import write_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--port', type=int, default=29871)
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    config = Path(args.config).resolve()
    cfg = OmegaConf.load(config)
    output = Path(cfg.run_root_dir) / cfg.run_id
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'run_status.json'
    if args.detach:
        if status_path.exists() or (output/'launch.json').exists():
            raise RuntimeError('Existing run must be inspected before restarting')
        with (output/'supervisor.log').open('x') as stream:
            child = subprocess.Popen([sys.executable, '-m', 'examples.LiLaWAM.run_training',
                '--config', str(config), '--gpu', args.gpu, '--port', str(args.port)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
        write_json(output/'launch.json', dict(pid=child.pid, birth=psutil.Process(child.pid).create_time(),
                                              gpu=args.gpu, config=str(config)))
        print(f'Supervisor PID {child.pid}; status: {status_path}')
        return
    report = dict(state='starting', pid=os.getpid(), birth=psutil.Process().create_time(),
                  gpu=args.gpu, config=str(config), max_steps=int(cfg.trainer.max_train_steps))
    child = None
    log = output/'trainer.log'
    def status():
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0, log.stat().st_size-65536))
                text = f.read().decode(errors='replace')
            steps = re.findall(r'(\d+)/' + str(cfg.trainer.max_train_steps) + r'\b', text)
            if steps: report['last_observed_step'] = max(map(int, steps))
        write_json(status_path, dict(report, time=time.strftime('%Y-%m-%d %H:%M:%S')))
    def stop(sig, frame):
        raise RuntimeError(f'Signal {sig}')
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop)
    status()
    try:
        if cfg.framework.name != 'LiLaWAMTrain': raise ValueError('Expected a LiLaWAMTrain config')
        vtt = Path(cfg.framework.lila.task_vectors_path)
        stats = Path(cfg.datasets.vla_data.normalization_statistics_path)
        audit = json.loads((vtt.parent/'preparation_audit.json').read_text())
        if audit['status'] != 'prepared': raise ValueError('Dataset preparation has not passed')
        files = [config, vtt, stats, *ROOT.glob('starVLA/model/modules/lila/*.py'),
                 ROOT/'starVLA/model/framework/WM4A/LiLaWAMTrain.py',
                 ROOT/'examples/LiLaWAM/train_files/data_registry/data_config.py',
                 ROOT/'starVLA/training/train_starvla.py']
        write_json(output/'input_manifest.json', {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
        free = int(subprocess.check_output(['nvidia-smi','-i',args.gpu,
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free < 24576: raise RuntimeError(f'GPU {args.gpu} has only {free} MiB free')
        with socket.socket() as sock: sock.bind(('127.0.0.1',args.port))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE='1',
                   NO_ALBUMENTATIONS_UPDATE='1', WANDB_MODE='disabled', OMP_NUM_THREADS='4')
        command = [str(Path(sys.executable).with_name('accelerate')), 'launch',
            '--config_file','starVLA/config/deepseeds/deepspeed_zero2.yaml',
            '--num_processes','1','--main_process_port',str(args.port),
            'starVLA/training/train_starvla.py','--config_yaml',str(config)]
        with log.open('x') as stream:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                     stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            report.update(state='training',child_pid=child.pid,command=command);status()
            while child.poll() is None:
                time.sleep(10);status()
        if child.returncode: raise RuntimeError(f'Trainer exited {child.returncode}; inspect {log}')
        if not (output/'final_model/pytorch_model.pt').exists(): raise RuntimeError('Missing final checkpoint')
        report.update(state='complete')
    except BaseException as error:
        report.update(state='failed',error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL)
        status()


if __name__=='__main__': main()
