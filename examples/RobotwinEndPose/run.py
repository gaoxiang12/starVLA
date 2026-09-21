"""Run matched feedback arms through smoke, training and unchanged fixed scenes."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np
from omegaconf import OmegaConf

from examples.RobotwinEndPose.prepare import ROOT, OUT, RUNS, digest, save

PYTHON = ROOT.parent/'.venvs/starVLA/bin/python'
SIM = ROOT.parent/'.venvs/RoboTwin/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--train-port', type=int, default=29831)
    parser.add_argument('--eval-port', type=int, default=6981)
    parser.add_argument('--phase', choices=['smoke','campaign'], required=True)
    parser.add_argument('--allow-shared-gpu', action='store_true',
        help='Allow existing processes only when at least 24 GiB is free; never stop them')
    parser.add_argument('--reuse-trained-smoke',action='store_true')
    args = parser.parse_args()
    if args.reuse_trained_smoke and args.phase != 'smoke':parser.error('Reuse restricted to smoke')
    import torch
    if not torch.cuda.is_available():
        save(OUT/f'{args.phase}_status.json',dict(state='blocked_no_cuda',
            reason='CUDA driver unavailable; no training or simulation started'))
        raise RuntimeError('CUDA driver unavailable; CPU preparation remains usable')
    status_path = OUT/f'{args.phase}_status.json'
    if status_path.exists():
        assert json.loads(status_path.read_text())['state'] == 'blocked_no_cuda', 'Existing run must be inspected, not overwritten'
    for variant in ('command','endpose'):
        assert json.loads((OUT/f'{variant}_audit_cpu.json').read_text())['state']=='cpu_verified_gpu_pending'
    if args.phase == 'campaign':
        assert json.loads((OUT/'smoke_status.json').read_text())['state']=='complete'
    protocol = json.loads((OUT/'protocol.json').read_text())
    sources = list((ROOT/'examples/RobotwinEndPose').rglob('*.py')) + [
        ROOT/'starVLA/robotwin_feedback.py',
        ROOT/'starVLA/model/framework/WM4A/GAWM.py',
        ROOT/'starVLA/model/modules/action_model/ACT_ActionHeader.py',
        ROOT/'starVLA/model/modules/spatial_focus.py',
        ROOT/'starVLA/dataloader/lerobot_datasets.py',
        ROOT/'starVLA/dataloader/gr00t_lerobot/datasets.py',
        ROOT/'starVLA/training/train_starvla.py',
        ROOT/'deployment/model_server/policy_wrapper.py',
        ROOT/'deployment/model_server/policy_norm_processor.py',
        ROOT/'examples/Robotwin/eval_files/model2robotwin_interface.py',
        ROOT/'examples/Robotwin/audits/run_qwen_gawm_ranking_case.py',
        ROOT.parent/'RoboTwin/envs/blocks_ranking_rgb.py',
        OUT/'protocol.json',OUT/'split.json',OUT/'initialization.json',OUT/'data_manifest.json']
    sources += list(OUT.glob('*.yaml')) + list(OUT.glob('*_statistics.json'))
    pinned = {str(p):digest(p) for p in sources}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
        PYTHONPATH=f'{ROOT}:{ROOT}/examples/Robotwin/eval_files', OMP_NUM_THREADS='4',
        NO_ALBUMENTATIONS_UPDATE='1',WANDB_MODE='disabled',STARVLA_DISABLE_TQDM='1',HF_HUB_OFFLINE='1')
    report = dict(state='starting',pid=os.getpid(),gpu=args.gpu,phase=args.phase,source_sha256=pinned,results=[])
    child = server = None

    def status(): save(status_path,dict(report,time=time.strftime('%Y-%m-%d %H:%M:%S')))
    def verify():
        for path, sha in pinned.items():
            assert digest(path) == sha, f'Source changed: {path}'
    def stop(sig, frame): raise RuntimeError(f'Signal {sig}')
    def cleanup(proc):
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
    def execute(command, log, timeout=86400):
        nonlocal child
        verify()
        with log.open('x') as stream:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        report.update(command=command,child_pid=child.pid);status()
        child.wait(timeout=timeout)
        assert child.returncode == 0, f'Failed command; see {log}'
        verify()
    def require_free_gpu():
        used, free = map(int, subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.used,memory.free',
            '--format=csv,noheader,nounits'],text=True).strip().split(','))
        assert used < 500 or (args.allow_shared_gpu and free >= 24576), (
            f'GPU {args.gpu}: used {used} MiB, free {free} MiB; no other job will be stopped')
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,stop)
    try:
        for variant in ('command','endpose'):
            stages = ['smoke'] if args.phase=='smoke' else ['warmup','joint']
            for stage in stages:
                require_free_gpu()
                cfg_path = OUT/f'{variant}_{stage}.yaml'; cfg=OmegaConf.load(cfg_path)
                run = RUNS/f'{variant}_{stage}'
                reuse = args.reuse_trained_smoke and run.exists()
                assert not run.exists() or reuse,run
                report.update(state='training',variant=variant,stage=stage);status()
                if not reuse:
                    execute([str(PYTHON.parent/'accelerate'),'launch','--config_file',
                        'starVLA/config/deepseeds/deepspeed_zero2.yaml','--num_processes','1',
                        '--main_process_port',str(args.train_port),'starVLA/training/train_starvla.py',
                        '--config_yaml',str(cfg_path)], OUT/f'{variant}_{stage}_trainer.log')
                metrics = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
                assert metrics[-1]['step'] == cfg.trainer.max_train_steps
                assert all(np.isfinite(v) for row in metrics for v in row.values() if isinstance(v,(int,float)))
                if not (run/'config.accessed.yaml').exists():
                    (run/'config.yaml').rename(run/'config.accessed.yaml')
                else:
                    assert (run/'config.yaml').read_text()==(run/'config.full.yaml').read_text()
                (run/'config.yaml').write_text((run/'config.full.yaml').read_text())
                checkpoint = run/'final_model/pytorch_model.pt'
                report.update(state='checking_deployment');status()
                execute([str(PYTHON),'-m','examples.RobotwinEndPose.audit','--variant',variant,
                    '--checkpoint',str(checkpoint),'--device','cuda'],OUT/f'{variant}_{stage}_deployment.log')
            require_free_gpu()
            with socket.socket() as sock: sock.bind(('127.0.0.1',args.eval_port))
            with (OUT/f'{variant}_{args.phase}_server.log').open('x') as stream:
                server=subprocess.Popen([str(PYTHON),'deployment/model_server/server_policy.py',
                    '--ckpt_path',str(checkpoint),'--port',str(args.eval_port),'--idle_timeout','-1'],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+600
            while True:
                assert server.poll() is None, 'Policy server failed'
                try:
                    with socket.create_connection(('127.0.0.1',args.eval_port),timeout=1): break
                except OSError:
                    if time.monotonic()>deadline: raise TimeoutError('Server startup')
                    time.sleep(2)
            records = protocol['records'][:1] if args.phase=='smoke' else protocol['records']
            for record in records:
                case=OUT/f'{variant}_{args.phase}_eval'/record['split']/f"scene_{record['scene_id']:03d}"
                case.parent.mkdir(parents=True,exist_ok=True)
                report.update(state='evaluating',scene=record);status()
                execute([str(SIM),'-m','examples.RobotwinEndPose.run_case','--seed',str(record['seed']),
                    '--checkpoint',str(checkpoint),'--port',str(args.eval_port),'--output',str(case)],
                    case.with_suffix('.log'),timeout=1800)
                result=json.loads((case/'result.json').read_text())
                assert result['state']=='complete' and result['seed']==record['seed']
                report['results'].append(dict(variant=variant,scene=record,success=result['success'],path=str(case/'result.json')))
            cleanup(server);server=None
        for record in records:
            paths=[OUT/f'{v}_{args.phase}_eval'/record['split']/f"scene_{record['scene_id']:03d}/result.json"
                for v in ('command','endpose')]
            pair=[json.loads(path.read_text()) for path in paths]
            assert pair[0]['initial']==pair[1]['initial'], f'Unpaired scene: {record}'
        summary = {}
        for split in ('development','test'):
            summary[split] = {v:dict(success=sum(r['success'] for r in report['results'] if r['variant']==v and r['scene']['split']==split),
                completed=sum(1 for r in report['results'] if r['variant']==v and r['scene']['split']==split))
                for v in ('command','endpose')}
        report.update(state='complete',summary=summary,paired_initials_verified=True)
    except BaseException as error:
        report.update(state='failed',error=repr(error));raise
    finally:
        cleanup(child);cleanup(server);status()


if __name__ == '__main__': main()
