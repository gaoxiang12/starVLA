"""Durable smoke/train/deployment/full-sorting campaign for one comparison arm."""
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
import psutil

from examples.Robotwin.audits.prepare_qwen_gawm_ranking import ROOT, OUT, SHARED, digest, save

PYTHON=ROOT.parent/'.venvs/starVLA/bin/python'
SIM_PYTHON=ROOT.parent/'.venvs/RoboTwin/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['gawm','qwen_mlp','qwen_act'],required=True)
    p.add_argument('--gpu',required=True)
    p.add_argument('--train-port',type=int,required=True)
    p.add_argument('--eval-port',type=int,required=True)
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--reuse-trained-smoke',action='store_true',
                   help='Re-audit an already completed 20-step smoke after an engineering fix')
    args=p.parse_args()
    if args.reuse_trained_smoke and not args.smoke_only:p.error('Reuse is restricted to smoke checkpoints')
    suffix='smoke' if args.smoke_only else 'campaign'
    status_path=OUT/f'{args.variant}_{suffix}_status.json'
    assert not status_path.exists(),status_path
    assert json.loads((OUT/'data_audit.json').read_text())['state']=='full_trajectory_shared_loader_verified'
    assert json.loads((OUT/f'{args.variant}_integration.json').read_text())['state']=='real_data_optimization_verified'
    if not args.smoke_only:
        for variant in ('gawm','qwen_mlp','qwen_act'):
            assert json.loads((OUT/f'{variant}_smoke_status.json').read_text())['state']=='complete'
    stages=['smoke'] if args.smoke_only else ['warmup','joint']
    preparation=json.loads((OUT/'preparation.json').read_text())
    prepared={r['name']:r for r in preparation['configs']}
    sources=[Path(__file__),OUT/'protocol.json',OUT/'data_audit.json',OUT/'preparation.json',
        ROOT/'examples/Robotwin/audits/audit_qwen_gawm_ranking.py',
        ROOT/'examples/Robotwin/audits/run_qwen_gawm_ranking_case.py',
        ROOT/'examples/Robotwin/audits/prepare_qwen_gawm_ranking.py',
        ROOT/'starVLA/model/framework/VLM4A/QwenGAWM.py',
        ROOT/'starVLA/model/framework/WM4A/GAWM.py',
        ROOT/'starVLA/model/modules/action_model/ACT_ActionHeader.py',
        ROOT/'starVLA/model/modules/action_model/MLP_ActionHeader.py',
        ROOT/'starVLA/dataloader/lerobot_datasets.py',ROOT/'starVLA/dataloader/gr00t_lerobot/datasets.py',
        ROOT/'starVLA/training/train_starvla.py',ROOT/'starVLA/model/modules/vlm/QWen3.py',
        ROOT/'deployment/model_server/policy_wrapper.py',ROOT/'deployment/model_server/policy_norm_processor.py',
        ROOT/'examples/Robotwin/eval_files/model2robotwin_interface.py',
        ROOT.parent/'RoboTwin/envs/blocks_ranking_rgb.py',ROOT.parent/'RoboTwin/task_config/_eval_step_limit.yml']
    for stage in stages:
        r=prepared[f'{args.variant}_{stage}'];path=Path(r['path'])
        assert digest(path)==r['sha256'];sources.append(path)
    pinned={str(path):digest(path) for path in sources}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
        PYTHONPATH=f'{ROOT}:{ROOT}/examples/Robotwin/eval_files',OMP_NUM_THREADS='4',
        NO_ALBUMENTATIONS_UPDATE='1',WANDB_MODE='disabled',STARVLA_DISABLE_TQDM='1',HF_HUB_OFFLINE='1')
    child=server=None
    report=dict(state='starting',variant=args.variant,gpu=args.gpu,supervisor_pid=os.getpid(),
        supervisor_birth=psutil.Process().create_time(),source_sha256=pinned,completed_stages=[],results=[])
    def status():
        save(status_path,dict(report,time=time.strftime('%Y-%m-%d %H:%M:%S'),
            child_pid=child.pid if child and child.poll() is None else None))
    def verify_sources():
        for path,sha in pinned.items():
            assert digest(path)==sha,f'Pinned source changed: {path}'
    def stop(sig,frame):raise RuntimeError(f'Signal {sig}')
    def cleanup(proc):
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
    def execute(command,log_path,timeout=None):
        nonlocal child
        verify_sources()
        with log_path.open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        report.update(command=command,child_birth=psutil.Process(child.pid).create_time())
        start=time.monotonic()
        while child.poll() is None:
            status()
            if timeout and time.monotonic()-start>timeout:raise RuntimeError(f'Child timeout: {log_path}')
            time.sleep(5)
        assert child.returncode==0,f'Child failed ({child.returncode}): {log_path}'
        verify_sources()
    def wait_gpu():
        while True:
            used=int(subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.used',
                '--format=csv,noheader,nounits'],text=True).strip())
            if used<500:return
            report.update(state='waiting_for_gpu',used_mib=used);status();time.sleep(10)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    try:
        for index,stage in enumerate(stages):
            wait_gpu()
            name=f'{args.variant}_{stage}';cfg_path=OUT/f'{name}.yaml';cfg=OmegaConf.load(cfg_path)
            run=SHARED/name
            if args.reuse_trained_smoke:
                assert (run/'final_model/pytorch_model.pt').is_file(),run
            else:
                assert not run.exists(),run
            report.update(state='training',current_stage=stage,output=str(run),optimizer_steps=int(cfg.trainer.max_train_steps),
                initial_checkpoint=str(cfg.trainer.pretrained_checkpoint))
            if not args.reuse_trained_smoke:
                execute([str(PYTHON.parent/'accelerate'),'launch','--config_file','starVLA/config/deepseeds/deepspeed_zero2.yaml',
                    '--num_processes','1','--main_process_port',str(args.train_port+index),
                    'starVLA/training/train_starvla.py','--config_yaml',str(cfg_path)],OUT/f'{name}_trainer.log')
            metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            assert metrics[-1]['step']==int(cfg.trainer.max_train_steps)
            assert all(np.isfinite(v) for row in metrics for v in row.values() if isinstance(v,(int,float)))
            checkpoint=run/'final_model/pytorch_model.pt'
            # Access-tracked snapshots can omit loader-only deployment settings.
            # Export the actual full run config and retain the compact snapshot for audit.
            full=run/'config.full.yaml';compact=run/'config.yaml'
            assert full.is_file()
            if not (run/'config.accessed.yaml').exists():
                compact.rename(run/'config.accessed.yaml')
            else:
                assert compact.read_text()==full.read_text()
            compact.write_text(full.read_text())
            save(run/'deployment_config_export.json',dict(source=str(full),sha256=digest(full),
                compact_snapshot='config.accessed.yaml',reason='Preserve image size, color, masks and embodiment metadata'))
            report.update(state='auditing_checkpoint',checkpoint=str(checkpoint))
            audit=OUT/f'{name}_deployment_audit.json'
            execute([str(PYTHON),'-m','examples.Robotwin.audits.audit_qwen_gawm_ranking','--mode','checkpoint',
                '--checkpoint',str(checkpoint),'--output',str(audit)],OUT/f'{name}_deployment_audit.log')
            assert json.loads(audit.read_text())['state']=='strict_deployment_verified'
            report['completed_stages'].append(dict(stage=stage,checkpoint=str(checkpoint),metrics=metrics[-1]))
        wait_gpu()
        with socket.socket() as sock:sock.bind(('127.0.0.1',args.eval_port))
        with (OUT/f'{args.variant}_{suffix}_server.log').open('x') as log:
            server=subprocess.Popen([str(PYTHON),'deployment/model_server/server_policy.py','--ckpt_path',str(checkpoint),
                '--port',str(args.eval_port),'--idle_timeout','-1'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        report.update(state='starting_server',server_pid=server.pid)
        deadline=time.monotonic()+600
        while True:
            if server.poll() is not None:raise RuntimeError('Server failed during startup')
            try:
                with socket.create_connection(('127.0.0.1',args.eval_port),timeout=1):break
            except OSError:
                if time.monotonic()>deadline:raise RuntimeError('Server startup timeout')
                status();time.sleep(2)
        protocol=json.loads((OUT/'protocol.json').read_text())
        records=protocol['records'][:1] if args.smoke_only else protocol['records']
        for record in records:
            case=OUT/f'{args.variant}_{suffix}_eval'/record['split']/f"scene_{record['scene_id']:03d}"
            case.parent.mkdir(parents=True,exist_ok=True)
            report.update(state='evaluating',current_scene=record)
            execute([str(SIM_PYTHON),'examples/Robotwin/audits/run_qwen_gawm_ranking_case.py',
                '--seed',str(record['seed']),'--checkpoint',str(checkpoint),'--port',str(args.eval_port),
                '--output',str(case)],case.with_suffix('.log'),timeout=1800)
            result=json.loads((case/'result.json').read_text())
            assert result['state']=='complete' and result['seed']==record['seed']
            report['results'].append(dict(scene=record,success=result['success'],path=str(case/'result.json')))
            status()
        report.update(state='complete',completed_scenes=len(report['results']),
                      successes=sum(r['success'] for r in report['results']))
    except BaseException as error:
        report.update(state='failed',error=repr(error));raise
    finally:
        cleanup(child);cleanup(server);status()


if __name__=='__main__':main()
