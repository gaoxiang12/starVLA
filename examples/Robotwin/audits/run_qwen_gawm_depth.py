"""Independent GPU campaign: 20-step smoke, 500+4500 training, depth ablations; exactly 20 development scenes."""
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

from examples.Robotwin.audits.prepare_qwen_gawm_depth import ROOT,BASE,OUT as PARENT,SHARED,QWEN,VARIANTS,digest,save,validate_config,validate_protocol

PYTHON=ROOT.parent/'.venvs/starVLA/bin/python'
SIM_PYTHON=ROOT.parent/'.venvs/RoboTwin/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--train-port',type=int,required=True)
    p.add_argument('--eval-port',type=int,required=True)
    args=p.parse_args()
    OUT=PARENT/args.variant
    status_path=OUT/'status.json';assert not status_path.exists()
    assert json.loads((OUT/'data_audit.json').read_text())['state']=='depth_data_verified'
    assert json.loads((OUT/'integration.json').read_text())['state']=='depth_optimization_verified'
    initialization=json.loads((OUT/'initialization_audit.json').read_text())
    assert initialization['state']=='depth_transfer_verified' and initialization['action_head_unchanged'] is True
    assert initialization['feature_boundary']['state']=='feature_boundary_verified'
    preparation=json.loads((OUT/'preparation.json').read_text())
    assert digest(QWEN)==json.loads((PARENT/'preparation.json').read_text())['qwen_checkpoint_sha256']
    for r in preparation['configs']:assert digest(r['path'])==r['sha256']
    sources=[Path(__file__),OUT/'preparation.json',OUT/'protocol.json',OUT/'data_audit.json',OUT/'integration.json',
        ROOT/'starVLA/model/framework/VLM4A/QwenGAWM.py',
        ROOT/'starVLA/model/framework/VLM4A/QwenGAWMDepth.py',
        PARENT/'preparation.json',PARENT/'protocol.json',
        PYTHON.parent.parent/'lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py',
        ROOT/'starVLA/model/modules/action_model/ACT_ActionHeader.py',
        ROOT/'starVLA/model/modules/vlm/QWen3.py',ROOT/'starVLA/model/framework/base_framework.py',
        ROOT/'starVLA/training/train_starvla.py',ROOT/'starVLA/dataloader/lerobot_datasets.py',
        ROOT/'starVLA/dataloader/gr00t_lerobot/datasets.py',ROOT/'starVLA/dataloader/gr00t_lerobot/registry.py',
        ROOT/'examples/Robotwin/train_files/data_registry/data_config.py',
        ROOT/'examples/UnifiedPretrain/train_files/data_registry/data_config.py',
        ROOT/'examples/Robotwin/audits/audit_qwen_gawm_depth.py',
        ROOT/'examples/Robotwin/audits/run_qwen_gawm_chunk_case.py',
        ROOT/'examples/Robotwin/audits/summarize_qwen_gawm_depth.py',
        ROOT/'examples/Robotwin/eval_files/model2robotwin_interface.py',
        ROOT/'deployment/model_server/policy_wrapper.py',ROOT/'deployment/model_server/policy_norm_processor.py',
        ROOT.parent/'RoboTwin/envs/blocks_ranking_rgb.py',ROOT.parent/'RoboTwin/task_config/_eval_step_limit.yml']
    sources += [OUT/'initialization_audit.json',ROOT/'examples/Robotwin/audits/prepare_qwen_gawm_depth.py',
        ROOT/'examples/Robotwin/audits/audit_qwen_gawm_ranking.py']
    sources += [Path(r['path']) for r in preparation['configs']]
    pinned={str(path):digest(path) for path in sources}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,PYTHONPATH=f'{ROOT}:{ROOT}/examples/Robotwin/eval_files',
        PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',
        NO_ALBUMENTATIONS_UPDATE='1',WANDB_MODE='disabled',STARVLA_DISABLE_TQDM='1')
    child=server=None
    report=dict(state='starting',variant=args.variant,gpu=args.gpu,supervisor_pid=os.getpid(),supervisor_birth=psutil.Process().create_time(),
                completed_stages=[],results=[],source_sha256=pinned)
    def status():save(status_path,dict(report,time=time.strftime('%Y-%m-%d %H:%M:%S'),
        child_pid=child.pid if child and child.poll() is None else None))
    def check():
        for path,sha in pinned.items():assert digest(path)==sha,f'Pinned source changed: {path}'
    def cleanup(proc):
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
    def execute(command,log_path,timeout=None):
        nonlocal child
        check()
        with log_path.open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        report.update(command=command,child_birth=psutil.Process(child.pid).create_time())
        start=time.monotonic()
        while child.poll() is None:
            status()
            if timeout and time.monotonic()-start>timeout:raise RuntimeError(f'Timeout: {log_path}')
            time.sleep(5)
        assert child.returncode==0,f'Child failed ({child.returncode}): {log_path}'
        check()
    def wait_gpu():
        while True:
            used=int(subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.used',
                '--format=csv,noheader,nounits'],text=True).strip())
            if used<500:return
            report.update(state='waiting_for_gpu',used_mib=used);status();time.sleep(10)
    def start_server(checkpoint,port,label):
        nonlocal server
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
        with (OUT/f'{label}_server.log').open('x') as log:
            server=subprocess.Popen([str(PYTHON),'deployment/model_server/server_policy.py','--ckpt_path',str(checkpoint),
                '--port',str(port),'--idle_timeout','-1'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        report.update(state='starting_server',server_pid=server.pid,server_birth=psutil.Process(server.pid).create_time())
        start=time.monotonic()
        while True:
            if server.poll() is not None:raise RuntimeError('Server exited during startup')
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=1):break
            except OSError:
                if time.monotonic()-start>600:raise RuntimeError('Server startup timeout')
                status();time.sleep(2)
    def scene(checkpoint,record,port,smoke=False):
        case=OUT/('smoke_eval' if smoke else 'eval')/record['split']/f"scene_{record['scene_id']:03d}"
        case.parent.mkdir(parents=True,exist_ok=True)
        report.update(state='smoke_evaluating' if smoke else 'evaluating',current_scene=record)
        command=[str(SIM_PYTHON),'examples/Robotwin/audits/run_qwen_gawm_chunk_case.py',
            '--seed',str(record['seed']),'--checkpoint',str(checkpoint),'--port',str(port),'--output',str(case),
            '--action-horizon','16','--execute-horizon','16']
        if smoke:command+=['--smoke-actions','33']
        execute(command,case.with_suffix('.log'),timeout=1800)
        result=json.loads((case/'result.json').read_text());assert result['state']=='complete'
        assert result['seed']==record['seed'] and result['engineering_smoke']==smoke
        if smoke:
            baseline=json.loads((BASE/'qwen_act_smoke_eval/development/scene_000/result.json').read_text())
            assert result['initial']==baseline['initial']
            assert result['actions']==33 or result['success']
            report['smoke_evaluation']=dict(path=str(case/'result.json'),actions=result['actions'],
                note='Engineering budget only; excluded from all benchmark success rates')
        else:report['results'].append(dict(scene=record,success=result['success'],path=str(case/'result.json')))
    def stop(sig,frame):raise InterruptedError(f'Signal {sig}')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    protocol=json.loads((OUT/'protocol.json').read_text());validate_protocol(protocol)
    try:
        for index,stage in enumerate(('smoke','warmup','joint')):
            wait_gpu();cfg_path=OUT/f'{stage}.yaml';cfg=OmegaConf.load(cfg_path);run=SHARED/cfg.run_id
            validate_config(cfg,stage,args.variant)
            assert not run.exists()
            report.update(state='training',stage=stage,output=str(run),optimizer_steps=int(cfg.trainer.max_train_steps))
            execute([str(PYTHON.parent/'accelerate'),'launch','--config_file','starVLA/config/deepseeds/deepspeed_zero2.yaml',
                '--num_processes','1','--main_process_port',str(args.train_port+index),
                'starVLA/training/train_starvla.py','--config_yaml',str(cfg_path)],OUT/f'{stage}_trainer.log')
            metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            assert metrics[-1]['step']==int(cfg.trainer.max_train_steps)
            assert all(np.isfinite(v) for row in metrics for v in row.values() if isinstance(v,(int,float)))
            checkpoint=run/'final_model/pytorch_model.pt'
            (run/'config.yaml').rename(run/'config.accessed.yaml')
            (run/'config.yaml').write_text((run/'config.full.yaml').read_text())
            report.update(state='auditing_checkpoint',checkpoint=str(checkpoint))
            audit=OUT/f'{stage}_deployment_audit.json'
            execute([str(PYTHON),'-m','examples.Robotwin.audits.audit_qwen_gawm_depth','--mode','checkpoint',
                '--checkpoint',str(checkpoint),'--output',str(audit)],OUT/f'{stage}_deployment_audit.log')
            assert json.loads(audit.read_text())['state']=='strict_deployment_verified'
            report['completed_stages'].append(dict(stage=stage,checkpoint=str(checkpoint),metrics=metrics[-1]))
            if stage=='smoke':
                start_server(checkpoint,args.eval_port,'smoke')
                scene(checkpoint,protocol['records'][0],args.eval_port,smoke=True)
                cleanup(server);server=None
        report.update(state='measuring_performance')
        execute([str(PYTHON),'-m','examples.Robotwin.audits.audit_qwen_gawm_depth','--mode','performance',
            '--checkpoint',str(checkpoint),'--output',str(OUT/'performance.json')],OUT/'performance.log')
        wait_gpu();start_server(checkpoint,args.eval_port+1,'formal')
        for record in protocol['records']:
            assert record['split']=='development'
            scene(checkpoint,record,args.eval_port+1);status()
        cleanup(server);server=None
        assert len(report['results'])==20
        execute([str(PYTHON),'-m','examples.Robotwin.audits.summarize_qwen_gawm_depth','--variant',args.variant],
            OUT/'development_comparison.log')
        report.update(state='complete',summarized_splits=['development'],completed_scenes=20,
                      successes=sum(r['success'] for r in report['results']),test_scenes=0)
    except InterruptedError as error:
        report.update(state='stopped_by_user',error=repr(error));raise
    except BaseException as error:
        report.update(state='failed',error=repr(error));raise
    finally:cleanup(child);cleanup(server);status()


if __name__=='__main__':main()
