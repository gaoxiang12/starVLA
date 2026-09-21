"""Run matched RGB-only control/dense/local experiments, restoring prior jobs."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

ROOT=Path(__file__).resolve().parents[3]
CAMP=ROOT/'playground/Checkpoints/gawm_rgb_focus_20260907'
PREVIOUS=ROOT/'playground/Checkpoints/gawm_ranking_spatial_ablation_20260907'
PYTHON=ROOT.parent/'.venvs/starVLA/bin/python'
ACTIVE={}
PAUSED=[]
HISTORY=[]
GPU={'control':'0','dense':'4','local':'5'}


def save(name,payload):
    path=CAMP/name
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(path)


def status(state):
    save('status.json',dict(state=state,time=time.strftime('%Y-%m-%d %H:%M:%S'),supervisor_pid=os.getpid(),
         active={v:{k:x for k,x in item.items() if k!='process'} for v,item in ACTIVE.items()},history=HISTORY))


def matches(item):
    try:return psutil.Process(item['pid']).create_time()==item['birth']
    except psutil.NoSuchProcess:return False


def pause_previous():
    candidates=[]
    for p in psutil.process_iter(['pid','cmdline']):
        if str(PREVIOUS/'supervise_experiment.py') in (p.info['cmdline'] or []): candidates.append(p)
    if len(candidates)!=1:raise RuntimeError('Expected exactly one previous ranking supervisor')
    parent=candidates[0]
    for p in [parent]+parent.children(recursive=True):
        try:
            if p.status()==psutil.STATUS_STOPPED:continue
            item=dict(pid=p.pid,birth=p.create_time(),cmd=p.cmdline())
            os.kill(p.pid,signal.SIGSTOP);PAUSED.append(item)
            save('paused_previous.json',PAUSED)
        except psutil.NoSuchProcess:pass
    print('Paused previous ranking experiment; preserving its progress',flush=True)


def restore_previous():
    # Restore children before the scheduler to avoid premature queue changes.
    for item in reversed(PAUSED):
        if matches(item):os.kill(item['pid'],signal.SIGCONT)
    save('previous_resumed.json',dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),processes=len(PAUSED)))


def launch(variant,stage):
    smoke=stage=='smoke'
    run_id=f'gawm_rgb_focus_{variant}_{"smoke_v2" if smoke else "5k"}_20260907'
    run=ROOT/'playground/Checkpoints'/run_id
    run.mkdir(exist_ok=True)
    gpu=GPU[variant]
    if stage in ('smoke','train'):
        cfg=CAMP/f'{variant}_smoke.yaml' if smoke else ROOT/f'examples/Robotwin/train_files/starvla_gawm_rgb_focus_{variant}.yaml'
        if (run/'config.full.yaml').exists():raise RuntimeError(f'Refusing to overwrite {run}')
        command=[str(PYTHON.parent/'accelerate'),'launch','--config_file',str(ROOT/'starVLA/config/deepseeds/deepspeed_zero2.yaml'),
                 '--num_processes','1','--main_process_port',str(29800+int(gpu)),str(ROOT/'starVLA/training/train_starvla.py'),'--config_yaml',str(cfg)]
    else:
        command=['bash',str(ROOT/'examples/Robotwin/eval_files/eval_ranking_single.sh'),run_id,'blocks_ranking_rgb',gpu,str(6000+int(gpu))]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,WANDB_MODE='disabled',PYTHONNOUSERSITE='1',NO_ALBUMENTATIONS_UPDATE='1',
             PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4')
    with (run/f'{stage}.log').open('x') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    ACTIVE[variant]=dict(process=p,pid=p.pid,stage=stage,run_id=run_id,gpu=gpu)
    print('Launched',variant,stage,p.pid,flush=True)


def verify(item):
    run=ROOT/'playground/Checkpoints'/item['run_id']
    if item['stage']=='eval':
        if not (run/'eval_summary/robotwin_eval_summary.json').is_file():raise RuntimeError('Missing final evaluation summary')
        return
    expected=10 if item['stage']=='smoke' else 5000
    metrics=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['step']==expected and (run/'final_model/pytorch_model.pt').stat().st_size>1000000
    for row in metrics:
        assert all(math.isfinite(v) for v in row.values() if isinstance(v,float))
    assert (run/'validation_per_task.jsonl').is_file()
    if 'incompatible or unexpected checkpoint parameters' in (run/f'{item["stage"]}.log').read_text():
        raise RuntimeError('Unexpected checkpoint migration skips')


def shutdown(signum,frame):raise RuntimeError(f'Stopping supervisor on signal {signum}')


def main():
    assert (CAMP/'preflight.json').is_file()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,shutdown)
    try:
        pause_previous()
        for v in GPU:launch(v,'smoke')
        while ACTIVE:
            for v,item in list(ACTIVE.items()):
                code=item['process'].poll()
                if code is None:continue
                if code:raise RuntimeError(f'{v} {item["stage"]} failed: {code}')
                verify(item);HISTORY.append({k:x for k,x in item.items() if k!='process'})
                del ACTIVE[v]
            status('smoke_running' if ACTIVE else 'smoke_passed')
            if ACTIVE:time.sleep(10)
        save('smokes_passed.json',dict(history=HISTORY))
        # Optional gate allows manual inspection of real smoke outputs before training.
        while not (CAMP/'release_training').exists():
            status('awaiting_smoke_review');time.sleep(10)
        for v in GPU:launch(v,'train')
        while ACTIVE:
            for v,item in list(ACTIVE.items()):
                code=item['process'].poll()
                if code is None:continue
                if code:raise RuntimeError(f'{v} {item["stage"]} failed: {code}')
                verify(item);HISTORY.append({k:x for k,x in item.items() if k!='process'})
                del ACTIVE[v]
                if item['stage']=='train':launch(v,'eval')
            status('running' if ACTIVE else 'complete')
            if ACTIVE:time.sleep(15)
    except BaseException as e:
        save('failure.json',dict(error=repr(e),time=time.strftime('%Y-%m-%d %H:%M:%S')))
        for item in ACTIVE.values():
            if item['process'].poll() is None:
                os.killpg(item['pid'],signal.SIGTERM)
        time.sleep(5)
        for item in ACTIVE.values():
            if item['process'].poll() is None:os.killpg(item['pid'],signal.SIGKILL)
        status('failed')
        raise
    finally:restore_previous()


if __name__=='__main__':main()
