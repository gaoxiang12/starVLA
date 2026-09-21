"""Switch this experiment at a saved step, retaining the paused source until healthy."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

from examples.LiLaWAM.official_robotwin_data import write_json


def checked_process(pid, token):
    p = psutil.Process(pid)
    if p.uids().real != os.getuid() or token not in ' '.join(p.cmdline()):
        raise RuntimeError(f'Process identity mismatch: {pid}, expected {token}')
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--target-run', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    args = parser.parse_args()
    old, new = args.source_run, args.target_run
    if (new/'stage1').exists():
        raise RuntimeError('Target already has training state; refusing to overwrite')
    new.mkdir(parents=True, exist_ok=True)
    journal = {'status':'waiting_for_eight_gpu_verification', 'source_run':str(old), 'target_run':str(new)}
    def update(status, **kw):
        journal.update(status=status, updated=time.time(), **kw)
        write_json(new/'handoff_status.json',journal)
        print(json.dumps(journal),flush=True)
    deadline=time.monotonic()+1800
    while not args.verification.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('Eight-GPU verification did not finish')
        update('waiting_for_eight_gpu_verification')
        time.sleep(10)
    proof=json.loads(args.verification.read_text())
    if proof.get('status')!='passed' or proof.get('world_size')!=8 or not proof.get('replicas_identical'):
        raise RuntimeError('Eight-GPU verification did not pass')
    state=json.loads((old/'run_status.json').read_text())
    if state['status']!='stage1':
        raise RuntimeError('Expected source stage1')
    supervisor=checked_process(state['pid'],'examples.LiLaWAM.run_official_robotwin')
    trainer=checked_process(state['child_pid'],'examples.LiLaWAM.train_official_robotwin')
    before=json.loads((old/'stage1/checkpoint_status.json').read_text())['global_step']
    update('waiting_for_checkpoint', initial_checkpoint=before,
           source_supervisor_pid=supervisor.pid,source_trainer_pid=trainer.pid)
    deadline=time.monotonic()+1800
    while True:
        checkpoint=json.loads((old/'stage1/checkpoint_status.json').read_text())
        if checkpoint['global_step']>before:
            break
        if time.monotonic()>deadline or not trainer.is_running():
            raise RuntimeError('Source did not reach a new checkpoint')
        time.sleep(.2)
    paused=False
    child=None
    try:
        supervisor.suspend()
        trainer.suspend()
        paused=True
        source_step=checkpoint['global_step']
        (new/'stage1').mkdir()
        os.link(old/'stage1/latest.pt',new/'stage1/latest.pt')
        os.link(old/'stage1/policy.pt',new/'stage1/policy.pt')
        for path in (old/'stage1').glob('checkpoint_epoch_*.pt'):
            os.link(path,new/'stage1'/path.name)
        (new/'upstream_source').symlink_to(old/'upstream_source',target_is_directory=True)
        (new/'smoke').mkdir()
        shutil.copy2(args.verification,new/'smoke/smoke_result.json')
        shutil.copy2(old/'reproduction_manifest.json',new/'source_reproduction_manifest.json')
        update('source_paused',resume_global_step=source_step,source_last_status=json.loads((old/'stage1/status.json').read_text()))
        env=os.environ.copy()
        env.update(PYTHONUNBUFFERED='1',NO_ALBUMENTATIONS_UPDATE='1',OMP_NUM_THREADS='4')
        command=[sys.executable,'-u','-m','examples.LiLaWAM.run_official_robotwin',
                 '--output',str(new),'--gpus','0,1,2,3,4,5,6,7','--num-processes','8']
        with (new/'supervisor.log').open('a') as log:
            child=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[2],env=env,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(new/'supervisor_process.json',{'pid':child.pid,'started':time.time(),'command':command})
        update('validating_target',target_supervisor_pid=child.pid)
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            if child.poll() is not None:
                raise RuntimeError(f'Target supervisor exited {child.returncode}')
            p=new/'stage1/status.json'
            if p.exists():
                current=json.loads(p.read_text())
                if current.get('status')=='failed':
                    raise RuntimeError(f'Target training failed: {current}')
                if current.get('world_size')==8 and current.get('global_step',0)>=source_step+60:
                    break
            time.sleep(3)
        else:
            raise TimeoutError('Target did not complete 60 updates')
        # Both old processes remain paused until the target has restored and trained.
        # Kill only their verified process tree; unrelated GPU processes are untouched.
        descendants=trainer.children(recursive=True)
        for p in descendants:
            try:p.kill()
            except psutil.NoSuchProcess:pass
        trainer.kill()
        supervisor.kill()
        psutil.wait_procs([trainer,supervisor,*descendants],timeout=10)
        paused=False
        update('completed',target_global_step=current['global_step'],world_size=8,
               global_batch_size=128,per_device_batch_size=16)
        write_json(old/'handoff_to_ddp8.json',journal)
        write_json(old/'run_status.json',{'status':'handed_off','target_run':str(new),
                   'resume_global_step':source_step,'updated':time.time()})
    except Exception as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
        if paused:
            trainer.resume()
            supervisor.resume()
        update('rolled_back_to_source',error=repr(exc))
        raise


if __name__=='__main__':
    main()
