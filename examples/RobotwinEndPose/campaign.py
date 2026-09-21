"""Durable coordinator: complete both smoke arms before formal training."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import psutil
from examples.RobotwinEndPose.prepare import ROOT, OUT, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--reuse-trained-smoke',action='store_true')
    args=parser.parse_args()
    path=OUT/'coordinator_status.json'
    if args.detach:
        assert not path.exists(), 'An existing campaign must be inspected before restarting'
        with (OUT/'coordinator.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-m','examples.RobotwinEndPose.campaign','--gpu',args.gpu]
                + (['--reuse-trained-smoke'] if args.reuse_trained_smoke else []),
                cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(OUT/'launch.json',dict(pid=proc.pid,birth=psutil.Process(proc.pid).create_time(),gpu=args.gpu))
        print(f'Feedback campaign coordinator PID {proc.pid}, GPU {args.gpu}')
        return
    child=None
    report=dict(state='starting',pid=os.getpid(),birth=psutil.Process().create_time(),gpu=args.gpu)
    def status(): save(path,dict(report,time=time.strftime('%Y-%m-%d %H:%M:%S')))
    def stop(sig,frame): raise RuntimeError(f'Signal {sig}')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    try:
        for phase in ('smoke','campaign'):
            report.update(state='running',phase=phase);status()
            child=subprocess.Popen([sys.executable,'-m','examples.RobotwinEndPose.run',
                '--gpu',args.gpu,'--phase',phase,'--allow-shared-gpu']
                + (['--reuse-trained-smoke'] if args.reuse_trained_smoke and phase=='smoke' else []),
                cwd=ROOT,start_new_session=True)
            report['child_pid']=child.pid;status()
            while child.poll() is None:
                status();time.sleep(10)
            assert child.returncode==0, f'{phase} exited {child.returncode}'
            assert json.loads((OUT/f'{phase}_status.json').read_text())['state']=='complete'
        report.update(state='complete')
    except BaseException as error:
        report.update(state='failed',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        status()


if __name__=='__main__':main()
