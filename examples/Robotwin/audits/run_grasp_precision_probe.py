"""Single development-scene diagnostic; distinct from the fixed final comparison."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import psutil
from websockets.sync.client import connect

from examples.Robotwin.audits.run_grasp_lift_development import ROOT,PYTHON,SIM_PYTHON,digest,save
from examples.Robotwin.audits.measure_grasp_precision import measure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--gpu',choices=['0','4','5','6'],required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protocol_path=ROOT/'examples/Robotwin/audits/grasp_lift_protocol_20260909.json'
    protocol=json.loads(protocol_path.read_text())
    record=next(r for r in protocol['records'] if r['seed']==args.seed and r['split']=='development')
    checkpoint=args.checkpoint.resolve()
    assert checkpoint.is_file()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),protocol_path,ROOT/'examples/Robotwin/audits/record_pregrasp_precision.py',
        ROOT/'examples/Robotwin/audits/run_grasp_lift_case.py',ROOT/'examples/Robotwin/audits/grasp_lift_scoring.py',
        ROOT/'starVLA/model/modules/action_model/CartesianResidualACT.py',ROOT/'starVLA/model/framework/WM4A/GAWMCartesian.py',
        ROOT/'deployment/model_server/policy_norm_processor.py']
    manifest=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),scene=record,gpu=args.gpu,
        port=args.port,mode='full',execute_horizon=16,precision='float32',
        source_sha256={str(p.resolve()):digest(p) for p in sources},
        note='Post-hoc selected development failure; early checkpoint diagnostic only. No full-comparison or final-test success-rate claim.')
    save(out/'manifest.json',manifest)
    state=dict(state='starting',supervisor_pid=os.getpid(),supervisor_birth=psutil.Process().create_time())
    server=case=None

    def status():
        save(out/'status.json',dict(state,time=time.strftime('%Y-%m-%d %H:%M:%S'),
            server_pid=server.pid if server is not None and server.poll() is None else None,
            case_pid=case.pid if case is not None and case.poll() is None else None))

    def stop(sig,frame):
        raise RuntimeError(f'Signal {sig}')

    def cleanup(process):
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)

    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4',NO_ALBUMENTATIONS_UPDATE='1',PYTHONPATH=f'{ROOT}:{ROOT}/examples/Robotwin/eval_files')
    try:
        with socket.socket() as sock:sock.bind(('127.0.0.1',args.port))
        with (out/'server.log').open('x') as log:
            server=subprocess.Popen([str(PYTHON),str(ROOT/'deployment/model_server/server_policy.py'),
                '--ckpt_path',str(checkpoint),'--port',str(args.port),'--idle_timeout','-1'],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        deadline=time.monotonic()+180
        while True:
            if server.poll() is not None:raise RuntimeError(f'Server exited {server.returncode}')
            try:
                with connect(f'ws://127.0.0.1:{args.port}',open_timeout=1) as websocket:
                    websocket.recv(timeout=2)
                break
            except (OSError,TimeoutError):
                if time.monotonic()>deadline:raise RuntimeError('Server startup timeout')
                status();time.sleep(2)
        with (out/'case.log').open('x') as log:
            case=subprocess.Popen([str(SIM_PYTHON),str(ROOT/'examples/Robotwin/audits/record_pregrasp_precision.py'),
                '--checkpoint',str(checkpoint),'--seed',str(args.seed),'--mode','full','--port',str(args.port),
                '--output',str(out/'case'),'--execute-horizon','16'],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        state['state']='running'
        while case.poll() is None:status();time.sleep(5)
        if case.returncode:raise RuntimeError(f'Case exited {case.returncode}')
        reference=ROOT/f"playground/Checkpoints/gawm_grasp_lift_development_20260909/rgb_v2_5k_full/scene_{record['scene_id']:03d}"
        precision=measure(out/'case',reference)
        save(out/'case/precision.json',precision)
        state.update(state='complete',result=precision['scored_result'],precision=str(out/'case/precision.json'))
    except BaseException as error:
        state.update(state='failed',error=repr(error));raise
    finally:
        cleanup(case);cleanup(server);status()


if __name__=='__main__':main()
