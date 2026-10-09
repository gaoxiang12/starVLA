"""Evaluate the public official-HDF5 GAWM checkpoint on RoboTwin Clean or Randomized."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time

from examples.Robotwin.eval_files.run_lila_benchmark import parse_counters, stop
from examples.Robotwin.eval_files.summarize_robotwin_eval import ALL_TASKS

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'examples/Robotwin/eval_files'
CANCEL = threading.Event()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def run_task(task, gpu, port, args):
    out = args.output / task
    out.mkdir()
    row = dict(task=task, gpu=gpu, state='starting', started_at=time.time())
    save(out/'status.json', row)
    server = simulator = None
    vulkan = ROOT / '.cache/robotwin_evaluation/vulkan'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1',
               VK_ICD_FILENAMES=str(vulkan/'nvidia_icd.json'),
               LD_LIBRARY_PATH=str(vulkan/'root/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH',''),
               CUDA_HOME=str(ROOT/'.cache/robotwin_evaluation/cuda-12.8'),
               TORCH_CUDA_ARCH_LIST='12.0', MAX_JOBS='4',
               TORCH_EXTENSIONS_DIR=str(ROOT/'.cache/robotwin_evaluation/torch_extensions'),
               ROBOTWIN_PATH=str(args.robotwin), ROBOTWIN_PYTHON=str(ROOT/'.venv-robotwin/bin/python'),
               ROBOTWIN_POLICY_NAME='gawm_hdf5_interface', ROBOTWIN_TEST_NUM=str(args.episodes),
               ROBOTWIN_POLICY_IMAGE_CHANNEL_ORDER=args.image_channel_order,
               ROBOTWIN_EVAL_VIDEO_LOG='0', DEPLOY_POLICY_TEMPLATE_PATH=str(HERE/'deploy_policy_lila_wam.yml'))
    try:
        with socket.socket() as s: s.bind(('127.0.0.1',port))
        with (out/'server.log').open('w') as log:
            server = subprocess.Popen([str(ROOT/'.venv/bin/python'),str(HERE/'gawm_hdf5_server.py'),
                '--checkpoint',str(args.checkpoint),'--port',str(port),
                '--image-channel-order',args.image_channel_order],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline = time.monotonic()+180
        while True:
            if CANCEL.is_set(): raise RuntimeError('Evaluation cancelled')
            if server.poll() is not None: raise RuntimeError(f'Policy server exited {server.returncode}')
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=1): break
            except OSError:
                if time.monotonic()>deadline: raise TimeoutError('Policy server startup timed out')
                time.sleep(1)
        row.update(state='evaluating',server_pid=server.pid)
        save(out/'status.json',row)
        cmd=['bash',str(HERE/'eval.sh'),task,args.task_config,args.output.name,str(args.seed),str(gpu),str(args.checkpoint),str(port)]
        with (out/'eval.log').open('w') as log:
            simulator = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+args.task_timeout
            while simulator.poll() is None:
                if CANCEL.wait(1): raise RuntimeError('Evaluation cancelled')
                if time.monotonic()>deadline: raise TimeoutError('Task evaluation timed out')
            if simulator.returncode: raise RuntimeError(f'Simulator exited {simulator.returncode}')
        evaluation_log=(out/'eval.log').read_text(errors='replace')
        if 'OIDN Error:' in evaluation_log:
            raise RuntimeError('Camera denoiser failed; rollout results are invalid')
        episodes=parse_counters(evaluation_log,args.episodes)
        successes=sum(e['success'] for e in episodes)
        row.update(state='complete',successes=successes,trials=len(episodes),
                   success_rate=successes/len(episodes),episodes=episodes)
    except Exception as error:
        row.update(state='failed',error=repr(error))
    finally:
        stop(simulator);stop(server)
        row['elapsed_seconds']=time.time()-row['started_at']
        save(out/'status.json',row)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--robotwin',type=Path,default=ROOT/'thirdparty/RoboTwin')
    p.add_argument('--tasks',nargs='+',default=['all'])
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--gpus',nargs='+',default=['1','2','3','4','5','6','7'])
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--image-channel-order',choices=['rgb','bgr'],required=True)
    p.add_argument('--base-port',type=int,default=5894)
    p.add_argument('--task-timeout',type=int,default=14400)
    p.add_argument('--task-config',choices=['demo_clean','demo_randomized'],default='demo_clean')
    args=p.parse_args()
    tasks=list(ALL_TASKS) if args.tasks==['all'] else args.tasks
    if not tasks or len(set(tasks))!=len(tasks) or set(tasks)-set(ALL_TASKS) or args.episodes<1:
        p.error('Expected unique official tasks and positive episode count')
    if len(set(args.gpus))!=len(args.gpus):p.error('GPU slots must be unique')
    args.checkpoint=args.checkpoint.resolve();args.output=args.output.resolve();args.robotwin=args.robotwin.resolve()
    # Evaluation scenes draw wall/table textures from the unseen split.
    textures=args.robotwin/'assets/background_texture/unseen'
    if args.task_config=='demo_randomized' and not (textures.is_dir() and any(textures.iterdir())):
        p.error(f'Randomized evaluation needs background textures in {textures}')
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in usage.splitlines():
        index,memory,util=[int(x) for x in line.split(',')]
        if str(index) in args.gpus and (memory>200 or util>5):raise RuntimeError(f'GPU {index} is occupied')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:CANCEL.set())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'supervisor.pid').write_text(str(os.getpid()))
    sources=[args.checkpoint,args.checkpoint.parent.parent/'dataset_statistics.json',
             HERE/'gawm_hdf5_server.py',HERE/'gawm_hdf5_interface.py',HERE/'robotwin_eval_runner.py',
             args.robotwin/'script/eval_policy.py',args.robotwin/'task_config'/f'{args.task_config}.yml',Path(__file__),
             args.checkpoint.parent.parent/'config.full.yaml',ROOT/'.cache/robotwin_evaluation/environment.json']
    protocol=dict(checkpoint=str(args.checkpoint),tasks=tasks,task_config=args.task_config,episodes_per_task=args.episodes,
                  seed=args.seed,gpus=args.gpus,execute_horizon=16,action_chunk_size=32,smooth_actions=True,
                  task_success='official eval_success',expert_filter=True,
                  simulator_image_channel_order='rgb',policy_image_channel_order=args.image_channel_order,
                  swap_rb_before_normalization=args.image_channel_order == 'bgr',
                  robotwin_commit=subprocess.check_output(['git','-C',str(args.robotwin),'rev-parse','HEAD'],text=True).strip(),
                  source_sha256={str(f):sha256(f) for f in sources},started_at=time.time(),
                  benchmark_note=f'{args.episodes} episodes/task; official final reporting uses 100 episodes/task')
    protocol['environment']=json.loads((ROOT/'.cache/robotwin_evaluation/environment.json').read_text())
    save(args.output/'protocol.json',protocol)
    results=[]
    def summary():
        done=[r for r in results if r['state']=='complete'];trials=sum(r['trials'] for r in done);successes=sum(r['successes'] for r in done)
        return dict(state='running',completed_tasks=len(done),total_tasks=len(tasks),trials=trials,successes=successes,
                    observed_success_rate=successes/trials if trials else None,
                    failed_tasks=[r['task'] for r in results if r['state']=='failed'],results=results)
    save(args.output/'summary.json',summary())
    # One sequential task stream per GPU, so simulation and its policy share a
    # device without competing with another evaluation task.
    lock=threading.Lock()
    def worker(slot,gpu):
        for task in tasks[slot::len(args.gpus)]:
            if CANCEL.is_set():break
            row=run_task(task,gpu,args.base_port+slot,args)
            with lock:
                results.append(row);save(args.output/'summary.json',summary())
            print(task,row['state'],row.get('success_rate',row.get('error')),flush=True)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        for future in as_completed([pool.submit(worker,i,gpu) for i,gpu in enumerate(args.gpus)]):future.result()
    result=summary();result['state']='complete' if result['completed_tasks']==len(tasks) else 'incomplete'
    result['sources_unchanged']=all(sha256(f)==h for f,h in protocol['source_sha256'].items())
    result['finished_at']=time.time();save(args.output/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
    if result['state']!='complete' or not result['sources_unchanged']:raise SystemExit(1)


if __name__=='__main__':main()
