"""Durable four-suite evaluation using the shared policy server and LIBERO client."""
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

ROOT = Path(__file__).resolve().parents[2]
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def progress(path, trials):
    content = path.read_text(errors='replace') if path.exists() else ''
    counts = re.findall(r'# episodes completed so far: (\d+).*?# successes: (\d+)', content, re.S)
    episodes, successes = map(int, counts[-1]) if counts else (0, 0)
    task_rates = re.findall(r'Current task success rate: ([\d.]+)', content)
    return dict(episodes=episodes, successes=successes,
                success_rate=successes / episodes if episodes else None,
                per_task=[dict(task_id=i, episodes=trials, successes=round(float(rate)*trials))
                          for i, rate in enumerate(task_rates)],
                finished=f'Total episodes: {10*trials}' in content)


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--base-port', type=int, default=29910)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--execute-horizon', type=int, default=8)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file() or not 1 <= args.trials <= 50:
        raise ValueError('Expected existing checkpoint and 1..50 initial states per task')
    output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (output / 'supervisor.log').open('x') as stream:
            child = subprocess.Popen([sys.executable, '-m', 'examples.LiLaWAM.run_libero_eval',
                *[a for a in sys.argv[1:] if a != '--detach']], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(output / 'launch.json', dict(pid=child.pid, birth=psutil.Process(child.pid).create_time()))
        print(f'Evaluation supervisor PID {child.pid}; status: {output / "run_status.json"}')
        return
    gpus = args.gpus.split(',')
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError('Select four distinct GPUs')
    report = dict(state='starting', checkpoint=str(checkpoint), seed=args.seed,
                  trials_per_task=args.trials, expected_episodes=40*args.trials,
                  execute_horizon=args.execute_horizon, pid=os.getpid(), suites={})
    def status():
        report['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        write_json(output / 'run_status.json', report)
    def interrupted(sig, frame):
        raise RuntimeError(f'Signal {sig}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    jobs, streams = [], []
    status()
    try:
        for i, gpu in enumerate(gpus):
            free = int(subprocess.check_output(['nvidia-smi', '-i', gpu,
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            if free < 12000:
                raise RuntimeError(f'GPU {gpu} has insufficient free memory: {free} MiB')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', args.base_port+i))
        files = [checkpoint, checkpoint.parent / 'config.yaml',
                 ROOT / 'starVLA/model/framework/WM4A/LiLaWAMTrain.py',
                 ROOT / 'starVLA/model/modules/lila/core.py',
                 ROOT / 'starVLA/model/modules/lila/primitives.py',
                 ROOT / 'examples/LIBERO/eval_files/eval_libero.py',
                 ROOT / 'examples/LIBERO/eval_files/model2libero_interface.py',
                 ROOT / 'deployment/model_server/policy_wrapper.py',
                 ROOT / 'deployment/model_server/policy_norm_processor.py',
                 ROOT / 'examples/LiLaWAM/run_libero_eval.py']
        hashes = {}
        for file in files:
            if not file.exists():
                continue
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''):
                    digest.update(block)
            hashes[str(file)] = digest.hexdigest()
        write_json(output / 'manifest.json', dict(arguments=vars(args), sha256=hashes))
        for i, (suite, gpu) in enumerate(zip(SUITES, gpus)):
            directory = output / suite
            directory.mkdir(exist_ok=False)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MUJOCO_GL='egl',
                       PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=gpu,
                       PYTHONPATH=str(ROOT), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                       HF_HUB_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1',
                       TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                       TRITON_CACHE_DIR=str(directory / 'triton_cache'))
            env.pop('DEBUG', None)
            server_command = [sys.executable, 'deployment/model_server/server_policy.py',
                '--ckpt_path', str(checkpoint), '--port', str(args.base_port+i), '--use_bf16',
                '--policy-seed', str(args.seed), '--policy-seed-log', str(directory/'policy_queries.jsonl')]
            client_command = [sys.executable, 'examples/LIBERO/eval_files/eval_libero.py',
                '--args.pretrained-path', str(checkpoint), '--args.host', '127.0.0.1',
                '--args.port', str(args.base_port+i), '--args.task-suite-name', suite,
                '--args.num-trials-per-task', str(args.trials), '--args.seed', str(args.seed),
                '--args.execute-horizon', str(args.execute_horizon), '--args.unnorm-key', 'franka',
                '--args.video-out-path', str(directory/'videos'), '--args.job-name', output.name]
            pair = []
            for name, command in [('server', server_command), ('eval', client_command)]:
                stream = (directory/f'{name}.log').open('x')
                streams.append(stream)
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT)
                pair.append(process)
                jobs.append(process)
            report['suites'][suite] = dict(state='running', gpu=gpu, port=args.base_port+i,
                server_pid=pair[0].pid, client_pid=pair[1].pid, commands=[server_command, client_command])
            status()
        report['state'] = 'evaluating'
        deadline = time.monotonic() + 12*3600
        while time.monotonic() < deadline:
            active = False
            for i, suite in enumerate(SUITES):
                server, client = jobs[2*i:2*i+2]
                row = report['suites'][suite]
                row.update(progress(output/suite/'eval.log', args.trials))
                if row['state'] != 'running':
                    continue
                if client.poll() is not None:
                    row['state'] = 'complete' if client.returncode == 0 and row['finished'] else 'failed'
                    row['exit_code'] = client.returncode
                    stop(server)
                elif server.poll() is not None:
                    row.update(state='failed', error=f'Server exited {server.returncode}')
                    stop(client)
                else:
                    active = True
            report['episodes'] = sum(r['episodes'] for r in report['suites'].values())
            report['successes'] = sum(r['successes'] for r in report['suites'].values())
            report['success_rate'] = report['successes']/report['episodes'] if report['episodes'] else None
            status()
            if not active:
                report['state'] = 'complete' if all(r['state']=='complete' for r in report['suites'].values()) else 'failed'
                status()
                write_json(output/'results.json', report)
                return
            time.sleep(10)
        raise TimeoutError('Evaluation exceeded 12 hours')
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        status()
        raise
    finally:
        for process in jobs:
            stop(process)
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()
