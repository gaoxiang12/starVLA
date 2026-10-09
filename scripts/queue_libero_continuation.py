"""Wait for the original GPUs, resume a prepared run, then evaluate milestones."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time


def available_gpus(rows):
    return [int(index) for index, memory, utilization in
            (row.split(',') for row in rows.splitlines() if row.strip())
            if int(memory) <= 200 and int(utilization) <= 5]


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def fetch_evaluation_checkpoint(run, root, host, controller, step):
    """Mirror a completed milestone from remote rank zero for local evaluation."""
    if host == controller:
        return
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
           '-o', f'UserKnownHostsFile={root}/.cache/multinode/known_hosts_{host}']
    state = run / 'checkpoints' / f'steps_{step}_training_state'
    result = subprocess.run(ssh + [host, 'cat ' + shlex.quote(str(state / 'complete.json'))],
                            check=True, capture_output=True, text=True, timeout=30)
    metadata = json.loads(result.stdout)
    if metadata['step'] != step:
        raise ValueError('Remote checkpoint step mismatch')
    checkpoint = run / 'checkpoints' / f'steps_{step}_pytorch_model.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix('.partial')
    subprocess.run(['rsync', '-a', '--no-owner', '--no-group', '-e', shlex.join(ssh),
                    f'{host}:{checkpoint}', str(temporary)], check=True, timeout=300)
    if temporary.stat().st_size == 0:
        raise ValueError('Remote checkpoint was empty')
    temporary.replace(checkpoint)
    state.mkdir(exist_ok=True)
    write_json(state / 'complete.json', metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    control = run / 'continuation'
    plan = json.loads((control / 'plan.json').read_text())
    root = Path(plan['root'])
    status_file = control / 'status.json'
    child = None
    stopped = False

    def interrupt(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def record(phase, **extra):
        write_json(status_file, dict(status=phase, updated_at=time.time(),
                                    pid=os.getpid(), plan=plan, **extra))

    def probe(host):
        command = ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                   '--format=csv,noheader,nounits']
        if host != plan['controller_host']:
            command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                       '-o', f'UserKnownHostsFile={root}/.cache/multinode/known_hosts_{host}',
                       host] + command
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
            return dict(host=host, available=available_gpus(result.stdout), raw=result.stdout)
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            return dict(host=host, available=[], error=str(exc))

    def wait_for_gpus(hosts, required, phase):
        while not stopped:
            with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
                observations = list(pool.map(probe, hosts))
            record(phase, gpu_observations=observations)
            if all(set(required[row['host']] if isinstance(required, dict) else required)
                   <= set(row['available']) for row in observations):
                return True
            for _ in range(60):
                if stopped:
                    return False
                time.sleep(1)
        return False

    def execute(command, log_path, phase):
        nonlocal child
        if stopped:
            raise InterruptedError('Queue stopped')
        with log_path.open('a') as log:
            child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record(phase, child_pid=child.pid, command=command)
        while child.poll() is None and not stopped:
            time.sleep(1)
        if stopped:
            raise InterruptedError('Queue stopped')
        if child.returncode:
            raise RuntimeError(f'Process exited {child.returncode}; see {log_path}')
        child = None

    with (control / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_file.exists():
            previous = json.loads(status_file.read_text())
            if previous['status'] not in ('prepared', 'waiting_gpus'):
                raise RuntimeError('Queue already launched; inspect state before retrying')
        try:
            if not wait_for_gpus(plan['hosts'], plan.get('gpu_map', plan['gpus']), 'waiting_gpus'):
                raise InterruptedError('Queue stopped')
            execute(plan['training_command'], control / 'training.log', 'training')
            done = json.loads((run / 'training_complete.json').read_text())
            assert done['completed_steps'] == plan['target_step']
            results = []
            for step in plan['evaluation_steps']:
                fetch_evaluation_checkpoint(run, root, plan.get('checkpoint_host', plan['controller_host']),
                                            plan['controller_host'], step)
                if not wait_for_gpus([plan['controller_host']], plan['evaluation_gpus'],
                                     f'waiting_eval_gpus_{step}'):
                    raise InterruptedError('Queue stopped')
                state = run / 'checkpoints' / f'steps_{step}_training_state'
                assert json.loads((state / 'complete.json').read_text())['step'] == step
                bundle = control / 'eval_model'
                (bundle / 'checkpoints').mkdir(parents=True, exist_ok=True)
                checkpoint = run / 'checkpoints' / f'steps_{step}_pytorch_model.pt'
                target = bundle / 'checkpoints' / checkpoint.name
                os.link(checkpoint, target)
                (bundle / 'config.yaml').write_bytes((run / 'config.full.yaml').read_bytes())
                (bundle / 'dataset_statistics.json').write_bytes((run / 'dataset_statistics.json').read_bytes())
                output = run / 'evaluations' / f'libero4_10ep_seed7_step{step}'
                # Bind-test ports immediately before starting the established evaluator.
                from eval_libero_after_training import choose_port_base
                command = [sys.executable, str(run / 'source_snapshot/scripts/run_libero_checkpoint_eval.py'),
                           '--checkpoint', str(target), '--output', str(output),
                           '--gpus', ','.join(map(str, plan['evaluation_gpus'])),
                           '--trials', '10', '--seed', '7',
                           '--port-base', str(choose_port_base(plan['evaluation_gpus']))]
                execute(command, control / f'eval_{step}.log', f'evaluating_{step}')
                summary = json.loads((output / 'summary.json').read_text())
                assert summary['status'] == 'complete' and summary['episodes'] == 400
                results.append(dict(step=step, successes=summary['successes'], summary=str(output / 'summary.json')))
                write_json(control / 'evaluation_results.json', results)
            record('complete', results=results)
        except InterruptedError:
            record('stopped')
        except BaseException as exc:
            record('failed', error=repr(exc))
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=180)


if __name__ == '__main__':
    main()
