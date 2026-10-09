"""Run a prepared LIBERO fine-tune only after a named RoboTwin run succeeds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import itertools
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from omegaconf import OmegaConf
from queue_libero_continuation import available_gpus, fetch_evaluation_checkpoint, write_json
from eval_libero_after_training import choose_port_base


def dependency_complete(run, expected_steps):
    status_path = run / 'cluster_status.json'
    if not status_path.exists():
        return False
    state = json.loads(status_path.read_text())
    if state.get('status') in ('failed', 'stopped'):
        raise RuntimeError(f'RoboTwin ended with {state["status"]}; do not treat failure as completion')
    if state.get('status') != 'complete':
        return False
    done = json.loads((run / 'training_complete.json').read_text())
    saved = run / 'checkpoints' / f'steps_{expected_steps}_training_state'
    marker = json.loads((saved / 'complete.json').read_text())
    if done.get('completed_steps') != expected_steps or marker.get('step') != expected_steps:
        raise ValueError('RoboTwin final checkpoint step mismatch')
    world = state['world_size']
    if marker.get('world_size') != world or len(list(saved.glob('random_states_*.pkl'))) != world:
        raise ValueError('RoboTwin final training state is incomplete')
    return True


def select_layout(observations, controller):
    # Prefer forty GPUs, but do not wait for a particular occupied card.
    pools = {r['host']: sorted(g for g in r['available'] if 0 <= g < 8) for r in observations}
    hosts = list(pools)
    choices = [[n for n in (8, 4, 2, 0) if n <= len(pools[h])] for h in hosts]
    for target in (40, 32, 20):
        layouts = [counts for counts in itertools.product(*choices) if sum(counts) == target]
        if not layouts:
            continue
        # Prefer the local controller as rank zero when available.
        counts = max(layouts, key=lambda c: (c[hosts.index(controller)] if controller in hosts else 0, -sum(n > 0 for n in c)))
        selected = sorted([(h, n) for h, n in zip(hosts, counts) if n], key=lambda x: (-x[1], x[0] != controller, x[0]))
        return {h: ','.join(map(str, pools[h][:n])) for h, n in selected}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args()
    queue = args.queue.resolve()
    plan = json.loads((queue / 'plan.json').read_text())
    root, run = Path(plan['root']), Path(plan['run'])
    stopped = False
    child = None
    def interrupt(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    def record(status, **kwargs):
        write_json(queue / 'status.json', {'status': status, 'pid': os.getpid(), 'updated_at': time.time(), 'run': str(run), **kwargs})
    def pause(seconds=30):
        for _ in range(seconds):
            if stopped:
                raise InterruptedError('Queue stopped')
            time.sleep(1)
    def probe(host):
        command = ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits']
        if host != plan['controller_host']:
            command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host] + command
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
            return {'host': host, 'available': available_gpus(result.stdout)}
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            return {'host': host, 'available': [], 'error': type(exc).__name__}
    def execute(command, log_name, status, cwd=root):
        nonlocal child
        if stopped:
            raise InterruptedError('Queue stopped')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(queue / 'source_snapshot')
        with (queue / log_name).open('a') as log:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record(status, command=command, child_pid=child.pid)
        while child.poll() is None:
            pause(2)
        if child.returncode:
            raise RuntimeError(f'{status} failed with exit {child.returncode}; see {log_name}')
        child = None
    with (queue / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = queue / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text())['status'] not in ('prepared', 'waiting_robotwin', 'waiting_gpus'):
            raise RuntimeError('Queue already launched; inspect its state before restarting')
        if run.exists():
            raise FileExistsError('Target run already exists; refusing duplicate launch')
        try:
            while not dependency_complete(Path(plan['dependency_run']), plan['dependency_target_step']):
                record('waiting_robotwin', dependency_run=plan['dependency_run'], dependency_target_step=plan['dependency_target_step'])
                pause()
            while True:
                with ThreadPoolExecutor(max_workers=len(plan['hosts'])) as pool:
                    observations = list(pool.map(probe, plan['hosts']))
                layout = select_layout(observations, plan['controller_host'])
                record('waiting_gpus', observations=observations, proposed_layout=layout)
                if layout:
                    break
                pause()
            world_size = sum(len(g.split(',')) for g in layout.values())
            cfg = OmegaConf.load(queue / 'training_config.yaml')
            cfg.training_overrides.datasets.vla_data.per_device_batch_size = 160 // world_size
            cfg.datasets.vla_data.per_device_batch_size = 160 // world_size
            launch_config = queue / 'launch_config.yaml'
            OmegaConf.save(cfg, launch_config)
            write_json(queue / 'launch_plan.json', {'gpu_map': layout, 'world_size': world_size, 'global_batch_size': 160, 'per_device_batch_size': 160 // world_size})
            launcher = queue / 'source_snapshot/scripts/run_gawm_c_cluster.py'
            command = [sys.executable, str(launcher), '--config', str(launch_config), '--run-id', run.name, '--hosts', ','.join(layout), '--gpu-map', json.dumps(layout), '--controller-host', plan['controller_host'], '--source-snapshot', str(queue / 'source_snapshot')]
            execute(command, 'training.log', 'training')
            done = json.loads((run / 'training_complete.json').read_text())
            if done.get('completed_steps') != plan['target_step']:
                raise ValueError('LIBERO did not finish the requested fine-tuning steps')
            rank_zero = next(iter(layout))
            for step in plan['evaluation_steps']:
                fetch_evaluation_checkpoint(run, root, rank_zero, plan['controller_host'], step)
            # The frozen evaluator needs the environment and simulator assets at
            # its repository root. Add links only after the source was provisioned.
            snapshot = run / 'source_snapshot'
            for rel in ('.venv', 'playground/LIBERO', '.cache/libero_eval/egl'):
                link = snapshot / rel
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(root / rel, target_is_directory=True)
            results = []
            for step in plan['evaluation_steps']:
                while True:
                    free = probe(plan['controller_host'])['available']
                    if free:
                        break
                    record('waiting_eval_gpus', step=step)
                    pause()
                bundle = queue / f'eval_model_{step}'
                (bundle / 'checkpoints').mkdir(parents=True, exist_ok=True)
                checkpoint = run / 'checkpoints' / f'steps_{step}_pytorch_model.pt'
                target = bundle / 'checkpoints' / checkpoint.name
                os.link(checkpoint, target)
                (bundle / 'config.yaml').write_bytes((run / 'config.full.yaml').read_bytes())
                (bundle / 'dataset_statistics.json').write_bytes((run / 'dataset_statistics.json').read_bytes())
                output = run / 'evaluations' / f'libero4_10ep_seed7_finetune{step}'
                command = [sys.executable, str(snapshot / 'scripts/run_libero_checkpoint_eval.py'), '--checkpoint', str(target), '--output', str(output), '--gpus', ','.join(map(str, free)), '--trials', '10', '--seed', '7', '--port-base', str(choose_port_base(free))]
                execute(command, f'eval_{step}.log', f'evaluating_{step}', cwd=snapshot)
                result = json.loads((output / 'summary.json').read_text())
                if result.get('status') != 'complete' or result.get('episodes') != 400:
                    raise ValueError('Evaluation is incomplete')
                results.append({'step': step, 'successes': result['successes'], 'episodes': 400, 'summary': str(output / 'summary.json')})
                write_json(queue / 'evaluation_results.json', results)
            record('complete', evaluations=results)
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
