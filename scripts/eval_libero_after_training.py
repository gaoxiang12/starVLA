"""Wait for a successful C training run, then evaluate its final LIBERO policy."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from filelock import FileLock
from omegaconf import OmegaConf


def write_json(path, record):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(path)


def training_checkpoint(run, expected_step):
    """Return None while training runs; never evaluate failed or partial saves."""
    status_file = run / 'cluster_status.json'
    if not status_file.exists():
        return None
    cluster = json.loads(status_file.read_text())
    if cluster['status'] in ('failed', 'stopped'):
        raise RuntimeError(f"Training ended with status {cluster['status']}")
    if cluster['status'] != 'complete':
        return None
    done = json.loads((run / 'training_complete.json').read_text())
    state = run / 'checkpoints' / f'steps_{expected_step}_training_state'
    marker = json.loads((state / 'complete.json').read_text())
    if done.get('completed_steps') != expected_step or marker.get('step') != expected_step:
        raise RuntimeError('Final training step does not match evaluation plan')
    world_size = int(cluster['world_size'])
    if marker.get('world_size') != world_size:
        raise RuntimeError('Checkpoint world size mismatch')
    checkpoint = run / 'checkpoints' / f'steps_{expected_step}_pytorch_model.pt'
    required = [checkpoint, state / 'optimizer.bin', state / 'scheduler.bin',
                run / 'config.full.yaml', run / 'dataset_statistics.json']
    required += [state / f'random_states_{rank}.pkl' for rank in range(world_size)]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'Incomplete final checkpoint: {path}')
    return checkpoint


def free_gpus(candidates):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                      '--format=csv,noheader,nounits'], text=True)
    used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in output.splitlines()}
    return [gpu for gpu in candidates if used.get(gpu, 10**9) <= 512]


def choose_port_base(gpus):
    for base in range(21000, 26000, 16):
        sockets = []
        try:
            for gpu in gpus:
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(('0.0.0.0', base + gpu))
            return base
        except OSError:
            pass
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError('No free evaluation server ports')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    control = run / 'auto_eval'
    plan = json.loads((control / 'plan.json').read_text())
    child = None
    stopped = False

    def interrupt(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    # A second watcher cannot launch a duplicate evaluation for this run.
    with FileLock(str(control / 'watcher.lock'), timeout=0):
        status_path = control / 'status.json'
        if status_path.exists():
            previous = json.loads(status_path.read_text())
            if previous['status'] not in ('waiting_training', 'waiting_gpus'):
                raise RuntimeError('Evaluation already started or ended; inspect its status before retrying')
        (control / 'watcher.pid').write_text(str(os.getpid()))
        status = dict(status='waiting_training', watcher_pid=os.getpid(), plan=plan)

        def record(**updates):
            status.update(updates, updated_at=time.time())
            write_json(status_path, status)

        try:
            record()
            while not stopped:
                checkpoint = training_checkpoint(run, plan['checkpoint_step'])
                if checkpoint is None:
                    time.sleep(15)
                    continue
                record(status='waiting_gpus')
                gpus = free_gpus(plan['candidate_gpus'])
                if not gpus:
                    time.sleep(15)
                    continue
                # Stable metadata for evaluation; do not overwrite the training
                # configuration or depend on later edits to the workspace.
                bundle = control / 'model'
                (bundle / 'checkpoints').mkdir(parents=True, exist_ok=True)
                target = bundle / 'checkpoints' / checkpoint.name
                if not target.exists():
                    os.link(checkpoint, target)
                (bundle / 'config.yaml').write_bytes((run / 'config.full.yaml').read_bytes())
                (bundle / 'dataset_statistics.json').write_bytes((run / 'dataset_statistics.json').read_bytes())
                command = [sys.executable, str(control / 'source_snapshot/scripts/run_libero_checkpoint_eval.py'),
                           '--checkpoint', str(target), '--output', plan['output_dir'],
                           '--gpus', ','.join(map(str, gpus)), '--trials', str(plan['trials_per_task']),
                           '--seed', str(plan['seed']), '--port-base', str(choose_port_base(gpus))]
                record(status='launching', gpus=gpus, command=command)
                with (control / 'evaluation.log').open('w') as log:
                    child = subprocess.Popen(command, cwd=control / 'source_snapshot',
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                record(status='evaluating', evaluation_pid=child.pid)
                while child.poll() is None and not stopped:
                    time.sleep(5)
                if stopped:
                    break
                if child.returncode:
                    raise RuntimeError(f'Evaluation exited {child.returncode}; see {control / "evaluation.log"}')
                summary = json.loads((Path(plan['output_dir']) / 'summary.json').read_text())
                if summary['status'] != 'complete' or summary['episodes'] != plan['total_episodes']:
                    raise RuntimeError('Evaluation did not complete all requested episodes')
                record(status='complete', summary=str(Path(plan['output_dir']) / 'summary.json'),
                       episodes=summary['episodes'], successes=summary['successes'],
                       success_rate=summary['success_rate'], finished_at=time.time())
                return
            record(status='stopped', finished_at=time.time())
        except BaseException as error:
            record(status='failed', error=repr(error), finished_at=time.time())
            raise
        finally:
            if child is not None and child.poll() is None:
                # The evaluator handles SIGTERM and cleans up its own servers.
                child.terminate()
                child.wait(timeout=180)


if __name__ == '__main__':
    main()
