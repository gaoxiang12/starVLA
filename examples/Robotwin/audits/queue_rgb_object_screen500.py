"""Screen the fixed 500-step object policy when an allowed evaluation GPU has room."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from examples.Robotwin.audits.run_rgb_contact_refinement import digest, save

ROOT = Path(__file__).resolve().parents[3]
CP = ROOT / 'playground/Checkpoints'
RUN = CP / 'gawm_rgb_object_readout_2k_20260908'
CAMP = CP / 'gawm_rgb_object_screen500_queue_20260908'
WEIGHTS = RUN / 'checkpoints/steps_500_pytorch_model.pt'
OUTPUT = RUN / 'screen10_step500'
PYTHON = ROOT.parent / '.venvs/starVLA/bin/python'


def main():
    CAMP.mkdir(exist_ok=False)
    report = json.loads((CP / 'gawm_rgb_object_localization_20260908/checkpoints/step500.json').read_text())
    assert report['state'] == 'complete'
    checksum = digest(WEIGHTS)
    assert checksum == report['checkpoint_sha256']
    assert not OUTPUT.exists()
    save(CAMP / 'manifest.json', dict(supervisor_pid=os.getpid(), checkpoint=str(WEIGHTS),
        checkpoint_sha256=checksum, allowed_gpus=['4', '6'], required_free_mib=12*1024,
        eval_episodes=10, seed=0, execute_horizon=16, channel_order='rgb', spatial_ablation='full',
        output=str(OUTPUT), port=6694,
        note='Early exploratory screen of the fixed 500-step weights. Does not replace the '
             'scheduled matched 2000-step screen, select a best checkpoint, or change training. '
             'No processes are stopped to obtain memory. GPUs 1/2/3/7 are excluded.'))
    active, selected, previous_candidate = None, None, None

    def status(state, **extra):
        save(CAMP / 'status.json', dict(state=state, supervisor_pid=os.getpid(), gpu=selected,
            pid=active.pid if active and active.poll() is None else None,
            time=time.strftime('%Y-%m-%d %H:%M:%S'), **extra))

    def stop(signum, frame):
        raise RuntimeError(f'Signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        while selected is None:
            free = {gpu: int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free',
                         '--format=csv,noheader,nounits'], text=True).strip()) for gpu in ('4', '6')}
            candidate = next((gpu for gpu in ('4', '6') if free[gpu] >= 12*1024), None)
            # Require two consecutive observations to avoid launching into a brief gap.
            if candidate is not None and candidate == previous_candidate:
                selected = candidate
                break
            previous_candidate = candidate
            status('waiting_for_gpu_memory', free_mib=free)
            time.sleep(15)
        assert digest(WEIGHTS) == checksum
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 6694))
        command = [str(PYTHON), str(ROOT / 'examples/Robotwin/audits/run_rgb_color_diagnostic.py'),
            '--gpu', selected, '--port', '6694', '--episodes', '10', '--orders', 'rgb',
            '--execute-horizon', '16', '--spatial-ablation', 'full',
            '--checkpoint', str(WEIGHTS), '--output', str(OUTPUT)]
        env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', PYTHONNOUSERSITE='1',
                   NO_ALBUMENTATIONS_UPDATE='1', WANDB_MODE='disabled')
        save(CAMP / 'command.json', dict(command=command, free_mib_at_admission=free))
        with (CAMP / 'eval.log').open('x') as log:
            active = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while active.poll() is None:
            status('evaluating')
            time.sleep(15)
        if active.returncode:
            raise RuntimeError(f'500-step screening exited {active.returncode}')
        assert json.loads((OUTPUT / 'status.json').read_text())['state'] == 'complete'
        rows = [json.loads(line) for line in (OUTPUT / 'rgb/ranking_episode_metrics.jsonl').read_text().splitlines()]
        assert len(rows) == 10 and {r['trial'] for r in rows} == set(range(10))
        status('complete', episodes=10, successes=sum(r['success'] for r in rows))
    except BaseException as exc:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
