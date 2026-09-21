"""Convert successful recovery clips only after their owned collection queue completes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

from examples.Robotwin.audits.run_rgb_contact_refinement import save

ROOT = Path(__file__).resolve().parents[3]
CP = ROOT / 'playground/Checkpoints'
QUEUE = CP / 'gawm_rgb_recovery_train20_queue_20260908'
COLLECTION = CP / 'gawm_rgb_recovery_train20_20260908'
OUTPUT = CP / 'gawm_rgb_recovery_train20_conversion_20260908'
DATASET = Path('/data/gaoxiang/RoboTwinRecoveryTrain20_20260908/Clean/blocks_ranking_rgb')


def main():
    OUTPUT.mkdir(exist_ok=False)
    queued = json.loads((QUEUE / 'status.json').read_text())
    process = psutil.Process(queued['pid'])
    assert process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    save(OUTPUT / 'manifest.json', dict(pid=os.getpid(), queue_pid=process.pid,
        queue_create_time=process.create_time(), collection=str(COLLECTION), dataset=str(DATASET),
        note='CPU conversion of verified successful expert clips only. No automatic model training.'))
    active = None

    def status(state, **extra):
        save(OUTPUT / 'status.json', dict(state=state, pid=os.getpid(), child_pid=active.pid if active else None,
             time=time.strftime('%Y-%m-%d %H:%M:%S'), **extra))

    def stop(signum, frame):
        raise RuntimeError(f'Signal {signum}')

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        while True:
            queued = json.loads((QUEUE / 'status.json').read_text())
            if queued['state'] == 'complete':
                if not queued['accepted_raw_clips']:
                    status('complete_without_data', accepted_raw_clips=0)
                    return
                break
            if queued['state'] in ('failed', 'cleanup_pending'):
                raise RuntimeError(f'Collection queue failed: {queued.get("error")}')
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                raise RuntimeError('Collection queue terminated without a complete result')
            status('waiting_for_collection', queue_state=queued['state'])
            time.sleep(15)
        cmd = [str(ROOT.parent / '.venvs/starVLA/bin/python'), '-u', '-m',
            'examples.Robotwin.audits.convert_rgb_recovery_campaign', '--campaign', str(COLLECTION),
            '--plan-output', str(OUTPUT / 'converted_dataset.json'), '--output', str(DATASET),
            '--labels', str(OUTPUT / 'labels'), '--raw-index', str(DATASET.parent / 'recovery_raw_index')]
        with (OUTPUT / 'conversion.log').open('x') as log:
            active = subprocess.Popen(cmd, cwd=ROOT,
                env=dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4'),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while active.poll() is None:
            status('converting')
            time.sleep(15)
        assert active.returncode == 0, f'Converter exited {active.returncode}'
        converted = json.loads((OUTPUT / 'converted_dataset.json').read_text())
        assert converted['state'] == 'converted_labels_ready'
        status('complete', episodes=len(converted['episodes']), frames=converted['frames'],
               loader_audit_complete=False, training_enabled=False)
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
