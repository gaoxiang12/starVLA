"""Queue recovery collection behind the oracle diagnostic and final matched validation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

from examples.Robotwin.audits.run_rgb_contact_refinement import digest, save

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = ROOT / 'playground/Checkpoints'
OUTPUT = CHECKPOINTS / 'gawm_rgb_recovery_train20_queue_20260908'
COLLECTION = CHECKPOINTS / 'gawm_rgb_recovery_train20_20260908'
SOURCES = ROOT / 'examples/Robotwin/audits/rgb_recovery_train20_sources_20260908.json'


def main():
    assert not COLLECTION.exists()
    OUTPUT.mkdir(exist_ok=False)
    gates = [('oracle', CHECKPOINTS / 'gawm_rgb_expert_action_replay_20260908/status.json', 'pid'),
             ('final_validation', CHECKPOINTS / 'gawm_rgb_matched_validation_2000_20260908/status.json', 'supervisor_pid')]
    handles = {}
    for name, path, pid_key in gates:
        data = json.loads(path.read_text())
        process = psutil.Process(data[pid_key])
        assert process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        handles[name] = process
    manifest = dict(pid=os.getpid(), sources=str(SOURCES), source_sha256=digest(SOURCES),
                    collection=str(COLLECTION), gpu='0',
                    dependencies={name: dict(pid=p.pid, create_time=p.create_time()) for name, p in handles.items()},
                    note='Require all three oracle scenes successful and final matched validation complete. Then collect 20 training-source prefixes, accepting only successful expert continuations. No automatic training.')
    save(OUTPUT / 'manifest.json', manifest)
    active = None

    def status(state, **extra):
        save(OUTPUT / 'status.json', dict(state=state, pid=os.getpid(),
             time=time.strftime('%Y-%m-%d %H:%M:%S'), child_pid=active.pid if active else None, **extra))

    def stop(signum, frame):
        raise RuntimeError(f'Signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        pending = list(gates)
        while pending:
            for gate in list(pending):
                name, path, _ = gate
                data = json.loads(path.read_text())
                if data['state'] == 'complete':
                    if name == 'oracle':
                        if len(data['cases']) != 3 or data['oracle_successes'] != 3:
                            raise RuntimeError('Oracle replay was not 3/3; inspect execution before expanding collection')
                    else:
                        assert not data.get('failures')
                    pending.remove(gate)
                elif data['state'] == 'failed':
                    raise RuntimeError(f'{name} failed: {data.get("error", data.get("failures"))}')
                elif not handles[name].is_running() or handles[name].status() == psutil.STATUS_ZOMBIE:
                    raise RuntimeError(f'{name} process terminated without a complete result')
            status('waiting_for_dependencies', pending=[g[0] for g in pending])
            if pending:
                time.sleep(15)
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
                                                '--format=csv,noheader,nounits'], text=True).strip())
            if free >= 12 * 1024:
                break
            status('waiting_for_gpu_memory', free_mib=free, required_mib=12*1024)
            time.sleep(15)
        assert digest(SOURCES) == manifest['source_sha256'] and not COLLECTION.exists()
        cmd = [str(ROOT.parent / '.venvs/starVLA/bin/python'), '-u', '-m',
               'examples.Robotwin.audits.run_rgb_recovery_pilot', '--sources', str(SOURCES),
               '--output', str(COLLECTION), '--raw-root',
               '/data/gaoxiang/RoboTwinGenerated_raw/RecoveryTrain20_20260908/blocks_ranking_rgb',
               '--gpu', '0', '--port', '6670', '--allow-tabletop-handoff']
        with (OUTPUT / 'collection.log').open('x') as log:
            active = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)),
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        while active.poll() is None:
            status('collecting')
            time.sleep(15)
        assert active.returncode == 0, f'Collection supervisor exited {active.returncode}'
        result = json.loads((COLLECTION / 'status.json').read_text())
        assert result['state'] == 'complete'
        status('complete', accepted_raw_clips=result['accepted_raw_clips'])
    except BaseException as exc:
        if active is not None and active.poll() is None:
            # The collector supervisor owns its server/collector process groups.
            active.terminate()
            try:
                active.wait(timeout=60)
            except subprocess.TimeoutExpired:
                status('cleanup_pending', error=repr(exc))
                raise
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
