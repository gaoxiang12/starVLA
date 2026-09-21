"""Run the accepted flow/regression training pair sequentially on GPU 0."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

from examples.Robotwin.audits.run_grasp_precision_training_smoke import ROOT, OUT, digest, save


def main():
    status_path = OUT/'compact_training_pair_status.json'
    assert not status_path.exists()
    acceptance = json.loads((OUT/'compact_trained_smoke_audit.json').read_text())
    assert acceptance['state'] == 'compact_real_trainer_and_deployment_verified'
    files = [ROOT/'starVLA/model/framework/WM4A/GAWM.py',
             ROOT/'starVLA/model/framework/WM4A/GAWMCompactExpert.py',
             ROOT/'starVLA/model/modules/action_model/CompactFlowActionHead.py',
             ROOT/'starVLA/training/train_starvla.py',
             ROOT/'examples/Robotwin/audits/run_grasp_precision_training_smoke.py']
    sources = {str(p): digest(p) for p in files}
    report = dict(state='starting', supervisor_pid=os.getpid(), supervisor_birth=psutil.Process().create_time(),
                  gpu='0', modes=['flow','regression'], completed=[], source_sha256=sources)
    child = None

    def stop(sig, frame):
        raise RuntimeError(f'Signal {sig}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        for mode in report['modes']:
            for path, expected in sources.items():
                assert digest(path) == expected, f'Experiment source changed: {path}'
            record, = [r for r in acceptance['records'] if r['mode'] == mode]
            assert digest(record['formal_config']) == record['formal_config_sha256']
            with (OUT/f'compact_{mode}_train1000_supervisor.log').open('x') as log:
                child = subprocess.Popen(['/data/gaoxiang/Code/.venvs/starVLA/bin/python', '-u',
                    str(ROOT/'examples/Robotwin/audits/run_grasp_precision_training_smoke.py'),
                    '--variant', f'compact_{mode}', '--gpu', '0', '--port', '29880', '--stage', 'train1000'],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            report.update(state='training', current_mode=mode, child_pid=child.pid,
                          child_birth=psutil.Process(child.pid).create_time())
            save(status_path, report)
            if child.wait() != 0:
                raise RuntimeError(f'{mode} training supervisor failed')
            result = json.loads((OUT/f'compact_{mode}_train1000_status.json').read_text())
            assert result['state'] == 'trainer_completed_checkpoint_audit_pending' and result['optimizer_steps'] == 1000
            report['completed'].append(mode)
        report['state'] = 'training_pair_completed_checkpoint_audits_pending'
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        report['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save(status_path, report)


if __name__ == '__main__':
    main()
