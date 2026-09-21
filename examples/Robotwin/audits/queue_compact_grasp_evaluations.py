"""Evaluate the formal compact pair sequentially on GPU 5 after verified training."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import psutil

from examples.Robotwin.audits.run_grasp_lift_development import ROOT, PYTHON, save, digest


def live(pid, birth):
    try:
        process = psutil.Process(pid)
        return abs(process.create_time()-birth) < .001 and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def main():
    audit = ROOT/'playground/Checkpoints/gawm_grasp_precision_training_20260909'
    status_path = audit/'compact_evaluation_queue.json'
    assert not status_path.exists()
    pair_path = audit/'compact_training_pair_status.json'
    pair = json.loads(pair_path.read_text())
    pair_identity = pair['supervisor_pid'], pair['supervisor_birth']
    assert live(*pair_identity)
    smoke_path = audit/'seeded_server_smoke/report.json'
    assert json.loads(smoke_path.read_text())['state'] == 'passed'
    evaluator = ROOT/'examples/Robotwin/audits/run_compact_grasp_evaluation.py'
    checkpoint_audit = ROOT/'examples/Robotwin/audits/audit_compact_expert_checkpoint.py'
    paths = [evaluator, checkpoint_audit, Path(__file__), smoke_path,
             ROOT/'deployment/model_server/server_policy.py',
             ROOT/'deployment/model_server/tools/seeded_episode_policy.py']
    pinned = dict(pair['source_sha256'], **{str(path.resolve()): digest(path) for path in paths})
    plan = dict(state='waiting_for_training', supervisor_pid=os.getpid(),
                supervisor_birth=psutil.Process().create_time(), pair_identity=pair_identity,
                training_gpu='0', evaluation_gpu='5', port=6885, policy_seed=20260909,
                modes=['flow', 'regression'], completed=[], source_sha256=pinned)
    child = None

    def status():
        save(status_path, dict(plan, time=time.strftime('%Y-%m-%d %H:%M:%S'),
            child_pid=child.pid if child is not None and child.poll() is None else None))

    def check_sources():
        for name, sha in pinned.items():
            assert digest(Path(name)) == sha, f'Pinned implementation changed: {name}'

    def stop(sig, frame):
        raise RuntimeError(f'Signal {sig}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='4', NO_ALBUMENTATIONS_UPDATE='1')
    try:
        for mode in plan['modes']:
            train_path = audit/f'compact_{mode}_train1000_status.json'
            plan.update(state='waiting_for_training', current_mode=mode, training_status=str(train_path))
            identity = None
            while True:
                current_pair = json.loads(pair_path.read_text())
                assert (current_pair['supervisor_pid'], current_pair['supervisor_birth']) == pair_identity
                if train_path.exists():
                    train = json.loads(train_path.read_text())
                    found = train['supervisor_pid'], train['supervisor_birth']
                    if identity is None:
                        identity = found
                    assert found == identity
                    assert train['variant'] == f'compact_{mode}' and train['steps_requested'] == 1000
                    if train['state'] == 'trainer_completed_checkpoint_audit_pending':
                        break
                    assert train['state'] != 'failed', train.get('error')
                    assert live(*identity), 'Training identity ended without completed status'
                else:
                    assert current_pair['state'] == 'training' and live(*pair_identity), 'Pair ended before this training started'
                assert current_pair['state'] != 'failed', current_pair.get('error')
                status()
                time.sleep(10)
            check_sources()
            assert train['optimizer_steps'] == 1000
            checkpoint = Path(train['final_checkpoint'])
            run = checkpoint.parents[1]
            metrics = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            assert metrics[-1]['step'] == 1000
            assert all(np.isfinite(v) for row in metrics for v in row.values() if isinstance(v, (float, int)))
            for name, sha in train['source_sha256'].items():
                assert digest(Path(name)) == sha, f'Training source changed: {name}'
            plan.update(state='auditing_checkpoint', checkpoint=str(checkpoint),
                        checkpoint_sha256=digest(checkpoint), training_identity=identity)
            audit_result = audit/f'compact_{mode}_train1000_deployment_audit.json'
            with (audit/f'compact_{mode}_train1000_deployment_audit.log').open('x') as log:
                child = subprocess.Popen([str(PYTHON), str(checkpoint_audit), '--checkpoint', str(checkpoint),
                    '--output', str(audit_result)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                status()
                time.sleep(5)
            assert child.returncode == 0, f'Deployment audit failed: {audit_result}'
            verified = json.loads(audit_result.read_text())
            assert verified['state'] == 'strict_deployment_weights_and_real_payload_verified'
            assert verified['mode'] == mode and verified['checkpoint_sha256'] == plan['checkpoint_sha256']
            # Refuse to contend with a new owner of the dedicated evaluation GPU.
            used = subprocess.check_output(['nvidia-smi', '-i', '5', '--query-gpu=memory.used',
                                            '--format=csv,noheader,nounits'], text=True)
            assert int(used.strip()) < 500, f'GPU 5 is occupied ({used.strip()} MiB)'
            check_sources()
            output = ROOT/f'playground/Checkpoints/gawm_grasp_precision_compact_{mode}_train1000_eval_20260909'
            assert not output.exists()
            plan.update(state='evaluating', output=str(output))
            with (audit/f'compact_{mode}_eval_supervisor.log').open('x') as log:
                child = subprocess.Popen([str(PYTHON), '-u', str(evaluator), '--checkpoint', str(checkpoint),
                    '--reference', str(ROOT/'playground/Checkpoints/gawm_grasp_lift_development_20260909/rgb_v2_5k_full'),
                    '--gpu', '5', '--port', '6885', '--policy-seed', str(plan['policy_seed']),
                    '--output', str(output)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                status()
                time.sleep(10)
            assert child.returncode == 0, f'{mode} evaluation failed; no replacement scenes'
            result = json.loads((output/'status.json').read_text())
            assert result['state'] == 'complete' and result['completed'] == 20
            plan['completed'].append(dict(mode=mode, output=str(output), summary=result['summary']))
        plan['state'] = 'complete'
    except BaseException as error:
        plan.update(state='failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        status()


if __name__ == '__main__':
    main()
