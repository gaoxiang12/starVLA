"""Wait on a specific live training identity, verify final weights, then evaluate."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
from omegaconf import OmegaConf
import psutil
import torch

from examples.Robotwin.audits.run_grasp_lift_development import ROOT, PYTHON, save, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-status', type=Path, required=True)
    parser.add_argument('--gpu', choices=['4','6'], required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    assert not args.status.exists() and not args.output.exists()
    initial = json.loads(args.training_status.read_text())
    identity = initial['supervisor_pid'], initial['supervisor_birth']
    assert initial['gpu'] == args.gpu and initial['steps_requested'] == 1000
    process = psutil.Process(identity[0])
    assert abs(process.create_time()-identity[1]) < .001
    assert 'run_grasp_precision_training_smoke.py' in ' '.join(process.cmdline())
    evaluator = ROOT/'examples/Robotwin/audits/run_grasp_precision_development.py'
    plan = dict(state='waiting_for_training', supervisor_pid=os.getpid(),
        supervisor_birth=psutil.Process().create_time(), training_pid=identity[0],
        training_birth=identity[1], training_status=str(args.training_status.resolve()),
        gpu=args.gpu, output=str(args.output.resolve()), evaluator_sha256=digest(evaluator),
        required_optimizer_steps=1000, evaluation_scenes=20)
    child = None

    def status():
        save(args.status, dict(plan, time=time.strftime('%Y-%m-%d %H:%M:%S'),
            evaluator_pid=child.pid if child is not None and child.poll() is None else None))

    def stop(sig, frame):
        raise RuntimeError(f'Signal {sig}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        while True:
            current = json.loads(args.training_status.read_text())
            assert (current['supervisor_pid'],current['supervisor_birth']) == identity
            if current['state'] == 'trainer_completed_checkpoint_audit_pending':
                break
            if current['state'] == 'failed':
                raise RuntimeError(f'Training failed: {current.get("error")}')
            # A stale status file alone never proves that training is still live.
            process = psutil.Process(identity[0])
            assert abs(process.create_time()-identity[1]) < .001 and process.status() != psutil.STATUS_ZOMBIE
            status()
            time.sleep(10)
        assert current['optimizer_steps'] == 1000
        checkpoint = Path(current['final_checkpoint'])
        run = checkpoint.parents[1]
        cfg = OmegaConf.load(run/'config.full.yaml')
        rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        assert rows[-1]['step'] == 1000
        assert all(np.isfinite(value) for row in rows for value in row.values() if isinstance(value,(int,float)))
        if cfg.framework.name == 'GAWMCartesian':
            for name in ('starVLA/model/modules/action_model/CartesianResidualACT.py',
                         'starVLA/model/framework/WM4A/GAWMCartesian.py'):
                path = ROOT/name
                assert digest(path) == current['source_sha256'][str(path)], 'Trained model implementation changed'
            # Numerical health gate, not a claim of successful grasp alignment.
            error = float(np.median([r['cartesian_actual_target_error_mm'] for r in rows[-5:]]))
            assert error < 50, f'Unstable final Cartesian training geometry: {error} mm'
            assert min(r['cartesian_ik_converged_fraction'] for r in rows[-5:]) >= .95
        source_stats = json.loads(Path(cfg.datasets.vla_data.normalization_statistics_path).read_text())
        stored_stats = json.loads((run/'dataset_statistics.json').read_text())
        assert source_stats['aloha']['action'] == stored_stats['aloha']['action']
        assert source_stats['aloha']['state'] == stored_stats['aloha']['state']
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert all(torch.isfinite(value).all() for value in weights.values() if torch.is_floating_point(value))
        if cfg.framework.name == 'GAWMCartesian':
            assert weights['action_models.aloha.pose_projection.layers.2.weight'].abs().sum() > 0
        del weights
        assert digest(evaluator) == plan['evaluator_sha256']
        plan.update(state='evaluating', checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
            training_verified_steps=1000)
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4',
            NO_ALBUMENTATIONS_UPDATE='1')
        with args.status.with_suffix('.evaluation.log').open('x') as log:
            child = subprocess.Popen([str(PYTHON),'-u',str(evaluator),'--checkpoint',str(checkpoint),
                '--reference',str(ROOT/'playground/Checkpoints/gawm_grasp_lift_development_20260909/rgb_v2_5k_full'),
                '--gpu',args.gpu,'--port',str(args.port),'--output',str(args.output)],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
        while child.poll() is None:
            status()
            time.sleep(10)
        if child.returncode:
            raise RuntimeError(f'Evaluator exited {child.returncode}')
        result = json.loads((args.output/'status.json').read_text())
        assert result['state'] == 'complete' and result['completed'] == 20
        plan.update(state='complete', evaluation_summary=result['summary'])
    except BaseException as error:
        plan.update(state='failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        status()


if __name__ == '__main__':
    main()
