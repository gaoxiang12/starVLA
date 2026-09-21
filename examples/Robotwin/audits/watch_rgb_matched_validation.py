"""Run fixed 500-step validation once each matched experiment saves its weights."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import psutil

from examples.Robotwin.audits.run_rgb_contact_refinement import digest, save

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = ROOT / 'playground/Checkpoints'
AUDITS = ROOT / 'examples/Robotwin/audits'
PYTHON = ROOT.parent / '.venvs/starVLA/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, default=500)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    with (args.output / 'claim.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), step=args.step), stream)
    pair = json.loads((CHECKPOINTS / 'gawm_rgb_contact_refinement_20260907/status.json').read_text())
    goal = json.loads((CHECKPOINTS / 'gawm_rgb_goal_readout_20260907/status.json').read_text())
    records = dict(pair['jobs'])
    records['goal'] = goal
    processes, runs = {}, {}
    for name in ('baseline', 'contact', 'goal'):
        record = records[name]
        assert record['state'] == 'running' and record['stage'] == 'train'
        process = psutil.Process(record['pid'])
        assert f'starvla_gawm_rgb_refine_{name}.yaml' in ' '.join(process.cmdline())
        processes[name] = process
        runs[name] = CHECKPOINTS / f'gawm_rgb_focus_refine_{name}_2k_20260907'
    save(args.output / 'manifest.json', dict(step=args.step, samples=128, gpu=args.gpu,
         source_processes={name: dict(pid=p.pid, birth=p.create_time()) for name, p in processes.items()},
         split=str(AUDITS / 'rgb_scene_safe_validation_20260907.json'),
         note='Read-only matched-step validation, no training interruption or parameter changes. '
              'Checkpoint summary is written after torch.save; no partially written checkpoint is read.'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4',
               NO_ALBUMENTATIONS_UPDATE='1', PYTHONNOUSERSITE='1')
    pending, done, failures = list(runs), {}, {}
    active = None
    current = None

    def status(state, **extra):
        save(args.output / 'status.json', dict(state=state, supervisor_pid=os.getpid(),
             time=time.strftime('%Y-%m-%d %H:%M:%S'), pending=pending, done=done, failures=failures,
             current=current, child_pid=active.pid if active and active.poll() is None else None, **extra))

    def execute(command, log):
        nonlocal active
        with log.open('x') as stream:
            active = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while active.poll() is None:
            status('analyzing')
            time.sleep(15)
        if active.returncode:
            raise RuntimeError(f'Diagnostic exited {active.returncode}: {log}')

    def stop(signum, frame):
        raise RuntimeError(f'Watcher received signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        while pending:
            for name in list(pending):
                run, process = runs[name], processes[name]
                summary = run / 'summary.jsonl'
                ready = summary.exists() and any(json.loads(line)['steps'] == args.step
                                                for line in summary.read_text().splitlines())
                if not ready:
                    if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                        failures[name] = 'Training process ended without requested checkpoint'
                        pending.remove(name)
                    continue
                checkpoint = run / 'checkpoints' / f'steps_{args.step}_pytorch_model.pt'
                assert checkpoint.is_file()
                current = name
                prefix = args.output / name
                execute([str(PYTHON), str(AUDITS / 'analyze_rgb_focus_usage.py'), '--run', str(run),
                         '--checkpoint', str(checkpoint), '--samples', '128', '--split-manifest',
                         str(AUDITS / 'rgb_scene_safe_validation_20260907.json'),
                         '--output', str(prefix.with_suffix('.usage.json'))], prefix.with_suffix('.usage.log'))
                execute([str(PYTHON), str(AUDITS / 'audit_rgb_contact_error.py'), '--run', str(run),
                         '--step', str(args.step), '--split-manifest',
                         str(AUDITS / 'rgb_scene_safe_validation_20260907.json'), '--urdf',
                         '/data/gaoxiang/Code/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf',
                         '--output', str(prefix.with_suffix('.contact.json'))], prefix.with_suffix('.contact.log'))
                done[name] = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint))
                pending.remove(name)
                current = None
            if pending:
                status('waiting_for_checkpoint')
                time.sleep(15)
        if failures:
            status('failed')
            return
        contact = {name: json.loads((args.output / f'{name}.contact.json').read_text()) for name in runs}
        usage = {name: json.loads((args.output / f'{name}.usage.json').read_text()) for name in runs}
        key = lambda row: (row['episode'], row['event'], row['arm'], row['lookahead'])
        expected = [key(row) for row in contact['baseline']['rows']]
        baseline_errors = np.asarray([row['tcp_error_mm'] for row in contact['baseline']['rows']])
        paired = {}
        for name in runs:
            assert [key(row) for row in contact[name]['rows']] == expected
            assert usage[name]['validation_episode_ids'] == usage['baseline']['validation_episode_ids']
            errors = np.asarray([row['tcp_error_mm'] for row in contact[name]['rows']])
            delta = baseline_errors - errors
            paired[name] = dict(samples=len(delta), mean_tcp_improvement_mm=float(delta.mean()),
                                median_tcp_improvement_mm=float(np.median(delta)),
                                fraction_lower_tcp_error=float((delta > 0).mean()))
        save(args.output / 'comparison.json', dict(step=args.step, paired_contact=paired,
             contact_by_lookahead={name: report['by_lookahead'] for name, report in contact.items()},
             action_and_ablation={name: report['scores'] for name, report in usage.items()},
             note='Same fixed validation inputs. Positive paired improvement means lower TCP error. '
                  'Teacher-observation geometry and action error do not establish closed-loop task success.'))
        status('complete')
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
