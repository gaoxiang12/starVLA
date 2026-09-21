"""Collect a fixed 20-scene train-source correction set, stopping on any invalid case."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from examples.Robotwin.audits.run_grasp_lift_development import ROOT, SIM_PYTHON, digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    args = parser.parse_args()
    assert args.gpu in ('0', '4', '5', '6')
    out, raw = args.output.resolve(), args.raw_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    sources_path = ROOT/'examples/Robotwin/audits/pregrasp_correction_train_sources_20260909.json'
    sources = json.loads(sources_path.read_text())
    assert len(sources['records']) == 20
    rng = np.random.default_rng(20260909)
    cases = []
    for i, source in enumerate(sources['records']):
        radius = [10., 20., 30., 40., 60.][i % 5]
        angle = float(rng.uniform(-np.pi, np.pi))
        cases.append(dict(source_episode=source['source_episode'], scene_seed=source['scene_seed'],
            offset_xy_mm=[radius*np.cos(angle), radius*np.sin(angle)], radius_mm=radius,
            yaw_deg=float([0, 7, -7, 14, -14][i % 5])))
    pinned = [sources_path, Path(sources['split']), Path(sources['seed_file']),
              *[ROOT/'examples/Robotwin/audits'/name for name in (
                  'collect_pregrasp_correction.py', 'run_pregrasp_correction_collection.py',
                  'grasp_lift_scoring.py', 'recovery_frame_capture.py', 'verify_rgb_recovery_sources.py')]]
    manifest = dict(cases=cases, source_sha256={str(p):digest(p) for p in pinned},
        gpu=args.gpu, raw_root=str(raw), sources=str(sources_path), training_enabled=False,
        note='Twenty fixed original training scenes; radius schedule 10/20/30/40/60 mm and yaw 0/+7/-7/+14/-14 deg. '
             'Directions fixed before outcomes. No scene replacement or automatic retry. Expert data, not policy evaluation.')
    save(out/'manifest.json', manifest)
    report = dict(state='starting', supervisor_pid=os.getpid(), cases=cases, completed=[], training_enabled=False)
    active = None

    def status():
        save(out/'status.json', dict(report, case_pid=active.pid if active is not None and active.poll() is None else None,
                                   time=time.strftime('%Y-%m-%d %H:%M:%S')))

    def stop(sig, frame):
        raise RuntimeError(f'Signal {sig}')

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED='1',
        PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', NO_ALBUMENTATIONS_UPDATE='1',
        PYTHONPATH=f'{ROOT}:{ROOT}/examples/Robotwin/eval_files')
    try:
        for row in cases:
            for path, expected in manifest['source_sha256'].items():
                assert digest(path) == expected, f'Pinned collection source changed: {path}'
            name = f"source_{row['source_episode']:06d}_offset{int(row['radius_mm'])}"
            command = [str(SIM_PYTHON), str(ROOT/'examples/Robotwin/audits/collect_pregrasp_correction.py'),
                '--sources', str(sources_path), '--source-episode', str(row['source_episode']),
                '--offset-xy-mm', *map(str,row['offset_xy_mm']), '--yaw-deg', str(row['yaw_deg']),
                '--output', str(out/name), '--raw-output', str(raw/name/'demo_clean')]
            report.update(state='collecting', current_case=row)
            with (out/f'{name}.log').open('x') as log:
                active = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while active.poll() is None:
                status()
                time.sleep(5)
            report['completed'].append(dict(case=name, exitcode=active.returncode))
            status()
            if active.returncode:
                raise RuntimeError(f'Invalid correction case {name}, exit {active.returncode}; no replacement')
            result = json.loads((out/name/'result.json').read_text())
            assert result['state'] == 'raw_verified' and result['result']['first_attempt_success']
        report['state'] = 'complete'
    except BaseException as exc:
        report.update(state='failed', error=repr(exc))
        raise
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=5)
        status()


if __name__ == '__main__':
    main()
