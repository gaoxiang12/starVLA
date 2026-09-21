"""CPU-only localization checks at fixed checkpoints of the live object experiment."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

from examples.Robotwin.audits.run_rgb_contact_refinement import digest, live, save

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT.parent / '.venvs/starVLA/bin/python'
AUDIT = ROOT / 'examples/Robotwin/audits/audit_rgb_object_localization.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--training-config', type=Path,
        default=ROOT / 'examples/Robotwin/train_files/starvla_gawm_rgb_object_readout.yaml')
    parser.add_argument('--reference-input-sha256',
        help='Require an identical native-frame manifest to an earlier experiment')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.run = args.run.resolve(); args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    cfg = args.run / 'config.full.yaml'
    files = [cfg, args.training_config.resolve(), AUDIT, ROOT / 'starVLA/model/modules/rgb_object_readout.py',
             ROOT / 'starVLA/model/modules/spatial_goal_readout.py',
             ROOT / 'starVLA/dataloader/rgb_object_supervision.py']
    hashes = {str(path): digest(path) for path in files}
    training = psutil.Process(args.training_pid)
    command = training.cmdline()
    assert Path(command[command.index('--config_yaml')+1]).resolve() == args.training_config.resolve()
    save(args.output / 'manifest.json', dict(supervisor_pid=os.getpid(), steps=[500, 1000, 2000],
        training_handle=dict(pid=training.pid, birth=training.create_time()), run=str(args.run),
        pinned_files_sha256=hashes, device='cpu',
        reference_input_sha256=args.reference_input_sha256,
        note='Read-only 160-frame / 20-episode native-head localization checks. '
             'No additional GPU work, optimizer update, policy intervention, or success-rate estimate.'))
    active, current, done = None, None, {}

    def status(state, **extra):
        save(args.output / 'status.json', dict(state=state, supervisor_pid=os.getpid(),
            step=current, done=done, child_pid=active.pid if active and active.poll() is None else None,
            time=time.strftime('%Y-%m-%d %H:%M:%S'), **extra))

    def stop(signum, frame):
        raise RuntimeError(f'Signal {signum}')

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        for current in (500, 1000, 2000):
            checkpoint = args.run / 'checkpoints' / f'steps_{current}_pytorch_model.pt'
            while True:
                summary = args.run / 'summary.jsonl'
                ready = summary.exists() and any(json.loads(line)['steps'] == current
                                                for line in summary.read_text().splitlines())
                if ready:
                    assert checkpoint.is_file()
                    break
                if not live(training):
                    raise RuntimeError(f'Training ended without completed checkpoint {current}')
                status('waiting_for_checkpoint')
                time.sleep(15)
            assert all(digest(Path(path)) == expected for path, expected in hashes.items()), 'Pinned analysis inputs changed'
            output = args.output / f'step{current}.json'
            command = [str(PYTHON), str(AUDIT), '--config', str(cfg), '--checkpoint', str(checkpoint),
                       '--temperature-probe', '1', '.5', '.25', '--output', str(output)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', PYTHONPATH=str(ROOT),
                       NO_ALBUMENTATIONS_UPDATE='1', PYTHONNOUSERSITE='1')
            with (args.output / f'step{current}.log').open('x') as log:
                active = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while active.poll() is None:
                status('analyzing')
                time.sleep(15)
            if active.returncode:
                raise RuntimeError(f'Localization audit exited {active.returncode}')
            result = json.loads(output.read_text())
            assert result['state'] == 'complete' and result['frames'] == 160
            if args.reference_input_sha256:
                assert result['input_manifest_sha256'] == args.reference_input_sha256
            done[str(current)] = dict(checkpoint_sha256=result['checkpoint_sha256'],
                input_manifest_sha256=result['input_manifest_sha256'], summary=result['summary'],
                temperature_probe=result['temperature_probe'])
            assert len({row['input_manifest_sha256'] for row in done.values()}) == 1
            save(args.output / 'comparison.json', dict(checkpoints=done,
                note='Same held-out native images and heuristic visible-region targets. '
                     'Temperature entries are coordinate-only counterfactuals, not modified policy evaluation.'))
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
