"""Own one frozen policy server and an isolated three-source recovery collector."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from examples.Robotwin.audits.run_rgb_contact_refinement import digest, save

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--port', type=int, default=6670)
    parser.add_argument('--allow-tabletop-handoff', action='store_true')
    args = parser.parse_args()
    output, raw_root = args.output.resolve(), args.raw_root.resolve()
    sources_path = args.sources.resolve()
    sources = json.loads(sources_path.read_text())
    checkpoint = Path(sources['policy_checkpoint']).resolve()
    assert checkpoint.is_file() and not raw_root.exists()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(sources=str(sources_path), source_manifest_sha256=digest(sources_path),
                    checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                    gpu=args.gpu, port=args.port, supervisor_pid=os.getpid(), raw_root=str(raw_root),
                    allow_tabletop_handoff=args.allow_tabletop_handoff,
                    collector_code_sha256=digest(ROOT / 'examples/Robotwin/audits/collect_rgb_recovery_pilot.py'),
                    recording_hook_code_sha256=digest(ROOT / 'examples/Robotwin/audits/recovery_frame_capture.py'),
                    note='Collection pilot, no automatic training, no evaluation seed100000 reuse. '
                         'Only this supervisor\'s server and collector are managed.')
    save(output / 'manifest.json', manifest)
    processes = {}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(ROOT),
               STARVLA_PYTHON=str(ROOT.parent / '.venvs/starVLA/bin/python'),
               ROBOTWIN_USE_BF16='0', ROBOTWIN_SPATIAL_ABLATION='full', ROBOTWIN_SERVER_IDLE_TIMEOUT='-1',
               PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', NO_ALBUMENTATIONS_UPDATE='1')

    def status(state, **extra):
        save(output / 'status.json', dict(state=state, supervisor_pid=os.getpid(),
             time=time.strftime('%Y-%m-%d %H:%M:%S'),
             processes={name: dict(pid=p.pid, returncode=p.poll()) for name, p in processes.items()}, **extra))

    def launch(name, command):
        with (output / f'{name}.log').open('x') as log:
            processes[name] = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def stop(signum, frame):
        raise RuntimeError(f'Signal {signum}')

    def cleanup():
        for name in ('collector', 'server'):
            p = processes.get(name)
            if p is not None and p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        launch('server', ['bash', str(ROOT / 'examples/Robotwin/eval_files/run_policy_server.sh'),
                          str(checkpoint), args.gpu, str(args.port)])
        launch('collector', [str(ROOT.parent / '.venvs/RoboTwin/bin/python'), '-u',
             str(ROOT / 'examples/Robotwin/audits/collect_rgb_recovery_pilot.py'),
             '--robotwin-root', str(ROOT.parent / 'RoboTwin'), '--sources', str(sources_path),
             '--checkpoint', str(checkpoint), '--port', str(args.port), '--output', str(output),
             '--raw-root', str(raw_root),
             *(['--allow-tabletop-handoff'] if args.allow_tabletop_handoff else [])])
        while processes['collector'].poll() is None:
            if processes['server'].poll() is not None:
                raise RuntimeError(f"Owned policy server exited {processes['server'].returncode}")
            status('running')
            time.sleep(15)
        if processes['collector'].returncode:
            raise RuntimeError(f"Collector exited {processes['collector'].returncode}")
        collected = json.loads((output / 'collector_status.json').read_text())
        assert collected['state'] == 'complete'
        cleanup()
        status('complete', accepted_raw_clips=collected['accepted_raw_clips'])
    except BaseException as exc:
        cleanup()
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
