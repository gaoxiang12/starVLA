"""Bounded development-only grasp campaign, with a fresh simulator per scene."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT.parent / '.venvs/starVLA/bin/python'
SIM_PYTHON = ROOT.parent / '.venvs/RoboTwin/bin/python'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--mode', choices=['full', 'near'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--execute-horizon', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.limit <= 20:
        parser.error('Only 1..20 predeclared development scenes may be run')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve()
    protocol_path = ROOT / 'examples/Robotwin/audits/grasp_lift_protocol_20260909.json'
    protocol = json.loads(protocol_path.read_text())
    selected = [r for r in protocol['records'] if r['split'] == 'development'][:args.limit]
    pinned = [protocol_path, ROOT/'examples/Robotwin/audits/run_grasp_lift_case.py',
              ROOT/'examples/Robotwin/audits/grasp_lift_scoring.py',
              ROOT/'examples/Robotwin/audits/audit_rgb_scene_and_color.py',
              ROOT/'examples/Robotwin/audits/rgb_scene_and_color_20260907.json',
              ROOT/'examples/Robotwin/eval_files/model2robotwin_interface.py']
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), mode=args.mode,
                    gpu=args.gpu, port=args.port, execute_horizon=args.execute_horizon,
                    source_sha256={str(p):digest(p) for p in pinned}, scenes=selected,
                    supervisor_pid=os.getpid(), precision='float32',
                    note='Development diagnostic only. Setup/runtime failures remain in denominator accounting as invalid, not scored policy failures. No scene replacement; final test scenes are untouched.')
    save(out/'manifest.json', manifest)
    results = []
    active = server = None
    report = dict(state='starting', results=results, supervisor_pid=os.getpid())

    def status():
        save(out/'status.json', dict(report, results=results, time=time.strftime('%Y-%m-%d %H:%M:%S'),
            server_pid=server.pid if server is not None else None,
            case_pid=active.pid if active is not None and active.poll() is None else None))

    def stop(sig, frame):
        raise RuntimeError(f'Signal {sig}')

    def cleanup(process):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONNOUSERSITE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
               PYTHONPATH=f'{ROOT}:{ROOT}/examples/Robotwin/eval_files', NO_ALBUMENTATIONS_UPDATE='1')
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', args.port))
        command = [str(PYTHON), str(ROOT/'deployment/model_server/server_policy.py'),
                   '--ckpt_path', str(checkpoint), '--port', str(args.port), '--idle_timeout', '-1']
        with (out/'server.log').open('x') as log:
            server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic()+180
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}')
            try:
                with socket.create_connection(('127.0.0.1', args.port), timeout=1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError('Server startup timeout; original server will be cleaned up')
                status()
                time.sleep(2)
        for record in selected:
            for path, expected in manifest['source_sha256'].items():
                if digest(path) != expected:
                    raise RuntimeError(f'Pinned source changed: {path}')
            case = out/f"scene_{record['scene_id']:03d}"
            command = [str(SIM_PYTHON), str(ROOT/'examples/Robotwin/audits/run_grasp_lift_case.py'),
                '--seed', str(record['seed']), '--mode', args.mode, '--output', str(case),
                '--checkpoint', str(checkpoint), '--port', str(args.port),
                '--execute-horizon', str(args.execute_horizon)]
            report.update(state='running', current_scene=record)
            with (out/f"scene_{record['scene_id']:03d}.log").open('x') as log:
                active = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, start_new_session=True)
            while active.poll() is None:
                status()
                time.sleep(10)
            result_path = case/'result.json'
            if active.returncode or not result_path.exists():
                results.append(dict(scene=record, state='invalid', exitcode=active.returncode,
                                    status_path=str(case/'status.json')))
                # Stop on infrastructure/setup failure; do not skip to easier scenes.
                raise RuntimeError(f'Case failed: {record}, exit {active.returncode}')
            result = json.loads(result_path.read_text())
            assert result['seed'] == record['seed'] and result['state'] == 'complete'
            results.append(dict(scene=record, state='complete', result=result['result'],
                                termination=result['termination'], result_path=str(result_path)))
            status()
        report.update(state='complete', completed=len(results),
                      successes=sum(r['result']['success'] for r in results),
                      first_attempt_successes=sum(r['result']['first_attempt_success'] for r in results))
    except BaseException as exc:
        report.update(state='failed', error=repr(exc))
        raise
    finally:
        cleanup(active)
        cleanup(server)
        status()


if __name__ == '__main__':
    main()
