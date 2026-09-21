"""Bounded post-hoc pregrasp feedback probe; final test scenes remain untouched."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from examples.Robotwin.audits.run_grasp_lift_development import ROOT, PYTHON, SIM_PYTHON, digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.gpu in ('0', '4', '5', '6')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT/'playground/Checkpoints/gawm_rgb_focus_local_v2_5k_20260907/final_model/pytorch_model.pt'
    # Positive reference plus both previously selected failures. No model tuning,
    # seed replacement, automatic retries, or training follows this probe.
    cases = [dict(seed=41000001, execute_horizon=16),
             dict(seed=41000002, execute_horizon=1), dict(seed=41000003, execute_horizon=1)]
    protocol = json.loads((ROOT/'examples/Robotwin/audits/grasp_lift_protocol_20260909.json').read_text())
    development = {row['seed'] for row in protocol['records'] if row['split'] == 'development'}
    assert all(row['seed'] in development for row in cases)
    sources = [ROOT/'examples/Robotwin/audits'/name for name in (
        'run_pregrasp_feedback_probe.py', 'record_pregrasp_precision.py', 'run_grasp_lift_case.py',
        'grasp_lift_scoring.py', 'grasp_lift_protocol_20260909.json')]
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        source_sha256={str(p):digest(p) for p in sources}, gpu=args.gpu, port=args.port,
        cases=cases, max_actions=600, max_sim_seconds=120, max_attempts=2,
        note='Post-hoc selected development scenes, not an unbiased success-rate estimate. '
             'Same float32 checkpoint and full initialization; only execution horizon changes for failures. '
             'Grasp geometry and contacts are passive audit data, never policy input.')
    save(out/'manifest.json', manifest)
    server = active = None
    report = dict(state='starting', supervisor_pid=os.getpid(), results=[])

    def status():
        save(out/'status.json', dict(report, time=time.strftime('%Y-%m-%d %H:%M:%S'),
            server_pid=server.pid if server is not None and server.poll() is None else None,
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

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONNOUSERSITE='1',
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', NO_ALBUMENTATIONS_UPDATE='1',
        PYTHONPATH=f'{ROOT}:{ROOT}/examples/Robotwin/eval_files')
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', args.port))
        with (out/'server.log').open('x') as log:
            server = subprocess.Popen([str(PYTHON), str(ROOT/'deployment/model_server/server_policy.py'),
                '--ckpt_path', str(checkpoint), '--port', str(args.port), '--idle_timeout', '-1'],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
        deadline = time.monotonic()+180
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}')
            try:
                with socket.create_connection(('127.0.0.1', args.port), timeout=1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError('Server startup timeout')
                status()
                time.sleep(2)
        for row in cases:
            for path, expected in manifest['source_sha256'].items():
                assert digest(path) == expected, f'Pinned source changed: {path}'
            name = f"seed_{row['seed']}_execute{row['execute_horizon']}"
            report.update(state='running', current_case=row)
            with (out/f'{name}.log').open('x') as log:
                active = subprocess.Popen([str(SIM_PYTHON),
                    str(ROOT/'examples/Robotwin/audits/record_pregrasp_precision.py'),
                    '--seed', str(row['seed']), '--mode', 'full', '--output', str(out/name),
                    '--checkpoint', str(checkpoint), '--port', str(args.port),
                    '--execute-horizon', str(row['execute_horizon'])],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
            while active.poll() is None:
                status()
                time.sleep(5)
            if active.returncode or not (out/name/'pregrasp_physics_poses.npz').exists():
                raise RuntimeError(f'Invalid case {name}, exit {active.returncode}')
            result = json.loads((out/name/'result.json').read_text())
            assert result['state'] == 'complete'
            report['results'].append(dict(case=row, result_path=str(out/name/'result.json'),
                                          result=result['result']))
            status()
        report['state'] = 'complete'
    except BaseException as exc:
        report.update(state='failed', error=repr(exc))
        raise
    finally:
        cleanup(active)
        cleanup(server)
        status()


if __name__ == '__main__':
    main()
