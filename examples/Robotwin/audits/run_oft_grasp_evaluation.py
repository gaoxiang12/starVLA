"""Evaluate released OFT on the frozen 20 scenes after a strict deployment audit."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import psutil

from examples.Robotwin.audits.run_grasp_lift_development import ROOT, PYTHON, SIM_PYTHON, digest, save
from examples.Robotwin.audits.measure_grasp_precision import measure, summarize
from examples.Robotwin.audits.oft_reference_adapter import CONTRACT, UNNORM_KEY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--execute-horizon', type=int, choices=[16, 50], required=True)
    parser.add_argument('--gpu', choices=['0','4','5','6'], required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    assert checkpoint.is_file()
    reference = args.reference.resolve()
    reference_status = json.loads((reference/'status.json').read_text())
    assert reference_status['state'] == 'complete' and reference_status['completed'] == 20
    protocol_path = ROOT/'examples/Robotwin/audits/grasp_lift_protocol_20260909.json'
    protocol = json.loads(protocol_path.read_text())
    selected = [r for r in protocol['records'] if r['split'] == 'development']
    assert len(selected) == 20
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = [protocol_path, Path(__file__), ROOT/'examples/Robotwin/audits/measure_grasp_precision.py',
        ROOT/'examples/Robotwin/audits/record_pregrasp_precision.py',
        ROOT/'examples/Robotwin/audits/run_grasp_lift_case.py',
        ROOT/'examples/Robotwin/audits/grasp_lift_scoring.py',
        ROOT/'examples/Robotwin/audits/audit_rgb_scene_and_color.py',
        ROOT/'examples/Robotwin/audits/rgb_scene_and_color_20260907.json',
        ROOT/'examples/Robotwin/eval_files/model2robotwin_interface.py',
        ROOT/'examples/Robotwin/audits/serve_oft_reference.py',
        ROOT/'examples/Robotwin/audits/run_oft_grasp_case.py',
        ROOT/'examples/Robotwin/audits/oft_reference_adapter.py',
        ROOT/'starVLA/model/framework/VLM4A/QwenOFT.py',
        ROOT/'starVLA/model/modules/vlm/QWen3.py',
        ROOT/'starVLA/model/modules/action_model/MLP_ActionHeader.py',
        ROOT/'starVLA/model/framework/base_framework.py',
        ROOT/'starVLA/model/framework/share_tools.py',
        ROOT/'examples/Robotwin/train_files/data_registry/data_config.py',
        ROOT/'deployment/model_server/policy_wrapper.py',
        ROOT/'deployment/model_server/policy_norm_processor.py']
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        reference=str(reference), contract=CONTRACT,
        source_sha256={str(p.resolve()):digest(p) for p in files}, scenes=selected,
        mode='full', execute_horizon=args.execute_horizon,
        precision='BF16 VLM / FP32 regression head', gpu=args.gpu, port=args.port,
        supervisor_pid=os.getpid(), supervisor_birth=psutil.Process().create_time(),
        note='Development only; physical truth is passive recording. No replacement or test-scene use.')
    save(out/'manifest.json', manifest)
    results = []
    report = dict(state='starting', supervisor_pid=os.getpid(), supervisor_birth=manifest['supervisor_birth'])
    active = server = None

    def status():
        save(out/'status.json', dict(report, results=results, summary=dict(summarize(results),
            any_block_lifted_and_held=sum(r['any_block_lifted_and_held'] for r in results),
            wrong_object_held=sum(r['wrong_object_held'] for r in results)),
            time=time.strftime('%Y-%m-%d %H:%M:%S'),
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
        PYTHONPATH=f'{ROOT}:{ROOT}/examples/Robotwin/eval_files',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', args.port))
        with (out/'server.log').open('x') as log:
            server = subprocess.Popen([str(PYTHON), str(ROOT/'examples/Robotwin/audits/serve_oft_reference.py'),
                '--checkpoint', str(checkpoint), '--port', str(args.port), '--output', str(out),
                '--smoke-video', str(reference/'scene_000/rollout.mp4')],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        deadline = time.monotonic()+1200
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}')
            try:
                from websockets.sync.client import connect
                from deployment.model_server.tools import msgpack_numpy
                with connect(f'ws://127.0.0.1:{args.port}', open_timeout=2, proxy=None) as ws:
                    metadata = msgpack_numpy.unpackb(ws.recv(timeout=5))
                    assert metadata['action_chunk_size'] == 50
                    assert metadata['available_unnorm_keys'] == [UNNORM_KEY]
                    assert metadata['oft_reference_contract'] == CONTRACT
                    assert metadata['deployment_audit_state'] == 'strict_weights_normalization_real_payload_verified'
                    save(out/'server_metadata.json', metadata)
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError('Server startup timeout')
                status()
                time.sleep(2)
        for record in selected:
            for path, expected in manifest['source_sha256'].items():
                assert digest(path) == expected, f'Pinned source changed: {path}'
            name = f"scene_{record['scene_id']:03d}"
            case = out/name
            report.update(state='running', current_scene=record)
            with (out/f'{name}.log').open('x') as log:
                active = subprocess.Popen([str(SIM_PYTHON),
                    str(ROOT/'examples/Robotwin/audits/run_oft_grasp_case.py'),
                    '--seed', str(record['seed']), '--mode', 'full', '--output', str(case),
                    '--checkpoint', str(checkpoint), '--port', str(args.port), '--execute-horizon', str(args.execute_horizon)],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            while active.poll() is None:
                status()
                time.sleep(5)
            if active.returncode or not (case/'result.json').exists():
                report['invalid_scene'] = dict(scene=record, exit_code=active.returncode)
                raise RuntimeError(f'Invalid scene {name}; stopping without replacement')
            row = measure(case, reference/name, False)
            assert row['seed'] == record['seed']
            save(case/'precision.json', row)
            rng_rows = [json.loads(line) for line in (out/'policy_queries.jsonl').read_text().splitlines()]
            episode_rng = [r for r in rng_rows if r['episode_index'] == len(results)]
            assert episode_rng and episode_rng[0]['episode_start'] is True
            assert [r['query_index'] for r in episode_rng] == list(range(len(episode_rng)))
            assert not any(r['episode_start'] for r in episode_rng[1:])
            assert max(r['episode_index'] for r in rng_rows) == len(results)
            save(case/'policy_queries.json', episode_rng)
            assert all(not r['state_present'] for r in episode_rng)
            row['any_block_lifted_and_held'] = any(row['scored_result']['lifted_and_held'])
            row['wrong_object_held'] = row['scored_result']['wrong_object_held']
            save(case/'precision.json', row)
            results.append(row)
            status()
        report.update(state='complete', completed=20)
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        cleanup(active)
        cleanup(server)
        status()


if __name__ == '__main__':
    main()
