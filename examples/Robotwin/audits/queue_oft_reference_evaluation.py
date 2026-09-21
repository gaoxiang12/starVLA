"""Wait for pinned assets, then audit and evaluate OFT on the development scenes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil
import yaml

from examples.Robotwin.audits.run_grasp_lift_development import ROOT, PYTHON, digest, save

OUT = ROOT/'playground/Checkpoints/oft_grasp_precision_reference_20260909'
OFT = Path('/data/gaoxiang/ckpts/Qwen3-VL-OFT-RoboTwin2-All')
BASE = Path('/data/gaoxiang/ckpts/Qwen3-VL-4B-Instruct')
WEIGHT_SHA = 'fea31ea190fd729aad23677c8d8b40b22b6441b5a22d0b71dcacfbbbaba175f1'


def main():
    status_path = OUT/'evaluation_queue.json'
    assert not status_path.exists()
    download_path = OUT/'download_status.json'
    download = json.loads(download_path.read_text())
    identity = download['pid'], download['birth']
    sources = [Path(__file__), ROOT/'examples/Robotwin/audits/run_oft_grasp_evaluation.py',
               ROOT/'examples/Robotwin/audits/run_oft_grasp_case.py',
               ROOT/'examples/Robotwin/audits/oft_reference_adapter.py',
               ROOT/'examples/Robotwin/audits/serve_oft_reference.py',
               ROOT/'starVLA/model/framework/VLM4A/QwenOFT.py',
               ROOT/'starVLA/model/modules/vlm/QWen3.py',
               ROOT/'starVLA/model/modules/action_model/MLP_ActionHeader.py',
               ROOT/'deployment/model_server/policy_wrapper.py',
               ROOT/'deployment/model_server/policy_norm_processor.py',
               ROOT/'starVLA/model/framework/base_framework.py',
               ROOT/'starVLA/model/framework/share_tools.py',
               ROOT/'starVLA/dataloader/gr00t_lerobot/transform/state_action.py',
               ROOT/'examples/Robotwin/audits/grasp_lift_protocol_20260909.json',
               ROOT/'examples/Robotwin/audits/run_grasp_lift_case.py',
               ROOT/'examples/Robotwin/audits/record_pregrasp_precision.py',
               ROOT/'examples/Robotwin/audits/measure_grasp_precision.py',
               ROOT/'examples/Robotwin/audits/grasp_lift_scoring.py',
               ROOT/'examples/Robotwin/eval_files/model2robotwin_interface.py',
               ROOT/'examples/Robotwin/train_files/data_registry/data_config.py']
    pinned = {str(path): digest(path) for path in sources}
    report = dict(state='waiting_for_download', supervisor_pid=os.getpid(),
                  supervisor_birth=psutil.Process().create_time(), download_identity=identity,
                  gpu='6', ports={'50': 6896, '16': 6897}, execute_horizons=[50, 16], completed=[], source_sha256=pinned,
                  note='Development-only red grasp comparison; no training or full sorting benchmark.')
    child = None

    def status():
        save(status_path, dict(report, child_pid=child.pid if child and child.poll() is None else None,
                               time=time.strftime('%Y-%m-%d %H:%M:%S')))

    def check_sources():
        for path, sha in pinned.items():
            assert digest(Path(path)) == sha, f'Pinned source changed: {path}'

    def stop(sig, frame):
        raise RuntimeError(f'Signal {sig}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        while True:
            download = json.loads(download_path.read_text())
            assert (download['pid'], download['birth']) == identity
            assert download['state'] != 'failed', download.get('error')
            if download['state'] == 'complete':
                break
            process = psutil.Process(identity[0])
            assert abs(process.create_time()-identity[1]) < .001
            assert process.status() != psutil.STATUS_ZOMBIE
            status()
            time.sleep(10)
        check_sources()
        assert {r['repo']: r['revision'] for r in download['completed']} == {
            'StarVLA/Qwen3-VL-OFT-RoboTwin2-All': '9fae49581755f57944ec57f6787d214c0c0143f5',
            'Qwen/Qwen3-VL-4B-Instruct': 'ebb281ec70b05090aa6165b016eac8ec08e71b17'}
        report['state'] = 'verifying_assets'
        status()
        for asset in download['completed']:
            for file in asset['files']:
                assert Path(file['path']).stat().st_size == file['bytes']
                assert digest(Path(file['path'])) == file['sha256']
        original = OFT/'checkpoints/steps_140000_pytorch_model.pt'
        assert digest(original) == WEIGHT_SHA
        # Derived config is kept beside a shared-storage hard link to the same
        # weight inode; the published source config and checkpoint stay intact.
        deployment = OFT/'deployments/starvla_grasp_precision_20260909'
        (deployment/'checkpoints').mkdir(parents=True, exist_ok=False)
        checkpoint = deployment/'checkpoints/steps_140000_pytorch_model.pt'
        os.link(original, checkpoint)
        assert original.stat().st_ino == checkpoint.stat().st_ino
        cfg = yaml.safe_load((OFT/'config.yaml').read_text())
        cfg['framework']['qwenvl']['base_vlm'] = str(BASE)
        (deployment/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        (deployment/'dataset_statistics.json').write_bytes((OFT/'dataset_statistics.json').read_bytes())
        (OUT/'deployment').symlink_to(deployment, target_is_directory=True)
        save(OUT/'deployment_manifest.json', dict(
            checkpoint=str(checkpoint), checkpoint_sha256=WEIGHT_SHA,
            original_config_sha256=digest(OFT/'config.yaml'),
            deployment_config_sha256=digest(deployment/'config.yaml'),
            normalization_sha256=digest(deployment/'dataset_statistics.json'),
            config_changes={'framework.qwenvl.base_vlm': str(BASE)},
            source_download_manifest=str(download_path), shared_weight_inode=checkpoint.stat().st_ino))
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE='1',
                   OMP_NUM_THREADS='4', NO_ALBUMENTATIONS_UPDATE='1')
        for horizon in report['execute_horizons']:
            port = report['ports'][str(horizon)]
            report.update(state='waiting_for_free_gpu', execute_horizon=horizon, port=port)
            while True:
                used = int(subprocess.check_output(['nvidia-smi', '-i', '6', '--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'], text=True).strip())
                if used < 500:
                    break
                report['gpu_used_mib'] = used
                status()
                time.sleep(10)
            check_sources()
            output = ROOT/f'playground/Checkpoints/oft_grasp_precision_h{horizon}_eval_20260909'
            report.update(state='auditing_then_evaluating', output=str(output))
            with (OUT/f'h{horizon}_supervisor.log').open('x') as log:
                child = subprocess.Popen([str(PYTHON), '-u',
                    str(ROOT/'examples/Robotwin/audits/run_oft_grasp_evaluation.py'),
                    '--checkpoint', str(checkpoint), '--reference',
                    str(ROOT/'playground/Checkpoints/gawm_grasp_precision_v2_reference_20260909'),
                    '--execute-horizon', str(horizon), '--gpu', '6', '--port', str(port),
                    '--output', str(output)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                status()
                time.sleep(10)
            assert child.returncode == 0, f'OFT h{horizon} failed; no replacement scenes'
            result = json.loads((output/'status.json').read_text())
            assert result['state'] == 'complete' and result['completed'] == 20
            report['completed'].append(dict(execute_horizon=horizon, output=str(output), summary=result['summary']))
        report['state'] = 'complete'
    except BaseException as error:
        report.update(state='failed', error=repr(error))
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
