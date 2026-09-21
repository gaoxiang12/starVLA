"""Run a short real mixed-data training smoke after oracle GPU cleanup."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil
import torch
import yaml

from examples.Robotwin.audits.run_rgb_contact_refinement import check_training, digest, save

ROOT = Path(__file__).resolve().parents[3]
CP = ROOT / 'playground/Checkpoints'
CAMP = CP / 'gawm_rgb_recovery_mix_smoke_campaign_20260908'
RUN = CP / 'gawm_rgb_recovery_mix_smoke20_20260908'
CONFIG = ROOT / 'examples/Robotwin/train_files/starvla_gawm_rgb_recovery_mix_smoke.yaml'
PYTHON = ROOT.parent / '.venvs/starVLA/bin/python'


def main():
    CAMP.mkdir(exist_ok=False)
    assert not RUN.exists()
    source = json.loads((CP / 'gawm_rgb_recovery_tabletop_pilot_20260908/smoke_input_audit.json').read_text())
    assert source['state'] == 'complete' and source['recovery_samples'] >= 2
    oracle_status = CP / 'gawm_rgb_expert_action_replay_20260908/status.json'
    oracle = json.loads(oracle_status.read_text())
    handle = psutil.Process(oracle['pid'])
    assert handle.is_running() and handle.status() != psutil.STATUS_ZOMBIE
    cfg = yaml.safe_load(CONFIG.read_text())
    checkpoint = ROOT / cfg['trainer']['pretrained_checkpoint']
    save(CAMP / 'manifest.json', dict(pid=os.getpid(), config=str(CONFIG), config_sha256=digest(CONFIG),
        source_checkpoint=str(checkpoint), source_sha256=digest(checkpoint), gpu='0',
        oracle_pid=handle.pid, oracle_create_time=handle.create_time(),
        note='20-step integration smoke, not a matched performance experiment or a final recovery recipe.'))
    active = None
    stage = 'waiting_for_oracle'

    def status(state, **extra):
        save(CAMP / 'status.json', dict(state=state, stage=stage, pid=os.getpid(),
             child_pid=active.pid if active else None, time=time.strftime('%Y-%m-%d %H:%M:%S'), **extra))

    def stop(signum, frame):
        raise RuntimeError(f'Signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4',
               NO_ALBUMENTATIONS_UPDATE='1', PYTHONNOUSERSITE='1', WANDB_MODE='disabled', PYTHONUNBUFFERED='1')

    def execute(command, log):
        nonlocal active
        with log.open('x') as stream:
            active = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while active.poll() is None:
            status('running')
            time.sleep(10)
        assert active.returncode == 0, f'{stage} exited {active.returncode}'

    try:
        while True:
            oracle = json.loads(oracle_status.read_text())
            live = handle.is_running() and handle.status() != psutil.STATUS_ZOMBIE
            if oracle['state'] == 'failed':
                raise RuntimeError(f'Oracle failed: {oracle.get("error")}')
            if not live:
                assert oracle['state'] == 'complete' and oracle['oracle_successes'] == 3
                break
            status('waiting')
            time.sleep(10)
        # Avoid racing the already-scheduled final validator and subsequent collection.
        latest = json.loads((CP / 'gawm_rgb_focus_refine_baseline_2k_20260907/metrics.jsonl').read_text().splitlines()[-1])['step']
        free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
                                           '--format=csv,noheader,nounits'], text=True).strip())
        if latest >= 1800 or free < 12*1024:
            status('deferred_without_training', baseline_step=latest, free_mib=free,
                   reason='No clear short GPU window before final validation; leave existing jobs unchanged.')
            return
        RUN.mkdir(exist_ok=False)
        stage = 'train_smoke'
        execute([str(PYTHON.parent / 'accelerate'), 'launch', '--config_file',
            str(ROOT / 'starVLA/config/deepseeds/deepspeed_zero2.yaml'), '--num_processes', '1',
            '--main_process_port', '29850', str(ROOT / 'starVLA/training/train_starvla.py'),
            '--config_yaml', str(CONFIG)], RUN / 'train.log')
        metrics = check_training(RUN, 20)
        before = torch.load(checkpoint, map_location='cpu', weights_only=True)
        after = torch.load(RUN / 'final_model/pytorch_model.pt', map_location='cpu', weights_only=True)
        updates = {}
        for prefix in ('action_models.aloha.', 'spatial_focus.', 'backbone.encoder.'):
            changed, maximum = 0, 0.0
            for key, value in after.items():
                if key.startswith(prefix) and value.is_floating_point():
                    difference = (value.float()-before[key].to(value.dtype).float()).abs()
                    changed += int((difference != 0).sum())
                    maximum = max(maximum, float(difference.max()))
            updates[prefix] = dict(changed_elements_after_matching_dtype=changed, max_absolute_change=maximum)
        assert updates['action_models.aloha.']['changed_elements_after_matching_dtype'] > 0
        save(CAMP / 'parameter_updates.json', updates)
        del before, after
        stage = 'heldout_inference'
        execute([str(PYTHON), str(ROOT / 'examples/Robotwin/audits/analyze_rgb_focus_usage.py'),
            '--run', str(RUN), '--checkpoint', str(RUN / 'final_model/pytorch_model.pt'),
            '--samples', '16', '--split-manifest', str(ROOT / 'examples/Robotwin/audits/rgb_scene_safe_validation_20260907.json'),
            '--output', str(CAMP / 'heldout_usage.json')], CAMP / 'heldout_usage.log')
        assert digest(RUN / 'final_model/pytorch_model.pt') != digest(checkpoint)
        status('complete', final_step=20, final_training_action_l1=metrics['l1_action_loss'],
               note='Training and inference integration complete; no task success-rate claim.')
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
