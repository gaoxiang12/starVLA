"""Run a bounded training stage and retain process identity and exit status."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import psutil
from omegaconf import OmegaConf

from examples.Robotwin.audits.prepare_grasp_precision_training import ROOT, OUT, digest


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['joint', 'cartesian', 'compact_flow', 'compact_regression'], required=True)
    parser.add_argument('--gpu', choices=['0', '4', '6'], required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--stage', choices=['smoke', 'train1000', 'train1000_r2'], default='smoke')
    args = parser.parse_args()
    if args.variant.startswith('compact_'):
        assert args.stage in ('smoke', 'train1000')
        integration = ROOT/'playground/Checkpoints/gawm_compact_expert_integration_smoke_20260909/report.json'
        assert json.loads(integration.read_text())['state'] == 'real_data_gpu_optimizer_reload_verified'
    audit = json.loads((OUT/'mixture_audit.json').read_text())
    assert audit['state'] == 'shared_mixture_verified'
    if args.stage.startswith('train1000'):
        if args.variant.startswith('compact_'):
            acceptance = OUT/'compact_trained_smoke_audit.json'
            assert json.loads(acceptance.read_text())['state'] == 'compact_real_trainer_and_deployment_verified'
        else:
            assert json.loads((OUT/'trained_smoke_audit.json').read_text())['state'] == 'real_trainer_smokes_verified'
    if args.stage == 'train1000_r2':
        assert json.loads((OUT/'gradient_fix_audit.json').read_text())['state'] == 'gradient_path_verified'
    config_path = OUT/f'{args.variant}_{args.stage}.yaml'
    cfg = OmegaConf.load(config_path)
    steps = 20 if args.stage == 'smoke' else 1000
    assert cfg.trainer.max_train_steps == steps and not cfg.trainer.is_resume
    output = ROOT/cfg.run_root_dir/cfg.run_id
    assert not output.exists(), 'Never append to an existing training attempt'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    sources = [config_path, OUT/'mixture_audit.json', OUT/'split.json', OUT/'training_anchors.json',
        ROOT/'starVLA/model/framework/WM4A/GAWMCartesian.py',
        ROOT/'starVLA/model/modules/action_model/CartesianResidualACT.py',
        ROOT/'starVLA/model/modules/robotwin_pose_kinematics.py',
        ROOT/'starVLA/dataloader/training_anchor_bounds.py',
        ROOT/'starVLA/dataloader/lerobot_datasets.py',
        ROOT/'starVLA/dataloader/gr00t_lerobot/datasets.py',
        ROOT/'starVLA/training/train_starvla.py']
    if args.variant.startswith('compact_'):
        sources += [integration, Path(__file__), ROOT/'starVLA/model/framework/WM4A/GAWM.py',
            ROOT/'starVLA/model/framework/WM4A/GAWMCompactExpert.py',
            ROOT/'starVLA/model/modules/action_model/CompactFlowActionHead.py',
            ROOT/'starVLA/model/modules/world_model/__init__.py']
        if args.stage == 'train1000':
            sources.append(acceptance)
    command = ['/data/gaoxiang/Code/.venvs/starVLA/bin/accelerate', 'launch',
        '--config_file', 'starVLA/config/deepseeds/deepspeed_zero2.yaml',
        '--num_processes', '1', '--main_process_port', str(args.port),
        'starVLA/training/train_starvla.py', '--config_yaml', str(config_path)]
    report = dict(state='starting', variant=args.variant, gpu=args.gpu, output=str(output),
        supervisor_pid=os.getpid(), supervisor_birth=psutil.Process().create_time(), command=command,
        source_sha256={str(p):digest(p) for p in sources},
        checkpoint_sha256=digest(ROOT/cfg.trainer.pretrained_checkpoint),
        normalization_sha256=digest(cfg.datasets.vla_data.normalization_statistics_path),
        steps_requested=steps, rollout_evaluation=False, stage=args.stage)
    status = OUT/f'{args.variant}_{args.stage}_status.json'
    save(status, report)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(ROOT),
        PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
        NO_ALBUMENTATIONS_UPDATE='1', WANDB_MODE='disabled', STARVLA_DISABLE_TQDM='1')
    child = None

    def stop(sig, frame):
        raise RuntimeError(f'Signal {sig}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        with (OUT/f'{args.variant}_{args.stage}_trainer.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        report.update(state='running', trainer_launcher_pid=child.pid,
                      trainer_launcher_birth=psutil.Process(child.pid).create_time())
        save(status, report)
        result = child.wait()
        report['exit_code'] = result
        if result != 0:
            raise RuntimeError(f'Trainer exited {result}')
        metrics = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
        frequency = cfg.trainer.logging_frequency
        assert [m['step'] for m in metrics] == list(range(frequency,steps+1,frequency))
        assert (output/'final_model/pytorch_model.pt').is_file()
        assert (output/'validation_per_task.jsonl').is_file()
        report.update(state='trainer_completed_checkpoint_audit_pending', optimizer_steps=steps,
                      final_checkpoint=str(output/'final_model/pytorch_model.pt'))
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        report['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save(status, report)


if __name__ == '__main__':
    main()
