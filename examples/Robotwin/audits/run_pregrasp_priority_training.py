"""Train the prepared Cartesian approach-priority ablation with pinned provenance."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from omegaconf import OmegaConf
import psutil

from examples.Robotwin.audits.run_grasp_lift_development import ROOT, save, digest


def main():
    out = ROOT/'playground/Checkpoints/gawm_grasp_precision_training_20260909'
    status_path = out/'cartesian_approach_priority_train1000_status.json'
    assert not status_path.exists()
    preparation_path = out/'approach_priority_preparation.json'
    preparation = json.loads(preparation_path.read_text())
    sampling_path = out/'approach_priority_sampling_audit.json'
    sampling = json.loads(sampling_path.read_text())
    assert sampling['state'] == 'configured_phase_exposure_measured'
    assert sampling['baseline_original_draws_exact'] and sampling['sample_draws'] == 10000
    assert json.loads((out/'gradient_fix_audit.json').read_text())['state'] == 'gradient_path_verified'
    record, = [r for r in preparation['configs'] if r['variant'] == 'cartesian']
    config_path, baseline_path = Path(record['path']), Path(record['base_path'])
    assert digest(config_path) == record['sha256'] and digest(baseline_path) == record['base_sha256']
    cfg, baseline = OmegaConf.load(config_path), OmegaConf.load(baseline_path)
    edited = OmegaConf.to_container(cfg, resolve=True)
    expected = OmegaConf.to_container(baseline, resolve=True)
    correction, = edited['datasets']['vla_data']['dataset_options']
    edited['run_id'] = expected['run_id']
    for key in ('spatial_supervision_dir', 'priority_sampling_probability'):
        old = expected['datasets']['vla_data']['dataset_options'][correction]
        new = edited['datasets']['vla_data']['dataset_options'][correction]
        if key in old:
            new[key] = old[key]
        else:
            new.pop(key)
    assert edited == expected, 'Changes exceed prepared sampling ablation'
    assert cfg.trainer.max_train_steps == 1000 and not cfg.trainer.is_resume
    output = ROOT/cfg.run_root_dir/cfg.run_id
    assert not output.exists()
    labels = Path(preparation['label_directory'])
    files = [config_path, baseline_path, preparation_path, sampling_path, Path(__file__),
        out/'split.json', out/'training_anchors.json', out/'gradient_fix_audit.json',
        ROOT/'starVLA/model/framework/WM4A/GAWM.py',
        ROOT/'starVLA/model/framework/WM4A/GAWMCartesian.py',
        ROOT/'starVLA/model/modules/action_model/CartesianResidualACT.py',
        ROOT/'starVLA/model/modules/robotwin_pose_kinematics.py',
        ROOT/'starVLA/dataloader/training_anchor_bounds.py',
        ROOT/'starVLA/dataloader/lerobot_datasets.py',
        ROOT/'starVLA/dataloader/gr00t_lerobot/datasets.py',
        ROOT/'starVLA/training/train_starvla.py']
    for r in preparation['records']:
        path = labels/f"episode_{r['episode']:06d}.npz"
        assert digest(path) == r['output_sha256']
        files.append(path)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 29884))
    command = ['/data/gaoxiang/Code/.venvs/starVLA/bin/accelerate', 'launch',
        '--config_file', 'starVLA/config/deepseeds/deepspeed_zero2.yaml',
        '--num_processes', '1', '--main_process_port', '29884',
        'starVLA/training/train_starvla.py', '--config_yaml', str(config_path)]
    report = dict(state='starting', variant='cartesian_approach_priority', gpu='4',
        output=str(output), supervisor_pid=os.getpid(), supervisor_birth=psutil.Process().create_time(),
        source_sha256={str(p):digest(p) for p in files}, command=command, steps_requested=1000,
        checkpoint_sha256=digest(ROOT/cfg.trainer.pretrained_checkpoint),
        normalization_sha256=digest(cfg.datasets.vla_data.normalization_statistics_path),
        comparison='Same v2 initialization and 1000-step Cartesian r2 budget; only correction anchor sampling changes.')
    save(status_path, report)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='4', PYTHONPATH=str(ROOT), PYTHONNOUSERSITE='1',
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', NO_ALBUMENTATIONS_UPDATE='1',
        WANDB_MODE='disabled', STARVLA_DISABLE_TQDM='1')
    child = None

    def stop(sig, frame):
        raise RuntimeError(f'Signal {sig}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        with (out/'cartesian_approach_priority_train1000_trainer.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        report.update(state='running', trainer_launcher_pid=child.pid,
                      trainer_launcher_birth=psutil.Process(child.pid).create_time())
        save(status_path, report)
        assert child.wait() == 0, 'Trainer exited unsuccessfully'
        rows = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == list(range(20, 1001, 20))
        checkpoint = output/'final_model/pytorch_model.pt'
        assert checkpoint.is_file() and (output/'validation_per_task.jsonl').is_file()
        report.update(state='trainer_completed_checkpoint_audit_pending', optimizer_steps=1000,
                      final_checkpoint=str(checkpoint))
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
        save(status_path, report)


if __name__ == '__main__':
    main()
