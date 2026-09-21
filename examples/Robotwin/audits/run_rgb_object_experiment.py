"""Finish the live object-branch smoke, validate gradients, then train and screen."""
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import psutil
import yaml

from examples.Robotwin.audits.run_rgb_contact_refinement import check_training, digest, live, save

ROOT = Path(__file__).resolve().parents[3]
CP = ROOT / 'playground/Checkpoints'
CAMP = CP / 'gawm_rgb_object_experiment_20260908'
SMOKE = CP / 'gawm_rgb_object_smoke20_gpu0_20260908'
PREFLIGHT = CP / 'gawm_rgb_object_preflight_20260908'
RUN = CP / 'gawm_rgb_object_readout_2k_20260908'
AUDITS = ROOT / 'examples/Robotwin/audits'
CONFIG = ROOT / 'examples/Robotwin/train_files/starvla_gawm_rgb_object_readout.yaml'
PYTHON = ROOT.parent / '.venvs/starVLA/bin/python'


def main():
    CAMP.mkdir(exist_ok=False)
    cfg = yaml.safe_load(CONFIG.read_text())
    baseline = yaml.safe_load((CONFIG.parent / 'starvla_gawm_rgb_refine_baseline.yaml').read_text())
    comparison = copy.deepcopy(cfg)
    baseline.pop('run_id'); comparison.pop('run_id')
    focus = comparison['framework']['spatial_focus']
    additions = {key: focus.pop(key) for key in ('object_readout', 'object_heatmap_loss_weight',
                                                'object_coordinate_loss_weight', 'object_visibility_loss_weight')}
    assert additions['object_readout'] and comparison == baseline
    smoke_cfg = yaml.safe_load((CONFIG.parent / 'starvla_gawm_rgb_object_smoke_gpu0.yaml').read_text())
    smoke_comparison = copy.deepcopy(smoke_cfg)
    smoke_comparison['run_id'] = cfg['run_id']
    for key in ('max_train_steps', 'gradient_accumulation_steps', 'num_warmup_steps',
                'save_interval', 'eval_interval', 'logging_frequency'):
        smoke_comparison['trainer'][key] = cfg['trainer'][key]
    assert smoke_comparison == cfg
    source = ROOT / cfg['trainer']['pretrained_checkpoint']
    assert digest(source) == 'cddf57200f2c26bc3c5a68a1c479987f94b78bac3603fda7b372123f965d7427'
    warmstart = json.loads((PREFLIGHT / 'warmstart.json').read_text())
    assert warmstart['passed'] and warmstart['branch'] == 'objects'
    assert warmstart['original_tensors_preserved'] == 577 and warmstart['initial_max_action_change'] == 0
    files = [CONFIG, ROOT / 'starVLA/model/framework/WM4A/GAWM.py',
             ROOT / 'starVLA/model/modules/spatial_focus.py', ROOT / 'starVLA/model/modules/rgb_object_readout.py',
             ROOT / 'starVLA/dataloader/rgb_object_supervision.py',
             ROOT / 'starVLA/training/train_starvla.py',
             ROOT / cfg['datasets']['vla_data']['episode_split_manifest'],
             ROOT / cfg['datasets']['vla_data']['normalization_statistics_path']]
    hashes = {str(path): digest(path) for path in files}
    launch = json.loads((SMOKE / 'launch.json').read_text())
    try:
        smoke_process = psutil.Process(launch['pid'])
        assert 'starvla_gawm_rgb_object_smoke_gpu0.yaml' in ' '.join(smoke_process.cmdline())
        smoke_handle = dict(pid=smoke_process.pid, birth=smoke_process.create_time())
    except psutil.NoSuchProcess:
        check_training(SMOKE, 20)
        smoke_process, smoke_handle = None, None
    save(CAMP / 'manifest.json', dict(supervisor_pid=os.getpid(), gpu='0', smoke=launch,
        smoke_handle=smoke_handle, source=str(source), source_sha256=digest(source), config=cfg,
        pinned_files_sha256=hashes, object_parameters=warmstart['added_parameters'],
        eval_episodes=10, eval_seed=0, eval_execute_horizon=16,
        note='Same original 950 training / 20 scene-safe validation episodes as baseline; no recovery mixture. '
             'Formal 2000 steps start from original v2 final5000, not smoke weights. '
             'Only object branch plus its image-derived auxiliary objectives differ from baseline.'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4',
               NO_ALBUMENTATIONS_UPDATE='1', PYTHONNOUSERSITE='1', WANDB_MODE='disabled')
    active, stage = None, 'waiting_for_existing_smoke'

    def status(state, **extra):
        save(CAMP / 'status.json', dict(state=state, stage=stage, supervisor_pid=os.getpid(),
             pid=active.pid if active and active.poll() is None else None,
             time=time.strftime('%Y-%m-%d %H:%M:%S'), **extra))

    def admission(minimum_mib):
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
                                                '--format=csv,noheader,nounits'], text=True).strip())
            if free >= minimum_mib:
                return
            status('waiting_for_gpu_memory', free_mib=free, required_mib=minimum_mib)
            time.sleep(15)

    def execute(command, log, minimum_mib=12*1024):
        nonlocal active
        admission(minimum_mib)
        assert all(digest(Path(path)) == expected for path, expected in hashes.items()), 'Pinned inputs changed'
        with log.open('x') as stream:
            active = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                      stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while active.poll() is None:
            status('running')
            time.sleep(15)
        if active.returncode:
            raise RuntimeError(f'{stage} exited {active.returncode}')

    def stop(signum, frame):
        raise RuntimeError(f'Signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        while smoke_process is not None and live(smoke_process):
            status('waiting_for_existing_smoke', smoke_handle=smoke_handle)
            time.sleep(15)
        check_training(SMOKE, 20)
        stage = 'smoke_action_gradients'
        execute([str(PYTHON), str(AUDITS / 'audit_rgb_goal_gradients.py'), '--run', str(SMOKE),
                 '--branch', 'objects', '--output', str(CAMP / 'smoke_gradients.json')], CAMP / 'smoke_gradients.log')
        gradients = json.loads((CAMP / 'smoke_gradients.json').read_text())
        assert gradients['passed']
        assert any(row['modes']['full']['object_supervision']['object_label_fraction'] > 0
                   for row in gradients['batches'])
        stage = 'smoke_validation'
        execute([str(PYTHON), str(AUDITS / 'analyze_rgb_focus_usage.py'), '--run', str(SMOKE),
                 '--checkpoint', str(SMOKE / 'final_model/pytorch_model.pt'), '--samples', '128',
                 '--split-manifest', str(AUDITS / 'rgb_scene_safe_validation_20260907.json'),
                 '--output', str(CAMP / 'smoke_usage.json')], CAMP / 'smoke_usage.log')
        scores = json.loads((CAMP / 'smoke_usage.json').read_text())['scores']
        assert scores['full']['action_l1'] < .02 and scores['no_objects']['mean_change_from_full'] > 1e-6
        stage = 'train'
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 29875))
        RUN.mkdir(exist_ok=False)
        execute([str(PYTHON.parent / 'accelerate'), 'launch', '--config_file',
                 str(ROOT / 'starVLA/config/deepseeds/deepspeed_zero2.yaml'), '--num_processes', '1',
                 '--main_process_port', '29875', str(ROOT / 'starVLA/training/train_starvla.py'),
                 '--config_yaml', str(CONFIG)], RUN / 'train.log', minimum_mib=16*1024)
        check_training(RUN, 2000)
        formal_rows = [json.loads(line) for line in (RUN / 'metrics.jsonl').read_text().splitlines()]
        assert all('object_heatmap_loss' in row for row in formal_rows)
        stage = 'eval'
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 6690))
        execute([str(PYTHON), str(AUDITS / 'run_rgb_color_diagnostic.py'), '--gpu', '0', '--port', '6690',
                 '--episodes', '10', '--orders', 'rgb', '--execute-horizon', '16', '--spatial-ablation', 'full',
                 '--checkpoint', str(RUN / 'final_model/pytorch_model.pt'), '--output', str(RUN / 'screen10')],
                RUN / 'eval.log')
        assert json.loads((RUN / 'screen10/status.json').read_text())['state'] == 'complete'
        stage = 'comparison'
        subprocess.run([str(PYTHON), str(AUDITS / 'summarize_rgb_matched_outcomes.py'),
            '--group', f'baseline={CP}/gawm_rgb_focus_refine_baseline_2k_20260907/screen10/rgb/ranking_episode_metrics.jsonl',
            '--group', f'objects={RUN}/screen10/rgb/ranking_episode_metrics.jsonl',
            '--output', str(CAMP / 'closed_loop_comparison.json')], cwd=ROOT, env=env, check=True)
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
