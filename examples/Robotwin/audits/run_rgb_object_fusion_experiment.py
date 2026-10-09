"""Supervise smoke validation, fresh 2000-step object fusion training, and evaluation."""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import psutil
import yaml

from examples.Robotwin.audits.run_rgb_contact_refinement import check_training, digest, live, save

ROOT = Path(__file__).resolve().parents[3]
CP = ROOT / 'playground/Checkpoints'
AUDITS = ROOT / 'examples/Robotwin/audits'
CONFIG = ROOT / 'examples/Robotwin/train_files/starvla_gawm_rgb_object_fusion.yaml'
SMOKE_CONFIG = CONFIG.with_name('starvla_gawm_rgb_object_fusion_smoke.yaml')
SMOKE = CP / 'gawm_rgb_object_fusion_smoke20_20260908'
RUN = CP / 'gawm_rgb_object_fusion_2k_20260908'
CAMP = CP / 'gawm_rgb_object_fusion_experiment_20260908'
PYTHON = ROOT.parent / '.venvs/starVLA/bin/python'


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    late = yaml.safe_load(CONFIG.with_name('starvla_gawm_rgb_object_readout.yaml').read_text())
    compare = copy.deepcopy(cfg)
    compare['run_id'] = late['run_id']
    compare['framework']['name'] = late['framework']['name']
    assert compare['framework']['spatial_focus'].pop('object_label_red_core') is True
    assert compare['framework']['spatial_focus'].pop('object_action_loss_weight') == .5
    assert compare == late, 'Unexpected protocol difference from late-object experiment'
    assert cfg['trainer']['max_train_steps'] == 2000
    assert cfg['framework']['spatial_focus']['teacher_end_steps'] == 0
    smoke_cfg = yaml.safe_load(SMOKE_CONFIG.read_text())
    smoke_cfg['run_id'] = cfg['run_id']
    for key in ('max_train_steps', 'gradient_accumulation_steps', 'num_warmup_steps',
                'save_interval', 'eval_interval', 'logging_frequency'):
        smoke_cfg['trainer'][key] = cfg['trainer'][key]
    assert smoke_cfg == cfg
    preflight = json.loads((CP/'gawm_rgb_object_fusion_preflight_20260908/audit.json').read_text())
    assert preflight['state'] == 'complete' and preflight['old_tensors_retained_exact'] == 577
    assert preflight['config_sha256'] == digest(CONFIG)
    assert preflight['training_labels_unreachable_in_inference'] and preflight['restored_full_exact']
    source = ROOT / cfg['trainer']['pretrained_checkpoint']
    assert digest(source) == preflight['source_sha256'] == 'cddf57200f2c26bc3c5a68a1c479987f94b78bac3603fda7b372123f965d7427'
    files = [CONFIG, SMOKE_CONFIG, ROOT/cfg['datasets']['vla_data']['episode_split_manifest'],
             ROOT/cfg['datasets']['vla_data']['normalization_statistics_path']]
    files += [ROOT/path for path in (
        'starVLA/model/framework/WM4A/GAWM.py', 'starVLA/model/framework/WM4A/GAWMObjectFusion.py',
        'starVLA/model/modules/spatial_focus.py', 'starVLA/model/modules/rgb_object_readout.py',
        'starVLA/model/modules/object_memory_fusion.py', 'starVLA/model/modules/rgb_object_core_loss.py',
        'starVLA/dataloader/rgb_object_supervision.py', 'starVLA/dataloader/rgb_object_supervision_core.py',
        'starVLA/model/modules/world_model/__init__.py', 'starVLA/training/train_starvla.py',
        'examples/Robotwin/audits/spatial_ablation.py', 'examples/Robotwin/audits/analyze_rgb_focus_usage.py')]
    hashes = {str(path):digest(path) for path in files}
    launch = json.loads((SMOKE/'launch.json').read_text())
    assert launch['config_sha256'] == digest(SMOKE_CONFIG)
    try:
        smoke_process = psutil.Process(launch['pid'])
        assert str(SMOKE_CONFIG) in ' '.join(smoke_process.cmdline())
        smoke_handle = dict(pid=smoke_process.pid, birth=smoke_process.create_time())
    except psutil.NoSuchProcess:
        check_training(SMOKE, 20)
        smoke_process, smoke_handle = None, None
    CAMP.mkdir(exist_ok=False)
    save(CAMP/'manifest.json', dict(supervisor_pid=os.getpid(), gpu='6', source=str(source),
        source_sha256=preflight['source_sha256'], config=cfg, pinned_files_sha256=hashes,
        smoke_handle=smoke_handle, added_parameters=preflight['added_parameters'],
        note='Same source,950 train/20 scene-safe val,2000 steps and optimizer as late-object model. '
             'Changes jointly: inside-decoder object memory, shared object-only action loss(.5), '
             'red-core label safeguard. This comparison does not isolate those three contributions. '
             'Formal weights start fresh from v2 source, never from smoke. Ten same-seed full RGB episodes.'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='6', PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4',
               NO_ALBUMENTATIONS_UPDATE='1', PYTHONNOUSERSITE='1', WANDB_MODE='disabled')
    active, stage = None, 'waiting_for_smoke'

    def status(state, **extra):
        save(CAMP/'status.json', dict(state=state, stage=stage, supervisor_pid=os.getpid(),
            pid=active.pid if active and active.poll() is None else None,
            time=time.strftime('%Y-%m-%d %H:%M:%S'), **extra))

    def execute(command, log, minimum_mib=12*1024):
        nonlocal active
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', '6', '--query-gpu=memory.free',
                                               '--format=csv,noheader,nounits'], text=True).strip())
            if free >= minimum_mib:
                break
            status('waiting_for_gpu_memory', free_mib=free, required_mib=minimum_mib)
            time.sleep(15)
        assert all(digest(Path(path)) == expected for path, expected in hashes.items()), 'Pinned inputs changed'
        with log.open('x') as stream:
            active = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while active.poll() is None:
            status('running')
            time.sleep(15)
        if active.returncode:
            raise RuntimeError(f'{stage} exited {active.returncode}')

    try:
        while smoke_process is not None and live(smoke_process):
            status('waiting_for_smoke', smoke_handle=smoke_handle)
            time.sleep(15)
        check_training(SMOKE, 20)
        rows = [json.loads(line) for line in (SMOKE/'metrics.jsonl').read_text().splitlines()]
        assert all('object_action_l1' in row for row in rows)
        stage = 'smoke_validation'
        execute([str(PYTHON), str(AUDITS/'analyze_rgb_focus_usage.py'), '--run', str(SMOKE),
            '--checkpoint', str(SMOKE/'final_model/pytorch_model.pt'), '--samples', '128',
            '--split-manifest', str(AUDITS/'rgb_scene_safe_validation_20260907.json'),
            '--output', str(CAMP/'smoke_usage.json')], CAMP/'smoke_usage.log')
        scores = json.loads((CAMP/'smoke_usage.json').read_text())['scores']
        assert scores['full']['action_l1'] < .02, 'Smoke policy catastrophic-damage guard, not an efficacy test'
        assert scores['no_objects']['mean_change_from_full'] > 1e-6
        stage = 'train'
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 29887))
        RUN.mkdir(exist_ok=False)
        execute([str(PYTHON.parent/'accelerate'), 'launch', '--config_file',
            str(ROOT/'starVLA/config/deepseeds/deepspeed_zero2.yaml'), '--num_processes', '1',
            '--main_process_port', '29887', str(ROOT/'starVLA/training/train_starvla.py'),
            '--config_yaml', str(CONFIG)], RUN/'train.log', minimum_mib=16*1024)
        check_training(RUN, 2000)
        stage = 'eval'
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 6696))
        execute([str(PYTHON), str(AUDITS/'run_rgb_color_diagnostic.py'), '--gpu', '6', '--port', '6696',
            '--episodes', '10', '--orders', 'rgb', '--execute-horizon', '16',
            '--checkpoint', str(RUN/'final_model/pytorch_model.pt'), '--output', str(RUN/'screen10')], RUN/'eval.log')
        assert json.loads((RUN/'screen10/status.json').read_text())['state'] == 'complete'
        status('complete')
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
