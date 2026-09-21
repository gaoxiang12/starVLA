"""Matched v2 continuation with/without TCP and closing-transition supervision."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = ROOT / 'playground/Checkpoints'
CAMP = CHECKPOINTS / 'gawm_rgb_contact_refinement_20260907'
SOURCE = CHECKPOINTS / 'gawm_rgb_focus_local_v2_5k_20260907'
SMOKE = CHECKPOINTS / 'gawm_rgb_focus_contact_logged_smoke5_20260907'
PYTHON = ROOT.parent / '.venvs/starVLA/bin/python'


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def check_training(run, steps, contact=False):
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert rows and rows[-1]['step'] == steps, (run, 'incomplete training')
    assert all(math.isfinite(value) for row in rows for value in row.values()
               if isinstance(value, float)), (run, 'nonfinite metric')
    if contact:
        assert all('contact_objective_loss' in row for row in rows)
        assert any(row['contact_objective_loss'] > 0 for row in rows)
    assert (run / 'final_model/pytorch_model.pt').stat().st_size > 1_000_000
    return rows[-1]


def process_handle(pid, expected):
    process = psutil.Process(pid)
    assert expected in ' '.join(process.cmdline()), (pid, 'unexpected process')
    return process


def live(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def main():
    CAMP.mkdir(exist_ok=True)
    with (CAMP / 'claim.json').open('x') as stream:
        json.dump(dict(supervisor_pid=os.getpid(), time=time.time()), stream)
    configs = {}
    for name in ('baseline', 'contact'):
        path = ROOT / f'examples/Robotwin/train_files/starvla_gawm_rgb_refine_{name}.yaml'
        configs[name] = (path, yaml.safe_load(path.read_text()))
    left, right = (copy.deepcopy(configs[name][1]) for name in ('baseline', 'contact'))
    left.pop('run_id'); right.pop('run_id')
    objective = right['framework'].pop('contact_objective')
    assert left == right and objective['enabled'], 'Unmatched experimental configurations'
    assert left['framework']['spatial_focus']['teacher_end_steps'] == 0
    assert left['trainer']['max_train_steps'] == 2000
    dependencies = []
    source_status = json.loads((SOURCE / 'supervisor_status.json').read_text())
    if source_status.get('stage') == 'train' and source_status['state'] == 'running':
        try:
            dependencies.append(process_handle(source_status['pid'], 'starvla_gawm_rgb_focus_local_v2.yaml'))
        except psutil.NoSuchProcess:
            check_training(SOURCE, 5000)
    else:
        check_training(SOURCE, 5000)
    smoke_pid = json.loads((SMOKE / 'launch.json').read_text())['pid']
    try:
        dependencies.append(process_handle(smoke_pid, 'starvla_gawm_rgb_contact_logged_smoke.yaml'))
    except psutil.NoSuchProcess:
        check_training(SMOKE, 5, contact=True)
    jobs = {}
    active = {}

    def status(state, **extra):
        save(CAMP / 'status.json', dict(state=state, supervisor_pid=os.getpid(),
             time=time.strftime('%Y-%m-%d %H:%M:%S'), jobs=jobs, **extra))

    def stop(signum, frame):
        raise RuntimeError(f'Supervisor received signal {signum}')

    def launch(name, stage):
        path, config = configs[name]
        run = CHECKPOINTS / config['run_id']
        gpu = '4' if name == 'baseline' else '5'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(ROOT),
                   OMP_NUM_THREADS='4', NO_ALBUMENTATIONS_UPDATE='1', PYTHONNOUSERSITE='1',
                   WANDB_MODE='disabled', PYTHONUNBUFFERED='1')
        if stage == 'train':
            run.mkdir(exist_ok=False)
            command = [str(PYTHON.parent / 'accelerate'), 'launch', '--config_file',
                       str(ROOT / 'starVLA/config/deepseeds/deepspeed_zero2.yaml'),
                       '--num_processes', '1', '--main_process_port', '2983' + gpu,
                       str(ROOT / 'starVLA/training/train_starvla.py'), '--config_yaml', str(path)]
        else:
            command = [str(PYTHON), str(ROOT / 'examples/Robotwin/audits/run_rgb_color_diagnostic.py'),
                       '--gpu', gpu, '--port', '66' + gpu + '0', '--episodes', '10', '--orders', 'rgb',
                       '--execute-horizon', '16', '--spatial-ablation', 'full',
                       '--checkpoint', str(run / 'final_model/pytorch_model.pt'),
                       '--output', str(run / 'screen10')]
        with (run / f'{stage}.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        active[name] = process
        jobs[name] = dict(stage=stage, state='running', pid=process.pid, gpu=gpu, run=str(run))

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        while any(live(process) for process in dependencies):
            status('waiting_for_training', dependencies=[dict(pid=p.pid, birth=p.create_time())
                                                        for p in dependencies if live(p)])
            time.sleep(15)
        source_last = check_training(SOURCE, 5000)
        smoke_last = check_training(SMOKE, 5, contact=True)
        checkpoint = SOURCE / 'final_model/pytorch_model.pt'
        save(CAMP / 'manifest.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
             configurations={name: dict(path=str(path), sha256=digest(path), config=config)
                             for name, (path, config) in configs.items()},
             source_last_metrics=source_last, smoke_last_metrics=smoke_last,
             additional_steps=2000, seed=42, eval_seed=0, eval_episodes=10,
             note='Same final v2 weights, data, crops, LR restart and steps; only contact objective differs. '
                  'Screening result requires subsequent larger evaluation. Existing evaluations keep running.'))
        for name in configs:
            launch(name, 'train')
        while active:
            for name, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[name]
                job = jobs[name]
                run = Path(job['run'])
                if code != 0:
                    job.update(state='failed', returncode=code)
                    continue
                try:
                    if job['stage'] == 'train':
                        check_training(run, 2000, contact=name == 'contact')
                        launch(name, 'eval')
                    else:
                        result = json.loads((run / 'screen10/status.json').read_text())
                        assert result['state'] == 'complete'
                        metrics = [json.loads(line) for line in
                                   (run / 'screen10/rgb/ranking_episode_metrics.jsonl').read_text().splitlines()]
                        assert len(metrics) == 10
                        job.update(state='complete', episodes=10,
                                   successes=sum(row['success'] for row in metrics),
                                   lifted_episodes=sum(row['any_block_lifted'] for row in metrics))
                except Exception as exc:
                    job.update(state='failed', error=repr(exc))
            status('running' if active else ('complete' if all(j['state'] == 'complete'
                                                              for j in jobs.values()) else 'failed'))
            if active:
                time.sleep(15)
    except BaseException as exc:
        for process in active.values():
            if process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    descendants = parent.children(recursive=True)
                    parent.terminate()
                    for child in descendants:
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
