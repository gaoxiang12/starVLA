"""Train and screen matched recovery replay experiments after real data acceptance."""
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import yaml

from examples.Robotwin.audits.run_rgb_contact_refinement import check_training, digest, save

ROOT = Path(__file__).resolve().parents[3]
CP = ROOT / 'playground/Checkpoints'
CAMP = CP / 'gawm_rgb_recovery_training_20260908'
DATA = CP / 'gawm_rgb_recovery_train20_conversion_20260908'
PYTHON = ROOT.parent / '.venvs/starVLA/bin/python'


def main():
    CAMP.mkdir(exist_ok=False)
    configs = {name: ROOT / f'examples/Robotwin/train_files/starvla_gawm_rgb_recovery_{name}.yaml'
               for name in ('uniform', 'priority')}
    values = {name: yaml.safe_load(path.read_text()) for name, path in configs.items()}
    plan_path = DATA / 'converted_dataset.json'
    plan = json.loads(plan_path.read_text())
    loader = json.loads((DATA / 'loader_audit.json').read_text())
    assert plan['loader_audit_complete'] and loader['state'] == 'complete'
    assert len(plan['episodes']) == len(loader['episodes']) == 10
    assert all(row['priority_anchors_verified'] for row in loader['episodes'])
    assert plan['frames'] == loader['frames'] == 5387
    left, right = (copy.deepcopy(values[name]) for name in configs)
    for cfg in (left, right):
        cfg.pop('run_id')
    options = [next(iter(cfg['datasets']['vla_data']['dataset_options'].values())) for cfg in (left, right)]
    assert [opt.pop('priority_sampling_probability') for opt in options] == [0.0, 0.5]
    assert left == right, 'Recovery groups differ beyond priority replay'
    baseline = yaml.safe_load((ROOT / 'examples/Robotwin/train_files/starvla_gawm_rgb_refine_baseline.yaml').read_text())
    baseline.pop('run_id')
    reference = copy.deepcopy(left)
    reference['datasets']['vla_data'].pop('dataset_options')
    reference['datasets']['vla_data']['data_mix'] = baseline['datasets']['vla_data']['data_mix']
    assert reference == baseline, 'Non-mixture settings differ from the completed control'
    audits = {}
    for name, path in configs.items():
        audit = json.loads((DATA / f'mixture_{name}_audit.json').read_text())
        assert audit['state'] == 'complete' and audit['config_sha256'] == digest(path)
        assert audit['conversion_plan_sha256'] == digest(plan_path)
        assert audit['training_episode_counts'] == [950, 10] and audit['planned_samples'] == 64000
        assert len(audit['validation_episode_ids']) == 20
        assert set(audit['recovery_samples_by_episode']) == {str(i) for i in range(10)}
        audits[name] = audit
    assert audits['priority']['correction_samples'] > audits['uniform']['correction_samples']
    checkpoint = ROOT / values['uniform']['trainer']['pretrained_checkpoint']
    assert digest(checkpoint) == plan['policy_checkpoint_sha256']
    assignments = {'uniform': ('0', 29860, 6680), 'priority': ('5', 29865, 6685)}
    free_memory = {}
    for name, (gpu, train_port, eval_port) in assignments.items():
        free = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free',
                                            '--format=csv,noheader,nounits'], text=True).strip())
        assert free >= 18*1024, f'GPU{gpu} has only {free} MiB free'
        free_memory[gpu] = free
        assert not (CP / values[name]['run_id']).exists()
        for port in (train_port, eval_port):
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', port))
    save(CAMP / 'manifest.json', dict(supervisor_pid=os.getpid(), checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint), conversion_plan=str(plan_path),
        conversion_plan_sha256=digest(plan_path), configurations={name: dict(path=str(path),
            sha256=digest(path), config=values[name]) for name, path in configs.items()},
        assignments=assignments, free_mib_at_admission=free_memory,
        data_episodes=[950, 10], recovery_frames=5387, eval_episodes=10, eval_seed=0,
        note='2000-step matched v2 continuation. Both recovery groups retain event sampling 0.35; '
             'uniform means no additional phase priority. Existing no-recovery control already completed.'))
    active, jobs = {}, {}

    def status(state):
        save(CAMP / 'status.json', dict(state=state, supervisor_pid=os.getpid(),
            time=time.strftime('%Y-%m-%d %H:%M:%S'), jobs=jobs))

    def launch(name, stage):
        gpu, train_port, eval_port = assignments[name]
        run = CP / values[name]['run_id']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', PYTHONPATH=str(ROOT),
                   PYTHONNOUSERSITE='1', NO_ALBUMENTATIONS_UPDATE='1', WANDB_MODE='disabled')
        if stage == 'train':
            run.mkdir(exist_ok=False)
            command = [str(PYTHON.parent / 'accelerate'), 'launch', '--config_file',
                str(ROOT / 'starVLA/config/deepseeds/deepspeed_zero2.yaml'), '--num_processes', '1',
                '--main_process_port', str(train_port), str(ROOT / 'starVLA/training/train_starvla.py'),
                '--config_yaml', str(configs[name])]
        else:
            command = [str(PYTHON), str(ROOT / 'examples/Robotwin/audits/run_rgb_color_diagnostic.py'),
                '--gpu', gpu, '--port', str(eval_port), '--episodes', '10', '--orders', 'rgb',
                '--execute-horizon', '16', '--spatial-ablation', 'full',
                '--checkpoint', str(run / 'final_model/pytorch_model.pt'), '--output', str(run / 'screen10')]
        with (run / f'{stage}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[name] = process
        jobs[name] = dict(state='running', stage=stage, pid=process.pid, gpu=gpu, run=str(run))

    def stop(signum, frame):
        raise RuntimeError(f'Signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        for name in configs:
            launch(name, 'train')
        while active:
            for name, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[name]
                job = jobs[name]
                try:
                    assert code == 0, f'{job["stage"]} exited {code}'
                    run = Path(job['run'])
                    if job['stage'] == 'train':
                        check_training(run, 2000)
                        launch(name, 'eval')
                    else:
                        completed = json.loads((run / 'screen10/status.json').read_text())
                        assert completed['state'] == 'complete'
                        metrics = run / 'screen10/rgb/ranking_episode_metrics.jsonl'
                        rows = [json.loads(s) for s in metrics.read_text().splitlines()]
                        assert len(rows) == 10 and len({r['trial'] for r in rows}) == 10
                        job.update(state='complete', episodes=10, successes=sum(r['success'] for r in rows))
                except Exception as exc:
                    job.update(state='failed', error=repr(exc))
            status('running' if active else ('complete' if all(j['state']=='complete' for j in jobs.values())
                                             else 'complete_with_failures'))
            if active:
                time.sleep(15)
    except BaseException:
        for name, process in active.items():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            jobs[name].update(state='interrupted')
        status('failed')
        raise


if __name__ == '__main__':
    main()
