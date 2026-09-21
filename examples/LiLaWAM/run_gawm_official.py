"""Run experiment B from random policy initialization, with a fixed-step evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from omegaconf import OmegaConf
from examples.LiLaWAM.official_robotwin_data import write_json, digest

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path('/data/gaoxiang/ckpts/lila_starvla/lila_robotwin_official_ddp8_20260916/upstream_source')
BASELINE = SOURCE.parent / 'eval/step117000_blocks_ranking_rgb_clean_10ep_seed0/summary.json'
VENV = Path('/data/gaoxiang/Code/.venvs')


def snapshot(root):
    target = root / 'code_snapshot'
    if target.exists():
        manifest = json.loads((root / 'code_manifest.json').read_text())
        for name, sha in manifest.items():
            if digest(target / name) != sha:
                raise ValueError(f'Frozen source changed: {name}')
        return target
    target.mkdir()
    for folder in ('starVLA', 'deployment', 'examples'):
        for path in (REPO / folder).rglob('*'):
            if not path.is_file() or path.suffix not in {'.py', '.yaml', '.yml', '.json', '.sh'}:
                continue
            if '__pycache__' in path.parts or path.is_symlink():
                continue
            dest = target / path.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    manifest = {str(p.relative_to(target)): digest(p) for p in target.rglob('*') if p.is_file()}
    write_json(root / 'code_manifest.json', manifest)
    return target


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = snapshot(root)
    parity = code / 'examples/LiLaWAM/audits/gawm_official_B_data_parity_20260917.json'
    if json.loads(parity.read_text())['status'] != 'passed':
        raise ValueError('A/B data parity audit is required')
    policy_audit = json.loads((code/'examples/LiLaWAM/audits/gawm_official_B_policy_contract_20260917.json').read_text())
    if policy_audit['status'] != 'passed' or not policy_audit['dino_last_features_bitwise_equal']:
        raise ValueError('DINO parity and inference contract audit are required')
    configs = []
    for stage in (1, 2):
        cfg = OmegaConf.load(code / f'examples/LiLaWAM/train_files/gawm_official_stage{stage}.yaml')
        path = root / f'stage{stage}.yaml'
        if path.exists() and OmegaConf.load(path) != cfg:
            raise ValueError('Existing experiment config differs from frozen source')
        OmegaConf.save(cfg, path)
        configs.append(path)
    baseline_manifest = json.loads((SOURCE.parent/'stage1/provenance.json').read_text())
    assert baseline_manifest['batch_size'] == 128
    assert baseline_manifest['stats_sha256'] == digest(SOURCE/'utils/stat-500-all.json')
    manifest = dict(experiment='B: GAWM under the LiLa reproduction recipe', seed=42,
        policy_initialization='random; frozen official DINOv3-L only', dataset_episodes=27071,
        dataset_frames=6120962, tasks=50, train_data_mode='both (demo_clean and demo_randomized)',
        train_gpus=args.gpus, global_batch=128, steps_per_epoch=47820,
        stage1_steps=573840, stage2_steps=191280, stage2_optimizer='fresh AdamW',
        scheduler_horizon_epochs=40, milestone_step=117000,
        evaluation=dict(task='blocks_ranking_rgb', task_config='demo_clean', episodes=100,
                        seed=0, gpu=args.eval_gpu, video=True, baseline=str(BASELINE)),
        differences_from_A=['GAWM fixed 8x8 spatial pooling of final-layer DINO features',
            'GAWM native ACT/L1, residual world model, cosine/diversity/variance losses',
            'Native canonical task-text encoder instead of LiLa VTT',
            'Two future recorded offsets +16,+32 instead of only +32',
            'Trainable architecture/parameter count; same bf16 and Adam recipe'],
        matched=['same author HDF5 inventory and shuffle seed', 'head RGB 320x240 OpenCV linear + ImageNet',
            'same frozen pretrained DINOv3-L asset', 'actual endpose16, native action14, author minmax',
            'endpoint-clamped tail supervision', 'global batch128 and optimizer-update budget',
            'AdamW betas(.9,.99), decay.01, clip1, two-stage learning rates/schedulers',
            'action chunk32, execute16, identical B-spline smoothing and evaluation task settings'],
        dino_sha256=digest(Path(cfg.model.vision_encoder.checkpoint_path)/'model.safetensors'),
        stats_sha256=digest(SOURCE/'utils/stat-500-all.json'), data_parity_sha256=digest(parity),
        interpretation='B is an architecture/objective bundle under a matched recipe; failure alone does not prove intrinsic architectural failure.')
    write_json(root / 'experiment_manifest.json', manifest)
    env = dict(os.environ, HF_HUB_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled', PYTHONPATH=str(code))
    count = len(args.gpus.split(','))
    stop_event = threading.Event()

    def evaluate_milestone():
        status_path = root / 'eval_status.json'
        policy = root / 'stage1/policy_step_117000.pt'
        write_json(status_path, dict(status='waiting_for_checkpoint', checkpoint=str(policy), step=117000))
        try:
            while not policy.exists():
                if stop_event.wait(15):
                    write_json(status_path, dict(status='training_stopped_before_milestone'))
                    return
            evaluation = root / 'eval/step117000_blocks_ranking_rgb_clean_100ep_seed0'
            evaluation.parent.mkdir(exist_ok=True)
            if not (evaluation/'summary.json').exists():
                if evaluation.exists():
                    raise ValueError(f'Incomplete evaluation exists; inspect before retry: {evaluation}')
                settings = dict(source_root=str(SOURCE), config_path=str(configs[0]), checkpoint_path=str(policy),
                    norm_stats_path=str(SOURCE/'utils/stat-500-all.json'),
                    vision_encoder_path=cfg.model.vision_encoder.checkpoint_path)
                eval_config = root / 'eval_step117000.yaml'
                OmegaConf.save(OmegaConf.create({'framework': {'name': 'GAWMOfficial', 'official_robotwin': settings}}), eval_config)
                command = [sys.executable, '-m', 'examples.Robotwin.eval_files.run_lila_benchmark',
                    '--config', str(eval_config), '--output', str(evaluation), '--tasks', 'blocks_ranking_rgb',
                    '--episodes', '100', '--seed', '0', '--policy-seed', '0', '--gpus', args.eval_gpu,
                    '--base-port', '5834', '--python', str(VENV/'starVLA/bin/python'),
                    '--sim-python', str(VENV/'RoboTwin/bin/python'), '--robotwin', '/data/gaoxiang/Code/RoboTwin', '--video']
                with (root/'eval_step117000.log').open('a') as log:
                    child = subprocess.Popen(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
                    write_json(status_path, dict(status='evaluating', pid=child.pid, command=command))
                    if child.wait():
                        raise RuntimeError('Evaluation failed; inspect eval_step117000.log')
            b = json.loads((evaluation/'summary.json').read_text())
            if b['state'] != 'complete' or b['trials'] != 100 or not b['sources_unchanged']:
                raise RuntimeError('Incomplete or modified evaluation')
            comparison = dict(B_summary=b, A_summary_path=str(BASELINE), conclusion='Descriptive comparison; one training seed and limited A trials.')
            if BASELINE.exists():
                a = json.loads(BASELINE.read_text())
                ar = {ep['seed']: ep['success'] for ep in a['results'][0]['episodes']}
                br = {ep['seed']: ep['success'] for ep in b['results'][0]['episodes']}
                comparison['paired_seeds'] = [dict(seed=s, A=ar[s], B=br[s]) for s in sorted(ar.keys() & br.keys())]
                comparison['A_summary'] = a
            write_json(root/'comparison_step117000.json', comparison)
            write_json(status_path, dict(status='completed', summary=str(evaluation/'summary.json'),
                successes=b['successes'], trials=b['trials']))
        except Exception as exc:
            write_json(status_path, dict(status='failed', error=repr(exc)))

    def train(label, output, config, extra):
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={count}',
            '--module', 'examples.LiLaWAM.train_official_robotwin', '--config', str(config),
            '--output', str(output), '--source', str(SOURCE), '--stats', str(SOURCE/'utils/stat-500-all.json'),
            '--workers-per-rank', '2', *extra]
        with (root/f'{label}.log').open('a') as log:
            child = subprocess.Popen(command, cwd=code, env=dict(env, CUDA_VISIBLE_DEVICES=args.gpus), stdout=log, stderr=subprocess.STDOUT)
            write_json(root/'run_status.json', dict(status=label, supervisor_pid=os.getpid(), child_pid=child.pid,
                gpus=args.gpus, command=command, started=time.time()))
            if child.wait():
                raise RuntimeError(f'{label} failed; inspect {root}/{label}.log')

    watcher = None
    try:
        smoke = root / f'smoke_ddp{count}_frozen'
        if not (smoke/'smoke_result.json').exists():
            train('smoke', smoke, configs[0], ['--smoke-steps', '3'])
        if json.loads((smoke/'smoke_result.json').read_text())['status'] != 'passed':
            raise ValueError('Distributed smoke did not pass')
        watcher = threading.Thread(target=evaluate_milestone)
        watcher.start()
        for stage, epochs in ((1, 12), (2, 4)):
            output = root / f'stage{stage}'
            status = output/'status.json'
            if status.exists() and json.loads(status.read_text()).get('status') == 'completed':
                continue
            extra = ['--stop-epochs', str(epochs)]
            if (output/'latest.pt').exists():
                extra += ['--resume', str(output/'latest.pt')]
            elif stage == 2:
                extra += ['--init-from', str(root/'stage1/latest.pt')]
            train(f'stage{stage}', output, configs[stage-1], extra)
        watcher.join()
        write_json(root/'run_status.json', dict(status='training_completed', evaluation_status=str(root/'eval_status.json')))
    finally:
        stop_event.set()
        if watcher is not None:
            watcher.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--eval-gpu', default='7')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        write_json(args.output/'run_status.json', dict(status='failed', error=repr(exc)))
        raise
