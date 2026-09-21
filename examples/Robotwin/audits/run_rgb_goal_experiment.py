"""Gate an explicit goal readout experiment on real-input and training checks."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml

from examples.Robotwin.audits.run_rgb_contact_refinement import check_training, digest, save

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = ROOT / 'playground/Checkpoints'
CAMP = CHECKPOINTS / 'gawm_rgb_goal_readout_20260907'
SMOKE = CHECKPOINTS / 'gawm_rgb_focus_goal_smoke20_20260907'
RUN = CHECKPOINTS / 'gawm_rgb_focus_refine_goal_2k_20260907'
PYTHON = ROOT.parent / '.venvs/starVLA/bin/python'
AUDITS = ROOT / 'examples/Robotwin/audits'
CONFIGS = ROOT / 'examples/Robotwin/train_files'


def main():
    CAMP.mkdir(exist_ok=True)
    with (CAMP / 'claim.json').open('x') as stream:
        json.dump(dict(supervisor_pid=os.getpid()), stream)
    baseline = yaml.safe_load((CONFIGS / 'starvla_gawm_rgb_refine_baseline.yaml').read_text())
    config = yaml.safe_load((CONFIGS / 'starvla_gawm_rgb_refine_goal.yaml').read_text())
    comparison = copy.deepcopy(config)
    baseline.pop('run_id'); comparison.pop('run_id')
    assert comparison['framework']['spatial_focus'].pop('goal_readout') is True
    assert baseline == comparison, 'Goal experiment must match the baseline except for its added readout'
    checkpoint = ROOT / config['trainer']['pretrained_checkpoint']
    check_training(checkpoint.parent.parent, 5000)
    pair = json.loads((CHECKPOINTS / 'gawm_rgb_contact_refinement_20260907/manifest.json').read_text())
    assert digest(checkpoint) == pair['checkpoint_sha256']
    save(CAMP / 'manifest.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=pair['checkpoint_sha256'],
         config=config, gpu='6', eval_seed=0, eval_episodes=10,
         note='Third matched continuation arm. Same v2 final weights and training protocol as baseline/contact; '
              'only the explicit predicted goal readout is added, contact objective disabled.'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='6', PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4',
               NO_ALBUMENTATIONS_UPDATE='1', PYTHONNOUSERSITE='1', WANDB_MODE='disabled', PYTHONUNBUFFERED='1')
    active = None
    stage = 'preflight'

    def status(state, **extra):
        save(CAMP / 'status.json', dict(state=state, stage=stage, supervisor_pid=os.getpid(),
             pid=active.pid if active else None, time=time.strftime('%Y-%m-%d %H:%M:%S'), **extra))

    def execute(command, log):
        nonlocal active
        with log.open('x') as stream:
            active = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while active.poll() is None:
            status('running')
            time.sleep(15)
        if active.returncode:
            raise RuntimeError(f'{stage} exited {active.returncode}')

    def train(config_name):
        return [str(PYTHON.parent / 'accelerate'), 'launch', '--config_file',
                str(ROOT / 'starVLA/config/deepseeds/deepspeed_zero2.yaml'), '--num_processes', '1',
                '--main_process_port', '29846', str(ROOT / 'starVLA/training/train_starvla.py'),
                '--config_yaml', str(CONFIGS / config_name)]

    def stop(signum, frame):
        raise RuntimeError(f'Supervisor received signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        execute([str(PYTHON), str(AUDITS / 'check_rgb_goal_warmstart.py'), '--config',
                 str(CONFIGS / 'starvla_gawm_rgb_goal_smoke.yaml'), '--output', str(CAMP / 'warmstart.json')],
                CAMP / 'warmstart.log')
        assert json.loads((CAMP / 'warmstart.json').read_text())['passed']
        stage = 'smoke'
        SMOKE.mkdir(exist_ok=False)
        execute(train('starvla_gawm_rgb_goal_smoke.yaml'), SMOKE / 'train.log')
        check_training(SMOKE, 20)
        stage = 'smoke_ablation'
        execute([str(PYTHON), str(AUDITS / 'analyze_rgb_focus_usage.py'), '--run', str(SMOKE),
                 '--checkpoint', str(SMOKE / 'final_model/pytorch_model.pt'), '--samples', '64',
                 '--split-manifest', str(AUDITS / 'rgb_scene_safe_validation_20260907.json'),
                 '--output', str(CAMP / 'smoke_usage.json')], CAMP / 'smoke_usage.log')
        scores = json.loads((CAMP / 'smoke_usage.json').read_text())['scores']
        assert scores['no_goal']['mean_change_from_full'] > 1e-5, 'Goal path did not become active'
        assert scores['full']['action_l1'] < .02, 'Smoke action error is excessive'
        stage = 'train'
        RUN.mkdir(exist_ok=False)
        execute(train('starvla_gawm_rgb_refine_goal.yaml'), RUN / 'train.log')
        check_training(RUN, 2000)
        stage = 'eval'
        execute([str(PYTHON), str(AUDITS / 'run_rgb_color_diagnostic.py'), '--gpu', '6', '--port', '6660',
                 '--episodes', '10', '--orders', 'rgb', '--execute-horizon', '16',
                 '--checkpoint', str(RUN / 'final_model/pytorch_model.pt'), '--output', str(RUN / 'screen10')],
                RUN / 'eval.log')
        assert json.loads((RUN / 'screen10/status.json').read_text())['state'] == 'complete'
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
