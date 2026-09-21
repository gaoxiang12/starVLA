"""Wait for audited author data, smoke-test, then train the official two-stage recipe."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf
from examples.LiLaWAM.official_robotwin_data import digest, write_json, expected_episode_count

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path('/data/gaoxiang/Code/LiLa-WAM')


def status(root, phase, **kwargs):
    value = {'status': phase, 'pid':os.getpid(), 'updated':time.time(), **kwargs}
    write_json(root / 'run_status.json', value)
    print(json.dumps(value), flush=True)


def choose_gpu(root, gpus, count=1):
    while True:
        r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                           check=True, capture_output=True, text=True)
        values = {p[0].strip():tuple(int(x) for x in p[1:]) for p in (line.split(',') for line in r.stdout.splitlines())}
        used = {gpu:row[0] for gpu,row in values.items()}
        available = [gpu for gpu in gpus if gpu in values and
                     (values[gpu][0] < 100 if count == 1 else values[gpu][1]-values[gpu][0] > 16000 and values[gpu][2] < 10)]
        if len(available) >= count:
            return ','.join(available[:count])
        status(root, 'waiting_for_free_gpu', candidates=gpus, memory_used_mb=used)
        time.sleep(30)


def execute(root, label, args, gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', WANDB_MODE='disabled')
    count = len(gpu.split(','))
    launcher = ['-m','torch.distributed.run','--standalone',f'--nproc_per_node={count}',
                '--module','examples.LiLaWAM.train_official_robotwin'] if count > 1 else ['-m','examples.LiLaWAM.train_official_robotwin']
    command = [sys.executable, '-u', *launcher, *map(str,args)]
    with (root / f'{label}.log').open('a') as log:
        p = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        status(root, label, child_pid=p.pid, gpu=gpu, world_size=count, command=command)
        code = p.wait()
    if code:
        raise RuntimeError(f'{label} exited {code}; inspect {root / (label + ".log")}')


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # A lock prevents accidentally launching duplicate supervisors for one run.
    import fcntl
    lock = (root / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = root / 'upstream_source'
    if not source.exists():
        source.mkdir()
        for folder in ('models', 'dataloader', 'utils', 'data-500-taskcond'):
            shutil.copytree(SOURCE / folder, source / folder, ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('README.md', 'train.py'):
            shutil.copy2(SOURCE / name, source / name)
    configs = []
    for stage in (1,2):
        c = OmegaConf.load(REPO / f'examples/LiLaWAM/train_files/robotwin_official_stage{stage}.yaml')
        c.dataset.dataset_dir = str(args.data / 'data')
        c.dataset.task_cond_dir = str(source / 'data-500-taskcond')
        path = root / f'stage{stage}.yaml'
        OmegaConf.save(c, path)
        configs.append(path)
    manifest = {'upstream_commit':subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        'data_root':str(args.data), 'seed':42, 'stage1_epochs':12, 'stage2_epochs':4,
        'stage1_lr':[2e-4,5e-5], 'stage2_lr':[4e-5,1e-5], 'scheduler_horizon_epochs_each_stage':40,
        'stage2_min_lr_note':'Explicit reproduction choice: README does not specify a valid lower bound; 1e-5 avoids increasing LR.',
        'initialization':'Frozen pretrained DINOv3 only; all policy/future-decoder parameters random at stage 1.',
        'preprocessing':'Author processed HDF5; author-supplied normalization statistics and per-task VTT, copied and hashed.',
        'padding':'Preserve upstream endpoint clamping and upstream loss behavior, including tail repeats.',
        'evaluation':'Not automatically restarted; training completion does not establish 90% success.',
        'source_sha256':{str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()},
        'runner_sha256':{p.name:digest(p) for p in (Path(__file__),REPO/'examples/LiLaWAM/train_official_robotwin.py')},
        'dino_weights_sha256':digest(Path(OmegaConf.load(configs[0]).model.vision_encoder.checkpoint_path)/'model.safetensors')}
    write_json(root / 'reproduction_manifest.json', manifest)
    while not (args.data / 'dataset_audit.json').exists():
        state = args.data / 'prepare_process.json'
        if state.exists():
            pid = json.loads(state.read_text())['pid']
            proc = Path(f'/proc/{pid}')
            if not proc.exists() or 'official_robotwin_data' not in (proc/'cmdline').read_bytes().decode(errors='replace'):
                raise RuntimeError(f'Data preparation stopped before audit completed: {args.data / "prepare.log"}')
        status(root, 'waiting_for_data', preparation_log=str(args.data/'prepare.log'))
        time.sleep(30)
    audit = json.loads((args.data/'dataset_audit.json').read_text())
    report = source / 'utils/outlier_files 500-all.txt'
    if (audit.get('status') != 'passed' or audit.get('episodes') != expected_episode_count(report)
            or audit.get('tasks') != 50 or audit.get('outlier_report_sha256') != digest(report)):
        raise ValueError('The complete 50-task data audit did not pass')
    gpu = choose_gpu(root, args.gpus.split(','), args.num_processes)
    common = ['--source',source,'--stats',source/'utils/stat-500-all.json','--audit',args.data/'dataset_audit.json']
    if args.num_processes > 1:
        common.extend(['--workers-per-rank','2'])
    if not (root/'smoke/smoke_result.json').exists():
        execute(root,'smoke',['--config',configs[0],'--output',root/'smoke','--smoke-steps','3',*common],gpu)
    # Smoke uses its own random model; stage 1 initializes a separate model from the seed.
    for stage, epochs in ((1,12),(2,4)):
        output = root / f'stage{stage}'
        existing = output / 'status.json'
        if existing.exists() and json.loads(existing.read_text()).get('status') == 'completed':
            continue
        command = ['--config',configs[stage-1],'--output',output,'--stop-epochs',str(epochs),*common]
        if (output/'latest.pt').exists():
            command.extend(['--resume',output/'latest.pt'])
        elif stage == 2:
            command.extend(['--init-from',root/'stage1/latest.pt'])
        execute(root,f'stage{stage}',command,gpu)
    cfg = {'framework':{'name':'LiLaWAM','lila_wam':{'source_root':str(source),
        'config_path':str(root/'stage2/config.yaml'),'checkpoint_path':str(root/'stage2/policy.pt'),
        'vision_encoder_path':OmegaConf.load(configs[0]).model.vision_encoder.checkpoint_path,
        'task_cond_dir':str(source/'data-500-taskcond'),'norm_stats_path':str(source/'utils/stat-500-all.json'),
        'device':'cuda','use_bf16':True}}}
    OmegaConf.save(OmegaConf.create(cfg),root/'starvla_inference.yaml')
    status(root,'trained_not_evaluated',checkpoint=str(root/'stage2/policy.pt'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('/data/gaoxiang/LiLaWAM_RoboTwin_Official'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='6,7')
    p.add_argument('--num-processes',type=int,default=1)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        status(args.output,'failed',error=repr(exc))
        raise
