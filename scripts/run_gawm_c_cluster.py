"""Launch the public C trainer using an isolated source snapshot per node."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

from omegaconf import OmegaConf

from starVLA.local_settings import cluster_hosts
from starVLA.training.recipe import apply_training_recipe, resolve_training_budget, resume_contract


def resolve_gpu_map(hosts, gpus, gpu_map=None):
    if len(set(hosts)) != len(hosts):
        raise ValueError('Training hosts must be unique')
    mapping = {host: gpus for host in hosts} if gpu_map is None else json.loads(gpu_map)
    if set(mapping) != set(hosts):
        raise ValueError('GPU map must specify exactly the training hosts')
    offset = 0
    for host in hosts:
        value = mapping[host]
        indices = [int(x) for x in value.split(',')]
        if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
            raise ValueError(f'Invalid GPU selection for {host}')
        mapping[host] = ','.join(map(str, indices))
        # The frozen trainer calls a barrier before DDP binds a device. NCCL
        # guesses that device as global_rank % local_device_count. Keep this
        # equal to LOCAL_RANK; descending power-of-two node sizes satisfy it.
        if offset % len(indices):
            raise ValueError('Node rank offset must be divisible by its GPU count; order larger nodes first')
        offset += len(indices)
    return mapping


def validate_cluster_batch(cfg, world_size):
    """Validate the effective recipe, including per-device and accumulation overrides."""
    resolved = apply_training_recipe(cfg)
    batch = int(resolved.datasets.vla_data.per_device_batch_size)
    accumulation = int(resolved.trainer.gradient_accumulation_steps)
    if world_size < 1 or batch < 1 or accumulation < 1:
        raise ValueError('World size, per-device batch and accumulation must be positive')
    actual = world_size * batch * accumulation
    expected = int(resolved.trainer.expected_global_batch_size)
    if actual != expected:
        raise ValueError(f'Configuration expects global batch {expected}, but '
                         f'{world_size} GPUs x {batch} samples x {accumulation} accumulation = {actual}')
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', default='examples/LIBERO/train_files/starvla_gawm_c_160k.yaml')
    parser.add_argument('--hosts', default=cluster_hosts())
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--gpu-map', help='JSON object mapping every host to its CUDA device list')
    parser.add_argument('--initial-state', help='Complete, validated training state to import into a new run')
    parser.add_argument('--source-snapshot', help='Existing frozen source to preserve when migrating a run')
    parser.add_argument('--controller-host', help='Local host address; defaults to the first training host')
    parser.add_argument('--resume', action='store_true', help='Resume an existing run with its frozen source and latest complete state')
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true', help='Resolve batch and estimated budget without launching or writing a run')
    args = parser.parse_args()
    if args.resume and (args.initial_state or args.source_snapshot):
        parser.error('Use initial-state/source-snapshot only for a new run')
    root = Path.cwd().resolve()
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        parser.error('run-id must be a new directory name')
    cfg = OmegaConf.load(args.config)
    if args.resume:
        existing_run = Path(cfg.run_root_dir).resolve() / args.run_id
        cfg = OmegaConf.load(existing_run / 'input_config.yaml')
        cfg.training_overrides.trainer.is_resume = True
    cfg.run_id = args.run_id
    if args.smoke_steps:
        cfg.training_overrides.trainer.max_train_steps = args.smoke_steps
        cfg.training_overrides.trainer.logging_frequency = 1
        cfg.trainer.max_train_steps = args.smoke_steps
        cfg.trainer.logging_frequency = 1
    if args.initial_state:
        cfg.trainer.is_resume = True
        cfg.training_overrides.trainer.is_resume = True
    hosts = args.hosts.split(',')
    controller = args.controller_host or hosts[0]
    gpu_map = resolve_gpu_map(hosts, args.gpus, args.gpu_map)
    world_size = sum(len(value.split(',')) for value in gpu_map.values())
    resolved = validate_cluster_batch(cfg, world_size)
    if args.dry_run:
        frames = int(resolved.datasets.vla_data.expected_frames)
        resolve_training_budget(resolved, range(frames), resolved.trainer.expected_global_batch_size)
        print(OmegaConf.to_yaml(OmegaConf.create({
            'run_id': args.run_id, 'world_size': world_size, 'gpu_map': gpu_map,
            'per_device_batch_size': resolved.datasets.vla_data.per_device_batch_size,
            'budget_source': 'expected_frames; actual dataset is verified during training',
            'trainer': resolved.trainer})))
        return
    asset = Path(cfg.framework.world_model.vision_encoder_path).resolve()
    if not (asset / 'config.json').is_file() or not list(asset.glob('*.safetensors')):
        raise FileNotFoundError(f'Pretrained DINO weights required at {asset}')
    run = Path(cfg.run_root_dir).resolve() / args.run_id
    run.mkdir(parents=True, exist_ok=args.resume)
    (run / 'logs').mkdir(exist_ok=args.resume)
    (run / 'supervisor.pid').write_text(str(os.getpid()))
    exclusions = cfg.datasets.vla_data.get('episode_exclusions_file')
    if exclusions and not args.resume:
        shutil.copy2(exclusions, run / 'video_exclusions.json')
        cfg.datasets.vla_data.episode_exclusions_file = str(run / 'video_exclusions.json')
    OmegaConf.save(cfg, run / 'input_config.yaml')
    snapshot = run / 'source_snapshot'
    if args.source_snapshot:
        shutil.copytree(args.source_snapshot, snapshot)
    tracked = [] if args.resume or args.source_snapshot else subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], text=True
    ).split('\0')
    for name in tracked:
        if not name or not Path(name).is_file() or not name.startswith(('starVLA/', 'examples/', 'scripts/')):
            continue
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, dest)
    # Snapshot the orchestration entry too, although it may not be committed yet.
    for name in (() if args.resume else ('scripts/run_gawm_c_cluster.py', 'scripts/activate_env.sh')):
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, dest)
    if not args.resume:
        if args.source_snapshot:
            origin = Path(args.source_snapshot).resolve()
            (run / 'source_snapshot_origin.txt').write_text(str(origin) + '\n')
            for name in ('git_diff.patch', 'git_commit.txt'):
                shutil.copy2(origin.parent / name, run / name)
        else:
            (run / 'git_diff.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD']))
            (run / 'git_commit.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
    if args.initial_state:
        source = Path(args.initial_state).resolve()
        metadata = json.loads((source / 'complete.json').read_text())
        if metadata['world_size'] != world_size or metadata['global_batch_size'] != resolved.trainer.expected_global_batch_size:
            raise ValueError('Imported state does not match the cluster batch layout')
        if metadata['contract'] != resume_contract(resolved):
            raise ValueError('Imported state contract differs from the configuration')
        if len(list(source.glob('random_states_*.pkl'))) != world_size:
            raise ValueError('Imported state is missing rank RNG records')
        shutil.copytree(source, run / 'checkpoints' / f'steps_{metadata["step"]}_training_state')
        shutil.copy2(source / 'dataset_statistics.json', run / 'dataset_statistics.json')
        if (source / 'migration.json').exists():
            shutil.copy2(source / 'migration.json', run / 'migration.json')

    def ssh(host):
        return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6',
                '-o', f'UserKnownHostsFile={root}/.cache/multinode/known_hosts_{host}']

    def remote(host, command, **kwargs):
        call = ['bash', '-c', command] if host == controller else ssh(host) + [host, command]
        return subprocess.run(call, check=True, timeout=60, **kwargs)

    def provision(host):
        remote(host, 'mkdir -p ' + shlex.quote(str(run)) + ' ' + shlex.quote(str(asset)))
        for source, target in ((str(run) + '/', str(run) + '/'), (str(asset) + '/', str(asset) + '/')):
            subprocess.run(['rsync', '-a', '--no-owner', '--no-group', '-e', shlex.join(ssh(host)),
                            source, f'{host}:{target}'], check=True, timeout=300)

    # A controller can supervise remote-only training while its GPUs are busy.
    # Check the selected GPUs immediately before provisioning/launching.
    for host in hosts:
        result = remote(host, 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits', capture_output=True, text=True)
        selected = set(map(int, gpu_map[host].split(',')))
        available = set()
        for line in result.stdout.splitlines():
            index, memory, utilization = map(int, line.split(','))
            if index in selected and (memory > 200 or utilization > 5):
                raise RuntimeError(f'Refusing occupied GPU {host}:{index}: {memory} MiB, {utilization}%')
            available.add(index)
        if not selected <= available:
            raise ValueError(f'GPU selection missing on {host}')
    if args.resume:
        previous = json.loads((run / 'cluster_status.json').read_text())
        old_map = previous.get('gpu_map', {host: previous['gpus'] for host in previous['hosts']})
        if previous['hosts'] != hosts or old_map != gpu_map:
            raise ValueError('Resume must preserve training hosts and GPU selection')
        code = ('from pathlib import Path; import json; '
                f'p=max(Path({str(run / "checkpoints")!r}).glob("steps_*_training_state/complete.json"), '
                'key=lambda x:int(x.parent.name.split("_")[1])); '
                f'assert len(list(p.parent.glob("random_states_*.pkl"))) == {world_size}; '
                'print(p.parent.name)')
        latest = remote(hosts[0], 'python3 -c ' + shlex.quote(code), capture_output=True, text=True).stdout.strip()
        if hosts[0] != controller:
            target = run / 'checkpoints' / latest
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(['rsync', '-a', '--no-owner', '--no-group', '-e', shlex.join(ssh(hosts[0])),
                            f'{hosts[0]}:{target}/', str(target) + '/'], check=True, timeout=300)
        for marker in [run / 'training_complete.json', *run.glob('STATUS.*')]:
            marker.unlink(missing_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(provision, [h for h in hosts if h != controller]))
    port_code = 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
    port = int(remote(hosts[0], 'python3 -c ' + shlex.quote(port_code), capture_output=True, text=True).stdout)
    state = dict(run_id=args.run_id, hosts=hosts, gpus=args.gpus, gpu_map=gpu_map, world_size=world_size,
                 status='starting', started_at=time.time(), supervisor_pid=os.getpid(),
                 port=port, validation_only=bool(args.smoke_steps), config=str(run / 'input_config.yaml'),
                 controller_host=controller, checkpoint_host=hosts[0])

    def mirror_metadata():
        if hosts[0] == controller:
            return
        # Checkpoints stay on rank zero; mirror small metadata for local status
        # tools without copying live checkpoint files or overwriting our status.
        subprocess.run(['rsync', '-a', '--no-owner', '--no-group',
                        '--include=config.*.yaml', '--include=dataset_statistics.json',
                        '--include=metrics.jsonl', '--include=summary.jsonl', '--exclude=*',
                        '-e', shlex.join(ssh(hosts[0])), f'{hosts[0]}:{run}/', str(run) + '/'],
                       check=True, timeout=60)

    def record():
        temporary = run / 'cluster_status.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(run / 'cluster_status.json')

    stopping = False
    def interrupt(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def stop_node(host):
        code = ('import os,signal,pathlib; '
                f'p=int(pathlib.Path({str(run / "train.pid")!r}).read_text()); '
                'c=pathlib.Path("/proc/"+str(p)+"/cmdline").read_bytes(); '
                f'assert {args.run_id.encode()!r} in c; os.killpg(p,signal.SIGTERM)')
        try:
            remote(host, 'python3 -c ' + shlex.quote(code), capture_output=True)
        except (OSError, subprocess.SubprocessError):
            pass

    jobs = []
    record()
    try:
        for rank, host in enumerate(hosts):
            nproc = len(gpu_map[host].split(','))
            command = [str(root / '.venv/bin/python'), '-m', 'torch.distributed.run',
                       '--nnodes', str(len(hosts)), '--nproc_per_node', str(nproc),
                       '--node_rank', str(rank), '--master_addr', hosts[0], '--master_port', str(port),
                       '--module', 'starVLA.training.train_starvla',
                       '--config_yaml', str(run / 'input_config.yaml')]
            env = dict(CUDA_VISIBLE_DEVICES=gpu_map[host], NCCL_SOCKET_IFNAME='eth0', GLOO_SOCKET_IFNAME='eth0',
                       TORCH_ELASTIC_WORKER_IDENTICAL='0',
                       NCCL_IB_DISABLE='1', NCCL_DEBUG='WARN', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1',
                       PYTHONUNBUFFERED='1', STARVLA_DISABLE_TQDM='1', PYTHONNOUSERSITE='1',
                       ACCELERATE_USE_DEEPSPEED='false', ACCELERATE_MIXED_PRECISION='bf16',
                       PYTHONPATH=str(snapshot), HF_HOME=str(root / '.cache/huggingface'),
                       MPLCONFIGDIR=str(root / '.cache/matplotlib'), TRITON_CACHE_DIR=str(root / '.cache/triton'))
            body = (f'cd {shlex.quote(str(snapshot))} && '
                    f'echo $$ > {shlex.quote(str(run / "train.pid"))} && exec env '
                    + ' '.join(shlex.quote(f'{key}={value}') for key, value in env.items())
                    + ' ' + shlex.join(command))
            launch = ['setsid', '--wait', 'bash', '-c', body]
            call = launch if host == controller else ssh(host) + [host, shlex.join(launch)]
            log = (run / 'logs' / f'{host}.log').open('a' if args.resume else 'w')
            proc = subprocess.Popen(call, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((host, proc, log))
        state['status'] = 'running'
        record()
        print('CLUSTER STARTED ' + json.dumps(state), flush=True)
        while True:
            codes = {host: proc.poll() for host, proc, _ in jobs}
            mirror_metadata()
            if stopping:
                state['status'] = 'stopped'
                break
            if any(code is not None and code != 0 for code in codes.values()):
                state.update(status='failed', exit_codes=codes)
                break
            if all(code == 0 for code in codes.values()):
                config = OmegaConf.load(run / 'config.full.yaml')
                step = int(config.trainer.max_train_steps)
                saved = run / 'checkpoints' / f'steps_{step}_training_state'
                verify = ('from pathlib import Path; p=Path(' + repr(str(saved)) + '); '
                          'assert (p/"complete.json").exists(); '
                          f'assert len(list(p.glob("random_states_*.pkl"))) == {state["world_size"]}')
                remote(hosts[0], 'python3 -c ' + shlex.quote(verify), capture_output=True)
                state.update(status='complete', completed_steps=step)
                (run / 'training_complete.json').write_text(json.dumps(state, indent=2))
                break
            time.sleep(10)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        if state['status'] != 'complete':
            with ThreadPoolExecutor(max_workers=5) as pool:
                list(pool.map(stop_node, [h for h, _, _ in jobs]))
        for _, proc, log in jobs:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
            log.close()
        state['finished_at'] = time.time()
        record()
        (run / f'STATUS.{state["status"]}').touch()
        print('CLUSTER FINISHED ' + json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
