"""Wait for idle GPUs on a dataset replica, then prepare and publish RoboTwin VTT.

Only data preparation is launched. Running GPU jobs are never terminated.
The queue is resumable through atomic per-task and per-shard files.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starVLA.local_settings import cluster_host, ssh_target

REPO = Path(__file__).resolve().parents[1]
ROOT = os.environ.get('STARVLA_ROBOTWIN_DATA', str(REPO / 'playground/Datasets/LiLaWAM_RoboTwin_Official'))
ENCODER = str(REPO / 'playground/Pretrained/dinov3-vitl16-pretrain-lvd1689m')
CONFIG = REPO / 'examples/Robotwin/train_files/starvla_gawm_l_robotwin_head_front_vtt_c_12plus4.yaml'


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main(args):
    from omegaconf import OmegaConf
    from starVLA.model.modules.gawm_l_vision import VTTConditioner

    work = Path(args.output).resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    host = ssh_target(args.host)
    opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o',
            'UserKnownHostsFile=' + str(REPO / f'.cache/multinode/known_hosts_{args.host}')]

    def remote(code):
        command = str(REPO / '.venv/bin/python') + ' -c ' + shlex.quote(code)
        result = subprocess.run(['ssh', *opts, host, command], capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        return json.loads(result.stdout)

    remote(f"from pathlib import Path; Path({str(work)!r}).mkdir(parents=True,exist_ok=True);print('null')")
    worker = work / 'prepare.py'
    subprocess.run(['scp', *opts, str(REPO / 'scripts/prepare_robotwin_official_vtt.py'),
                    host + ':' + str(worker)], check=True, timeout=60)
    state_path = work / 'queue_status.json'
    jobs = json.loads(state_path.read_text()).get('jobs', {}) if state_path.exists() else {}
    base = [str(REPO / '.venv/bin/python'), '-u', str(worker), '--root', ROOT,
            '--encoder', ENCODER, '--output', str(work), '--shards', '4', '--batch-images', '64']
    while True:
        try:
            probe = remote(f'''
import json,subprocess
from pathlib import Path
work=Path({str(work)!r});jobs={jobs!r}
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus={{int(row.split(',')[0]):[int(v.strip()) for v in row.split(',')[1:]] for row in rows.splitlines()}}
complete=[s for s in range(4) if (work/f'shard_{{s}}.json').exists()]
exits={{str(s):json.loads((work/f'exit_{{s}}.json').read_text()) for s in range(4) if (work/f'exit_{{s}}.json').exists()}}
alive={{s:Path('/proc/'+str(j['pid'])).exists() for s,j in jobs.items()}}
print(json.dumps(dict(gpus=gpus,complete=complete,exits=exits,alive=alive,tasks=len(list((work/'tasks').glob('*.json'))))))
''')
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            print(f'Transient node probe failure: {error}', flush=True)
            time.sleep(60)
            continue
        complete = set(probe['complete'])
        for shard, job in jobs.items():
            if int(shard) in complete:
                continue
            code = probe['exits'].get(shard)
            if code is not None or not probe['alive'].get(shard):
                write_json(state_path, dict(status='failed',jobs=jobs,probe=probe,failed_shard=shard))
                raise RuntimeError(f'Shard {shard} stopped before completion; inspect remote shard log')
        if complete == set(range(4)):
            break
        free = [int(g) for g, usage in probe['gpus'].items() if usage == [0, 0]]
        for shard in range(4):
            if shard in complete or str(shard) in jobs or not free:
                continue
            gpu = free.pop(0)
            command = base + ['--shard', str(shard)]
            wrapper = f'''import os,subprocess,json
from pathlib import Path
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES={str(gpu)!r},OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
code=subprocess.call({command!r},env=env)
p=Path({str(work / f'exit_{shard}.json')!r});tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(code));tmp.replace(p)
'''
            result = remote(f'''
import json,subprocess
from pathlib import Path
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
usage={{int(r.split(',')[0]):[int(v.strip()) for v in r.split(',')[1:]] for r in rows.splitlines()}}
if usage[{gpu}]!=[0,0]:
    print('null')
else:
    with Path({str(work / f'shard_{shard}.log')!r}).open('a') as log:
        p=subprocess.Popen([{str(REPO / '.venv/bin/python')!r},'-u','-c',{wrapper!r}],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=p.pid,gpu={gpu})))
''')
            if result:
                jobs[str(shard)] = result
                print(json.dumps(dict(event='launched',shard=shard,**result)), flush=True)
                write_json(state_path, dict(status='extracting',jobs=jobs,probe=probe))
        status = 'extracting' if jobs else 'waiting_for_idle_gpu'
        write_json(state_path, dict(status=status,host=args.host,jobs=jobs,probe=probe,
                                   updated_at=time.strftime('%Y-%m-%d %H:%M:%S')))
        time.sleep(60)
    merged = remote(f'''
import subprocess,json
subprocess.run({base + ['--merge']!r},check=True,stdout=subprocess.DEVNULL)
print(json.dumps(json.loads(open({str(work / 'data_readiness.json')!r}).read())))
''')
    for name in ('robotwin_official_train_vtt.json', 'data_readiness.json'):
        subprocess.run(['scp', *opts, host + ':' + str(work / name), str(work / name)], check=True, timeout=60)
    from omegaconf import OmegaConf
    from starVLA.model.modules.gawm_l_vision import VTTConditioner
    cfg = OmegaConf.load(CONFIG)
    payload = json.loads((work / 'robotwin_official_train_vtt.json').read_text())
    assert sorted(payload['vectors']) == list(cfg.framework.lang_cond.task_names)
    verify_cfg = OmegaConf.create(OmegaConf.to_container(cfg.framework.lang_cond))
    verify_cfg.task_vectors_path = str(work / 'robotwin_official_train_vtt.json')
    conditioner = VTTConditioner(verify_cfg, feature_dim=1024, output_dim=384)
    assert conditioner.ready
    target = Path(cfg.framework.lang_cond.task_vectors_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = (work / 'robotwin_official_train_vtt.json').read_bytes()
    if target.exists() and target.read_bytes() != content:
        raise FileExistsError(f'Refusing to overwrite different VTT: {target}')
    temporary = target.with_suffix('.tmp');temporary.write_bytes(content);temporary.replace(target)
    digest = hashlib.sha256(content).hexdigest()
    replicas = {}
    for name in ('worker-1', 'worker-2', 'worker-3', 'worker-4'):
        node = cluster_host(name)
        node_opts = ['-o','BatchMode=yes','-o','ConnectTimeout=8','-o',
                     'UserKnownHostsFile=' + str(REPO / f'.cache/multinode/known_hosts_{node}')]
        destination = ssh_target(node)
        mkdir = f"from pathlib import Path;Path({str(target.parent)!r}).mkdir(parents=True,exist_ok=True)"
        subprocess.run(['ssh',*node_opts,destination,str(REPO / '.venv/bin/python')+' -c '+shlex.quote(mkdir)],check=True,timeout=60)
        incoming = str(target) + '.incoming_20260926'
        subprocess.run(['scp',*node_opts,str(target),destination+':'+incoming],check=True,timeout=60)
        publish = f'''from pathlib import Path
import hashlib
incoming=Path({incoming!r});target=Path({str(target)!r});expected={digest!r}
assert hashlib.sha256(incoming.read_bytes()).hexdigest()==expected
assert not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()==expected
incoming.replace(target)
print(expected)
'''
        result = subprocess.check_output(['ssh',*node_opts,destination,str(REPO / '.venv/bin/python')+' -c '+shlex.quote(publish)],text=True,timeout=60)
        assert result.strip() == digest
        replicas[node] = digest
    write_json(state_path, dict(status='complete',jobs=jobs,data_readiness=merged,
                               asset=str(target),asset_sha256=digest,config=str(CONFIG),replicas=replicas))
    print(json.dumps(dict(status='complete',asset=str(target),sha256=digest)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default=cluster_host('worker-4'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    try:
        main(args)
    except Exception as error:
        print(f'Queue failed: {error}', file=sys.stderr, flush=True)
        status_path = Path(args.output).resolve() / 'queue_status.json'
        if status_path.parent.exists():
            previous = json.loads(status_path.read_text()) if status_path.exists() else {}
            previous.update(status='failed',error=str(error))
            write_json(status_path,previous)
        raise
