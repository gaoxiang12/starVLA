"""Wait for the requested latest-B evaluation, then run controlled experiment C."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from omegaconf import OmegaConf
from examples.LiLaWAM.official_robotwin_data import write_json, digest
from examples.LiLaWAM.run_gawm_official import snapshot, SOURCE, VENV

BASELINE = Path('/data/gaoxiang/ckpts/lila_starvla/gawm_official_B_20260917')
GATE = BASELINE/'eval/step316000_blocks_ranking_rgb_clean_100ep_seed0/summary.json'
MODEL_FILES = ['examples/LiLaWAM/gawm_official.py', 'starVLA/model/framework/WM4A/GAWM.py',
    'starVLA/model/modules/world_model/GAWM.py',
    'starVLA/model/modules/action_model/ACT_ActionHeader.py', 'starVLA/model/modules/action_model/action_loss.py',
    'starVLA/task_language.py']


def validated_result(path):
    result=json.loads(path.read_text())
    if result['state']!='complete' or result['trials']!=100 or not result['sources_unchanged']:
        raise ValueError(f'Evaluation is incomplete or its sources changed: {path}')
    if len(result['results'])!=1 or result['results'][0]['task']!='blocks_ranking_rgb':
        raise ValueError('Unexpected evaluation task')
    return result


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code=snapshot(root)
    matched={name:digest(code/name)==digest(BASELINE/'code_snapshot'/name) for name in MODEL_FILES}
    if not all(matched.values()): raise ValueError(f'B/C model source mismatch: {matched}')
    audit=json.loads((code/'examples/LiLaWAM/audits/gawm_C_parity_20260919.json').read_text())
    if (audit['status']!='passed' or not audit['three_step_model_bitwise_equal']
            or not audit['resume_same_data_rng_loss']):
        raise ValueError('C must pass update and resume parity before production')
    configs=[]
    for stage in (1,2):
        cfg=OmegaConf.load(code/f'examples/LiLaWAM/train_files/gawm_official_C_stage{stage}.yaml')
        cfg.output_dir=str(root/f'stage{stage}')
        path=root/f'stage{stage}.yaml'
        if path.exists() and OmegaConf.load(path)!=cfg: raise ValueError('Existing C configuration changed')
        OmegaConf.save(cfg,path);configs.append(path)
    write_json(root/'experiment_manifest.json',dict(experiment='C',baseline=str(BASELINE),
        initialization='random policy, bitwise identical to B at seed42; no trained B weights loaded',
        model_source_equal=matched,code_manifest=str(root/'code_manifest.json'),
        optimizer_update='inherited VLATrainer._train_step',training_loop='inherited VLATrainer.train',
        backend='StarVLA Accelerate/DDP, same DDP options as B; not historical DeepSpeed configuration',
        data='same official HDF5 and reader as B; no dataset conversion',
        world_size=8,local_batch=16,global_batch=128,gradient_accumulation_steps=1,
        stage1_steps=573840,stage2_steps=191280,evaluation_steps=[117000,316000],
        gate=str(GATE),gate_requirement='Latest B 316k evaluation must finish before any production C training',
        parity_audit=str(code/'examples/LiLaWAM/audits/gawm_C_parity_20260919.json')))
    launch=json.loads((BASELINE/'latest_eval_launch.json').read_text())
    while not GATE.exists():
        import psutil
        process=psutil.Process(launch['pid'])
        if process.status()==psutil.STATUS_ZOMBIE or 'run_lila_benchmark' not in ' '.join(process.cmdline()):
            raise RuntimeError('B evaluation stopped without a summary; C has not started')
        write_json(root/'run_status.json',dict(status='ready_waiting_for_B_evaluation',supervisor_pid=os.getpid(),
            updated=time.time(),evaluation_pid=launch['pid'],expected_summary=str(GATE)))
        time.sleep(15)
    latest=validated_result(GATE)
    prior=validated_result(BASELINE/'eval/step117000_blocks_ranking_rgb_clean_100ep_seed0/summary.json')
    pairs={ep['seed']:ep['success'] for ep in prior['results'][0]['episodes']}
    write_json(root/'B_latest_comparison.json',dict(latest_step=316000,latest_successes=latest['successes'],
        previous_step=117000,previous_successes=prior['successes'],trials=100,
        paired=[dict(seed=ep['seed'],B117k=pairs.get(ep['seed']),B316k=ep['success']) for ep in latest['results'][0]['episodes']]))
    env=dict(os.environ,HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',NO_ALBUMENTATIONS_UPDATE='1',WANDB_MODE='disabled',
        STARVLA_DISABLE_TQDM='1',ACCELERATE_MIXED_PRECISION='bf16',PYTHONPATH=str(code))
    env.pop('ACCELERATE_USE_DEEPSPEED',None)
    cancelled=threading.Event()

    def evaluate():
        try:
            for step in (117000,316000):
                policy=root/f'stage1/policy_step_{step}.pt'
                while not policy.exists():
                    write_json(root/'eval_status.json',dict(status='waiting_for_checkpoint',step=step,checkpoint=str(policy)))
                    if cancelled.wait(15): return
                output=root/f'eval/step{step}_blocks_ranking_rgb_clean_100ep_seed0'
                output.parent.mkdir(exist_ok=True)
                if not (output/'summary.json').exists():
                    if output.exists(): raise RuntimeError(f'Incomplete evaluation exists: {output}')
                    settings=dict(source_root=str(SOURCE),config_path=str(configs[0]),checkpoint_path=str(policy),
                        norm_stats_path=str(SOURCE/'utils/stat-500-all.json'),
                        vision_encoder_path='/data/gaoxiang/ckpts/dinov3-vitl16-pretrain-lvd1689m')
                    config=root/f'eval_step{step}.yaml'
                    OmegaConf.save(OmegaConf.create({'framework':{'name':'GAWMOfficial','official_robotwin':settings}}),config)
                    command=[sys.executable,'-m','examples.Robotwin.eval_files.run_lila_benchmark',
                        '--config',str(config),'--output',str(output),'--tasks','blocks_ranking_rgb',
                        '--episodes','100','--seed','0','--policy-seed','0','--gpus','7','--base-port','5838',
                        '--python',str(VENV/'starVLA/bin/python'),'--sim-python',str(VENV/'RoboTwin/bin/python'),
                        '--robotwin','/data/gaoxiang/Code/RoboTwin','--video']
                    with (root/f'eval_step{step}.log').open('a') as log:
                        child=subprocess.Popen(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
                        write_json(root/'eval_status.json',dict(status='evaluating',step=step,pid=child.pid))
                        if child.wait(): raise RuntimeError(f'C {step} evaluation failed')
                result=validated_result(output/'summary.json')
                b=prior if step==117000 else latest
                bp={ep['seed']:ep['success'] for ep in b['results'][0]['episodes']}
                write_json(root/f'comparison_step{step}.json',dict(step=step,B_successes=b['successes'],
                    C_successes=result['successes'],trials=100,C_summary=str(output/'summary.json'),
                    paired=[dict(seed=ep['seed'],B=bp.get(ep['seed']),C=ep['success']) for ep in result['results'][0]['episodes']]))
                write_json(root/'eval_status.json',dict(status='completed',step=step,successes=result['successes'],trials=100))
        except Exception as exc:
            write_json(root/'eval_status.json',dict(status='failed',error=repr(exc)))

    watcher=threading.Thread(target=evaluate);watcher.start()
    try:
        for stage in (1,2):
            output=root/f'stage{stage}'
            status=output/'status.json'
            if status.exists() and json.loads(status.read_text()).get('status')=='completed': continue
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                '--module','starVLA.training.train_gawm_official','--config',str(configs[stage-1])]
            if (output/'latest.pt').exists(): command+=['--resume',str(output/'latest.pt')]
            elif stage==2: command+=['--init-from',str(root/'stage1/latest.pt')]
            with (root/f'stage{stage}.log').open('a') as log:
                child=subprocess.Popen(command,cwd=code,env=dict(env,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'),stdout=log,stderr=subprocess.STDOUT)
                write_json(root/'run_status.json',dict(status=f'stage{stage}',supervisor_pid=os.getpid(),child_pid=child.pid,
                    started=time.time(),B316k_successes=latest['successes'],B316k_trials=100,command=command))
                if child.wait(): raise RuntimeError(f'C stage {stage} failed')
        watcher.join()
        write_json(root/'run_status.json',dict(status='training_completed',evaluation_status=str(root/'eval_status.json')))
    finally:
        cancelled.set();watcher.join()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('/data/gaoxiang/ckpts/lila_starvla/gawm_official_C_20260919'))
    args=p.parse_args()
    try: run(args)
    except Exception as exc:
        write_json(args.output/'run_status.json',dict(status='failed',error=repr(exc)))
        raise
