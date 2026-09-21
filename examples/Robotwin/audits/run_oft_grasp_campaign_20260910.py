"""Run audited training stages, strict deployment checks, then fixed grasp scenes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
from omegaconf import OmegaConf
import psutil

from examples.Robotwin.audits.prepare_oft_grasp_training_20260910 import OUT, SHARED, ROOT, digest
from examples.Robotwin.audits.run_grasp_lift_development import PYTHON, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['smoke','data_only','act','oft'], required=True)
    parser.add_argument('--gpu', choices=['0','5','6'], required=True)
    parser.add_argument('--train-port', type=int, required=True)
    parser.add_argument('--eval-port', type=int, required=True)
    args = parser.parse_args()
    status_path = OUT/f'{args.variant}_status.json'
    assert not status_path.exists()
    assert json.loads((OUT/'data_audit.json').read_text())['state']=='shared_loader_first_grasp_windows_verified'
    assert json.loads((OUT/'integration.json').read_text())['state']=='real_data_gradients_optimizer_strict_reload_verified'
    if args.variant != 'smoke':
        assert json.loads((OUT/'smoke_status.json').read_text())['state']=='complete'
    stages = {'smoke':['oft_smoke20'], 'data_only':['data_only1000'],
              'act':['act_warmup500','act_joint4500'], 'oft':['oft_warmup500','oft_joint4500']}[args.variant]
    preparation = json.loads((OUT/'preparation.json').read_text())
    prepared = {r['name']:r for r in preparation['configs']}
    sources = [Path(__file__), OUT/'preparation.json', OUT/'data_audit.json', OUT/'integration.json',
        OUT/'first_grasp_anchors.json', Path(preparation['source_split']),
        ROOT/'examples/Robotwin/audits/audit_oft_grasp_training_20260910.py',
        ROOT/'starVLA/model/framework/WM4A/GAWM.py',
        ROOT/'starVLA/model/framework/WM4A/GAWMOFT.py',
        ROOT/'starVLA/model/modules/action_model/OFTSpatialActionHead.py',
        ROOT/'starVLA/model/modules/action_model/MLP_ActionHeader.py',
        ROOT/'starVLA/model/modules/action_model/ACT_ActionHeader.py',
        ROOT/'starVLA/model/modules/world_model/__init__.py',
        ROOT/'starVLA/model/modules/world_model/GAWM.py',
        ROOT/'starVLA/dataloader/lerobot_datasets.py',
        ROOT/'starVLA/dataloader/gr00t_lerobot/datasets.py',
        ROOT/'starVLA/dataloader/training_anchor_bounds.py',
        ROOT/'starVLA/training/train_starvla.py',
        ROOT/'starVLA/training/trainer_utils/trainer_tools.py',
        ROOT/'starVLA/training/trainer_utils/action_validation.py',
        ROOT/'deployment/model_server/policy_wrapper.py',
        ROOT/'deployment/model_server/policy_norm_processor.py',
        ROOT/'deployment/model_server/server_policy.py',
        ROOT/'examples/Robotwin/audits/run_grasp_precision_development.py',
        ROOT/'examples/Robotwin/audits/record_pregrasp_precision.py',
        ROOT/'examples/Robotwin/audits/run_grasp_lift_case.py',
        ROOT/'examples/Robotwin/audits/grasp_lift_scoring.py',
        ROOT/'examples/Robotwin/audits/grasp_lift_protocol_20260909.json',
        ROOT/'examples/Robotwin/eval_files/model2robotwin_interface.py']
    for stage in stages:
        p = Path(prepared[stage]['path'])
        assert digest(p) == prepared[stage]['sha256']
        sources.append(p)
    pinned = {str(p.resolve()):digest(p) for p in sources}
    report = dict(state='starting', variant=args.variant, gpu=args.gpu, stages=stages,
        completed_stages=[], source_sha256=pinned, supervisor_pid=os.getpid(),
        supervisor_birth=psutil.Process().create_time(), evaluation_port=args.eval_port,
        evaluation_scenes=0 if args.variant=='smoke' else 20)
    child = None

    def status():
        save(status_path,dict(report, child_pid=child.pid if child and child.poll() is None else None,
                              time=time.strftime('%Y-%m-%d %H:%M:%S')))

    def check_sources():
        for p, sha in pinned.items():
            assert digest(Path(p)) == sha, f'Pinned source changed: {p}'

    def wait_gpu():
        while True:
            used = int(subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.used',
                                              '--format=csv,noheader,nounits'],text=True).strip())
            if used < 500:
                return
            report.update(state='waiting_for_free_gpu',used_mib=used)
            status(); time.sleep(10)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(ROOT),
        PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
        NO_ALBUMENTATIONS_UPDATE='1', WANDB_MODE='disabled', STARVLA_DISABLE_TQDM='1')

    def execute(command, log_path):
        nonlocal child
        report['command'] = command
        check_sources()
        with log_path.open('x') as log:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        report['child_birth'] = psutil.Process(child.pid).create_time()
        while child.poll() is None:
            status(); time.sleep(10)
        assert child.returncode == 0, f'Child failed ({child.returncode}); see {log_path}'
        check_sources()

    def stop(sig,frame):
        raise RuntimeError(f'Signal {sig}')

    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,stop)
    try:
        checkpoint = None
        for index,stage in enumerate(stages):
            wait_gpu()
            cfg_path = OUT/f'{stage}.yaml'
            cfg = OmegaConf.load(cfg_path)
            assert not cfg.trainer.is_resume
            run = Path(cfg.run_root_dir)/cfg.run_id
            assert run == SHARED/stage and not run.exists()
            initial = Path(cfg.trainer.pretrained_checkpoint)
            if not initial.is_absolute():initial=ROOT/initial
            assert initial.is_file()
            if checkpoint is not None:assert initial.resolve() == checkpoint.resolve()
            steps = int(cfg.trainer.max_train_steps)
            report.update(state='training', current_stage=stage, current_output=str(run),
                          stage_optimizer_steps=steps, initial_checkpoint=str(initial),
                          initial_checkpoint_sha256=digest(initial))
            execute(['/data/gaoxiang/Code/.venvs/starVLA/bin/accelerate','launch',
                '--config_file','starVLA/config/deepseeds/deepspeed_zero2.yaml',
                '--num_processes','1','--main_process_port',str(args.train_port+index),
                'starVLA/training/train_starvla.py','--config_yaml',str(cfg_path)], OUT/f'{stage}_trainer.log')
            metrics = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            assert metrics[-1]['step'] == steps
            assert all(np.isfinite(v) for row in metrics for v in row.values() if isinstance(v,(int,float)))
            checkpoint = run/'final_model/pytorch_model.pt'
            assert checkpoint.is_file()
            report.update(state='auditing_checkpoint',checkpoint=str(checkpoint))
            audit_path = OUT/f'{stage}_deployment_audit.json'
            execute([str(PYTHON),'-m','examples.Robotwin.audits.audit_oft_grasp_training_20260910',
                     '--mode','checkpoint','--checkpoint',str(checkpoint),'--output',str(audit_path)],
                    OUT/f'{stage}_deployment_audit.log')
            verified = json.loads(audit_path.read_text())
            assert verified['state']=='strict_weights_normalization_and_real_payload_verified'
            report['completed_stages'].append(dict(stage=stage, optimizer_steps=steps,
                checkpoint=str(checkpoint), checkpoint_sha256=verified['checkpoint_sha256'],
                audit=str(audit_path), last_metrics=metrics[-1]))
        if args.variant != 'smoke':
            wait_gpu()
            result_dir = OUT/f'{args.variant}_eval20'
            report.update(state='evaluating', evaluation_output=str(result_dir))
            execute([str(PYTHON),'-u','examples/Robotwin/audits/run_grasp_precision_development.py',
                '--checkpoint',str(checkpoint),'--reference',
                str(ROOT/'playground/Checkpoints/gawm_grasp_precision_v2_reference_20260909'),
                '--gpu',args.gpu,'--port',str(args.eval_port),'--output',str(result_dir)],
                OUT/f'{args.variant}_evaluation.log')
            evaluated = json.loads((result_dir/'status.json').read_text())
            assert evaluated['state']=='complete' and evaluated['completed']==20
            report['evaluation_summary'] = evaluated['summary']
        report.update(state='complete', total_optimizer_steps=sum(s['optimizer_steps'] for s in report['completed_stages']))
    except BaseException as error:
        report.update(state='failed',error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        status()


if __name__=='__main__':
    main()
