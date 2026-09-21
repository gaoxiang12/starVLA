"""Real full-trajectory data, optimization and strict deployment audits."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import torch

from examples.Robotwin.audits.prepare_qwen_gawm_ranking import ROOT, OUT, OLD, digest, save
from examples.Robotwin.audits.audit_oft_grasp_training_20260910 import packed, ORDER
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.policy_wrapper import PolicyServerWrapper


def data_audit():
    cfg = OmegaConf.load(OUT/'gawm_smoke.yaml')
    split = json.loads((OLD/'split.json').read_text())
    rows = {r['episode_index']:r for r in json.loads((OLD/'training_anchors.json').read_text())['episodes']}
    source = Path(split['dataset'])
    required = ['info.json','episodes.jsonl','tasks.jsonl','modality.json','stats.json','stats_gr00t.json']
    assert all((source/'meta'/name).is_file() for name in required)
    mixture = get_vla_dataset(cfg.datasets.vla_data, mode='train')
    child, = mixture.datasets
    assert set(child.trajectory_ids) == set(split['train_episode_ids'])
    norm = PolicyNormProcessor(str(ROOT/cfg.trainer.pretrained_checkpoint), unnorm_key='aloha')
    audited, samples = [], []
    for ep in sorted(rows):
        path = Path(rows[ep]['parquet_path'])
        assert digest(path) == rows[ep]['parquet_sha256']
        raw = np.asarray(pq.read_table(path,columns=['action'])['action'].to_pylist())
        assert raw.shape[1] == 14 and len(raw) > 16 and np.isfinite(raw).all()
        audited.append(dict(episode=ep, frames=len(raw), sha256=rows[ep]['parquet_sha256']))
        if ep not in sorted(rows)[::48]:
            continue
        for anchor in (0, len(raw)//2, len(raw)-17, len(raw)-2):
            sample = packed(child,ep,anchor)
            valid = np.asarray(sample['action_valid_mask'],bool)
            recovered = norm.unapply_actions(sample['action'])
            indices = anchor+np.arange(1,17)
            np.testing.assert_array_equal(valid, indices < len(raw))
            np.testing.assert_allclose(recovered[valid], raw[indices[valid]][:,ORDER],atol=.005,rtol=0)
            assert sample['lang']=='blocks ranking rgb' and len(sample['image'])==3
            samples.append(dict(episode=ep,anchor=anchor,valid_actions=int(valid.sum())))
    vcfg = OmegaConf.create(OmegaConf.to_container(cfg.datasets.vla_data,resolve=True))
    vcfg.episode_split='validation'
    val = get_vla_dataset(vcfg,mode='validation').datasets[0]
    assert set(val.trajectory_ids)==set(split['validation_episode_ids'])
    # Verify the Qwen loader sees identical RGB/labels, with no different sampling windows.
    qcfg = OmegaConf.load(OUT/'qwen_act_smoke.yaml')
    qchild = get_vla_dataset(qcfg.datasets.vla_data, mode='train').datasets[0]
    assert set(qchild.trajectory_ids)==set(child.trajectory_ids)
    for row in samples[::4]:
        a,b = packed(child,row['episode'],row['anchor']), packed(qchild,row['episode'],row['anchor'])
        np.testing.assert_array_equal(a['action'],b['action'])
        for x,y in zip(a['image'],b['image']): np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
    save(OUT/'data_audit.json',dict(state='full_trajectory_shared_loader_verified',
        train_episodes=len(audited),validation_episodes=len(val.trajectory_ids),
        train_frames=sum(r['frames'] for r in audited),episodes=audited,boundary_samples=samples,
        metadata_sha256={n:digest(source/'meta'/n) for n in required},
        qwen_gawm_images_labels_equal=True, quality_score=9,
        quality_notes='Complete existing converted sorting demonstrations; audited full action traces, metadata and sampled video decoding. Sparse physical timestamps remain an explicit limitation.',
        no_grasp_window=True, no_correction_mixture=True))


def real_sample(cfg):
    child = get_vla_dataset(cfg.datasets.vla_data,mode='train').datasets[0]
    return packed(child,int(child.trajectory_ids[0]),32)


def integration(variant):
    cfg = OmegaConf.load(OUT/f'{variant}_smoke.yaml')
    torch.manual_seed(42)
    model = build_framework(cfg)
    source = torch.load(ROOT/cfg.trainer.pretrained_checkpoint,map_location='cpu',weights_only=True,mmap=True)
    source = model.remap_checkpoint_state_dict(source)
    if variant.startswith('qwen'):
        prefix='qwen_vl_interface.'
        model.qwen_vl_interface.load_state_dict({k[len(prefix):]:v for k,v in source.items() if k.startswith(prefix)},strict=True)
    else:
        state = model.state_dict()
        compatible = {k:v for k,v in source.items() if k in state and v.shape==state[k].shape}
        assert set(compatible)==set(state)
        model.load_state_dict(compatible,strict=True)
    del source
    model.to('cuda').train()
    sample = real_sample(cfg)
    groups = [{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('action_model')], 'lr':1e-4},
              {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('action_model')], 'lr':1e-5}]
    opt = torch.optim.AdamW(groups)
    losses=[]
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        result=model([sample]); loss=result['action_loss']
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if variant=='qwen_act':
            assert model.action_model.decoder.layers[0].multihead_attn.in_proj_weight.grad.abs().sum()>0
        if variant.startswith('qwen'):
            assert model.qwen_vl_interface.model.model.language_model.layers[-1].self_attn.q_proj.weight.grad.abs().sum()>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step(); losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], losses
    model.eval()
    first=model.predict_action([sample])['normalized_actions']
    if variant.startswith('qwen'):
        poisoned=dict(sample,state=np.full((1,14),np.nan),future_images='invalid',action='invalid',native_images='invalid')
        np.testing.assert_array_equal(first,model.predict_action([poisoned])['normalized_actions'])
    save(OUT/f'{variant}_integration.json',dict(state='real_data_optimization_verified',losses=losses,
        parameters=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,action_shape=list(first.shape)))


def checkpoint_audit(checkpoint, output):
    wrapper=PolicyServerWrapper(str(checkpoint),device='cuda',use_bf16=False,unnorm_key='aloha')
    model=wrapper._framework
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    current=model.state_dict()
    assert set(saved)==set(current)
    for name,value in saved.items():
        assert torch.isfinite(value).all() and value.shape==current[name].shape, name
        torch.testing.assert_close(current[name].cpu(),value.to(current[name].dtype),atol=0,rtol=0)
    count=len(saved)
    del saved,current
    cfg=OmegaConf.load(checkpoint.parents[1]/'config.full.yaml')
    sample=real_sample(cfg)
    proc=wrapper._get_processor('aloha')
    baseline=OmegaConf.load(OUT/'gawm_smoke.yaml')
    reference=PolicyNormProcessor(str(ROOT/baseline.trainer.pretrained_checkpoint),unnorm_key='aloha')
    random=np.random.default_rng(713).uniform(-1.5,1.5,(1024,14)).astype(np.float32)
    np.testing.assert_array_equal(proc.unapply_actions(random),reference.unapply_actions(random))
    # The loader state is normalized; deployment receives raw drive targets.
    raw_path=json.loads((OLD/'training_anchors.json').read_text())['episodes'][0]['parquet_path']
    raw=np.asarray(pq.read_table(raw_path,columns=['action'])['action'].to_pylist())[32:33,ORDER]
    payload=dict(sample,state=raw)
    direct=dict(payload,state=proc.apply_state(raw)) if model.expects_normalized_state else payload
    normalized=model.predict_action([direct])['normalized_actions']
    actual=wrapper.predict_action([payload],unnorm_key='aloha')['actions']
    np.testing.assert_array_equal(actual,np.stack([proc.unapply_actions(normalized[0])]))
    assert actual.shape==(1,16,14) and np.isfinite(actual).all()
    if cfg.framework.name=='QwenGAWM':
        poisoned=dict(payload,state=np.full((1,14),np.nan),future_images='invalid',action='invalid',native_images='invalid')
        np.testing.assert_array_equal(actual,wrapper.predict_action([poisoned],unnorm_key='aloha')['actions'])
    save(output,dict(state='strict_deployment_verified',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        loaded_tensors=count,normalization_samples=1024,direct_wrapper_max_difference=0.,action_shape=list(actual.shape)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['data','integration','checkpoint'],required=True)
    p.add_argument('--variant',choices=['gawm','qwen_mlp','qwen_act'])
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.mode=='data':data_audit()
    elif args.mode=='integration':integration(args.variant)
    else:checkpoint_audit(args.checkpoint,args.output)
