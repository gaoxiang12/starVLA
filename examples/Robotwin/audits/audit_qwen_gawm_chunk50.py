"""Audit 50-step labels, full ACT gradients and strict checkpoint deployment."""
import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import torch

from examples.Robotwin.audits.prepare_qwen_gawm_chunk50 import ROOT, BASE, OUT, digest, save
from examples.Robotwin.audits.audit_oft_grasp_training_20260910 import packed, ORDER
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.policy_norm_processor import PolicyNormProcessor


def reference_norm():
    cfg=OmegaConf.load(BASE/'gawm_smoke.yaml')
    return PolicyNormProcessor(str(ROOT/cfg.trainer.pretrained_checkpoint),unnorm_key='aloha')


def data_audit():
    cfg=OmegaConf.load(OUT/'smoke.yaml')
    child=get_vla_dataset(cfg.datasets.vla_data,mode='train').datasets[0]
    old_cfg=OmegaConf.load(BASE/'qwen_act_smoke.yaml')
    original=get_vla_dataset(old_cfg.datasets.vla_data,mode='train').datasets[0]
    assert set(child.trajectory_ids)==set(original.trajectory_ids) and len(child.trajectory_ids)==475
    assert child.all_steps==original.all_steps
    assert list(child.modality_configs['action'].delta_indices)==list(range(1,51))
    ref=reference_norm()
    previous=json.loads((BASE/'data_audit.json').read_text())
    dataset=Path(cfg.datasets.vla_data.data_root_dir)/'RoboTwinGenerated/Clean/blocks_ranking_rgb'
    for name,sha in previous['metadata_sha256'].items():assert digest(dataset/'meta'/name)==sha
    rows=[];values=[];closed=[]
    for i,row in enumerate(previous['episodes']):
        ep=row['episode'];path=dataset/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'
        assert digest(path)==row['sha256']
        if i%48:continue
        raw=np.asarray(pq.read_table(path,columns=['action'])['action'].to_pylist())
        for anchor in (0,len(raw)//2,len(raw)-51,len(raw)-17,len(raw)-2):
            sample=packed(child,ep,anchor);old=packed(original,ep,anchor)
            assert sample['action'].shape==(50,14)
            mask=np.asarray(sample['action_valid_mask'],bool);indices=anchor+np.arange(1,51)
            np.testing.assert_array_equal(mask,indices<len(raw))
            np.testing.assert_allclose(ref.unapply_actions(sample['action'])[mask],raw[indices[mask]][:,ORDER],atol=.005,rtol=0)
            np.testing.assert_array_equal(sample['action'][:16],old['action'])
            np.testing.assert_array_equal(mask[:16],old['action_valid_mask'])
            for a,b in zip(sample['image'],old['image']):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
            values.append(sample['action'][mask]);closed.append((sample['action'][mask,12:]<.2).mean())
            rows.append(dict(episode=ep,anchor=anchor,valid_actions=int(mask.sum())))
    cfg.datasets.vla_data.episode_split='validation'
    val=get_vla_dataset(cfg.datasets.vla_data,mode='validation').datasets[0]
    split=json.loads(Path(cfg.datasets.vla_data.episode_split_manifest).read_text())
    assert set(val.trajectory_ids)==set(split['validation_episode_ids'])
    values=np.concatenate(values);assert np.isfinite(values).all()
    save(OUT/'data_audit.json',dict(state='50_step_labels_verified',train_episodes=475,validation_episodes=len(val.trajectory_ids),
        observation_anchors_unchanged=True,first_16_labels_and_rgb_equal=True,boundary_samples=rows,
        per_dimension_min=values.min(0).tolist(),per_dimension_max=values.max(0).tolist(),
        sampled_closed_gripper_fraction=float(np.mean(closed)),parent_data_audit_sha256=digest(BASE/'data_audit.json')))


def sample_for(cfg):
    child=get_vla_dataset(cfg.datasets.vla_data,mode='train').datasets[0]
    ep=int(child.trajectory_ids[0]);return packed(child,ep,32),ep


def integration():
    cfg=OmegaConf.load(OUT/'smoke.yaml');torch.manual_seed(42)
    model=build_framework(cfg)
    source=torch.load(cfg.trainer.pretrained_checkpoint,map_location='cpu',mmap=True,weights_only=True)
    model.remap_checkpoint_state_dict(source)
    prefix='qwen_vl_interface.'
    model.qwen_vl_interface.load_state_dict({k[len(prefix):]:v for k,v in source.items() if k.startswith(prefix)},strict=True)
    del source
    model.to('cuda').train();sample,ep=sample_for(cfg)
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('action_model')],lr=1e-4),
            dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('action_model')],lr=1e-5)]
    opt=torch.optim.AdamW(groups);losses=[]
    for _ in range(12):
        opt.zero_grad(set_to_none=True);loss=model([sample])['action_loss']
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.action_model.action_queries.weight.grad[-1].abs().sum()>0
        assert model.action_model.decoder.layers[0].multihead_attn.in_proj_weight.grad.abs().sum()>0
        assert model.qwen_vl_interface.model.model.language_model.layers[-1].self_attn.q_proj.weight.grad.abs().sum()>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
    assert losses[-1]<losses[0]
    model.eval();prediction=model.predict_action([sample])['normalized_actions']
    assert prediction.shape==(1,50,14) and np.isfinite(prediction).all()
    poison=dict(sample,state=np.full((1,14),np.nan),future_images='invalid',action='invalid',native_images='invalid')
    np.testing.assert_array_equal(prediction,model.predict_action([poison])['normalized_actions'])
    save(OUT/'integration.json',dict(state='50_step_optimization_verified',episode=ep,anchor=32,losses=losses,
        action_shape=list(prediction.shape),last_action_query_gradient_verified=True,
        peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))


def checkpoint_audit(checkpoint,output):
    wrapper=PolicyServerWrapper(str(checkpoint),device='cuda',use_bf16=False,unnorm_key='aloha')
    model=wrapper._framework;current=model.state_dict()
    source=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
    assert set(source)==set(current)
    for name,value in source.items():
        assert torch.isfinite(value).all() and value.shape==current[name].shape
        torch.testing.assert_close(current[name].cpu(),value.to(current[name].dtype),atol=0,rtol=0)
    count=len(source);del source,current
    cfg=OmegaConf.load(checkpoint.parents[1]/'config.full.yaml');sample,ep=sample_for(cfg)
    proc=wrapper._get_processor('aloha');random=np.random.default_rng(713).uniform(-1.5,1.5,(1024,14)).astype(np.float32)
    np.testing.assert_array_equal(proc.unapply_actions(random),reference_norm().unapply_actions(random))
    normalized=model.predict_action([sample])['normalized_actions']
    actual=wrapper.predict_action([sample],unnorm_key='aloha')['actions']
    np.testing.assert_array_equal(actual,np.stack([proc.unapply_actions(normalized[0])]))
    assert actual.shape==(1,50,14) and np.isfinite(actual).all()
    poison=dict(sample,state=np.full((1,14),np.nan),future_images='invalid',action='invalid',native_images='invalid')
    np.testing.assert_array_equal(actual,wrapper.predict_action([poison],unnorm_key='aloha')['actions'])
    assert wrapper.metadata['action_chunk_sizes']['aloha']==50
    assert wrapper.metadata['action_chunk_size']==50
    save(output,dict(state='strict_50_step_deployment_verified',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        loaded_tensors=count,action_shape=list(actual.shape),normalization_samples=1024,direct_wrapper_max_difference=0.))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['data','integration','checkpoint'],required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.mode=='data':data_audit()
    elif args.mode=='integration':integration()
    else:checkpoint_audit(args.checkpoint,args.output)
