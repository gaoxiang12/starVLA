"""Prove OFT Qwen transfer and GAWM world-model gradients, matching data, gradients and deployment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch
from safetensors import safe_open

from examples.Robotwin.audits.prepare_qwen_gawm_world import ROOT,BASE,OUT,QWEN,digest,save,validate_config
from examples.Robotwin.audits.audit_qwen_gawm_ranking import real_sample,checkpoint_audit
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def initialize_qwen(model):
    source=torch.load(QWEN,map_location='cpu',weights_only=True,mmap=True)
    model.remap_checkpoint_state_dict(source)
    prefix='qwen_vl_interface.'
    subset={k[len(prefix):]:v for k,v in source.items() if k.startswith(prefix)}
    before={k:v.detach().clone() for k,v in model.action_model.state_dict().items()}
    model.qwen_vl_interface.load_state_dict(subset,strict=True)
    current=model.qwen_vl_interface.state_dict()
    for k,v in subset.items():torch.testing.assert_close(current[k],v.to(current[k].dtype),atol=0,rtol=0)
    for k,v in model.action_model.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
    return dict(qwen_tensors_verified=len(subset),world_and_act_unchanged_by_transfer=True,
        qwen_checkpoint=str(QWEN),world_parameters=sum(p.numel() for p in model.action_model.world_model.parameters()))


def data_audit():
    for stage in ('smoke','warmup','joint'):validate_config(OmegaConf.load(OUT/f'{stage}.yaml'),stage)
    parent=json.loads((BASE/'data_audit.json').read_text())
    cfg=OmegaConf.load(OUT/'smoke.yaml');source=Path(cfg.datasets.vla_data.data_root_dir)/'RoboTwinGenerated/Clean/blocks_ranking_rgb'
    for name,sha in parent['metadata_sha256'].items():assert digest(source/'meta'/name)==sha
    for row in parent['episodes']:
        ep=row['episode'];assert digest(source/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet')==row['sha256']
    dataset=get_vla_dataset(cfg.datasets.vla_data,mode='train').datasets[0]
    split=json.loads(Path(cfg.datasets.vla_data.episode_split_manifest).read_text())
    assert set(dataset.trajectory_ids)==set(split['train_episode_ids'])
    assert list(dataset.modality_configs['action'].delta_indices)==list(range(1,17))
    sample=real_sample(cfg)
    assert sample['action'].shape==(16,14) and np.isfinite(sample['action']).all()
    assert list(dataset.modality_configs['video'].delta_indices)==[0,6,12]
    from examples.Robotwin.audits.audit_oft_grasp_training_20260910 import packed
    import pyarrow.parquet as pq
    for row in parent['episodes'][::48]:
        ep=row['episode'];count=row['frames']
        for anchor in (32,count-2):
            item=packed(dataset,ep,anchor)
            assert len(item['future_images'])==2 and all(len(v)==3 for v in item['future_images'])
            np.testing.assert_array_equal(item['future_frame_valid_mask'],anchor+np.array([0,6,12])<count)
    cfg.datasets.vla_data.episode_split='validation'
    val=get_vla_dataset(cfg.datasets.vla_data,mode='validation').datasets[0]
    assert set(val.trajectory_ids)==set(split['validation_episode_ids'])
    save(OUT/'data_audit.json',dict(state='world_data_verified',train_episodes=len(dataset.trajectory_ids),
        validation_episodes=len(val.trajectory_ids),parent_data_audit_sha256=digest(BASE/'data_audit.json'),
        future_recorded_offsets=[6,12],physical_time_offsets_known=False))


def integration():
    cfg=OmegaConf.load(OUT/'smoke.yaml');validate_config(cfg,'smoke');torch.manual_seed(42)
    model=build_framework(cfg)
    proof=initialize_qwen(model);save(OUT/'initialization_audit.json',dict(state='oft_qwen_transfer_verified',**proof))
    model.to('cuda').train();sample=real_sample(cfg)
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('action_model')],lr=1e-4),
            dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('action_model')],lr=1e-5)]
    opt=torch.optim.AdamW(groups);losses=[]
    for _ in range(12):
        opt.zero_grad(set_to_none=True);loss=model([sample])['action_loss']
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.action_model.act.decoder.layers[0].multihead_attn.in_proj_weight.grad.abs().sum()>0
        assert model.qwen_vl_interface.model.model.language_model.layers[-1].self_attn.q_proj.weight.grad.abs().sum()>0
        assert model.action_model.world_model.residual_predictor.out.weight.grad.abs().sum()>0
        if len(losses)>0:
            assert model.action_model.world_model.residual_predictor.blocks[0].attn.in_proj_weight.grad.abs().sum()>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
    assert losses[-1]<losses[0]
    model.eval();prediction=model.predict_action([sample])['normalized_actions']
    assert prediction.shape==(1,16,14) and np.isfinite(prediction).all()
    poison=dict(sample,state=np.full((1,14),np.nan),future_images='invalid',action='invalid',native_images='invalid')
    np.testing.assert_array_equal(prediction,model.predict_action([poison])['normalized_actions'])
    save(OUT/'integration.json',dict(state='qwen_world_optimization_verified',losses=losses,
        action_shape=list(prediction.shape),initialization_audit_sha256=digest(OUT/'initialization_audit.json'),
        world_diagnostics={k:float(v) for k,v in model([sample]).items() if k.startswith(('latent_','delta_'))},
        peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['data','integration','checkpoint'],required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.mode=='data':data_audit()
    elif args.mode=='integration':integration()
    else:checkpoint_audit(args.checkpoint,args.output)
