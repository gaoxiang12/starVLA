"""Prove original VLM initialization, matching data, gradients and deployment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch
from safetensors import safe_open

from examples.Robotwin.audits.prepare_qwen_gawm_vlm_only import ROOT,BASE,OUT,BASE_VLM,digest,save,validate_config
from examples.Robotwin.audits.audit_qwen_gawm_ranking import real_sample,checkpoint_audit
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def verify_original_vlm(model):
    """Compare every original safetensors key, allowing only the proven tied LM head alias."""
    index=json.loads((BASE_VLM/'model.safetensors.index.json').read_text())['weight_map']
    current=model.qwen_vl_interface.model.state_dict()
    extra=set(current)-set(index)
    assert not set(index)-set(current)
    assert extra <= {'lm_head.weight'},extra
    checked=0
    for shard in sorted(set(index.values())):
        with safe_open(BASE_VLM/shard,framework='pt',device='cpu') as f:
            assert set(f.keys())=={k for k,v in index.items() if v==shard}
            for name in f.keys():
                original=f.get_tensor(name)
                assert torch.isfinite(original).all()
                torch.testing.assert_close(current[name],original.to(current[name].dtype),atol=0,rtol=0)
                checked+=1
    if extra:
        torch.testing.assert_close(current['lm_head.weight'],current['model.language_model.embed_tokens.weight'],atol=0,rtol=0)
    h=hashlib.sha256()
    for name,value in model.action_model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return dict(original_vlm_tensors_verified=checked,tied_aliases=sorted(extra),
                random_act_sha256=h.hexdigest(),oft_checkpoint_loaded=False)


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
    cfg.datasets.vla_data.episode_split='validation'
    val=get_vla_dataset(cfg.datasets.vla_data,mode='validation').datasets[0]
    assert set(val.trajectory_ids)==set(split['validation_episode_ids'])
    save(OUT/'data_audit.json',dict(state='matched_16_step_data_verified',train_episodes=len(dataset.trajectory_ids),
        validation_episodes=len(val.trajectory_ids),parent_data_audit_sha256=digest(BASE/'data_audit.json'),
        config_differences_only=['trainer.pretrained_checkpoint','trainer.reload_modules','run_id','run_root_dir']))


def integration():
    cfg=OmegaConf.load(OUT/'smoke.yaml');validate_config(cfg,'smoke');torch.manual_seed(42)
    model=build_framework(cfg)
    proof=verify_original_vlm(model);save(OUT/'initialization_audit.json',dict(state='original_vlm_weights_verified',**proof))
    model.to('cuda').train();sample=real_sample(cfg)
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('action_model')],lr=1e-4),
            dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('action_model')],lr=1e-5)]
    opt=torch.optim.AdamW(groups);losses=[]
    for _ in range(12):
        opt.zero_grad(set_to_none=True);loss=model([sample])['action_loss']
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.action_model.decoder.layers[0].multihead_attn.in_proj_weight.grad.abs().sum()>0
        assert model.qwen_vl_interface.model.model.language_model.layers[-1].self_attn.q_proj.weight.grad.abs().sum()>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
    assert losses[-1]<losses[0]
    model.eval();prediction=model.predict_action([sample])['normalized_actions']
    assert prediction.shape==(1,16,14) and np.isfinite(prediction).all()
    poison=dict(sample,state=np.full((1,14),np.nan),future_images='invalid',action='invalid',native_images='invalid')
    np.testing.assert_array_equal(prediction,model.predict_action([poison])['normalized_actions'])
    save(OUT/'integration.json',dict(state='vlm_only_optimization_verified',losses=losses,
        action_shape=list(prediction.shape),initialization_audit_sha256=digest(OUT/'initialization_audit.json'),
        peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['data','integration','checkpoint'],required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.mode=='data':data_audit()
    elif args.mode=='integration':integration()
    else:checkpoint_audit(args.checkpoint,args.output)
