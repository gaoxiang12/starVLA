"""Data, exact feature-boundary, transfer, gradient and deployment checks."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import torch
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from examples.Robotwin.audits.prepare_qwen_gawm_depth import ROOT,BASE,OUT,QWEN,VARIANTS,digest,save,validate_config,validate_protocol
from examples.Robotwin.audits.audit_qwen_gawm_ranking import real_sample,checkpoint_audit
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils.trainer_tools import resize_images


def data_audit():
    parent=json.loads((BASE/'data_audit.json').read_text())
    for v in VARIANTS:
        for stage in ('smoke','warmup','joint'):validate_config(OmegaConf.load(OUT/v/f'{stage}.yaml'),stage,v)
        validate_protocol(json.loads((OUT/v/'protocol.json').read_text()))
    cfg=OmegaConf.load(OUT/'vit/smoke.yaml')
    source=Path(cfg.datasets.vla_data.data_root_dir)/'RoboTwinGenerated/Clean/blocks_ranking_rgb'
    for name,sha in parent['metadata_sha256'].items():assert digest(source/'meta'/name)==sha
    for row in parent['episodes']:
        ep=row['episode'];assert digest(source/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet')==row['sha256']
    data=get_vla_dataset(cfg.datasets.vla_data,mode='train').datasets[0]
    split=json.loads(Path(cfg.datasets.vla_data.episode_split_manifest).read_text())
    assert set(data.trajectory_ids)==set(split['train_episode_ids'])
    assert list(data.modality_configs['action'].delta_indices)==list(range(1,17))
    sample=real_sample(cfg);assert sample['action'].shape==(16,14) and np.isfinite(sample['action']).all()
    cfg.datasets.vla_data.episode_split='validation'
    val=get_vla_dataset(cfg.datasets.vla_data,mode='validation').datasets[0]
    assert set(val.trajectory_ids)==set(split['validation_episode_ids'])
    report=dict(state='depth_data_verified',train_episodes=len(data.trajectory_ids),validation_episodes=len(val.trajectory_ids),parent_audit_sha256=digest(BASE/'data_audit.json'),evaluation_scenes=20,test_scenes=0)
    save(OUT/'data_audit.json',report)
    for v in VARIANTS:save(OUT/v/'data_audit.json',report)


def transfer(model):
    source=torch.load(QWEN,map_location='cpu',weights_only=True,mmap=True)
    retained=model.remap_checkpoint_state_dict(source);prefix='qwen_vl_interface.'
    before={k:v.detach().clone() for k,v in model.action_model.state_dict().items()}
    model.qwen_vl_interface.load_state_dict({k[len(prefix):]:v for k,v in retained.items()},strict=True)
    for k,v in model.qwen_vl_interface.state_dict().items():torch.testing.assert_close(v,retained[prefix+k].to(v.dtype),atol=0,rtol=0)
    for k,v in model.action_model.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
    return dict(retained_oft_tensors=len(retained),action_head_unchanged=True,
        total_parameters=sum(p.numel() for p in model.parameters()),qwen_parameters=sum(p.numel() for p in model.qwen_vl_interface.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


@torch.no_grad()
def feature_boundary_audit(model,sample):
    cfg=model.config;variant=model.depth_variant
    images=resize_images(to_pil_preserve(sample['image']),target_size=cfg.datasets.vla_data.obs_image_size)
    processor=model.qwen_vl_interface.processor
    text=cfg.framework.task_instruction+' Please predict the next 16 robot actions: <action>'+'🔍'*16+'<action>.'
    messages=[{'role':'user','content':[{'type':'image','image':x} for x in images]+[{'type':'text','text':text}]}]
    inputs=processor.apply_chat_template(messages,tokenize=True,padding=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
    standalone=processor.image_processor(images=images,return_tensors='pt').to('cuda')
    for k in ('pixel_values','image_grid_thw'):torch.testing.assert_close(inputs[k],standalone[k],atol=0,rtol=0)
    source=torch.load(QWEN,map_location='cpu',weights_only=True,mmap=True)
    saved=[]
    if variant=='half':
        rcfg=OmegaConf.load(BASE/'qwen_act_warmup.yaml');full=build_framework(rcfg)
        prefix='qwen_vl_interface.'
        full.qwen_vl_interface.load_state_dict({k[len(prefix):]:v for k,v in source.items() if k.startswith(prefix)},strict=True)
        full.to('cuda').eval()
        handle=full.qwen_vl_interface.model.model.language_model.layers[17].register_forward_hook(lambda m,a,o:saved.append(o.detach()))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            full.qwen_vl_interface.model.model(**inputs,use_cache=False,return_dict=True)
            expected=full.qwen_vl_interface.model.model.language_model.norm(saved[0])
            actual=model.qwen_vl_interface.model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state
        handle.remove();torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        assert len(model.qwen_vl_interface.model.model.language_model.layers)==18
        count=18
    else:
        vc=AutoConfig.from_pretrained(cfg.framework.qwenvl.base_vlm,local_files_only=True).vision_config;vc._attn_implementation='sdpa'
        full=Qwen3VLVisionModel(vc).to(torch.bfloat16)
        prefix='qwen_vl_interface.model.model.visual.'
        full.load_state_dict({k[len(prefix):]:v for k,v in source.items() if k.startswith(prefix)},strict=True)
        full.to('cuda').eval()
        handle=full.blocks[-1].register_forward_hook(lambda m,a,o:saved.append(o.detach()))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            merged,_=full(inputs['pixel_values'].to(torch.bfloat16),grid_thw=inputs['image_grid_thw'])
            actual,_=model.qwen_vl_interface.model.model.visual(inputs['pixel_values'].to(torch.bfloat16),grid_thw=inputs['image_grid_thw'])
        handle.remove();expected=saved[0] if variant=='vit' else merged
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        assert not any('language_model' in k or 'deepstack_merger' in k for k in model.state_dict())
        if variant=='vit':assert not any('.merger.' in k for k in model.state_dict())
        count=0
    result=dict(state='feature_boundary_verified',language_layers_present=count,feature_shape=list(actual.shape),max_difference=0.,preprocessing_identical=True)
    del source,full,expected,actual,saved;gc.collect();torch.cuda.empty_cache()
    return result


def integration(variant):
    dest=OUT/variant;cfg=OmegaConf.load(dest/'smoke.yaml');validate_config(cfg,'smoke',variant)
    torch.manual_seed(42);model=build_framework(cfg);proof=transfer(model)
    model.to('cuda').eval();sample=real_sample(cfg)
    boundary=feature_boundary_audit(model,sample)
    save(dest/'initialization_audit.json',dict(state='depth_transfer_verified',**proof,feature_boundary=boundary))
    torch.cuda.reset_peak_memory_stats();model.train()
    groups=[]
    for name,lr in [('action_model',1e-4),('qwen_vl_interface',1e-5)]:
        params=[p for n,p in model.named_parameters() if n.startswith(name) and p.requires_grad]
        if params:groups.append(dict(params=params,lr=lr))
    opt=torch.optim.AdamW(groups);losses=[]
    for _ in range(12):
        opt.zero_grad(set_to_none=True);loss=model([sample])['action_loss'];assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.action_model.decoder.layers[0].multihead_attn.in_proj_weight.grad.abs().sum()>0
        if variant=='half':
            lang=model.qwen_vl_interface.model.model.language_model
            assert lang.layers[17].self_attn.q_proj.weight.grad.abs().sum()>0
            assert all(p.grad is None for p in lang.layers[:14].parameters())
        else:assert all(p.grad is None and not p.requires_grad for p in model.qwen_vl_interface.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
    assert losses[-1]<losses[0],losses
    model.eval();a=model.predict_action([sample])['normalized_actions']
    poison=dict(sample,state=np.full((1,14),np.nan),future_images='invalid',action='invalid',native_images='invalid')
    np.testing.assert_array_equal(a,model.predict_action([poison])['normalized_actions'])
    assert a.shape==(1,16,14) and np.isfinite(a).all()
    save(dest/'integration.json',dict(state='depth_optimization_verified',losses=losses,action_shape=list(a.shape),peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,initialization_sha256=digest(dest/'initialization_audit.json')))


def performance(checkpoint,output):
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    wrapper=PolicyServerWrapper(str(checkpoint),device='cuda',use_bf16=False,unnorm_key='aloha')
    model=wrapper._framework;cfg=OmegaConf.load(checkpoint.parents[1]/'config.full.yaml');sample=real_sample(cfg)
    for _ in range(3):model.predict_action([sample])
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();durations=[]
    for _ in range(20):
        torch.cuda.synchronize();start=time.perf_counter();model.predict_action([sample]);torch.cuda.synchronize()
        durations.append(time.perf_counter()-start)
    save(output,dict(state='performance_measured',checkpoint=str(checkpoint),batch=1,warmup_queries=3,measured_queries=20,
        includes_cpu_image_preprocessing=True,includes_network_or_simulator=False,wrapper_use_bf16=False,
        median_seconds=float(np.median(durations)),p90_seconds=float(np.quantile(durations,.9)),durations=durations,
        peak_allocated_gpu_mib=torch.cuda.max_memory_allocated()/2**20,parameters=sum(p.numel() for p in model.parameters())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['data','integration','checkpoint','performance'],required=True)
    p.add_argument('--variant',choices=VARIANTS);p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.mode=='data':data_audit()
    elif a.mode=='integration':integration(a.variant)
    elif a.mode=='checkpoint':checkpoint_audit(a.checkpoint,a.output)
    else:performance(a.checkpoint,a.output)
