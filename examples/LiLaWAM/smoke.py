"""Real shared-loader forward/backward, masks, inference and strict reload."""
import argparse
import copy
import gc
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.WM4A.LiLaWAMTrain import LiLaWAMTrain
from examples.LiLaWAM.prepare import write_json
from examples.LiLaWAM.prepare import modality_array
from deployment.model_server.policy_wrapper import PolicyServerWrapper


def packed(child, ep, anchor):
    child.transforms.eval()
    sample=child._pack_sample(child.transforms(child.get_step_data(ep,anchor)))
    sample=child._attach_action_validity(sample,ep,anchor)
    return child._attach_future_frame_validity(sample,ep,anchor)


def smoke(config, output, device='cpu', steps=2):
    torch.set_num_threads(4);torch.manual_seed(42)
    cfg=OmegaConf.load(config)
    mixture=get_vla_dataset(cfg.datasets.vla_data,mode='validation')
    report=dict(config=str(config),device=device,datasets=[],steps=steps,status='running')
    samples=[]
    for child in mixture.datasets:
        ep=int(child.trajectory_ids[0]); length=int(child.trajectory_lengths[0])
        middle=packed(child,ep,min(32,length-2)); tail=packed(child,ep,length-2)
        for item in (middle,tail):
            assert item['action'].shape[0]==32 and len(item['image'])==3 and len(item['future_images'])==1
            assert np.isfinite(item['state']).all() and np.isfinite(item['action']).all()
        assert not tail['future_frame_valid_mask'][-1]
        assert not tail['action_valid_mask'][-1]
        indexed=child[0]; indexed_mix=mixture[(mixture.datasets.index(child),0)]
        assert indexed['action'].shape==middle['action'].shape==indexed_mix['action'].shape
        samples.append(middle)
        report['datasets'].append(dict(name=child.dataset_name,train_episodes=len(child.trajectory_ids),
            state_shape=list(middle['state'].shape),action_shape=list(middle['action'].shape),
            view_valid_mask=middle['view_valid_mask'],tail_action_valid=int(np.sum(tail['action_valid_mask'])),
            normalized_state_min=float(middle['state'].min()),normalized_state_max=float(middle['state'].max()),
            normalized_action_min=float(middle['action'].min()),normalized_action_max=float(middle['action'].max())))
    model=LiLaWAMTrain(cfg).to(device)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4)
    losses=[]
    for i in range(steps):
        model.train();opt.zero_grad(set_to_none=True)
        sample=samples[i%len(samples)]
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
            out=model([sample]);loss=out['action_loss']
        assert torch.isfinite(loss);loss.backward()
        grads={}
        for name,module in [('adapter',model.core.adapter),('dit',model.core.blocks),
                            ('future',model.core.future),('head',model.core.heads[sample['robot_tag']])]:
            g=[p.grad for p in module.parameters() if p.grad is not None]
            assert g and all(torch.isfinite(v).all() for v in g)
            grads[name]=sum(v.float().abs().sum().item() for v in g)
            assert grads[name]>0
        assert all(p.grad is None for p in model.encoder.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        losses.append(dict(metrics={k:float(v.detach()) for k,v in out.items()},gradients=grads))
    model.eval();del opt;gc.collect()
    if device.startswith('cuda'): torch.cuda.empty_cache()
    torch.manual_seed(999)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
        prediction=model.predict_action([samples[0]])['normalized_actions']
    poisoned=copy.copy(samples[0]);poisoned.update(action='unused',future_images='unused')
    torch.manual_seed(999)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
        other=model.predict_action([poisoned])['normalized_actions']
    np.testing.assert_array_equal(prediction,other)
    assert np.isfinite(prediction).all()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    bundle=output/'final_model';bundle.mkdir(exist_ok=True)
    checkpoint=bundle/'pytorch_model.pt'
    OmegaConf.save(cfg,output/'config.yaml')
    mixture.save_dataset_statistics(output/'dataset_statistics.json')
    torch.save(model.state_dict(),checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True)
    tensor_count=len(state)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad)
    del state,model;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    wrapper=PolicyServerWrapper(str(checkpoint),device=device,use_bf16=False,unnorm_key=samples[0]['robot_tag'])
    child=mixture.datasets[0];ep=int(child.trajectory_ids[0]);anchor=min(32,int(child.trajectory_lengths[0])-2)
    raw=modality_array(child,child.get_trajectory_data(ep),'state')[anchor:anchor+1]
    payload=dict(samples[0],state=raw)
    proc=wrapper._get_processor(samples[0]['robot_tag'])
    normalized=dict(payload,state=proc.apply_state(raw))
    np.testing.assert_allclose(normalized['state'],samples[0]['state'].astype(np.float32),atol=3e-3,rtol=5e-4)
    torch.manual_seed(77)
    direct=wrapper._framework.predict_action([normalized])['normalized_actions']
    torch.manual_seed(77)
    deployed=wrapper.predict_action([payload])['actions']
    np.testing.assert_allclose(deployed,proc.unapply_actions(direct[0])[None],atol=0,rtol=0)
    report.update(status='passed',losses=losses,inference_shape=list(prediction.shape),
                  strict_reload_tensors=tensor_count,no_future_label_leakage=True,
                  trainable_parameters=trainable,frozen_parameters=frozen,
                  deployment_action_parity=True,state_normalization_parity=True)
    write_json(output/'smoke_report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--steps',type=int,default=2)
    a=p.parse_args();smoke(a.config,a.output,a.device,a.steps)
