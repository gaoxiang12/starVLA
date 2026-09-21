"""Compare unchanged LiLa objectives with batched, masked visual computation."""
import argparse
import copy
import gc
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from examples.LiLaWAM.prepare import write_json
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.WM4A.LiLaWAMTrain import LiLaWAMTrain


def profile(config, output):
    torch.set_num_threads(4);torch.manual_seed(42)
    cfg=OmegaConf.load(config)
    mix=get_vla_dataset(cfg.datasets.vla_data,mode='validation')
    samples=[mix[(i%len(mix.datasets),i*31)] for i in range(64)]
    model=LiLaWAMTrain(cfg).cuda().to(torch.bfloat16).train()
    results=[]
    for batch,chunk,efficient in [(8,3,False),(8,32,True),(16,32,True),(32,64,True),(64,64,True)]:
        model.encoder_batch_size=chunk
        model.efficient_views=model.core.efficient_views=efficient
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            elapsed=[]
            for i in range(4):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize()
                start=time.perf_counter();torch.manual_seed(91)
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(samples[:batch])
                assert torch.isfinite(out['action_loss'])
                out['action_loss'].backward();torch.cuda.synchronize()
                elapsed.append(time.perf_counter()-start)
            result=dict(batch=batch,encoder_chunk=chunk,efficient_views=efficient,
                seconds_per_microbatch=float(np.median(elapsed[1:])),
                estimated_seconds_per_128_samples=float(np.median(elapsed[1:])*128/batch),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                loss=float(out['action_loss'].detach()))
            if batch==8:
                tensors={n:p.grad.detach().float().clone() for n,p in model.named_parameters()
                         if p.grad is not None and n.startswith(('core.heads.franka','core.fusion'))}
                if not efficient:
                    reference_loss=result['loss'];reference_grads=tensors
                else:
                    result['loss_relative_difference']=abs(result['loss']-reference_loss)/max(abs(reference_loss),1e-8)
                    result['selected_gradient_relative_l2']=float((sum((tensors[n]-v).square().sum() for n,v in reference_grads.items())/
                        sum(v.square().sum() for v in reference_grads.values())).sqrt())
            del out
        except torch.OutOfMemoryError:
            result=dict(batch=batch,encoder_chunk=chunk,efficient_views=efficient,error='OOM')
            model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        results.append(result);print(result,flush=True);write_json(output,results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();profile(a.config,a.output)
