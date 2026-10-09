"""Measure actual GAWM latency/kernel dispatch and SDPA backend eligibility.

No optimizer updates, parameter conversion, or checkpoint writes are made.
Raw BF16 kernel comparisons are explicitly separate from full-policy timings.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F

from starVLA.dataloader.robotwin_official_hdf5 import RoboTwinOfficialDataset
from starVLA.model.framework.WM4A.GAWM import GAWM
from starVLA.training.recipe import prepare_parameter_precision


def measure(function, repeats=20, warmup=5):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed=[]
    for _ in range(repeats):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record();function();end.record();end.synchronize()
        elapsed.append(begin.elapsed_time(end))
    return dict(median_ms=statistics.median(elapsed), p95_ms=float(np.percentile(elapsed,95)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),repeats=repeats)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)  # Same as RoboTwin server.
    cfg=OmegaConf.load(args.run/'config.full.yaml')
    model=GAWM(cfg)
    prepare_parameter_precision(model,cfg)
    step=int(cfg.trainer.max_train_steps)
    weights=torch.load(args.run/'checkpoints'/f'steps_{step}_pytorch_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(weights,strict=True);del weights
    model.cuda().eval()
    dataset=RoboTwinOfficialDataset(cfg.datasets.vla_data)
    examples=[dataset[i*max(1,len(dataset)//4)] for i in range(4)]
    result=dict(torch=torch.__version__,device=torch.cuda.get_device_name(),run=str(args.run),
        parameter_dtype=str(next(model.visual_token_pooler.parameters()).dtype),
        backbone_attention=model.backbone.encoder.config._attn_implementation,
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        notes=['Policy timing includes tensor preprocessing and CPU action transfer; excludes JPEG decode/network/simulator.',
               'Training measurement is forward+backward only, with no optimizer/DDP or data loading.',
               'BF16 kernel microbenchmarks do not change the policy core precision.'])
    with torch.inference_mode():
        result['policy_batch1']=measure(lambda:model.predict_action(examples[:1]))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
            model.predict_action(examples[:1]);torch.cuda.synchronize()
        result['policy_attention_operators']=[dict(name=e.key,calls=e.count) for e in profiler.key_averages()
                                            if any(word in e.key.lower() for word in ('attention','flash','sdpa'))]
    model.train()
    def train_step():
        model.zero_grad(set_to_none=True)
        model(examples)['action_loss'].backward()
    result['training_forward_backward_batch4']=measure(train_step,repeats=10,warmup=3)
    model.zero_grad(set_to_none=True)
    kernel_rows=[]
    # Actual baseline shapes: DINO S+, adapter cross attention, WM, ACT cross.
    for name,heads,lq,lk,dim in [('dino',6,305,305,64),('adapter_cross',8,64,305,96),
                                  ('world_model',6,384,384,64),('act_cross',8,32,386,48)]:
        q=torch.randn(4,heads,lq,dim,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(4,heads,lk,dim,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        with sdpa_kernel(SDPBackend.MATH):
            reference=F.scaled_dot_product_attention(q,k,v)
        for backend in (None,SDPBackend.MATH,SDPBackend.FLASH_ATTENTION):
            row=dict(module=name,backend='auto' if backend is None else backend.name,dtype='bfloat16')
            try:
                with nullcontext() if backend is None else sdpa_kernel(backend):
                    fn=lambda:F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)
                    actual=fn();row.update(measure(fn,repeats=50),max_abs_error_vs_math=(actual-reference).abs().max().item(),status='supported')
            except RuntimeError as error:
                row.update(status='unsupported',reason=str(error))
            kernel_rows.append(row)
    result['bf16_kernel_microbenchmarks']=kernel_rows
    result['completed_at']=time.time()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
