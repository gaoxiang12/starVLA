"""Compare GAWM's GAWM-L front-end against a local upstream vla_model_fm.py.

No source download or training. Example:
PYTHONPATH=. .venv/bin/python scripts/check_gawm_l_vision_parity.py \
  --upstream-file .cache/gawm_l_alignment/models/vla_model_fm.py \
  --output .cache/gawm_l_alignment/parity.json
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from starVLA.model.modules.gawm_l_vision import GAWMLVisualPooler


def compare(upstream_file, output):
    path=Path(upstream_file)
    # Only the two reviewed visual primitives are needed, not upstream policy,
    # environment imports or its action/world-model implementations.
    tree=ast.parse(path.read_text())
    classes=[node for node in tree.body if isinstance(node,ast.ClassDef)
             and node.name in ('MultiLayerConcatFusion','VisualFeatureAdapter')]
    if len(classes)!=2:
        raise ValueError('Expected the official fusion and visual adapter definitions')
    namespace={'torch':torch,'nn':nn}
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(path),'exec'),namespace)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    local=GAWMLVisualPooler(1024,384,3,64,3,768,4,8).eval()
    fusion=namespace['MultiLayerConcatFusion'](1024,3,1024,'linear',True).eval()
    adapter=namespace['VisualFeatureAdapter'](1024,768,64,8,4,dropout=0.).eval()
    fusion.load_state_dict(local.fusion.state_dict(),strict=True)
    adapter.load_state_dict(local.adapter.state_dict(),strict=True)
    values=torch.randn(1,1,3,3,305,1024)
    actual=local.official_tokens(values)
    features=[values[:,:,:,i].reshape(3,305,1024) for i in range(3)]
    expected=adapter(fusion(features)).reshape(1,1,3,64,768)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-6)
    actual.square().mean().backward();expected.square().mean().backward()
    max_gradient_error=0.
    for part,reference in [(local.fusion,fusion),(local.adapter,adapter)]:
        for (name,p),(ref_name,q) in zip(part.named_parameters(),reference.named_parameters()):
            assert name==ref_name
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=2e-6)
            max_gradient_error=max(max_gradient_error,float((p.grad-q.grad).abs().max()))
    result=dict(status='passed',upstream_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                dimensions=dict(dino=1024,layers=3,tokens_including_prefix=305,views=3,queries=64,adapter_dim=768,depth=4,heads=8),
                forward_max_abs=float((actual-expected).detach().abs().max()),gradient_max_abs=max_gradient_error,
                scope='Official per-view fusion and adapter; GAWM bridge is deliberately separate')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-file',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    compare(args.upstream_file,args.output)
