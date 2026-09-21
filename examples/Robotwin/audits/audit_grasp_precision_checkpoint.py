"""Strictly inspect a saved Cartesian checkpoint through the deployment loader."""
import argparse
import json
from pathlib import Path

import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from examples.Robotwin.audits.run_grasp_lift_development import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    wrapper=PolicyServerWrapper(str(args.checkpoint.resolve()),device='cpu',use_bf16=False,unnorm_key='aloha')
    model=wrapper._framework
    assert type(model).__name__=='GAWMCartesian'
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    loaded=model.state_dict()
    # The general loader warns and drops incompatible keys. This experiment
    # requires exact structure and values, so verify that nothing was dropped.
    assert set(saved)==set(loaded)
    for name,value in saved.items():
        assert value.shape==loaded[name].shape,name
        assert torch.isfinite(value).all(),name
        torch.testing.assert_close(loaded[name],value.to(loaded[name].dtype),atol=0,rtol=0)
    head=model.action_models['aloha']
    assert all(not p.requires_grad for p in head.action_projection.parameters())
    last=head.pose_projection.layers[-1]
    assert last.weight.abs().sum()>0
    metadata=wrapper.metadata
    assert metadata['action_specs']['aloha']==dict(action_spec_id='aloha_dual_joint_contgrip_next_recorded_14',action_dim=14,action_horizon=16,state_dim=14)
    report=dict(state='strict_deployment_checkpoint_load_verified',checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=digest(args.checkpoint),loaded_tensors=len(loaded),
        parameter_count=sum(p.numel() for p in model.parameters()),
        added_pose_head_weight_norm=float(last.weight.detach().norm()),
        added_pose_head_bias_norm=float(last.bias.detach().norm()),
        deployment_metadata=metadata,missing_or_dropped_keys=[],device='cpu',
        inference_executed=False,note='Real trained weights loaded through deployment factory and normalization alias, then compared exactly. No GPU rollout or grasp precision claim.')
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='deployment_metadata'}),flush=True)


if __name__=='__main__':
    main()
