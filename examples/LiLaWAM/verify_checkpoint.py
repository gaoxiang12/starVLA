"""Verify real trainer checkpoints through the standard policy-server wrapper."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from examples.LiLaWAM.prepare import modality_array, write_json
from examples.LiLaWAM.smoke import packed


def verify(checkpoint, device):
    torch.set_num_threads(4)
    checkpoint=Path(checkpoint).absolute();root=checkpoint.parent.parent
    cfg=OmegaConf.load(root/'config.yaml')
    cfg.datasets.vla_data.normalization_statistics_path=str(root/'dataset_statistics.json')
    dataset=get_vla_dataset(cfg.datasets.vla_data,mode='validation')
    wrapper=PolicyServerWrapper(str(checkpoint),device=device,use_bf16=False,
                                unnorm_key=dataset.datasets[0].tag)
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    current=wrapper._framework.state_dict()
    assert set(weights)==set(current)
    for name,tensor in current.items():
        torch.testing.assert_close(tensor.cpu(),weights[name].to(tensor.dtype),rtol=0,atol=0)
    report={'checkpoint':str(checkpoint),'strict_tensors':len(weights),'datasets':[],'status':'passed'}
    del weights,current
    for child in dataset.datasets:
        ep=int(child.trajectory_ids[0]);anchor=min(32,int(child.trajectory_lengths[0])-2)
        sample=packed(child,ep,anchor)
        raw=modality_array(child,child.get_trajectory_data(ep),'state')[anchor:anchor+1]
        payload=dict(sample,state=raw)
        proc=wrapper._get_processor(child.tag)
        normalized=dict(payload,state=proc.apply_state(raw))
        np.testing.assert_allclose(normalized['state'],sample['state'].astype(np.float32),atol=3e-3,rtol=5e-4)
        torch.manual_seed(77);direct=wrapper._framework.predict_action([normalized])['normalized_actions']
        torch.manual_seed(77);deployed=wrapper.predict_action([payload],unnorm_key=child.tag)['actions']
        np.testing.assert_array_equal(deployed,proc.unapply_actions(direct[0])[None])
        poison=copy.copy(payload);poison.update(action='unused',future_images='unused',future_frame_valid_mask='unused')
        torch.manual_seed(77);other=wrapper.predict_action([poison],unnorm_key=child.tag)['actions']
        np.testing.assert_array_equal(other,deployed)
        assert np.isfinite(deployed).all()
        report['datasets'].append(dict(name=child.dataset_name,shape=list(deployed.shape),
            state_loader_server_parity=True,action_server_parity=True,no_future_label_leakage=True))
    write_json(root/'serving_audit.json',report)
    print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();verify(a.checkpoint,a.device)
