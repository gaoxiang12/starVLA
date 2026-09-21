"""Verify the actual shared loader and a native trainer checkpoint end to end."""
import argparse
import json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import torch
from accelerate import PartialState

from examples.LiLaWAM.prepare import write_json
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from deployment.model_server.policy_wrapper import PolicyServerWrapper


def verify(config, checkpoint, output):
    PartialState()
    cfg=OmegaConf.load(config)
    dataset=get_vla_dataset(cfg.datasets.vla_data)
    assert len(dataset)==cfg.datasets.vla_data.expected_frames
    report=dict(status='running', frames=len(dataset), datasets=len(dataset.datasets),
                loader='shared LeRobot', checked=[])
    stats=json.loads(Path(cfg.datasets.vla_data.normalization_statistics_path).read_text())['aloha']
    def normalized(x,modality):
        low=np.array(stats[modality]['min'],np.float32)
        span=np.array(stats[modality]['max'],np.float32)-low
        span[span<1e-6]=1
        return 2*(x-low)/span-1
    for child in dataset.datasets:
        # Exercise current and terminal samples, all cameras and repeated tails.
        for index in (0,len(child)-1):
            ex=child[index]
            ep,anchor=child.all_steps[index]
            raw=child.get_trajectory_data(ep)
            actions=np.stack(raw['action']).astype(np.float32)
            states=np.stack(raw['observation.state']).astype(np.float32)
            expected=normalized(actions[np.minimum(anchor+np.arange(32),len(actions)-1)],'action')
            np.testing.assert_allclose(ex['action'],expected,rtol=1e-6,atol=2e-7)
            np.testing.assert_allclose(ex['state'][0],normalized(states[anchor],'state'),rtol=1e-6,atol=2e-7)
            assert len(ex['image'])==3 and all(im.size==(320,240) for im in ex['image'])
            assert ex['view_valid_mask']==[True,True,True]
            if index==len(child)-1:
                assert sum(ex['action_valid_mask'])==1 and not ex['future_frame_valid_mask'][1]
                for now,future in zip(ex['image'],ex['future_images'][0]):
                    np.testing.assert_array_equal(np.asarray(now),np.asarray(future))
        report['checked'].append(child.dataset_name)
        child.curr_traj_data=None;child.curr_traj_id=None
    child=next(c for c in dataset.datasets if c.dataset_name.endswith('blocks_ranking_rgb'))
    ex=child[0];ep,anchor=child.all_steps[0]
    state=np.asarray(child.get_trajectory_data(ep)['observation.state'].iloc[anchor],np.float32)
    wrapper=PolicyServerWrapper(checkpoint,use_bf16=True,unnorm_key='aloha')
    framework=wrapper._framework
    assert not framework.encoder.training and all(not p.requires_grad for p in framework.encoder.parameters())
    payload=dict(lang=ex['lang'],image=ex['image'],state=state,robot_tag='aloha')
    torch.manual_seed(51)
    output_actions=wrapper.predict_action([payload])['actions']
    proc=wrapper._get_processor('aloha')
    direct=dict(payload,state=proc.apply_state(state))
    torch.manual_seed(51)
    prediction=framework.predict_action([direct])['normalized_actions']
    expected=framework.postprocess_actions(np.stack([proc.unapply_actions(prediction[0])]))
    np.testing.assert_array_equal(output_actions,expected)
    assert output_actions.shape==(1,32,14) and np.isfinite(output_actions).all()
    report.update(status='passed', checkpoint=str(checkpoint), physical_action_shape=list(output_actions.shape),
                  wrapper_max_abs_error=float(np.max(np.abs(output_actions-expected))),
                  encoder_frozen=True, parameter_dtype=str(next(framework.core.parameters()).dtype),
                  metadata=wrapper.metadata)
    write_json(output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='checked'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='examples/LiLaWAM/train_files/robotwin_3view_aligned_stage1.yaml')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();verify(a.config,a.checkpoint,a.output)
