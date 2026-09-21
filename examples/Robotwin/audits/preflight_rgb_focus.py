"""Verify original checkpoint preservation and native-image/label data wiring."""
import gc
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from starVLA.model.framework.base_framework import build_framework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def main():
    root=Path(__file__).resolve().parents[3]
    out=root/'playground/Checkpoints/gawm_rgb_focus_20260907'
    original=torch.load(root/'playground/Checkpoints/gawm_s_robotwin_continuous_next_49tasks_40k_20260905/final_model/pytorch_model.pt',map_location='cpu',weights_only=True)
    reports=[]
    for variant in ('control','dense','local'):
        cfg=OmegaConf.load(root/f'examples/Robotwin/train_files/starvla_gawm_rgb_focus_{variant}.yaml')
        torch.manual_seed(42)
        model=build_framework(cfg)
        migrated=model.remap_checkpoint_state_dict(original)
        result=model.load_state_dict(migrated,strict=False)
        assert not result.unexpected_keys
        assert all(k.startswith('spatial_focus.') for k in result.missing_keys)
        if variant=='control': assert not result.missing_keys
        state=model.state_dict()
        assert len(original)==540
        assert all(torch.equal(state[k],v) for k,v in original.items())
        complete=dict(state)
        model.load_state_dict(complete,strict=True)
        reports.append(dict(variant=variant,original_tensors_preserved=len(original),new_tensors=len(result.missing_keys),
                            total_parameters=sum(p.numel() for p in model.parameters()),
                            added_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith('spatial_focus.'))))
        del model,state,complete,migrated
        gc.collect()
    data=get_vla_dataset(cfg.datasets.vla_data,mode='train',seed=42)
    examples=[data[(0,i)] for i in range(16)]
    assert all(x['native_images'][0].size==(320,240) for x in examples)
    assert all(x['image'][0].size==(224,224) for x in examples)
    assert all(np.asarray(x['action']).shape==(16,14) for x in examples)
    assert set(data.datasets[0].trajectory_ids).isdisjoint(range(0,1000,20))
    assert len(data.datasets[0].trajectory_ids)==950
    (out/'preflight.json').write_text(json.dumps(dict(models=reports,native_images_wh=[320,240],training_episodes=950,
            samples_checked=16,valid_spatial_labels=sum(np.asarray(x['spatial_target_valid']).sum().item() for x in examples)),indent=2)+'\n')
    print(reports,flush=True)


if __name__=='__main__': main()
