"""Real loader, state sensitivity, gradients and deployment parity checks."""
import argparse
import copy
import gc
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from examples.RobotwinEndPose.prepare import ROOT, OUT, RUNS, DATASET, BASE, save, digest
from starVLA.robotwin_feedback import FEEDBACK
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.policy_wrapper import PolicyServerWrapper


def packed(child, ep, anchor):
    child.transforms.eval()
    sample = child._pack_sample(child.transforms(child.get_step_data(ep, anchor)))
    sample = child._attach_action_validity(sample, ep, anchor)
    sample = child._attach_future_frame_validity(sample, ep, anchor)
    return child._attach_spatial_supervision(sample, ep, anchor)


def raw_feedback(ep, anchor, variant):
    path = f'/data/gaoxiang/RoboTwinGenerated_raw/Clean/blocks_ranking_rgb/demo_clean/data/episode{ep}.hdf5'
    with h5py.File(path) as f:
        observation = dict(endpose={k: f['endpose'][k][anchor] for k in
            ('left_endpose','right_endpose','left_gripper','right_gripper')},
            joint_action={'vector': f['joint_action/vector'][anchor]})
    return FEEDBACK[variant](observation)


def audit(variant, checkpoint=None, device='cpu', optimize=False):
    torch.set_num_threads(4); torch.manual_seed(42)
    cfg = OmegaConf.load(OUT/f'{variant}_smoke.yaml')
    if checkpoint is None:
        # A metadata-only deployment bundle; large weights stay in shared storage.
        bundle = OUT/f'{variant}_initial'
        (bundle/'final_model').mkdir(parents=True, exist_ok=True)
        checkpoint = bundle/'final_model/pytorch_model.pt'
        if not checkpoint.is_symlink(): checkpoint.symlink_to(RUNS/'initial/pytorch_model.pt')
        OmegaConf.save(cfg, bundle/'config.yaml')
        (bundle/'dataset_statistics.json').write_text((OUT/f'{variant}_statistics.json').read_text())
    checkpoint = Path(checkpoint).absolute()  # Keep bundle path, not symlink target.
    proc = PolicyNormProcessor(str(checkpoint), unnorm_key='aloha')
    mixture = get_vla_dataset(cfg.datasets.vla_data, mode='validation')
    child, = mixture.datasets
    split = json.loads((OUT/'split.json').read_text())
    assert set(child.trajectory_ids) == set(split['train_episode_ids'])
    vcfg = copy.deepcopy(cfg.datasets.vla_data); vcfg.episode_split = 'validation'
    valchild, = get_vla_dataset(vcfg, mode='validation').datasets
    assert set(valchild.trajectory_ids) == set(split['validation_episode_ids'])
    original_cfg = OmegaConf.load(BASE/'gawm_joint.yaml')
    original, = get_vla_dataset(original_cfg.datasets.vla_data, mode='validation').datasets
    reference_checkpoint = Path(OmegaConf.load(BASE/'gawm_warmup.yaml').trainer.pretrained_checkpoint)
    if not reference_checkpoint.is_absolute(): reference_checkpoint = ROOT/reference_checkpoint
    reference = PolicyNormProcessor(str(reference_checkpoint), unnorm_key='aloha')
    values = np.random.default_rng(123).uniform(-1.5,1.5,(1024,14)).astype(np.float32)
    np.testing.assert_array_equal(proc.unapply_actions(values),reference.unapply_actions(values))
    lengths = dict(zip(child.trajectory_ids,child.trajectory_lengths))
    checks = []
    max_state_difference = 0.
    for ep in sorted(lengths)[::48]:
        for anchor in (0, int(lengths[ep])//2, int(lengths[ep])-2):
            sample, old = packed(child,ep,anchor), packed(original,ep,anchor)
            np.testing.assert_array_equal(sample['action'],old['action'])
            np.testing.assert_array_equal(sample['action_valid_mask'],old['action_valid_mask'])
            np.testing.assert_array_equal(sample['future_frame_valid_mask'],old['future_frame_valid_mask'])
            for a,b in zip(sample['image'],old['image']): np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
            expected_state = proc.apply_state(raw_feedback(ep,anchor,variant))[None].astype(sample['state'].dtype)
            max_state_difference = max(max_state_difference,float(np.max(np.abs(sample['state'].astype(np.float32)-expected_state.astype(np.float32)))))
            np.testing.assert_allclose(sample['state'],expected_state,atol=5e-4,rtol=5e-4)
            assert np.isfinite(sample['state']).all()
            checks.append(dict(episode=int(ep),anchor=anchor))
    # Real indexed and mixture sampling must both use the registered state schema.
    assert child[0]['state'].shape == (1,16) and mixture[(0,0)]['state'].shape == (1,16)
    for ep in valchild.trajectory_ids:
        sample = packed(valchild,ep,0)
        expected_state = proc.apply_state(raw_feedback(ep,0,variant))[None].astype(sample['state'].dtype)
        max_state_difference = max(max_state_difference,float(np.max(np.abs(sample['state'].astype(np.float32)-expected_state.astype(np.float32)))))
        np.testing.assert_allclose(sample['state'],expected_state,atol=5e-4,rtol=5e-4)
    wrapper = PolicyServerWrapper(str(checkpoint),device=device,use_bf16=False,unnorm_key='aloha')
    model = wrapper._framework
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    assert set(saved) == set(model.state_dict())
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), saved[name].to(value.dtype), atol=0,rtol=0)
    count = len(saved); del saved
    ep = int(child.trajectory_ids[0]); sample = packed(child,ep,32)
    payload = dict(sample,state=raw_feedback(ep,32,variant)[None])
    direct_payload = dict(payload,state=proc.apply_state(payload['state']))
    direct = model.predict_action([direct_payload])['normalized_actions']
    deployed = wrapper.predict_action([payload],unnorm_key='aloha')['actions']
    np.testing.assert_array_equal(deployed,proc.unapply_actions(direct[0])[None])
    assert deployed.shape == (1,16,14) and np.isfinite(deployed).all()
    poisoned = dict(payload, action='not an inference input', future_images='not an inference input',
        spatial_target_xy='not an inference input', spatial_target_valid='not an inference input')
    np.testing.assert_array_equal(deployed,wrapper.predict_action([poisoned],unnorm_key='aloha')['actions'])
    changed = dict(direct_payload,state=np.asarray(direct_payload['state']).copy())
    changed['state'][...,0] += .2
    sensitivity = float(np.max(np.abs(direct-model.predict_action([changed])['normalized_actions'])))
    assert sensitivity > 0
    losses = []
    if optimize:
        model.train()
        for name,p in model.named_parameters():
            p.requires_grad_(name.startswith(('action_models.aloha.state_projection.','spatial_focus.query.')))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = model([sample])['action_loss']
            assert torch.isfinite(loss); loss.backward()
            for prefix in ('action_models.aloha.state_projection.','spatial_focus.query.'):
                grads = [p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                assert sum(float(g.abs().sum()) for g in grads) > 0, f'No effective gradient: {prefix}'
            optimizer.step(); losses.append(float(loss.detach()))
    report = dict(state='cpu_verified_gpu_pending' if device=='cpu' else 'gpu_verified',
        variant=variant,checkpoint=str(checkpoint),device=device,loaded_tensors=count,
        train_episodes=len(child.trajectory_ids),validation_episodes=len(valchild.trajectory_ids),
        boundary_checks=checks,action_normalization_samples=1024,action_labels_images_masks_unchanged=True,
        state_loader_deployment_max_difference=max_state_difference,
        state_loader_deployment_tolerance=dict(atol=5e-4,rtol=5e-4,note='Loader casts to float16'),
        direct_wrapper_max_difference=0.,
        normalized_action_state_sensitivity=sensitivity,gradient_smoke_losses=losses,
        output_shape=list(deployed.shape),formal_training_started=False)
    report_path = OUT/f'{variant}_audit_{device}.json' if checkpoint.parent.parent.name.endswith('_initial') else checkpoint.parent.parent/f'feedback_audit_{device}.json'
    save(report_path,report); print(json.dumps(report,indent=2))
    del wrapper,model;gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=list(FEEDBACK),required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--optimize',action='store_true')
    args = parser.parse_args()
    audit(args.variant,args.checkpoint,args.device,args.optimize)
