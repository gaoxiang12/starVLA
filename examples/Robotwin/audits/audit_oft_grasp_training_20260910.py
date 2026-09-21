"""Real-loader, gradient, and strict deployment checks for the small OFT policy."""
import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from examples.Robotwin.audits.prepare_oft_grasp_training_20260910 import OUT, OLD, SHARED, ROOT, digest
from examples.Robotwin.audits.run_grasp_lift_development import save
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

ORDER = [0,1,2,3,4,5,7,8,9,10,11,12,6,13]


def packed(child, episode, anchor):
    child.transforms.eval()
    sample = child._pack_sample(child.transforms(child.get_step_data(episode, anchor)))
    sample = child._attach_action_validity(sample, episode, anchor)
    sample = child._attach_future_frame_validity(sample, episode, anchor)
    return child._attach_spatial_supervision(sample, episode, anchor)


def data_audit():
    target = OUT/'data_audit.json'
    assert not target.exists()
    cfg = OmegaConf.load(OUT/'data_only1000.yaml')
    original_cfg = OmegaConf.load(OLD/'joint_train1000.yaml')
    candidate = OmegaConf.to_container(cfg, resolve=True)
    reference = OmegaConf.to_container(original_cfg, resolve=True)
    for key in ('run_id', 'run_root_dir'):
        candidate[key] = reference[key]
    candidate['datasets']['vla_data']['training_anchor_manifest'] = reference['datasets']['vla_data']['training_anchor_manifest']
    assert candidate == reference, 'Data-only configuration changed more than source windows/output paths'
    mixture = get_vla_dataset(cfg.datasets.vla_data, mode='train')
    original, correction = mixture.datasets
    split = json.loads((OLD/'split.json').read_text())
    rows = {r['episode_index']:r for r in json.loads((OUT/'first_grasp_anchors.json').read_text())['episodes']}
    assert set(original.trajectory_ids) == set(split['train_episode_ids']) == set(rows)
    assert set(correction.trajectory_ids) == set(range(20))
    np.testing.assert_allclose(mixture.dataset_sampling_weights, [.8,.2], atol=1e-12)
    assert len(original.all_steps) == 55536
    for episode, step in original.all_steps:
        assert step+16 < rows[int(episode)]['target_end_exclusive']
    counts, closed, seen, raw_cache = [0,0], 0, [set(),set()], {}

    def raw(episode):
        if episode not in raw_cache:
            raw_cache[episode] = np.asarray(pq.read_table(rows[episode]['parquet_path'], columns=['action'])['action'].to_pylist())
        return raw_cache[episode]

    for index in range(10000):
        child, ep, step = mixture.sample_step(index)
        ep, step = int(ep), int(step)
        which = int(child is correction)
        counts[which] += 1; seen[which].add(ep)
        if child is original:
            assert step+16 < rows[ep]['target_end_exclusive']
            arm = 6 if rows[ep]['active_arm'] == 'left' else 13
            closed += int(raw(ep)[step+1,arm] < .2)
    assert seen == [set(rows), set(range(20))]
    assert .18 < counts[1]/10000 < .22 and .4 < closed/counts[0] < .55
    checkpoint = ROOT/cfg.trainer.pretrained_checkpoint
    proc = PolicyNormProcessor(str(checkpoint), unnorm_key='aloha')
    samples = []
    for ep in sorted(rows)[::24]:
        r = rows[ep]
        for anchor in sorted({0, r['first_close_frame']+4, r['target_end_exclusive']-17}):
            s = packed(original, ep, anchor)
            assert s['action'].shape == (16,14) and s['state'].shape == (1,14)
            assert s['action_valid_mask'].all() and s['future_frame_valid_mask'].all()
            assert s['lang'] == 'blocks ranking rgb'
            recovered = proc.unapply_actions(s['action'])
            expected = raw(ep)[anchor+np.arange(1,17)][:,ORDER]
            np.testing.assert_allclose(recovered, expected, atol=.005, rtol=0)
            expected_state = proc.apply_state(raw(ep)[anchor:anchor+1,ORDER])
            np.testing.assert_allclose(s['state'], expected_state, atol=.002, rtol=0)
            samples.append(dict(episode=ep, anchor=anchor,
                                first_grippers=s['action'][0,12:].astype(float).tolist()))
    val_cfg = OmegaConf.create(OmegaConf.to_container(cfg.datasets.vla_data, resolve=True))
    val_cfg.episode_split = 'validation'
    validation = get_vla_dataset(val_cfg, mode='validation')
    assert len(validation.datasets) == 1
    assert set(validation.datasets[0].trajectory_ids) == set(split['validation_episode_ids'])
    save(target, dict(state='shared_loader_first_grasp_windows_verified', legal_original_anchors=55536,
        random_draws=10000, dataset_draw_counts=counts, original_closed_first_action_draws=closed,
        original_closed_first_action_fraction=closed/counts[0], boundary_samples=samples,
        validation_episode_ids=sorted(map(int, validation.datasets[0].trajectory_ids)),
        normalization_sha256=digest(cfg.datasets.vla_data.normalization_statistics_path),
        data_only_config_differences=['training_anchor_manifest','run_id','run_root_dir']))
    print(target, flush=True)


def checkpoint_audit(checkpoint, output):
    assert not output.exists()
    wrapper = PolicyServerWrapper(str(checkpoint.resolve()), device='cuda', use_bf16=False, unnorm_key='aloha')
    model = wrapper._framework
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    loaded = model.state_dict()
    assert set(saved) == set(loaded)
    for name, value in saved.items():
        assert value.shape == loaded[name].shape and torch.isfinite(value).all(), name
        torch.testing.assert_close(loaded[name].cpu(), value.to(dtype=loaded[name].dtype), atol=0, rtol=0)
    cfg_path = checkpoint.parents[1]/'config.full.yaml'
    if not cfg_path.is_file():
        cfg_path = checkpoint.parents[1]/'config.yaml'
    cfg = OmegaConf.load(cfg_path)
    mixture = get_vla_dataset(cfg.datasets.vla_data, mode='train')
    child = mixture.datasets[0]
    episode = int(child.trajectory_ids[0])
    row, = [r for r in json.loads((OUT/'first_grasp_anchors.json').read_text())['episodes'] if r['episode_index']==episode]
    anchor = row['first_close_frame']+4
    sample = packed(child, episode, anchor)
    raw = np.asarray(pq.read_table(row['parquet_path'], columns=['action'])['action'].to_pylist())[anchor:anchor+1,ORDER]
    proc = wrapper._get_processor('aloha')
    baseline_cfg = OmegaConf.load(OUT/'data_only1000.yaml')
    reference = PolicyNormProcessor(str(ROOT/baseline_cfg.trainer.pretrained_checkpoint), unnorm_key='aloha')
    random = np.random.default_rng(713).uniform(-1.5,1.5,(1024,14)).astype(np.float32)
    np.testing.assert_array_equal(proc.unapply_actions(random), reference.unapply_actions(random))
    payload = dict(image=sample['image'], native_images=sample['native_images'], state=raw,
                   lang=sample['lang'], robot_tag='aloha')
    direct = dict(payload, state=proc.apply_state(raw)) if model.expects_normalized_state else payload
    normalized = model.predict_action([direct])['normalized_actions']
    expected = np.stack([proc.unapply_actions(normalized[0])])
    actual = wrapper.predict_action([payload], unnorm_key='aloha')['actions']
    assert actual.shape == (1,16,14) and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)
    isolation = None
    if type(model).__name__ == 'GAWMOFT':
        poisoned = dict(payload, state=np.full((1,14), np.nan), action=np.full((16,14), np.nan),
                        future_images='forbidden future observation', native_images='not used')
        other = wrapper.predict_action([poisoned], unnorm_key='aloha')['actions']
        np.testing.assert_array_equal(actual, other)
        isolation = 'Exact unchanged outputs with poisoned state/actions/future/native extras'
    save(output, dict(state='strict_weights_normalization_and_real_payload_verified',
        checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest(checkpoint),
        framework=type(model).__name__, loaded_tensors=len(loaded),
        parameters=sum(p.numel() for p in model.parameters()), metadata=wrapper.metadata,
        reference_normalization_max_difference=0., direct_wrapper_max_difference=0.,
        normalization_random_samples=1024, test_episode=episode, test_anchor=anchor,
        state_future_label_isolation=isolation, action_shape=list(actual.shape)))
    print(output, flush=True)


def integration():
    assert json.loads((OUT/'data_audit.json').read_text())['state']=='shared_loader_first_grasp_windows_verified'
    target = OUT/'integration.json'
    assert not target.exists()
    cfg = OmegaConf.load(OUT/'oft_smoke20.yaml')
    torch.manual_seed(42)
    model = build_framework(cfg)
    old = torch.load(ROOT/cfg.trainer.pretrained_checkpoint, map_location='cpu', weights_only=True)
    state = model.state_dict()
    matching = {k:v for k,v in old.items() if k in state and v.shape == state[k].shape}
    result = model.load_state_dict(matching, strict=False)
    assert all(k.startswith('action_models.aloha.') for k in result.missing_keys)
    assert 'action_models.aloha.action_queries.weight' in matching
    for k,v in matching.items():
        loaded_value = model.state_dict()[k]
        torch.testing.assert_close(loaded_value, v.to(dtype=loaded_value.dtype), atol=0, rtol=0)
    torch.testing.assert_close(model.embodiment_embedding.weight,
        old['embodiment_embedding.weight'].to(model.embodiment_embedding.weight), atol=0, rtol=0)
    del state, old
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(build_param_lr_groups(model,cfg), lr=1e-4)
    mixture = get_vla_dataset(cfg.datasets.vla_data, mode='train')
    child = mixture.datasets[0]
    rows = json.loads((OUT/'first_grasp_anchors.json').read_text())['episodes']
    examples = [packed(child,r['episode_index'],r['first_close_frame']+4) for r in rows[:4]]
    losses, gradients = [], []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(examples)['action_loss']
        assert torch.isfinite(loss)
        loss.backward()
        encoder = sum(float(p.grad.abs().sum()) for p in model.backbone.encoder.parameters() if p.grad is not None)
        head = sum(float(p.grad.abs().sum()) for p in model.action_models['aloha'].parameters() if p.grad is not None)
        assert encoder > 0 and head > 0
        assert all(p.grad is None for p in model.world_model.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        losses.append(float(loss.detach())); gradients.append(dict(encoder=encoder,head=head))
    run = SHARED/'integration3'
    (run/'checkpoints').mkdir(parents=True, exist_ok=False)
    checkpoint = run/'checkpoints/pytorch_model.pt'
    torch.save({k:v.cpu() for k,v in model.state_dict().items()},checkpoint)
    OmegaConf.save(cfg,run/'config.yaml')
    (run/'dataset_statistics.json').write_bytes(Path(cfg.datasets.vla_data.normalization_statistics_path).read_bytes())
    parameter_count = sum(p.numel() for p in model.parameters())
    shared_count = len(matching)
    assert parameter_count < 200_000_000, 'Keep this experiment below the requested small-model scale'
    del model, optimizer, matching, examples, loss
    torch.cuda.empty_cache()
    checkpoint_audit(checkpoint,OUT/'integration_deployment.json')
    save(target,dict(state='real_data_gradients_optimizer_strict_reload_verified', losses=losses,
        gradient_sums=gradients, shared_tensors_loaded=shared_count,
        new_missing_keys=result.missing_keys, parameters=parameter_count,
        note='Engineering check only; three updates are not a learned grasp result.'))
    print(target,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['data','integration','checkpoint'],required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='data':data_audit()
    elif args.mode=='integration':integration()
    else:
        if args.checkpoint is None or args.output is None:parser.error('checkpoint/output required')
        checkpoint_audit(args.checkpoint,args.output)
