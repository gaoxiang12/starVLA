"""Real shared-loader/GPU integration check for compact flow and regression."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import types

import numpy as np
from omegaconf import OmegaConf
import torch

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def old_action_path(self, head, visual, state, memories, actions=None, action_valid_mask=None):
    queries = self._pool_visual_tokens_to_action_queries(visual, state=state, action_model=head)
    if memories is not None:
        queries = self.spatial_focus.refine_queries(queries, memories)
    return head.predict_action(queries), None, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    audit = root / 'playground/Checkpoints/gawm_grasp_precision_training_20260909'
    cfg = OmegaConf.load(audit / 'joint_train1000.yaml')
    cfg.datasets.vla_data.num_workers = 0
    mixture = get_vla_dataset(cfg.datasets.vla_data, mode='train')
    examples, sample_records = [], []
    for child_index, anchor in [(0, 20), (1, 0), (1, 73)]:
        child = mixture.datasets[child_index]
        episode = int(child.trajectory_ids[0])
        child.transforms.eval()
        ex = child._pack_sample(child.transforms(child.get_step_data(episode, anchor)))
        ex = child._attach_action_validity(ex, episode, anchor)
        ex = child._attach_future_frame_validity(ex, episode, anchor)
        ex = child._attach_spatial_supervision(ex, episode, anchor)
        ex.update(robot_tag='aloha', action_spec_id=child.action_spec_id, state_spec_id=child.state_spec_id)
        examples.append(ex)
        sample_records.append(dict(child=child_index, episode=episode, anchor=anchor,
                                   valid_actions=int(ex['action_valid_mask'].sum())))
    assert sample_records[-1]['valid_actions'] == 1
    inputs = [{k: v for k, v in ex.items() if k != 'action' and not k.startswith(('spatial_', 'future', 'pregrasp_', 'action_valid'))}
              for ex in examples]
    source = root / cfg.trainer.pretrained_checkpoint
    saved = torch.load(source, map_location='cpu', weights_only=True)
    base = build_framework(cfg)
    base.load_state_dict(saved, strict=True)
    base.cuda().float().eval()
    baseline = base.predict_action(inputs)['normalized_actions']
    base._predict_action_chunk = types.MethodType(old_action_path, base)
    previous = base.predict_action(inputs)['normalized_actions']
    np.testing.assert_array_equal(baseline, previous)
    del base
    gc.collect(); torch.cuda.empty_cache()
    reports = []
    for mode in ('flow', 'regression'):
        torch.manual_seed(42)
        config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        config.framework.name = 'GAWMCompactExpert'
        config.framework.compact_expert = dict(mode=mode, hidden_dim=384, depth=6,
            heads=6, planning_tokens=8, inference_steps=4)
        mode_dir = out / mode
        mode_dir.mkdir()
        OmegaConf.save(config, mode_dir / 'config.yaml')
        model = build_framework(config)
        loaded = model.load_state_dict(saved, strict=False)
        assert loaded.unexpected_keys and all(k.startswith('action_models.aloha.') for k in loaded.unexpected_keys)
        assert loaded.missing_keys and all(k.startswith(('action_models.aloha.', 'expert_task_projection.')) for k in loaded.missing_keys)
        common = model.state_dict()
        common_count = 0
        for name, value in saved.items():
            if name in common:
                torch.testing.assert_close(common[name], value.to(common[name]), atol=0, rtol=0)
                common_count += 1
        del common
        model.cuda().float().train()
        head = model.action_models['aloha']
        initial = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.)
        logs = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(examples=examples, optimizer_step=step)
            assert all(torch.isfinite(v).all() for v in loss.values() if torch.is_tensor(v))
            loss['action_loss'].backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            head_gradient = sum(float(p.grad.float().norm()) for p in head.parameters() if p.grad is not None)
            visual_gradient = sum(float(p.grad.float().norm()) for p in model.spatial_focus.project.parameters() if p.grad is not None)
            assert head_gradient > 0 and visual_gradient > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            logs.append(dict(step=step + 1, losses={k: float(v.detach()) for k, v in loss.items()},
                             head_gradient=head_gradient, visual_gradient=visual_gradient))
        changed = [k for k, value in head.state_dict().items() if not torch.equal(value.cpu(), initial[k])]
        assert changed
        model.eval()
        torch.manual_seed(713)
        before = model.predict_action(inputs)['normalized_actions']
        assert before.shape == (3, 16, 14) and np.isfinite(before).all()
        weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        checkpoint = mode_dir / 'pytorch_model.pt'
        torch.save(weights, checkpoint)
        parameters = sum(p.numel() for p in model.parameters())
        del optimizer, model, head, weights, initial, loss
        gc.collect(); torch.cuda.empty_cache()
        rebuilt = build_framework(config)
        reloaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
        rebuilt.load_state_dict(reloaded, strict=True)
        rebuilt.cuda().float().eval()
        torch.manual_seed(713)
        after = rebuilt.predict_action(inputs)['normalized_actions']
        np.testing.assert_array_equal(before, after)
        reports.append(dict(mode=mode, optimizer_steps=3, parameters=parameters,
                            inherited_tensors_exact=common_count, discarded_old_head_keys=loaded.unexpected_keys,
                            initialized_keys=loaded.missing_keys, changed_head_tensors=changed, logs=logs,
                            save_reload_max_difference=float(np.abs(before-after).max())))
        del rebuilt, reloaded
        gc.collect(); torch.cuda.empty_cache()
    report = dict(state='real_data_gpu_optimizer_reload_verified', samples=sample_records,
                  baseline_action_hook_max_difference=float(np.abs(baseline-previous).max()),
                  variants=reports, source_checkpoint=str(source),
                  source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__), root/'starVLA/model/framework/WM4A/GAWM.py',
                       root/'starVLA/model/framework/WM4A/GAWMCompactExpert.py',
                       root/'starVLA/model/modules/action_model/CompactFlowActionHead.py')},
                  limitations=['Three AdamW updates on three train samples, not formal convergence or grasp evaluation.',
                               'Outer BF16 autocast with existing GAWM precision scopes; not a DeepSpeed smoke.',
                               'Flow training L1 is a noised-action denoising estimate, not deployment action error.',
                               'Reload check uses the real model factory, not yet PolicyServerWrapper.'])
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('variants','source_sha256')}), flush=True)


if __name__ == '__main__':
    main()
