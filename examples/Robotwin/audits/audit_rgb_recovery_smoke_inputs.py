"""Check deterministic smoke sampling and the full recovery command range."""
import itertools
import json
import math
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from examples.Robotwin.audits.convert_rgb_recovery_pilot import ROOT, CAMP
from starVLA.dataloader.lerobot_datasets import EmbodimentBatchSampler, get_vla_dataset


def main():
    output = CAMP / 'smoke_input_audit.json'
    assert not output.exists()
    cfg = OmegaConf.load(ROOT / 'examples/Robotwin/train_files/starvla_gawm_rgb_recovery_mix_smoke.yaml')
    mix = get_vla_dataset(cfg.datasets.vla_data, mode='train')
    sampler = EmbodimentBatchSampler(mix, batch_size=2, seed=cfg.seed,
        embodiment_weights=cfg.datasets.vla_data.embodiment_sampling_weights)
    count = int(cfg.trainer.max_train_steps * cfg.trainer.gradient_accumulation_steps)
    batches = list(itertools.islice(sampler, count))
    rows = []
    for microbatch, indices in enumerate(batches):
        for dataset_index, sample_index in indices:
            child, episode, anchor = mix.sample_step(sample_index, dataset_index=dataset_index)
            rows.append(dict(microbatch=microbatch, dataset_index=dataset_index,
                             sample_index=sample_index, episode=int(episode), anchor=int(anchor)))
    recovery = [r for r in rows if r['dataset_index'] == 1]
    assert len(rows) == 80 and len(recovery) >= 2
    checkpoint = ROOT / cfg.trainer.pretrained_checkpoint
    norm = PolicyNormProcessor(str(checkpoint), unnorm_key='aloha')
    result = json.loads((CAMP / 'cases/source_000003/result.json').read_text())
    with h5py.File(result['hdf5']) as raw:
        actions = np.asarray(raw['joint_action/vector'], dtype=np.float32)
    ordered = actions[:, [0,1,2,3,4,5,7,8,9,10,11,12,6,13]]
    fields = {}
    for keys in (norm.action_keys, norm.state_keys):
        for key, start, stop in zip(keys, (0,6,12,13), (6,12,13,14)):
            fields[key] = ordered[:, start:stop].copy()
    transformed = norm.transform(fields)
    normalized = np.concatenate([np.asarray(transformed[k]) for k in norm.action_keys], axis=-1)
    assert normalized.shape == (535, 14) and np.isfinite(normalized).all()
    restored = norm.unapply_actions(normalized)
    np.testing.assert_allclose(restored, ordered, atol=1e-6, rtol=0)
    assert (np.abs(normalized[:, 12:]) <= 1+1e-6).all()
    slope, threshold = 1-math.tanh(3)**2, math.tanh(3)
    absolute = np.abs(normalized[:, :12])
    required_joint_logits = np.where(absolute > threshold,
        3+(absolute-threshold)/slope, np.arctanh(np.minimum(absolute, threshold)))
    report = dict(state='complete', expected_microbatches=count, expected_samples=len(rows),
        recovery_samples=len(recovery), recovery_sample_schedule=recovery,
        source_range_frames=len(actions), normalized_min=normalized.min(0).tolist(),
        normalized_max=normalized.max(0).tolist(),
        joint_fraction_outside_unit_range=float((absolute > 1).mean()),
        max_joint_unit_range_excess=float(np.maximum(absolute-1, 0).max()),
        joint_fraction_beyond_unit_range_by_0_001=float((absolute > 1.001).mean()),
        max_required_joint_preactivation=float(required_joint_logits.max()),
        output_activation=cfg.framework.action_model.embodiment_heads.aloha.output_activation,
        action_normalization_roundtrip_max_error=float(np.abs(restored-ordered).max()),
        note='Sampling schedule assumes one process, epoch 0, batch 2, accumulation 2 as configured. Joint tanh_linear_tail is unbounded; only grippers retain tanh. This is a preflight, not proof the optimizer ran.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
