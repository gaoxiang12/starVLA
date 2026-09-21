"""Verify trained compact weights and the actual deployment normalization path."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    wrapper = PolicyServerWrapper(str(args.checkpoint.resolve()), device='cpu', use_bf16=False, unnorm_key='aloha')
    model = wrapper._framework
    assert type(model).__name__ == 'GAWMCompactExpert'
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    loaded = model.state_dict()
    assert set(saved) == set(loaded)
    for name, value in saved.items():
        assert value.shape == loaded[name].shape and torch.isfinite(value).all()
        torch.testing.assert_close(loaded[name], value.to(loaded[name]), atol=0, rtol=0)
    assert wrapper.metadata['action_specs']['aloha'] == dict(
        action_spec_id='aloha_dual_joint_contgrip_next_recorded_14', action_dim=14, action_horizon=16, state_dim=14)
    # Saved config.yaml omits default dataset fields; the training loader needs
    # its fully resolved configuration, as used by the trainer itself.
    cfg = OmegaConf.load(args.checkpoint.parents[1] / 'config.full.yaml')
    baseline_path = Path(__file__).resolve().parents[3] / cfg.trainer.pretrained_checkpoint
    reference = PolicyNormProcessor(str(baseline_path), unnorm_key='aloha')
    processor = wrapper._get_processor('aloha')
    random = np.random.default_rng(42).uniform(-1.5, 1.5, (1024, 14)).astype(np.float32)
    np.testing.assert_array_equal(processor.unapply_actions(random), reference.unapply_actions(random))
    # Real train-scene initial image and drive state; no action or spatial labels in inference.
    mixture = get_vla_dataset(cfg.datasets.vla_data, mode='train')
    child = mixture.datasets[0]
    episode = int(child.trajectory_ids[0])
    child.transforms.eval()
    sample = child._pack_sample(child.transforms(child.get_step_data(episode, 0)))
    raw_state = np.asarray([[0.] * 12 + [1., 1.]], dtype=np.float32)
    np.testing.assert_array_equal(processor.apply_state(raw_state).astype(sample['state'].dtype), sample['state'])
    np.testing.assert_array_equal(processor.apply_state(raw_state), reference.apply_state(raw_state))
    payload = dict(image=sample['image'], native_images=sample['native_images'],
                   state=raw_state, lang=sample['lang'])
    normalized_payload = dict(payload, state=processor.apply_state(raw_state), robot_tag='aloha')
    torch.manual_seed(713)
    normalized = model.predict_action([normalized_payload])['normalized_actions']
    expected = np.stack([processor.unapply_actions(row) for row in normalized])
    torch.manual_seed(713)
    actual = wrapper.predict_action([payload])['actions']
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (1, 16, 14) and np.isfinite(actual).all()
    report = dict(state='strict_deployment_weights_and_real_payload_verified', mode=model.expert_mode,
                  checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  loaded_tensors=len(loaded), parameters=sum(p.numel() for p in model.parameters()),
                  metadata=wrapper.metadata, test_source_episode=episode,
                  normalization_random_actions_checked=1024, normalization_max_difference=0.,
                  wrapper_vs_direct_action_max_difference=float(np.abs(actual-expected).max()),
                  device='cpu', precision='float32', inference_executed=True,
                  note='Actual deployment wrapper with a real train-scene observation. No simulator rollout or success claim.')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'metadata'}), flush=True)


if __name__ == '__main__':
    main()
