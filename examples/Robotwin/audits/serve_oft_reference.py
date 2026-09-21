"""Strictly audit the released OFT checkpoint, then serve it locally."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from examples.Robotwin.audits.oft_reference_adapter import CONTRACT, INSTRUCTION, UNNORM_KEY
from examples.Robotwin.audits.run_grasp_lift_development import digest, save


def float_head_input(module, args):
    # The VLM returns BF16. QwenOFT's float32 autocast block disables autocast,
    # which does not cast these activations to the FP32 regression head dtype.
    return (args[0].to(module.fc1.weight.dtype), *args[1:])


def audit_normalization(wrapper):
    stats = wrapper._framework.norm_stats[UNNORM_KEY]['action']
    random = np.random.default_rng(713).uniform(-1.5, 1.5, (1024, 14)).astype(np.float32)
    # Include the binary boundary on both sides (not continuous gripper output).
    random[:4, 12:] = np.array([0., .49, .49001, 1.], dtype=np.float32)[:, None]
    lo, hi = np.asarray(stats['min'], np.float32), np.asarray(stats['max'], np.float32)
    expected = (random+1)/2*(hi-lo)+lo
    expected[:, 12:] = (random[:, 12:] > np.float32(.49)).astype(np.float32)
    actual = wrapper._get_processor(UNNORM_KEY).unapply_actions(random)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    return dict(samples=1024, maximum_difference=float(np.abs(actual-expected).max()),
                joints='min_max', grippers='binary > 0.49', output_order='L6,R6,Lgrip,Rgrip')


class CheckedPolicy:
    def __init__(self, wrapper, log):
        self.wrapper, self.log = wrapper, log
        self.episode, self.query = -1, 0

    def predict_action(self, examples, unnorm_key=None, **kwargs):
        assert unnorm_key == UNNORM_KEY and len(examples) == 1
        example = examples[0]
        assert 'state' not in example and example['lang'] == INSTRUCTION
        assert len(example['image']) == 3
        arrays = [np.asarray(im) for im in example['image']]
        assert all(im.shape == (224, 224, 3) and im.dtype == np.uint8 for im in arrays)
        if example.get('episode_start') is True:
            self.episode += 1
            self.query = 0
        assert self.episode >= 0
        start = time.monotonic()
        # Whitelist prevents native/history/state fields from reaching the VLM.
        payload = dict(image=arrays, lang=INSTRUCTION)
        result = self.wrapper.predict_action([payload], unnorm_key=UNNORM_KEY)
        action = result['actions']
        assert action.shape == (1, 50, 14) and np.isfinite(action).all()
        assert np.isin(action[..., 12:], [0, 1]).all()
        row = dict(episode_index=self.episode, query_index=self.query,
                   episode_start=bool(example.get('episode_start')), state_present=False,
                   instruction=INSTRUCTION, inference_seconds=time.monotonic()-start,
                   images_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in arrays],
                   actions_sha256=hashlib.sha256(action.tobytes()).hexdigest())
        with self.log.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        self.query += 1
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--smoke-video', type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.output/'deployment_audit.json'
    assert not audit_path.exists() and not (args.output/'policy_queries.jsonl').exists()
    report = dict(state='loading', checkpoint=str(args.checkpoint),
                  precision='BF16 VLM / FP32 regression head', contract=CONTRACT)
    save(audit_path, report)
    try:
        wrapper = PolicyServerWrapper(str(args.checkpoint), device='cpu', use_bf16=False,
                                      unnorm_key=UNNORM_KEY)
        model = wrapper._framework
        assert type(model).__name__ == 'Qwenvl_OFT'
        assert type(model.action_model).__name__ == 'L1RegressionActionHead'
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
        loaded = model.state_dict()
        assert set(saved) == set(loaded), (set(saved)-set(loaded), set(loaded)-set(saved))
        for name, value in saved.items():
            assert value.shape == loaded[name].shape and torch.isfinite(value).all(), name
            torch.testing.assert_close(loaded[name], value.to(loaded[name]), atol=0, rtol=0)
        report.update(checkpoint_sha256=digest(args.checkpoint), loaded_tensors=len(loaded),
                      parameters=sum(p.numel() for p in model.parameters()),
                      normalization=audit_normalization(wrapper), state='weights_verified')
        del saved, loaded
        assert wrapper.metadata['action_chunk_size'] == 50
        assert wrapper.metadata['available_unnorm_keys'] == [UNNORM_KEY]
        model.action_model.float()
        model.action_model.model.register_forward_pre_hook(float_head_input)
        model.to('cuda').eval()
        save(audit_path, report)

        # A real, previously recorded development observation for engineering
        # checks only. Formal rollouts use fresh lossless simulator observations.
        import av
        import cv2
        from PIL import Image
        with av.open(str(args.smoke_video)) as container:
            frame = next(container.decode(video=0)).to_ndarray(format='rgb24')
        assert frame.shape[1] % 3 == 0
        images = [cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
                  for im in np.split(frame, 3, axis=1)]
        payload = dict(image=images, lang=INSTRUCTION)
        suffix = f' Please predict the next 50 robot actions: <action>{model.action_token*50}<action>.'
        inputs = model.qwen_vl_interface.build_qwenvl_inputs(
            images=[[Image.fromarray(im) for im in images]], instructions=[INSTRUCTION+suffix])
        count = int((inputs['input_ids'] == model.action_token_id).sum())
        assert count == 50
        assert '[STATE]' not in model.qwen_vl_interface.processor.tokenizer.decode(inputs['input_ids'][0])
        report.update(action_token_count=count, image_grid_thw=inputs['image_grid_thw'].cpu().tolist())
        assert len(report['image_grid_thw']) == 3
        del inputs
        normalized = model.predict_action([payload])['normalized_actions']
        expected = np.stack([wrapper._get_processor(UNNORM_KEY).unapply_actions(normalized[0])])
        actual = wrapper.predict_action([payload], unnorm_key=UNNORM_KEY)['actions']
        assert actual.shape == (1, 50, 14) and np.isfinite(actual).all()
        np.testing.assert_array_equal(actual, expected)
        np.savez_compressed(args.output/'deployment_smoke_actions.npz', normalized=normalized, raw=actual)
        report.update(state='strict_weights_normalization_real_payload_verified',
                      smoke_video=str(args.smoke_video), smoke_video_sha256=digest(args.smoke_video),
                      smoke_images_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in images],
                      direct_vs_wrapper_max_difference=float(np.abs(actual-expected).max()),
                      dtype_compatibility='Explicit cast of VLM output into FP32 MLP; no changed weights or action semantics',
                      metadata=wrapper.metadata)
        save(audit_path, report)
        policy = CheckedPolicy(wrapper, args.output/'policy_queries.jsonl')
        metadata = dict(wrapper.metadata, oft_reference_contract=CONTRACT,
                        deployment_audit_state=report['state'], precision=report['precision'])
        WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port,
                              idle_timeout=-1, metadata=metadata).serve_forever()
    except BaseException as error:
        if report['state'] != 'strict_weights_normalization_real_payload_verified':
            report.update(state='failed', error=repr(error))
            save(audit_path, report)
        raise


if __name__ == '__main__':
    main()
