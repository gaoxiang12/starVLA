"""Exercise checkpoint loading and the physical-action server contract on real data."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from omegaconf import OmegaConf
import torch

from examples.LiLaWAM.gawm_official import GAWMOfficialPolicy
from examples.LiLaWAM.official_robotwin_data import write_json
from examples.LiLaWAM.train_official_robotwin import load_dataset_module
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.WM4A.LiLaWAM import upstream_models, preprocess_image


def check(config, output):
    cfg = OmegaConf.load(config)
    settings = cfg.framework.official_robotwin
    policy = GAWMOfficialPolicy(cfg)
    source = Path(settings.source_root)
    module = load_dataset_module(source)
    file = Path(policy.cfg.dataset.dataset_dir)/'blocks_ranking_rgb/demo_clean/data/episode0.hdf5'
    with h5py.File(file) as f:
        image = module._decode(f['observation/head_camera/rgb'][0])
        state = np.concatenate((f['endpose/left_endpose'][0], [f['endpose/left_gripper'][0]],
            f['endpose/right_endpose'][0], [f['endpose/right_gripper'][0]])).astype(np.float32)
    rng = torch.cuda.get_rng_state().cpu().numpy()
    request = dict(examples=[dict(task_name='blocks_ranking_rgb', image=[image], state=state)], cuda_rng_state=rng)
    server = WebsocketPolicyServer(policy, metadata=policy.metadata)
    first = server._route_message(request)
    second = server._route_message(request)
    assert first['ok'] and second['ok'], (first, second)
    actions = first['data']['actions']
    assert actions.shape == (1, 32, 14) and np.isfinite(actions).all()
    assert np.array_equal(actions, second['data']['actions'])
    assert np.array_equal(first['data']['cuda_rng_state'], rng), 'Deterministic ACT should not consume CUDA random draws'
    # Ensure the same frozen asset AND last-layer features as the upstream DINO loader.
    upstream = upstream_models(source)
    vision, _, _, _ = upstream.ModelFactory.create_vision_encoder(settings.vision_encoder_path, torch.bfloat16, 'cuda')
    backbone = policy.wrapper.policy.backbone
    a, b = vision.state_dict(), backbone.encoder.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a), 'DINO weights differ'
    pixels = torch.from_numpy(preprocess_image(image, [320, 240])).to('cuda', torch.bfloat16)[None]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        ref = vision(pixel_values=pixels).last_hidden_state[:, backbone.num_prefix_tokens:]
        actual = backbone.encode_patch_frames([[[pixels[0]]]])[:, 0, 0]
    assert torch.equal(ref, actual), float((ref-actual).abs().max())
    result = dict(status='passed', action_shape=list(actions.shape), finite_actions=True,
        deterministic_reload_policy=True, simulator_rng_preserved=True,
        dino_weights_bitwise_equal=True, dino_last_features_bitwise_equal=True,
        metadata=policy.metadata, checkpoint=str(settings.checkpoint_path),
        note='Inference contract check on recorded observation; not a closed-loop success measurement.')
    write_json(output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    check(a.config, a.output)
