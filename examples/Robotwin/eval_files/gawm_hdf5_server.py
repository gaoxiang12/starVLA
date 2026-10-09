"""Serve a public GAWM checkpoint trained on official RoboTwin HDF5 data."""
import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import torch


def preprocess_policy_image(image, image_size, channel_order):
    """Match the checkpoint's physical color order before ImageNet normalization.

    Live SAPIEN observations are RGB. Historical official-HDF5 GAWM runs
    swapped R/B after decoding upstream RGB-as-OpenCV JPEGs, so those
    checkpoints require BGR-valued inputs. This does not mutate observations.
    """
    from starVLA.model.framework.WM4A.LiLaWAM import preprocess_image
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected HxWx3 uint8 simulator RGB')
    if channel_order not in ('rgb', 'bgr'):
        raise ValueError(f'Unknown policy image channel order: {channel_order}')
    if channel_order == 'bgr':
        image = np.ascontiguousarray(image[..., ::-1])
    return preprocess_image(image, image_size)


class GAWMHDF5Policy:
    def __init__(self, checkpoint, smooth_actions=True, image_channel_order="rgb"):
        if image_channel_order not in ("rgb", "bgr"):
            raise ValueError(image_channel_order)
        self.image_channel_order = image_channel_order
        checkpoint = Path(checkpoint).resolve()
        run = checkpoint.parent.parent
        # Model implementation is exactly the training run's frozen source.
        sys.path.insert(0, str(run / 'source_snapshot'))
        from starVLA.model.framework.WM4A.GAWM import GAWM
        from starVLA.training.recipe import prepare_parameter_precision
        self.cfg = OmegaConf.load(run / 'config.full.yaml')
        if self.cfg.datasets.vla_data.dataset_py != 'robotwin_official_hdf5':
            raise ValueError('This server requires the official HDF5 observation/action contract')
        self.image_size = tuple(self.cfg.datasets.vla_data.obs_image_size)
        self.cameras = list(self.cfg.datasets.vla_data.get('cameras', ['head_camera']))
        if len(self.cameras) != int(self.cfg.framework.world_model.num_views):
            raise ValueError('Checkpoint camera count differs from model view count')
        # Tensor images are already resized/normalized by the adapter. The
        # generic PIL resize in predict_action must not run on these tensors.
        self.cfg.datasets.vla_data.obs_image_size = None
        self.stats = json.loads((run / 'dataset_statistics.json').read_text())['aloha']
        torch.backends.mha.set_fastpath_enabled(False)
        self.model = GAWM(self.cfg)
        prepare_parameter_precision(self.model, self.cfg)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.to('cuda').eval()
        self.history_latent = None
        self.history_task = None
        self.smooth_actions = smooth_actions
        self.metadata = dict(framework='GAWMOfficialHDF5', ckpt_path=str(checkpoint),
            action_chunk_size=32, execute_horizon=16, state_dim=16, action_dim=14,
            state_representation='endpose', action_order='left_arm6,left_gripper,right_arm6,right_gripper',
            normalization='official_min_max_float32_no_clipping',
            camera=self.cameras[0], cameras=self.cameras, image_size=list(self.image_size), smooth_actions=smooth_actions,
            simulator_image_channel_order='rgb', policy_image_channel_order=image_channel_order,
            swap_rb_before_normalization=image_channel_order == 'bgr')

    def limits(self, key):
        low = np.asarray(self.stats[key]['min'], dtype=np.float32)
        high = np.asarray(self.stats[key]['max'], dtype=np.float32)
        return low, np.where(high-low < 1e-6, 1., high-low)

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        from starVLA.task_language import canonical_task_text
        from starVLA.model.framework.WM4A.LiLaWAM import smooth_chunk
        if len(examples) != 1:
            raise ValueError('One simulator observation is required')
        x = examples[0]
        state = np.asarray(x['state'], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError('Expected finite 16D endpose state')
        task = x['task_name']
        if not task or Path(task).name != task:
            raise ValueError('An explicit canonical task ID is required')
        low, span = self.limits('state')
        if len(x['image']) != len(self.cameras):
            raise ValueError(f'Expected cameras in order {self.cameras}')
        pixels = [torch.from_numpy(preprocess_policy_image(im, self.image_size, self.image_channel_order))
                  for im in x['image']]
        history_kwargs = {}
        if getattr(self.model, 'compact_history', False):
            if x.get('reset_history', False) or self.history_task != task:
                self.history_latent = None
            history_kwargs['history_latent'] = self.history_latent
        result = self.model.predict_action([dict(image=pixels, state=2*(state-low)/span-1,
            lang=canonical_task_text(task), robot_tag='aloha',
            action_spec_id='robotwin_official_absolute_dual_joint_native_14', **history_kwargs)])
        if getattr(self.model, 'compact_history', False):
            self.history_latent = result.pop('_current_visual_latent')[0].detach()
            self.history_task = task
        normalized = np.asarray(result['normalized_actions'], dtype=np.float32)
        if normalized.shape != (1, 32, 14):
            raise ValueError(f'Unexpected action shape: {normalized.shape}')
        low, span = self.limits('action')
        actions = (normalized[0]+1)/2*span+low
        if self.smooth_actions:
            actions = smooth_chunk(actions).astype(np.float32)
        if not np.isfinite(actions).all():
            raise ValueError('Nonfinite physical actions')
        return {'actions': actions[None]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--no-smoothing', action='store_true')
    p.add_argument('--image-channel-order', choices=['rgb', 'bgr'], required=True,
                   help='Physical channel order expected by the checkpoint before normalization')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)
    policy = GAWMHDF5Policy(args.checkpoint, not args.no_smoothing, args.image_channel_order)
    logging.info('Policy metadata: %s', json.dumps(policy.metadata))
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    main()
