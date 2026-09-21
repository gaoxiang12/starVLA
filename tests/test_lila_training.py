import copy
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from PIL import Image

from starVLA.model.framework.WM4A.LiLaWAMTrain import LiLaWAMTrain


class TinyVision(nn.Module):
    """Small deterministic frozen encoder for contract/gradient tests."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, patch_size=16)
        self.patch = nn.Conv2d(3, 16, 16, 16)

    def forward(self, pixel_values, output_hidden_states=True):
        patches = self.patch(pixel_values).flatten(2).transpose(1, 2)
        x = torch.cat((patches.mean(1, keepdim=True), patches), dim=1)
        return SimpleNamespace(hidden_states=(x, x * .9, x * 1.1), last_hidden_state=x)


def config():
    heads = {tag: dict(action_dim=a, state_dim=s, action_horizon=4,
                       num_observed_views=v, action_spec_id=f'{tag}_action', state_spec_id=f'{tag}_state')
             for tag, a, s, v in [('franka', 7, 8, 2), ('aloha', 14, 14, 3)]}
    return dict(framework=dict(lila=dict(image_size=[32, 32], num_views=3, feat_layers=[-2, -1],
        future_target_layer=-1, hidden_dim=32, depth=2, num_heads=4, adapter_depth=1,
        queries_per_view=2, future_depth=1, future_heads=4, num_inference_steps=2,
        task_vectors=dict(format_version=1, vectors={f'{t}:pick object': [0.1]*16 for t in heads})),
        action_model=dict(embodiment_heads=heads)), datasets=dict(vla_data=dict(task_language_mode='canonical_metadata')))


def sample(tag='franka'):
    a, s, v = (7, 8, 2) if tag == 'franka' else (14, 14, 3)
    return dict(robot_tag=tag, lang='pick object', state=np.zeros((1, s), np.float32),
        action=np.zeros((4, a), np.float32), action_valid_mask=[True, True, False, False],
        image=[Image.new('RGB', (32,32), (20*i,40,80)) for i in range(3)],
        future_images=[[Image.new('RGB',(32,32),(100,30,50)) for _ in range(3)]],
        view_valid_mask=[i<v for i in range(3)], future_frame_valid_mask=[True,True])


class LiLaTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.model = LiLaWAMTrain(config(), vision_encoder=TinyVision())

    def test_gradients_and_frozen_encoder(self):
        for tag in ('franka','aloha'):
            self.model.zero_grad(set_to_none=True)
            self.model.train()
            loss = self.model([sample(tag)])['action_loss']
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertFalse(self.model.encoder.training)
            self.assertTrue(all(p.grad is None for p in self.model.encoder.parameters()))
            for module in (self.model.core.heads[tag], self.model.core.adapter,
                           self.model.core.future, self.model.core.blocks):
                grads=[p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads),0)

    def test_padding_and_missing_camera_do_not_affect_loss(self):
        a=sample(); b=copy.deepcopy(a)
        b['action'][2:]=1e9
        b['image'][2]=Image.new('RGB',(32,32),'white')
        b['future_images'][0][2]=Image.new('RGB',(32,32),'white')
        torch.manual_seed(55); x=self.model([a])['action_loss']
        torch.manual_seed(55); y=self.model([b])['action_loss']
        torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_future_mask_and_no_inference_leakage(self):
        a=sample(); a['future_frame_valid_mask']=[True,False]
        b=copy.deepcopy(a); b['future_images'][0]=[Image.new('RGB',(32,32),'white')]*3
        torch.manual_seed(3); x=self.model([a])
        torch.manual_seed(3); y=self.model([b])
        self.assertEqual(x['latent_loss'].item(),0)
        torch.testing.assert_close(x['action_loss'],y['action_loss'],rtol=0,atol=0)
        self.model.eval()
        b.update(action='ignored',future_images='ignored',future_frame_valid_mask='ignored')
        torch.manual_seed(3); x=self.model.predict_action([a])['normalized_actions']
        torch.manual_seed(3); y=self.model.predict_action([b])['normalized_actions']
        np.testing.assert_array_equal(x,y)

    def test_valid_third_camera_changes_robotwin_prediction(self):
        # adaLN-zero initially blocks cross-token interaction; simulate trained gates.
        for block in self.model.core.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].bias,.2)
        self.model.eval()
        a=sample('aloha'); b=copy.deepcopy(a)
        b['image'][2]=Image.new('RGB',(32,32),'white')
        torch.manual_seed(3); x=self.model.predict_action([a])['normalized_actions']
        torch.manual_seed(3); y=self.model.predict_action([b])['normalized_actions']
        self.assertGreater(np.max(np.abs(x-y)),1e-6)

    def test_routing_and_state_semantics(self):
        with self.assertRaises(ValueError): self.model([sample(),sample('aloha')])
        a=sample(); a['action_spec_id']='wrong'
        with self.assertRaises(ValueError): self.model([a])
        a=sample(); a['state']=np.zeros((1,16))
        with self.assertRaises(ValueError): self.model([a])

    def test_checkpoint_roundtrip(self):
        opt=torch.optim.AdamW(self.model.core.parameters(),lr=1e-3)
        self.model([sample()])['action_loss'].backward();opt.step()
        other=LiLaWAMTrain(config(),vision_encoder=TinyVision())
        with tempfile.TemporaryFile() as f:
            torch.save(self.model.state_dict(),f);f.seek(0)
            other.load_state_dict(torch.load(f,weights_only=True),strict=True)
        self.model.eval();other.eval()
        for tag,dim in [('franka',7),('aloha',14)]:
            torch.manual_seed(9); x=self.model.predict_action([sample(tag)])['normalized_actions']
            torch.manual_seed(9); y=other.predict_action([sample(tag)])['normalized_actions']
            self.assertEqual(x.shape,(1,4,dim));np.testing.assert_array_equal(x,y)

    def test_vtt_vocabulary_mismatch_fails(self):
        cfg=config()
        vectors=cfg['framework']['lila']['task_vectors']['vectors']
        vectors['franka:different task']=vectors.pop('franka:pick object')
        other=LiLaWAMTrain(cfg,vision_encoder=TinyVision())
        with self.assertRaisesRegex(ValueError,'vocabulary'):
            other.load_state_dict(self.model.state_dict())

    def test_trainer_config_snapshot_retains_model_settings(self):
        from omegaconf import OmegaConf
        from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig
        cfg=AccessTrackedConfig(OmegaConf.create(config()))
        LiLaWAMTrain(cfg,vision_encoder=TinyVision())
        with tempfile.TemporaryDirectory() as directory:
            path=directory+'/config.yaml'
            cfg.save_accessed_config(path,use_original_values=False)
            self.assertEqual(OmegaConf.to_container(OmegaConf.load(path).framework),config()['framework'])

    def test_attention_bypasses_native_bf16_inference_kernel(self):
        from unittest.mock import patch
        from starVLA.model.modules.lila.primitives import SafeMultiheadAttention
        layer=SafeMultiheadAttention(32,4,batch_first=True).eval()
        x=torch.randn(1,5,32)
        with torch.inference_mode(),patch('torch._native_multi_head_attention',side_effect=AssertionError('unsafe fast path')):
            out=layer(x,x,x,need_weights=False)[0]
        self.assertTrue(torch.isfinite(out).all())

    def test_registered_dataset_horizons_and_missing_views(self):
        from examples.LiLaWAM.train_files.data_registry.data_config import ROBOT_TYPE_CONFIG_MAP
        for key,cfg in ROBOT_TYPE_CONFIG_MAP.items():
            self.assertEqual(len(cfg.action_indices),32)
            self.assertEqual(cfg.video_indices,[0,32])
            self.assertEqual(len(cfg.video_keys),2 if key=='lila_libero' else 3)
            self.assertIsNone(cfg.control_hz)
            self.assertIsNone(cfg.future_time_offsets_s)

    def test_efficient_views_preserve_loss_gradients_and_inference(self):
        cfg=config();cfg['framework']['lila'].update(efficient_views=True,encoder_batch_size=8)
        fast=LiLaWAMTrain(cfg,vision_encoder=TinyVision())
        for block in self.model.core.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].bias,.2)
        fast.load_state_dict(self.model.state_dict(),strict=True)
        for invalid_future in (False,True):
            a=sample();b=sample()
            b['view_valid_mask']=[True,False,False]
            if invalid_future:
                a['future_frame_valid_mask']=b['future_frame_valid_mask']=[True,False]
            self.model.zero_grad(set_to_none=True);fast.zero_grad(set_to_none=True)
            torch.manual_seed(51);reference=self.model([a,b])['action_loss']
            torch.manual_seed(51);optimized=fast([a,b])['action_loss']
            torch.testing.assert_close(reference,optimized,rtol=2e-5,atol=2e-6)
            reference.backward();optimized.backward()
            for (name,p),(other_name,q) in zip(self.model.named_parameters(),fast.named_parameters()):
                self.assertEqual(name,other_name)
                self.assertEqual(p.grad is None,q.grad is None,name)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-3,atol=2e-6,msg=name)
        self.model.eval();fast.eval()
        torch.manual_seed(7);ref=self.model.predict_action([sample()])['normalized_actions']
        torch.manual_seed(7);out=fast.predict_action([sample()])['normalized_actions']
        np.testing.assert_allclose(ref,out,atol=2e-6,rtol=2e-5)


if __name__ == '__main__': unittest.main()
