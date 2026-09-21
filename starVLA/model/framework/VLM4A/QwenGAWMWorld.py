"""Qwen action-token memory -> original GAWM residual predictor -> three-frame ACT.

Token slots have action-placeholder identity, not camera/patch coordinates.
Real future observations supply detached diagnostics and training-only residual
scale statistics. The controlled campaign optimizes masked action L1 only.
"""
import numpy as np
import torch
from torch import nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.framework.VLM4A.QwenGAWM import QwenGAWM, gather_action_features
from starVLA.model.modules.world_model.visual_token_delta_world_model import VisualTokenLatentWorldModel
from starVLA.model.modules.action_model.action_loss import masked_action_l1_loss, action_l1_diagnostics
from starVLA.model.tools import FRAMEWORK_REGISTRY


class QwenWorldActionHead(nn.Module):
    def __init__(self, act, wm):
        super().__init__()
        self.input_projection = act.visual_projection
        act.visual_projection = nn.Identity()
        # Keep the parent's random current-frame embedding; add two future slots.
        original = act.frame_embedding.weight.detach().clone()
        act.num_frames = 3
        act.frame_embedding = nn.Embedding(3, act.hidden_dim)
        with torch.no_grad(): act.frame_embedding.weight[0].copy_(original[0])
        self.act = act
        self.world_model = VisualTokenLatentWorldModel(
            latent_dim=act.hidden_dim, goal_dim=None, n_future=2, num_tokens=16,
            dim=int(wm.residual_predictor_dim), depth=int(wm.residual_predictor_depth),
            num_heads=int(wm.residual_predictor_heads), ffn_dim=int(wm.residual_predictor_ffn),
            stats_momentum=float(wm.latent_stats_momentum), detach_input=True)

    def current_latent(self, features):
        return self.input_projection(features.to(self.input_projection.weight.dtype))[:, None]

    def decode(self, current, future):
        # GAWM intentionally keeps delta_scale in FP32, which promotes its
        # predicted memory even when DeepSpeed converts the decoder to BF16.
        memory = torch.cat([current, future], dim=1)
        return self.act(memory.to(self.act.memory_norm.weight.dtype)).float()

    def forward(self, features):
        current = self.current_latent(features[:, 0])
        return self.decode(current, self.world_model.regress_future(current))


@FRAMEWORK_REGISTRY.register('QwenGAWMWorld')
class QwenGAWMWorld(QwenGAWM):
    def __init__(self, config):
        super().__init__(config)
        if self.head_type != 'ACT': raise ValueError('World-model comparison requires ACT')
        wm = self.config.framework.world_model
        if int(wm.n_future) != 2 or list(wm.future_recorded_offsets) != [6, 12]:
            raise ValueError('Expected two recorded-frame horizons +6 and +12')
        if float(wm.loss_latent_weight) != 0 or float(wm.latent_cosine_weight) != 0:
            raise ValueError('This campaign is the L1-only architectural comparison')
        self.action_model = QwenWorldActionHead(self.action_model, wm)

    def _encode_features(self, examples):
        if not examples:
            raise ValueError('Nonempty examples required')
        images = []
        for example in examples:
            if example.get('robot_tag', 'aloha') != 'aloha':
                raise ValueError('QwenGAWM comparison is Aloha-only')
            if example.get('action_spec_id', self.embodiment_head_specs['aloha']['action_spec_id']) != self.embodiment_head_specs['aloha']['action_spec_id']:
                raise ValueError('Wrong action semantics')
            if len(example['image']) != 3 or not np.asarray(example.get('view_valid_mask', [True]*3)).all():
                raise ValueError('Three valid current RGB views required')
            lang = str(example['lang']).strip()
            if lang not in ('blocks ranking rgb', 'blocks_ranking_rgb', self.config.framework.task_instruction):
                raise ValueError(f'Unexpected instruction for single-task experiment: {lang!r}')
            current = to_pil_preserve(example['image'])
            images.append(resize_images(current, target_size=self.config.datasets.vla_data.obs_image_size))
        suffix = f' Please predict the next 16 robot actions: <action>{self.action_token*16}<action>.'
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images,
            instructions=[self.config.framework.task_instruction+suffix]*len(examples))
        # Call the multimodal base to avoid allocating unused vocabulary logits.
        device = next(self.qwen_vl_interface.parameters()).device
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            output = self.qwen_vl_interface.model.model(**inputs, use_cache=False, return_dict=True)
        features = gather_action_features(output.last_hidden_state, inputs['input_ids'],
                                          self.action_token_id, self.action_horizon)
        return features

    def forward(self, examples=None, **kwargs):
        # Future targets never enter the current Qwen sequence or ACT memory.
        # Encode targets first in eval/no-grad mode, then restore Qwen mode before
        # creating the current-frame graph (including checkpointed recomputation).
        future_latents=[]
        was_training=self.qwen_vl_interface.training
        try:
            self.qwen_vl_interface.eval()
            with torch.no_grad():
                for index in range(2):
                    targets=[]
                    for x in examples:
                        if x.get('future_images') is None or len(x['future_images']) != 2:
                            raise ValueError('Training requires two future RGB observations')
                        targets.append(dict(x, image=x['future_images'][index]))
                    features=self._encode_features(targets)
                    future_latents.append(self.action_model.current_latent(features))
        finally:
            self.qwen_vl_interface.train(was_training)
        features=self._encode_features(examples)
        device=features.device
        with torch.autocast(device.type, enabled=False):
            current=self.action_model.current_latent(features)
            latent=torch.cat([current, *future_latents], dim=1)
            valid=torch.as_tensor(np.asarray([x['future_frame_valid_mask'] for x in examples]),
                                  device=device,dtype=torch.bool)
            if valid.shape != (len(examples),3) or not valid[:,0].all():
                raise ValueError('Expected valid-current plus two future masks')
            token_mask=valid[:,:,None].expand(-1,-1,16)
            wm=self.action_model.world_model(latent,ctx_len=1,loss_mask=token_mask,update_stats=True)
            prediction=self.action_model.decode(current,wm['pred_future_latent'])
        target=torch.as_tensor(np.asarray([x['action'] for x in examples]),device=device,dtype=prediction.dtype)
        mask=torch.as_tensor(np.asarray([x['action_valid_mask'] for x in examples]),device=device,dtype=torch.bool)
        if target.shape != prediction.shape or mask.shape != prediction.shape[:2]:
            raise ValueError('Expected action [B,16,14] and mask [B,16]')
        target=torch.where(mask[...,None],target,0.)
        if not torch.isfinite(target).all(): raise ValueError('Nonfinite valid target')
        loss=masked_action_l1_loss(prediction,target,mask)
        metrics=action_l1_diagnostics(prediction,target,mask,gripper_indices=(12,13))
        metrics.update({k:v for k,v in wm.items() if k!='pred_future_latent'})
        return dict(action_loss=loss,l1_action_loss=loss.detach(),
                    **{k:v.detach() for k,v in metrics.items()})
