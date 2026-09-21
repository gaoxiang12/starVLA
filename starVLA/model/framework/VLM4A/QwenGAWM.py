"""Controlled Qwen action-token features with either the GAWM ACT or OFT MLP.

Only current RGB and language enter Qwen. ACT uses one memory frame whose
positions identify action tokens, not spatial cells or predicted future frames.
Both heads are freshly initialized and share the same masked L1 contract.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.ACT_ActionHeader import TurboStyleACTActionHead
from starVLA.model.modules.action_model.MLP_ActionHeader import L1RegressionActionHead
from starVLA.model.modules.action_model.action_loss import masked_action_l1_loss, action_l1_diagnostics
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def gather_action_features(hidden, input_ids, token_id, horizon):
    mask = input_ids.eq(token_id)
    if hidden.shape[:2] != mask.shape or not torch.all(mask.sum(1) == horizon):
        raise ValueError(f'Every input must contain exactly {horizon} action tokens')
    return hidden[mask].reshape(hidden.shape[0], horizon, hidden.shape[-1])


@dataclass
class QwenGAWMConfig:
    name: str = 'QwenGAWM'
    qwenvl: dict = field(default_factory=lambda: dict(
        base_vlm='/data/gaoxiang/ckpts/Qwen3-VL-4B-Instruct', attn_implementation='sdpa'))
    action_model: dict = field(default_factory=lambda: dict(
        action_horizon=16, action_dim=14, action_hidden_dim=384, action_model_type='ACT',
        act_num_heads=8, act_num_layers=3, act_dim_feedforward=2048,
        act_mlp_hidden_dim=512, act_dropout=.1, output_activation='tanh_linear_tail'))
    qwen_training: dict = field(default_factory=lambda: dict(train_last_n_layers=0))
    # An explicit single-task instruction avoids changing language across data sources.
    task_instruction: str = ('Start with the red block, followed by the green block and the blue block, '
                             'placing them in order left to right.')


@FRAMEWORK_REGISTRY.register('QwenGAWM')
class QwenGAWM(baseframework):
    def __init__(self, config):
        super().__init__()
        self.config = merge_framework_config(QwenGAWMConfig, config)
        settings = self.config.framework
        head = settings.action_model
        self.action_horizon = int(head.action_horizon)
        if self.action_horizon != 16 or int(head.action_dim) != 14:
            raise ValueError('This experiment requires the Aloha next-recorded 16x14 contract')
        self.embodiment_head_specs = {'aloha': dict(action_dim=14, action_horizon=16,
            state_dim=0, gripper_indices=[12, 13],
            action_spec_id='aloha_dual_joint_contgrip_next_recorded_14')}
        self.expects_normalized_state = False
        self.qwen_vl_interface = get_vlm_model(self.config)
        width = int(self.qwen_vl_interface.model.config.hidden_size)
        self.head_type = str(head.action_model_type)
        if self.head_type == 'ACT':
            self.action_model = TurboStyleACTActionHead(token_dim=width,
                hidden_dim=int(head.action_hidden_dim), action_dim=14, horizon=16,
                num_frames=1, num_visual_tokens=16, num_heads=int(head.act_num_heads),
                num_layers=int(head.act_num_layers), dim_feedforward=int(head.act_dim_feedforward),
                mlp_hidden_dim=int(head.act_mlp_hidden_dim), dropout=float(head.act_dropout),
                state_dim=0, output_activation=head.output_activation, gripper_indices=(12, 13))
        elif self.head_type == 'MLP':
            self.action_model = L1RegressionActionHead(input_dim=width, hidden_dim=width*2,
                action_dim=14, NUM_ACTIONS_CHUNK=16)
        else:
            raise ValueError(f'Unsupported comparison head: {self.head_type}')
        self.action_token = '🔍'
        ids = self.qwen_vl_interface.processor.tokenizer(self.action_token, add_special_tokens=False)['input_ids']
        if len(ids) != 1:
            raise ValueError('Action placeholder must tokenize to exactly one token')
        self.action_token_id = ids[0]
        self.qwen_vl_interface.requires_grad_(False)
        count = int(settings.qwen_training.train_last_n_layers)
        language = self.qwen_vl_interface.model.model.language_model
        if not 0 <= count <= len(language.layers):
            raise ValueError('Invalid train_last_n_layers')
        if count:
            for layer in language.layers[-count:]:
                layer.requires_grad_(True)
            language.norm.requires_grad_(True)
            self.qwen_vl_interface.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})

    def remap_checkpoint_state_dict(self, state_dict):
        """Reject incomplete Qwen transfers and incomplete experiment checkpoints."""
        expected = self.state_dict()
        # Published QwenOFT transfer is explicitly requested by the trainer.
        transfer = (bool(self.config.trainer.get('pretrained_checkpoint'))
                    and self.config.trainer.get('reload_modules') == 'qwen_vl_interface')
        prefix = 'qwen_vl_interface.'
        keys = {k for k in expected if k.startswith(prefix)} if transfer else set(expected)
        source_keys = {k for k in state_dict if k.startswith(prefix)} if transfer else set(state_dict)
        if keys != source_keys or any(expected[k].shape != state_dict[k].shape for k in keys):
            raise ValueError('Incomplete or incompatible Qwen/experiment checkpoint')
        return state_dict

    def _predict_tensor(self, examples):
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
        # Explicit dtype conversion handles FP32 deployment and BF16 DeepSpeed heads.
        with torch.autocast(device.type, enabled=False):
            features = features.to(next(self.action_model.parameters()).dtype)
            result = self.action_model(features[:, None] if self.head_type == 'ACT' else features)
        return result.float()

    def forward(self, examples=None, **kwargs):
        prediction = self._predict_tensor(examples)
        target = torch.as_tensor(np.asarray([x['action'] for x in examples]),
                                 device=prediction.device, dtype=prediction.dtype)
        mask = torch.as_tensor(np.asarray([x.get('action_valid_mask', np.ones(16, bool)) for x in examples]),
                               device=prediction.device, dtype=torch.bool)
        if target.shape != prediction.shape or mask.shape != prediction.shape[:2]:
            raise ValueError('Expected target [B,16,14] and action mask [B,16]')
        target = torch.where(mask[..., None], target, 0.)
        if not torch.isfinite(target).all():
            raise ValueError('Nonfinite valid target')
        loss = masked_action_l1_loss(prediction, target, mask)
        metrics = action_l1_diagnostics(prediction, target, mask, gripper_indices=(12, 13))
        return dict(action_loss=loss, l1_action_loss=loss.detach(),
                    **{k: v.detach() for k, v in metrics.items()})

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        return dict(normalized_actions=self._predict_tensor(examples).cpu().numpy())
