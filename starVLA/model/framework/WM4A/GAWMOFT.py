"""GAWM visual backbone with a current-image OFT-style direct action pathway.

Retains the inherited checkpoint namespaces and shared data/action contracts.
The old world predictor/pooler remain frozen for checkpoint compatibility but
are neither evaluated nor used to condition this policy. Only current RGB and
canonical task language reach the action pathway; training uses masked L1.
"""
import numpy as np
import torch

from starVLA.model.framework.WM4A.GAWM import GAWM
from starVLA.model.modules.action_model.OFTSpatialActionHead import OFTSpatialActionHead
from starVLA.model.modules.action_model.action_loss import masked_action_l1_loss, action_l1_diagnostics
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register('GAWMOFT')
class GAWMOFT(GAWM):
    def __init__(self, config):
        super().__init__(config)
        settings = self.config.framework.oft_action
        spec = self.embodiment_head_specs['aloha']
        if (int(spec['action_dim']) != 14 or int(spec['action_horizon']) != 16
                or spec['action_spec_id'] != 'aloha_dual_joint_contgrip_next_recorded_14'):
            raise ValueError('Initial GAWMOFT experiment requires the existing Aloha next-recorded 16x14 contract')
        self.action_models['aloha'] = OFTSpatialActionHead(
            patch_dim=self.backbone.encoder.config.hidden_size, task_dim=self.task_emb_dim,
            hidden_dim=int(settings.get('hidden_dim', 384)), num_views=self.num_views,
            grid_size=int(settings.get('grid_size', 14)), horizon=16, action_dim=14,
            depth=int(settings.get('depth', 4)), heads=int(settings.get('heads', 6)))
        self.world_model.requires_grad_(False)
        self.visual_token_pooler.requires_grad_(False)
        for tag, module in self.action_models.items():
            if tag != 'aloha':
                module.requires_grad_(False)
        self.expects_normalized_state = False
        self.use_state_cond = False

    def _actions_from_current_rgb(self, examples, *, action_valid=None):
        tag = self._resolve_batch_embodiment(examples)
        if tag != 'aloha':
            raise ValueError('GAWMOFT is currently Aloha-only')
        frames, masks = [], []
        size = self.config.datasets.vla_data.get('obs_image_size', [224, 224])
        for example in examples:
            current, inferred = self._pad_inference_views(example['image'])
            if size:
                current = resize_images(current, target_size=size)
            frames.append([current])
            masks.append(example.get('view_valid_mask', inferred))
        device = next(self.parameters()).device
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            patches = self.backbone.encode_patch_frames(frames)[:, 0]
        # Match GAWM's FP32 action-compute autocast. DeepSpeed stores model
        # parameters in BF16; disabling autocast does not upcast those weights.
        with torch.autocast(device_type=device.type, dtype=torch.float32,
                            enabled=device.type == 'cuda'):
            task = self._condition_task_on_embodiment(
                self._embed_task([x['lang'] for x in examples], patches.device), tag)
            valid = torch.as_tensor(masks, dtype=torch.bool, device=patches.device)
            return self.action_models[tag](patches.float(), task.float(), valid, action_valid)

    def forward(self, examples=None, **kwargs):
        if not examples:
            raise ValueError('Nonempty examples required')
        device = next(self.parameters()).device
        mask = self._action_valid_mask_tensor(examples, device, 16)
        target = torch.as_tensor(np.asarray([x['action'] for x in examples]), device=device, dtype=torch.float32)
        if target.shape != (len(examples), 16, 14):
            raise ValueError('GAWMOFT target must have shape [B,16,14]')
        if mask is not None:
            target = torch.where(mask[..., None], target, 0.)
        if not torch.isfinite(target).all():
            raise ValueError('Nonfinite valid action target')
        prediction = self._actions_from_current_rgb(examples, action_valid=mask)
        loss = masked_action_l1_loss(prediction, target, mask)
        metrics = action_l1_diagnostics(prediction, target, mask, gripper_indices=(12, 13))
        output = dict(action_loss=loss, l1_action_loss=loss.detach(), full_l1_action_loss=loss.detach(),
                      oft_l1_action_loss=loss.detach(),
                      oft_visual_tokens=loss.new_tensor(self.num_views*self.action_models['aloha'].grid_size**2))
        output.update({name: value.detach() for name, value in metrics.items()})
        output.update({name+'/aloha': value for name, value in output.copy().items() if name != 'action_loss'})
        return output

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        actions = self._actions_from_current_rgb(examples)
        return dict(normalized_actions=actions.float().cpu().numpy())
