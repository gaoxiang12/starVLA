"""RGB task candidate with object memory inside ACT and shared auxiliary decoding."""
from contextvars import ContextVar
import math

import numpy as np
import torch

from starVLA.model.framework.WM4A.GAWM import GAWM
from starVLA.model.modules.action_model.action_loss import masked_action_l1_loss
from starVLA.model.modules.object_memory_fusion import ObjectMemoryFusion
from starVLA.model.modules.rgb_object_readout import RGBObjectReadout
from starVLA.model.tools import FRAMEWORK_REGISTRY


_active_fusion = ContextVar('gawm_object_fusion_call', default=None)


class CoreSupervisedObjectReadout(RGBObjectReadout):
    def supervision(self, prediction, arrays, head_valid, cfg):
        if not cfg.get('object_label_red_core', False):
            return super().supervision(prediction, arrays, head_valid, cfg)
        # Candidate image rules remain exclusively in training supervision.
        from starVLA.model.modules.rgb_object_core_loss import object_core_supervision
        return object_core_supervision(prediction, arrays, head_valid, cfg)


@FRAMEWORK_REGISTRY.register('GAWMObjectFusion')
class GAWMObjectFusion(GAWM):
    """Keep the existing observation/action contract and unmodified base GAWM.

    Full actions use object tokens in every shared ACT decoder layer. An extra
    masked imitation loss uses that same decoder/projection with object+state
    memory only. Both consume learned current-image tokens, never target labels.
    Per-call context avoids persistent inference state and releases training
    graphs even when a forward raises an exception.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.spatial_focus is None or self.spatial_focus.object_readout is None:
            raise ValueError('GAWMObjectFusion requires spatial_focus.object_readout')
        head = self.action_models[self.focus_robot_tag]
        original = self.spatial_focus.object_readout
        reader = CoreSupervisedObjectReadout(original.identity.embedding_dim)
        reader.load_state_dict(original.state_dict(), strict=True)
        self.spatial_focus.object_readout = reader
        self.spatial_focus.early_fusion = ObjectMemoryFusion(reader.identity.embedding_dim, head.hidden_dim)
        self.object_action_weight = float(self.focus_cfg.get('object_action_loss_weight', .5))
        self._object_inference_ablation = 'full'
        if not math.isfinite(self.object_action_weight) or self.object_action_weight < 0:
            raise ValueError('object_action_loss_weight must be finite and nonnegative')

    def configure_object_memory_ablation(self, mode):
        self._object_inference_ablation = mode

    def _context(self):
        context = _active_fusion.get()
        if context is None or context['owner'] is not self:
            raise RuntimeError('Object fusion must run inside its framework forward/predict call')
        return context

    def _focus_memory(self, *args, **kwargs):
        memories, extra_valid, loss, metrics, xy = super()._focus_memory(*args, **kwargs)
        if memories is None or 'objects' not in memories:
            raise ValueError('Object fusion requires predicted current head tokens')
        context = self._context()
        context['objects'] = memories['objects']
        # Objects are already read inside ACT; avoid a second late readout.
        return {k:v for k,v in memories.items() if k != 'objects'}, extra_valid, loss, metrics, xy

    def _pool_visual_tokens_to_action_queries(self, visual_tokens, state, action_model):
        context = self._context()
        objects = context.pop('objects')
        if not context['training_forward'] and self._object_inference_ablation in ('no_objects', 'no_focus'):
            return action_model.decode_action_queries(visual_tokens, state=state)
        fusion = self.spatial_focus.early_fusion
        queries = fusion(action_model, visual_tokens, state, objects)
        if context['training_forward'] and self.object_action_weight > 0:
            auxiliary = fusion(action_model, visual_tokens, state, objects, objects_only=True)
            context['auxiliary_actions'] = action_model.predict_action(auxiliary)
        return queries

    def forward(self, examples=None, **kwargs):
        context = dict(owner=self, training_forward=True)
        token = _active_fusion.set(context)
        try:
            output = super().forward(examples, **kwargs)
            if self.object_action_weight > 0:
                prediction = context.pop('auxiliary_actions')
                targets = torch.as_tensor(np.asarray([ex['action'] for ex in examples]),
                                          device=prediction.device, dtype=torch.float32)
                mask = self._action_valid_mask_tensor(examples, prediction.device, prediction.shape[1])
                auxiliary_loss = masked_action_l1_loss(prediction, targets, mask)
                output['action_loss'] = output['action_loss'] + self.object_action_weight * auxiliary_loss
                output['object_action_l1'] = auxiliary_loss.detach()
            return output
        finally:
            context.clear()
            _active_fusion.reset(token)

    def predict_action(self, examples, **kwargs):
        context = dict(owner=self, training_forward=False)
        token = _active_fusion.set(context)
        try:
            return super().predict_action(examples, **kwargs)
        finally:
            context.clear()
            _active_fusion.reset(token)
