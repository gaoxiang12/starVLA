"""GAWM with a small action expert reading current dense/local spatial tokens."""
import torch
from torch import nn

from starVLA.model.framework.WM4A.GAWM import GAWM
from starVLA.model.modules.action_model.CompactFlowActionHead import CompactFlowActionHead
from starVLA.model.modules.action_model.action_loss import masked_action_l1_loss
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register('GAWMCompactExpert')
class GAWMCompactExpert(GAWM):
    def __init__(self, cfg):
        super().__init__(cfg)
        settings = self.config.framework.compact_expert
        self.expert_mode = str(settings.get('mode', 'flow'))
        if self.expert_mode not in ('flow', 'regression'):
            raise ValueError('Compact expert mode must be flow or regression')
        if self.spatial_focus is None or not self.focus_cfg.get('use_local', False):
            raise ValueError('Compact expert requires current dense and local spatial memory')
        if self.focus_robot_tag != 'aloha' or self.contact_objective is not None:
            raise ValueError('Initial compact expert experiment requires Aloha without contact auxiliary loss')
        spec = self.embodiment_head_specs['aloha']
        if spec['action_dim'] != 14 or spec['state_dim'] != 14 or spec['action_horizon'] != 16:
            raise ValueError('Compact expert experiment requires the 14D Aloha / 16-step contract')
        dimension = int(self.config.framework.world_model.visual_token_dim)
        self.expert_task_projection = nn.Linear(self.task_emb_dim, dimension)
        self.action_models['aloha'] = CompactFlowActionHead(
            context_dim=dimension, hidden_dim=int(settings.get('hidden_dim', 384)),
            action_dim=14, state_dim=14, horizon=16,
            depth=int(settings.get('depth', 6)), heads=int(settings.get('heads', 6)),
            planning_tokens=int(settings.get('planning_tokens', 8)),
            inference_steps=int(settings.get('inference_steps', 4)))

    def _focus_memory(self, patches, state, task, views, examples, robot_tag, step=0, supervision=False):
        if robot_tag != 'aloha':
            raise ValueError('Compact expert is currently Aloha-only')
        memories, extra_valid, loss, metrics, xy = super()._focus_memory(
            patches, state, task, views, examples, robot_tag, step, supervision)
        memories['expert_task'] = (self.expert_task_projection(task)[:, None],
                                   torch.ones(task.shape[0], 1, dtype=torch.bool, device=task.device))
        return memories, extra_valid, loss, metrics, xy

    def _predict_action_chunk(self, action_model, visual_tokens, state, memories,
                              actions=None, action_valid_mask=None):
        # Current-image patch features only. Predicted future grids still have
        # their existing auxiliary world-model loss, but do not drive this head.
        selected = [memories[key] for key in ('dense', 'local', 'expert_task')]
        context = torch.cat([tokens for tokens, _ in selected], dim=1)
        context_valid = torch.cat([valid for _, valid in selected], dim=1)
        if actions is None:
            if self.expert_mode == 'flow':
                prediction = action_model.predict_action(context, state, context_valid=context_valid)
            else:
                prediction = self._regress(action_model, context, state, context_valid)
            return prediction, None, {}
        if self.expert_mode == 'flow':
            output = action_model.loss(context, state, actions,
                context_valid=context_valid, action_valid=action_valid_mask)
            prediction, loss = output['estimated_actions'], output['loss']
            name = 'compact_flow_velocity_loss'
        else:
            prediction = self._regress(action_model, context, state, context_valid, action_valid_mask)
            loss = masked_action_l1_loss(prediction, actions, action_valid_mask)
            name = 'compact_regression_loss'
        return prediction, loss, {name: loss.detach(),
                                  'compact_context_tokens': loss.new_tensor(context.shape[1])}

    @staticmethod
    def _regress(head, context, state, context_valid, action_valid=None):
        # Same expert architecture/memory, single deterministic decoding pass.
        # Constant zero action inputs and time=1 never depend on target actions.
        zeros = context.new_zeros(context.shape[0], head.horizon, head.action_dim)
        ones = context.new_ones(context.shape[0])
        return head.velocity(context, state, zeros, ones,
                             context_valid=context_valid, action_valid=action_valid)
