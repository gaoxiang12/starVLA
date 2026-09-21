"""OFT Qwen visual-only and 18-language-layer ACT ablations.

Pure vision retains real camera/spatial token identity; it does not reinterpret
patches as action placeholder tokens. Removed modules are absent from state_dict.
"""
import re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.QwenGAWM import QwenGAWM,QwenGAWMConfig
from starVLA.model.modules.action_model.ACT_ActionHeader import TurboStyleACTActionHead
from starVLA.model.tools import FRAMEWORK_REGISTRY


def pool_camera_tokens(features, grid_thw, *, merged, merge_size=2):
    """Restore HF merge-block order to row/column order before 4x4 pooling."""
    pooled=[];offset=0
    for t,h,w in grid_thw.tolist():
        if t!=1 or h%merge_size or w%merge_size:
            raise ValueError('Expected still-image grids divisible by merge size')
        gh,gw=(h//merge_size,w//merge_size) if merged else (h,w)
        n=gh*gw;part=features[offset:offset+n];offset+=n
        if len(part)!=n:raise ValueError('Incomplete image token sequence')
        if merged:
            spatial=part.reshape(gh,gw,-1)
        else:
            spatial=part.reshape(h//merge_size,w//merge_size,merge_size,merge_size,-1)
            spatial=spatial.permute(0,2,1,3,4).reshape(h,w,-1)
        pooled.append(F.adaptive_avg_pool2d(spatial.permute(2,0,1).float()[None],(4,4))[0].flatten(1).T)
    if offset!=len(features):raise ValueError('Extra image tokens')
    return torch.stack(pooled)


class VisionOnlyInterface(nn.Module):
    def __init__(self,base_path,variant):
        super().__init__()
        cfg=AutoConfig.from_pretrained(base_path,local_files_only=True).vision_config
        cfg._attn_implementation='sdpa'
        visual=Qwen3VLVisionModel(cfg).to(torch.bfloat16)
        visual.deepstack_visual_indexes=[]
        visual.deepstack_merger_list=nn.ModuleList()
        if variant=='vit':visual.merger=nn.Identity()
        self.model=nn.Module();self.model.model=nn.Module()
        self.model.model.visual=visual
        self.processor=AutoProcessor.from_pretrained(base_path,local_files_only=True)
        self.requires_grad_(False)


@FRAMEWORK_REGISTRY.register('QwenGAWMDepth')
class QwenGAWMDepth(QwenGAWM):
    def __init__(self,config):
        variant=str(config.framework.depth_variant)
        if variant not in ('vit','merger','half'):raise ValueError('Unknown depth ablation')
        self.depth_variant=variant
        if variant=='half':
            super().__init__(config)
            language=self.qwen_vl_interface.model.model.language_model
            language.layers=nn.ModuleList(list(language.layers[:18]))
            language.config.num_hidden_layers=18
            self.qwen_vl_interface.model.config.text_config.num_hidden_layers=18
            self.qwen_vl_interface.requires_grad_(False)
            count=int(self.config.framework.qwen_training.train_last_n_layers)
            if count not in (0,4):raise ValueError('Expected frozen or last-four-layer adaptation')
            if count:
                for layer in language.layers[-count:]:layer.requires_grad_(True)
                language.norm.requires_grad_(True)
            assert len(language.layers)==18
        else:
            baseframework.__init__(self)
            self.config=merge_framework_config(QwenGAWMConfig,config)
            h=self.config.framework.action_model
            if int(h.action_horizon)!=16 or int(h.action_dim)!=14 or h.action_model_type!='ACT':
                raise ValueError('Only ACT16x14 supported')
            if int(self.config.framework.qwen_training.train_last_n_layers)!=0:
                raise ValueError('Pure vision tower and merger remain frozen')
            self.action_horizon=16;self.head_type='ACT';self.expects_normalized_state=False
            self.embodiment_head_specs={'aloha':dict(action_dim=14,action_horizon=16,state_dim=0,
                gripper_indices=[12,13],action_spec_id='aloha_dual_joint_contgrip_next_recorded_14')}
            self.qwen_vl_interface=VisionOnlyInterface(self.config.framework.qwenvl.base_vlm,variant)
            self.action_model=TurboStyleACTActionHead(token_dim=1024 if variant=='vit' else 2560,
                hidden_dim=int(h.action_hidden_dim),action_dim=14,horizon=16,num_frames=1,
                num_visual_tokens=48,num_heads=int(h.act_num_heads),num_layers=int(h.act_num_layers),
                dim_feedforward=int(h.act_dim_feedforward),mlp_hidden_dim=int(h.act_mlp_hidden_dim),
                dropout=float(h.act_dropout),state_dim=0,output_activation=h.output_activation,gripper_indices=(12,13))

    def remap_checkpoint_state_dict(self,state_dict):
        transfer=bool(self.config.trainer.get('pretrained_checkpoint')) and self.config.trainer.get('reload_modules')=='qwen_vl_interface'
        if not transfer:return super().remap_checkpoint_state_dict(state_dict)
        prefix='qwen_vl_interface.';expected={k:v for k,v in self.state_dict().items() if k.startswith(prefix)}
        if any(k not in state_dict or state_dict[k].shape!=v.shape for k,v in expected.items()):
            raise ValueError('Missing or incompatible retained OFT parameters')
        for k in state_dict:
            if not k.startswith(prefix) or k in expected:continue
            if self.depth_variant=='half':
                match=re.match(r'qwen_vl_interface\.model\.model\.language_model\.layers\.(\d+)\.',k)
                allowed=match is not None and 18<=int(match[1])<36
            else:
                allowed=k.startswith(prefix+'model.model.language_model.') or k.startswith(prefix+'model.lm_head.') or k.startswith(prefix+'model.model.visual.deepstack_merger_list.')
                if self.depth_variant=='vit':allowed=allowed or k.startswith(prefix+'model.model.visual.merger.')
            if not allowed:raise ValueError(f'Unexpected omitted Qwen parameter: {k}')
        return {k:state_dict[k] for k in expected}

    def _predict_tensor(self,examples):
        if self.depth_variant=='half':return super()._predict_tensor(examples)
        if not examples:raise ValueError('Nonempty batch required')
        images=[]
        for x in examples:
            if x.get('robot_tag','aloha')!='aloha' or x.get('action_spec_id',self.embodiment_head_specs['aloha']['action_spec_id'])!=self.embodiment_head_specs['aloha']['action_spec_id']:
                raise ValueError('Wrong embodiment/action specification')
            if len(x['image'])!=3 or not np.asarray(x.get('view_valid_mask',[True]*3)).all():
                raise ValueError('Three valid RGB views required')
            if str(x['lang']).strip() not in ('blocks ranking rgb','blocks_ranking_rgb',self.config.framework.task_instruction):
                raise ValueError('Unexpected task for single-task visual policy')
            images.extend(resize_images(to_pil_preserve(x['image']),target_size=self.config.datasets.vla_data.obs_image_size))
        visual=self.qwen_vl_interface.model.model.visual
        device=next(visual.parameters()).device
        inputs=self.qwen_vl_interface.processor.image_processor(images=images,return_tensors='pt').to(device)
        with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            features,_=visual(inputs['pixel_values'].to(next(visual.parameters()).dtype),grid_thw=inputs['image_grid_thw'])
            memory=pool_camera_tokens(features,inputs['image_grid_thw'],merged=self.depth_variant=='merger')
        memory=memory.reshape(len(examples),1,48,-1)
        with torch.autocast(device.type,enabled=False):
            return self.action_model(memory.to(next(self.action_model.parameters()).dtype)).float()
