"""Small dense/current and native-image crop branch for precise interaction."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image


def patch_coordinates(side, device):
    axis = (torch.arange(side, device=device, dtype=torch.float32) + .5) / side
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x, y), -1).reshape(-1, 2)


def native_crops(images, centers, fraction):
    """Crop before encoder resize; return actual normalized boxes after clipping."""
    result, boxes = [], []
    for views, xy_views in zip(images, centers.detach().cpu().numpy()):
        sample, sample_boxes = [], []
        for image, (x, y) in zip(views, xy_views):
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image, dtype=np.uint8))
            w, h = image.size
            cw, ch = max(1, round(w*fraction)), max(1, round(h*fraction))
            left = int(np.clip(round(float(x)*w-cw/2), 0, w-cw))
            top = int(np.clip(round(float(y)*h-ch/2), 0, h-ch))
            sample.append(image.crop((left, top, left+cw, top+ch)))
            sample_boxes.append([left/w, top/h, cw/w, ch/h])
        result.append([sample])
        boxes.append(sample_boxes)
    return result, torch.tensor(boxes, dtype=torch.float32, device=centers.device)


class SpatialFocus(nn.Module):
    def __init__(self, patch_dim, hidden_dim, state_dim, task_dim, num_views, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_views = num_views
        self.project = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim,hidden_dim), nn.LayerNorm(hidden_dim))
        self.position = nn.Sequential(nn.Linear(2,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,hidden_dim))
        self.view = nn.Embedding(num_views, hidden_dim)
        self.query = nn.Sequential(nn.Linear(state_dim+task_dim,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,hidden_dim))
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.offset = nn.Linear(hidden_dim, 2)
        self.box_position = nn.Sequential(nn.Linear(4,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,hidden_dim))
        self.kind = nn.Embedding(2,hidden_dim)
        self.gates = nn.Parameter(torch.zeros(2))
        self.query_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.attentions = nn.ModuleList([nn.MultiheadAttention(hidden_dim,4,batch_first=True) for _ in range(2)])
        self.residual_mode = str(cfg.get('residual_mode', 'scalar_gate'))
        if self.residual_mode not in ('scalar_gate', 'zero_projection'):
            raise ValueError(f'Unknown spatial residual mode: {self.residual_mode}')
        if self.residual_mode == 'zero_projection':
            # Preserve the initial policy while letting output directions learn
            # independently, without a single scalar suppressing all gradients.
            for attention in self.attentions:
                nn.init.zeros_(attention.out_proj.weight)
                nn.init.zeros_(attention.out_proj.bias)
            self.gates.requires_grad_(False)
        self.goal_readout = None
        if cfg.get('goal_readout', False):
            from starVLA.model.modules.spatial_goal_readout import SpatialGoalReadout
            self.goal_readout = SpatialGoalReadout(hidden_dim)
        self.object_readout = None
        if cfg.get('object_readout', False):
            if num_views != 3:
                raise ValueError('RGB object readout requires head/left/right camera order')
            from starVLA.model.modules.rgb_object_readout import RGBObjectReadout
            self.object_readout = RGBObjectReadout(hidden_dim)

    def residual_scales(self):
        if self.residual_mode == 'zero_projection':
            return torch.ones_like(self.gates)
        return self.gates.tanh()

    def locate(self, patches, state, task):
        b,v,n,_ = patches.shape
        side = math.isqrt(n)
        if side*side != n or v != self.num_views:
            raise ValueError("SpatialFocus requires square per-camera patches")
        coords = patch_coordinates(side,patches.device)
        features = self.project(patches.float())
        memory = features + self.position(coords)[None,None] + self.view.weight[None,:,None]
        query = self.query(torch.cat((state,task),-1))[:,None,None]
        if self.cfg.get('view_conditioned_query', False):
            # An additive camera embedding in memory alone contributes the same
            # scalar to every patch logit, which cancels in softmax. Conditioning
            # the query introduces camera-dependent feature/position scoring.
            # Opt-in preserves the behavior of already-running v1 experiments.
            query = query + self.view.weight[None,:,None]
        logits = (self.key(memory)*query).sum(-1) / math.sqrt(memory.shape[-1])
        probs = logits.softmax(-1)
        subpatch = self.offset(features).tanh() / (2*side)
        xy = (probs[...,None]*(coords[None,None]+subpatch)).sum(-2).clamp(0,1)
        return memory, logits, xy

    def supervision(self, logits, xy, examples):
        target = torch.as_tensor(np.asarray([x['spatial_target_xy'] for x in examples]),device=xy.device,dtype=torch.float32)
        mask = torch.as_tensor(np.asarray([x['spatial_target_valid'] for x in examples]),device=xy.device,dtype=torch.bool)
        views = torch.as_tensor(np.asarray([x['view_valid_mask'] for x in examples]),device=xy.device,dtype=torch.bool)
        mask &= views
        # Sanitise masked labels BEFORE arithmetic; 0*NaN is still NaN.
        if not torch.isfinite(target[mask]).all():
            raise ValueError('Nonfinite valid spatial target')
        target = torch.where(mask[...,None],target,torch.zeros_like(target))
        side = math.isqrt(logits.shape[-1])
        coords = patch_coordinates(side, xy.device)
        dist = ((coords[None,None]-target[:,:,None])*side).square().sum(-1)
        soft_target = (-dist/(2*.7**2)).softmax(-1)
        ce = -(soft_target*logits.log_softmax(-1)).sum(-1)
        denom = mask.sum().clamp_min(1)
        heatmap = (ce*mask).sum()/denom
        error = (xy-target).abs().sum(-1)
        coord = (error*mask).sum()/denom
        loss = float(self.cfg.get('heatmap_loss_weight',.002))*heatmap + float(self.cfg.get('coordinate_loss_weight',.02))*coord
        return loss, target, mask, dict(spatial_heatmap_loss=heatmap.detach(),spatial_coordinate_l1=coord.detach(),spatial_label_fraction=mask.float().mean())

    def crop_centers(self, predicted, target, valid, step):
        if not self.training:
            return predicted.detach(), 0.
        warm = int(self.cfg.get('teacher_warmup_steps',500))
        end = int(self.cfg.get('teacher_end_steps',3000))
        probability = max(0.,min(1.,(end-step)/max(1,end-warm)))
        use_teacher = (torch.rand_like(predicted[...,0]) < probability) & valid
        centers = torch.where(use_teacher[...,None],target,predicted.detach())
        centers = (centers+torch.randn_like(centers)*float(self.cfg.get('crop_jitter',.025))).clamp(0,1)
        return centers, probability

    def pack_dense(self, memory, view_mask):
        memory = memory+self.kind.weight[0]
        valid = view_mask[:,:,None].expand(*memory.shape[:3])
        return memory.flatten(1,2), valid.flatten(1,2)

    def pack_local(self, patches, boxes, view_mask):
        b,v,n,d = patches.shape
        side = math.isqrt(n)
        # Local patch coordinates are mapped back to the ORIGINAL full image.
        coords = patch_coordinates(side, patches.device)[None,None]
        coords = boxes[:,:,None,:2]+coords*boxes[:,:,None,2:]
        memory = self.project(patches.float())+self.position(coords)+self.view.weight[None,:,None]+self.box_position(boxes)[:,:,None]+self.kind.weight[1]
        valid = view_mask[:,:,None].expand(b,v,n).clone()
        if self.training:
            keep = torch.rand(b,v,1,device=patches.device) >= float(self.cfg.get('crop_dropout',.15))
            valid &= keep
        return memory.flatten(1,2),valid.flatten(1,2)

    def refine_queries(self, queries, memories):
        # A zero residual gate exactly preserves the pretrained policy at
        # initialization. Scaling appended tokens does NOT preserve it because
        # even zero-valued extra keys change softmax's denominator.
        for index,key in enumerate(('dense','local')):
            if key not in memories:
                continue
            memory,valid=memories[key]
            available=valid.any(-1)
            valid=valid.clone()
            valid[~available,0]=True
            memory=torch.where(available[:,None,None],memory,torch.zeros_like(memory))
            residual=self.attentions[index](self.query_norms[index](queries),memory,memory,key_padding_mask=~valid,need_weights=False)[0]
            queries=queries+self.residual_scales()[index]*residual*available[:,None,None]
        if 'goal' in memories:
            if self.goal_readout is None:
                raise ValueError('Goal memory requires the configured goal readout')
            queries = self.goal_readout(queries, *memories['goal'])
        if 'objects' in memories:
            if self.object_readout is None:
                raise ValueError('Object memory requires the configured RGB object readout')
            queries = self.object_readout.refine(queries, memories['objects'])
        return queries
