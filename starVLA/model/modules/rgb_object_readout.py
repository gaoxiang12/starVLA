"""Learn three color-identity tokens from current native head RGB at stride four."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from starVLA.model.modules.spatial_goal_readout import SpatialGoalReadout


class RGBObjectReadout(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU())
        self.heatmaps = nn.Conv2d(64, 3, 1)
        self.visibility = nn.Linear(64, 3)
        self.project = nn.Linear(64, hidden_dim)
        self.identity = nn.Embedding(3, hidden_dim)
        self.readout = SpatialGoalReadout(hidden_dim)

    @staticmethod
    def head_images(examples, head_valid, device):
        if head_valid.ndim != 1 or len(examples) != head_valid.numel():
            raise ValueError('One head validity flag is required per example')
        arrays = []
        for example, valid in zip(examples, head_valid.detach().cpu().tolist()):
            if not valid:
                arrays.append(np.zeros((240, 320, 3), dtype=np.uint8))
                continue
            native = example.get('native_images')
            if native is None or len(native) != 3:
                raise ValueError('RGB object readout requires native head/left/right images')
            array = np.asarray(native[0])
            if array.shape != (240, 320, 3) or array.dtype != np.uint8:
                raise ValueError('RGB object readout expects native uint8 RGB 320x240 head images')
            arrays.append(array)
        tensor = torch.as_tensor(np.stack(arrays), device=device, dtype=torch.float32)
        return tensor.permute(0, 3, 1, 2).contiguous() / 127.5 - 1., arrays

    def encode(self, images, head_valid):
        images = torch.where(head_valid[:, None, None, None], images, 0.)
        features = self.encoder(images)
        b, _, h, w = features.shape
        logits = self.heatmaps(features).flatten(2)
        y, x = torch.meshgrid((torch.arange(h, device=images.device)+.5)/h,
                              (torch.arange(w, device=images.device)+.5)/w, indexing='ij')
        coordinates = torch.stack((x, y), -1).reshape(-1, 2)
        xy = logits.softmax(-1) @ coordinates
        visibility = self.visibility(features.mean((-2, -1)))
        # Pool at 64 channels before the linear projection: mathematically
        # equivalent, without materializing three full 384-channel image maps.
        pooled = logits.softmax(-1) @ features.flatten(2).transpose(1, 2)
        valid = head_valid[:, None].expand(b, 3)
        tokens = self.readout.feature_norm(self.project(pooled)) + self.readout.coordinates(xy)
        tokens = (tokens + self.identity.weight[None]) * visibility.sigmoid()[..., None]
        tokens = torch.where(valid[..., None], tokens, 0.)
        return (tokens, valid), dict(logits=logits, xy=xy, visibility=visibility,
                                    coordinates=coordinates, grid_hw=(h, w))

    def supervision(self, prediction, arrays, head_valid, cfg):
        # Lazy import: neither color thresholds nor candidate labels are used
        # to select objects, positions, tokens, or actions during inference.
        from starVLA.dataloader.rgb_object_supervision import candidates
        labels = [candidates(rgb) for rgb in arrays]
        device = prediction['xy'].device
        target = torch.as_tensor(np.asarray([[(np.array(obj['center_xy'])+.5)/[320, 240]
                                if obj['accepted'] else [0., 0.] for obj in row]
                               for row in labels]), device=device, dtype=torch.float32)
        valid = torch.tensor([[obj['accepted'] for obj in row] for row in labels], device=device)
        valid &= head_valid[:, None]
        visible = torch.tensor([[obj['area'] >= 12 for obj in row] for row in labels],
                               device=device, dtype=torch.float32)
        visibility_valid = torch.tensor([[obj['area'] < 12 or obj['dominance'] >= .7
                                         for obj in row] for row in labels], device=device)
        visibility_valid &= head_valid[:, None]
        h, w = prediction['grid_hw']
        distance = ((prediction['coordinates'][None, None] - target[:, :, None]) *
                    target.new_tensor([w, h])).square().sum(-1)
        distribution = (-distance / 2.).softmax(-1)
        heatmap = (-(distribution * prediction['logits'].log_softmax(-1)).sum(-1) * valid).sum() / valid.sum().clamp_min(1)
        coordinate = ((prediction['xy'] - target).abs().sum(-1) * valid).sum() / valid.sum().clamp_min(1)
        presence = (F.binary_cross_entropy_with_logits(prediction['visibility'], visible, reduction='none') *
                    visibility_valid).sum() / visibility_valid.sum().clamp_min(1)
        loss = (float(cfg.get('object_heatmap_loss_weight', .01)) * heatmap +
                float(cfg.get('object_coordinate_loss_weight', .05)) * coordinate +
                float(cfg.get('object_visibility_loss_weight', .01)) * presence)
        return loss, dict(object_heatmap_loss=heatmap.detach(), object_coordinate_l1=coordinate.detach(),
                          object_visibility_loss=presence.detach(), object_label_fraction=valid.float().mean())

    def refine(self, queries, memory):
        return self.readout(queries, *memory)
