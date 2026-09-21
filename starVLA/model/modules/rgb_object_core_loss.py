"""Opt-in object auxiliary loss with uncertain red logo targets masked out."""
import numpy as np
import torch
from torch.nn import functional as F

from starVLA.dataloader.rgb_object_supervision_core import candidates_with_core


def object_core_supervision(prediction, arrays, head_valid, cfg):
    """Same losses as RGBObjectReadout, with the audited red-core label masks.

    This function is not yet selected by the pinned running model. Coordinate
    and presence masks are separate: rejecting an uncertain red positive must
    not create a negative presence label.
    """
    labels = [candidates_with_core(rgb) for rgb in arrays]
    device = prediction['xy'].device
    target = torch.as_tensor(np.asarray([[(np.array(obj['center_xy'])+.5)/[320, 240]
        if obj['accepted'] else [0., 0.] for obj in row] for row in labels]),
        device=device, dtype=torch.float32)
    valid = torch.tensor([[obj['accepted'] for obj in row] for row in labels], device=device)
    valid &= head_valid[:, None]
    visible = torch.tensor([[obj['area'] >= 12 for obj in row] for row in labels],
                           device=device, dtype=torch.float32)
    visibility_valid = torch.tensor([[obj['visibility_supervision_valid'] for obj in row]
                                    for row in labels], device=device)
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
