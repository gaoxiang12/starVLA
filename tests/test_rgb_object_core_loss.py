import numpy as np
import torch

from starVLA.model.modules.rgb_object_readout import RGBObjectReadout
from starVLA.model.modules.rgb_object_core_loss import object_core_supervision


def scene(red):
    rgb = np.full((240, 320, 3), 180, np.uint8)
    for i, color in enumerate((red, [0, 255, 0], [0, 0, 255])):
        rgb[100:120, 40+i*70:60+i*70] = color
    return rgb


def prediction(rgb):
    model = RGBObjectReadout(16)
    valid = torch.ones(1, dtype=torch.bool)
    images, arrays = model.head_images([dict(native_images=[rgb]*3)], valid, 'cpu')
    _, pred = model.encode(images, valid)
    return model, pred, arrays, valid


def test_unchanged_labels_preserve_loss_and_prediction_gradients():
    torch.set_num_threads(2)
    model, pred, arrays, valid = prediction(scene([255, 0, 0]))
    old, old_metrics = model.supervision(pred, arrays, valid, {})
    new, new_metrics = object_core_supervision(pred, arrays, valid, {})
    torch.testing.assert_close(old, new, atol=0, rtol=0)
    for key in old_metrics:
        torch.testing.assert_close(old_metrics[key], new_metrics[key], atol=0, rtol=0)
    tensors = (pred['logits'], pred['xy'], pred['visibility'])
    old_grad = torch.autograd.grad(old, tensors, retain_graph=True)
    new_grad = torch.autograd.grad(new, tensors)
    for a, b in zip(old_grad, new_grad):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_uncertain_logo_has_zero_auxiliary_gradient_but_other_colors_learn():
    torch.set_num_threads(2)
    _, pred, arrays, valid = prediction(scene([95, 37, 39]))
    loss, _ = object_core_supervision(pred, arrays, valid, {})
    tensors = (pred['logits'], pred['xy'], pred['visibility'])
    gradients = torch.autograd.grad(loss, tensors)
    for grad in gradients:
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad[0, 0]) == 0
        assert torch.count_nonzero(grad[0, 1:]) > 0


def test_missing_head_has_zero_loss_and_gradients():
    torch.set_num_threads(2)
    _, pred, arrays, valid = prediction(scene([95, 37, 39]))
    loss, metrics = object_core_supervision(pred, arrays, ~valid, {})
    assert loss == 0 and all(torch.isfinite(v) for v in metrics.values())
    gradients = torch.autograd.grad(loss, (pred['logits'], pred['xy'], pred['visibility']))
    assert all(torch.count_nonzero(g) == 0 for g in gradients)
