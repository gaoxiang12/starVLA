import numpy as np
import pytest
import torch
from PIL import Image

from starVLA.dataloader.rgb_object_supervision import candidates
from starVLA.model.modules.rgb_object_readout import RGBObjectReadout


def scene():
    rgb = np.full((240, 320, 3), 180, dtype=np.uint8)
    for index, x in enumerate((30, 150, 270)):
        rgb[100:120, x:x+20] = 0
        rgb[100:120, x:x+20, index] = 255
    return rgb


def test_color_identity_and_clipped_component_mask():
    rgb = scene()
    labels = candidates(rgb)
    assert [row['color'] for row in labels] == ['red', 'green', 'blue']
    assert all(row['accepted'] for row in labels)
    np.testing.assert_allclose([row['center_xy'] for row in labels], [[39.5, 109.5], [159.5, 109.5], [279.5, 109.5]])
    rgb[:, :50] = 180
    rgb[100:120, :20] = [255, 0, 0]
    red = candidates(rgb)[0]
    assert not red['accepted'] and red['touches_image_edge'] and red['area'] == 400


def test_native_input_does_not_resize_or_accept_broken_layout():
    rgb = scene()
    images, arrays = RGBObjectReadout.head_images([{'native_images': [Image.fromarray(rgb)] * 3}], torch.tensor([True]), 'cpu')
    assert images.shape == (1, 3, 240, 320)
    np.testing.assert_array_equal(arrays[0], rgb)
    with pytest.raises(ValueError, match='320x240'):
        RGBObjectReadout.head_images([{'native_images': [rgb[:100]] * 3}], torch.tensor([True]), 'cpu')


def test_zero_initial_action_residual_and_later_action_gradient(monkeypatch):
    torch.manual_seed(42)
    model = RGBObjectReadout(16)
    images = torch.randn(2, 3, 32, 40, requires_grad=True)
    valid = torch.tensor([True, False])
    # Inference must not depend on the training-only heuristic.
    def forbidden(_):
        raise AssertionError('Training label generator reached during inference')
    monkeypatch.setattr('starVLA.dataloader.rgb_object_supervision.candidates', forbidden)
    memories, prediction = model.encode(images, valid)
    queries = torch.randn(2, 4, 16)
    torch.testing.assert_close(model.refine(queries, memories), queries, atol=0, rtol=0)
    with torch.no_grad():
        model.readout.attention.out_proj.weight.normal_(std=.01)
    actions = model.refine(queries, memories)
    torch.testing.assert_close(actions[1], queries[1], atol=0, rtol=0)
    actions[0].square().mean().backward()
    for parameter in (model.encoder[0].weight, model.heatmaps.weight, model.visibility.weight,
                      model.identity.weight, model.project.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert images.grad[0].abs().sum() > 0 and images.grad[1].abs().sum() == 0


def test_masked_nan_head_cannot_affect_prediction_or_supervision():
    model = RGBObjectReadout(16)
    images = torch.full((1, 3, 32, 40), float('nan'))
    valid = torch.tensor([False])
    memories, prediction = model.encode(images, valid)
    assert torch.isfinite(memories[0]).all() and not memories[1].any()
    loss, metrics = model.supervision(prediction, [scene()], valid, {})
    assert loss.item() == 0 and all(torch.isfinite(v) for v in metrics.values())
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_image_supervision_trains_locator_without_action_warmup():
    torch.manual_seed(42)
    model = RGBObjectReadout(16)
    rgb = scene()
    images, arrays = model.head_images([{'native_images': [rgb] * 3}], torch.tensor([True]), 'cpu')
    images = torch.nn.functional.interpolate(images, size=(60, 80), mode='area')
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    losses = []
    for _ in range(6):
        optimizer.zero_grad()
        _, prediction = model.encode(images, torch.tensor([True]))
        loss, metrics = model.supervision(prediction, arrays, torch.tensor([True]), {})
        loss.backward()
        assert model.heatmaps.weight.grad.abs().sum() > 0
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert metrics['object_label_fraction'] == 1
