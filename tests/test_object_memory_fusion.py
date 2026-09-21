import numpy as np
import pytest
import torch

from starVLA.model.modules.action_model.ACT_ActionHeader import TurboStyleACTActionHead
from starVLA.model.modules.object_memory_fusion import ObjectMemoryFusion
from starVLA.model.modules.rgb_object_readout import RGBObjectReadout


def setup():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    head = TurboStyleACTActionHead(token_dim=24, hidden_dim=32, action_dim=14, horizon=16,
        num_frames=2, num_visual_tokens=4, num_heads=4, num_layers=2, dim_feedforward=64,
        mlp_hidden_dim=32, dropout=0., state_dim=14, state_hidden_dim=32, num_state_tokens=2)
    fusion = ObjectMemoryFusion(32, 32)
    return head.eval(), fusion.eval(), torch.randn(2, 2, 4, 24), torch.randn(2, 14)


def test_full_fusion_reaches_each_decoder_layer_and_retains_action_shape():
    head, fusion, visual, state = setup()
    tokens = torch.randn(2, 3, 32, requires_grad=True)
    valid = torch.tensor([[True, True, True], [True, False, True]])
    seen = []
    handles = [layer.multihead_attn.register_forward_pre_hook(
        lambda module, inputs: seen.append(inputs[1].shape[1])) for layer in head.decoder.layers]
    queries = fusion(head, visual, state, (tokens, valid))
    actions = head.predict_action(queries)
    assert actions.shape == (2, 16, 14) and torch.isfinite(actions).all()
    assert len(seen) == 2 and all(length == 13 for length in seen)
    actions.square().mean().backward()
    assert tokens.grad[0].abs().sum() > 0 and tokens.grad[1, 1].abs().sum() == 0
    for handle in handles:
        handle.remove()


def test_object_only_path_excludes_visual_shortcut_but_uses_objects():
    head, fusion, visual, state = setup()
    tokens = torch.randn(2, 3, 32)
    valid = torch.ones(2, 3, dtype=torch.bool)
    a = head.predict_action(fusion(head, visual, state, (tokens, valid), objects_only=True))
    b = head.predict_action(fusion(head, torch.full_like(visual, float('nan')), state,
                                  (tokens, valid), objects_only=True))
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    changed = tokens.clone(); changed[0, 0] += torch.linspace(-2, 2, 32)
    c = head.predict_action(fusion(head, visual, state, (changed, valid), objects_only=True))
    assert (a[0]-c[0]).abs().max() > 1e-5
    torch.testing.assert_close(a[1], c[1], atol=0, rtol=0)


@pytest.mark.parametrize('objects_only', [False, True])
def test_missing_objects_are_safe_with_proprioception(objects_only):
    head, fusion, visual, state = setup()
    valid = torch.zeros(2, 3, dtype=torch.bool)
    tokens = torch.full((2, 3, 32), float('nan'), requires_grad=True)
    a = fusion(head, visual, state, (tokens, valid), objects_only=objects_only)
    b = fusion(head, visual, state, (torch.zeros_like(tokens), valid), objects_only=objects_only)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.isfinite(a).all()
    a.square().mean().backward()
    assert torch.count_nonzero(tokens.grad) == 0


def test_action_gradient_reaches_learned_object_encoder_without_auxiliary_labels():
    head, fusion, visual, state = setup()
    reader = RGBObjectReadout(32)
    rgb = np.full((240, 320, 3), 180, dtype=np.uint8)
    for i in range(3):
        rgb[100:120, 40+i*80:60+i*80] = 0
        rgb[100:120, 40+i*80:60+i*80, i] = 255
    examples = [dict(native_images=[rgb]*3)]*2
    images, _ = reader.head_images(examples, torch.ones(2, dtype=torch.bool), 'cpu')
    memory, _ = reader.encode(images, torch.ones(2, dtype=torch.bool))
    actions = head.predict_action(fusion(head, visual, state, memory, objects_only=True))
    actions.square().mean().backward()
    for parameter in (reader.encoder[0].weight, reader.heatmaps.weight, reader.identity.weight,
                      reader.project.weight, fusion.project.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
