import pytest
import torch

from starVLA.model.modules.action_model.CompactFlowActionHead import CompactFlowActionHead


def setup():
    torch.manual_seed(9)
    head = CompactFlowActionHead(context_dim=24, hidden_dim=32, horizon=4,
                                 depth=4, heads=4, planning_tokens=2)
    memory = torch.randn(2, 7, 24)
    state = torch.randn(2, 14)
    action = torch.randn(2, 4, 14)
    noise = torch.randn_like(action)
    time = torch.tensor([.2, .8])
    return head, memory, state, action, noise, time


def test_padding_values_do_not_change_valid_predictions_or_loss():
    head, memory, state, action, noise, time = setup()
    mv = torch.tensor([[True]*4+[False]*3, [True]*6+[False]])
    av = torch.tensor([[True, True, False, False], [True, True, True, False]])
    baseline = head.loss(memory, state, action, noise=noise, time=time,
                         context_valid=mv, action_valid=av)
    padded_memory, padded_action, padded_noise = memory.clone(), action.clone(), noise.clone()
    padded_memory[~mv] = float('nan')
    padded_action[~av] = float('nan')
    padded_noise[~av] = float('nan')
    changed = head.loss(padded_memory, state, padded_action, noise=padded_noise, time=time,
                        context_valid=mv, action_valid=av)
    torch.testing.assert_close(baseline['loss'], changed['loss'], atol=0, rtol=0)
    torch.testing.assert_close(baseline['estimated_actions'], changed['estimated_actions'], atol=0, rtol=0)
    assert torch.count_nonzero(changed['estimated_actions'][~av]) == 0


def test_valid_visual_and_state_gradients_and_masked_target_gradients():
    head, memory, state, action, noise, time = setup()
    mv = torch.tensor([[True]*6+[False], [True]*6+[False]])
    av = torch.tensor([[True, True, False, False], [True, True, True, False]])
    memory.requires_grad_(); state.requires_grad_(); action.requires_grad_()
    result = head.loss(memory, state, action, noise=noise, time=time,
                       context_valid=mv, action_valid=av)
    result['loss'].backward()
    for value in (memory, state, action):
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert torch.count_nonzero(memory.grad[~mv]) == 0
    assert torch.count_nonzero(action.grad[~av]) == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())


def test_inference_uses_state_and_memory_with_reproducible_noise():
    head, memory, state, action, noise, time = setup()
    head.eval()
    first = head.predict_action(memory, state, initial_noise=noise)
    repeat = head.predict_action(memory, state, initial_noise=noise)
    torch.testing.assert_close(first, repeat, atol=0, rtol=0)
    assert first.shape == action.shape and torch.isfinite(first).all()
    assert not torch.allclose(first, head.predict_action(memory, state+1, initial_noise=noise))
    assert not torch.allclose(first, head.predict_action(memory+1, state, initial_noise=noise))


def test_empty_action_mask_is_zero_loss_and_empty_context_is_rejected():
    head, memory, state, action, noise, time = setup()
    av = torch.zeros(2, 4, dtype=torch.bool)
    action.fill_(float('nan'))
    result = head.loss(memory, state, action, noise=noise, time=time, action_valid=av)
    assert result['loss'].item() == 0
    result['loss'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
    with pytest.raises(ValueError, match='valid context'):
        head.predict_action(memory, state, context_valid=torch.zeros(2, 7, dtype=torch.bool))
