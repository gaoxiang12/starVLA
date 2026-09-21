import numpy as np
import pytest
import torch

from starVLA.model.framework.VLM4A.QwenGAWM import QwenGAWM, gather_action_features
from starVLA.model.modules.action_model.ACT_ActionHeader import TurboStyleACTActionHead


def test_action_token_gather_preserves_order_padding_and_gradients():
    hidden = torch.randn(2, 7, 12, requires_grad=True)
    ids = torch.tensor([[0,0,9,1,9,9,2], [0,1,9,9,9,1,2]])
    selected = gather_action_features(hidden, ids, 9, 3)
    torch.testing.assert_close(selected[0], hidden[0,[2,4,5]])
    selected.sum().backward()
    assert torch.all(hidden.grad[ids != 9] == 0)
    with pytest.raises(ValueError, match='exactly'):
        gather_action_features(hidden, ids, 9, 4)


def test_qwen_memory_trains_full_act_decoder_and_input_projection():
    torch.manual_seed(7)
    head = TurboStyleACTActionHead(token_dim=24, hidden_dim=32, action_dim=14,
        horizon=16, num_frames=1, num_visual_tokens=16, num_heads=4,
        num_layers=2, dim_feedforward=64, mlp_hidden_dim=32, dropout=0, state_dim=0)
    memory = torch.randn(2,1,16,24, requires_grad=True)
    actions = head(memory)
    assert actions.shape == (2,16,14)
    actions.abs().mean().backward()
    for grad in (memory.grad, head.visual_projection.weight.grad,
                 head.decoder.layers[0].multihead_attn.in_proj_weight.grad,
                 head.action_queries.weight.grad):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert not torch.allclose(actions, head(memory.detach()+1))


def test_masked_padding_nan_does_not_enter_loss_or_gradient():
    model = QwenGAWM.__new__(QwenGAWM)
    torch.nn.Module.__init__(model)
    prediction = torch.randn(1,16,14,requires_grad=True)
    model._predict_tensor = lambda examples: prediction
    target = np.zeros((16,14),np.float32)
    target[8:] = np.nan
    result = model.forward([dict(action=target,action_valid_mask=np.arange(16)<8)])
    torch.testing.assert_close(result['action_loss'], prediction[:,:8].abs().mean())
    result['action_loss'].backward()
    assert torch.all(prediction.grad[:,8:] == 0)


def test_paired_success_statistics_keep_gains_and_losses():
    from examples.Robotwin.audits.summarize_qwen_gawm_ranking import paired_statistics
    r=paired_statistics([1,1,0,0],[1,0,1,1])
    assert r['gained']==2 and r['lost']==1
    assert r['difference_percentage_points']==25
    assert r['exact_mcnemar_p']==1
    same=paired_statistics([0,1],[0,1])
    assert same['paired_bootstrap_95_percentile_ci_pp']==[0.,0.]
