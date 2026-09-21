import torch

from starVLA.model.modules.spatial_focus import SpatialFocus
from starVLA.model.modules.spatial_goal_readout import SpatialGoalReadout


def test_warm_start_preserves_trained_queries_and_only_adds_goal_weights():
    torch.manual_seed(3)
    old = SpatialFocus(8, 16, 4, 6, 3, {'residual_mode': 'zero_projection'})
    with torch.no_grad():
        old.attentions[0].out_proj.weight.normal_(std=.1)
    new = SpatialFocus(8, 16, 4, 6, 3, {'residual_mode': 'zero_projection', 'goal_readout': True})
    incompatible = new.load_state_dict(old.state_dict(), strict=False)
    assert incompatible.missing_keys and all(k.startswith('goal_readout.') for k in incompatible.missing_keys)
    assert not incompatible.unexpected_keys
    queries = torch.randn(2, 16, 16)
    dense = (torch.randn(2, 48, 16), torch.ones(2, 48, dtype=torch.bool))
    goal = (torch.randn(2, 3, 16), torch.ones(2, 3, dtype=torch.bool))
    torch.testing.assert_close(new.refine_queries(queries, {'dense': dense, 'goal': goal}),
                               old.refine_queries(queries, {'dense': dense}), atol=0, rtol=0)


def test_action_loss_reaches_waypoint_selection_after_first_output_update():
    torch.manual_seed(7)
    focus = SpatialFocus(8, 16, 4, 6, 3, {'goal_readout': True, 'view_conditioned_query': True})
    patches, state, task = torch.randn(2, 3, 16, 8), torch.randn(2, 4), torch.randn(2, 6)
    views = torch.ones(2, 3, dtype=torch.bool)
    queries, target = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
    optimizer = torch.optim.Adam(focus.parameters(), lr=.01)
    for iteration in range(2):
        optimizer.zero_grad(set_to_none=True)
        memory, logits, xy = focus.locate(patches, state, task)
        goal = focus.goal_readout.pack(memory, logits, xy, views)
        output = focus.refine_queries(queries, {'goal': goal})
        (output - target).square().mean().backward()
        if iteration == 0:
            assert focus.goal_readout.attention.out_proj.weight.grad.abs().sum() > 0
        else:
            # No auxiliary spatial loss is involved in this test.
            assert focus.query[0].weight.grad.abs().sum() > 0
            assert focus.key.weight.grad.abs().sum() > 0
            assert focus.offset.weight.grad.abs().sum() > 0
            assert focus.goal_readout.coordinates[0].weight.grad.abs().sum() > 0
        optimizer.step()
    with torch.no_grad():
        original = focus.refine_queries(queries, {'goal': focus.goal_readout.pack(memory, logits, xy, views)})
        moved = focus.refine_queries(queries, {'goal': focus.goal_readout.pack(memory, logits, 1 - xy, views)})
        assert (original - moved).abs().max() > 1e-5


def test_masked_nan_views_have_no_output_or_gradient_effect():
    torch.manual_seed(9)
    readout = SpatialGoalReadout(16)
    with torch.no_grad():
        readout.attention.out_proj.weight.normal_(std=.1)
    views = torch.tensor([[True, False, True], [False, False, False]])
    memory = torch.randn(2, 3, 4, 16).masked_fill(~views[..., None, None], float('nan')).requires_grad_()
    logits = torch.randn(2, 3, 4).masked_fill(~views[..., None], float('nan')).requires_grad_()
    xy = torch.rand(2, 3, 2).masked_fill(~views[..., None], float('nan')).requires_grad_()
    queries = torch.randn(2, 4, 16)
    output = readout(queries, *readout.pack(memory, logits, xy, views))
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[1], queries[1], atol=0, rtol=0)
    output.square().mean().backward()
    for tensor in (memory, logits, xy):
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad[~views].abs().sum() == 0
        assert tensor.grad[views].abs().sum() > 0
