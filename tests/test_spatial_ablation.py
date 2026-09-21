from types import SimpleNamespace

import pytest
import torch

from deployment.model_server.tools.spatial_ablation import apply_spatial_ablation


@pytest.mark.parametrize('mode,expected', [('full', 7.), ('no_dense', 5.), ('no_local', 3.), ('no_focus', 1.)])
def test_ablation_filters_memories_without_changing_queries_or_weights(mode, expected):
    weight = torch.nn.Parameter(torch.tensor(2.))
    def refine(queries, memories):
        return queries + sum(memories.values()) * weight
    model = SimpleNamespace(spatial_focus=SimpleNamespace(refine_queries=refine))
    queries = torch.tensor([1.])
    memories = dict(dense=torch.tensor([1.]), local=torch.tensor([2.]))
    apply_spatial_ablation(model, mode)
    torch.testing.assert_close(model.spatial_focus.refine_queries(queries, memories), torch.tensor([expected]))
    assert weight.item() == 2.
    assert queries.item() == 1.
    assert set(memories) == {'dense', 'local'}


def test_ablation_rejects_inapplicable_or_invalid_interventions():
    apply_spatial_ablation(SimpleNamespace(), 'full')
    with pytest.raises(ValueError):
        apply_spatial_ablation(SimpleNamespace(), 'no_focus')
    with pytest.raises(ValueError):
        apply_spatial_ablation(SimpleNamespace(), 'typo')


@pytest.mark.parametrize('mode,expected', [('full', 7), ('no_dense', 6), ('no_local', 5),
                                         ('no_goal', 3), ('no_focus', 0)])
def test_goal_memory_has_an_independent_ablation(mode, expected):
    model = SimpleNamespace(spatial_focus=SimpleNamespace(refine_queries=lambda q, m: q + sum(m.values())))
    apply_spatial_ablation(model, mode)
    assert model.spatial_focus.refine_queries(0, dict(dense=1, local=2, goal=4)) == expected


@pytest.mark.parametrize('mode,expected', [('full', 15), ('no_dense', 14), ('no_local', 13),
                                         ('no_goal', 11), ('no_objects', 7), ('no_focus', 0)])
def test_object_memory_is_preserved_unless_explicitly_ablated(mode, expected):
    model = SimpleNamespace(spatial_focus=SimpleNamespace(refine_queries=lambda q, m: q + sum(m.values())))
    apply_spatial_ablation(model, mode)
    assert model.spatial_focus.refine_queries(0, dict(dense=1, local=2, goal=4, objects=8)) == expected


def test_early_object_memory_receives_and_resets_independent_intervention():
    calls = []
    model = SimpleNamespace(spatial_focus=SimpleNamespace(refine_queries=lambda q, m: q),
                            configure_object_memory_ablation=calls.append)
    for mode in ('no_objects', 'no_focus', 'full'):
        apply_spatial_ablation(model, mode)
    assert calls == ['no_objects', 'no_focus', 'full']
    with pytest.raises(ValueError):
        apply_spatial_ablation(model, 'typo')
    assert calls[-1] == 'full'
