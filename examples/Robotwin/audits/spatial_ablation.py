"""Offline interventions for historical GAWM source snapshots with spatial memories.

Current GAWM and its policy server no longer expose this retired experiment.
"""


def apply_spatial_ablation(model, mode):
    selected = {
        'full': {'dense', 'local', 'goal', 'objects'},
        'no_focus': set(),
        'no_dense': {'local', 'goal', 'objects'},
        'no_local': {'dense', 'goal', 'objects'},
        'no_goal': {'dense', 'local', 'objects'},
        'no_objects': {'dense', 'local', 'goal'},
    }
    if mode not in selected:
        raise ValueError(f'Unknown spatial ablation: {mode}')
    object_hook = getattr(model, 'configure_object_memory_ablation', None)
    if object_hook is not None:
        object_hook(mode)
    if mode == 'full':
        return
    focus = getattr(model, 'spatial_focus', None)
    if focus is None:
        raise ValueError('Spatial ablation requires a model with spatial_focus')
    original = focus.refine_queries
    keep = selected[mode]
    focus.refine_queries = lambda queries, memories: original(
        queries, {key: value for key, value in memories.items() if key in keep})
