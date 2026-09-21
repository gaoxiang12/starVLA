"""Migrate learned spatial positions when changing a GAWM token grid."""
import logging

import torch
import torch.nn.functional as F


def resize_spatial_checkpoint(state_dict, *, num_views, grid_size):
    """Interpolate spatial positions per view; keep all content weights intact.

    This is an explicit warm-start operation, not a change to saved model
    semantics. Loading a checkpoint at its native grid is an exact no-op.
    """
    result = dict(state_dict)
    row_key = 'visual_token_pooler.row_embedding.weight'
    col_key = 'visual_token_pooler.col_embedding.weight'
    view_key = 'visual_token_pooler.view_embedding.weight'
    for key in (row_key, col_key, view_key):
        if key not in result or result[key].ndim != 2:
            raise ValueError(f'Spatial warm start requires a 2-D {key}')
    source_grid = result[row_key].shape[0]
    if result[col_key].shape[0] != source_grid:
        raise ValueError('Source row and column grids disagree')
    if result[view_key].shape[0] != num_views:
        raise ValueError('Spatial warm start cannot change camera view count/order')
    if source_grid == grid_size:
        return result

    for key in (row_key, col_key):
        value = result[key]
        resized = F.interpolate(value.float().T.unsqueeze(0), size=grid_size,
                                mode='linear', align_corners=False)
        result[key] = resized.squeeze(0).T.to(dtype=value.dtype)

    token_keys = ['world_model.residual_predictor.token_embedding']
    token_keys += [key for key in result
                   if key.startswith('action_models.') and key.endswith('.token_embedding.weight')]
    source_tokens = num_views * source_grid * source_grid
    for key in token_keys:
        if key not in result:
            raise ValueError(f'Missing spatial parameter {key}')
        value = result[key]
        if value.ndim not in (2, 4) or value.shape[-2] != source_tokens:
            raise ValueError(f'{key} shape {tuple(value.shape)} disagrees with source spatial grid')
        if value.ndim == 4 and value.shape[:2] != (1, 1):
            raise ValueError(f'Unexpected leading dimensions for {key}')
        channels = value.shape[-1]
        per_view = value.float().reshape(num_views, source_grid, source_grid, channels).permute(0, 3, 1, 2)
        resized = F.interpolate(per_view, size=(grid_size, grid_size), mode='bilinear', align_corners=False)
        target_shape = (*value.shape[:-2], num_views * grid_size * grid_size, channels)
        result[key] = resized.permute(0, 2, 3, 1).reshape(target_shape).to(dtype=value.dtype)
    logging.getLogger(__name__).info('Resized GAWM spatial positions %sx%s -> %sx%s, preserving %s camera views',
                                    source_grid, source_grid, grid_size, grid_size, num_views)
    return result
