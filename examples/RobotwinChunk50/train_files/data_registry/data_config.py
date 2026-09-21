"""Separate registry entry for 50 next-recorded-frame Aloha targets.

Reuses the full sorting dataset, action/state semantics and normalization.
No original registry entry or running 16-step dataset is changed.
"""
from examples.UnifiedPretrain.train_files.data_registry.data_config import RoboTwinContinuousNextWMDataConfig


class RoboTwinContinuousNext50DataConfig(RoboTwinContinuousNextWMDataConfig):
    action_indices = list(range(1, 51))


ROBOT_TYPE_CONFIG_MAP = {'robotwin_continuous_next50': RoboTwinContinuousNext50DataConfig()}
DATASET_NAMED_MIXTURES = {'robotwin_ranking_rgb_continuous_next50': [
    ('RoboTwinGenerated/Clean/blocks_ranking_rgb', 1.0, 'robotwin_continuous_next50'),
]}
