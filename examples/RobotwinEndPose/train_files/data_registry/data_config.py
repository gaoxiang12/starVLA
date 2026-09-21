"""Feedback-only ablation on unchanged RoboTwin continuous joint actions."""
from examples.UnifiedPretrain.train_files.data_registry.data_config import RoboTwinContinuousNextWMDataConfig
from starVLA.robotwin_feedback import feedback_keys


class RoboTwinFeedbackDataConfig(RoboTwinContinuousNextWMDataConfig):
    def __init__(self, variant):
        self.state_keys = feedback_keys(variant)
        self.state_key_dims = dict(zip(self.state_keys, (7, 1, 7, 1)))
        self.state_spec_id = ('aloha_native_world_ee_xyz_wxyz_contgrip_16' if variant == 'endpose'
                              else 'aloha_joint_command_padded_contgrip_16')


ROBOT_TYPE_CONFIG_MAP = {f'robotwin_feedback_{v}': RoboTwinFeedbackDataConfig(v)
                        for v in ('command', 'endpose')}
DATASET_NAMED_MIXTURES = {
    f'robotwin_ranking_rgb_feedback_{v}': [
        ('RoboTwinEndPose/Clean/blocks_ranking_rgb', 1., f'robotwin_feedback_{v}')]
    for v in ('command', 'endpose')}
