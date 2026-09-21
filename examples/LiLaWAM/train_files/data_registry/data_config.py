"""LiLa recorded-frame chunks over existing shared LIBERO/RoboTwin datasets."""
from examples.UnifiedPretrain.train_files.data_registry.data_config import (
    UnifiedLiberoWMDataConfig, RoboTwinContinuousNextWMDataConfig,
    DATASET_NAMED_MIXTURES as UNIFIED,
)
from examples.LIBERO.train_files.data_registry.data_config import DATASET_NAMED_MIXTURES as LIBERO
from examples.RobotwinEndPose.train_files.data_registry.data_config import RoboTwinFeedbackDataConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from examples.Robotwin.eval_files.summarize_robotwin_eval import ALL_TASKS


class LiLaLiberoDataConfig(UnifiedLiberoWMDataConfig):
    action_indices = list(range(32))
    state_indices = [0]
    video_indices = [0, 32]
    # No-op removal changes recorded-frame cadence; do not infer physical time
    # from the encoded video's FPS. Offsets below are explicitly recorded steps.
    control_hz = None
    future_time_offsets_s = None


class LiLaRoboTwinDataConfig(RoboTwinContinuousNextWMDataConfig):
    action_indices = list(range(1, 33))
    state_indices = [0]
    video_indices = [0, 32]


class LiLaRoboTwinEndPoseDataConfig(RoboTwinFeedbackDataConfig):
    action_indices = list(range(1, 33))
    state_indices = [0]
    video_indices = [0, 32]


class LiLaRoboTwinAlignedDataConfig(LiLaRoboTwinDataConfig):
    """Local three-camera commands with the official recorded-frame recipe.

    These files contain commanded joints, NOT the author's measured endpose.
    Keep a distinct spec and preserve native [L6,Lgrip,R6,Rgrip] ordering.
    """
    action_spec_id = 'aloha_dual_joint_contgrip_current_recorded_native_14'
    state_spec_id = 'aloha_joint_command_contgrip_native_14'
    action_keys = ['action.left_joints', 'action.left_gripper',
                   'action.right_joints', 'action.right_gripper']
    state_keys = [key.replace('action.', 'state.') for key in action_keys]
    action_indices = list(range(32))
    gripper_indices = (6, 13)

    def transform(self):
        transforms = []
        for keys in (self.action_keys, self.state_keys):
            transforms.extend([StateActionToTensor(apply_to=keys), StateActionTransform(
                apply_to=keys, normalization_modes={key: 'min_max_safe' for key in keys})])
        return ComposedModalityTransform(transforms=transforms)


ROBOT_TYPE_CONFIG_MAP = {
    'lila_libero': LiLaLiberoDataConfig(),
    'lila_robotwin': LiLaRoboTwinDataConfig(),
    'lila_robotwin_endpose': LiLaRoboTwinEndPoseDataConfig('endpose'),
    'lila_robotwin_aligned': LiLaRoboTwinAlignedDataConfig(),
}
DATASET_NAMED_MIXTURES = {
    'lila_libero': [(f'libero/{name}', w, 'lila_libero') for name, w, _ in LIBERO['libero_all']],
    'lila_libero90': [('libero/libero_90_no_noops_lerobot', 1., 'lila_libero')],
    'lila_libero_goal': [('libero/libero_goal_no_noops_1.0.0_lerobot', 1., 'lila_libero')],
    'lila_robotwin': [(name, w, 'lila_robotwin') for name, w, _ in
                      UNIFIED['robotwin_generated_clean1000_continuous_next_wm']],
    'lila_robotwin_ranking': [('RoboTwinGenerated/Clean/blocks_ranking_rgb', 1., 'lila_robotwin')],
    'lila_robotwin_endpose': [('RoboTwinEndPose/Clean/blocks_ranking_rgb', 1., 'lila_robotwin_endpose')],
    'lila_robotwin_aligned': [(f'RoboTwin/{split}/{task}', 1., 'lila_robotwin_aligned')
                             for split in ('Clean', 'Randomized') for task in ALL_TASKS],
    'lila_robotwin_aligned_smoke': [('RoboTwin/Clean/blocks_ranking_rgb', 1., 'lila_robotwin_aligned')],
}
# Cross-benchmark joint world-model recipes require verified physical-time
# alignment; the source no-op-filtered/segment-recorded datasets lack it.
# Keep the per-benchmark recipes explicit instead of silently mixing offsets.
