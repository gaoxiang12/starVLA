"""Unified LIBERO + RoboTwin + Bridge pretraining data registry."""

from examples.LIBERO.train_files.data_registry.data_config import (
    DATASET_NAMED_MIXTURES as LIBERO_MIXTURES,
    Libero4in1WMDataConfig,
)
from examples.Robotwin.train_files.data_registry.data_config import (
    DATASET_NAMED_MIXTURES as ROBOTWIN_MIXTURES,
    AgilexWMDataConfig,
)
from starVLA.dataloader.gr00t_lerobot.data_config import OxeBridgeDataConfig
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import (
    ComposedModalityTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


def _unified_state_action_transform(
    config,
    *,
    state_mode: str,
) -> ComposedModalityTransform:
    """Use one robust action normalization contract across embodiments."""

    action_modes = {key: "q99" for key in config.action_keys}
    for key in config.gripper_action_keys:
        action_modes[key] = "binary"
    state_modes = {key: state_mode for key in config.state_keys}
    for key in config.gripper_state_keys:
        state_modes[key] = "binary"
    return ComposedModalityTransform(
        transforms=[
            StateActionToTensor(apply_to=config.action_keys),
            StateActionTransform(
                apply_to=config.action_keys,
                normalization_modes=action_modes,
                q99_clip=1.0,
            ),
            StateActionToTensor(apply_to=config.state_keys),
            StateActionTransform(
                apply_to=config.state_keys,
                normalization_modes=state_modes,
            ),
        ]
    )


class UnifiedLiberoWMDataConfig(Libero4in1WMDataConfig):
    action_spec_id = "franka_eef_delta_7"
    state_spec_id = "franka_state_8"
    control_hz = 20
    future_time_offsets_s = (0.0, 0.2, 0.4)
    gripper_action_keys = ("action.gripper",)
    gripper_state_keys = ()
    action_absolute_overrides = {
        key: key == "action.gripper" for key in Libero4in1WMDataConfig.action_keys
    }

    def transform(self):
        return _unified_state_action_transform(self, state_mode="mean_std")


class UnifiedRoboTwinWMDataConfig(AgilexWMDataConfig):
    embodiment_tag = EmbodimentTag.ALOHA
    action_spec_id = "aloha_dual_joint_14"
    state_spec_id = "aloha_joint_state_14"
    control_hz = 30
    # Match the shared world's +0.2 s / +0.4 s prediction targets.
    video_indices = [0, 6, 12]
    future_time_offsets_s = (0.0, 0.2, 0.4)
    gripper_action_keys = ("action.left_gripper", "action.right_gripper")
    gripper_state_keys = ("state.left_gripper", "state.right_gripper")
    action_absolute_overrides = {key: True for key in AgilexWMDataConfig.action_keys}

    def transform(self):
        return _unified_state_action_transform(self, state_mode="q99")


class UnifiedBridgeWMDataConfig(OxeBridgeDataConfig):
    episode_blacklist_path = (
        "meta/task_language/bridge_pretrain_excluded_episodes.jsonl"
    )
    action_spec_id = "bridge_eef_delta_7"
    state_spec_id = "bridge_state_8"
    control_hz = 5
    video_keys = ["video.image_0", "video.image_1", "video.image_2"]
    video_indices = [0, 1, 2]
    action_indices = [0, 1, 2]
    future_time_offsets_s = (0.0, 0.2, 0.4)
    gripper_action_keys = ("action.gripper",)
    gripper_state_keys = ("state.gripper",)
    action_absolute_overrides = {
        key: key == "action.gripper" for key in OxeBridgeDataConfig.action_keys
    }

    def modality_config(self):
        config = super().modality_config()
        config["video"] = ModalityConfig(
            delta_indices=self.video_indices,
            modality_keys=self.video_keys,
        )
        config["action"] = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        return config

    def transform(self):
        return _unified_state_action_transform(self, state_mode="q99")


ROBOT_TYPE_CONFIG_MAP = {
    "unified_libero_wm": UnifiedLiberoWMDataConfig(),
    "unified_robotwin_wm": UnifiedRoboTwinWMDataConfig(),
    "unified_bridge_wm": UnifiedBridgeWMDataConfig(),
}


def _under(prefix, mixture, robot_type):
    return [(f"{prefix}/{name}", weight, robot_type) for name, weight, _ in mixture]


DATASET_NAMED_MIXTURES = {
    "unified_libero_robotwin_bridge_wm": (
        _under(
            "libero",
            LIBERO_MIXTURES["libero_all_wm_l10_augmented_l90"],
            "unified_libero_wm",
        )
        + _under(
            "RoboTwin",
            ROBOTWIN_MIXTURES["robotwin_all_wm"],
            "unified_robotwin_wm",
        )
        + [
            (
                "datasets/IPEC-COMMUNITY/bridge_orig_lerobot_git",
                1.0,
                "unified_bridge_wm",
            )
        ]
    )
}
