"""LIBERO benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys), # igore state modality for now since some datasets don't have state and we want to be able to use them, can add back later if needed
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


class Libero4in1WMDataConfig(Libero4in1DataConfig):
    """LIBERO config that also loads future frames for latent world-model training.

    The video modality loads frames at ``video_indices`` (current + future). The
    dataset's ``_pack_sample`` exposes frames [1:] as ``future_images`` when
    ``future_obs_frames`` is enabled in the run's data config. With
    ``action_horizon = 8`` and ``n_future = 2`` the natural cadence is every 4
    env-steps -> ``[0, 4, 8]`` (current latent + 2 future latents).
    """

    video_indices = [0, 4, 8]
    # Sample proprio state at the SAME cadence as video so state[t] aligns with
    # the current + future latent frames (current, +4, +8). This lets the
    # LeWMOFT state probe supervise predicted future latents against the
    # *future* physical state.
    state_indices = [0, 4, 8]

    def modality_config(self):
        cfg = super().modality_config()
        cfg["video"] = ModalityConfig(delta_indices=self.video_indices, modality_keys=self.video_keys)
        cfg["state"] = ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys)
        return cfg

    def transform(self):
        # Base transform normalizes actions; additionally standardize proprio
        # state (mean_std) so the state-probe MSE target is well-scaled across
        # dims (LIBERO proprio std ranges ~0.015 to ~0.72).
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={k: "mean_std" for k in self.state_keys},
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "libero_franka_wm": Libero4in1WMDataConfig(),
}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_all_wm": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
    ],
    "libero_goal_wm": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
    ],
    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
}
