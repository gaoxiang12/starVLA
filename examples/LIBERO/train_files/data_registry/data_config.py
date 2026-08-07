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


class Libero4in1SmoothLatentWMDataConfig(Libero4in1DataConfig):
    """Consecutive prediction targets plus a ranking-only same-episode frame."""

    video_indices = [0, 1, 2, 8]

    def modality_config(self):
        cfg = super().modality_config()
        cfg["video"] = ModalityConfig(
            delta_indices=self.video_indices, modality_keys=self.video_keys
        )
        return cfg


class Libero4in1SmoothLatentWMRolloutDataConfig(Libero4in1DataConfig):
    """Consecutive targets for a two-step autoregressive rollout plus far frame.

    Step 1 supervises [t,t+1,t+2] exactly as the single-shot config; the extra
    t+3/t+4 frames let the predictor be supervised on its own re-anchored
    output (rollout_steps=2, n_future=2); t+8 stays the same-episode
    temporal-order reference.
    """

    video_indices = [0, 1, 2, 3, 4, 8]

    def modality_config(self):
        cfg = super().modality_config()
        cfg["video"] = ModalityConfig(
            delta_indices=self.video_indices, modality_keys=self.video_keys
        )
        return cfg


class Libero4in1WMContext2Horizon20DataConfig(Libero4in1WMDataConfig):
    """LIBERO WM samples aligned to a 20-step action chunk and two-frame context."""

    observation_indices = [0, 1, 5, 10, 15, 20]
    video_indices = [0, 1, 5, 10, 15, 20]
    action_indices = list(range(1, 21))
    state_indices = [1]


class Libero4in1WMContext2Horizon8DataConfig(Libero4in1WMDataConfig):
    """Two-frame history with the baseline 8-step action and future horizon."""

    video_indices = [-1, 0, 4, 8]
    state_indices = [0]


class Libero4in1WMContext3Horizon8DataConfig(Libero4in1WMDataConfig):
    """Three-frame causal history plus the unchanged +4/+8 latent targets."""

    video_indices = [-8, -4, 0, 4, 8]
    state_indices = [-8, -4, 0]


class Libero4in1WMRolloutHorizon16DataConfig(Libero4in1WMDataConfig):
    """Four future frames at the deployed +4 cadence for multi-step rollout.

    The first two targets are exactly the baseline +4/+8 pair, so a two-step
    rollout run stays directly comparable to every single-shot checkpoint while
    additionally supervising the predictor on its own re-anchored output.
    """

    video_indices = [0, 4, 8, 12, 16]
    state_indices = [0]


class Libero4in1WMRolloutHorizon32DataConfig(Libero4in1WMDataConfig):
    """Eight future frames for a four-step rollout (about 1.6s at 20 Hz)."""

    video_indices = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    state_indices = [0]


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "libero_franka_wm": Libero4in1WMDataConfig(),
    "libero_franka_smooth_latent_wm": Libero4in1SmoothLatentWMDataConfig(),
    "libero_franka_smooth_latent_wm_rollout": (
        Libero4in1SmoothLatentWMRolloutDataConfig()
    ),
    "libero_franka_wm_ctx2_h8": Libero4in1WMContext2Horizon8DataConfig(),
    "libero_franka_wm_ctx3_h8": Libero4in1WMContext3Horizon8DataConfig(),
    "libero_franka_wm_ctx2_h20": Libero4in1WMContext2Horizon20DataConfig(),
    "libero_franka_wm_rollout_h16": Libero4in1WMRolloutHorizon16DataConfig(),
    "libero_franka_wm_rollout_h32": Libero4in1WMRolloutHorizon32DataConfig(),
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
    # The public IPEC LIBERO-90 conversion (LeRobot v2, no-op frames removed).
    # Keep it standalone so experiments can choose its sampling weight
    # explicitly instead of silently changing any existing mixture.
    "libero_90": [
        ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],
    "libero_90_wm": [
        ("libero_90_no_noops_lerobot", 1.0, "libero_franka_wm"),
    ],
    # Extend the current four-suite LIBERO mixture with LIBERO-90 while
    # retaining the recovered/teacher LIBERO-10 trajectories. LIBERO-90 gets
    # the same top-level weight as each original suite; auxiliary LIBERO-10
    # weights continue to preserve approximately equal probability per L10
    # trajectory.
    "libero_all_wm_l10_augmented_l90": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        (
            "libero_10_replayed_missing_seed7_1.0.0_lerobot",
            21.0 / 379.0,
            "libero_franka_wm",
        ),
        (
            "libero_10_qwen_teacher_task8_missing_seed7_1.0.0_lerobot",
            15.0 / 379.0,
            "libero_franka_wm",
        ),
        (
            "libero_10_qwen_teacher_remaining_missing_seed7_1.0.0_lerobot",
            74.0 / 379.0,
            "libero_franka_wm",
        ),
        ("libero_90_no_noops_lerobot", 1.0, "libero_franka_wm"),
    ],
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
    # Official raw LIBERO-10 demos absent from the IPEC conversion, replayed
    # under the current simulator and retained only when the success predicate
    # passes.  The replayed dataset has 21 episodes versus 379 originals, so
    # 21/379 preserves approximately equal probability per LIBERO-10 episode.
    "libero_all_wm_l10_replayed": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        (
            "libero_10_replayed_missing_seed7_1.0.0_lerobot",
            21.0 / 379.0,
            "libero_franka_wm",
        ),
    ],
    # Adds successful Qwen3-OFT rollouts from the still-missing official
    # LIBERO-10 training initial states. Auxiliary weights are proportional to
    # episode counts, preserving approximately equal probability per
    # LIBERO-10 trajectory without manually oversampling it.
    "libero_all_wm_l10_augmented": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
        (
            "libero_10_replayed_missing_seed7_1.0.0_lerobot",
            21.0 / 379.0,
            "libero_franka_wm",
        ),
        (
            "libero_10_qwen_teacher_task8_missing_seed7_1.0.0_lerobot",
            15.0 / 379.0,
            "libero_franka_wm",
        ),
        (
            "libero_10_qwen_teacher_remaining_missing_seed7_1.0.0_lerobot",
            74.0 / 379.0,
            "libero_franka_wm",
        ),
    ],
    "libero_all_wm_ctx2_h20": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm_ctx2_h20"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm_ctx2_h20"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm_ctx2_h20"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm_ctx2_h20"),
    ],
    "libero_all_wm_ctx2_h8": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm_ctx2_h8"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm_ctx2_h8"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm_ctx2_h8"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm_ctx2_h8"),
    ],
    "libero_goal_wm": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wm"),
    ],
    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
}

# Match the augmented LIBERO mixture used by the 220k warm-start checkpoint,
# while swapping only its frame schema to the two-context-frame variant.
DATASET_NAMED_MIXTURES["libero_all_wm_l10_augmented_ctx2_h8"] = [
    (dataset, weight, "libero_franka_wm_ctx2_h8")
    for dataset, weight, _robot_type in DATASET_NAMED_MIXTURES[
        "libero_all_wm_l10_augmented"
    ]
]

# Same trajectories and sampling weights, with a longer causal observation and
# proprio history for the action-free residual-correction experiment.
DATASET_NAMED_MIXTURES["libero_all_wm_l10_augmented_ctx3_h8"] = [
    (dataset, weight, "libero_franka_wm_ctx3_h8")
    for dataset, weight, _robot_type in DATASET_NAMED_MIXTURES[
        "libero_all_wm_l10_augmented"
    ]
]

# Same trajectories again, extended to +12/+16 (and +32) so the world model can
# be supervised on its own re-anchored predictions instead of only on
# teacher-forced single-shot targets.
DATASET_NAMED_MIXTURES["libero_all_wm_l10_augmented_rollout_h16"] = [
    (dataset, weight, "libero_franka_wm_rollout_h16")
    for dataset, weight, _robot_type in DATASET_NAMED_MIXTURES[
        "libero_all_wm_l10_augmented"
    ]
]

DATASET_NAMED_MIXTURES["libero_all_wm_l10_augmented_rollout_h32"] = [
    (dataset, weight, "libero_franka_wm_rollout_h32")
    for dataset, weight, _robot_type in DATASET_NAMED_MIXTURES[
        "libero_all_wm_l10_augmented"
    ]
]

# Same augmented LIBERO distribution with [t,t+1,t+2,t+8] video frames.  The
# first three define direct future prediction and local smoothness; t+8 is only
# a same-episode temporal-order reference.
DATASET_NAMED_MIXTURES["libero_all_smooth_latent_wm_l10_augmented"] = [
    (dataset, weight, "libero_franka_smooth_latent_wm")
    for dataset, weight, _robot_type in DATASET_NAMED_MIXTURES[
        "libero_all_wm_l10_augmented"
    ]
]

# Same distribution with [t,t+1,t+2,t+3,t+4,t+8] for a two-step rollout: step 1
# is teacher-forced on [t,t+1,t+2], step 2 is supervised on the predictor's own
# re-anchored output, and t+8 remains the same-episode temporal-order frame.
DATASET_NAMED_MIXTURES["libero_all_smooth_latent_wm_l10_augmented_rollout"] = [
    (dataset, weight, "libero_franka_smooth_latent_wm_rollout")
    for dataset, weight, _robot_type in DATASET_NAMED_MIXTURES[
        "libero_all_smooth_latent_wm_l10_augmented"
    ]
]
