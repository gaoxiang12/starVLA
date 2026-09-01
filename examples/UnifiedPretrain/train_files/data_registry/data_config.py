"""Unified LIBERO + RoboTwin + Bridge + DROID + KUKA + SO-family registry."""

import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from examples.UnifiedPretrain.data_tools.community_so100 import (
    select_so100_video_keys,
    validate_so_family_dataset_metadata,
)

from examples.LIBERO.train_files.data_registry.data_config import (
    DATASET_NAMED_MIXTURES as LIBERO_MIXTURES,
    Libero4in1WMDataConfig,
)
from examples.Robotwin.train_files.data_registry.data_config import (
    DATASET_NAMED_MIXTURES as ROBOTWIN_MIXTURES,
    AgilexWMDataConfig,
)
from starVLA.dataloader.gr00t_lerobot.data_config import (
    OxeBridgeDataConfig,
    OxeDroidDataConfig,
)
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotSingleDataset,
    ModalityConfig,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    LeRobotModalityMetadata,
)
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


class UnifiedKukaWMDataConfig:
    """KUKA iiwa: EEF delta XYZ/RPY plus an absolute open-gripper bit."""

    embodiment_tag = EmbodimentTag.KUKA
    episode_blacklist_path = "meta/pretrain_audit/excluded_episodes.jsonl"
    action_spec_id = "kuka_eef_delta_rpy_gripper_open_abs_7"
    state_spec_id = "kuka_eef_xyz_quaternion_xyzw_gripper_closed_8"
    control_hz = 10
    future_time_offsets_s = (0.0, 0.2, 0.4)
    video_keys = ["video.image"]
    state_keys = [
        "state.eef_position",
        "state.eef_quaternion_xyzw",
        "state.gripper_closed",
    ]
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta_rpy",
        "action.gripper_open",
    ]
    state_key_dims = {
        "state.eef_position": 3,
        "state.eef_quaternion_xyzw": 4,
        "state.gripper_closed": 1,
    }
    action_key_dims = {
        "action.eef_position_delta": 3,
        "action.eef_rotation_delta_rpy": 3,
        "action.gripper_open": 1,
    }
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    video_indices = [0, 2, 4]
    state_indices = [0]
    action_indices = list(range(8))
    gripper_action_keys = ("action.gripper_open",)
    gripper_state_keys = ("state.gripper_closed",)
    action_absolute_overrides = {
        "action.eef_position_delta": False,
        "action.eef_rotation_delta_rpy": False,
        "action.gripper_open": True,
    }

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.video_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return _unified_state_action_transform(self, state_mode="q99")

    def make_dataset(self, **kwargs):
        kwargs.pop("dataset_name", None)
        return _KukaSingleDataset(**kwargs)


class _LazyStepIndex(Sequence[tuple[int, int]]):
    """Compact flat-step index for very large LeRobot datasets."""

    def __init__(self, trajectory_ids, trajectory_lengths):
        self.trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
        lengths = np.asarray(trajectory_lengths, dtype=np.int64)
        self.ends = np.cumsum(lengths, dtype=np.int64)

    def __len__(self):
        return int(self.ends[-1]) if self.ends.size else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        trajectory_position = int(np.searchsorted(self.ends, index, side="right"))
        start = 0 if trajectory_position == 0 else int(self.ends[trajectory_position - 1])
        return int(self.trajectory_ids[trajectory_position]), index - start


class _DroidSingleDataset(LeRobotSingleDataset):
    def _get_all_steps(self):
        # The configured taxonomy blacklist already removes empty, ambiguous,
        # and rare task classes. Avoid materializing one Python tuple per frame.
        return _LazyStepIndex(self.trajectory_ids, self.trajectory_lengths)


class _KukaSingleDataset(LeRobotSingleDataset):
    def _get_all_steps(self):
        return _LazyStepIndex(self.trajectory_ids, self.trajectory_lengths)


class _CommunityEefSingleDataset(LeRobotSingleDataset):
    """Memory-bounded reader for the large per-episode Open-X exports."""

    def _get_all_steps(self):
        return _LazyStepIndex(self.trajectory_ids, self.trajectory_lengths)


class UnifiedDroidWMDataConfig(OxeDroidDataConfig):
    episode_blacklist_path = (
        "meta/task_language/droid_pretrain_excluded_episodes.jsonl"
    )
    action_spec_id = "droid_eef_delta_7"
    state_spec_id = "droid_eef_state_10"
    control_hz = 15
    video_indices = [0, 3, 6]
    observation_indices = [0]
    action_indices = list(range(16))
    future_time_offsets_s = (0.0, 0.2, 0.4)
    gripper_action_keys = ("action.gripper_position",)
    gripper_state_keys = ("state.gripper_position",)
    action_absolute_overrides = {
        "action.eef_position_delta": False,
        "action.eef_rotation_delta": False,
        "action.gripper_position": True,
    }

    def modality_config(self):
        config = super().modality_config()
        config["video"] = ModalityConfig(
            delta_indices=self.video_indices,
            modality_keys=self.video_keys,
        )
        return config

    def transform(self):
        action_modes = {key: "q99" for key in self.action_keys}
        action_modes["action.gripper_position"] = "binary"
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_modes,
                    target_rotations={
                        "action.eef_rotation_delta": "axis_angle"
                    },
                    q99_clip=1.0,
                ),
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={
                        "state.eef_position": "q99",
                        "state.gripper_position": "binary",
                    },
                    target_rotations={"state.eef_rotation": "rotation_6d"},
                    q99_clip=1.0,
                ),
            ]
        )

    def make_dataset(self, **kwargs):
        kwargs.pop("dataset_name", None)
        return _DroidSingleDataset(**kwargs)


class _UnifiedCommunityEefWMDataConfig:
    """Shared loader plumbing; subclasses retain distinct action semantics."""

    episode_blacklist_path = "meta/pretrain_audit/excluded_episodes.jsonl"
    future_time_offsets_s = (0.0, 0.2, 0.4)
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(8))
    gripper_action_keys = ("action.gripper_open",)
    gripper_state_keys = ()

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.video_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return _unified_state_action_transform(self, state_mode="q99")

    def make_dataset(self, **kwargs):
        kwargs.pop("dataset_name", None)
        return _CommunityEefSingleDataset(**kwargs)


class UnifiedTacoPlayWMDataConfig(_UnifiedCommunityEefWMDataConfig):
    embodiment_tag = EmbodimentTag.TACO_FRANKA
    action_spec_id = (
        "taco_franka_world_delta_scaled_xyz50_rpy20_gripper_open_abs_7_15hz"
    )
    state_spec_id = "taco_franka_eef_xyz_rpy_pad_gripper_state_8"
    control_hz = 15
    video_keys = ["video.rgb_static", "video.rgb_gripper"]
    video_indices = [0, 3, 6]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation_rpy",
        "state.pad",
        "state.gripper_position",
    ]
    state_key_dims = {
        "state.eef_position": 3,
        "state.eef_rotation_rpy": 3,
        "state.pad": 1,
        "state.gripper_position": 1,
    }
    action_keys = [
        "action.eef_position_delta_scaled",
        "action.eef_rotation_delta_rpy_scaled",
        "action.gripper_open",
    ]
    action_key_dims = {
        "action.eef_position_delta_scaled": 3,
        "action.eef_rotation_delta_rpy_scaled": 3,
        "action.gripper_open": 1,
    }
    action_absolute_overrides = {
        "action.eef_position_delta_scaled": False,
        "action.eef_rotation_delta_rpy_scaled": False,
        "action.gripper_open": True,
    }


class UnifiedBcZWMDataConfig(_UnifiedCommunityEefWMDataConfig):
    embodiment_tag = EmbodimentTag.GOOGLE_BCZ
    action_spec_id = (
        "google_bcz_eef_delta_xyz_axis_angle_gripper_open_abs_7_10hz"
    )
    state_spec_id = "google_bcz_eef_xyz_rpy_pad_gripper_position_8"
    control_hz = 10
    video_keys = ["video.image"]
    video_indices = [0, 2, 4]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation_rpy",
        "state.pad",
        "state.gripper_position",
    ]
    state_key_dims = {
        "state.eef_position": 3,
        "state.eef_rotation_rpy": 3,
        "state.pad": 1,
        "state.gripper_position": 1,
    }
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta_axis_angle",
        "action.gripper_open",
    ]
    action_key_dims = {
        "action.eef_position_delta": 3,
        "action.eef_rotation_delta_axis_angle": 3,
        "action.gripper_open": 1,
    }
    action_absolute_overrides = {
        "action.eef_position_delta": False,
        "action.eef_rotation_delta_axis_angle": False,
        "action.gripper_open": True,
    }


class UnifiedFractalWMDataConfig(_UnifiedCommunityEefWMDataConfig):
    embodiment_tag = EmbodimentTag.GOOGLE_RT1
    action_spec_id = "google_rt1_eef_delta_xyz_rpy_gripper_open_abs_7_3hz"
    state_spec_id = "google_rt1_eef_xyz_quaternion_xyzw_gripper_closed_8"
    control_hz = 3
    video_keys = ["video.image"]
    # At 3 Hz, these are the nearest frames to +0.2 s and +0.4 s.
    video_indices = [0, 1, 1]
    state_keys = [
        "state.eef_position",
        "state.eef_quaternion_xyzw",
        "state.gripper_closed",
    ]
    state_key_dims = {
        "state.eef_position": 3,
        "state.eef_quaternion_xyzw": 4,
        "state.gripper_closed": 1,
    }
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta_rpy",
        "action.gripper_open",
    ]
    action_key_dims = {
        "action.eef_position_delta": 3,
        "action.eef_rotation_delta_rpy": 3,
        "action.gripper_open": 1,
    }
    gripper_state_keys = ("state.gripper_closed",)
    action_absolute_overrides = {
        "action.eef_position_delta": False,
        "action.eef_rotation_delta_rpy": False,
        "action.gripper_open": True,
    }


class UnifiedFmbWMDataConfig(_UnifiedCommunityEefWMDataConfig):
    embodiment_tag = EmbodimentTag.FMB_FRANKA
    action_spec_id = "fmb_franka_eef_twist_normalized_gripper_open_abs_7_10hz"
    state_spec_id = "fmb_franka_eef_xyz_quaternion_xyzw_gripper_position_8"
    control_hz = 10
    video_keys = [
        "video.image_side_1",
        "video.image_side_2",
        "video.image_wrist_1",
    ]
    video_indices = [0, 2, 4]
    state_keys = [
        "state.eef_position",
        "state.eef_quaternion_xyzw",
        "state.gripper_position",
    ]
    state_key_dims = {
        "state.eef_position": 3,
        "state.eef_quaternion_xyzw": 4,
        "state.gripper_position": 1,
    }
    action_keys = [
        "action.eef_linear_twist_normalized",
        "action.eef_angular_twist_normalized",
        "action.gripper_open",
    ]
    action_key_dims = {
        "action.eef_linear_twist_normalized": 3,
        "action.eef_angular_twist_normalized": 3,
        "action.gripper_open": 1,
    }
    action_absolute_overrides = {
        "action.eef_linear_twist_normalized": False,
        "action.eef_angular_twist_normalized": False,
        "action.gripper_open": True,
    }


@lru_cache(maxsize=8)
def _manifest_payload(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _community_manifest_entry(data_cfg, dataset_path: Path) -> dict | None:
    if data_cfg is None:
        return None
    data_root = Path(str(data_cfg.get("data_root_dir", "."))).resolve()
    for configured_path in data_cfg.get("dataset_manifests", {}).values():
        manifest_path = Path(str(configured_path))
        if not manifest_path.is_absolute():
            manifest_path = data_root / manifest_path
        payload = _manifest_payload(str(manifest_path.resolve()))
        source_root = (data_root / str(payload.get("source_root", ""))).resolve()
        try:
            relative_path = dataset_path.resolve().relative_to(source_root).as_posix()
        except ValueError:
            continue
        for entry in payload.get("datasets", []):
            if entry.get("path") == relative_path:
                return entry
    return None


class _CommunitySoFamilySingleDataset(LeRobotSingleDataset):
    """Read curated SO-family LeRobot v3 roots without mutating raw metadata."""

    def __init__(self, *args, **kwargs):
        expected_data_config = kwargs.pop("expected_data_config")
        dataset_path = kwargs.get("dataset_path")
        if dataset_path is None and args:
            dataset_path = args[0]
        dataset_path = Path(dataset_path)
        self._community_info = json.loads(
            (dataset_path / "meta/info.json").read_text()
        )
        task_table = pd.read_parquet(dataset_path / "meta/tasks.parquet")
        tasks = [str(task) for task in task_table.reset_index()["task"]]
        reason = validate_so_family_dataset_metadata(
            self._community_info,
            tasks,
            expected_data_config=expected_data_config,
        )
        if reason is not None:
            raise ValueError(f"SO-family manifest contains invalid dataset: {reason}")
        video_keys = select_so100_video_keys(self._community_info)
        assert video_keys is not None
        self._community_video_keys = video_keys

        manifest_entry = _community_manifest_entry(
            kwargs.get("data_cfg"), dataset_path
        )
        if manifest_entry is not None:
            manifest_config = manifest_entry.get("data_config", expected_data_config)
            if manifest_config != expected_data_config:
                raise ValueError(
                    f"manifest routes {dataset_path} to {manifest_config!r}, "
                    f"but loader uses {expected_data_config!r}"
                )
            kwargs["episode_blacklist"] = manifest_entry.get(
                "excluded_episode_indices", []
            )

        if len(video_keys) == 1:
            modality_configs = dict(kwargs["modality_configs"])
            modality_configs["video"] = modality_configs["video"].model_copy(
                update={"modality_keys": ["video.primary_image"]}
            )
            kwargs["modality_configs"] = modality_configs
        super().__init__(*args, **kwargs)

    def _get_all_steps(self):
        return _LazyStepIndex(self.trajectory_ids, self.trajectory_lengths)

    def _get_lerobot_modality_meta(self):
        video = {
            "primary_image": {"original_key": self._community_video_keys[0]},
        }
        if len(self._community_video_keys) > 1:
            video["secondary_image"] = {
                "original_key": self._community_video_keys[1]
            }
        return LeRobotModalityMetadata.model_validate(
            {
                "state": {
                    "joints": {
                        "start": 0,
                        "end": 6,
                        "absolute": True,
                        "dtype": "float32",
                        "original_key": "observation.state",
                    }
                },
                "action": {
                    "joints": {
                        "start": 0,
                        "end": 6,
                        "absolute": True,
                        "dtype": "float32",
                        "original_key": "action",
                    }
                },
                "video": video,
                "annotation": {
                    "human.action.task_description": {
                        "original_key": "task_index"
                    }
                },
            }
        )

    @staticmethod
    def _statistics(raw):
        return {
            "min": raw["min"],
            "max": raw["max"],
            "mean": raw["mean"],
            "std": raw["std"],
            # LeRobot v3 community stats omit quantiles. They are retained as
            # conservative bounds for schema compatibility; this config uses
            # min/max normalization, so these values are not consumed.
            "q01": raw["min"],
            "q99": raw["max"],
        }

    def _get_metadata(self, embodiment_tag):
        stats = json.loads((self.dataset_path / "meta/stats.json").read_text())
        state_stats = stats.get("observation.state")
        action_stats = stats.get("action")
        if state_stats is None or action_stats is None:
            raise ValueError(f"SO-family statistics missing in {self.dataset_path}")
        video = {
            "primary_image": {
                "resolution": [224, 224],
                "channels": 3,
                "fps": 30,
            },
            # Mixture metadata describes the canonical schema shared by all
            # children. A one-view child omits this key from its runtime
            # modality config; _pack_sample pads it and marks it invalid.
            "secondary_image": {
                "resolution": [224, 224],
                "channels": 3,
                "fps": 30,
            },
        }
        return DatasetMetadata.model_validate(
            {
                "statistics": {
                    "state": {"joints": self._statistics(state_stats)},
                    "action": {"joints": self._statistics(action_stats)},
                },
                "modalities": {
                    # Samples are resized to 224x224 by _pack_sample. Canonical
                    # metadata keeps all selected community roots mergeable.
                    "video": video,
                    "state": {
                        "joints": {
                            "absolute": True,
                            "shape": [6],
                            "continuous": True,
                        }
                    },
                    "action": {
                        "joints": {
                            "absolute": True,
                            "shape": [6],
                            "continuous": True,
                        }
                    },
                },
                "embodiment_tag": embodiment_tag,
            }
        )


class _UnifiedSoFamilyWMDataConfig:
    lerobot_version = "v3.0"
    control_hz = 30
    future_time_offsets_s = (0.0, 0.2, 0.4)
    video_keys = ["video.primary_image", "video.secondary_image"]
    state_keys = ["state.joints"]
    action_keys = ["action.joints"]
    action_key_dims = {"action.joints": 6}
    state_key_dims = {"state.joints": 6}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    video_indices = [0, 6, 12]
    state_indices = [0]
    action_indices = list(range(16))
    action_absolute_overrides = {"action.joints": True}

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.video_indices, modality_keys=self.video_keys
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices, modality_keys=self.state_keys
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices, modality_keys=self.action_keys
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={"action.joints": "min_max"},
                ),
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={"state.joints": "min_max"},
                ),
            ]
        )

    def make_dataset(self, **kwargs):
        kwargs.pop("dataset_name", None)
        return _CommunitySoFamilySingleDataset(
            expected_data_config=self.data_config_key, **kwargs
        )


class UnifiedSo100WMDataConfig(_UnifiedSoFamilyWMDataConfig):
    data_config_key = "unified_so100_wm"
    embodiment_tag = EmbodimentTag.SO100
    action_spec_id = "so100_joint_abs_6"
    state_spec_id = "so100_joint_state_6"


class UnifiedSo101WMDataConfig(_UnifiedSoFamilyWMDataConfig):
    data_config_key = "unified_so101_wm"
    embodiment_tag = EmbodimentTag.SO101
    action_spec_id = "so101_joint_abs_6"
    state_spec_id = "so101_joint_state_6"


class UnifiedSoFollowerWMDataConfig(_UnifiedSoFamilyWMDataConfig):
    data_config_key = "unified_so_follower_wm"
    embodiment_tag = EmbodimentTag.SO_FOLLOWER
    action_spec_id = "so_follower_joint_abs_6"
    state_spec_id = "so_follower_joint_state_6"


ROBOT_TYPE_CONFIG_MAP = {
    "unified_libero_wm": UnifiedLiberoWMDataConfig(),
    "unified_robotwin_wm": UnifiedRoboTwinWMDataConfig(),
    "unified_bridge_wm": UnifiedBridgeWMDataConfig(),
    "unified_kuka_wm": UnifiedKukaWMDataConfig(),
    "unified_droid_wm": UnifiedDroidWMDataConfig(),
    "unified_taco_play_wm": UnifiedTacoPlayWMDataConfig(),
    "unified_bc_z_wm": UnifiedBcZWMDataConfig(),
    "unified_fractal_wm": UnifiedFractalWMDataConfig(),
    "unified_fmb_wm": UnifiedFmbWMDataConfig(),
    "unified_so100_wm": UnifiedSo100WMDataConfig(),
    "unified_so101_wm": UnifiedSo101WMDataConfig(),
    "unified_so_follower_wm": UnifiedSoFollowerWMDataConfig(),
}


def _under(prefix, mixture, robot_type):
    return [(f"{prefix}/{name}", weight, robot_type) for name, weight, _ in mixture]


DATASET_NAMED_MIXTURES = {
    "unified_robotwin_generated_clean500_wm": _under(
        "RoboTwinGenerated",
        ROBOTWIN_MIXTURES["robotwin_generated_clean500_wm"],
        "unified_robotwin_wm",
    ),
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
    ),
    "unified_droid_wm": [
        ("droid_lerobot", 1.0, "unified_droid_wm"),
    ],
    "unified_kuka_wm": [
        ("kuka_lerobot", 1.0, "unified_kuka_wm"),
    ],
    "unified_taco_play_wm": [
        ("taco_play_lerobot", 1.0, "unified_taco_play_wm"),
    ],
    "unified_bc_z_wm": [
        ("bc_z_lerobot", 1.0, "unified_bc_z_wm"),
    ],
    "unified_fractal_wm": [
        ("fractal20220817_data_lerobot", 1.0, "unified_fractal_wm"),
    ],
    "unified_fmb_wm": [
        ("fmb_dataset_lerobot", 1.0, "unified_fmb_wm"),
    ],
    "unified_so100_wm": [
        ("@manifest:community_so100", 1.0, "unified_so100_wm"),
    ],
    "unified_so_family_wm": [
        ("@manifest:community_so_family", 1.0, "unified_so100_wm"),
    ],
}

DATASET_NAMED_MIXTURES["unified_community_oxe_candidate_wm"] = (
    DATASET_NAMED_MIXTURES["unified_taco_play_wm"]
    + DATASET_NAMED_MIXTURES["unified_bc_z_wm"]
    + DATASET_NAMED_MIXTURES["unified_fractal_wm"]
    + DATASET_NAMED_MIXTURES["unified_fmb_wm"]
)

DATASET_NAMED_MIXTURES["unified_libero_robotwin_bridge_droid_wm"] = (
    DATASET_NAMED_MIXTURES["unified_libero_robotwin_bridge_wm"]
    + DATASET_NAMED_MIXTURES["unified_droid_wm"]
)

DATASET_NAMED_MIXTURES["unified_libero_robotwin_bridge_droid_so100_wm"] = (
    DATASET_NAMED_MIXTURES["unified_libero_robotwin_bridge_droid_wm"]
    + DATASET_NAMED_MIXTURES["unified_so100_wm"]
)

DATASET_NAMED_MIXTURES["unified_libero_robotwin_bridge_droid_so_family_wm"] = (
    DATASET_NAMED_MIXTURES["unified_libero_robotwin_bridge_droid_wm"]
    + DATASET_NAMED_MIXTURES["unified_so_family_wm"]
)

DATASET_NAMED_MIXTURES[
    "unified_libero_robotwin_bridge_droid_kuka_so_family_wm"
] = (
    DATASET_NAMED_MIXTURES["unified_libero_robotwin_bridge_droid_wm"]
    + DATASET_NAMED_MIXTURES["unified_kuka_wm"]
    + DATASET_NAMED_MIXTURES["unified_so_family_wm"]
)
