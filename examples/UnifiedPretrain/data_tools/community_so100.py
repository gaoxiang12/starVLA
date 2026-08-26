"""Shared validation helpers for the community SO-family pretraining subset."""

from __future__ import annotations

from pathlib import Path
from typing import Any


SO100_ACTION_NAMES = (
    "main_shoulder_pan",
    "main_shoulder_lift",
    "main_elbow_flex",
    "main_wrist_flex",
    "main_wrist_roll",
    "main_gripper",
)
SO_FOLLOWER_ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
SO100_EXCLUDED_NAME_TOKENS = ("test", "debug", "trial")
SO100_EXCLUDED_TASK_TOKENS = (
    "dummy task",
    "no action",
    "do nothing",
    "test run",
)

SO_FAMILY_DATA_CONFIGS = {
    "so100": "unified_so100_wm",
    "so100-blue": "unified_so100_wm",
    "so100-red": "unified_so100_wm",
    "so101": "unified_so101_wm",
    "so100_follower": "unified_so_follower_wm",
    "so101_follower": "unified_so_follower_wm",
}


def feature_names(feature: dict[str, Any] | None) -> tuple[str, ...]:
    """Return a flat, stable feature-name tuple from LeRobot metadata."""

    names = (feature or {}).get("names")
    if isinstance(names, list):
        return tuple(str(name) for name in names)
    if isinstance(names, dict):
        flattened = []
        for value in names.values():
            if isinstance(value, list):
                flattened.extend(str(name) for name in value)
        return tuple(flattened)
    return ()


def select_so100_video_keys(info: dict[str, Any]) -> tuple[str, ...] | None:
    """Choose one or two usable views with a stable exterior-first ordering."""

    candidates = []
    for key, feature in info.get("features", {}).items():
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            continue
        curation = (feature.get("info") or {}).get("curation")
        if isinstance(curation, dict) and curation.get("usable") is False:
            continue
        lower = str(key).lower()
        is_wrist = "wrist" in lower or (
            isinstance(curation, dict)
            and "wrist" in str(curation.get("view_label", "")).lower()
        )
        candidates.append((is_wrist, str(key)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    primary = candidates[0][1]
    wrist_candidates = [key for is_wrist, key in candidates if is_wrist]
    secondary = wrist_candidates[0] if wrist_candidates else None
    if secondary is None or secondary == primary:
        secondary = next((key for _, key in candidates if key != primary), None)
    return (primary, secondary) if secondary is not None else (primary,)


def classify_so_family_metadata(info: dict[str, Any]) -> str | None:
    """Return the StarVLA data-config key for a compatible SO-family schema."""

    robot_type = str(info.get("robot_type", ""))
    data_config = SO_FAMILY_DATA_CONFIGS.get(robot_type)
    if data_config is None:
        return None
    features = info.get("features", {})
    action = features.get("action")
    state = features.get("observation.state")
    expected_names = (
        SO_FOLLOWER_ACTION_NAMES
        if data_config == "unified_so_follower_wm"
        else SO100_ACTION_NAMES
    )
    if feature_names(action) != expected_names or feature_names(state) != expected_names:
        return None
    return data_config


def validate_so_family_dataset_metadata(
    info: dict[str, Any],
    tasks: list[str] | tuple[str, ...],
    *,
    expected_data_config: str | None = None,
) -> str | None:
    """Return an exclusion reason, or ``None`` for an SO-family candidate."""

    if info.get("codebase_version") != "v3.0":
        return "not_lerobot_v3"
    if int(info.get("fps", 0)) != 30:
        return "not_30hz"

    features = info.get("features", {})
    action = features.get("action")
    state = features.get("observation.state")
    if not isinstance(action, dict) or not isinstance(state, dict):
        return "missing_action_or_state"
    if tuple(action.get("shape", ())) != (6,) or tuple(state.get("shape", ())) != (6,):
        return "not_6d"

    data_config = SO_FAMILY_DATA_CONFIGS.get(str(info.get("robot_type", "")))
    if data_config is None:
        return "incompatible_so_family_schema"
    expected_names = (
        SO_FOLLOWER_ACTION_NAMES
        if data_config == "unified_so_follower_wm"
        else SO100_ACTION_NAMES
    )
    if feature_names(action) != expected_names:
        return "noncanonical_action_order"
    if feature_names(state) != expected_names:
        return "noncanonical_state_order"
    if expected_data_config is not None and data_config != expected_data_config:
        return "wrong_so_family_data_config"
    if select_so100_video_keys(info) is None:
        return "no_usable_views"

    normalized_tasks = [str(task).strip() for task in tasks]
    if not normalized_tasks or any(not task for task in normalized_tasks):
        return "missing_task_language"
    if any(
        token in task.lower()
        for task in normalized_tasks
        for token in SO100_EXCLUDED_TASK_TOKENS
    ):
        return "low_quality_task_language"
    return None


def validate_so100_dataset_metadata(
    info: dict[str, Any],
    dataset_name: str | Path,
    tasks: list[str] | tuple[str, ...],
) -> str | None:
    """Validate the original conservative, exact-SO100 two-view subset."""

    reason = validate_so_family_dataset_metadata(
        info, tasks, expected_data_config="unified_so100_wm"
    )
    if reason is not None:
        return reason
    if info.get("robot_type") != "so100":
        return "not_so100"
    video_keys = select_so100_video_keys(info)
    if video_keys is None or len(video_keys) < 2:
        return "fewer_than_two_usable_views"
    lower_name = str(dataset_name).lower()
    if any(token in lower_name for token in SO100_EXCLUDED_NAME_TOKENS):
        return "suspicious_dataset_name"
    return None
