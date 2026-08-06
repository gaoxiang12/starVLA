"""Stable task-language handling shared by training and deployment."""

from __future__ import annotations

from pathlib import PurePath
from typing import Optional


TASK_LANGUAGE_METADATA = "metadata"
TASK_LANGUAGE_DATASET_NAME = "dataset_name"
TASK_LANGUAGE_MODES = {
    TASK_LANGUAGE_METADATA,
    TASK_LANGUAGE_DATASET_NAME,
}


def normalize_task_language_mode(mode: Optional[str]) -> str:
    """Validate a task-language mode while preserving the legacy default."""
    normalized = str(mode or TASK_LANGUAGE_METADATA).strip().lower()
    if normalized not in TASK_LANGUAGE_MODES:
        raise ValueError(
            f"Unsupported task_language_mode={mode!r}; "
            f"expected one of {sorted(TASK_LANGUAGE_MODES)}"
        )
    return normalized


def canonical_task_text(task_name: str) -> str:
    """Turn a dataset/environment task name into one stable readable string."""
    leaf_name = PurePath(str(task_name).strip()).name
    words = " ".join(leaf_name.replace("-", "_").split("_"))
    return " ".join(words.lower().split())


def resolve_task_language(
    original_text: Optional[str],
    task_name: str,
    mode: Optional[str],
) -> str:
    """Resolve the language presented to the policy for one task instance."""
    normalized_mode = normalize_task_language_mode(mode)
    if normalized_mode == TASK_LANGUAGE_DATASET_NAME:
        text = canonical_task_text(task_name)
        if not text:
            raise ValueError("task_name must be non-empty in dataset_name mode")
        return text
    return str(original_text or "")
