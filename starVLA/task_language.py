"""Stable task-language handling shared by training and deployment.

The policy uses language as a task identifier rather than as an open-vocabulary
VLM prompt.  Keep the transformations deterministic and intentionally
conservative: dataset taxonomies win when one exists (RoboTwin), while Bridge
only receives lexical normalization that can be reproduced at serving time.
"""

from __future__ import annotations

from pathlib import PurePath
import re
import unicodedata
from typing import Any, Mapping, Optional


TASK_LANGUAGE_METADATA = "metadata"
TASK_LANGUAGE_DATASET_NAME = "dataset_name"
TASK_LANGUAGE_CANONICAL_METADATA = "canonical_metadata"
TASK_LANGUAGE_BRIDGE_CANONICAL = "bridge_canonical"
TASK_LANGUAGE_MODES = {
    TASK_LANGUAGE_METADATA,
    TASK_LANGUAGE_DATASET_NAME,
    TASK_LANGUAGE_CANONICAL_METADATA,
    TASK_LANGUAGE_BRIDGE_CANONICAL,
}


_BRIDGE_PHRASE_REPLACEMENTS = (
    (r"\bpick\s+(?:it\s+)?up\b", "pick"),
    (r"\bon\s+top\s+of\b", "on"),
    (r"\bonto\b", "on"),
    (r"\binto\b", "in"),
    (r"\binside\s+of\b", "in"),
    (r"\binside\b", "in"),
    (r"\bnext\s+to\b", "beside"),
    (r"\bmiddle\b", "center"),
    (r"\bupper\b", "top"),
    (r"\blower\b", "bottom"),
)

# Only the leading task verb is rewritten.  Applying these substitutions to
# every token would corrupt object names (for example, "can" or "stand").
_BRIDGE_ACTION_ALIASES = {
    "move": "move",
    "moved": "move",
    "moves": "move",
    "moving": "move",
    "moove": "move",
    "place": "move",
    "placed": "move",
    "places": "move",
    "placing": "move",
    "put": "move",
    "puts": "move",
    "putting": "move",
    "puting": "move",
    "close": "close",
    "closed": "close",
    "closes": "close",
    "closing": "close",
    "fold": "fold",
    "folded": "fold",
    "folding": "fold",
    "folds": "fold",
    "open": "open",
    "opened": "open",
    "opening": "open",
    "opens": "open",
    "pick": "pick",
    "picked": "pick",
    "picking": "pick",
    "picks": "pick",
    "pickup": "pick",
    "remove": "remove",
    "removed": "remove",
    "removing": "remove",
    "take": "take",
    "takes": "take",
    "taking": "take",
    "took": "take",
    "unfold": "unfold",
    "unfolded": "unfold",
    "unfolding": "unfold",
    "unfolds": "unfold",
}
_ARTICLES = {"a", "an", "the"}


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


def canonical_metadata_text(text: Optional[str]) -> str:
    """Match the text encoder's stable Unicode/case/whitespace normalization."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    return " ".join(normalized.split())


def canonical_bridge_task(text: Optional[str]) -> str:
    """Conservatively collapse Bridge surface aliases into stable task labels.

    This deliberately does not attempt semantic clustering.  Different object,
    source, target, direction, or action tokens remain different labels.
    """
    normalized = canonical_metadata_text(text)
    for pattern, replacement in _BRIDGE_PHRASE_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    tokens = [token for token in normalized.split() if token not in _ARTICLES]
    if tokens:
        tokens[0] = _BRIDGE_ACTION_ALIASES.get(tokens[0], tokens[0])
    return " ".join(tokens)


def configured_task_language_mode(
    config: Optional[Mapping[str, Any]], routing_key: Any = None
) -> str:
    """Resolve an optional per-embodiment mode with a global legacy fallback."""
    if config is None:
        return TASK_LANGUAGE_METADATA
    default_mode = config.get("task_language_mode", TASK_LANGUAGE_METADATA)
    per_key = config.get("task_language_modes", {}) or {}
    key = getattr(routing_key, "value", routing_key)
    selected = per_key.get(str(key), default_mode) if key else default_mode
    return normalize_task_language_mode(selected)


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
    if normalized_mode == TASK_LANGUAGE_CANONICAL_METADATA:
        return canonical_metadata_text(original_text)
    if normalized_mode == TASK_LANGUAGE_BRIDGE_CANONICAL:
        return canonical_bridge_task(original_text)
    return str(original_text or "")
