"""Stable task-language handling shared by training and deployment.

The policy uses language as a task identifier rather than as an open-vocabulary
VLM prompt.  Keep the transformations deterministic and intentionally
conservative: dataset taxonomies win when one exists (RoboTwin), while Bridge
only receives lexical normalization that can be reproduced at serving time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
import re
import unicodedata
from typing import Any, Mapping, Optional


TASK_LANGUAGE_METADATA = "metadata"
TASK_LANGUAGE_DATASET_NAME = "dataset_name"
TASK_LANGUAGE_CANONICAL_METADATA = "canonical_metadata"
TASK_LANGUAGE_BRIDGE_CANONICAL = "bridge_canonical"
TASK_LANGUAGE_BRIDGE_TAXONOMY = "bridge_taxonomy"
TASK_LANGUAGE_MODES = {
    TASK_LANGUAGE_METADATA,
    TASK_LANGUAGE_DATASET_NAME,
    TASK_LANGUAGE_CANONICAL_METADATA,
    TASK_LANGUAGE_BRIDGE_CANONICAL,
    TASK_LANGUAGE_BRIDGE_TAXONOMY,
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


@dataclass(frozen=True)
class BridgeTaskLabel:
    """Deterministic structured label for one Bridge instruction."""

    canonical_text: str
    family: str
    status: str
    confidence: str


_BRIDGE_TAXONOMY_REPLACEMENTS = (
    (r"\bright\s+top\b", "top right"),
    (r"\bleft\s+top\b", "top left"),
    (r"\bright\s+bottom\b", "bottom right"),
    (r"\bleft\s+bottom\b", "bottom left"),
    (r"\bcenter\s+top\b", "top center"),
    (r"\bcenter\s+bottom\b", "bottom center"),
    (r"\bmiddle\s+top\b", "top center"),
    (r"\bmiddle\s+bottom\b", "bottom center"),
    (r"\bin\s+to\b", "in"),
    (r"\bout\s+of\b", "from"),
    (r"\boff\s+of\b", "from"),
    (r"\babove\s+of\b", "above"),
    (r"\bunderneath\b", "under"),
    (r"\bnear\s+to\b", "near"),
    (r"\bon\s+right\s+side\s+of\b", "right of"),
    (r"\bon\s+left\s+side\s+of\b", "left of"),
    (r"\bon\s+right\s+of\b", "right of"),
    (r"\bon\s+left\s+of\b", "left of"),
    (r"\bto\s+right\s+side\s+of\b", "right of"),
    (r"\bto\s+left\s+side\s+of\b", "left of"),
    (r"\bright\s+side\s+of\b", "right of"),
    (r"\bleft\s+side\s+of\b", "left of"),
    (r"\bupper\s+right\b", "top right"),
    (r"\bupper\s+left\b", "top left"),
    (r"\blower\s+right\b", "bottom right"),
    (r"\blower\s+left\b", "bottom left"),
    (r"\brigth\b", "right"),
    (r"\bbotton\b", "bottom"),
    (r"\bbotom\b", "bottom"),
    (r"\bsliver\b", "silver"),
    (r"\bfabric\b", "cloth"),
    (r"\brag\b", "cloth"),
    (r"\bclothes\b", "cloth"),
)

_BRIDGE_KNOWN_FAMILIES = {
    "place",
    "remove",
    "pick",
    "open",
    "close",
    "fold",
    "unfold",
    "sweep",
    "turn",
    "flip",
    "topple",
    "upright",
    "slide",
    "push",
    "pull",
    "pour",
    "wipe",
    "zip",
    "unzip",
    "cover",
    "hold",
    "reach",
    "transition",
    "no_op",
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


def classify_bridge_task(text: Optional[str]) -> BridgeTaskLabel:
    """Map a Bridge instruction to a role-preserving controlled label.

    The classifier intentionally keeps objects, targets, relations, and ordered
    directions. It only merges surface forms that do not change the requested
    behavior. Empty or unrecognized descriptions are made explicit so a data
    pipeline can quarantine them instead of silently sharing one task ID.
    """

    normalized = canonical_bridge_task(text)
    if not normalized:
        return BridgeTaskLabel("", "unlabeled", "unlabeled", "none")

    for pattern, replacement in _BRIDGE_TAXONOMY_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = " ".join(normalized.split())

    if normalized == "lever vertical to front":
        normalized = "turn lever vertical to front"
    if normalized in {
        "nothing",
        "not moving anything",
        "no change in image",
        "robot arm did nothing",
        "robot did nothing",
        "arm did nothing",
        "arm did noothing",
    }:
        normalized = "no_op"

    # A pick/remove followed by a placement is classified by its final goal,
    # while retaining any source phrase in the object span.
    compound = re.match(
        r"^(?:pick|take|remove|grab|lift)\s+(.+?)\s+and\s+"
        r"(?:move|place|put)\s+(?:it\s+|them\s+)?(.+)$",
        normalized,
    )
    if compound:
        normalized = f"place {compound.group(1)} {compound.group(2)}"
    elif normalized.startswith("move "):
        normalized = "place " + normalized.removeprefix("move ")
    else:
        removal = re.match(
            r"^(?:take|remove|pick|grab)\s+(.+?)\s+(?:from|off)\s+(.+)$",
            normalized,
        )
        if removal:
            normalized = f"remove {removal.group(1)} from {removal.group(2)}"
        elif re.match(r"^(?:grab|lift)\s+", normalized):
            normalized = re.sub(r"^(?:grab|lift)\s+", "pick ", normalized)
        elif normalized.startswith("take "):
            normalized = "pick " + normalized.removeprefix("take ")

    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized.startswith("end effector reaching "):
        normalized = "reach " + normalized.removeprefix("end effector reaching ")
    elif normalized.startswith("end effector transition from "):
        normalized = "transition " + normalized.removeprefix(
            "end effector transition from "
        )
    elif normalized in {
        "robotic arm did not move",
        "robot arm did not move",
        "robot did not move",
    }:
        normalized = "no_op"

    family = normalized.split()[0] if normalized else "unlabeled"
    if family in _BRIDGE_KNOWN_FAMILIES:
        return BridgeTaskLabel(normalized, family, "classified", "high")
    return BridgeTaskLabel(normalized, family, "needs_review", "low")


def canonical_bridge_taxonomy_task(text: Optional[str]) -> str:
    """Return the controlled Bridge text used as a categorical task label."""

    return classify_bridge_task(text).canonical_text


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
    if normalized_mode == TASK_LANGUAGE_BRIDGE_TAXONOMY:
        return canonical_bridge_taxonomy_task(original_text)
    return str(original_text or "")
