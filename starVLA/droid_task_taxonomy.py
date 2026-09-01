"""Deterministic semantic task taxonomy for DROID instructions."""

from __future__ import annotations

import re

EMPTY_LABEL = ""
NO_ACTION_LABEL = "no action"
OTHER_LABEL = "perform object manipulation"


def normalize_text(text: str) -> str:
    text = str(text or "").lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def _direction(text: str) -> str | None:
    directions = (
        ("top left", r"\b(?:top|upper) left\b|\bleft (?:top|upper)\b"),
        ("top right", r"\b(?:top|upper) right\b|\bright (?:top|upper)\b"),
        ("bottom left", r"\b(?:bottom|lower) left\b|\bleft (?:bottom|lower)\b"),
        ("bottom right", r"\b(?:bottom|lower) right\b|\bright (?:bottom|lower)\b"),
        ("left", r"\b(?:to|towards?|further|slightly|over)? ?(?:the )?left\b"),
        ("right", r"\b(?:to|towards?|further|slightly|over)? ?(?:the )?right\b"),
        ("forward", r"\b(?:forward|forwards|away from you|farther away)\b"),
        ("backward", r"\b(?:backward|backwards|towards? you|closer to you)\b"),
        ("up", r"\b(?:upward|upwards|higher|raise)\b"),
        ("down", r"\b(?:downward|downwards|lower)\b"),
        ("center", r"\b(?:center|centre|middle)\b"),
    )
    for label, pattern in directions:
        if _has(text, pattern):
            return label
    return None


def _acts_on_container(text: str, verb: str) -> bool:
    """Distinguish an action verb from adjectives such as ``open drawer``."""
    forms = {
        "open": r"(?:open|opened)",
        "close": r"(?:close|closed)",
    }.get(verb, re.escape(verb))
    return _has(text, rf"^(?:fully |partially |slightly |completely )?{forms}\b") or _has(
        text, rf"\b(?:then|and|next|afterwards?|to) {forms}\b"
    )


def _transfer_label(text: str) -> str:
    if _has(text, r"\b(?:out of|from inside|from|off)\b") and _has(
        text, r"\b(?:remove|take|get|pick|retrieve|unload)\b"
    ):
        return "remove object from container"
    if _has(text, r"\b(?:into|inside|in)\b"):
        return "place object in container"
    if _has(text, r"\b(?:onto|on top of|on)\b"):
        return "place object on surface"
    if _has(text, r"\b(?:under|underneath|below)\b"):
        return "place object under target"
    if _has(text, r"\b(?:beside|next to|near|close to)\b"):
        return "place object beside target"
    direction = _direction(text)
    if direction:
        return f"move object {direction}"
    return "move object"


def _primary_label(text: str) -> str:
    # Highly distinctive manipulation skills take priority over generic
    # pick/move words that often occur as preparatory clauses.
    distinctive = (
        ("pour contents", r"\b(?:pour|empty contents|spill)\b"),
        ("wipe or clean surface", r"\b(?:wipe|clean|scrub|wash)\b"),
        ("fold object", r"\b(?:fold|crease)\b"),
        ("unfold object", r"\b(?:unfold|spread out|open up the (?:cloth|towel|shirt))\b"),
        ("stack objects", r"\b(?:stack|pile)\b"),
        ("unstack objects", r"\b(?:unstack|unpile)\b"),
        ("hang object", r"\b(?:hang|hook)\b"),
        ("unhang object", r"\b(?:unhang|unhook|take .* off .* hook)\b"),
        ("cover object", r"\b(?:cover|put .* lid|place .* lid)\b"),
        ("uncover object", r"\b(?:uncover|remove .* lid|take .* lid off)\b"),
        ("stir or mix contents", r"\b(?:stir|mix)\b"),
        ("scoop contents", r"\b(?:scoop|ladle)\b|\buse .{0,30}\bspoon\b.{0,30}\bscoop\b"),
        ("sweep objects", r"\b(?:sweep|brush)\b"),
        ("shake object", r"\bshake\b"),
        (
            "wrap object",
            r"\b(?:wrap|bundle)\b|"
            r"\b(?:elastic|rubber band|rope|cord)\b.{0,50}\baround\b|"
            r"\baround\b.{0,50}\b(?:bottle|cup|jar|object)\b",
        ),
        ("unwrap object", r"\b(?:unwrap|unroll|unwind)\b"),
        ("plug in object", r"\bplug (?:in|into)\b|\bconnect .{0,30}\b(?:cable|cord|plug|socket)\b"),
        ("unplug object", r"\b(?:unplug|disconnect)\b"),
        ("attach object", r"\b(?:attach|fasten)\b|\bstick (?:the|a|an|it)\b"),
        ("detach object", r"\b(?:detach|release)\b"),
        ("tie object", r"\b(?:tie|make a knot)\b"),
        ("untie object", r"\b(?:untie|undo .* knot)\b"),
        ("cut or tear object", r"\b(?:cut|tear|rip)\b"),
        ("peel object", r"\bpeel\b"),
        ("throw object", r"\b(?:throw|toss)\b"),
        (
            "draw or write",
            r"\b(?:write|spell)\b|\bdraw\b.{0,40}\b(?:line|shape|circle|square|board|paper)\b",
        ),
        ("erase marking", r"\berase\b"),
        ("arrange or sort objects", r"\b(?:arrange|organize|sort|separate|reorder|rearrange)\b"),
        ("swap objects", r"\b(?:swap|exchange|change .* positions?)\b"),
        ("straighten object", r"\b(?:straighten|align|flatten)\b"),
        ("straighten object", r"\b(?:spread|stretch|untangle|unravel)\b"),
        ("roll object", r"^(?:roll|coil|wind)\b|\b(?:then|and) (?:roll|coil|wind)\b|\broll up\b"),
        ("zip or unzip container", r"\b(?:zip|unzip)\b"),
        ("arrange or sort objects", r"\b(?:create|form|add)\b.{0,50}\b(?:word|letter|tile)\b"),
        ("swap objects", r"\breplace\b"),
    )
    for label, pattern in distinctive:
        if _has(text, pattern):
            return label

    if _has(text, r"^empty\b|\bempty .{0,60}\b(?:into|in|onto|on)\b"):
        return "pour contents"
    if _has(text, r"^spoon\b"):
        return "scoop contents"
    if _has(text, r"^press\b|\bpress on\b"):
        return "press control"
    if _has(text, r"^plug\b"):
        return "plug in object"
    if _has(text, r"^connect\b"):
        return "attach object"
    if _has(text, r"\b(?:upright|lie down|lay down|tilt|point .* (?:left|right))\b"):
        return "rotate object"
    if _has(text, r"^draw (?:the )?curtains?\b"):
        return "slide object"
    if _has(text, r"\b(?:lock|unlock)\b"):
        return "operate lock"

    if _has(text, r"\b(?:turn|switch|flick) (?:the )?.{0,40}\b(?:off|down)\b|\bswitch off\b"):
        return "switch device off"
    if _has(text, r"\b(?:turn|switch|flick) (?:the )?.{0,40}\b(?:on|up)\b|\bswitch on\b"):
        return "switch device on"
    if _has(text, r"\b(?:press|push|click|tap)\b") and _has(
        text, r"\b(?:button|switch|lever|key|pedal|handle)\b"
    ):
        return "press control"
    acts_open = _acts_on_container(text, "open")
    acts_close = _acts_on_container(text, "close")
    if acts_open and acts_close:
        return "open and close container or door"
    if acts_close:
        return "close container or door"
    if acts_open:
        return "open container or door"
    if _has(text, r"\b(?:rotate|twist|turn over|flip)\b") or (
        _has(text, r"^turn\b") and not _has(text, r"\b(?:on|off)\b")
    ):
        return "rotate object"
    if _has(text, r"\b(?:click|press|tap|touch)\b") and _has(
        text, r"\b(?:mouse|remote|device|object)\b"
    ):
        return "press control"
    if _has(text, r"\b(?:push|shove)\b"):
        direction = _direction(text)
        return f"push object {direction}" if direction else "push object"
    if _has(text, r"\b(?:pull|drag)\b"):
        direction = _direction(text)
        return f"pull object {direction}" if direction else "pull object"
    if _has(text, r"\bslide\b"):
        direction = _direction(text)
        return f"slide object {direction}" if direction else "slide object"
    if _has(text, r"\b(?:lift|raise|pick up and hold)\b"):
        return "lift object"
    if _has(
        text,
        r"^(?:lower|put down|set down|drop)\b|"
        r"\b(?:then|and) (?:lower|put down|set down|drop)\b",
    ):
        return "lower object"
    if _has(text, r"^(?:center|position|adjust|readjust)\b"):
        direction = _direction(text)
        return f"move object {direction}" if direction else "move object"
    if _has(
        text,
        r"\b(?:put|place|move|pick|take|get|remove|transfer|bring|set|lay|"
        r"reposition|shift|retrieve)\b",
    ):
        return _transfer_label(text)
    return OTHER_LABEL


def classify_droid_task(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return EMPTY_LABEL
    if _has(text, r"^(?:no action|not action|do nothing|none|null|not applicable|n a)$"):
        return NO_ACTION_LABEL

    primary = _primary_label(text)

    # Preserve common goal-changing close/open clauses around transport.  This
    # keeps genuinely different multi-step tasks distinct without retaining
    # incidental wording or object identities.
    transfer = _has(
        text,
        r"\b(?:put|place|move|pick|take|get|remove|transfer|bring|set|lay|reposition|retrieve)\b",
    )
    if transfer and primary not in {
        "open container or door",
        "close container or door",
        "cover object",
        "uncover object",
    }:
        has_open = _acts_on_container(text, "open")
        has_close = _acts_on_container(text, "close")
        if primary == "open and close container or door":
            primary = _transfer_label(text)
        if has_open and has_close:
            return f"open container, {primary}, and close container"
        if has_open:
            return f"open container and {primary}"
        if has_close:
            return f"{primary} and close container"
    return primary
