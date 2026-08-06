#!/usr/bin/env python3
"""Remove terminal formatting/control sequences from a byte stream."""

from __future__ import annotations

import re
import sys


# ECMA-48 CSI sequences (colors, cursor movement, erase commands), OSC
# sequences (terminal title/hyperlinks), and simple two-byte ESC sequences.
ANSI_ESCAPE_RE = re.compile(
    rb"\x1b(?:"
    rb"\[[0-?]*[ -/]*[@-~]"
    rb"|\][^\x07\x1b]*(?:\x07|\x1b\\)"
    rb"|[@-_]"
    rb")"
)
CONTROL_RE = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1a\x1c-\x1f\x7f]")


def strip_terminal_controls(data: bytes) -> bytes:
    """Return log bytes without ANSI escapes or non-text C0 controls."""

    data = ANSI_ESCAPE_RE.sub(b"", data)
    # Preserve terminal progress updates as ordinary log lines.
    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return CONTROL_RE.sub(b"", data)


def main() -> None:
    source = sys.stdin.buffer
    destination = sys.stdout.buffer
    for line in iter(source.readline, b""):
        destination.write(strip_terminal_controls(line))
        destination.flush()


if __name__ == "__main__":
    main()
