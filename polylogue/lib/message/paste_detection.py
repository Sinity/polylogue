"""Paste-detection heuristics for message content.

These run at materialization time and are intentionally simple and fast.
The goal is to distinguish typed-by-human prose from pasted/inserted content
(chatlogs, tool output, code dumps, structured documents).
"""

from __future__ import annotations

import re

# Messages longer than this are very likely dominated by pasted content.
_PASTE_LENGTH_THRESHOLD = 4000

# Patterns that strongly indicate chatlog forwarding.
_PASTE_FORWARDING_PATTERNS = (
    re.compile(r"previous\s+chatlog\s+attached\s+below", re.IGNORECASE),
    re.compile(r"showing\s+you\s+(the\s+)?(last|previous)\s+chatlog", re.IGNORECASE),
    re.compile(r"here\s+(is|are)\s+(the\s+)?(full\s+)?(chat\s*)?log", re.IGNORECASE),
    re.compile(r"attached\s+(chat\s*)?log\s+below", re.IGNORECASE),
    re.compile(r"below\s+is\s+(the\s+)?(full\s+)?(chat\s*)?log", re.IGNORECASE),
)

# Code fence markers.
_FENCE_PATTERN = re.compile(r"^```", re.MULTILINE)


def _code_fence_ratio(text: str) -> float:
    """Fraction of characters that live inside fenced code blocks."""
    parts = _FENCE_PATTERN.split(text)
    # After splitting, odd-indexed parts are inside fences (content between
    # opening ``` and closing ```).  Even indices are outside.
    if len(parts) < 3:
        return 0.0
    inside = sum(len(p) for p in parts[1::2])
    return inside / max(len(text), 1)


def _has_forwarding_pattern(text: str) -> bool:
    """True if the text matches known chatlog-forwarding patterns."""
    return any(pattern.search(text) for pattern in _PASTE_FORWARDING_PATTERNS)


def detect_paste(text: str | None) -> int:
    """Return 1 if the message text is dominated by pasted/inserted content.

    Heuristics (any one match is sufficient):
    1. Total text length exceeds 4000 characters.
    2. Text matches a known chatlog-forwarding pattern.
    3. More than 70% of characters live inside fenced code blocks.
    """
    if not text or not text.strip():
        return 0
    if len(text) > _PASTE_LENGTH_THRESHOLD:
        return 1
    if _has_forwarding_pattern(text):
        return 1
    if _code_fence_ratio(text) > 0.7:
        return 1
    return 0


__all__ = ["detect_paste"]
