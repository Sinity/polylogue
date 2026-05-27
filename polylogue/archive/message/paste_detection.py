"""Paste-detection heuristics for message content.

These run at materialization time and are intentionally simple and fast.
The goal is to distinguish typed-by-human prose from pasted/inserted content
(chatlogs, tool output, code dumps, structured documents).

The module also exposes :func:`has_paste_indicator` for hook event payloads
(Claude Code PreToolUse / PostToolUse / UserPromptSubmit, Codex equivalents):
hook payloads provide ground-truth paste markers like ``[Pasted text #1]``
before any expansion, so the daemon ingest path can flag messages even when
the assembled text would otherwise look like ordinary prose.
"""

from __future__ import annotations

import re
from typing import Any

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

# Claude Code expands clipboard pastes into ``[Pasted text #N]`` markers in
# the UserPromptSubmit hook payload before they are rewritten into the actual
# prompt text. The marker is the most reliable paste signal because it is
# emitted by the agent runtime, not inferred from message shape. The runtime
# also emits richer variants like ``[Pasted text #1 +6 lines]`` and
# ``[Pasted text #1 +6 lines] more prose``; everything past ``#N`` up to the
# closing bracket is annotation and must not be required by the matcher
# (#1583 — the original strict pattern silently missed all line-count
# variants).
_PASTE_MARKER_PATTERN = re.compile(
    r"\[Pasted\s+(text|content|image)\s*#\d+(?:[^\[\]]*)\]",
    re.IGNORECASE,
)

# Base64 blobs of meaningful size are almost never typed by hand. We find a
# contiguous run of >=512 base64-alphabet characters and then require a
# structural marker (mixed case, digit, or one of ``+`` / ``/`` / ``=``) so
# that long single-character fills like ``"x" * 4000`` do not trigger.
_BASE64_RUN_PATTERN = re.compile(r"[A-Za-z0-9+/=]{512,}")
_BASE64_STRUCTURAL_CHARS = re.compile(r"[+/=0-9]")

# Hook event payload field names that may carry user-visible text. Both
# Claude Code and Codex use the same field names for these.
_HOOK_PASTE_TEXT_FIELDS = ("prompt", "text", "content", "message", "tool_output", "tool_input")


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


def _has_base64_blob(text: str) -> bool:
    """True if the text contains a long base64-shaped blob.

    Requires both length (>=512 contiguous base64-alphabet chars) and a
    structural marker (mixed case, digit, or one of ``+`` / ``/`` / ``=``)
    so that uniform fills do not trigger the heuristic.
    """
    for match in _BASE64_RUN_PATTERN.finditer(text):
        run = match.group(0)
        if _BASE64_STRUCTURAL_CHARS.search(run):
            return True
        has_upper = any(c.isupper() for c in run)
        has_lower = any(c.islower() for c in run)
        if has_upper and has_lower:
            return True
    return False


def detect_paste(text: str | None) -> int:
    """Return 1 if the message text is dominated by pasted/inserted content.

    Heuristics (any one match is sufficient):
    1. Text contains an explicit ``[Pasted text #N]`` (or content/image) marker
       emitted by the agent runtime before clipboard expansion.
    2. Text contains a contiguous base64-like blob of >=512 characters.
    3. Total text length exceeds 4000 characters.
    4. Text matches a known chatlog-forwarding pattern.
    5. More than 70% of characters live inside fenced code blocks.
    """
    if not text or not text.strip():
        return 0
    if _PASTE_MARKER_PATTERN.search(text):
        return 1
    if _has_base64_blob(text):
        return 1
    if len(text) > _PASTE_LENGTH_THRESHOLD:
        return 1
    if _has_forwarding_pattern(text):
        return 1
    if _code_fence_ratio(text) > 0.7:
        return 1
    return 0


def _flatten_payload_text(value: Any) -> list[str]:
    """Walk a hook event payload and collect text-shaped leaves."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for field in _HOOK_PASTE_TEXT_FIELDS:
            if field in value:
                out.extend(_flatten_payload_text(value[field]))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_flatten_payload_text(item))
        return out
    return []


def has_paste_indicator(hook_payload: Any) -> bool:
    """Return True if a Claude Code / Codex hook event payload carries paste content.

    Hook events (``PreToolUse``, ``PostToolUse``, ``UserPromptSubmit``) provide
    ground-truth paste signals before any text expansion. This helper extracts
    the relevant text fields from the payload and runs :func:`detect_paste`
    over them. The function is intentionally permissive about input shape:
    it accepts a raw payload dict, a wrapped hook record (``{"payload": {...}}``),
    a list of records, or arbitrary nested JSON.
    """
    if hook_payload is None:
        return False
    if isinstance(hook_payload, dict) and "payload" in hook_payload and "event_type" in hook_payload:
        # Wrapped hook record; unwrap to inner payload.
        hook_payload = hook_payload["payload"]
    return any(detect_paste(text) == 1 for text in _flatten_payload_text(hook_payload))


__all__ = ["detect_paste", "has_paste_indicator"]
