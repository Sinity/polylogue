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


def has_paste_marker(text: str | None) -> bool:
    """Ground-truth paste signal: an explicit ``[Pasted text #N]`` marker.

    The agent runtime emits ``[Pasted text|content|image #N]`` (optionally
    ``+M lines``) before clipboard expansion. This is the only *positive*
    evidence that a paste actually occurred — it is asserted by the runtime,
    not inferred from message shape. Kept separate from the size/format
    heuristics so a ground-truth signal is never conflated with a proxy.
    """
    if not text or not text.strip():
        return False
    return _PASTE_MARKER_PATTERN.search(text) is not None


def has_paste_heuristic(text: str | None) -> bool:
    """Heuristic *proxy* for pasted/inserted content — NOT ground truth.

    Any one match is sufficient:
    1. A contiguous base64-like blob of >=512 characters.
    2. Total text length exceeds 4000 characters.
    3. A known chatlog-forwarding phrase.
    4. More than 70% of characters live inside fenced code blocks.

    These are shape signals: a long typed prose answer or a heavily
    code-quoting human reply will match without any paste having occurred.
    Callers must treat the result as a weak proxy (candidate), never as a
    confirmed paste.
    """
    if not text or not text.strip():
        return False
    if _has_base64_blob(text):
        return True
    if len(text) > _PASTE_LENGTH_THRESHOLD:
        return True
    if _has_forwarding_pattern(text):
        return True
    return _code_fence_ratio(text) > 0.7


def detect_paste(text: str | None) -> int:
    """Selection gate: 1 if *any* paste evidence (marker or proxy) is present.

    This is the union of :func:`has_paste_marker` (ground truth) and
    :func:`has_paste_heuristic` (proxy), used only to decide whether a hook
    event or message is worth closer paste inspection
    (see :func:`has_paste_indicator`). It is deliberately NOT a stored paste
    fact: the persisted ``messages.has_paste`` column is marker-derived (from
    ``paste_spans``), and ``paste_boundary`` distinguishes a marker
    (``projected``) from a heuristic-only proxy (``whole_message_fallback``)
    via :func:`resolve_paste_boundary_state`. Do not promote this union
    boolean into a ground-truth "a paste occurred" claim.
    """
    return int(has_paste_marker(text) or has_paste_heuristic(text))


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


#: #1655 — boundary-state vocabulary for paste evidence.
#: ``exact``: paste span boundaries known to the character; recoverable text.
#: ``projected``: paste boundaries inferred (heuristic marker match).
#: ``whole_message_fallback``: paste exists but no boundaries; whole message.
#: ``hash_only``: pastedContents recorded paste existence but content unrecoverable.
_PASTE_BOUNDARY_STATES = frozenset({"exact", "projected", "whole_message_fallback", "hash_only"})


def resolve_paste_boundary_state(
    *,
    message_text: str | None = None,
    history_has_paste: bool = False,
    history_has_content: bool = False,
    hook_has_paste: bool = False,
) -> str | None:
    """Resolve paste boundary state from available evidence sources.

    Priority: history exact-content > hook markers > history hash-only
    > in-text paste marker (ground truth, ``projected``) > heuristic proxy
    (``whole_message_fallback``) > none.

    The marker and the heuristic proxies are resolved as distinct states: a
    ``[Pasted text #N]`` marker carries a known boundary (``projected``),
    while a size/code-fence proxy only signals "a paste likely happened
    somewhere in this message" (``whole_message_fallback``). They are never
    collapsed into one state.
    """
    if history_has_paste and history_has_content:
        return "exact"
    if hook_has_paste:
        return "projected"
    if history_has_paste and not history_has_content:
        return "hash_only"
    if has_paste_marker(message_text):
        return "projected"
    if has_paste_heuristic(message_text):
        return "whole_message_fallback"
    return None


__all__ = [
    "detect_paste",
    "has_paste_heuristic",
    "has_paste_indicator",
    "has_paste_marker",
    "resolve_paste_boundary_state",
]
