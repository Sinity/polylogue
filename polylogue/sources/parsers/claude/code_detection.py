"""Claude Code payload shape detection."""

from __future__ import annotations

from collections.abc import Sequence

_CODE_ONLY_TYPES = frozenset(
    {
        "file-history-snapshot",
        "queue-operation",
        "custom-title",
        "user",
        "assistant",
        "summary",
        "progress",
        "result",
    }
)


def looks_like_code(payload: Sequence[object]) -> bool:
    """Return whether a payload matches the Claude Code record format."""
    if not isinstance(payload, list):
        return False
    for item in payload:
        if not isinstance(item, dict):
            continue
        if any(key in item for key in ("parentUuid", "leafUuid", "sessionId", "session_id")):
            return True
        item_type = item.get("type")
        if isinstance(item_type, str) and item_type in _CODE_ONLY_TYPES:
            return True
    return False


__all__ = ["looks_like_code"]
