"""Claude Code payload shape detection."""

from __future__ import annotations

from collections.abc import Sequence

from polylogue.sources.detection_projection import DetectorProjection

#: Type tokens specific enough to Claude Code's own record vocabulary that
#: their bare presence is sufficient evidence on its own.
_CODE_ONLY_TYPES = frozenset(
    {
        "file-history-snapshot",
        "queue-operation",
        "custom-title",
        "summary",
        "progress",
        "result",
    }
)
#: ``"user"``/``"assistant"`` are NOT specific to Claude Code -- they are
#: generic role/edge-type words that also appear as a bare ``"type"`` value on
#: unrelated structured data sitting under a watched Claude Code directory
#: (e.g. a third-party graph-edge index recording
#: ``{"conversation", "parent", "child", "type", "timestamp"}`` rows, whose
#: ``type`` happens to be "assistant"/"user" -- polylogue-9ykn). Treating them
#: as sufficient on their own is exactly how such a file misclassified as a
#: Claude Code session. Require them to co-occur with a genuine record
#: envelope marker instead -- every real Claude Code JSONL record carries at
#: least one of these regardless of its own ``type``.
_AMBIGUOUS_ROLE_TYPES = frozenset({"user", "assistant"})
#: Keys strong enough that their bare presence alone is sufficient evidence:
#: Claude Code's own camelCase transcript envelope.
_STRONG_SESSION_KEYS = ("parentUuid", "leafUuid", "sessionId")
#: Weaker companion markers that only count when paired with an ambiguous
#: role-word ``type`` value (see ``_AMBIGUOUS_ROLE_TYPES``) -- not strong
#: enough to stand alone (``"uuid"`` in particular is too generic a field name
#: across unrelated JSON shapes to be positive evidence by itself). The
#: snake_case ``session_id`` of Claude Code's stream-json output is one of
#: them: Hermes and other origins carry the same key on their own documents,
#: so alone it is no evidence of Claude Code.
_AMBIGUOUS_TYPE_ENVELOPE_MARKERS = frozenset({"uuid", "cwd", "version", "message", "session_id"})


def looks_like_code(payload: Sequence[object]) -> bool:
    """Return whether a payload matches the Claude Code record format."""
    if not isinstance(payload, list):
        return False
    for item in payload:
        if not isinstance(item, dict):
            continue
        if any(key in item for key in _STRONG_SESSION_KEYS):
            return True
        item_type = item.get("type")
        if not isinstance(item_type, str):
            continue
        if item_type in _CODE_ONLY_TYPES:
            return True
        if item_type in _AMBIGUOUS_ROLE_TYPES and any(key in item for key in _AMBIGUOUS_TYPE_ENVELOPE_MARKERS):
            return True
    return False


__all__ = ["looks_like_code"]


def detection_projection() -> DetectorProjection:
    """Keep envelope markers exactly; consume every other field and record."""
    fields: dict[str, DetectorProjection | None] = dict.fromkeys(
        (*_STRONG_SESSION_KEYS, *_AMBIGUOUS_TYPE_ENVELOPE_MARKERS)
    )
    fields["type"] = DetectorProjection()
    return DetectorProjection(fields=fields)
