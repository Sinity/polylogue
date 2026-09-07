"""Generation-agnostic reads over Claude Code / Codex hook payloads.

The harness emits a camelCase generation of every hook payload alongside the
snake_case one (``toolUseId`` for ``tool_use_id``, ``sessionId`` for
``session_id``, ...). A reader keyed on one spelling sees the other generation
as a payload whose fields were never sent, and an absent field is
indistinguishable from a field nothing sent -- which is why this class of
defect survives to the archive as empty values rather than as an error.

Readers therefore name the canonical (snake_case) key and resolve it here.
:data:`HOOK_READER_KEYS` declares every canonical key an in-tree hook reader
consumes, so :func:`matched_reader_keys` can answer "did any reader key match
this record at all" -- an empty answer is a signal that a generation exists
this vocabulary does not describe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Canonical keys whose camelCase spelling is not the mechanical conversion of
#: the snake_case one. ``tool_response`` is the harness's snake_case name for
#: the field its camelCase generation calls ``toolResult``.
_IRREGULAR_SPELLINGS: Mapping[str, tuple[str, ...]] = {
    "tool_response": ("toolResult", "tool_result"),
}

#: Every canonical payload key an in-tree hook reader consumes:
#: ``sources/live/hook_paste_enrichment`` (session id, timestamp),
#: ``archive/message/paste_detection`` (the text fields), ``sources/hook_producer``
#: (provider detection), ``context/claude_agent_dispatch_correlation`` (subagent
#: lineage, read-only) and ``storage/sqlite/archive_tiers/write`` (the same
#: lineage written as a ``session_links`` edge). A key belongs here when a
#: reader reads it, not when the harness emits it.
HOOK_READER_KEYS: tuple[str, ...] = (
    "agent_id",
    "agent_type",
    "content",
    "message",
    "model",
    "permission_mode",
    "prompt",
    "session_id",
    "source",
    "text",
    "timestamp",
    "tool_input",
    "tool_output",
    "tool_response",
    "tool_use_id",
    "turn_id",
)


def _camel(key: str) -> str:
    head, _, tail = key.partition("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail.split("_") if part)


def payload_key_spellings(key: str) -> tuple[str, ...]:
    """Every spelling a hook payload may use for one canonical key."""

    spellings = [key, _camel(key), *_IRREGULAR_SPELLINGS.get(key, ())]
    return tuple(dict.fromkeys(spelling for spelling in spellings if spelling))


def hook_payload_field(payload: Any, key: str) -> object | None:
    """Read one canonical key from a hook payload in either generation."""

    if not isinstance(payload, Mapping):
        return None
    for spelling in payload_key_spellings(key):
        value: object = payload.get(spelling)
        if value not in (None, "", {}, []):
            return value
    return None


def hook_record_field(record: Any, key: str) -> object | None:
    """Read one canonical key from a spooled hook record, envelope before payload.

    The envelope is normalized by ``sources/hook_producer``; the payload is the
    harness's own bytes and carries whichever generation the harness emitted.
    """

    value = hook_payload_field(record, key)
    if value is not None:
        return value
    if isinstance(record, Mapping):
        return hook_payload_field(record.get("payload"), key)
    return None


def matched_reader_keys(record: Any) -> frozenset[str]:
    """Canonical reader keys that resolve against one hook record.

    Empty means no reader consumes anything from this record: either the
    harness sent a generation :data:`HOOK_READER_KEYS`' spellings do not
    describe, or the record carries no readable evidence at all. Callers treat
    an empty result as a signal, never as a no-op.
    """

    return frozenset(key for key in HOOK_READER_KEYS if hook_record_field(record, key) is not None)


__all__ = [
    "HOOK_READER_KEYS",
    "hook_payload_field",
    "hook_record_field",
    "matched_reader_keys",
    "payload_key_spellings",
]
