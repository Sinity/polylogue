"""Benchmark harness for identity-hash serialization strategies (polylogue-fqp0).

py-spy profile of the 2026-07-19 rebuild found ``hash_payload`` at 32%
cumulative CPU during census, dominated by json serialization of the session
tree run at least twice per revision: once inside ``session_content_hash``
(the whole tree, nested) and again inside ``session_revision_projection``
(each message/attachment/event payload rebuilt from scratch and hashed
individually). This module compares four strategies for computing the same
projection:

- ``status_quo_projection`` -- frozen copy of the pre-fix double-serialization
  behavior (build payloads once inside ``session_content_hash``, then build
  them again for per-item hashing). Kept here as a fixed baseline so benchmark
  numbers stay comparable after ``pipeline/ids.py`` is patched.
- ``dedup_projection`` -- build every hash-stable payload exactly once and
  reuse it for both the whole-tree hash and the per-item hashes. Produces
  byte-identical output to status-quo (same payloads, same hash_payload
  calls, just not duplicated) -- epoch-free. This is what
  ``polylogue.pipeline.ids.session_revision_projection`` now implements.
- ``merkle_projection`` -- compose ``session_hash`` from the per-message hash
  list plus a normalized header instead of re-serializing the full tree.
  Eliminates the O(total content) whole-tree dump entirely, but changes
  ``session_hash`` bytes -- a durable-evidence epoch, NOT wired into the
  pipeline.
- ``orjson_projection`` -- swap ``hash_payload``'s stdlib ``json.dumps`` for
  ``orjson.dumps``. Faster C serializer, but ``orjson`` escapes non-ASCII
  differently than stdlib json's default ``ensure_ascii=True`` and rejects
  ``NaN``/``Infinity`` -- changes hash bytes for any non-ASCII content. Also
  NOT wired into the pipeline.

Only ``dedup_projection``'s logic is shipped (as the new
``session_revision_projection``); ``merkle_projection`` and
``orjson_projection`` exist to give the epoch-vs-cache-first decision on
polylogue-fqp0 real numbers instead of guesses.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass

import orjson

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.pipeline.ids import (
    SessionRevisionProjection,
    _attachment_hash_payload,
    _message_hash_payload,
    _normalize_for_hash,
    _session_hash_payload,
)
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.parsers.base_models import ParsedSessionEvent


def build_synthetic_session(
    *,
    message_count: int = 300,
    tool_call_every: int = 3,
    avg_text_chars: int = 800,
    session_event_count: int = 20,
) -> ParsedSession:
    """A realistic-shaped session: mixed text/tool_use/tool_result blocks.

    ``tool_call_every`` controls how often an assistant message carries a
    tool_use+tool_result pair (the ``tool_input`` dict is the one nested
    payload that gets its own recursive ``hash_payload`` call today).
    """
    filler = "lorem ipsum dolor sit amet consectetur adipiscing elit "
    messages: list[ParsedMessage] = []
    for i in range(message_count):
        role = Role.USER if i % 2 == 0 else Role.ASSISTANT
        text = f"message body {i} " + (filler * max(1, avg_text_chars // len(filler)))
        blocks: list[ParsedContentBlock] = []
        if role is Role.ASSISTANT and tool_call_every > 0 and i % tool_call_every == 0:
            blocks = [
                ParsedContentBlock(
                    type=BlockType.TOOL_USE,
                    tool_name="Bash",
                    tool_id=f"tool-{i}",
                    tool_input={
                        "command": f"echo synthetic-{i}",
                        "description": "synthetic benchmark tool call",
                        "extra_context": [f"line-{n}" for n in range(20)],
                    },
                ),
                ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id=f"tool-{i}", text=f"output-{i} " * 15),
            ]
        messages.append(
            ParsedMessage(
                provider_message_id=f"msg-{i:06d}",
                role=role,
                text=text,
                timestamp=f"2026-07-19T00:{i // 60 % 60:02d}:{i % 60:02d}Z",
                blocks=blocks,
            )
        )
    session_events = [
        ParsedSessionEvent(
            event_type="turn_context",
            timestamp=f"2026-07-19T00:00:{i % 60:02d}Z",
            payload={"turn": i, "detail": "x" * 200, "tags": [f"tag-{n}" for n in range(5)]},
        )
        for i in range(session_event_count)
    ]
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="hash-economics-benchmark-session",
        title="hash economics benchmark session",
        messages=messages,
        session_events=session_events,
    )


# ---------------------------------------------------------------------------
# status quo -- frozen copy of the pre-fix double-serialization behavior.
# ---------------------------------------------------------------------------


def _status_quo_session_content_hash(convo: ParsedSession) -> str:
    messages_payload = [
        _message_hash_payload(msg, msg.provider_message_id or f"msg-{idx}")
        for idx, msg in enumerate(convo.messages, start=1)
    ]
    attachments_payload = [_attachment_hash_payload(a) for a in convo.attachments]
    session_events_payload = [
        {
            "event_index": event_index,
            "event_type": _normalize_for_hash(event.event_type),
            "timestamp": _normalize_for_hash(event.timestamp),
            "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
            "payload": _hash_payload_stdlib(event.payload),
        }
        for event_index, event in enumerate(convo.session_events)
    ]
    return _hash_payload_stdlib(
        _session_hash_payload(
            title=convo.title,
            created_at=convo.created_at,
            updated_at=convo.updated_at,
            messages=messages_payload,
            attachments=attachments_payload,
            session_events=session_events_payload,
        )
    )


def _hash_payload_stdlib(payload: object) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def status_quo_projection(convo: ParsedSession) -> SessionRevisionProjection:
    """Pre-fix behavior: build every payload twice (once per hash target)."""
    session_hash_hex = _status_quo_session_content_hash(convo)

    message_payloads = [
        _message_hash_payload(message, message.provider_message_id or f"msg-{index}")
        for index, message in enumerate(convo.messages, start=1)
    ]
    attachment_payloads = [_attachment_hash_payload(a) for a in convo.attachments]
    event_payloads = [
        {
            "event_index": event_index,
            "event_type": _normalize_for_hash(event.event_type),
            "timestamp": _normalize_for_hash(event.timestamp),
            "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
            "payload": _hash_payload_stdlib(event.payload),
        }
        for event_index, event in enumerate(convo.session_events)
    ]
    return SessionRevisionProjection(
        session_hash=bytes.fromhex(session_hash_hex),
        message_hashes=tuple(bytes.fromhex(_hash_payload_stdlib(p)) for p in message_payloads),
        attachment_hashes=frozenset(bytes.fromhex(_hash_payload_stdlib(p)) for p in attachment_payloads),
        event_hashes=tuple(bytes.fromhex(_hash_payload_stdlib(p)) for p in event_payloads),
    )


# ---------------------------------------------------------------------------
# dedup -- build every hash-stable payload exactly once, reuse for both the
# whole-tree hash and the per-item hashes. Byte-identical to status quo
# (same payloads, same hash_payload calls, just not duplicated) -- epoch-free.
# This is what polylogue.pipeline.ids.session_revision_projection now ships.
# ---------------------------------------------------------------------------


def dedup_projection(convo: ParsedSession) -> SessionRevisionProjection:
    message_payloads = [
        _message_hash_payload(message, message.provider_message_id or f"msg-{index}")
        for index, message in enumerate(convo.messages, start=1)
    ]
    attachment_payloads = [_attachment_hash_payload(a) for a in convo.attachments]
    event_payloads = [
        {
            "event_index": event_index,
            "event_type": _normalize_for_hash(event.event_type),
            "timestamp": _normalize_for_hash(event.timestamp),
            "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
            "payload": _hash_payload_stdlib(event.payload),
        }
        for event_index, event in enumerate(convo.session_events)
    ]
    session_hash_hex = _hash_payload_stdlib(
        _session_hash_payload(
            title=convo.title,
            created_at=convo.created_at,
            updated_at=convo.updated_at,
            messages=message_payloads,
            attachments=attachment_payloads,
            session_events=event_payloads,
        )
    )
    return SessionRevisionProjection(
        session_hash=bytes.fromhex(session_hash_hex),
        message_hashes=tuple(bytes.fromhex(_hash_payload_stdlib(p)) for p in message_payloads),
        attachment_hashes=frozenset(bytes.fromhex(_hash_payload_stdlib(p)) for p in attachment_payloads),
        event_hashes=tuple(bytes.fromhex(_hash_payload_stdlib(p)) for p in event_payloads),
    )


# ---------------------------------------------------------------------------
# merkle -- session_hash composed from the per-message hash list. Changes
# session_hash bytes: NOT wired into the pipeline, benchmark-only.
# ---------------------------------------------------------------------------


def merkle_projection(convo: ParsedSession) -> SessionRevisionProjection:
    message_payloads = [
        _message_hash_payload(message, message.provider_message_id or f"msg-{index}")
        for index, message in enumerate(convo.messages, start=1)
    ]
    attachment_payloads = [_attachment_hash_payload(a) for a in convo.attachments]
    event_payloads = [
        {
            "event_index": event_index,
            "event_type": _normalize_for_hash(event.event_type),
            "timestamp": _normalize_for_hash(event.timestamp),
            "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
            "payload": _hash_payload_stdlib(event.payload),
        }
        for event_index, event in enumerate(convo.session_events)
    ]
    message_hashes = tuple(bytes.fromhex(_hash_payload_stdlib(p)) for p in message_payloads)
    attachment_hashes = frozenset(bytes.fromhex(_hash_payload_stdlib(p)) for p in attachment_payloads)
    event_hashes = tuple(bytes.fromhex(_hash_payload_stdlib(p)) for p in event_payloads)

    # Composed from digests + a small header only -- O(N) in message count,
    # not O(total content) like the whole-tree dump.
    header = {
        "title": _normalize_for_hash(convo.title),
        "created_at": _normalize_for_hash(convo.created_at),
        "updated_at": _normalize_for_hash(convo.updated_at),
        "message_hashes": [h.hex() for h in message_hashes],
        "attachment_hashes": sorted(h.hex() for h in attachment_hashes),
        "event_hashes": [h.hex() for h in event_hashes],
    }
    session_hash = bytes.fromhex(_hash_payload_stdlib(header))
    return SessionRevisionProjection(
        session_hash=session_hash,
        message_hashes=message_hashes,
        attachment_hashes=attachment_hashes,
        event_hashes=event_hashes,
    )


# ---------------------------------------------------------------------------
# orjson -- same dedup shape as the shipped fix, but hash_payload's stdlib
# json.dumps swapped for orjson.dumps(option=OPT_SORT_KEYS). Changes hash
# bytes for non-ASCII content (orjson does not escape to \\uXXXX); benchmark
# only, NOT wired into the pipeline.
# ---------------------------------------------------------------------------


def _hash_payload_orjson(payload: object) -> str:
    serialized = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(serialized).hexdigest()


def orjson_projection(convo: ParsedSession) -> SessionRevisionProjection:
    message_payloads = [
        _message_hash_payload(message, message.provider_message_id or f"msg-{index}")
        for index, message in enumerate(convo.messages, start=1)
    ]
    attachment_payloads = [_attachment_hash_payload(a) for a in convo.attachments]
    event_payloads = [
        {
            "event_index": event_index,
            "event_type": _normalize_for_hash(event.event_type),
            "timestamp": _normalize_for_hash(event.timestamp),
            "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
            "payload": _hash_payload_orjson(event.payload),
        }
        for event_index, event in enumerate(convo.session_events)
    ]
    session_hash_hex = _hash_payload_orjson(
        _session_hash_payload(
            title=convo.title,
            created_at=convo.created_at,
            updated_at=convo.updated_at,
            messages=message_payloads,
            attachments=attachment_payloads,
            session_events=event_payloads,
        )
    )
    return SessionRevisionProjection(
        session_hash=bytes.fromhex(session_hash_hex),
        message_hashes=tuple(bytes.fromhex(_hash_payload_orjson(p)) for p in message_payloads),
        attachment_hashes=frozenset(bytes.fromhex(_hash_payload_orjson(p)) for p in attachment_payloads),
        event_hashes=tuple(bytes.fromhex(_hash_payload_orjson(p)) for p in event_payloads),
    )


# ---------------------------------------------------------------------------
# ceiling -- dedup + merkle + orjson combined, for reference only. Shows the
# theoretical maximum if every non-epoch-free lever were pulled together;
# NOT a candidate for immediate implementation (needs a durable-evidence
# epoch bump like merkle/orjson alone).
# ---------------------------------------------------------------------------


def dedup_merkle_orjson_projection(convo: ParsedSession) -> SessionRevisionProjection:
    message_payloads = [
        _message_hash_payload(message, message.provider_message_id or f"msg-{index}")
        for index, message in enumerate(convo.messages, start=1)
    ]
    attachment_payloads = [_attachment_hash_payload(a) for a in convo.attachments]
    event_payloads = [
        {
            "event_index": event_index,
            "event_type": _normalize_for_hash(event.event_type),
            "timestamp": _normalize_for_hash(event.timestamp),
            "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
            "payload": _hash_payload_orjson(event.payload),
        }
        for event_index, event in enumerate(convo.session_events)
    ]
    message_hashes = tuple(bytes.fromhex(_hash_payload_orjson(p)) for p in message_payloads)
    attachment_hashes = frozenset(bytes.fromhex(_hash_payload_orjson(p)) for p in attachment_payloads)
    event_hashes = tuple(bytes.fromhex(_hash_payload_orjson(p)) for p in event_payloads)
    header = {
        "title": _normalize_for_hash(convo.title),
        "created_at": _normalize_for_hash(convo.created_at),
        "updated_at": _normalize_for_hash(convo.updated_at),
        "message_hashes": [h.hex() for h in message_hashes],
        "attachment_hashes": sorted(h.hex() for h in attachment_hashes),
        "event_hashes": [h.hex() for h in event_hashes],
    }
    session_hash = bytes.fromhex(_hash_payload_orjson(header))
    return SessionRevisionProjection(
        session_hash=session_hash,
        message_hashes=message_hashes,
        attachment_hashes=attachment_hashes,
        event_hashes=event_hashes,
    )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    label: str
    ns_per_session: float
    sessions_timed: int
    repeats: int


def time_projection(
    label: str,
    fn: Callable[[ParsedSession], SessionRevisionProjection],
    sessions: list[ParsedSession],
    *,
    repeats: int = 5,
) -> BenchmarkResult:
    """Median wall-time per session across ``repeats`` full passes."""
    totals: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for session in sessions:
            fn(session)
        totals.append(time.perf_counter() - start)
    totals.sort()
    median_total = totals[len(totals) // 2]
    return BenchmarkResult(
        label=label,
        ns_per_session=(median_total / len(sessions)) * 1e9,
        sessions_timed=len(sessions),
        repeats=repeats,
    )


__all__ = [
    "BenchmarkResult",
    "build_synthetic_session",
    "dedup_merkle_orjson_projection",
    "dedup_projection",
    "merkle_projection",
    "orjson_projection",
    "status_quo_projection",
    "time_projection",
]
