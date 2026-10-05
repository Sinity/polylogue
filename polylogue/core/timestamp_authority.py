"""Session timestamp authority shared by ingest, replay, and storage.

The precedence is deliberately explicit:

1. producer-supplied session ``created_at``/``updated_at``;
2. parseable message timestamps (``occurred_at_ms``);
3. parseable session-event timestamps;
4. acquisition file mtime, but only when the caller explicitly supplies it.

Acquisition metadata never outranks evidence emitted by the producer.  Missing
or malformed evidence remains ``None`` rather than becoming an ingest-time
clock value.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel

from polylogue.core.timestamps import parse_timestamp


def timestamp_millis(value: Any) -> int | None:
    parsed = parse_timestamp(value)
    return int(parsed.timestamp() * 1000) if parsed is not None else None


class OccurredAtBounds(Protocol):
    """A message collection that already knows its ``occurred_at_ms`` extrema.

    A disk-backed prepared session answers from its stored rows instead of
    decoding every message into a model only to read one field.
    """

    def occurred_at_bounds(self) -> tuple[int | None, int | None]: ...


def _message_bounds(messages: Iterable[Any]) -> tuple[int | None, int | None]:
    # Looked up on the type: the method is part of the collection's class
    # contract, not an attribute a test double happens to answer.
    if callable(getattr(type(messages), "occurred_at_bounds", None)):
        return cast(OccurredAtBounds, messages).occurred_at_bounds()
    derived_created: int | None = None
    derived_updated: int | None = None
    for message in messages:
        value = getattr(message, "occurred_at_ms", None)
        if value is None:
            continue
        point = int(value)
        derived_created = point if derived_created is None else min(derived_created, point)
        derived_updated = point if derived_updated is None else max(derived_updated, point)
    return derived_created, derived_updated


def _event_bounds(events: Iterable[Any]) -> tuple[int | None, int | None]:
    derived_created: int | None = None
    derived_updated: int | None = None
    for event in events:
        event_point = timestamp_millis(getattr(event, "timestamp", None))
        if event_point is None:
            continue
        derived_created = event_point if derived_created is None else min(derived_created, event_point)
        derived_updated = event_point if derived_updated is None else max(derived_updated, event_point)
    return derived_created, derived_updated


def session_evidence_timestamps(
    session: Any,
    *,
    fallback_timestamp: str | None = None,
) -> tuple[int | None, int | None]:
    """Return authoritative ``(created_ms, updated_ms)`` for a parsed session.

    Session-level fields are authoritative independently: a valid producer
    ``created_at`` does not prevent ``updated_at`` from being derived from the
    timeline, and vice versa.  Message evidence precedes event evidence because
    authored content is the stronger timeline signal.  Event timestamps are
    still valid evidence for event-only sources such as Hermes ATIF.

    The timeline is read only when a side needs it: two valid producer
    endpoints decide the pair on their own, so the messages are not visited.
    """
    raw_created = timestamp_millis(getattr(session, "created_at", None))
    raw_updated = timestamp_millis(getattr(session, "updated_at", None))
    created_provenance = getattr(session, "created_at_provenance", "unknown")
    updated_provenance = getattr(session, "updated_at_provenance", "unknown")
    # A normalized derived value is not producer evidence on a later pass. Keep
    # recomputing it from the current timeline, while retaining it only when no
    # timeline is available at all.
    producer_created = raw_created if created_provenance not in {"derived", "fallback"} else None
    producer_updated = raw_updated if updated_provenance not in {"derived", "fallback"} else None
    derived_created: int | None = None
    derived_updated: int | None = None
    if producer_created is None or producer_updated is None:
        derived_created, derived_updated = _message_bounds(getattr(session, "messages", ()) or ())
        if derived_created is None:
            derived_created, derived_updated = _event_bounds(getattr(session, "session_events", ()) or ())
    created = (
        producer_created
        if producer_created is not None
        else (derived_created if derived_created is not None else raw_created)
    )
    updated = (
        producer_updated
        if producer_updated is not None
        else (derived_updated if derived_updated is not None else raw_updated)
    )
    if created is None and updated is None and fallback_timestamp is not None:
        fallback = timestamp_millis(fallback_timestamp)
        created = fallback
        updated = fallback
    # A one-sided producer observation still describes a closed interval: use
    # the supplied endpoint for the absent side before writing it to storage.
    # This also gives force-upsert SQL one coherent pair to preserve.
    if created is not None and updated is None and producer_created is not None:
        updated = created
    elif updated is not None and created is None and producer_updated is not None:
        created = updated
    # A malformed producer pair must not create an impossible interval. Keep the
    # producer-owned side fixed when only one side was supplied; when both sides
    # came from the same evidence class, preserve the interval's extrema.
    if created is not None and updated is not None and created > updated:
        if producer_created is not None and producer_updated is None:
            updated = created
        elif producer_updated is not None and producer_created is None:
            created = updated
        else:
            created, updated = min(created, updated), max(created, updated)
    return created, updated


_TimestampModel = TypeVar("_TimestampModel", bound=BaseModel)


def normalize_session_timestamps(session: _TimestampModel, *, fallback_timestamp: str | None = None) -> _TimestampModel:
    """Fill and repair parsed-session fields according to the authority ladder."""
    created_ms, updated_ms = session_evidence_timestamps(session, fallback_timestamp=fallback_timestamp)
    raw_created = timestamp_millis(getattr(session, "created_at", None))
    raw_updated = timestamp_millis(getattr(session, "updated_at", None))
    created_kind = (
        "producer"
        if getattr(session, "created_at_provenance", "unknown") not in {"derived", "fallback"}
        and raw_created is not None
        else "derived"
    )
    updated_kind = (
        "producer"
        if getattr(session, "updated_at_provenance", "unknown") not in {"derived", "fallback"}
        and raw_updated is not None
        else "derived"
    )
    if created_ms is not None and raw_created is None:
        created_kind = (
            "fallback"
            if fallback_timestamp is not None and created_ms == timestamp_millis(fallback_timestamp)
            else "derived"
        )
    if updated_ms is not None and raw_updated is None:
        updated_kind = (
            "fallback"
            if fallback_timestamp is not None and updated_ms == timestamp_millis(fallback_timestamp)
            else "derived"
        )
    updates: dict[str, object] = {
        "created_at_provenance": created_kind,
        "updated_at_provenance": updated_kind,
    }
    if raw_created != created_ms and created_ms is not None:
        updates["created_at"] = _iso_from_millis(created_ms)
    if raw_updated != updated_ms and updated_ms is not None:
        updates["updated_at"] = _iso_from_millis(updated_ms)
    return session.model_copy(update=updates)


def producer_timestamp_flags(session: Any) -> tuple[bool, bool]:
    """Return whether each session timestamp is truthful producer evidence."""
    return (
        getattr(session, "created_at_provenance", "unknown") in {"unknown", "producer"}
        and timestamp_millis(getattr(session, "created_at", None)) is not None,
        getattr(session, "updated_at_provenance", "unknown") in {"unknown", "producer"}
        and timestamp_millis(getattr(session, "updated_at", None)) is not None,
    )


def _iso_from_millis(value: int) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(value / 1000, UTC).isoformat()


__all__ = [
    "OccurredAtBounds",
    "normalize_session_timestamps",
    "producer_timestamp_flags",
    "session_evidence_timestamps",
    "timestamp_millis",
]
