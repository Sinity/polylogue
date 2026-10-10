"""Lower marker evidence through the existing assertion service."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable, Iterator

from polylogue.core.enums import AssertionStatus, AssertionVisibility
from polylogue.markers.models import MarkerCandidate, marker_provenance
from polylogue.markers.registry import MARKER_REGISTRY, MarkerRegistry
from polylogue.storage.sqlite.archive_tiers.user_write import (
    AssertionWriteBatch,
    assertion_write_batch,
    mark_assertion_status,
    record_retired_marker_assertion,
)


def candidates_for_block(
    message_id: str, block_id: str, text: str, *, registry: MarkerRegistry = MARKER_REGISTRY
) -> tuple[MarkerCandidate, ...]:
    return tuple(iter_candidates_for_block(message_id, block_id, text, registry=registry))


def iter_candidates_for_block(
    message_id: str, block_id: str, text: str, *, registry: MarkerRegistry = MARKER_REGISTRY
) -> Iterator[MarkerCandidate]:
    """Yield block candidates without collecting every marker in one block."""
    from polylogue.markers.parser import iter_parse_markers

    provenance = marker_provenance(message_id, block_id)
    for match in iter_parse_markers(text, registry=registry):
        spec = registry.get(match.kind)
        yield MarkerCandidate(match, provenance, None if spec is None or match.malformed else spec.lowering_target)


def assertion_id_for_marker(candidate: MarkerCandidate) -> str | None:
    """Return the stable assertion identity for one lowerable marker."""
    if candidate.assertion_kind is None:
        return None
    match = candidate.match
    digest = hashlib.sha256("\x1f".join((match.kind, match.raw_text, *candidate.evidence_refs)).encode()).hexdigest()[
        :32
    ]
    return f"marker-{digest}"


def lower_markers(
    conn: sqlite3.Connection, candidates: Iterable[MarkerCandidate], *, now_ms: int | None = None
) -> tuple[str, ...]:
    """Persist candidates as private, agent-authority assertions.

    A marker's deterministic id makes replay idempotent, but it is not a
    license to replace a human's assertion or judgment at that id.  Existing
    non-agent rows and every terminal agent judgment are therefore preserved.
    """
    with assertion_write_batch(conn) as writer:
        return tuple(iter_lower_markers(writer, candidates, now_ms=now_ms))


def iter_lower_markers(
    writer: AssertionWriteBatch, candidates: Iterable[MarkerCandidate], *, now_ms: int | None = None
) -> Iterator[str]:
    """Lower candidates incrementally without retaining every assertion ID."""
    for candidate in candidates:
        conn = writer.connection
        assertion_kind = candidate.assertion_kind
        if assertion_kind is None:
            continue
        assertion_id = assertion_id_for_marker(candidate)
        if assertion_id is None:
            continue
        if conn.execute("SELECT 1 FROM retired_marker_assertions WHERE assertion_id = ?", (assertion_id,)).fetchone():
            # A later carrier re-owned this evidence before this one arrived.
            continue
        existing = conn.execute(
            "SELECT author_kind, status FROM assertions WHERE assertion_id = ?",
            (assertion_id,),
        ).fetchone()
        if existing is not None and (
            str(existing[0]) != "agent"
            or str(existing[1])
            in {
                AssertionStatus.ACCEPTED.value,
                AssertionStatus.REJECTED.value,
                AssertionStatus.DEFERRED.value,
                AssertionStatus.SUPERSEDED.value,
                AssertionStatus.DELETED.value,
            }
        ):
            yield assertion_id
            continue
        match = candidate.match
        writer.upsert(
            assertion_id=assertion_id,
            target_ref=candidate.evidence_refs[0],
            kind=assertion_kind,
            key=match.kind,
            value={"marker_kind": match.kind, "arguments": dict(match.arguments)},
            body_text=match.body,
            author_ref=candidate.evidence_refs[0],
            author_kind="agent",
            evidence_refs=candidate.evidence_refs,
            status=AssertionStatus.CANDIDATE,
            visibility=AssertionVisibility.PRIVATE,
            now_ms=now_ms,
        )
        yield assertion_id


def retire_marker_assertions(
    conn: sqlite3.Connection, assertion_ids: Iterable[str], *, now_ms: int | None = None
) -> tuple[str, ...]:
    """Record re-owned marker ids and supersede their live agent candidates.

    The retirement is durable, so delivery order does not matter: a carrier
    lowered later skips a retired id in :func:`lower_markers`. Only an
    untouched agent ``candidate`` is superseded; a human's assertion or any
    judgment already made at that id is preserved.
    """
    return tuple(iter_retire_marker_assertions(conn, assertion_ids, now_ms=now_ms))


def iter_retire_marker_assertions(
    conn: sqlite3.Connection, assertion_ids: Iterable[str], *, now_ms: int | None = None
) -> Iterator[str]:
    """Record retirements incrementally without collecting changed IDs."""
    from polylogue.storage.sqlite.archive_tiers.user_write import _now_ms

    timestamp = _now_ms() if now_ms is None else now_ms
    for assertion_id in assertion_ids:
        record_retired_marker_assertion(conn, assertion_id, now_ms=timestamp)
        existing = conn.execute(
            "SELECT author_kind, status FROM assertions WHERE assertion_id = ?",
            (assertion_id,),
        ).fetchone()
        if existing is None or str(existing[0]) != "agent":
            continue
        if existing[1] is not None and str(existing[1]) != AssertionStatus.CANDIDATE.value:
            continue
        if mark_assertion_status(conn, assertion_id, AssertionStatus.SUPERSEDED, now_ms=timestamp):
            yield assertion_id


__all__ = [
    "assertion_id_for_marker",
    "candidates_for_block",
    "iter_lower_markers",
    "iter_retire_marker_assertions",
    "lower_markers",
    "retire_marker_assertions",
]
