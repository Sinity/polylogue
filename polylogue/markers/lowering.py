"""Lower marker evidence through the existing assertion service."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable

from polylogue.core.enums import AssertionStatus, AssertionVisibility
from polylogue.markers.models import MarkerCandidate, marker_provenance
from polylogue.markers.registry import MARKER_REGISTRY, MarkerRegistry
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion


def candidates_for_block(
    message_id: str, block_id: str, text: str, *, registry: MarkerRegistry = MARKER_REGISTRY
) -> tuple[MarkerCandidate, ...]:
    from polylogue.markers.parser import parse_markers

    provenance = marker_provenance(message_id, block_id)
    result: list[MarkerCandidate] = []
    for match in parse_markers(text, registry=registry):
        spec = registry.get(match.kind)
        result.append(
            MarkerCandidate(match, provenance, None if spec is None or match.malformed else spec.lowering_target)
        )
    return tuple(result)


def lower_markers(
    conn: sqlite3.Connection, candidates: Iterable[MarkerCandidate], *, now_ms: int | None = None
) -> tuple[str, ...]:
    """Persist candidates as private, agent-authority assertions."""
    ids: list[str] = []
    for candidate in candidates:
        if candidate.assertion_kind is None:
            continue
        match = candidate.match
        digest = hashlib.sha256(
            "\x1f".join((match.kind, match.raw_text, *candidate.evidence_refs)).encode()
        ).hexdigest()[:32]
        assertion_id = f"marker-{digest}"
        upsert_assertion(
            conn,
            assertion_id=assertion_id,
            target_ref=candidate.evidence_refs[0],
            kind=candidate.assertion_kind,
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
        ids.append(assertion_id)
    return tuple(ids)


__all__ = ["candidates_for_block", "lower_markers"]
