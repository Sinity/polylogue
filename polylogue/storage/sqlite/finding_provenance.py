"""Queryable provenance projection over one ``AssertionKind.FINDING`` claim.

Findings are ordinary assertion rows (``polylogue.finding.v1`` value payload,
see ``storage/sqlite/archive_tiers/user_write.py``): a prior review flagged
that finding provenance "must be QUERYABLE, not prose". This module answers
that -- given a finding's assertion id, it re-derives the finding's own
declared evidence refs (``query_ref``, ``result_set_ref``, ``baseline_ref``,
``current_ref``) plus every generic ``evidence_refs`` entry, resolves each
against live user-tier storage, and reports an honest current/stale/unknown
staleness verdict. It does not (yet) carry a code SHA or corpus-datasheet
hash -- those require build-info threading that is out of scope here and
tracked as a named follow-up (see the module docstring in
``polylogue.surfaces.payloads.FindingProvenancePayload``).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

from polylogue.core.enums import AssertionKind
from polylogue.core.refs import ObjectRef
from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionEnvelope, read_assertion_envelope
from polylogue.storage.sqlite.query_objects import get_query, get_result_set

StalenessVerdict = Literal["current", "stale", "unknown"]


@dataclass(frozen=True, slots=True)
class FindingEvidenceResolution:
    ref: str
    resolvable: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class FindingProvenance:
    assertion_id: str
    claim_key: str | None
    target_ref: str
    finding_kind: str | None
    query_ref: str | None
    result_set_ref: str | None
    baseline_ref: str | None
    current_ref: str | None
    detector_ref: str | None
    status: str
    evidence: tuple[FindingEvidenceResolution, ...]
    staleness_verdict: StalenessVerdict
    created_at_ms: int
    updated_at_ms: int


def compute_finding_provenance(conn: sqlite3.Connection, assertion_id: str) -> FindingProvenance | None:
    """Return the provenance projection for one finding, or ``None`` if absent/not-a-finding."""

    envelope = read_assertion_envelope(conn, assertion_id)
    if envelope is None or envelope.kind != AssertionKind.FINDING.value:
        return None
    return _provenance_from_envelope(conn, envelope)


def _provenance_from_envelope(conn: sqlite3.Connection, envelope: ArchiveAssertionEnvelope) -> FindingProvenance:
    value = envelope.value if isinstance(envelope.value, dict) else {}
    query_ref = _str_or_none(value.get("query_ref"))
    result_set_ref = _str_or_none(value.get("result_set_ref"))
    baseline_ref = _str_or_none(value.get("baseline_ref"))
    current_ref = _str_or_none(value.get("current_ref"))
    finding_kind = _str_or_none(value.get("finding_kind"))

    declared_refs = [ref for ref in (query_ref, result_set_ref, baseline_ref, current_ref) if ref is not None]
    all_refs = list(dict.fromkeys([*declared_refs, *envelope.evidence_refs]))
    resolutions = tuple(_resolve_evidence_ref(conn, ref) for ref in all_refs)
    resolved_by_ref = {resolution.ref: resolution.resolvable for resolution in resolutions}

    if not declared_refs:
        staleness: StalenessVerdict = "unknown"
    elif all(resolved_by_ref.get(ref, False) for ref in declared_refs):
        staleness = "current"
    elif any(ref in resolved_by_ref and not resolved_by_ref[ref] for ref in declared_refs):
        staleness = "stale"
    else:
        staleness = "unknown"

    return FindingProvenance(
        assertion_id=envelope.assertion_id,
        claim_key=envelope.key,
        target_ref=envelope.target_ref,
        finding_kind=finding_kind,
        query_ref=query_ref,
        result_set_ref=result_set_ref,
        baseline_ref=baseline_ref,
        current_ref=current_ref,
        detector_ref=envelope.author_ref,
        status=envelope.status,
        evidence=resolutions,
        staleness_verdict=staleness,
        created_at_ms=envelope.created_at_ms,
        updated_at_ms=envelope.updated_at_ms,
    )


def _resolve_evidence_ref(conn: sqlite3.Connection, ref: str) -> FindingEvidenceResolution:
    try:
        parsed = ObjectRef.parse(ref)
    except ValueError:
        return FindingEvidenceResolution(ref=ref, resolvable=False, reason="unparseable ref")
    if parsed.kind == "query":
        found = get_query(conn, parsed.object_id) is not None
        return FindingEvidenceResolution(ref=ref, resolvable=found, reason=None if found else "query not found")
    if parsed.kind == "result-set":
        found = get_result_set(conn, parsed.object_id) is not None
        return FindingEvidenceResolution(ref=ref, resolvable=found, reason=None if found else "result set not found")
    if parsed.kind == "assertion":
        found = read_assertion_envelope(conn, parsed.object_id) is not None
        return FindingEvidenceResolution(ref=ref, resolvable=found, reason=None if found else "assertion not found")
    return FindingEvidenceResolution(
        ref=ref,
        resolvable=False,
        reason=f"resolution not implemented for ref kind {parsed.kind!r}",
    )


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = [
    "FindingEvidenceResolution",
    "FindingProvenance",
    "StalenessVerdict",
    "compute_finding_provenance",
]
