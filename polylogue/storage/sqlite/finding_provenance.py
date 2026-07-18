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
``polylogue.surfaces.payloads.FindingProvenancePayload``).  This legacy
current/stale helper is not the 37t.14 evidence-integrity authority and is not
used to compute public-claim support.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal, cast

from polylogue.core.enums import AssertionKind
from polylogue.core.refs import ObjectRef
from polylogue.insights.measurement.public_claims import (
    PublicClaimDisclosure,
    PublicClaimPresetName,
    PublicFindingInput,
)
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ArchiveAssertionEnvelope,
    list_assertion_claims,
    read_assertion_envelope,
    read_latest_candidate_judgment,
)
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


def list_public_finding_inputs(conn: sqlite3.Connection) -> tuple[PublicFindingInput, ...]:
    """Adapt public-declared FINDING rows to the storage-neutral projection input.

    This function reads lifecycle and judgment facts only.  It never resolves
    evidence refs or computes support, cycle, staleness, frame, or privacy
    ancestry verdicts; those enter later through the 37t.14 provider seam.
    """

    findings = list_assertion_claims(conn, kinds=(AssertionKind.FINDING,), statuses=None)
    return tuple(_public_finding_input(conn, finding) for finding in findings if _has_public_claim(finding))


def _has_public_claim(envelope: ArchiveAssertionEnvelope) -> bool:
    return isinstance(envelope.value, dict) and isinstance(envelope.value.get("public_claim"), dict)


def _public_finding_input(conn: sqlite3.Connection, envelope: ArchiveAssertionEnvelope) -> PublicFindingInput:
    value = envelope.value if isinstance(envelope.value, dict) else {}
    declaration = value.get("public_claim")
    if not isinstance(declaration, dict):
        raise ValueError(f"finding {envelope.assertion_id!r} has no public_claim declaration")

    publication = _required_str(declaration.get("publication"), field="publication", assertion_id=envelope.assertion_id)
    scope = _required_str(declaration.get("scope"), field="scope", assertion_id=envelope.assertion_id)
    caveat = _required_str(declaration.get("caveat"), field="caveat", assertion_id=envelope.assertion_id)
    disclosure_value = _required_str(
        declaration.get("disclosure"), field="disclosure", assertion_id=envelope.assertion_id
    )
    if disclosure_value not in {"public", "held_private"}:
        raise ValueError(f"finding {envelope.assertion_id!r} has invalid public disclosure {disclosure_value!r}")
    disclosure = cast(PublicClaimDisclosure, disclosure_value)

    public_refs = _required_str_list(
        declaration.get("public_evidence_refs"),
        field="public_evidence_refs",
        assertion_id=envelope.assertion_id,
    )
    preset_values = _required_str_list(
        declaration.get("presets"),
        field="presets",
        assertion_id=envelope.assertion_id,
    )
    try:
        presets = tuple(PublicClaimPresetName(item) for item in preset_values)
    except ValueError as exc:
        raise ValueError(f"finding {envelope.assertion_id!r} has an unknown public-claim preset") from exc

    statistic = value.get("statistic")
    if not isinstance(statistic, dict):
        raise ValueError(f"finding {envelope.assertion_id!r} statistic must be a mapping")

    judgment_ref, judgment_decision = _finding_judgment(conn, envelope)
    return PublicFindingInput(
        assertion_ref=f"assertion:{envelope.assertion_id}",
        claim_key=_required_str(envelope.key, field="claim key", assertion_id=envelope.assertion_id),
        publication=publication,
        scope=scope,
        caveat=caveat,
        public_evidence_refs=public_refs,
        presets=presets,
        disclosure=disclosure,
        statistic=statistic,
        finding_epoch=_str_or_none(value.get("source_epoch")),
        evaluation_ref=_str_or_none(value.get("evaluation_ref")),
        frame_ref=_str_or_none(value.get("frame_ref")),
        assertion_status=envelope.status,
        assertion_visibility=envelope.visibility,
        author_kind=_required_str(envelope.author_kind, field="author kind", assertion_id=envelope.assertion_id),
        judgment_ref=judgment_ref,
        judgment_decision=judgment_decision,
        supersedes=tuple(envelope.supersedes),
        updated_at_ms=envelope.updated_at_ms,
    )


def _finding_judgment(conn: sqlite3.Connection, envelope: ArchiveAssertionEnvelope) -> tuple[str | None, str | None]:
    candidate_ids: list[str] = []
    if envelope.status.value in {"candidate", "accepted", "rejected", "deferred", "superseded"}:
        candidate_ids.append(envelope.assertion_id)
    candidate_ids.extend(ref.removeprefix("assertion:") for ref in envelope.supersedes if ref.startswith("assertion:"))

    resulting_ref = f"assertion:{envelope.assertion_id}"
    for candidate_id in candidate_ids:
        judgment = read_latest_candidate_judgment(conn, candidate_id)
        if judgment is None or not isinstance(judgment.value, dict):
            continue
        decision = _str_or_none(judgment.value.get("decision"))
        declared_result = _str_or_none(judgment.value.get("resulting_assertion_ref"))
        if envelope.status.value == "active" and declared_result != resulting_ref:
            continue
        return f"assertion:{judgment.assertion_id}", decision
    return None, None


def _required_str(value: object, *, field: str, assertion_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"finding {assertion_id!r} public {field} must be a non-empty string")
    return value.strip()


def _required_str_list(value: object, *, field: str, assertion_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"finding {assertion_id!r} public {field} must be a non-empty string list")
    return tuple(dict.fromkeys(item.strip() for item in value))


__all__ = [
    "FindingEvidenceResolution",
    "FindingProvenance",
    "StalenessVerdict",
    "compute_finding_provenance",
    "list_public_finding_inputs",
]
