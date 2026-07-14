"""Minimal user-tier write/read helpers.

These functions are intentionally narrow: they provide deterministic upsert
helpers and compact envelope readers for user-only tables in ``user.py``.

Writer module: user.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from polylogue.core.assertions import (
    AssertionContextPolicy,
    AssertionStaleness,
    AssertionValue,
    constrain_assertion_context_policy,
)
from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
from polylogue.core.json import JSONValue
from polylogue.core.refs import ObjectRef, normalize_object_ref_text, normalize_public_ref_text

if TYPE_CHECKING:
    from polylogue.insights.judgment.types import ComparativeJudgment
    from polylogue.insights.pathology import PathologyFinding
    from polylogue.insights.transforms import DecisionCandidate, SessionDigest, TransformRawRef


ASSERTION_DEFAULT_STATUS: Final[AssertionStatus] = AssertionStatus.ACTIVE
ASSERTION_DEFAULT_VISIBILITY: Final[AssertionVisibility] = AssertionVisibility.PRIVATE
ASSERTION_DEFAULT_AUTHOR_KIND: Final = "user"
ASSERTION_DEFAULT_AUTHOR_REF: Final = "user:local"
ASSERTION_DEFAULT_CONTEXT_POLICY: Final[dict[str, JSONValue]] = AssertionContextPolicy.default().as_json_document()

#: Candidate-lifecycle terminal outcomes (set only by ``judge_assertion_candidate``
#: via ``mark_assertion_status``, never by ``upsert_assertion`` itself). A
#: non-user-authored write that lands on an assertion_id already in one of
#: these states must not resurrect it back to candidate/active (37t.15).
_ASSERTION_TERMINAL_JUDGED_STATUSES: Final[frozenset[AssertionStatus]] = frozenset(
    {
        AssertionStatus.ACCEPTED,
        AssertionStatus.REJECTED,
        AssertionStatus.DEFERRED,
        AssertionStatus.SUPERSEDED,
        AssertionStatus.DELETED,
    }
)

#: Non-injected candidate context policy every non-user-authored write is
#: coerced to unless it is already terminal-judged (37t.15). Matches the
#: shape existing candidate writers (transform/pathology) already set by hand.
_ASSERTION_AGENT_CANDIDATE_CONTEXT_POLICY: Final[dict[str, JSONValue]] = {"inject": False, "promotion_required": True}


def _default_context_policy() -> dict[str, JSONValue]:
    return dict(ASSERTION_DEFAULT_CONTEXT_POLICY)


def _normalize_assertion_kind(kind: str | AssertionKind) -> AssertionKind:
    return AssertionKind.from_string(kind)


def _normalize_assertion_status(status: str | AssertionStatus | None) -> AssertionStatus:
    return AssertionStatus.from_string(status) if status is not None else ASSERTION_DEFAULT_STATUS


def _normalize_assertion_visibility(visibility: str | AssertionVisibility | None) -> AssertionVisibility:
    return AssertionVisibility.from_string(visibility) if visibility is not None else ASSERTION_DEFAULT_VISIBILITY


def _normalize_assertion_author_kind(author_kind: str | None) -> str:
    return author_kind if author_kind is not None else ASSERTION_DEFAULT_AUTHOR_KIND


def _normalize_assertion_author_ref(author_ref: str | None) -> str:
    return author_ref if author_ref is not None else ASSERTION_DEFAULT_AUTHOR_REF


def _normalize_assertion_value(value: object | None) -> AssertionValue:
    return AssertionValue.from_raw(value)


def _normalize_assertion_staleness(staleness: Mapping[str, object] | None) -> AssertionStaleness | None:
    return AssertionStaleness.from_raw(staleness)


def _normalize_assertion_context_policy(
    context_policy: Mapping[str, object] | AssertionContextPolicy | None,
) -> AssertionContextPolicy:
    """Return the stored assertion context policy.

    Assertions are durable user-tier state first. They must not flow into future
    context unless a caller opts in explicitly, so even caller-supplied policies
    get an explicit ``inject`` key when omitted. This keeps generic overlay
    writes private/no-inject while preserving explicit promotion/candidate
    policy such as ``promotion_required``.
    """

    if isinstance(context_policy, AssertionContextPolicy):
        return context_policy
    return AssertionContextPolicy.from_raw(context_policy)


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _read_payload_text(value: str | None) -> dict[str, object]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return dict(parsed)
    return {}


def _json_dumps(payload: dict[str, object] | None) -> str:
    if payload is None:
        return "{}"
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dumps_optional(value: object | None) -> str | None:
    """Serialize an arbitrary JSON-able value, or ``None`` when value is ``None``."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads_optional(value: object | None) -> object | None:
    """Parse a stored JSON column back to a Python value, tolerating malformed text."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed


def _loads_str_list(value: object | None) -> list[str]:
    parsed = _loads_optional(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _loads_dict_optional(value: object | None) -> dict[str, object] | None:
    parsed = _loads_optional(value)
    if isinstance(parsed, dict):
        return dict(parsed)
    return None


def _assertion_value_document(value: JSONValue | None) -> dict[str, object]:
    payload: dict[str, object] = {}
    if isinstance(value, dict):
        payload.update(value)
    return payload


def _deterministic_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return f"{prefix}:{digest.hexdigest()}"


def assertion_id_for_mark(target_type: str, target_id: str, mark_type: str) -> str:
    """Return the mirrored assertion id for one legacy mark row."""
    mark_id = _deterministic_id("mark", target_type, target_id, mark_type)
    return _deterministic_id(f"assertion-{AssertionKind.MARK}", mark_id)


def assertion_id_for_suppression(session_id: str) -> str:
    """Return the mirrored assertion id for one legacy suppression row."""
    return _deterministic_id(f"assertion-{AssertionKind.SUPPRESSION}", session_id)


def assertion_id_for_annotation(annotation_id: str) -> str:
    """Return the mirrored assertion id for one legacy annotation row."""
    return _deterministic_id(f"assertion-{AssertionKind.ANNOTATION}", annotation_id)


def assertion_id_for_correction(correction_id: str) -> str:
    """Return the mirrored assertion id for one legacy correction row."""
    return _deterministic_id(f"assertion-{AssertionKind.CORRECTION}", correction_id)


def correction_id_for(target_type: str, target_id: str, correction_type: str) -> str:
    """Return the stable assertion-backed correction id."""

    return _deterministic_id("correction", target_type, target_id, correction_type)


def assertion_id_for_saved_view(view_id: str) -> str:
    """Return the mirrored assertion id for one legacy saved-view row."""
    return _deterministic_id(f"assertion-{AssertionKind.SAVED_QUERY}", view_id)


def assertion_id_for_recall_pack(recall_pack_id: str) -> str:
    """Return the mirrored assertion id for one legacy recall-pack row."""
    return _deterministic_id(f"assertion-{AssertionKind.RECALL_PACK}", recall_pack_id)


def assertion_id_for_workspace(workspace_id: str) -> str:
    """Return the mirrored assertion id for one legacy workspace row."""
    return _deterministic_id(f"assertion-{AssertionKind.WORKSPACE_NOTE}", workspace_id)


def assertion_id_for_blackboard_note(note_id: str) -> str:
    """Return the mirrored assertion id for one legacy blackboard note row."""
    return _deterministic_id(f"assertion-{AssertionKind.NOTE}", note_id)


def assertion_id_for_session_tag(session_id: str, tag: str, tag_source: str) -> str:
    """Return the mirrored assertion id for one normalized session tag."""
    normalized_tag = tag.strip().lower()
    return _deterministic_id(f"assertion-{AssertionKind.TAG}", session_id, normalized_tag, tag_source)


def assertion_id_for_session_metadata(session_id: str, key: str) -> str:
    """Return the assertion id for one user metadata key."""
    normalized_key = key.strip()
    return _deterministic_id(f"assertion-{AssertionKind.METADATA}", session_id, normalized_key)


def assertion_id_for_transform_candidate(
    *,
    session_id: str,
    transform_id: str,
    transform_version: int,
    candidate_index: int,
    candidate_kind: str,
    candidate_text: str,
    evidence_refs: Sequence[str],
) -> str:
    """Return the assertion id for one deterministic transform candidate."""

    return _deterministic_id(
        f"assertion-{AssertionKind.TRANSFORM_CANDIDATE}",
        session_id,
        transform_id,
        str(transform_version),
        str(candidate_index),
        candidate_kind,
        candidate_text,
        *evidence_refs,
    )


def assertion_id_for_pathology_finding(
    *,
    session_id: str,
    finding_kind: str,
    detector_version: int,
    finding_detail: str,
    evidence_refs: Sequence[str],
) -> str:
    """Return the assertion id for one deterministic pathology finding."""

    return _deterministic_id(
        f"assertion-{AssertionKind.PATHOLOGY}",
        session_id,
        finding_kind,
        str(detector_version),
        finding_detail,
        *evidence_refs,
    )


def assertion_id_for_finding(
    *,
    claim_key: str,
    target_ref: str,
    value: Mapping[str, object],
    evidence_refs: Sequence[str],
    detector_ref: str,
) -> str:
    """Return the deterministic id for one ``polylogue.finding.v1`` claim."""

    return _deterministic_id(
        f"assertion-{AssertionKind.FINDING}",
        claim_key,
        target_ref,
        _json_dumps(dict(value)),
        *sorted(evidence_refs),
        detector_ref,
    )


def assertion_id_for_candidate_judgment(candidate_assertion_id: str, decision: str) -> str:
    return _deterministic_id(f"assertion-{AssertionKind.JUDGMENT}", candidate_assertion_id, decision)


def assertion_id_for_promoted_candidate(candidate_assertion_id: str) -> str:
    return _deterministic_id("assertion-promoted", candidate_assertion_id)


def _target_ref(target_type: str | None, target_id: str | None) -> str | None:
    """Compose and validate an assertion ``target_ref`` from ``(type, id)``.

    Returns ``None`` when no archive target is attached (e.g. an unscoped
    blackboard note); callers fall back to a typed assertion ref for the
    NOT NULL ``target_ref`` column.
    """
    if target_type and target_id:
        return normalize_object_ref_text(f"{target_type}:{target_id}")
    return None


@dataclass(frozen=True, slots=True)
class ArchiveSuppressionEnvelope:
    session_id: str
    reason: str | None
    mode: str
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class ArchiveMarkEnvelope:
    mark_id: str
    target_type: str
    target_id: str
    mark_type: str
    label: str | None
    created_at_ms: int
    updated_at_ms: int
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class ArchiveAnnotationEnvelope:
    annotation_id: str
    target_type: str
    target_id: str
    body: str
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class ArchiveCorrectionEnvelope:
    correction_id: str
    target_type: str
    target_id: str
    correction_type: str
    payload: dict[str, object]
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class ArchiveSavedViewEnvelope:
    view_id: str
    name: str
    query: dict[str, object]
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class ArchiveRecallPackEnvelope:
    recall_pack_id: str
    name: str
    payload: dict[str, object]
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class ArchiveWorkspaceEnvelope:
    workspace_id: str
    name: str
    settings: dict[str, object]
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class ArchiveBlackboardNoteEnvelope:
    note_id: str
    target_type: str | None
    target_id: str | None
    body: str
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class ArchiveAssertionEnvelope:
    assertion_id: str
    scope_ref: str | None
    target_ref: str
    key: str | None
    kind: AssertionKind
    value: JSONValue | None
    body_text: str | None
    author_ref: str | None
    author_kind: str | None
    evidence_refs: list[str]
    status: AssertionStatus
    visibility: AssertionVisibility
    confidence: float | None
    staleness: dict[str, JSONValue] | None
    context_policy: dict[str, JSONValue]
    supersedes: list[str]
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class ArchiveAssertionJudgmentEnvelope:
    candidate: ArchiveAssertionEnvelope
    judgment: ArchiveAssertionEnvelope
    resulting_assertion: ArchiveAssertionEnvelope | None = None
    outcome: str = "applied"


@dataclass(frozen=True, slots=True)
class ArchiveAssertionBulkJudgmentItemEnvelope:
    candidate_ref: str
    decision: str
    reason: str | None = None
    inject: bool = False
    actor_ref: str = ASSERTION_DEFAULT_AUTHOR_REF
    replacement_kind: str | AssertionKind | None = None
    replacement_body_text: str | None = None
    replacement_value: object | None = None


@dataclass(frozen=True, slots=True)
class ArchiveAssertionBulkJudgmentResultEnvelope:
    candidate_ref: str
    outcome: str
    result: ArchiveAssertionJudgmentEnvelope | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveAssertionBulkJudgmentEnvelope:
    items: tuple[ArchiveAssertionBulkJudgmentResultEnvelope, ...]

    @property
    def applied_count(self) -> int:
        return sum(item.outcome == "applied" for item in self.items)

    @property
    def idempotent_count(self) -> int:
        return sum(item.outcome == "idempotent" for item in self.items)

    @property
    def failed_count(self) -> int:
        return sum(item.outcome == "failed" for item in self.items)


@dataclass(frozen=True, slots=True)
class ArchiveAssertionCandidateReviewEnvelope:
    candidate: ArchiveAssertionEnvelope
    latest_judgment: ArchiveAssertionEnvelope | None = None


@dataclass(frozen=True, slots=True)
class FindingAssertion:
    """One detector-produced claim stored through the assertion lifecycle.

    Query and result-set refs deliberately remain opaque at this layer. The
    rxdo substrate owns their grammar; this writer preserves them as public
    evidence refs so future provenance readers can resolve ancestry.
    """

    claim_key: str
    target_ref: str
    body_text: str
    finding_kind: str
    statistic: Mapping[str, JSONValue]
    n: int
    query_ref: str
    result_set_ref: str
    detector_ref: str
    baseline_ref: str | None = None
    current_ref: str | None = None
    expected: Mapping[str, JSONValue] | None = None
    evidence_refs: Sequence[str] = ()
    scope_ref: str | None = None
    confidence: float | None = None


def upsert_suppression(
    conn: sqlite3.Connection,
    session_id: str,
    reason: str | None,
    *,
    mode: str = "hide",
    now_ms: int | None = None,
) -> ArchiveSuppressionEnvelope:
    """Upsert one suppression assertion by ``session_id``."""
    conn.execute("PRAGMA foreign_keys = ON")
    timestamp = now_ms if now_ms is not None else _now_ms()
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_suppression(session_id),
        target_ref=f"session:{session_id}",
        kind=AssertionKind.SUPPRESSION,
        value={"mode": mode},
        body_text=reason,
        author_kind="user",
        now_ms=timestamp,
    )
    return read_archive_suppression_envelope(conn, session_id)


def upsert_mark(
    conn: sqlite3.Connection,
    target_type: str,
    target_id: str,
    mark_type: str,
    *,
    label: str | None = None,
    metadata: dict[str, object] | None = None,
    now_ms: int | None = None,
) -> ArchiveMarkEnvelope:
    """Insert-or-update one mark assertion with deterministic ``mark_id``."""
    conn.execute("PRAGMA foreign_keys = ON")
    timestamp = now_ms if now_ms is not None else _now_ms()
    mark_id = _deterministic_id("mark", target_type, target_id, mark_type)
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_mark(target_type, target_id, mark_type),
        target_ref=f"{target_type}:{target_id}",
        kind=AssertionKind.MARK,
        key=mark_type,
        value=metadata if metadata else None,
        body_text=label,
        author_kind="user",
        now_ms=timestamp,
    )
    return read_archive_mark_envelope(conn, mark_id)


def upsert_session_tag_assertion(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    tag: str,
    tag_source: str,
    method: str | None = None,
    author_ref: str | None = None,
    author_kind: str | None = None,
    confidence: float | None = None,
    evidence: dict[str, object] | None = None,
    now_ms: int | None = None,
) -> ArchiveAssertionEnvelope | None:
    """Mirror one user session tag as a first-class tag assertion."""
    if tag_source != "user":
        return None
    normalized_tag = tag.strip().lower()
    if not normalized_tag:
        raise ValueError("tag cannot be empty")
    if len(normalized_tag) > 200:
        raise ValueError("tag exceeds maximum length of 200 characters")
    timestamp = now_ms if now_ms is not None else _now_ms()
    value: dict[str, object] = {"tag_source": tag_source}
    if method is not None:
        value["method"] = method
    if evidence is not None:
        value["evidence"] = evidence
    return upsert_assertion(
        conn,
        assertion_id=assertion_id_for_session_tag(session_id, normalized_tag, tag_source),
        target_ref=f"session:{session_id}",
        kind=AssertionKind.TAG,
        key=normalized_tag,
        value=value,
        body_text=normalized_tag,
        author_ref=author_ref,
        author_kind=author_kind,
        confidence=confidence,
        now_ms=timestamp,
        # Session tags are categorization, not epistemic claims -- no
        # judgment-queue path exists for TAG, and add_user_tags' existing-row
        # short-circuit would strand an agent-tagged session as a permanently
        # unreachable candidate (37t.15 chokepoint carve-out; see
        # upsert_assertion's require_promotion docstring).
        require_promotion=False,
    )


def upsert_session_metadata_assertion(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    key: str,
    value: object,
    now_ms: int | None = None,
) -> ArchiveAssertionEnvelope:
    """Upsert one user metadata key as a first-class metadata assertion."""
    normalized_key = key.strip()
    if not normalized_key:
        raise ValueError("metadata key cannot be empty")
    timestamp = now_ms if now_ms is not None else _now_ms()
    return upsert_assertion(
        conn,
        assertion_id=assertion_id_for_session_metadata(session_id, normalized_key),
        target_ref=f"session:{session_id}",
        kind=AssertionKind.METADATA,
        key=normalized_key,
        value=value,
        author_kind="user",
        now_ms=timestamp,
    )


def upsert_annotation(
    conn: sqlite3.Connection,
    target_type: str,
    target_id: str,
    body: str,
    *,
    annotation_id: str | None = None,
    now_ms: int | None = None,
) -> ArchiveAnnotationEnvelope:
    """Insert-or-update one annotation assertion with deterministic identity."""
    conn.execute("PRAGMA foreign_keys = ON")
    timestamp = now_ms if now_ms is not None else _now_ms()
    resolved_id = annotation_id or _deterministic_id("annotation", target_type, target_id, body)
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_annotation(resolved_id),
        target_ref=f"{target_type}:{target_id}",
        kind=AssertionKind.ANNOTATION,
        key=resolved_id,
        body_text=body,
        author_kind="user",
        now_ms=timestamp,
    )
    return read_archive_annotation_envelope(conn, resolved_id)


def upsert_correction(
    conn: sqlite3.Connection,
    target_type: str,
    target_id: str,
    correction_type: str,
    payload: dict[str, object],
    *,
    author_ref: str | None = None,
    author_kind: str | None = None,
    now_ms: int | None = None,
) -> ArchiveCorrectionEnvelope:
    """Insert-or-update one correction assertion with deterministic id."""
    conn.execute("PRAGMA foreign_keys = ON")
    timestamp = now_ms if now_ms is not None else _now_ms()
    correction_id = correction_id_for(target_type, target_id, correction_type)
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_correction(correction_id),
        target_ref=f"{target_type}:{target_id}",
        kind=AssertionKind.CORRECTION,
        key=correction_type,
        value=payload,
        author_ref=author_ref,
        author_kind=author_kind,
        now_ms=timestamp,
    )
    return read_archive_correction_envelope(conn, correction_id)


def upsert_saved_view(
    conn: sqlite3.Connection,
    name: str,
    query: dict[str, object],
    *,
    view_id: str | None = None,
    now_ms: int | None = None,
) -> ArchiveSavedViewEnvelope:
    """Insert-or-update a saved query assertion keyed by name."""
    timestamp = now_ms if now_ms is not None else _now_ms()
    resolved_id = view_id or _deterministic_id("view", name)
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_saved_view(resolved_id),
        target_ref=f"saved_view:{resolved_id}",
        kind=AssertionKind.SAVED_QUERY,
        key=name,
        value=query,
        author_kind="user",
        now_ms=timestamp,
    )
    return read_archive_saved_view_envelope(conn, name)


def upsert_recall_pack(
    conn: sqlite3.Connection,
    name: str,
    payload: dict[str, object],
    *,
    recall_pack_id: str | None = None,
    now_ms: int | None = None,
) -> ArchiveRecallPackEnvelope:
    """Insert-or-update one recall-pack assertion."""
    timestamp = now_ms if now_ms is not None else _now_ms()
    resolved_id = recall_pack_id or _deterministic_id("recall-pack", name, _json_dumps(payload))
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_recall_pack(resolved_id),
        target_ref=f"recall_pack:{resolved_id}",
        kind=AssertionKind.RECALL_PACK,
        key=name,
        value=payload,
        body_text=name,
        author_kind="user",
        now_ms=timestamp,
    )
    return read_archive_recall_pack_envelope(conn, resolved_id)


def upsert_workspace(
    conn: sqlite3.Connection,
    name: str,
    settings: dict[str, object],
    *,
    workspace_id: str | None = None,
    now_ms: int | None = None,
) -> ArchiveWorkspaceEnvelope:
    """Insert-or-update one workspace assertion keyed by name."""
    timestamp = now_ms if now_ms is not None else _now_ms()
    resolved_id = workspace_id or _deterministic_id("workspace", name)
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_workspace(resolved_id),
        target_ref=f"workspace:{resolved_id}",
        kind=AssertionKind.WORKSPACE_NOTE,
        key=name,
        value=settings,
        body_text=name,
        author_kind="user",
        now_ms=timestamp,
    )
    return read_archive_workspace_envelope(conn, name)


def upsert_blackboard_note(
    conn: sqlite3.Connection,
    body: str,
    *,
    target_type: str | None = None,
    target_id: str | None = None,
    note_id: str | None = None,
    author_ref: str | None = None,
    author_kind: str = "user",
    evidence_refs: Sequence[str] | None = None,
    staleness: Mapping[str, object] | None = None,
    context_policy: Mapping[str, object] | AssertionContextPolicy | None = None,
    now_ms: int | None = None,
) -> ArchiveBlackboardNoteEnvelope:
    """Insert-or-update one blackboard note assertion."""
    timestamp = now_ms if now_ms is not None else _now_ms()
    resolved_id = note_id or _deterministic_id("blackboard-note", target_type or "", target_id or "", body)
    upsert_assertion(
        conn,
        assertion_id=assertion_id_for_blackboard_note(resolved_id),
        target_ref=_target_ref(target_type, target_id) or ObjectRef(kind="assertion", object_id=resolved_id).format(),
        kind=AssertionKind.NOTE,
        key=resolved_id,
        body_text=body,
        author_ref=author_ref,
        author_kind=author_kind,
        evidence_refs=evidence_refs,
        staleness=staleness,
        context_policy=context_policy,
        now_ms=timestamp,
    )
    return read_archive_blackboard_note_envelope(conn, resolved_id)


def read_archive_suppression_envelope(conn: sqlite3.Connection, session_id: str) -> ArchiveSuppressionEnvelope:
    assertion = read_assertion_envelope(conn, assertion_id_for_suppression(session_id))
    if assertion is None or assertion.status == AssertionStatus.DELETED:
        raise KeyError(session_id)
    assertion_value = _assertion_value_document(assertion.value)
    return ArchiveSuppressionEnvelope(
        session_id=session_id,
        reason=assertion.body_text,
        mode=str(assertion_value.get("mode") or "hide"),
        created_at_ms=assertion.created_at_ms,
        updated_at_ms=assertion.updated_at_ms,
    )


def read_archive_mark_envelope(conn: sqlite3.Connection, mark_id: str) -> ArchiveMarkEnvelope:
    assertion = read_assertion_envelope(conn, _deterministic_id(f"assertion-{AssertionKind.MARK}", mark_id))
    if assertion is None or assertion.status == AssertionStatus.DELETED:
        raise KeyError(mark_id)
    target_type, target_id = _split_target_ref(assertion.target_ref)
    assertion_value = _assertion_value_document(assertion.value)
    return ArchiveMarkEnvelope(
        mark_id=mark_id,
        target_type=target_type,
        target_id=target_id,
        mark_type=str(assertion.key or ""),
        label=assertion.body_text,
        created_at_ms=assertion.created_at_ms,
        updated_at_ms=assertion.updated_at_ms,
        metadata=assertion_value,
    )


def read_archive_annotation_envelope(conn: sqlite3.Connection, annotation_id: str) -> ArchiveAnnotationEnvelope:
    assertion = read_assertion_envelope(conn, assertion_id_for_annotation(annotation_id))
    if assertion is None or assertion.status == AssertionStatus.DELETED:
        raise KeyError(annotation_id)
    target_type, target_id = _split_target_ref(assertion.target_ref)
    return ArchiveAnnotationEnvelope(
        annotation_id=annotation_id,
        target_type=target_type,
        target_id=target_id,
        body=assertion.body_text or "",
        created_at_ms=assertion.created_at_ms,
        updated_at_ms=assertion.updated_at_ms,
    )


def read_archive_correction_envelope(
    conn: sqlite3.Connection,
    correction_id: str,
) -> ArchiveCorrectionEnvelope:
    assertion = read_assertion_envelope(conn, assertion_id_for_correction(correction_id))
    if assertion is None or assertion.status == AssertionStatus.DELETED:
        raise KeyError(correction_id)
    target_type, target_id = _split_target_ref(assertion.target_ref)
    assertion_value = _assertion_value_document(assertion.value)
    return ArchiveCorrectionEnvelope(
        correction_id=correction_id,
        target_type=target_type,
        target_id=target_id,
        correction_type=str(assertion.key or ""),
        payload=assertion_value,
        created_at_ms=assertion.created_at_ms,
        updated_at_ms=assertion.updated_at_ms,
    )


def read_archive_saved_view_envelope(conn: sqlite3.Connection, name: str) -> ArchiveSavedViewEnvelope:
    assertion = _read_assertion_by_kind_key(conn, AssertionKind.SAVED_QUERY, name)
    if assertion is None:
        raise KeyError(name)
    assertion_value = _assertion_value_document(assertion.value)
    return ArchiveSavedViewEnvelope(
        view_id=_id_from_prefixed_target(assertion.target_ref, "saved_view:"),
        name=str(assertion.key or name),
        query=assertion_value,
        created_at_ms=assertion.created_at_ms,
        updated_at_ms=assertion.updated_at_ms,
    )


def read_archive_recall_pack_envelope(conn: sqlite3.Connection, recall_pack_id: str) -> ArchiveRecallPackEnvelope:
    assertion = read_assertion_envelope(conn, assertion_id_for_recall_pack(recall_pack_id))
    if assertion is None or assertion.status == AssertionStatus.DELETED:
        raise KeyError(recall_pack_id)
    assertion_value = _assertion_value_document(assertion.value)
    return ArchiveRecallPackEnvelope(
        recall_pack_id=recall_pack_id,
        name=str(assertion.key or ""),
        payload=assertion_value,
        created_at_ms=assertion.created_at_ms,
        updated_at_ms=assertion.updated_at_ms,
    )


def read_archive_workspace_envelope(conn: sqlite3.Connection, name: str) -> ArchiveWorkspaceEnvelope:
    assertion = _read_assertion_by_kind_key(conn, AssertionKind.WORKSPACE_NOTE, name)
    if assertion is None:
        raise KeyError(name)
    assertion_value = _assertion_value_document(assertion.value)
    return ArchiveWorkspaceEnvelope(
        workspace_id=_id_from_prefixed_target(assertion.target_ref, "workspace:"),
        name=str(assertion.key or name),
        settings=assertion_value,
        created_at_ms=assertion.created_at_ms,
        updated_at_ms=assertion.updated_at_ms,
    )


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table_name,),
        ).fetchone()
        is not None
    )


def _read_mirrored_assertion(conn: sqlite3.Connection, assertion_id: str) -> ArchiveAssertionEnvelope | None:
    if not _table_exists(conn, "assertions"):
        return None
    return read_assertion_envelope(conn, assertion_id)


def _split_target_ref(target_ref: str) -> tuple[str, str]:
    target_type, sep, target_id = target_ref.partition(":")
    if not sep:
        return "", target_ref
    return target_type, target_id


def _id_from_prefixed_target(target_ref: str, prefix: str) -> str:
    if target_ref.startswith(prefix):
        return target_ref[len(prefix) :]
    return target_ref


def _is_active_assertion(envelope: ArchiveAssertionEnvelope) -> bool:
    return envelope.status != AssertionStatus.DELETED


def list_assertions_by_kind(
    conn: sqlite3.Connection,
    kind: str | AssertionKind,
) -> list[ArchiveAssertionEnvelope]:
    """List active assertions of one kind, newest first."""
    if not _table_exists(conn, "assertions"):
        return []
    normalized_kind = _normalize_assertion_kind(kind)
    rows = conn.execute(
        f"SELECT {_ASSERTION_COLUMNS} FROM assertions WHERE kind = ? ORDER BY updated_at_ms DESC, assertion_id",
        (normalized_kind.value,),
    ).fetchall()
    return [envelope for row in rows if _is_active_assertion(envelope := _assertion_row_to_envelope(row))]


def _read_assertion_by_kind_key(
    conn: sqlite3.Connection,
    kind: str | AssertionKind,
    key: str,
) -> ArchiveAssertionEnvelope | None:
    if not _table_exists(conn, "assertions"):
        return None
    normalized_kind = _normalize_assertion_kind(kind)
    row = conn.execute(
        f"""
        SELECT {_ASSERTION_COLUMNS}
        FROM assertions
        WHERE kind = ?
          AND key = ?
          AND COALESCE(status, ?) != ?
        ORDER BY updated_at_ms DESC, assertion_id
        """,
        (normalized_kind.value, key, ASSERTION_DEFAULT_STATUS.value, AssertionStatus.DELETED.value),
    ).fetchone()
    if row is None:
        return None
    return _assertion_row_to_envelope(row)


def _blackboard_envelope_from_assertion(assertion: ArchiveAssertionEnvelope) -> ArchiveBlackboardNoteEnvelope:
    """Project one note assertion into the blackboard read envelope."""
    raw_target_type, raw_target_id = _split_target_ref(assertion.target_ref)
    target_type: str | None = raw_target_type
    target_id: str | None = raw_target_id
    note_id = str(assertion.key or assertion.target_ref)
    if raw_target_type == "assertion":
        note_id = raw_target_id
        target_type = None
        target_id = None
    elif raw_target_type == "":
        target_type = None
        target_id = None
    return ArchiveBlackboardNoteEnvelope(
        note_id=note_id,
        target_type=target_type,
        target_id=target_id,
        body=assertion.body_text or "",
        created_at_ms=assertion.created_at_ms,
        updated_at_ms=assertion.updated_at_ms,
    )


def read_archive_blackboard_note_envelope(conn: sqlite3.Connection, note_id: str) -> ArchiveBlackboardNoteEnvelope:
    assertion = read_assertion_envelope(conn, assertion_id_for_blackboard_note(note_id))
    if assertion is None or assertion.status == AssertionStatus.DELETED:
        raise KeyError(note_id)
    return _blackboard_envelope_from_assertion(assertion)


def list_archive_blackboard_note_envelopes(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
) -> list[ArchiveBlackboardNoteEnvelope]:
    """List blackboard notes from note assertions, newest first."""
    envelopes = [
        _blackboard_envelope_from_assertion(assertion)
        for assertion in list_assertions_by_kind(conn, AssertionKind.NOTE)
        # Candidate notes are deliberately visible only through the judgment
        # queue.  The blackboard is the active, operator-approved read model.
        if assertion.status == AssertionStatus.ACTIVE
    ]
    if limit is not None and limit > 0:
        return envelopes[:limit]
    return envelopes


def upsert_assertion(
    conn: sqlite3.Connection,
    *,
    assertion_id: str,
    target_ref: str,
    kind: str | AssertionKind,
    scope_ref: str | None = None,
    key: str | None = None,
    value: object | None = None,
    body_text: str | None = None,
    author_ref: str | None = None,
    author_kind: str | None = None,
    evidence_refs: Sequence[str] | None = None,
    status: str | AssertionStatus | None = None,
    visibility: str | AssertionVisibility | None = None,
    confidence: float | None = None,
    staleness: Mapping[str, object] | None = None,
    context_policy: Mapping[str, object] | AssertionContextPolicy | None = None,
    supersedes: Sequence[str] | None = None,
    now_ms: int | None = None,
    require_promotion: bool = True,
) -> ArchiveAssertionEnvelope:
    """Insert-or-update one assertion row keyed by caller-supplied ``assertion_id``.

    Single write chokepoint for the QUOTED->OPERATOR promotion gate (37t.15):
    any non-``user`` ``author_kind`` (agent, transform, detector, or any
    future automated writer) is coerced to ``status=CANDIDATE`` with a
    non-injected ``{"inject": False, "promotion_required": True}`` context
    policy, overriding whatever the caller requested -- an automated writer
    cannot self-promote a claim to authoritative/injectable. If the
    assertion_id already carries a terminal judgment outcome (accepted,
    rejected, deferred, superseded, or deleted -- set only by
    ``judge_assertion_candidate`` via ``mark_assertion_status``, never by this
    function), that outcome is preserved instead: a later automated write to
    the same id must not resurrect a judged-rejected row back to candidate.

    ``require_promotion=False`` is the sole, explicit escape hatch (the design
    note's "allowlist argument, not author_kind sniffing"): a handful of
    overlay kinds (session tags today) are categorization, not epistemic
    claims -- they have no judgment-queue/review path
    (:data:`ASSERTION_CLAIM_KINDS`) and their idempotent add/remove semantics
    would strand an agent-authored row as a permanently-unreachable candidate
    (the existing-row short-circuit in callers like ``add_user_tags`` never
    re-upserts once any non-deleted row exists). Callers must set this
    per-call-site deliberately; it is never inferred from ``kind``/
    ``author_kind`` inside this function.
    """
    conn.execute("PRAGMA foreign_keys = ON")
    timestamp = now_ms if now_ms is not None else _now_ms()
    existing = conn.execute(
        "SELECT created_at_ms, status FROM assertions WHERE assertion_id = ?",
        (assertion_id,),
    ).fetchone()
    created_at_ms = int(existing[0]) if existing is not None else timestamp
    existing_status = (
        _normalize_assertion_status(existing[1]) if existing is not None and existing[1] is not None else None
    )

    normalized_target_ref = normalize_object_ref_text(target_ref)
    normalized_scope_ref = normalize_object_ref_text(scope_ref) if scope_ref is not None else None
    normalized_author_ref = (
        normalize_object_ref_text(_normalize_assertion_author_ref(author_ref))
        if author_ref is not None
        else ASSERTION_DEFAULT_AUTHOR_REF
    )
    resolved_kind = _normalize_assertion_kind(kind)
    resolved_value = _normalize_assertion_value(value)
    resolved_staleness = _normalize_assertion_staleness(staleness)
    normalized_evidence_refs = [normalize_public_ref_text(ref) for ref in evidence_refs or ()]
    resolved_status = _normalize_assertion_status(status)
    resolved_visibility = _normalize_assertion_visibility(visibility)
    resolved_author_kind = _normalize_assertion_author_kind(author_kind)
    resolved_context_policy = _normalize_assertion_context_policy(context_policy)

    if require_promotion and resolved_author_kind != ASSERTION_DEFAULT_AUTHOR_KIND:
        if existing_status is not None and existing_status in _ASSERTION_TERMINAL_JUDGED_STATUSES:
            resolved_status = existing_status
        else:
            resolved_status = AssertionStatus.CANDIDATE
            resolved_context_policy = AssertionContextPolicy.from_raw(_ASSERTION_AGENT_CANDIDATE_CONTEXT_POLICY)

    resolved_context_policy = constrain_assertion_context_policy(
        resolved_context_policy,
        author_kind=resolved_author_kind,
        author_ref=normalized_author_ref,
        status=resolved_status,
    )

    evidence_refs_json = _dumps_optional(normalized_evidence_refs)
    supersedes_json = _dumps_optional(list(supersedes or ()))

    conn.execute(
        """
        INSERT INTO assertions (
            assertion_id, scope_ref, target_ref, key, kind, value_json, body_text,
            author_ref, author_kind, evidence_refs_json, status, visibility, confidence,
            staleness_json, context_policy_json, supersedes_json, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(assertion_id) DO UPDATE SET
            scope_ref = excluded.scope_ref,
            target_ref = excluded.target_ref,
            key = excluded.key,
            kind = excluded.kind,
            value_json = excluded.value_json,
            body_text = excluded.body_text,
            author_ref = excluded.author_ref,
            author_kind = excluded.author_kind,
            evidence_refs_json = excluded.evidence_refs_json,
            status = excluded.status,
            visibility = excluded.visibility,
            confidence = excluded.confidence,
            staleness_json = excluded.staleness_json,
            context_policy_json = excluded.context_policy_json,
            supersedes_json = excluded.supersedes_json,
            updated_at_ms = excluded.updated_at_ms
        """,
        (
            assertion_id,
            normalized_scope_ref,
            normalized_target_ref,
            key,
            resolved_kind.value,
            _dumps_optional(resolved_value.as_json_value()),
            body_text,
            normalized_author_ref,
            resolved_author_kind,
            evidence_refs_json,
            resolved_status.value,
            resolved_visibility.value,
            confidence,
            _dumps_optional(
                None if resolved_staleness is None else resolved_staleness.as_json_document(),
            ),
            _dumps_optional(resolved_context_policy.as_json_document()),
            supersedes_json,
            created_at_ms,
            timestamp,
        ),
    )
    envelope = read_assertion_envelope(conn, assertion_id)
    assert envelope is not None
    return envelope


def upsert_transform_candidate_assertions(
    conn: sqlite3.Connection,
    digest: SessionDigest,
    *,
    now_ms: int | None = None,
) -> list[ArchiveAssertionEnvelope]:
    """Mirror session digest decision candidates as non-injected assertions."""

    timestamp = now_ms if now_ms is not None else _now_ms()
    scope_ref = f"transform:{digest.transform.transform_id}@v{digest.transform.transform_version}"
    target_ref = f"session:{digest.session_id}"
    envelopes: list[ArchiveAssertionEnvelope] = []

    for index, candidate in enumerate(digest.decision_candidates):
        evidence_refs = _transform_candidate_evidence_refs(candidate)
        assertion_id = assertion_id_for_transform_candidate(
            session_id=digest.session_id,
            transform_id=digest.transform.transform_id,
            transform_version=digest.transform.transform_version,
            candidate_index=index,
            candidate_kind=candidate.kind,
            candidate_text=candidate.text,
            evidence_refs=evidence_refs,
        )
        existing = read_assertion_envelope(conn, assertion_id)
        if existing is not None and existing.status != AssertionStatus.CANDIDATE:
            envelopes.append(existing)
            continue
        envelopes.append(
            upsert_assertion(
                conn,
                assertion_id=assertion_id,
                scope_ref=scope_ref,
                target_ref=target_ref,
                key=f"candidate/{candidate.kind}/{index}",
                kind=AssertionKind.TRANSFORM_CANDIDATE,
                value={
                    "candidate_index": index,
                    "candidate_kind": candidate.kind,
                    "session_id": digest.session_id,
                    "source_origin": digest.transform.source_origin,
                    "transform_id": digest.transform.transform_id,
                    "transform_version": digest.transform.transform_version,
                },
                body_text=candidate.text,
                author_ref=scope_ref,
                author_kind="transform",
                evidence_refs=evidence_refs,
                status=AssertionStatus.CANDIDATE,
                visibility=AssertionVisibility.PRIVATE,
                context_policy={"inject": False, "promotion_required": True},
                now_ms=timestamp,
            )
        )

    return envelopes


def upsert_pathology_findings_as_assertions(
    conn: sqlite3.Connection,
    session_id: str,
    findings: Sequence[PathologyFinding],
    *,
    now_ms: int | None = None,
) -> list[ArchiveAssertionEnvelope]:
    """Mirror deterministic pathology detector findings as candidate assertions (#2383).

    Each finding becomes a private, non-injected ``AssertionKind.PATHOLOGY``
    candidate awaiting explicit promotion (Ref #2182), keyed by a deterministic
    id so a rebuild over identical evidence is idempotent. An operator-promoted
    finding (status != CANDIDATE) is never silently downgraded back to candidate.
    """

    timestamp = now_ms if now_ms is not None else _now_ms()
    target_ref = f"session:{session_id}"
    envelopes: list[ArchiveAssertionEnvelope] = []

    for finding in findings:
        evidence_refs = [ref.format() for ref in finding.evidence_refs]
        scope_ref = f"insight:pathology-detector@v{finding.detector_version}"
        assertion_id = assertion_id_for_pathology_finding(
            session_id=session_id,
            finding_kind=finding.kind,
            detector_version=finding.detector_version,
            finding_detail=finding.detail,
            evidence_refs=evidence_refs,
        )
        existing = read_assertion_envelope(conn, assertion_id)
        if existing is not None and existing.status != AssertionStatus.CANDIDATE:
            envelopes.append(existing)
            continue
        envelopes.append(
            upsert_assertion(
                conn,
                assertion_id=assertion_id,
                scope_ref=scope_ref,
                target_ref=target_ref,
                key=finding.kind,
                kind=AssertionKind.PATHOLOGY,
                value={
                    "pathology_kind": finding.kind,
                    "severity": finding.severity,
                    "occurrence_count": finding.occurrence_count,
                    "detector_version": finding.detector_version,
                    "session_id": session_id,
                },
                body_text=finding.detail,
                author_ref=scope_ref,
                author_kind="detector",
                evidence_refs=evidence_refs,
                status=AssertionStatus.CANDIDATE,
                visibility=AssertionVisibility.PRIVATE,
                context_policy={"inject": False, "promotion_required": True},
                now_ms=timestamp,
            )
        )

    return envelopes


def upsert_comparative_judgment_assertion(
    conn: sqlite3.Connection,
    judgment: ComparativeJudgment,
    *,
    author_kind: str,
    now_ms: int | None = None,
) -> ArchiveAssertionEnvelope:
    """Store one comparative judgment (rxdo.9.11, mechanism K) as an assertion row.

    ``judgment.judgment_id`` (a content hash -- see
    :func:`polylogue.insights.judgment.comparative.build_comparative_judgment`)
    is used verbatim as the assertion id, so recording an identical verdict
    twice is idempotent rather than duplicative.

    ``author_kind`` follows the existing promotion gate
    (:func:`upsert_assertion`): a non-``"user"`` author (an agent judge) is
    coerced to a CANDIDATE, non-injected row regardless of the caller's
    request -- agent judgments stay candidates awaiting operator promotion,
    per the recursive-safety spine (rxdo.9.15's cascade routes THOSE verdicts
    to the operator; this function does not decide routing).
    """

    from polylogue.insights.judgment.comparative import comparative_judgment_to_value

    timestamp = now_ms if now_ms is not None else _now_ms()
    value = comparative_judgment_to_value(judgment)
    evidence_refs = list(dict.fromkeys((*judgment.items, *judgment.evidence_refs)))
    is_user_authored = author_kind == ASSERTION_DEFAULT_AUTHOR_KIND
    return upsert_assertion(
        conn,
        assertion_id=judgment.judgment_id,
        scope_ref="insight:comparative-judgment@v1",
        target_ref=judgment.items[0],
        key=judgment.dimension,
        kind=AssertionKind.COMPARATIVE_JUDGMENT,
        value=value,
        body_text=judgment.rationale if judgment.rationale_visible else None,
        author_ref=judgment.judge.actor_ref,
        author_kind=author_kind,
        evidence_refs=evidence_refs,
        status=AssertionStatus.ACTIVE if is_user_authored else AssertionStatus.CANDIDATE,
        visibility=AssertionVisibility.PRIVATE,
        context_policy={"inject": False, "promotion_required": not is_user_authored},
        now_ms=timestamp,
    )


def list_comparative_judgments(conn: sqlite3.Connection) -> list[ComparativeJudgment]:
    """Read back every live comparative-judgment assertion row.

    ``list_assertions_by_kind`` only excludes ``DELETED`` rows, but the
    candidate-judgment lifecycle this kind reuses as-is
    (:func:`judge_assertion_candidate`) never deletes: a rejected verdict is
    left at ``status=REJECTED`` (a terminal, non-live row) and an accepted
    verdict leaves the original candidate at ``status=ACCEPTED`` while
    writing a *separate*, differently-id'd promoted row at
    ``status=ACTIVE`` (:func:`_promote_candidate_assertion`). Filtering only
    on non-``DELETED`` would therefore resurrect rejected verdicts as live
    judgments and double-count accepted ones (original ``ACCEPTED`` row plus
    the promoted ``ACTIVE`` row, parsed into two distinct
    :class:`ComparativeJudgment` objects). ``ACTIVE`` is the sole live
    status here: user-authored judgments are written directly as ``ACTIVE``
    (see :func:`upsert_comparative_judgment_assertion`), and agent-authored
    candidates only reach ``ACTIVE`` via the promoted row once accepted.
    """

    from polylogue.insights.judgment.comparative import comparative_judgment_from_value

    judgments: list[ComparativeJudgment] = []
    for envelope in list_assertions_by_kind(conn, AssertionKind.COMPARATIVE_JUDGMENT):
        if envelope.status != AssertionStatus.ACTIVE:
            continue
        if not isinstance(envelope.value, dict):
            continue
        raw_items = envelope.value.get("items", ())
        item_refs = {str(item) for item in raw_items} if isinstance(raw_items, list) else set()
        # evidence_refs_json is stored as items UNION evidence_refs (so every
        # comparison subject resolves as ordinary evidence); strip the item
        # refs back out to reconstruct the original standalone evidence_refs.
        extra_evidence_refs = [ref for ref in envelope.evidence_refs if ref not in item_refs]
        judgments.append(
            comparative_judgment_from_value(envelope.assertion_id, envelope.value, evidence_refs=extra_evidence_refs)
        )
    return judgments


_FINDING_KINDS: Final[frozenset[str]] = frozenset(
    {"query-delta", "query-drift", "measure", "pathology", "claim-vs-evidence"}
)


def _validate_finding_ref(value: str, *, field: str) -> str:
    """Keep finding evidence refs opaque until the rxdo substrate owns grammar."""

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"finding {field} must be a non-empty ref")
    return normalized


def _finding_value(finding: FindingAssertion) -> dict[str, object]:
    """Validate and project the additive ``polylogue.finding.v1`` payload."""

    if finding.finding_kind not in _FINDING_KINDS:
        raise ValueError(f"unsupported finding_kind: {finding.finding_kind!r}")
    if not finding.claim_key.strip():
        raise ValueError("finding claim_key must be non-empty")
    if isinstance(finding.n, bool) or finding.n < 0:
        raise ValueError("finding n must be a non-negative integer")
    statistic = dict(finding.statistic)
    if not {"op", "value", "unit"}.issubset(statistic):
        raise ValueError("finding statistic must contain op, value, and unit")

    query_ref = _validate_finding_ref(finding.query_ref, field="query_ref")
    result_set_ref = _validate_finding_ref(finding.result_set_ref, field="result_set_ref")
    value: dict[str, object] = {
        "_schema": "polylogue.finding.v1",
        "finding_kind": finding.finding_kind,
        "statistic": statistic,
        "n": finding.n,
        "query_ref": query_ref,
        "result_set_ref": result_set_ref,
    }
    if finding.baseline_ref is not None:
        value["baseline_ref"] = _validate_finding_ref(finding.baseline_ref, field="baseline_ref")
    if finding.current_ref is not None:
        value["current_ref"] = _validate_finding_ref(finding.current_ref, field="current_ref")
    if finding.finding_kind == "query-delta" and {"baseline_ref", "current_ref"} - value.keys():
        raise ValueError("query-delta findings require baseline_ref and current_ref")
    if finding.expected is not None:
        expected = dict(finding.expected)
        if not {"measure", "op", "value"}.issubset(expected):
            raise ValueError("finding expected must contain measure, op, and value")
        # Future rigor work can add bands, tolerances, or direction-only
        # expectations without a user-tier schema migration.
        value["expected"] = expected
    return value


def upsert_findings_as_assertions(
    conn: sqlite3.Connection,
    findings: Sequence[FindingAssertion],
    *,
    now_ms: int | None = None,
) -> list[ArchiveAssertionEnvelope]:
    """Write detector findings as private, non-injected assertion candidates.

    This mirrors :func:`upsert_pathology_findings_as_assertions`: repeated
    materialization maps identical claim/evidence inputs to the same row, and
    terminal operator judgments are never overwritten by a detector.
    """

    timestamp = now_ms if now_ms is not None else _now_ms()
    envelopes: list[ArchiveAssertionEnvelope] = []
    for finding in findings:
        value = _finding_value(finding)
        query_ref = str(value["query_ref"])
        result_set_ref = str(value["result_set_ref"])
        evidence_refs = [
            query_ref,
            result_set_ref,
            *(_validate_finding_ref(ref, field="evidence_ref") for ref in finding.evidence_refs),
        ]
        for field in ("baseline_ref", "current_ref"):
            ref = value.get(field)
            if isinstance(ref, str):
                evidence_refs.append(ref)
        evidence_refs = sorted(set(evidence_refs))
        detector_ref = _validate_finding_ref(finding.detector_ref, field="detector_ref")
        assertion_id = assertion_id_for_finding(
            claim_key=finding.claim_key.strip(),
            target_ref=finding.target_ref,
            value=value,
            evidence_refs=evidence_refs,
            detector_ref=detector_ref,
        )
        existing = read_assertion_envelope(conn, assertion_id)
        if existing is not None and existing.status != AssertionStatus.CANDIDATE:
            envelopes.append(existing)
            continue
        envelopes.append(
            upsert_assertion(
                conn,
                assertion_id=assertion_id,
                scope_ref=finding.scope_ref or detector_ref,
                target_ref=finding.target_ref,
                key=finding.claim_key.strip(),
                kind=AssertionKind.FINDING,
                value=value,
                body_text=finding.body_text,
                author_ref=detector_ref,
                author_kind="detector",
                evidence_refs=evidence_refs,
                status=AssertionStatus.CANDIDATE,
                visibility=AssertionVisibility.PRIVATE,
                confidence=finding.confidence,
                context_policy={"inject": False, "promotion_required": True},
                now_ms=timestamp,
            )
        )
    return envelopes


def _transform_candidate_evidence_refs(candidate: DecisionCandidate) -> list[str]:
    return [_format_transform_raw_ref(ref) for ref in candidate.raw_refs]


def _format_transform_raw_ref(ref: TransformRawRef) -> str:
    return ref.to_evidence_ref().format()


def mark_assertion_status(
    conn: sqlite3.Connection,
    assertion_id: str,
    status: str | AssertionStatus,
    *,
    now_ms: int | None = None,
) -> bool:
    """Update one assertion status without deleting its evidence row."""
    timestamp = now_ms if now_ms is not None else _now_ms()
    resolved_status = _normalize_assertion_status(status)
    cursor = conn.execute(
        """
        UPDATE assertions
        SET status = ?, updated_at_ms = ?
        WHERE assertion_id = ?
          AND COALESCE(status, ?) != ?
        """,
        (
            resolved_status.value,
            timestamp,
            assertion_id,
            ASSERTION_DEFAULT_STATUS.value,
            resolved_status.value,
        ),
    )
    return int(cursor.rowcount) > 0


def list_assertion_candidates(
    conn: sqlite3.Connection,
    *,
    target_ref: str | None = None,
    kinds: Sequence[str | AssertionKind] | None = None,
    limit: int | None = None,
) -> list[ArchiveAssertionEnvelope]:
    """List candidate assertion claims awaiting explicit judgment."""

    return list_assertion_claims(
        conn,
        kinds=ASSERTION_CANDIDATE_JUDGMENT_KINDS if kinds is None else kinds,
        target_ref=target_ref,
        statuses=(AssertionStatus.CANDIDATE,),
        limit=limit,
    )


ASSERTION_CANDIDATE_REVIEW_STATUSES: tuple[AssertionStatus, ...] = (
    AssertionStatus.CANDIDATE,
    AssertionStatus.ACCEPTED,
    AssertionStatus.REJECTED,
    AssertionStatus.DEFERRED,
    AssertionStatus.SUPERSEDED,
)

# Candidate capture is broader than successor-context claims. Terminal notes
# and corrections must be reviewable even though neither belongs in the normal
# active-claim reader (which deliberately excludes ordinary overlays).
ASSERTION_CANDIDATE_JUDGMENT_KINDS: tuple[AssertionKind, ...] = (
    AssertionKind.DECISION,
    AssertionKind.CAVEAT,
    AssertionKind.BLOCKER,
    AssertionKind.LESSON,
    AssertionKind.RUN_STATE,
    AssertionKind.TRANSFORM_CANDIDATE,
    AssertionKind.PATHOLOGY,
    AssertionKind.FINDING,
    AssertionKind.NOTE,
    AssertionKind.CORRECTION,
    AssertionKind.COMPARATIVE_JUDGMENT,
)


def list_assertion_candidate_reviews(
    conn: sqlite3.Connection,
    *,
    target_ref: str | None = None,
    kinds: Sequence[str | AssertionKind] | None = None,
    statuses: Sequence[str | AssertionStatus] | None = ASSERTION_CANDIDATE_REVIEW_STATUSES,
    limit: int | None = None,
) -> list[ArchiveAssertionCandidateReviewEnvelope]:
    """List candidate assertion lifecycle rows with their latest judgment.

    This is the review read model: it deliberately excludes promoted active
    assertions and ordinary overlay rows, while retaining accepted/rejected/
    deferred/superseded candidate rows so rebuilds cannot make old inference
    candidates look actionable again.
    """

    candidates = list_assertion_claims(
        conn,
        kinds=ASSERTION_CANDIDATE_JUDGMENT_KINDS if kinds is None else kinds,
        target_ref=target_ref,
        statuses=statuses,
        limit=limit,
    )
    return [
        ArchiveAssertionCandidateReviewEnvelope(
            candidate=candidate,
            latest_judgment=_latest_candidate_judgment(conn, candidate.assertion_id),
        )
        for candidate in candidates
    ]


def _latest_candidate_judgment(
    conn: sqlite3.Connection,
    candidate_assertion_id: str,
) -> ArchiveAssertionEnvelope | None:
    if not _table_exists(conn, "assertions"):
        return None
    row = conn.execute(
        f"""
        SELECT {_ASSERTION_COLUMNS}
        FROM assertions
        WHERE target_ref = ?
          AND kind = ?
          AND COALESCE(status, ?) != ?
        ORDER BY updated_at_ms DESC, assertion_id DESC
        LIMIT 1
        """,
        (
            f"assertion:{candidate_assertion_id}",
            AssertionKind.JUDGMENT.value,
            ASSERTION_DEFAULT_STATUS.value,
            AssertionStatus.DELETED.value,
        ),
    ).fetchone()
    if row is None:
        return None
    return _assertion_row_to_envelope(row)


def read_latest_candidate_judgment(
    conn: sqlite3.Connection,
    candidate_assertion_id: str,
) -> ArchiveAssertionEnvelope | None:
    """Return the durable latest judgment attached to one candidate."""

    return _latest_candidate_judgment(conn, _assertion_id_from_ref(candidate_assertion_id))


def judge_assertion_candidate(
    conn: sqlite3.Connection,
    *,
    candidate_ref: str,
    decision: str,
    reason: str | None = None,
    actor_ref: str = ASSERTION_DEFAULT_AUTHOR_REF,
    inject: bool = False,
    replacement_kind: str | AssertionKind | None = None,
    replacement_body_text: str | None = None,
    replacement_value: object | None = None,
    now_ms: int | None = None,
) -> ArchiveAssertionJudgmentEnvelope:
    """Record an explicit operator judgment for one candidate assertion."""

    normalized_decision = decision.strip().lower()
    if normalized_decision not in {"accept", "reject", "defer", "supersede"}:
        raise ValueError("candidate assertion decision must be accept, reject, defer, or supersede")
    normalized_replacement_kind = None if replacement_kind is None else _normalize_assertion_kind(replacement_kind)
    normalized_replacement_value = (
        None if replacement_value is None else _normalize_assertion_value(replacement_value).as_json_value()
    )
    if normalized_decision != "supersede" and any(
        value is not None
        for value in (normalized_replacement_kind, replacement_body_text, normalized_replacement_value)
    ):
        raise ValueError("replacement fields are only valid for a supersede judgment")
    candidate_id = _assertion_id_from_ref(candidate_ref)
    candidate = read_assertion_envelope(conn, candidate_id)
    if candidate is None:
        raise ValueError(f"candidate assertion not found: {candidate_ref}")
    if candidate.status != AssertionStatus.CANDIDATE:
        latest = _latest_candidate_judgment(conn, candidate_id)
        if latest is not None and _is_exact_judgment_retry(
            latest,
            decision=normalized_decision,
            reason=reason,
            actor_ref=actor_ref,
            inject=inject,
            replacement_kind=normalized_replacement_kind,
            replacement_body_text=replacement_body_text,
            replacement_value=normalized_replacement_value,
        ):
            return ArchiveAssertionJudgmentEnvelope(
                candidate=candidate,
                judgment=latest,
                resulting_assertion=_resulting_assertion_from_judgment(conn, latest),
                outcome="idempotent",
            )
        raise ValueError(f"candidate assertion has a conflicting prior judgment: {candidate_ref}")

    timestamp = now_ms if now_ms is not None else _now_ms()
    resulting_assertion: ArchiveAssertionEnvelope | None = None
    candidate_status = {
        "accept": AssertionStatus.ACCEPTED,
        "reject": AssertionStatus.REJECTED,
        "defer": AssertionStatus.DEFERRED,
        "supersede": AssertionStatus.SUPERSEDED,
    }[normalized_decision]

    if normalized_decision in {"accept", "supersede"}:
        resulting_assertion = _promote_candidate_assertion(
            conn,
            candidate,
            actor_ref=actor_ref,
            inject=inject,
            replacement_kind=normalized_replacement_kind,
            replacement_body_text=replacement_body_text,
            replacement_value=normalized_replacement_value,
            now_ms=timestamp,
        )

    mark_assertion_status(conn, candidate_id, candidate_status, now_ms=timestamp)
    refreshed = read_assertion_envelope(conn, candidate_id)
    if refreshed is not None:
        candidate = refreshed

    evidence_refs = [*candidate.evidence_refs, f"assertion:{candidate_id}"]
    if resulting_assertion is not None:
        evidence_refs.append(f"assertion:{resulting_assertion.assertion_id}")
    judgment_value: dict[str, JSONValue] = {
        "decision": normalized_decision,
        "candidate_ref": f"assertion:{candidate_id}",
        "reason": reason,
        "inject_authorized": inject if normalized_decision in {"accept", "supersede"} else False,
        "resulting_assertion_ref": None
        if resulting_assertion is None
        else f"assertion:{resulting_assertion.assertion_id}",
    }
    if normalized_decision == "supersede":
        judgment_value.update(
            replacement_kind=None if normalized_replacement_kind is None else normalized_replacement_kind.value,
            replacement_body_text=replacement_body_text,
            replacement_value=normalized_replacement_value,
        )
    judgment = upsert_assertion(
        conn,
        assertion_id=assertion_id_for_candidate_judgment(candidate_id, normalized_decision),
        scope_ref=candidate.scope_ref,
        target_ref=f"assertion:{candidate_id}",
        key=f"{normalized_decision}/{candidate_id}",
        kind=AssertionKind.JUDGMENT,
        value=judgment_value,
        body_text=reason,
        author_ref=actor_ref,
        author_kind="user",
        evidence_refs=evidence_refs,
        status=AssertionStatus.ACTIVE,
        visibility=AssertionVisibility.PRIVATE,
        context_policy={"inject": False},
        supersedes=(f"assertion:{candidate_id}",),
        now_ms=timestamp,
    )
    return ArchiveAssertionJudgmentEnvelope(
        candidate=candidate,
        judgment=judgment,
        resulting_assertion=resulting_assertion,
    )


def judge_assertion_candidates(
    conn: sqlite3.Connection,
    items: Sequence[ArchiveAssertionBulkJudgmentItemEnvelope],
    *,
    now_ms: int | None = None,
) -> ArchiveAssertionBulkJudgmentEnvelope:
    """Apply independent candidate judgments in one transaction.

    A malformed or conflicting item rolls back only its savepoint.  Repeated
    refs are intentionally collapsed before execution so one request cannot
    create duplicate review history for the same candidate.
    """

    unique_items: dict[str, ArchiveAssertionBulkJudgmentItemEnvelope] = {}
    ordered_candidate_ids: list[str] = []
    result_order: list[str | ArchiveAssertionBulkJudgmentResultEnvelope] = []
    conflicting_duplicates: set[str] = set()

    for item in items:
        try:
            candidate_id = _assertion_id_from_ref(item.candidate_ref)
        except ValueError as exc:
            result_order.append(
                ArchiveAssertionBulkJudgmentResultEnvelope(
                    candidate_ref=item.candidate_ref, outcome="failed", error=str(exc)
                )
            )
            continue
        previous = unique_items.get(candidate_id)
        if previous is None:
            unique_items[candidate_id] = item
            ordered_candidate_ids.append(candidate_id)
            result_order.append(candidate_id)
        elif not _same_bulk_judgment_input(previous, item):
            conflicting_duplicates.add(candidate_id)

    results_by_candidate_id: dict[str, ArchiveAssertionBulkJudgmentResultEnvelope] = {}
    batch_savepoint = "assertion_judgment_batch"
    conn.execute(f"SAVEPOINT {batch_savepoint}")
    try:
        for index, candidate_id in enumerate(ordered_candidate_ids):
            item = unique_items[candidate_id]
            if candidate_id in conflicting_duplicates:
                results_by_candidate_id[candidate_id] = ArchiveAssertionBulkJudgmentResultEnvelope(
                    candidate_ref=f"assertion:{candidate_id}",
                    outcome="failed",
                    error="conflicting duplicate judgment inputs for candidate",
                )
                continue
            savepoint = f"assertion_judgment_{index}"
            conn.execute(f"SAVEPOINT {savepoint}")
            try:
                result = judge_assertion_candidate(
                    conn,
                    candidate_ref=f"assertion:{candidate_id}",
                    decision=item.decision,
                    reason=item.reason,
                    actor_ref=item.actor_ref,
                    inject=item.inject,
                    replacement_kind=item.replacement_kind,
                    replacement_body_text=item.replacement_body_text,
                    replacement_value=item.replacement_value,
                    now_ms=now_ms,
                )
            except (TypeError, ValueError) as exc:
                conn.execute(f"ROLLBACK TO {savepoint}")
                results_by_candidate_id[candidate_id] = ArchiveAssertionBulkJudgmentResultEnvelope(
                    candidate_ref=f"assertion:{candidate_id}", outcome="failed", error=str(exc)
                )
            else:
                results_by_candidate_id[candidate_id] = ArchiveAssertionBulkJudgmentResultEnvelope(
                    candidate_ref=f"assertion:{candidate_id}", outcome=result.outcome, result=result
                )
            finally:
                conn.execute(f"RELEASE {savepoint}")
    except BaseException:
        conn.execute(f"ROLLBACK TO {batch_savepoint}")
        conn.execute(f"RELEASE {batch_savepoint}")
        raise
    else:
        conn.execute(f"RELEASE {batch_savepoint}")
    return ArchiveAssertionBulkJudgmentEnvelope(
        items=tuple(
            entry if isinstance(entry, ArchiveAssertionBulkJudgmentResultEnvelope) else results_by_candidate_id[entry]
            for entry in result_order
        )
    )


def _same_bulk_judgment_input(
    left: ArchiveAssertionBulkJudgmentItemEnvelope,
    right: ArchiveAssertionBulkJudgmentItemEnvelope,
) -> bool:
    """Compare duplicate input after its candidate reference was normalized."""

    return (
        left.decision,
        left.reason,
        left.actor_ref,
        left.inject,
        left.replacement_kind,
        left.replacement_body_text,
        left.replacement_value,
    ) == (
        right.decision,
        right.reason,
        right.actor_ref,
        right.inject,
        right.replacement_kind,
        right.replacement_body_text,
        right.replacement_value,
    )


def _is_exact_judgment_retry(
    judgment: ArchiveAssertionEnvelope,
    *,
    decision: str,
    reason: str | None,
    actor_ref: str,
    inject: bool,
    replacement_kind: AssertionKind | None,
    replacement_body_text: str | None,
    replacement_value: JSONValue | None,
) -> bool:
    value = judgment.value if isinstance(judgment.value, dict) else {}
    if decision == "supersede" and not {
        "replacement_kind",
        "replacement_body_text",
        "replacement_value",
    }.issubset(value):
        # Earlier judgment rows did not preserve replacement intent, so they
        # cannot prove an edited retry is exact.  Fail closed rather than
        # silently treating a possible correction as already applied.
        return False
    return (
        value.get("decision") == decision
        and value.get("reason") == reason
        and judgment.author_ref == actor_ref
        and bool(value.get("inject_authorized", False)) == inject
        and value.get("replacement_kind") == (None if replacement_kind is None else replacement_kind.value)
        and value.get("replacement_body_text") == replacement_body_text
        and value.get("replacement_value") == replacement_value
    )


def _resulting_assertion_from_judgment(
    conn: sqlite3.Connection,
    judgment: ArchiveAssertionEnvelope,
) -> ArchiveAssertionEnvelope | None:
    value = judgment.value if isinstance(judgment.value, dict) else {}
    result_ref = value.get("resulting_assertion_ref")
    if not isinstance(result_ref, str):
        return None
    return read_assertion_envelope(conn, _assertion_id_from_ref(result_ref))


def _assertion_id_from_ref(value: str) -> str:
    if value.startswith("assertion:"):
        ref = ObjectRef.parse(value)
        if ref.kind != "assertion" or ref.qualifiers:
            raise ValueError("candidate_ref must be an assertion ref")
        return ref.object_id
    return value


def _promote_candidate_assertion(
    conn: sqlite3.Connection,
    candidate: ArchiveAssertionEnvelope,
    *,
    actor_ref: str,
    inject: bool,
    replacement_kind: str | AssertionKind | None,
    replacement_body_text: str | None,
    replacement_value: object | None,
    now_ms: int,
) -> ArchiveAssertionEnvelope:
    context_policy: dict[str, JSONValue] = dict(candidate.context_policy)
    context_policy["inject"] = inject
    context_policy.pop("promotion_required", None)
    return upsert_assertion(
        conn,
        assertion_id=assertion_id_for_promoted_candidate(candidate.assertion_id),
        scope_ref=candidate.scope_ref,
        target_ref=candidate.target_ref,
        key=candidate.key,
        kind=replacement_kind or _candidate_active_kind(candidate),
        value=replacement_value if replacement_value is not None else candidate.value,
        body_text=replacement_body_text if replacement_body_text is not None else candidate.body_text,
        author_ref=actor_ref,
        author_kind="user",
        evidence_refs=(*candidate.evidence_refs, f"assertion:{candidate.assertion_id}"),
        status=AssertionStatus.ACTIVE,
        visibility=candidate.visibility,
        confidence=candidate.confidence,
        staleness=candidate.staleness,
        context_policy=context_policy,
        supersedes=(f"assertion:{candidate.assertion_id}",),
        now_ms=now_ms,
    )


def _candidate_active_kind(candidate: ArchiveAssertionEnvelope) -> AssertionKind:
    if candidate.kind == AssertionKind.TRANSFORM_CANDIDATE and isinstance(candidate.value, dict):
        candidate_kind = candidate.value.get("candidate_kind")
        if isinstance(candidate_kind, str) and candidate_kind:
            return _normalize_assertion_kind(candidate_kind)
    return candidate.kind


def _assertion_row_to_envelope(row: sqlite3.Row) -> ArchiveAssertionEnvelope:
    value = _normalize_assertion_value(_loads_optional(row[5])).as_json_value()
    staleness = _normalize_assertion_staleness(_loads_dict_optional(row[13]))
    context_policy = _normalize_assertion_context_policy(_loads_dict_optional(row[14]))
    return ArchiveAssertionEnvelope(
        assertion_id=str(row[0]),
        scope_ref=str(row[1]) if row[1] is not None else None,
        target_ref=str(row[2]),
        key=str(row[3]) if row[3] is not None else None,
        kind=_normalize_assertion_kind(str(row[4])),
        value=value,
        body_text=str(row[6]) if row[6] is not None else None,
        author_ref=_normalize_assertion_author_ref(str(row[7]) if row[7] is not None else None),
        author_kind=_normalize_assertion_author_kind(str(row[8]) if row[8] is not None else None),
        evidence_refs=_loads_str_list(row[9]),
        status=_normalize_assertion_status(str(row[10]) if row[10] is not None else None),
        visibility=_normalize_assertion_visibility(str(row[11]) if row[11] is not None else None),
        confidence=float(row[12]) if row[12] is not None else None,
        staleness=None if staleness is None else staleness.as_json_document(),
        context_policy=context_policy.as_json_document(),
        supersedes=_loads_str_list(row[15]),
        created_at_ms=int(row[16]),
        updated_at_ms=int(row[17]),
    )


_ASSERTION_COLUMNS = (
    "assertion_id, scope_ref, target_ref, key, kind, value_json, body_text, "
    "author_ref, author_kind, evidence_refs_json, status, visibility, confidence, "
    "staleness_json, context_policy_json, supersedes_json, created_at_ms, updated_at_ms"
)


def read_assertion_envelope(conn: sqlite3.Connection, assertion_id: str) -> ArchiveAssertionEnvelope | None:
    """Read one assertion by id, or ``None`` when absent."""
    row = conn.execute(
        f"SELECT {_ASSERTION_COLUMNS} FROM assertions WHERE assertion_id = ?",
        (assertion_id,),
    ).fetchone()
    if row is None:
        return None
    return _assertion_row_to_envelope(row)


def list_assertions_for_target(
    conn: sqlite3.Connection,
    target_ref: str,
    *,
    kind: str | AssertionKind | None = None,
) -> list[ArchiveAssertionEnvelope]:
    """List assertions for one ``target_ref``, optionally filtered by ``kind``."""
    if kind is None:
        rows = conn.execute(
            f"SELECT {_ASSERTION_COLUMNS} FROM assertions WHERE target_ref = ? ORDER BY created_at_ms, assertion_id",
            (target_ref,),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {_ASSERTION_COLUMNS} FROM assertions "
            "WHERE target_ref = ? AND kind = ? ORDER BY created_at_ms, assertion_id",
            (target_ref, _normalize_assertion_kind(kind).value),
        ).fetchall()
    return [_assertion_row_to_envelope(row) for row in rows]


def assertion_envelope_to_payload(envelope: ArchiveAssertionEnvelope) -> dict[str, object]:
    """Return a JSON-serializable backup/export payload for one assertion."""

    return {
        "assertion_id": envelope.assertion_id,
        "scope_ref": envelope.scope_ref,
        "target_ref": envelope.target_ref,
        "key": envelope.key,
        "kind": envelope.kind.value,
        "value": envelope.value,
        "body_text": envelope.body_text,
        "author_ref": envelope.author_ref,
        "author_kind": envelope.author_kind,
        "evidence_refs": list(envelope.evidence_refs),
        "status": envelope.status.value,
        "visibility": envelope.visibility.value,
        "confidence": envelope.confidence,
        "staleness": envelope.staleness,
        "context_policy": envelope.context_policy,
        "supersedes": list(envelope.supersedes),
        "created_at_ms": envelope.created_at_ms,
        "updated_at_ms": envelope.updated_at_ms,
    }


def list_assertions_for_export(
    conn: sqlite3.Connection,
    *,
    kinds: Sequence[str | AssertionKind] | None = None,
    statuses: Sequence[str | AssertionStatus] | None = None,
    limit: int | None = None,
) -> list[ArchiveAssertionEnvelope]:
    """List assertion rows for durable user-tier export.

    Unlike ``list_assertion_claims``, this is deliberately all-kinds and
    all-statuses by default: backup/export must include marks, overlays,
    accepted claims, deleted rows, and private transform candidates.
    """

    if not _table_exists(conn, "assertions"):
        return []

    where: list[str] = []
    params: list[object] = []

    if kinds is not None:
        normalized_kinds = tuple(_normalize_assertion_kind(kind).value for kind in kinds)
        if not normalized_kinds:
            return []
        placeholders = ", ".join("?" for _ in normalized_kinds)
        where.append(f"kind IN ({placeholders})")
        params.extend(normalized_kinds)

    if statuses is not None:
        normalized_statuses = tuple(_normalize_assertion_status(status).value for status in statuses)
        if not normalized_statuses:
            return []
        placeholders = ", ".join("?" for _ in normalized_statuses)
        where.append(f"COALESCE(status, ?) IN ({placeholders})")
        params.append(ASSERTION_DEFAULT_STATUS.value)
        params.extend(normalized_statuses)

    sql = f"SELECT {_ASSERTION_COLUMNS} FROM assertions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at_ms, assertion_id"
    if limit is not None and limit >= 0:
        sql += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(sql, tuple(params)).fetchall()
    return [_assertion_row_to_envelope(row) for row in rows]


ASSERTION_CLAIM_KINDS: tuple[AssertionKind, ...] = (
    AssertionKind.DECISION,
    AssertionKind.CAVEAT,
    AssertionKind.BLOCKER,
    AssertionKind.LESSON,
    AssertionKind.RUN_STATE,
    AssertionKind.TRANSFORM_CANDIDATE,
    AssertionKind.PATHOLOGY,
    AssertionKind.FINDING,
    AssertionKind.COMPARATIVE_JUDGMENT,
)


def list_assertion_claims(
    conn: sqlite3.Connection,
    *,
    kinds: Sequence[str | AssertionKind] = ASSERTION_CLAIM_KINDS,
    target_ref: str | None = None,
    scope_ref: str | None = None,
    statuses: Sequence[str | AssertionStatus] | None = (AssertionStatus.ACTIVE, AssertionStatus.CANDIDATE),
    context_inject: bool | None = None,
    annotation_schema_prefix: str | None = None,
    annotation_schema_qualified_id: str | None = None,
    annotation_schema_excluded_qualified_id: str | None = None,
    annotation_target_kind: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[ArchiveAssertionEnvelope]:
    """List lifecycle claims for successor-context/profile consumers.

    This helper intentionally covers authored/transform claims, not every
    overlay assertion row. Marks, annotations, saved views, recall packs, and
    workspaces keep their domain-specific read helpers above.
    """

    if not _table_exists(conn, "assertions"):
        return []

    where: list[str] = []
    params: list[object] = []

    normalized_kinds = tuple(_normalize_assertion_kind(kind).value for kind in kinds)
    if normalized_kinds:
        placeholders = ", ".join("?" for _ in normalized_kinds)
        where.append(f"kind IN ({placeholders})")
        params.extend(normalized_kinds)
    else:
        return []

    if target_ref is not None:
        where.append("target_ref = ?")
        params.append(target_ref)
    if scope_ref is not None:
        where.append("scope_ref = ?")
        params.append(scope_ref)
    if statuses is not None:
        if not statuses:
            return []
        normalized_statuses = tuple(_normalize_assertion_status(status).value for status in statuses)
        placeholders = ", ".join("?" for _ in normalized_statuses)
        where.append(f"COALESCE(status, ?) IN ({placeholders})")
        params.append(ASSERTION_DEFAULT_STATUS.value)
        params.extend(normalized_statuses)
    if annotation_schema_prefix is not None:
        where.append("substr(json_extract(value_json, '$._schema'), 1, length(?)) = ?")
        params.extend((annotation_schema_prefix, annotation_schema_prefix))
    if annotation_schema_qualified_id is not None:
        where.append("json_extract(value_json, '$._schema') = ?")
        params.append(annotation_schema_qualified_id)
    if annotation_schema_excluded_qualified_id is not None:
        where.append("json_extract(value_json, '$._schema') != ?")
        params.append(annotation_schema_excluded_qualified_id)
    if annotation_target_kind is not None:
        target_prefix = f"{annotation_target_kind}:"
        where.append("substr(target_ref, 1, length(?)) = ?")
        params.extend((target_prefix, target_prefix))

    sql = f"SELECT {_ASSERTION_COLUMNS} FROM assertions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY updated_at_ms DESC, assertion_id"
    if limit is not None and limit >= 0 and context_inject is None:
        sql += " LIMIT ? OFFSET ?"
        params.extend((limit, max(offset, 0)))

    rows = conn.execute(sql, tuple(params)).fetchall()
    claims = [_assertion_row_to_envelope(row) for row in rows]
    if context_inject is not None:
        claims = [claim for claim in claims if bool(claim.context_policy.get("inject")) is context_inject]
    if context_inject is not None:
        start = max(offset, 0)
        if limit is not None and limit >= 0:
            return claims[start : start + limit]
        return claims[start:]
    return claims


def count_assertion_claims(
    conn: sqlite3.Connection,
    *,
    kinds: Sequence[str | AssertionKind],
    statuses: Sequence[str | AssertionStatus],
    annotation_schema_prefix: str | None = None,
    annotation_schema_qualified_id: str | None = None,
    annotation_schema_excluded_qualified_id: str | None = None,
    annotation_target_kind: str | None = None,
) -> int:
    """Count a typed assertion selection without materializing claim rows."""

    if not _table_exists(conn, "assertions") or not kinds or not statuses:
        return 0
    normalized_kinds = tuple(_normalize_assertion_kind(kind).value for kind in kinds)
    normalized_statuses = tuple(_normalize_assertion_status(status).value for status in statuses)
    kind_placeholders = ", ".join("?" for _ in normalized_kinds)
    status_placeholders = ", ".join("?" for _ in normalized_statuses)
    where = [
        f"kind IN ({kind_placeholders})",
        f"COALESCE(status, ?) IN ({status_placeholders})",
    ]
    params: list[object] = [*normalized_kinds, ASSERTION_DEFAULT_STATUS.value, *normalized_statuses]
    if annotation_schema_prefix is not None:
        where.append("substr(json_extract(value_json, '$._schema'), 1, length(?)) = ?")
        params.extend((annotation_schema_prefix, annotation_schema_prefix))
    if annotation_schema_qualified_id is not None:
        where.append("json_extract(value_json, '$._schema') = ?")
        params.append(annotation_schema_qualified_id)
    if annotation_schema_excluded_qualified_id is not None:
        where.append("json_extract(value_json, '$._schema') != ?")
        params.append(annotation_schema_excluded_qualified_id)
    if annotation_target_kind is not None:
        target_prefix = f"{annotation_target_kind}:"
        where.append("substr(target_ref, 1, length(?)) = ?")
        params.extend((target_prefix, target_prefix))
    row = conn.execute(f"SELECT count(*) FROM assertions WHERE {' AND '.join(where)}", tuple(params)).fetchone()
    return int(row[0]) if row is not None else 0


__all__ = [
    "ASSERTION_CLAIM_KINDS",
    "ASSERTION_CANDIDATE_JUDGMENT_KINDS",
    "ASSERTION_CANDIDATE_REVIEW_STATUSES",
    "ASSERTION_DEFAULT_AUTHOR_KIND",
    "ASSERTION_DEFAULT_AUTHOR_REF",
    "ASSERTION_DEFAULT_CONTEXT_POLICY",
    "ASSERTION_DEFAULT_STATUS",
    "ASSERTION_DEFAULT_VISIBILITY",
    "ArchiveAnnotationEnvelope",
    "ArchiveAssertionEnvelope",
    "ArchiveAssertionBulkJudgmentEnvelope",
    "ArchiveAssertionBulkJudgmentItemEnvelope",
    "ArchiveAssertionBulkJudgmentResultEnvelope",
    "ArchiveAssertionCandidateReviewEnvelope",
    "ArchiveAssertionJudgmentEnvelope",
    "ArchiveBlackboardNoteEnvelope",
    "ArchiveSuppressionEnvelope",
    "ArchiveMarkEnvelope",
    "ArchiveCorrectionEnvelope",
    "ArchiveRecallPackEnvelope",
    "ArchiveSavedViewEnvelope",
    "ArchiveWorkspaceEnvelope",
    "FindingAssertion",
    "AssertionKind",
    "AssertionStatus",
    "AssertionVisibility",
    "assertion_envelope_to_payload",
    "assertion_id_for_annotation",
    "assertion_id_for_blackboard_note",
    "assertion_id_for_candidate_judgment",
    "assertion_id_for_correction",
    "assertion_id_for_finding",
    "assertion_id_for_mark",
    "assertion_id_for_promoted_candidate",
    "assertion_id_for_session_metadata",
    "assertion_id_for_recall_pack",
    "assertion_id_for_saved_view",
    "assertion_id_for_session_tag",
    "assertion_id_for_suppression",
    "assertion_id_for_pathology_finding",
    "assertion_id_for_transform_candidate",
    "assertion_id_for_workspace",
    "correction_id_for",
    "count_assertion_claims",
    "judge_assertion_candidate",
    "judge_assertion_candidates",
    "list_archive_blackboard_note_envelopes",
    "list_assertion_candidates",
    "list_assertion_candidate_reviews",
    "list_assertion_claims",
    "list_assertions_for_export",
    "list_assertions_by_kind",
    "list_assertions_for_target",
    "mark_assertion_status",
    "read_archive_annotation_envelope",
    "read_archive_blackboard_note_envelope",
    "read_archive_correction_envelope",
    "read_archive_mark_envelope",
    "read_archive_recall_pack_envelope",
    "read_archive_saved_view_envelope",
    "read_archive_suppression_envelope",
    "read_archive_workspace_envelope",
    "read_assertion_envelope",
    "read_latest_candidate_judgment",
    "upsert_annotation",
    "upsert_assertion",
    "upsert_blackboard_note",
    "upsert_correction",
    "upsert_findings_as_assertions",
    "upsert_mark",
    "upsert_recall_pack",
    "upsert_saved_view",
    "upsert_session_metadata_assertion",
    "upsert_session_tag_assertion",
    "upsert_suppression",
    "upsert_pathology_findings_as_assertions",
    "upsert_transform_candidate_assertions",
    "upsert_workspace",
]
