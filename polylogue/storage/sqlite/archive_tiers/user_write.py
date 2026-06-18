"""Minimal user-tier write/read helpers.

These functions are intentionally narrow: they provide deterministic upsert
helpers and compact envelope readers for user-only tables in ``user.py``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from polylogue.core.json import JSONValue, require_json_value
from polylogue.core.refs import ObjectRef, normalize_object_ref_text, normalize_public_ref_text

if TYPE_CHECKING:
    from polylogue.insights.transforms import DecisionCandidate, RecoveryDigest, TransformRawRef


class AssertionKind(StrEnum):
    """Closed v0 vocabulary for ``assertions.kind`` (#1883).

    The unified assertions table collapses the legacy user-tier overlays; each
    kind below corresponds to a meaning carried by one of those mini-systems or
    by the agent-blackboard surface. The DDL stores ``kind`` as plain ``TEXT``
    (no CHECK) so the vocabulary can grow without a schema bump; this enum is
    the authoritative documentation and typing aid.
    """

    MARK = "mark"
    HIGHLIGHT = "highlight"
    ANNOTATION = "annotation"
    CORRECTION = "correction"
    SUPPRESSION = "suppression"
    TAG = "tag"
    METADATA = "metadata"
    SAVED_QUERY = "saved_query"
    RECALL_PACK = "recall_pack"
    WORKSPACE_NOTE = "workspace_note"
    NOTE = "note"
    DECISION = "decision"
    CAVEAT = "caveat"
    LESSON = "lesson"
    BLOCKER = "blocker"
    HANDOFF = "handoff"
    RUN_STATE = "run_state"
    PROMPT_EVAL = "prompt_eval"
    TRANSFORM_CANDIDATE = "transform_candidate"


ASSERTION_DEFAULT_STATUS: Final = "active"
ASSERTION_DEFAULT_VISIBILITY: Final = "private"
ASSERTION_DEFAULT_AUTHOR_KIND: Final = "user"
ASSERTION_DEFAULT_AUTHOR_REF: Final = "user:local"
ASSERTION_DEFAULT_CONTEXT_POLICY: Final[dict[str, JSONValue]] = {"inject": False}


def _default_context_policy() -> dict[str, JSONValue]:
    return dict(ASSERTION_DEFAULT_CONTEXT_POLICY)


def _normalize_assertion_status(status: str | None) -> str:
    return status if status is not None else ASSERTION_DEFAULT_STATUS


def _normalize_assertion_visibility(visibility: str | None) -> str:
    return visibility if visibility is not None else ASSERTION_DEFAULT_VISIBILITY


def _normalize_assertion_author_kind(author_kind: str | None) -> str:
    return author_kind if author_kind is not None else ASSERTION_DEFAULT_AUTHOR_KIND


def _normalize_assertion_author_ref(author_ref: str | None) -> str:
    return author_ref if author_ref is not None else ASSERTION_DEFAULT_AUTHOR_REF


def _normalize_assertion_context_policy(context_policy: dict[str, object] | None) -> dict[str, JSONValue]:
    """Return the stored assertion context policy.

    Assertions are durable user-tier state first. They must not flow into future
    context unless a caller opts in explicitly, so even caller-supplied policies
    get an explicit ``inject`` key when omitted. This keeps generic overlay
    writes private/no-inject while preserving explicit promotion/candidate
    policy such as ``promotion_required``.
    """

    if context_policy is None:
        return _default_context_policy()
    normalized = {
        key: require_json_value(value, context=f"assertion context_policy.{key}")
        for key, value in context_policy.items()
    }
    normalized.setdefault("inject", False)
    return normalized


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
    kind: str
    value: object | None
    body_text: str | None
    author_ref: str | None
    author_kind: str | None
    evidence_refs: list[str]
    status: str | None
    visibility: str | None
    confidence: float | None
    staleness: dict[str, object] | None
    context_policy: dict[str, JSONValue] | None
    supersedes: list[str]
    created_at_ms: int
    updated_at_ms: int


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
        author_kind="user",
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
    staleness: dict[str, object] | None = None,
    context_policy: dict[str, object] | None = None,
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
    if assertion is None or assertion.status == "deleted":
        raise KeyError(session_id)
    assertion_value = assertion.value if isinstance(assertion.value, dict) else {}
    return ArchiveSuppressionEnvelope(
        session_id=session_id,
        reason=assertion.body_text,
        mode=str(assertion_value.get("mode") or "hide"),
        created_at_ms=assertion.created_at_ms,
        updated_at_ms=assertion.updated_at_ms,
    )


def read_archive_mark_envelope(conn: sqlite3.Connection, mark_id: str) -> ArchiveMarkEnvelope:
    assertion = read_assertion_envelope(conn, _deterministic_id(f"assertion-{AssertionKind.MARK}", mark_id))
    if assertion is None or assertion.status == "deleted":
        raise KeyError(mark_id)
    target_type, target_id = _split_target_ref(assertion.target_ref)
    assertion_value = assertion.value if isinstance(assertion.value, dict) else {}
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
    if assertion is None or assertion.status == "deleted":
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
    if assertion is None or assertion.status == "deleted":
        raise KeyError(correction_id)
    target_type, target_id = _split_target_ref(assertion.target_ref)
    assertion_value = assertion.value if isinstance(assertion.value, dict) else {}
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
    assertion_value = assertion.value if isinstance(assertion.value, dict) else {}
    return ArchiveSavedViewEnvelope(
        view_id=_id_from_prefixed_target(assertion.target_ref, "saved_view:"),
        name=str(assertion.key or name),
        query=assertion_value,
        created_at_ms=assertion.created_at_ms,
        updated_at_ms=assertion.updated_at_ms,
    )


def read_archive_recall_pack_envelope(conn: sqlite3.Connection, recall_pack_id: str) -> ArchiveRecallPackEnvelope:
    assertion = read_assertion_envelope(conn, assertion_id_for_recall_pack(recall_pack_id))
    if assertion is None or assertion.status == "deleted":
        raise KeyError(recall_pack_id)
    assertion_value = assertion.value if isinstance(assertion.value, dict) else {}
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
    assertion_value = assertion.value if isinstance(assertion.value, dict) else {}
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
    return envelope.status != "deleted"


def list_assertions_by_kind(conn: sqlite3.Connection, kind: str) -> list[ArchiveAssertionEnvelope]:
    """List active assertions of one kind, newest first."""
    if not _table_exists(conn, "assertions"):
        return []
    rows = conn.execute(
        f"SELECT {_ASSERTION_COLUMNS} FROM assertions WHERE kind = ? ORDER BY updated_at_ms DESC, assertion_id",
        (kind,),
    ).fetchall()
    return [envelope for row in rows if _is_active_assertion(envelope := _assertion_row_to_envelope(row))]


def _read_assertion_by_kind_key(
    conn: sqlite3.Connection,
    kind: str,
    key: str,
) -> ArchiveAssertionEnvelope | None:
    if not _table_exists(conn, "assertions"):
        return None
    row = conn.execute(
        f"""
        SELECT {_ASSERTION_COLUMNS}
        FROM assertions
        WHERE kind = ?
          AND key = ?
          AND COALESCE(status, '') != 'deleted'
        ORDER BY updated_at_ms DESC, assertion_id
        """,
        (kind, key),
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
    if assertion is None or assertion.status == "deleted":
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
    ]
    if limit is not None and limit > 0:
        return envelopes[:limit]
    return envelopes


def upsert_assertion(
    conn: sqlite3.Connection,
    *,
    assertion_id: str,
    target_ref: str,
    kind: str,
    scope_ref: str | None = None,
    key: str | None = None,
    value: object | None = None,
    body_text: str | None = None,
    author_ref: str | None = None,
    author_kind: str | None = None,
    evidence_refs: Sequence[str] | None = None,
    status: str | None = None,
    visibility: str | None = None,
    confidence: float | None = None,
    staleness: dict[str, object] | None = None,
    context_policy: dict[str, object] | None = None,
    supersedes: Sequence[str] | None = None,
    now_ms: int | None = None,
) -> ArchiveAssertionEnvelope:
    """Insert-or-update one assertion row keyed by caller-supplied ``assertion_id``."""
    conn.execute("PRAGMA foreign_keys = ON")
    timestamp = now_ms if now_ms is not None else _now_ms()
    existing = conn.execute(
        "SELECT created_at_ms FROM assertions WHERE assertion_id = ?",
        (assertion_id,),
    ).fetchone()
    created_at_ms = int(existing[0]) if existing is not None else timestamp

    normalized_target_ref = normalize_object_ref_text(target_ref)
    normalized_scope_ref = normalize_object_ref_text(scope_ref) if scope_ref is not None else None
    normalized_author_ref = (
        normalize_object_ref_text(_normalize_assertion_author_ref(author_ref))
        if author_ref is not None
        else ASSERTION_DEFAULT_AUTHOR_REF
    )
    normalized_evidence_refs = [normalize_public_ref_text(ref) for ref in evidence_refs or ()]
    resolved_status = _normalize_assertion_status(status)
    resolved_visibility = _normalize_assertion_visibility(visibility)
    resolved_author_kind = _normalize_assertion_author_kind(author_kind)
    resolved_context_policy = _normalize_assertion_context_policy(context_policy)
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
            kind,
            _dumps_optional(value),
            body_text,
            normalized_author_ref,
            resolved_author_kind,
            evidence_refs_json,
            resolved_status,
            resolved_visibility,
            confidence,
            _dumps_optional(staleness),
            _dumps_optional(resolved_context_policy),
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
    digest: RecoveryDigest,
    *,
    now_ms: int | None = None,
) -> list[ArchiveAssertionEnvelope]:
    """Mirror recovery digest decision candidates as non-injected assertions."""

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
        if existing is not None and existing.status != "candidate":
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
                status="candidate",
                visibility="private",
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
    status: str,
    *,
    now_ms: int | None = None,
) -> bool:
    """Update one assertion status without deleting its evidence row."""
    timestamp = now_ms if now_ms is not None else _now_ms()
    cursor = conn.execute(
        """
        UPDATE assertions
        SET status = ?, updated_at_ms = ?
        WHERE assertion_id = ?
          AND COALESCE(status, '') != ?
        """,
        (status, timestamp, assertion_id, status),
    )
    return int(cursor.rowcount) > 0


def _assertion_row_to_envelope(row: sqlite3.Row) -> ArchiveAssertionEnvelope:
    return ArchiveAssertionEnvelope(
        assertion_id=str(row[0]),
        scope_ref=str(row[1]) if row[1] is not None else None,
        target_ref=str(row[2]),
        key=str(row[3]) if row[3] is not None else None,
        kind=str(row[4]),
        value=_loads_optional(row[5]),
        body_text=str(row[6]) if row[6] is not None else None,
        author_ref=_normalize_assertion_author_ref(str(row[7]) if row[7] is not None else None),
        author_kind=_normalize_assertion_author_kind(str(row[8]) if row[8] is not None else None),
        evidence_refs=_loads_str_list(row[9]),
        status=_normalize_assertion_status(str(row[10]) if row[10] is not None else None),
        visibility=_normalize_assertion_visibility(str(row[11]) if row[11] is not None else None),
        confidence=float(row[12]) if row[12] is not None else None,
        staleness=_loads_dict_optional(row[13]),
        context_policy=_normalize_assertion_context_policy(_loads_dict_optional(row[14])),
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
    kind: str | None = None,
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
            (target_ref, kind),
        ).fetchall()
    return [_assertion_row_to_envelope(row) for row in rows]


def assertion_envelope_to_payload(envelope: ArchiveAssertionEnvelope) -> dict[str, object]:
    """Return a JSON-serializable backup/export payload for one assertion."""

    return {
        "assertion_id": envelope.assertion_id,
        "scope_ref": envelope.scope_ref,
        "target_ref": envelope.target_ref,
        "key": envelope.key,
        "kind": envelope.kind,
        "value": envelope.value,
        "body_text": envelope.body_text,
        "author_ref": envelope.author_ref,
        "author_kind": envelope.author_kind,
        "evidence_refs": list(envelope.evidence_refs),
        "status": envelope.status,
        "visibility": envelope.visibility,
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
    statuses: Sequence[str] | None = None,
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
        normalized_kinds = tuple(str(kind) for kind in kinds)
        if not normalized_kinds:
            return []
        placeholders = ", ".join("?" for _ in normalized_kinds)
        where.append(f"kind IN ({placeholders})")
        params.extend(normalized_kinds)

    if statuses is not None:
        normalized_statuses = tuple(str(status) for status in statuses)
        if not normalized_statuses:
            return []
        placeholders = ", ".join("?" for _ in normalized_statuses)
        where.append(f"COALESCE(status, ?) IN ({placeholders})")
        params.append(ASSERTION_DEFAULT_STATUS)
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
)


def list_assertion_claims(
    conn: sqlite3.Connection,
    *,
    kinds: Sequence[str | AssertionKind] = ASSERTION_CLAIM_KINDS,
    target_ref: str | None = None,
    scope_ref: str | None = None,
    statuses: Sequence[str] | None = ("active", "candidate"),
    context_inject: bool | None = None,
    limit: int | None = None,
) -> list[ArchiveAssertionEnvelope]:
    """List lifecycle claims for recovery/work-packet/profile consumers.

    This helper intentionally covers authored/transform claims, not every
    overlay assertion row. Marks, annotations, saved views, recall packs, and
    workspaces keep their domain-specific read helpers above.
    """

    if not _table_exists(conn, "assertions"):
        return []

    where: list[str] = []
    params: list[object] = []

    normalized_kinds = tuple(str(kind) for kind in kinds)
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
        placeholders = ", ".join("?" for _ in statuses)
        where.append(f"COALESCE(status, ?) IN ({placeholders})")
        params.append(ASSERTION_DEFAULT_STATUS)
        params.extend(str(status) for status in statuses)

    sql = f"SELECT {_ASSERTION_COLUMNS} FROM assertions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY updated_at_ms DESC, assertion_id"

    rows = conn.execute(sql, tuple(params)).fetchall()
    claims = [_assertion_row_to_envelope(row) for row in rows]
    if context_inject is not None:
        claims = [
            claim
            for claim in claims
            if bool((claim.context_policy or ASSERTION_DEFAULT_CONTEXT_POLICY).get("inject")) is context_inject
        ]
    if limit is not None and limit >= 0:
        return claims[:limit]
    return claims


__all__ = [
    "ASSERTION_CLAIM_KINDS",
    "ASSERTION_DEFAULT_AUTHOR_KIND",
    "ASSERTION_DEFAULT_AUTHOR_REF",
    "ASSERTION_DEFAULT_CONTEXT_POLICY",
    "ASSERTION_DEFAULT_STATUS",
    "ASSERTION_DEFAULT_VISIBILITY",
    "ArchiveAnnotationEnvelope",
    "ArchiveAssertionEnvelope",
    "ArchiveBlackboardNoteEnvelope",
    "ArchiveSuppressionEnvelope",
    "ArchiveMarkEnvelope",
    "ArchiveCorrectionEnvelope",
    "ArchiveRecallPackEnvelope",
    "ArchiveSavedViewEnvelope",
    "ArchiveWorkspaceEnvelope",
    "AssertionKind",
    "assertion_envelope_to_payload",
    "assertion_id_for_annotation",
    "assertion_id_for_blackboard_note",
    "assertion_id_for_correction",
    "assertion_id_for_mark",
    "assertion_id_for_recall_pack",
    "assertion_id_for_saved_view",
    "assertion_id_for_suppression",
    "assertion_id_for_transform_candidate",
    "assertion_id_for_workspace",
    "correction_id_for",
    "list_archive_blackboard_note_envelopes",
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
    "upsert_annotation",
    "upsert_assertion",
    "upsert_blackboard_note",
    "upsert_correction",
    "upsert_mark",
    "upsert_recall_pack",
    "upsert_saved_view",
    "upsert_suppression",
    "upsert_transform_candidate_assertions",
    "upsert_workspace",
]
