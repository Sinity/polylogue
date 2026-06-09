"""Minimal user-tier write/read helpers.

These functions are intentionally narrow: they provide deterministic upsert
helpers and compact envelope readers for user-only tables in ``user.py``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime


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


def _deterministic_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return f"{prefix}:{digest.hexdigest()}"


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


def upsert_suppression(
    conn: sqlite3.Connection,
    session_id: str,
    reason: str | None,
    *,
    mode: str = "hide",
    now_ms: int | None = None,
) -> ArchiveSuppressionEnvelope:
    """Upsert one suppression row by ``session_id``."""
    conn.execute("PRAGMA foreign_keys = ON")
    timestamp = now_ms if now_ms is not None else _now_ms()

    existing = conn.execute(
        "SELECT created_at_ms FROM suppressions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    created_at_ms = int(existing[0]) if existing is not None else timestamp

    conn.execute(
        """
        INSERT INTO suppressions (session_id, reason, mode, created_at_ms, updated_at_ms)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            reason = excluded.reason,
            mode = excluded.mode,
            updated_at_ms = excluded.updated_at_ms
        """,
        (session_id, reason, mode, created_at_ms, timestamp),
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
    """Insert-or-update one mark row with deterministic ``mark_id``."""
    conn.execute("PRAGMA foreign_keys = ON")
    timestamp = now_ms if now_ms is not None else _now_ms()
    mark_id = _deterministic_id("mark", target_type, target_id, mark_type)
    metadata_json = _json_dumps(metadata)
    existing = conn.execute("SELECT created_at_ms FROM marks WHERE mark_id = ?", (mark_id,)).fetchone()
    created_at_ms = int(existing[0]) if existing is not None else timestamp

    conn.execute(
        """
        INSERT INTO marks (
            mark_id, target_type, target_id, mark_type, label, created_at_ms, updated_at_ms, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(mark_id) DO UPDATE SET
            target_type = excluded.target_type,
            target_id = excluded.target_id,
            mark_type = excluded.mark_type,
            label = excluded.label,
            metadata_json = excluded.metadata_json,
            updated_at_ms = excluded.updated_at_ms
        """,
        (mark_id, target_type, target_id, mark_type, label, created_at_ms, timestamp, metadata_json),
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
    """Insert-or-update one annotation row with deterministic default identity."""
    conn.execute("PRAGMA foreign_keys = ON")
    timestamp = now_ms if now_ms is not None else _now_ms()
    resolved_id = annotation_id or _deterministic_id("annotation", target_type, target_id, body)
    existing = conn.execute("SELECT created_at_ms FROM annotations WHERE annotation_id = ?", (resolved_id,)).fetchone()
    created_at_ms = int(existing[0]) if existing is not None else timestamp

    conn.execute(
        """
        INSERT INTO annotations (
            annotation_id, target_type, target_id, body, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(annotation_id) DO UPDATE SET
            target_type = excluded.target_type,
            target_id = excluded.target_id,
            body = excluded.body,
            updated_at_ms = excluded.updated_at_ms
        """,
        (resolved_id, target_type, target_id, body, created_at_ms, timestamp),
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
    """Insert-or-update one correction row with deterministic ``correction_id``."""
    conn.execute("PRAGMA foreign_keys = ON")
    timestamp = now_ms if now_ms is not None else _now_ms()
    correction_id = _deterministic_id("correction", target_type, target_id, correction_type)
    payload_json = _json_dumps(payload)
    existing = conn.execute(
        "SELECT created_at_ms FROM corrections WHERE target_type = ? AND target_id = ? AND correction_type = ?",
        (target_type, target_id, correction_type),
    ).fetchone()
    created_at_ms = int(existing[0]) if existing is not None else timestamp

    conn.execute(
        """
        INSERT INTO corrections (
            correction_id, target_type, target_id, correction_type, payload_json, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(target_type, target_id, correction_type) DO UPDATE SET
            payload_json = excluded.payload_json,
            updated_at_ms = excluded.updated_at_ms
        """,
        (
            correction_id,
            target_type,
            target_id,
            correction_type,
            payload_json,
            created_at_ms,
            timestamp,
        ),
    )
    return read_archive_correction_envelope(conn, correction_id)


def upsert_saved_view(
    conn: sqlite3.Connection,
    name: str,
    query: dict[str, object],
    *,
    now_ms: int | None = None,
) -> ArchiveSavedViewEnvelope:
    """Insert-or-update a saved query view keyed by name."""
    timestamp = now_ms if now_ms is not None else _now_ms()
    view_id = _deterministic_id("view", name)
    existing = conn.execute("SELECT created_at_ms FROM saved_views WHERE name = ?", (name,)).fetchone()
    created_at_ms = int(existing[0]) if existing is not None else timestamp

    conn.execute(
        """
        INSERT INTO saved_views (view_id, name, query_json, created_at_ms, updated_at_ms)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            query_json = excluded.query_json,
            updated_at_ms = excluded.updated_at_ms
        """,
        (view_id, name, _json_dumps(query), created_at_ms, timestamp),
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
    """Insert-or-update one recall pack by deterministic or supplied id."""
    timestamp = now_ms if now_ms is not None else _now_ms()
    resolved_id = recall_pack_id or _deterministic_id("recall-pack", name, _json_dumps(payload))
    existing = conn.execute(
        "SELECT created_at_ms FROM recall_packs WHERE recall_pack_id = ?",
        (resolved_id,),
    ).fetchone()
    created_at_ms = int(existing[0]) if existing is not None else timestamp

    conn.execute(
        """
        INSERT INTO recall_packs (recall_pack_id, name, payload_json, created_at_ms, updated_at_ms)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(recall_pack_id) DO UPDATE SET
            name = excluded.name,
            payload_json = excluded.payload_json,
            updated_at_ms = excluded.updated_at_ms
        """,
        (resolved_id, name, _json_dumps(payload), created_at_ms, timestamp),
    )
    return read_archive_recall_pack_envelope(conn, resolved_id)


def upsert_workspace(
    conn: sqlite3.Connection,
    name: str,
    settings: dict[str, object],
    *,
    now_ms: int | None = None,
) -> ArchiveWorkspaceEnvelope:
    """Insert-or-update one workspace keyed by name."""
    timestamp = now_ms if now_ms is not None else _now_ms()
    workspace_id = _deterministic_id("workspace", name)
    existing = conn.execute("SELECT created_at_ms FROM workspaces WHERE name = ?", (name,)).fetchone()
    created_at_ms = int(existing[0]) if existing is not None else timestamp

    conn.execute(
        """
        INSERT INTO workspaces (workspace_id, name, settings_json, created_at_ms, updated_at_ms)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            settings_json = excluded.settings_json,
            updated_at_ms = excluded.updated_at_ms
        """,
        (workspace_id, name, _json_dumps(settings), created_at_ms, timestamp),
    )
    return read_archive_workspace_envelope(conn, name)


def upsert_blackboard_note(
    conn: sqlite3.Connection,
    body: str,
    *,
    target_type: str | None = None,
    target_id: str | None = None,
    note_id: str | None = None,
    now_ms: int | None = None,
) -> ArchiveBlackboardNoteEnvelope:
    """Insert-or-update one blackboard note, optionally scoped to an archive target."""
    timestamp = now_ms if now_ms is not None else _now_ms()
    resolved_id = note_id or _deterministic_id("blackboard-note", target_type or "", target_id or "", body)
    existing = conn.execute(
        "SELECT created_at_ms FROM blackboard_notes WHERE note_id = ?",
        (resolved_id,),
    ).fetchone()
    created_at_ms = int(existing[0]) if existing is not None else timestamp

    conn.execute(
        """
        INSERT INTO blackboard_notes (note_id, target_type, target_id, body, created_at_ms, updated_at_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(note_id) DO UPDATE SET
            target_type = excluded.target_type,
            target_id = excluded.target_id,
            body = excluded.body,
            updated_at_ms = excluded.updated_at_ms
        """,
        (resolved_id, target_type, target_id, body, created_at_ms, timestamp),
    )
    return read_archive_blackboard_note_envelope(conn, resolved_id)


def read_archive_suppression_envelope(conn: sqlite3.Connection, session_id: str) -> ArchiveSuppressionEnvelope:
    row = conn.execute(
        """
        SELECT session_id, reason, mode, created_at_ms, updated_at_ms
        FROM suppressions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        raise KeyError(session_id)
    return ArchiveSuppressionEnvelope(
        session_id=str(row[0]),
        reason=str(row[1]) if row[1] is not None else None,
        mode=str(row[2]),
        created_at_ms=int(row[3]),
        updated_at_ms=int(row[4]),
    )


def read_archive_mark_envelope(conn: sqlite3.Connection, mark_id: str) -> ArchiveMarkEnvelope:
    row = conn.execute(
        """
        SELECT mark_id, target_type, target_id, mark_type, label, created_at_ms, updated_at_ms, metadata_json
        FROM marks
        WHERE mark_id = ?
        """,
        (mark_id,),
    ).fetchone()
    if row is None:
        raise KeyError(mark_id)
    return ArchiveMarkEnvelope(
        mark_id=str(row[0]),
        target_type=str(row[1]),
        target_id=str(row[2]),
        mark_type=str(row[3]),
        label=str(row[4]) if row[4] is not None else None,
        created_at_ms=int(row[5]),
        updated_at_ms=int(row[6]),
        metadata=_read_payload_text(row[7] if isinstance(row[7], str) else None),
    )


def read_archive_annotation_envelope(conn: sqlite3.Connection, annotation_id: str) -> ArchiveAnnotationEnvelope:
    row = conn.execute(
        """
        SELECT annotation_id, target_type, target_id, body, created_at_ms, updated_at_ms
        FROM annotations
        WHERE annotation_id = ?
        """,
        (annotation_id,),
    ).fetchone()
    if row is None:
        raise KeyError(annotation_id)
    return ArchiveAnnotationEnvelope(
        annotation_id=str(row[0]),
        target_type=str(row[1]),
        target_id=str(row[2]),
        body=str(row[3]),
        created_at_ms=int(row[4]),
        updated_at_ms=int(row[5]),
    )


def read_archive_correction_envelope(
    conn: sqlite3.Connection,
    correction_id: str,
) -> ArchiveCorrectionEnvelope:
    row = conn.execute(
        """
        SELECT correction_id, target_type, target_id, correction_type, payload_json, created_at_ms, updated_at_ms
        FROM corrections
        WHERE correction_id = ?
        """,
        (correction_id,),
    ).fetchone()
    if row is None:
        raise KeyError(correction_id)
    return ArchiveCorrectionEnvelope(
        correction_id=str(row[0]),
        target_type=str(row[1]),
        target_id=str(row[2]),
        correction_type=str(row[3]),
        payload=_read_payload_text(row[4] if isinstance(row[4], str) else None),
        created_at_ms=int(row[5]),
        updated_at_ms=int(row[6]),
    )


def read_archive_saved_view_envelope(conn: sqlite3.Connection, name: str) -> ArchiveSavedViewEnvelope:
    row = conn.execute(
        """
        SELECT view_id, name, query_json, created_at_ms, updated_at_ms
        FROM saved_views
        WHERE name = ?
        """,
        (name,),
    ).fetchone()
    if row is None:
        raise KeyError(name)
    return ArchiveSavedViewEnvelope(
        view_id=str(row[0]),
        name=str(row[1]),
        query=_read_payload_text(row[2] if isinstance(row[2], str) else None),
        created_at_ms=int(row[3]),
        updated_at_ms=int(row[4]),
    )


def read_archive_recall_pack_envelope(conn: sqlite3.Connection, recall_pack_id: str) -> ArchiveRecallPackEnvelope:
    row = conn.execute(
        """
        SELECT recall_pack_id, name, payload_json, created_at_ms, updated_at_ms
        FROM recall_packs
        WHERE recall_pack_id = ?
        """,
        (recall_pack_id,),
    ).fetchone()
    if row is None:
        raise KeyError(recall_pack_id)
    return ArchiveRecallPackEnvelope(
        recall_pack_id=str(row[0]),
        name=str(row[1]),
        payload=_read_payload_text(row[2] if isinstance(row[2], str) else None),
        created_at_ms=int(row[3]),
        updated_at_ms=int(row[4]),
    )


def read_archive_workspace_envelope(conn: sqlite3.Connection, name: str) -> ArchiveWorkspaceEnvelope:
    row = conn.execute(
        """
        SELECT workspace_id, name, settings_json, created_at_ms, updated_at_ms
        FROM workspaces
        WHERE name = ?
        """,
        (name,),
    ).fetchone()
    if row is None:
        raise KeyError(name)
    return ArchiveWorkspaceEnvelope(
        workspace_id=str(row[0]),
        name=str(row[1]),
        settings=_read_payload_text(row[2] if isinstance(row[2], str) else None),
        created_at_ms=int(row[3]),
        updated_at_ms=int(row[4]),
    )


def read_archive_blackboard_note_envelope(conn: sqlite3.Connection, note_id: str) -> ArchiveBlackboardNoteEnvelope:
    row = conn.execute(
        """
        SELECT note_id, target_type, target_id, body, created_at_ms, updated_at_ms
        FROM blackboard_notes
        WHERE note_id = ?
        """,
        (note_id,),
    ).fetchone()
    if row is None:
        raise KeyError(note_id)
    return ArchiveBlackboardNoteEnvelope(
        note_id=str(row[0]),
        target_type=str(row[1]) if row[1] is not None else None,
        target_id=str(row[2]) if row[2] is not None else None,
        body=str(row[3]),
        created_at_ms=int(row[4]),
        updated_at_ms=int(row[5]),
    )


__all__ = [
    "ArchiveAnnotationEnvelope",
    "ArchiveBlackboardNoteEnvelope",
    "ArchiveSuppressionEnvelope",
    "ArchiveMarkEnvelope",
    "ArchiveCorrectionEnvelope",
    "ArchiveRecallPackEnvelope",
    "ArchiveSavedViewEnvelope",
    "ArchiveWorkspaceEnvelope",
    "read_archive_annotation_envelope",
    "read_archive_blackboard_note_envelope",
    "read_archive_correction_envelope",
    "read_archive_mark_envelope",
    "read_archive_recall_pack_envelope",
    "read_archive_saved_view_envelope",
    "read_archive_suppression_envelope",
    "read_archive_workspace_envelope",
    "upsert_annotation",
    "upsert_blackboard_note",
    "upsert_correction",
    "upsert_mark",
    "upsert_recall_pack",
    "upsert_saved_view",
    "upsert_suppression",
    "upsert_workspace",
]
