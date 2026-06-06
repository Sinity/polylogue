"""Storage for the learning-feedback loop (#1131).

User corrections are persisted in the ``user_corrections`` table, which is
intentionally separate from any content-hashed session payload. The
table is declared in
:mod:`polylogue.storage.sqlite.schema_ddl_archive` (``USER_CORRECTIONS_DDL``)
and the schema version was bumped to 15 in the same change so existing
databases reject mismatched versions until the operator runs the explicit
upgrade.

This module owns the SQL surface: insert/upsert, list, delete, and the
``hash_invariant_columns`` helper used by tests to assert that nothing in
this path touches the session's content hash.

The functions take a raw async SQLite connection (``aiosqlite.Connection``)
to match the rest of ``polylogue.storage.sqlite.queries``. Higher-level
mixins in ``polylogue.storage.repository.archive.repository_writes`` open
the connection and call into here.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from polylogue.insights.feedback import (
    CorrectionKind,
    LearningCorrection,
    now_utc,
    parse_correction_kind,
)

if TYPE_CHECKING:
    import aiosqlite


def _new_correction_id() -> str:
    return f"correction-{uuid.uuid4().hex}"


async def upsert_correction(
    conn: aiosqlite.Connection,
    *,
    session_id: str,
    kind: CorrectionKind,
    payload: dict[str, str],
    note: str | None = None,
) -> LearningCorrection:
    """Insert or replace the single correction of ``kind`` for ``session_id``.

    Returns the stored :class:`LearningCorrection`. Replacing an existing
    correction reuses the same ``correction_id`` so downstream callers
    have a stable surrogate key across edits (the
    ``(session_id, insight_kind)`` UNIQUE constraint enforces the
    invariant in the DB layer).
    """

    payload_json = json.dumps(payload, sort_keys=True)
    created_at = now_utc()

    # Look up the existing row so we preserve the surrogate ID on update.
    cursor = await conn.execute(
        "SELECT correction_id FROM user_corrections WHERE session_id = ? AND insight_kind = ?",
        (session_id, kind.value),
    )
    row = await cursor.fetchone()
    correction_id = row[0] if row is not None else _new_correction_id()

    await conn.execute(
        "INSERT INTO user_corrections "
        "  (correction_id, session_id, insight_kind, payload_json, note, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (session_id, insight_kind) DO UPDATE SET "
        "  payload_json = excluded.payload_json, "
        "  note = excluded.note, "
        "  created_at = excluded.created_at",
        (correction_id, session_id, kind.value, payload_json, note, created_at.isoformat()),
    )

    return LearningCorrection(
        session_id=session_id,
        kind=kind,
        payload=payload,
        note=note,
        created_at=created_at,
    )


async def list_corrections(
    conn: aiosqlite.Connection,
    *,
    session_id: str | None = None,
    kind: CorrectionKind | None = None,
) -> list[LearningCorrection]:
    """List stored corrections, optionally filtered by session and/or kind.

    Returns rows in deterministic ``(session_id, insight_kind)``
    order so callers (rebuild paths, tests, CLI output) see stable
    ordering across calls.
    """

    clauses: list[str] = []
    params: list[object] = []
    if session_id is not None:
        clauses.append("session_id = ?")
        params.append(session_id)
    if kind is not None:
        clauses.append("insight_kind = ?")
        params.append(kind.value)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        "SELECT session_id, insight_kind, payload_json, note, created_at "
        f"FROM user_corrections {where} "
        "ORDER BY session_id, insight_kind"
    )
    cursor = await conn.execute(sql, params)
    rows = await cursor.fetchall()

    out: list[LearningCorrection] = []
    for row in rows:
        try:
            payload_raw = json.loads(row[2])
        except (json.JSONDecodeError, TypeError):
            payload_raw = {}
        if not isinstance(payload_raw, dict):
            payload_raw = {}
        payload = {str(key): str(value) for key, value in payload_raw.items()}
        try:
            kind_value = parse_correction_kind(str(row[1]))
        except ValueError:
            # Persisted row uses an unrecognized kind (e.g. removed in a
            # later version). Skip rather than corrupt the typed surface.
            continue
        from datetime import datetime

        try:
            created_at = datetime.fromisoformat(str(row[4]))
        except ValueError:
            created_at = now_utc()
        out.append(
            LearningCorrection(
                session_id=str(row[0]),
                kind=kind_value,
                payload=payload,
                note=str(row[3]) if row[3] is not None else None,
                created_at=created_at,
            )
        )
    return out


async def delete_correction(
    conn: aiosqlite.Connection,
    *,
    session_id: str,
    kind: CorrectionKind,
) -> bool:
    """Delete the single correction of ``kind`` for ``session_id``.

    Returns ``True`` when a row was deleted, ``False`` when none existed.
    """

    cursor = await conn.execute(
        "DELETE FROM user_corrections WHERE session_id = ? AND insight_kind = ?",
        (session_id, kind.value),
    )
    return (cursor.rowcount or 0) > 0


async def clear_corrections(
    conn: aiosqlite.Connection,
    *,
    session_id: str,
) -> int:
    """Delete every correction for ``session_id``. Returns the count."""

    cursor = await conn.execute(
        "DELETE FROM user_corrections WHERE session_id = ?",
        (session_id,),
    )
    return cursor.rowcount or 0


# ---------------------------------------------------------------------------
# Test seam — declarative list of tables / columns this storage path must
# never touch. The hash-invariant test (see tests) reads the session's
# ``content_hash`` before and after a correction round-trip and confirms it
# is unchanged.
# ---------------------------------------------------------------------------

CONTENT_HASH_GUARDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("sessions", "content_hash"),
    ("sessions", "title"),
)
"""``(table, column)`` pairs whose values must be identical before and
after any correction lifecycle. Documented here so the invariant lives
next to the code that could break it."""


_SUPPORTED_KINDS: frozenset[CorrectionKind] = frozenset(CorrectionKind)


def supports_kind(kind: CorrectionKind | str) -> bool:
    """Return whether ``kind`` is a recognized correction kind.

    Surfaces should call ``parse_correction_kind`` for the typed parse;
    this helper is for callers that just want a boolean check without
    raising.
    """

    if isinstance(kind, CorrectionKind):
        return kind in _SUPPORTED_KINDS
    try:
        return parse_correction_kind(kind) in _SUPPORTED_KINDS
    except ValueError:
        return False


__all__ = [
    "CONTENT_HASH_GUARDED_COLUMNS",
    "clear_corrections",
    "delete_correction",
    "list_corrections",
    "supports_kind",
    "upsert_correction",
]


def _list_corrections_sequence_typed(
    items: Sequence[LearningCorrection],
) -> list[LearningCorrection]:
    """Type-only helper used in tests to surface ``Sequence`` semantics."""

    return list(items)
