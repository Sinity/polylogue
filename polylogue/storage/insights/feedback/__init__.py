"""Storage for the learning-feedback loop (#1131).

User corrections are persisted outside the content-hashed session payload.
Current split-archive writes use assertion rows in the user tier; older
single-file archives with ``user_corrections`` are still read and written by
the fallback path used in tests and historical local archives.

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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.insights.feedback import (
    CorrectionKind,
    LearningCorrection,
    now_utc,
    parse_correction_kind,
)
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ASSERTION_DEFAULT_AUTHOR_KIND,
    ASSERTION_DEFAULT_AUTHOR_REF,
    ASSERTION_DEFAULT_CONTEXT_POLICY,
    ASSERTION_DEFAULT_STATUS,
    ASSERTION_DEFAULT_VISIBILITY,
    AssertionKind,
    assertion_id_for_correction,
    correction_id_for,
)

if TYPE_CHECKING:
    import aiosqlite


async def _table_exists(conn: aiosqlite.Connection, table_name: str) -> bool:
    cursor = await conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    )
    return await cursor.fetchone() is not None


async def _attached_table_exists(conn: aiosqlite.Connection, schema_name: str, table_name: str) -> bool:
    cursor = await conn.execute(
        f"SELECT 1 FROM {schema_name}.sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    )
    return await cursor.fetchone() is not None


async def _attach_user_tier_if_present(conn: aiosqlite.Connection) -> bool:
    cursor = await conn.execute("PRAGMA database_list")
    rows = await cursor.fetchall()
    main_path: Path | None = None
    attached = False
    for row in rows:
        name = str(row[1])
        if name == "main":
            main_path = Path(str(row[2]))
        elif name == "user_tier":
            attached = True
    if attached:
        return True
    if main_path is None:
        return False
    user_db = main_path.parent / "user.db"
    if not user_db.exists():
        return False
    await conn.execute("ATTACH DATABASE ? AS user_tier", (str(user_db),))
    return True


async def _uses_archive_user_tier(conn: aiosqlite.Connection) -> bool:
    if await _table_exists(conn, "user_corrections"):
        return False
    return await _attach_user_tier_if_present(conn) and await _attached_table_exists(conn, "user_tier", "assertions")


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
    if await _uses_archive_user_tier(conn):
        stored_payload: dict[str, object] = {"payload": dict(payload), "note": note}
        stored_json = json.dumps(stored_payload, sort_keys=True)
        now_ms = int(created_at.timestamp() * 1000)
        correction_id = correction_id_for("insight", session_id, kind.value)
        assertion_id = assertion_id_for_correction(correction_id)
        cursor = await conn.execute(
            """
            SELECT created_at_ms
            FROM user_tier.assertions
            WHERE assertion_id = ?
            """,
            (assertion_id,),
        )
        row = await cursor.fetchone()
        created_ms = int(row[0]) if row is not None else now_ms
        await conn.execute(
            """
            INSERT INTO user_tier.assertions (
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
                context_policy_json = excluded.context_policy_json,
                supersedes_json = excluded.supersedes_json,
                updated_at_ms = excluded.updated_at_ms
            """,
            (
                assertion_id,
                "insight-feedback",
                f"insight:{session_id}",
                kind.value,
                AssertionKind.CORRECTION.value,
                stored_json,
                note,
                ASSERTION_DEFAULT_AUTHOR_REF,
                ASSERTION_DEFAULT_AUTHOR_KIND,
                json.dumps([], sort_keys=True, separators=(",", ":")),
                ASSERTION_DEFAULT_STATUS,
                ASSERTION_DEFAULT_VISIBILITY,
                None,
                None,
                json.dumps(ASSERTION_DEFAULT_CONTEXT_POLICY, sort_keys=True, separators=(",", ":")),
                json.dumps([], sort_keys=True, separators=(",", ":")),
                created_ms,
                now_ms,
            ),
        )
        return LearningCorrection(
            session_id=session_id,
            kind=kind,
            payload=payload,
            note=note,
            created_at=created_at,
        )

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
    if await _uses_archive_user_tier(conn):
        archive_clauses: list[str] = ["kind = ?", "COALESCE(status, '') != 'deleted'"]
        archive_params: list[object] = [AssertionKind.CORRECTION.value]
        if session_id is not None:
            archive_clauses.append("target_ref = ?")
            archive_params.append(f"insight:{session_id}")
        if kind is not None:
            archive_clauses.append("key = ?")
            archive_params.append(kind.value)
        cursor = await conn.execute(
            "SELECT target_ref, key, value_json, updated_at_ms "
            f"FROM user_tier.assertions WHERE {' AND '.join(archive_clauses)} "
            "ORDER BY target_ref, key",
            archive_params,
        )
        rows = await cursor.fetchall()
        archive_out: list[LearningCorrection] = []
        for row in rows:
            try:
                payload_raw = json.loads(row[2])
            except (json.JSONDecodeError, TypeError):
                payload_raw = {}
            if not isinstance(payload_raw, dict):
                payload_raw = {}
            payload_section = payload_raw.get("payload")
            if isinstance(payload_section, dict):
                payload_dict = dict(payload_section)
                note = payload_raw.get("note")
            else:
                payload_dict = payload_raw
                note = None
            try:
                kind_value = parse_correction_kind(str(row[1]))
            except ValueError:
                continue
            target_ref = str(row[0])
            _target_kind, _separator, resolved_session_id = target_ref.partition(":")
            created_at = datetime.fromtimestamp(int(row[3]) / 1000, tz=UTC)
            archive_out.append(
                LearningCorrection(
                    session_id=resolved_session_id or target_ref,
                    kind=kind_value,
                    payload={str(key): str(value) for key, value in payload_dict.items()},
                    note=str(note) if note is not None else None,
                    created_at=created_at,
                )
            )
        return archive_out

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

    cursor = (
        await conn.execute(
            """
            UPDATE user_tier.assertions
            SET status = 'deleted'
            WHERE assertion_id = ?
              AND COALESCE(status, '') != 'deleted'
            """,
            (assertion_id_for_correction(correction_id_for("insight", session_id, kind.value)),),
        )
        if await _uses_archive_user_tier(conn)
        else await conn.execute(
            "DELETE FROM user_corrections WHERE session_id = ? AND insight_kind = ?",
            (session_id, kind.value),
        )
    )
    return (cursor.rowcount or 0) > 0


async def clear_corrections(
    conn: aiosqlite.Connection,
    *,
    session_id: str,
) -> int:
    """Delete every correction for ``session_id``. Returns the count."""

    cursor = (
        await conn.execute(
            """
            UPDATE user_tier.assertions
            SET status = 'deleted'
            WHERE kind = ?
              AND target_ref = ?
              AND COALESCE(status, '') != 'deleted'
            """,
            (AssertionKind.CORRECTION.value, f"insight:{session_id}"),
        )
        if await _uses_archive_user_tier(conn)
        else await conn.execute(
            "DELETE FROM user_corrections WHERE session_id = ?",
            (session_id,),
        )
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
