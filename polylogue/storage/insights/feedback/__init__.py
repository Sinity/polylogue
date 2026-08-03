"""Storage for the learning-feedback loop (#1131).

User corrections are persisted outside the content-hashed session payload,
as ``AssertionKind.CORRECTION`` rows in the user tier's unified ``assertions``
table (``user.db``).

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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.insights.feedback import (
    CorrectionKind,
    LearningCorrection,
    now_utc,
    parse_correction_kind,
)
from polylogue.storage.introspection import table_exists_async
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


async def _attached_table_exists(conn: aiosqlite.Connection, schema_name: str, table_name: str) -> bool:
    return await table_exists_async(conn, table_name, schema=schema_name)


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
    return await _attach_user_tier_if_present(conn) and await _attached_table_exists(conn, "user_tier", "assertions")


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

    created_at = now_utc()
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
    out: list[LearningCorrection] = []
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
        out.append(
            LearningCorrection(
                session_id=resolved_session_id or target_ref,
                kind=kind_value,
                payload={str(key): str(value) for key, value in payload_dict.items()},
                note=str(note) if note is not None else None,
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
        """
        UPDATE user_tier.assertions
        SET status = 'deleted'
        WHERE assertion_id = ?
          AND COALESCE(status, '') != 'deleted'
        """,
        (assertion_id_for_correction(correction_id_for("insight", session_id, kind.value)),),
    )
    return (cursor.rowcount or 0) > 0


async def clear_corrections(
    conn: aiosqlite.Connection,
    *,
    session_id: str,
) -> int:
    """Delete every correction for ``session_id``. Returns the count."""

    cursor = await conn.execute(
        """
        UPDATE user_tier.assertions
        SET status = 'deleted'
        WHERE kind = ?
          AND target_ref = ?
          AND COALESCE(status, '') != 'deleted'
        """,
        (AssertionKind.CORRECTION.value, f"insight:{session_id}"),
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
