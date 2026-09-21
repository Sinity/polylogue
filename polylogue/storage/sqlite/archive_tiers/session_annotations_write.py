"""Session tag CRUD: the write tier's session-annotation contract.

Writer module: index, user.
Twin-write contract: session-tag-assertion-mirror.

Extracted from ``archive_tiers/write.py`` (polylogue-1r9c hotspot-map slice
1): this is a self-contained read/write contract over ``session_tags`` that
shares no
state with the session/message/block writer in ``write.py`` beyond the
connection they're handed and a couple of small serialization helpers
duplicated-by-reference below (``_json_dumps`` is imported lazily from
``write.py`` at call time — see the module-level note on that import for why
it is lazy, not a module-level import).

``write.py`` re-exports every public name here for backward compatibility —
external callers keep importing from
``polylogue.storage.sqlite.archive_tiers.write`` unchanged. This module is
the single source of truth; ``write.py`` holds no duplicate definitions.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from polylogue.core.sqlite_introspection import table_exists as _table_exists


@dataclass(frozen=True, slots=True)
class ArchiveSessionTag:
    session_id: str
    tag: str
    tag_source: str
    method: str | None
    confidence: float | None
    evidence: dict[str, object] | None


def _json_loads(raw_json: str | bytes) -> dict[str, object]:
    if isinstance(raw_json, bytes):
        raw_json = raw_json.decode("utf-8")
    loaded = json.loads(raw_json or "{}")
    return loaded if isinstance(loaded, dict) else {}


def _json_tuple(raw_json: str | bytes) -> tuple[str, ...]:
    if isinstance(raw_json, bytes):
        raw_json = raw_json.decode("utf-8")
    loaded = json.loads(raw_json or "[]")
    return tuple(str(item) for item in loaded) if isinstance(loaded, list) else ()


def _json_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float | str | bytes | bytearray):
        return int(value)
    return 0


def upsert_session_tag(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    tag: str,
    tag_source: str,
    method: str | None = None,
    confidence: float | None = None,
    evidence: dict[str, object] | None = None,
) -> ArchiveSessionTag:
    """Upsert one unified user/auto tag row for an archive session."""
    from polylogue.storage.sqlite.archive_tiers.write import _json_dumps

    conn.execute("PRAGMA foreign_keys = ON")
    normalized_tag = tag.strip().lower()
    if not normalized_tag:
        raise ValueError("tag cannot be empty")
    if len(normalized_tag) > 200:
        raise ValueError("tag exceeds maximum length of 200 characters")
    with conn:
        conn.execute(
            """
            INSERT INTO session_tags (
                session_id, tag, tag_source, method, confidence, evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, tag, tag_source) DO UPDATE SET
                method = excluded.method,
                confidence = excluded.confidence,
                evidence_json = excluded.evidence_json
            """,
            (
                session_id,
                normalized_tag,
                tag_source,
                method,
                confidence,
                _json_dumps(evidence) if evidence is not None else None,
            ),
        )
        _mirror_session_tag_assertion_if_available(
            conn,
            session_id=session_id,
            tag=normalized_tag,
            tag_source=tag_source,
            method=method,
            confidence=confidence,
            evidence=evidence,
        )
    return read_session_tags(conn, session_id=session_id, tag_source=tag_source)[normalized_tag]


def _mirror_session_tag_assertion_if_available(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    tag: str,
    tag_source: str,
    method: str | None,
    confidence: float | None,
    evidence: dict[str, object] | None,
) -> None:
    """Mirror user tag writes when the active tier owns assertions."""
    if tag_source != "user" or not _table_exists(conn, "assertions"):
        return
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_session_tag_assertion

    upsert_session_tag_assertion(
        conn,
        session_id=session_id,
        tag=tag,
        tag_source=tag_source,
        method=method,
        confidence=confidence,
        evidence=evidence,
    )


def read_session_tags(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    tag_source: str | None = None,
) -> dict[str, ArchiveSessionTag]:
    """Read archive session tags keyed by normalized tag."""
    conn.row_factory = sqlite3.Row
    params: list[object] = [session_id]
    source_filter = ""
    if tag_source is not None:
        source_filter = "AND tag_source = ?"
        params.append(tag_source)
    rows = conn.execute(
        f"""
        SELECT session_id, tag, tag_source, method, confidence, evidence_json
        FROM session_tags
        WHERE session_id = ?
          {source_filter}
        ORDER BY tag_source, tag
        """,
        tuple(params),
    ).fetchall()
    return {
        row["tag"]: ArchiveSessionTag(
            session_id=row["session_id"],
            tag=row["tag"],
            tag_source=row["tag_source"],
            method=row["method"],
            confidence=row["confidence"],
            evidence=_json_loads(row["evidence_json"]) if row["evidence_json"] is not None else None,
        )
        for row in rows
    }


__all__ = [
    "ArchiveSessionTag",
    "read_session_tags",
    "upsert_session_tag",
]
