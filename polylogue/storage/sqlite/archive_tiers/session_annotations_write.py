"""Session tag/work-event/phase CRUD: the write tier's session-annotation contract.

Writer module: index, user.
Twin-write contract: session-tag-assertion-mirror.

Extracted from ``archive_tiers/write.py`` (polylogue-1r9c hotspot-map slice
1): this is a self-contained read/write contract over three tables
(``session_tags``, ``session_work_events``, ``session_phases``) that share no
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


@dataclass(frozen=True, slots=True)
class ArchiveSessionTag:
    session_id: str
    tag: str
    tag_source: str
    method: str | None
    confidence: float | None
    evidence: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class ArchiveSessionWorkEvent:
    event_id: str
    session_id: str
    position: int
    work_event_type: str
    summary: str
    confidence: float
    start_index: int
    end_index: int
    started_at_ms: int | None
    ended_at_ms: int | None
    duration_ms: int
    file_paths: tuple[str, ...]
    tools_used: tuple[str, ...]
    evidence: dict[str, object]
    inference: dict[str, object]
    search_text: str
    input_high_water_mark: str | None = None
    input_high_water_mark_source: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveSessionPhase:
    phase_id: str
    session_id: str
    position: int
    start_index: int
    end_index: int
    started_at_ms: int | None
    ended_at_ms: int | None
    duration_ms: int
    tool_counts: dict[str, int]
    word_count: int
    evidence: dict[str, object]
    inference: dict[str, object]
    search_text: str
    input_high_water_mark: str | None = None
    input_high_water_mark_source: str | None = None


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


def _refresh_session_profile_count(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    table: str,
    column: str,
) -> None:
    count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (session_id,)).fetchone()[0]
    conn.execute(
        f"""
        INSERT INTO session_profiles (session_id, {column})
        VALUES (?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            {column} = excluded.{column}
        """,
        (session_id, count),
    )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


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


def upsert_session_work_event(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    position: int,
    work_event_type: str,
    summary: str,
    confidence: float = 0.0,
    start_index: int = 0,
    end_index: int = 0,
    started_at_ms: int | None = None,
    ended_at_ms: int | None = None,
    duration_ms: int = 0,
    file_paths: tuple[str, ...] = (),
    tools_used: tuple[str, ...] = (),
    evidence: dict[str, object] | None = None,
    inference: dict[str, object] | None = None,
    search_text: str = "",
    input_high_water_mark: str | None = None,
    input_high_water_mark_source: str | None = None,
) -> ArchiveSessionWorkEvent:
    """Upsert one deterministic session work-event row."""
    from polylogue.storage.sqlite.archive_tiers.write import _json_dumps

    conn.execute("PRAGMA foreign_keys = ON")
    with conn:
        conn.execute(
            """
            INSERT INTO session_work_events (
                session_id, position, work_event_type, summary, confidence,
                start_index, end_index, started_at_ms, ended_at_ms, duration_ms,
                file_paths_json, tools_used_json,
                input_high_water_mark, input_high_water_mark_source,
                evidence_json, inference_json, search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, position) DO UPDATE SET
                work_event_type = excluded.work_event_type,
                summary = excluded.summary,
                confidence = excluded.confidence,
                start_index = excluded.start_index,
                end_index = excluded.end_index,
                started_at_ms = excluded.started_at_ms,
                ended_at_ms = excluded.ended_at_ms,
                duration_ms = excluded.duration_ms,
                file_paths_json = excluded.file_paths_json,
                tools_used_json = excluded.tools_used_json,
                input_high_water_mark = excluded.input_high_water_mark,
                input_high_water_mark_source = excluded.input_high_water_mark_source,
                evidence_json = excluded.evidence_json,
                inference_json = excluded.inference_json,
                search_text = excluded.search_text
            """,
            (
                session_id,
                position,
                work_event_type,
                summary,
                confidence,
                start_index,
                end_index,
                started_at_ms,
                ended_at_ms,
                duration_ms,
                _json_dumps(list(file_paths)),
                _json_dumps(list(tools_used)),
                input_high_water_mark,
                input_high_water_mark_source,
                _json_dumps(evidence or {}),
                _json_dumps(inference or {}),
                search_text,
            ),
        )
        _refresh_session_profile_count(conn, session_id, table="session_work_events", column="work_event_count")
    return read_session_work_events(conn, session_id=session_id)[position]


def read_session_work_events(
    conn: sqlite3.Connection,
    *,
    session_id: str,
) -> dict[int, ArchiveSessionWorkEvent]:
    """Read deterministic session work events keyed by position."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT event_id, session_id, position, work_event_type, summary, confidence,
            start_index, end_index, started_at_ms, ended_at_ms, duration_ms,
            file_paths_json, tools_used_json,
            input_high_water_mark, input_high_water_mark_source,
            evidence_json, inference_json, search_text
        FROM session_work_events
        WHERE session_id = ?
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    return {
        row["position"]: ArchiveSessionWorkEvent(
            event_id=row["event_id"],
            session_id=row["session_id"],
            position=row["position"],
            work_event_type=row["work_event_type"],
            summary=row["summary"],
            confidence=row["confidence"],
            start_index=row["start_index"],
            end_index=row["end_index"],
            started_at_ms=row["started_at_ms"],
            ended_at_ms=row["ended_at_ms"],
            duration_ms=row["duration_ms"],
            file_paths=_json_tuple(row["file_paths_json"]),
            tools_used=_json_tuple(row["tools_used_json"]),
            evidence=_json_loads(row["evidence_json"]),
            inference=_json_loads(row["inference_json"]),
            search_text=row["search_text"],
            input_high_water_mark=row["input_high_water_mark"],
            input_high_water_mark_source=row["input_high_water_mark_source"],
        )
        for row in rows
    }


def upsert_session_phase(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    position: int,
    start_index: int = 0,
    end_index: int = 0,
    started_at_ms: int | None = None,
    ended_at_ms: int | None = None,
    duration_ms: int = 0,
    tool_counts: dict[str, int] | None = None,
    word_count: int = 0,
    evidence: dict[str, object] | None = None,
    inference: dict[str, object] | None = None,
    search_text: str = "",
    input_high_water_mark: str | None = None,
    input_high_water_mark_source: str | None = None,
) -> ArchiveSessionPhase:
    """Upsert one deterministic session phase row."""
    from polylogue.storage.sqlite.archive_tiers.write import _json_dumps

    conn.execute("PRAGMA foreign_keys = ON")
    with conn:
        conn.execute(
            """
            INSERT INTO session_phases (
                session_id, position, start_index, end_index,
                started_at_ms, ended_at_ms, duration_ms, tool_counts_json, word_count,
                input_high_water_mark, input_high_water_mark_source,
                evidence_json, inference_json, search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, position) DO UPDATE SET
                start_index = excluded.start_index,
                end_index = excluded.end_index,
                started_at_ms = excluded.started_at_ms,
                ended_at_ms = excluded.ended_at_ms,
                duration_ms = excluded.duration_ms,
                tool_counts_json = excluded.tool_counts_json,
                word_count = excluded.word_count,
                input_high_water_mark = excluded.input_high_water_mark,
                input_high_water_mark_source = excluded.input_high_water_mark_source,
                evidence_json = excluded.evidence_json,
                inference_json = excluded.inference_json,
                search_text = excluded.search_text
            """,
            (
                session_id,
                position,
                start_index,
                end_index,
                started_at_ms,
                ended_at_ms,
                duration_ms,
                _json_dumps(tool_counts or {}),
                word_count,
                input_high_water_mark,
                input_high_water_mark_source,
                _json_dumps(evidence or {}),
                _json_dumps(inference or {}),
                search_text,
            ),
        )
        _refresh_session_profile_count(conn, session_id, table="session_phases", column="phase_count")
    return read_session_phases(conn, session_id=session_id)[position]


def read_session_phases(
    conn: sqlite3.Connection,
    *,
    session_id: str,
) -> dict[int, ArchiveSessionPhase]:
    """Read deterministic session phases keyed by position."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT phase_id, session_id, position, start_index, end_index,
            started_at_ms, ended_at_ms, duration_ms, tool_counts_json, word_count,
            input_high_water_mark, input_high_water_mark_source,
            evidence_json, inference_json, search_text
        FROM session_phases
        WHERE session_id = ?
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    return {
        row["position"]: ArchiveSessionPhase(
            phase_id=row["phase_id"],
            session_id=row["session_id"],
            position=row["position"],
            start_index=row["start_index"],
            end_index=row["end_index"],
            started_at_ms=row["started_at_ms"],
            ended_at_ms=row["ended_at_ms"],
            duration_ms=row["duration_ms"],
            tool_counts={str(key): _json_int(value) for key, value in _json_loads(row["tool_counts_json"]).items()},
            word_count=row["word_count"],
            evidence=_json_loads(row["evidence_json"]),
            inference=_json_loads(row["inference_json"]),
            search_text=row["search_text"],
            input_high_water_mark=row["input_high_water_mark"],
            input_high_water_mark_source=row["input_high_water_mark_source"],
        )
        for row in rows
    }


__all__ = [
    "ArchiveSessionPhase",
    "ArchiveSessionTag",
    "ArchiveSessionWorkEvent",
    "read_session_phases",
    "read_session_tags",
    "read_session_work_events",
    "upsert_session_phase",
    "upsert_session_tag",
    "upsert_session_work_event",
]
