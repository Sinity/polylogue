"""Minimal archive index parsed-session writer/read helpers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from polylogue.archive.viewport.viewports import ToolCategory, classify_tool
from polylogue.core.enums import BlockType, ContentBlockType, PasteBoundary
from polylogue.core.identity_law import message_id as archive_message_id
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.json import JSONValue
from polylogue.core.sources import origin_from_provider
from polylogue.core.timestamps import parse_timestamp
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.storage.search.query_support import normalize_fts5_query


@dataclass(frozen=True, slots=True)
class ArchiveBlockRow:
    block_id: str
    message_id: str
    block_type: str
    text: str | None
    tool_name: str | None = None
    tool_id: str | None = None
    semantic_type: str | None = None
    tool_input: str | None = None
    metadata: str | None = None
    language: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveAttachmentRow:
    attachment_id: str
    message_id: str | None
    display_name: str | None = None
    media_type: str | None = None
    byte_count: int = 0
    upload_origin: str | None = None
    source_url: str | None = None
    caption: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveMessageRow:
    message_id: str
    native_id: str | None
    role: str
    position: int
    variant_index: int
    is_active_path: bool
    is_active_leaf: bool
    blocks: tuple[ArchiveBlockRow, ...]
    message_type: str = "message"
    word_count: int = 0
    has_tool_use: bool = False
    has_thinking: bool = False
    has_paste: bool = False
    occurred_at: str | None = None
    duration_ms: int = 0
    parent_message_id: str | None = None
    attachments: tuple[ArchiveAttachmentRow, ...] = ()


@dataclass(frozen=True, slots=True)
class ArchiveAgentPolicy:
    policy_id: str
    session_id: str
    position: int
    approval_policy: str | None
    sandbox_policy: str | None
    network_policy: str | None
    observed_at_ms: int | None
    source_message_id: str | None


@dataclass(frozen=True, slots=True)
class ArchiveSessionEnvelope:
    session_id: str
    native_id: str
    origin: str
    title: str | None
    active_leaf_message_id: str | None
    messages: tuple[ArchiveMessageRow, ...]
    parent_session_id: str | None = None
    root_session_id: str | None = None
    branch_type: str | None = None
    title_source: str | None = None
    instructions_text: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    working_directories: tuple[str, ...] = ()
    git_branch: str | None = None
    git_repository_url: str | None = None
    orphan_attachments: tuple[ArchiveAttachmentRow, ...] = ()


@dataclass(frozen=True, slots=True)
class ArchiveInsightMaterialization:
    insight_type: str
    session_id: str
    materializer_version: int
    materialized_at_ms: int
    source_updated_at_ms: int | None
    source_sort_key_ms: int | None
    input_high_water_mark_ms: int | None
    input_row_count: int


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


@dataclass(frozen=True, slots=True)
class ArchiveSessionPhase:
    phase_id: str
    session_id: str
    position: int
    phase_type: str
    confidence: float
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


def write_parsed_session_to_archive(
    conn: sqlite3.Connection,
    session: ParsedSession,
    *,
    content_hash: str | None = None,
    raw_id: str | None = None,
    merge_append: bool = False,
) -> str:
    """Write one parsed session into an initialized archive index DB."""
    conn.execute("PRAGMA foreign_keys = ON")
    origin = origin_from_provider(session.source_name)
    native_id = session.provider_session_id
    session_id = archive_session_id(origin.value, native_id)
    messages = _normalized_messages(session.messages)
    active_leaf_message_id = _active_leaf_message_id(session_id, messages, session.active_leaf_message_provider_id)
    session_content_hash = (
        bytes.fromhex(content_hash) if content_hash is not None else _hash_bytes("session", origin.value, native_id)
    )

    with conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, branch_type, active_leaf_message_id,
                title, title_source, git_branch, git_repository_url, commit_hash,
                instructions_text, reported_duration_ms,
                message_count, word_count, tool_use_count, thinking_count,
                paste_count, user_message_count, assistant_message_count, system_message_count,
                tool_message_count, user_word_count, assistant_word_count,
                content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(origin, native_id) DO UPDATE SET
                raw_id = excluded.raw_id,
                branch_type = excluded.branch_type,
                active_leaf_message_id = excluded.active_leaf_message_id,
                title = COALESCE(excluded.title, sessions.title),
                title_source = COALESCE(excluded.title_source, sessions.title_source),
                git_branch = excluded.git_branch,
                git_repository_url = excluded.git_repository_url,
                commit_hash = excluded.commit_hash,
                instructions_text = COALESCE(excluded.instructions_text, sessions.instructions_text),
                reported_duration_ms = excluded.reported_duration_ms,
                content_hash = excluded.content_hash,
                created_at_ms = COALESCE(sessions.created_at_ms, excluded.created_at_ms),
                updated_at_ms = MAX(COALESCE(sessions.updated_at_ms, 0), COALESCE(excluded.updated_at_ms, 0))
            """,
            (
                native_id,
                origin.value,
                raw_id,
                _enum_value(session.branch_type),
                active_leaf_message_id,
                session.title,
                _enum_value(session.title_source),
                session.git_branch,
                session.git_repository_url,
                session.git_commit_hash,
                session.instructions_text,
                session.reported_duration_ms,
                len(messages),
                sum(_word_count(message.text) for message in messages),
                sum(_has_block(message, ContentBlockType.TOOL_USE) for message in messages),
                sum(_has_block(message, ContentBlockType.THINKING) for message in messages),
                sum(_has_paste(message) for message in messages),
                sum(1 for message in messages if _enum_value(message.role) == "user"),
                sum(1 for message in messages if _enum_value(message.role) == "assistant"),
                sum(1 for message in messages if _enum_value(message.role) == "system"),
                sum(1 for message in messages if _enum_value(message.role) == "tool"),
                sum(_word_count(message.text) for message in messages if _enum_value(message.role) == "user"),
                sum(_word_count(message.text) for message in messages if _enum_value(message.role) == "assistant"),
                session_content_hash,
                _timestamp_ms(session.created_at),
                _timestamp_ms(session.updated_at),
            ),
        )
        position_offset = 0
        if merge_append:
            row = conn.execute(
                "SELECT COALESCE(MAX(position) + 1, 0) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            position_offset = int(row[0] or 0) if row is not None else 0
            conn.execute("UPDATE messages SET is_active_leaf = 0 WHERE session_id = ?", (session_id,))
            active_leaf_message_id = _active_leaf_message_id(
                session_id,
                messages,
                session.active_leaf_message_provider_id,
                position_offset=position_offset,
            )
            conn.execute(
                "UPDATE sessions SET active_leaf_message_id = ? WHERE session_id = ?",
                (active_leaf_message_id, session_id),
            )
        else:
            _clear_session_projection_rows(conn, session_id)
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        _write_messages(conn, session_id, messages, position_offset=position_offset)
        _write_blocks(conn, session_id, messages, position_offset=position_offset)
        _write_attachments(conn, session_id, messages, session.attachments, position_offset=position_offset)
        _write_paste_spans(conn, session_id, messages, position_offset=position_offset)
        _write_parent_links(conn, session_id, messages, position_offset=position_offset)
        _write_session_link(conn, session_id, session)
        _write_session_events(conn, session_id, messages, session.session_events, position_offset=position_offset)
        _write_working_dirs(conn, session_id, session.working_directories)
        _write_repo_edges(conn, session_id, session)
        _write_reported_costs(conn, session_id, session)
        _refresh_session_counts(conn, session_id)
        _resolve_session_graph(conn, session_id, native_id, origin.value)
    return session_id


def _clear_session_projection_rows(conn: sqlite3.Connection, session_id: str) -> None:
    """Clear rows owned by parsed-session replacement before rewriting it."""
    for table in (
        "attachment_refs",
        "session_events",
        "session_agent_policies",
        "session_working_dirs",
        "session_repos",
        "session_commits",
        "session_reported_costs",
        "session_model_usage",
    ):
        conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM session_links WHERE src_session_id = ?", (session_id,))


def upsert_session_profile_costs(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    cost_credits: float | None = None,
    cost_usd: float | None = None,
    cost_is_estimated: bool = False,
    cost_provenance: str | None = None,
    priced_with: str | None = None,
    priced_at_ms: int | None = None,
) -> None:
    """Upsert a minimal cost/pricing slice for an existing profile row."""
    conn.execute("PRAGMA foreign_keys = ON")
    with conn:
        conn.execute(
            """
            INSERT INTO session_profiles (
                session_id, cost_credits, cost_usd, cost_is_estimated, cost_provenance, priced_with, priced_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                cost_credits = excluded.cost_credits,
                cost_usd = excluded.cost_usd,
                cost_is_estimated = excluded.cost_is_estimated,
                cost_provenance = excluded.cost_provenance,
                priced_with = excluded.priced_with,
                priced_at_ms = excluded.priced_at_ms
            """,
            (
                session_id,
                cost_credits,
                cost_usd,
                1 if cost_is_estimated else 0,
                cost_provenance,
                priced_with,
                priced_at_ms,
            ),
        )


def apply_insight_materialization(
    conn: sqlite3.Connection,
    *,
    insight_type: str,
    session_id: str,
    materializer_version: int,
    materialized_at_ms: int,
    source_updated_at_ms: int | None = None,
    source_sort_key_ms: int | None = None,
    input_high_water_mark_ms: int | None = None,
    input_row_count: int = 0,
) -> None:
    """Stamp one session-insight materialization row without committing.

    The bulk insight rebuild (``rebuild_session_insights_sync``) materializes
    every insight table inside one transaction so a SIGKILL mid-rebuild rolls
    the WAL back to the prior insights. It therefore stamps materialization
    through this no-commit primitive; the committing ``upsert_*`` wrapper below
    is for callers that stamp a single insight as its own unit of work.
    """
    conn.execute(
        """
        INSERT INTO insight_materialization (
            insight_type, session_id, materializer_version, materialized_at_ms,
            source_updated_at_ms, source_sort_key_ms, input_high_water_mark_ms, input_row_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(insight_type, session_id) DO UPDATE SET
            materializer_version = excluded.materializer_version,
            materialized_at_ms = excluded.materialized_at_ms,
            source_updated_at_ms = excluded.source_updated_at_ms,
            source_sort_key_ms = excluded.source_sort_key_ms,
            input_high_water_mark_ms = excluded.input_high_water_mark_ms,
            input_row_count = excluded.input_row_count
        """,
        (
            insight_type,
            session_id,
            materializer_version,
            materialized_at_ms,
            source_updated_at_ms,
            source_sort_key_ms,
            input_high_water_mark_ms,
            input_row_count,
        ),
    )


def upsert_insight_materialization(
    conn: sqlite3.Connection,
    *,
    insight_type: str,
    session_id: str,
    materializer_version: int,
    materialized_at_ms: int,
    source_updated_at_ms: int | None = None,
    source_sort_key_ms: int | None = None,
    input_high_water_mark_ms: int | None = None,
    input_row_count: int = 0,
) -> ArchiveInsightMaterialization:
    """Upsert the shared materialization state for one session insight."""
    conn.execute("PRAGMA foreign_keys = ON")
    with conn:
        apply_insight_materialization(
            conn,
            insight_type=insight_type,
            session_id=session_id,
            materializer_version=materializer_version,
            materialized_at_ms=materialized_at_ms,
            source_updated_at_ms=source_updated_at_ms,
            source_sort_key_ms=source_sort_key_ms,
            input_high_water_mark_ms=input_high_water_mark_ms,
            input_row_count=input_row_count,
        )
    return read_insight_materialization(conn, insight_type, session_id)


def read_insight_materialization(
    conn: sqlite3.Connection,
    insight_type: str,
    session_id: str,
) -> ArchiveInsightMaterialization:
    """Read the shared materialization state for one session insight."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT insight_type, session_id, materializer_version, materialized_at_ms,
            source_updated_at_ms, source_sort_key_ms, input_high_water_mark_ms, input_row_count
        FROM insight_materialization
        WHERE insight_type = ? AND session_id = ?
        """,
        (insight_type, session_id),
    ).fetchone()
    if row is None:
        raise KeyError(f"{insight_type}:{session_id}")
    return ArchiveInsightMaterialization(
        insight_type=row["insight_type"],
        session_id=row["session_id"],
        materializer_version=row["materializer_version"],
        materialized_at_ms=row["materialized_at_ms"],
        source_updated_at_ms=row["source_updated_at_ms"],
        source_sort_key_ms=row["source_sort_key_ms"],
        input_high_water_mark_ms=row["input_high_water_mark_ms"],
        input_row_count=row["input_row_count"],
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
    return read_session_tags(conn, session_id=session_id, tag_source=tag_source)[normalized_tag]


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
) -> ArchiveSessionWorkEvent:
    """Upsert one deterministic session work-event row."""
    conn.execute("PRAGMA foreign_keys = ON")
    with conn:
        conn.execute(
            """
            INSERT INTO session_work_events (
                session_id, position, work_event_type, summary, confidence,
                start_index, end_index, started_at_ms, ended_at_ms, duration_ms,
                file_paths_json, tools_used_json, evidence_json, inference_json, search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            file_paths_json, tools_used_json, evidence_json, inference_json, search_text
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
        )
        for row in rows
    }


def upsert_session_phase(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    position: int,
    phase_type: str,
    confidence: float = 0.0,
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
) -> ArchiveSessionPhase:
    """Upsert one deterministic session phase row."""
    conn.execute("PRAGMA foreign_keys = ON")
    with conn:
        conn.execute(
            """
            INSERT INTO session_phases (
                session_id, position, phase_type, confidence, start_index, end_index,
                started_at_ms, ended_at_ms, duration_ms, tool_counts_json, word_count,
                evidence_json, inference_json, search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, position) DO UPDATE SET
                phase_type = excluded.phase_type,
                confidence = excluded.confidence,
                start_index = excluded.start_index,
                end_index = excluded.end_index,
                started_at_ms = excluded.started_at_ms,
                ended_at_ms = excluded.ended_at_ms,
                duration_ms = excluded.duration_ms,
                tool_counts_json = excluded.tool_counts_json,
                word_count = excluded.word_count,
                evidence_json = excluded.evidence_json,
                inference_json = excluded.inference_json,
                search_text = excluded.search_text
            """,
            (
                session_id,
                position,
                phase_type,
                confidence,
                start_index,
                end_index,
                started_at_ms,
                ended_at_ms,
                duration_ms,
                _json_dumps(tool_counts or {}),
                word_count,
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
        SELECT phase_id, session_id, position, phase_type, confidence, start_index, end_index,
            started_at_ms, ended_at_ms, duration_ms, tool_counts_json, word_count,
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
            phase_type=row["phase_type"],
            confidence=row["confidence"],
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
        )
        for row in rows
    }


def read_archive_session_envelope(conn: sqlite3.Connection, session_id: str) -> ArchiveSessionEnvelope:
    """Read a compact archive envelope used by self-verify and writer tests."""
    conn.row_factory = sqlite3.Row
    session = conn.execute(
        """
        SELECT session_id, native_id, origin, title, active_leaf_message_id,
               parent_session_id, root_session_id, branch_type,
               title_source, instructions_text,
               created_at_ms, updated_at_ms, git_branch, git_repository_url
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if session is None:
        raise KeyError(session_id)
    working_directories = tuple(
        str(row["path"])
        for row in conn.execute(
            """
            SELECT path
            FROM session_working_dirs
            WHERE session_id = ?
            ORDER BY position, path
            """,
            (session_id,),
        ).fetchall()
    )

    attachment_rows = conn.execute(
        """
        SELECT r.message_id AS message_id, a.attachment_id AS attachment_id,
               a.display_name AS display_name, a.media_type AS media_type, a.byte_count AS byte_count,
               r.upload_origin AS upload_origin, r.source_url AS source_url, r.caption AS caption
        FROM attachment_refs r
        JOIN attachments a ON a.attachment_id = r.attachment_id
        WHERE r.session_id = ?
        ORDER BY r.message_id, a.attachment_id
        """,
        (session_id,),
    ).fetchall()
    attachments_by_message: dict[str | None, list[ArchiveAttachmentRow]] = {}
    for attachment in attachment_rows:
        attachments_by_message.setdefault(attachment["message_id"], []).append(
            ArchiveAttachmentRow(
                attachment_id=attachment["attachment_id"],
                message_id=attachment["message_id"],
                display_name=attachment["display_name"],
                media_type=attachment["media_type"],
                byte_count=int(attachment["byte_count"] or 0),
                upload_origin=attachment["upload_origin"],
                source_url=attachment["source_url"],
                caption=attachment["caption"],
            )
        )

    message_rows = conn.execute(
        """
        SELECT message_id, native_id, role, position, variant_index, is_active_path, is_active_leaf,
               message_type, word_count, has_tool_use, has_thinking, has_paste, occurred_at_ms,
               duration_ms, parent_message_id
        FROM messages
        WHERE session_id = ?
        ORDER BY position, variant_index
        """,
        (session_id,),
    ).fetchall()
    messages: list[ArchiveMessageRow] = []
    for message in message_rows:
        block_rows = conn.execute(
            """
            SELECT block_id, message_id, block_type, text, tool_name, tool_id, semantic_type,
                   tool_input, language
            FROM blocks
            WHERE message_id = ?
            ORDER BY position
            """,
            (message["message_id"],),
        ).fetchall()
        messages.append(
            ArchiveMessageRow(
                message_id=message["message_id"],
                native_id=message["native_id"],
                role=message["role"],
                position=message["position"],
                variant_index=message["variant_index"],
                is_active_path=bool(message["is_active_path"]),
                is_active_leaf=bool(message["is_active_leaf"]),
                blocks=tuple(
                    ArchiveBlockRow(
                        block_id=block["block_id"],
                        message_id=block["message_id"],
                        block_type=block["block_type"],
                        text=block["text"],
                        tool_name=block["tool_name"],
                        tool_id=block["tool_id"],
                        semantic_type=block["semantic_type"],
                        tool_input=block["tool_input"],
                        language=block["language"],
                    )
                    for block in block_rows
                ),
                message_type=message["message_type"],
                word_count=int(message["word_count"] or 0),
                has_tool_use=bool(message["has_tool_use"]),
                has_thinking=bool(message["has_thinking"]),
                has_paste=bool(message["has_paste"]),
                occurred_at=_iso_from_ms(message["occurred_at_ms"]),
                duration_ms=int(message["duration_ms"] or 0),
                parent_message_id=message["parent_message_id"],
                attachments=tuple(attachments_by_message.get(message["message_id"], ())),
            )
        )

    return ArchiveSessionEnvelope(
        session_id=session["session_id"],
        native_id=session["native_id"],
        origin=session["origin"],
        title=session["title"],
        active_leaf_message_id=session["active_leaf_message_id"],
        messages=tuple(messages),
        parent_session_id=session["parent_session_id"],
        root_session_id=session["root_session_id"],
        branch_type=session["branch_type"],
        title_source=session["title_source"],
        instructions_text=session["instructions_text"],
        created_at=_iso_from_ms(session["created_at_ms"]),
        updated_at=_iso_from_ms(session["updated_at_ms"]),
        working_directories=working_directories,
        git_branch=session["git_branch"],
        git_repository_url=session["git_repository_url"],
        orphan_attachments=tuple(attachments_by_message.get(None, ())),
    )


def read_session_agent_policies(conn: sqlite3.Connection, session_id: str) -> list[ArchiveAgentPolicy]:
    """Read all agent-policy rows for a session, ordered by position."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT policy_id, session_id, position, approval_policy,
               sandbox_policy, network_policy, observed_at_ms, source_message_id
        FROM session_agent_policies
        WHERE session_id = ?
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    return [
        ArchiveAgentPolicy(
            policy_id=str(row["policy_id"]),
            session_id=str(row["session_id"]),
            position=int(row["position"]),
            approval_policy=row["approval_policy"],
            sandbox_policy=row["sandbox_policy"],
            network_policy=row["network_policy"],
            observed_at_ms=row["observed_at_ms"],
            source_message_id=row["source_message_id"],
        )
        for row in rows
    ]


def search_archive_blocks(conn: sqlite3.Connection, query: str) -> list[str]:
    """Return block ids matched by the archive contentless FTS table."""
    match_query = normalize_fts5_query(query)
    if match_query is None:
        return []
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT b.block_id
        FROM messages_fts f
        JOIN blocks b ON b.rowid = f.rowid
        WHERE f.text MATCH ?
        ORDER BY rank
        """,
        (match_query,),
    ).fetchall()
    return [row["block_id"] for row in rows]


def rebuild_archive_messages_fts(conn: sqlite3.Connection) -> int:
    """Rebuild the archive message FTS index from canonical ``blocks`` rows."""
    conn.execute("DELETE FROM messages_fts")
    conn.execute(
        """
        INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
        SELECT rowid, block_id, message_id, session_id, block_type, search_text
        FROM blocks
        WHERE search_text != ''
        """
    )
    row = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()
    return int(row[0] if row is not None else 0)


def _write_messages(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
) -> None:
    for fallback_position, message in enumerate(messages):
        position = position_offset + (message.position if message.position is not None else fallback_position)
        variant_index = message.variant_index if message.variant_index is not None else 0
        conn.execute(
            """
            INSERT OR REPLACE INTO messages (
                session_id, native_id, parent_message_id, position, role, message_type,
                model_name, model_effort, has_tool_use, has_thinking, has_paste, paste_boundary,
                variant_index, is_active_path, is_active_leaf, word_count,
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                duration_ms, content_hash, occurred_at_ms
            ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                message.provider_message_id or None,
                position,
                _enum_value(message.role),
                _enum_value(message.message_type),
                message.model_name,
                message.model_effort,
                _has_block(message, ContentBlockType.TOOL_USE),
                _has_block(message, ContentBlockType.THINKING),
                _has_paste(message),
                _paste_boundary(message),
                variant_index,
                1 if message.is_active_path is not False else 0,
                1 if message.is_active_leaf else 0,
                _word_count(message.text),
                message.input_tokens,
                message.output_tokens,
                message.cache_read_tokens,
                message.cache_write_tokens,
                message.duration_ms,
                _hash_bytes(
                    "message", session_id, message.provider_message_id or "", str(position), str(variant_index)
                ),
                message.occurred_at_ms if message.occurred_at_ms is not None else _timestamp_ms(message.timestamp),
            ),
        )


def _write_blocks(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
) -> None:
    for fallback_position, message in enumerate(messages):
        message_id = _message_id(session_id, message, fallback_position, position_offset=position_offset)
        blocks = _message_blocks(message)
        for position, block in enumerate(blocks):
            conn.execute(
                """
                INSERT OR REPLACE INTO blocks (
                    message_id, session_id, position, block_type, text, tool_name,
                    tool_id, tool_input, semantic_type, media_type, language
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    position,
                    _block_type(block).value,
                    block.text,
                    block.tool_name,
                    block.tool_id,
                    _json_dumps(block.tool_input) if block.tool_input is not None else None,
                    _semantic_type(block),
                    block.media_type,
                    _block_language(block),
                ),
            )


def _refresh_session_counts(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        """
        UPDATE sessions
        SET message_count = (SELECT COUNT(*) FROM messages WHERE session_id = sessions.session_id),
            word_count = COALESCE((SELECT SUM(word_count) FROM messages WHERE session_id = sessions.session_id), 0),
            tool_use_count = COALESCE((SELECT SUM(has_tool_use) FROM messages WHERE session_id = sessions.session_id), 0),
            thinking_count = COALESCE((SELECT SUM(has_thinking) FROM messages WHERE session_id = sessions.session_id), 0),
            paste_count = COALESCE((SELECT SUM(has_paste) FROM messages WHERE session_id = sessions.session_id), 0),
            user_message_count = (
                SELECT COUNT(*) FROM messages WHERE session_id = sessions.session_id AND role = 'user'
            ),
            assistant_message_count = (
                SELECT COUNT(*) FROM messages WHERE session_id = sessions.session_id AND role = 'assistant'
            ),
            system_message_count = (
                SELECT COUNT(*) FROM messages WHERE session_id = sessions.session_id AND role = 'system'
            ),
            tool_message_count = (
                SELECT COUNT(*) FROM messages WHERE session_id = sessions.session_id AND role = 'tool'
            ),
            user_word_count = COALESCE((
                SELECT SUM(word_count) FROM messages WHERE session_id = sessions.session_id AND role = 'user'
            ), 0),
            assistant_word_count = COALESCE((
                SELECT SUM(word_count) FROM messages WHERE session_id = sessions.session_id AND role = 'assistant'
            ), 0)
        WHERE session_id = ?
        """,
        (session_id,),
    )


def _write_attachments(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    attachments: Iterable[ParsedAttachment],
    *,
    position_offset: int = 0,
) -> None:
    by_native_message_id = {
        message.provider_message_id: _message_id(
            session_id, message, fallback_position, position_offset=position_offset
        )
        for fallback_position, message in enumerate(messages)
        if message.provider_message_id
    }
    for attachment in attachments:
        attachment_id = _attachment_id(session_id, attachment)
        message_id = (
            by_native_message_id.get(attachment.message_provider_id) if attachment.message_provider_id else None
        )
        if message_id is None:
            continue
        conn.execute(
            """
            INSERT INTO attachments (
                attachment_id, display_name, media_type, byte_count, blob_hash, ref_count
            ) VALUES (?, ?, ?, ?, ?, 0)
            ON CONFLICT(attachment_id) DO UPDATE SET
                display_name = COALESCE(excluded.display_name, attachments.display_name),
                media_type = COALESCE(excluded.media_type, attachments.media_type),
                byte_count = excluded.byte_count,
                blob_hash = excluded.blob_hash
            """,
            (
                attachment_id,
                attachment.name,
                attachment.mime_type,
                attachment.size_bytes or 0,
                _attachment_blob_hash(attachment_id, attachment),
            ),
        )
        ref_position = _attachment_position(attachment)
        conn.execute(
            """
            INSERT OR REPLACE INTO attachment_refs (
                attachment_id, session_id, message_id, position, upload_origin, source_url, caption
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attachment_id,
                session_id,
                message_id,
                ref_position,
                attachment.upload_origin,
                _attachment_source_url(attachment),
                _attachment_caption(attachment),
            ),
        )
        ref_id = f"{message_id}:attachment:{ref_position}"
        _write_attachment_native_ids(conn, ref_id, attachment)
    conn.execute(
        """
        UPDATE attachments
        SET ref_count = (
            SELECT COUNT(*) FROM attachment_refs WHERE attachment_refs.attachment_id = attachments.attachment_id
        )
        """
    )


def _write_paste_spans(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
) -> None:
    for fallback_position, message in enumerate(messages):
        if not _has_paste(message):
            continue
        message_id = _message_id(session_id, message, fallback_position, position_offset=position_offset)
        text = message.text or ""
        for evidence in message.paste_spans:
            boundary = PasteBoundary(evidence.boundary_state)
            start_offset = evidence.start_offset if evidence.start_offset is not None else 0
            end_offset = evidence.end_offset if evidence.end_offset is not None else len(text)
            conn.execute(
                """
                INSERT OR REPLACE INTO paste_spans (
                    message_id, session_id, position, start_offset, end_offset, boundary_state,
                    source_event_id, source_marker, content_hash, observed_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    evidence.position,
                    start_offset,
                    end_offset,
                    boundary.value,
                    evidence.source_event_id,
                    evidence.source_marker,
                    evidence.content_hash or _hash_bytes("paste", message_id, str(evidence.position), text),
                    evidence.observed_at_ms
                    if evidence.observed_at_ms is not None
                    else message.occurred_at_ms
                    if message.occurred_at_ms is not None
                    else _timestamp_ms(message.timestamp),
                ),
            )


def _write_parent_links(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
) -> None:
    by_native_id = {
        message.provider_message_id: _message_id(
            session_id, message, fallback_position, position_offset=position_offset
        )
        for fallback_position, message in enumerate(messages)
        if message.provider_message_id
    }
    for fallback_position, message in enumerate(messages):
        if not message.parent_message_provider_id:
            continue
        parent_message_id = by_native_id.get(message.parent_message_provider_id)
        if parent_message_id is None:
            continue
        conn.execute(
            """
            UPDATE messages
            SET parent_message_id = ?
            WHERE message_id = ?
            """,
            (
                parent_message_id,
                _message_id(session_id, message, fallback_position, position_offset=position_offset),
            ),
        )


def _write_session_link(conn: sqlite3.Connection, session_id: str, session: ParsedSession) -> None:
    if not session.parent_session_provider_id:
        return
    link_type = _enum_value(session.branch_type) or "continuation"
    conn.execute(
        """
        INSERT OR REPLACE INTO session_links (
            src_session_id, dst_origin, dst_native_id, link_type, status, method, confidence, evidence_json, observed_at_ms
        ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?)
        """,
        (
            session_id,
            origin_from_provider(session.source_name).value,
            session.parent_session_provider_id,
            link_type,
            "parser-parent",
            1.0,
            _json_dumps({"parent_session_provider_id": session.parent_session_provider_id}),
            _timestamp_ms(session.updated_at) or _timestamp_ms(session.created_at) or 0,
        ),
    )


def _resolve_session_graph(conn: sqlite3.Connection, session_id: str, native_id: str, origin: str) -> None:
    conn.execute(
        """
        UPDATE sessions
        SET root_session_id = session_id
        WHERE session_id = ? AND root_session_id IS NULL
        """,
        (session_id,),
    )
    _resolve_outbound_session_links(conn, session_id, origin)
    inbound_rows = conn.execute(
        """
        SELECT links.src_session_id
        FROM session_links links
        WHERE links.dst_native_id = ?
          AND links.resolved_dst_session_id IS NULL
          AND links.dst_origin = ?
        """,
        (native_id, origin),
    ).fetchall()
    for row in inbound_rows:
        conn.execute(
            """
            UPDATE session_links
            SET resolved_dst_session_id = ?,
                resolved_at_ms = COALESCE(resolved_at_ms, observed_at_ms)
            WHERE src_session_id = ?
              AND dst_native_id = ?
              AND resolved_dst_session_id IS NULL
            """,
            (session_id, row[0], native_id),
        )

    impacted_session_ids = {session_id, *(str(row[0]) for row in inbound_rows)}
    old_root_ids = _root_ids(conn, impacted_session_ids)
    for impacted_session_id in impacted_session_ids:
        _refresh_session_projection(conn, impacted_session_id, seen=set())
    root_ids_to_refresh = old_root_ids | _root_ids(conn, impacted_session_ids)
    for root_session_id in root_ids_to_refresh:
        _refresh_thread(conn, root_session_id)


def _resolve_outbound_session_links(conn: sqlite3.Connection, session_id: str, origin: str) -> None:
    conn.execute(
        """
        UPDATE session_links
        SET resolved_dst_session_id = (
                SELECT dst.session_id
                FROM sessions dst
                WHERE dst.native_id = session_links.dst_native_id
                  AND dst.origin = session_links.dst_origin
                LIMIT 1
            ),
            resolved_at_ms = COALESCE(resolved_at_ms, observed_at_ms)
        WHERE src_session_id = ?
          AND resolved_dst_session_id IS NULL
          AND EXISTS (
                SELECT 1
                FROM sessions dst
                WHERE dst.native_id = session_links.dst_native_id
                  AND dst.origin = session_links.dst_origin
          )
        """,
        (session_id,),
    )


def _refresh_session_projection(conn: sqlite3.Connection, session_id: str, *, seen: set[str]) -> None:
    if session_id in seen:
        return
    seen.add(session_id)
    parent_link = conn.execute(
        """
        SELECT resolved_dst_session_id, link_type
        FROM session_links
        WHERE src_session_id = ? AND resolved_dst_session_id IS NOT NULL
        ORDER BY observed_at_ms IS NULL, observed_at_ms, dst_origin, dst_native_id, link_type
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    if parent_link is None:
        unresolved_link = conn.execute(
            """
            SELECT link_type
            FROM session_links
            WHERE src_session_id = ?
            ORDER BY observed_at_ms IS NULL, observed_at_ms, dst_origin, dst_native_id, link_type
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        branch_type: str | None
        if unresolved_link is not None:
            branch_type = str(unresolved_link[0])
        else:
            existing_branch = conn.execute(
                "SELECT branch_type FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            branch_type = str(existing_branch[0]) if existing_branch is not None and existing_branch[0] else None
        conn.execute(
            """
            UPDATE sessions
            SET parent_session_id = NULL,
                root_session_id = session_id,
                branch_type = ?
            WHERE session_id = ?
            """,
            (branch_type, session_id),
        )
        return

    parent_session_id = str(parent_link[0])
    _refresh_session_projection(conn, parent_session_id, seen=seen)
    parent_root_row = conn.execute(
        """
        SELECT COALESCE(root_session_id, session_id)
        FROM sessions
        WHERE session_id = ?
        """,
        (parent_session_id,),
    ).fetchone()
    parent_root_id = str(parent_root_row[0]) if parent_root_row is not None else parent_session_id
    conn.execute(
        """
        UPDATE sessions
        SET parent_session_id = ?,
            root_session_id = ?,
            branch_type = ?
        WHERE session_id = ?
        """,
        (parent_session_id, parent_root_id, str(parent_link[1]), session_id),
    )


def _refresh_thread(conn: sqlite3.Connection, root_session_id: str) -> None:
    root = conn.execute(
        """
        SELECT session_id, origin, created_at_ms, updated_at_ms, COALESCE(root_session_id, session_id) AS actual_root_id
        FROM sessions
        WHERE session_id = ?
        """,
        (root_session_id,),
    ).fetchone()
    if root is None:
        return
    if root[4] != root_session_id:
        conn.execute("DELETE FROM thread_sessions WHERE thread_id = ?", (root_session_id,))
        conn.execute("DELETE FROM threads WHERE thread_id = ?", (root_session_id,))
        return
    conn.execute(
        """
        INSERT INTO threads (thread_id, created_at_ms, session_count, depth)
        VALUES (?, ?, 0, 0)
        ON CONFLICT(thread_id) DO UPDATE SET
            created_at_ms = excluded.created_at_ms
        """,
        (root_session_id, root[2] or root[3] or 0),
    )
    conn.execute("DELETE FROM thread_sessions WHERE thread_id = ?", (root_session_id,))
    session_rows = conn.execute(
        """
        SELECT session_id
        FROM sessions
        WHERE COALESCE(root_session_id, session_id) = ?
        ORDER BY sort_key_ms IS NULL, sort_key_ms, session_id
        """,
        (root_session_id,),
    ).fetchall()
    for position, row in enumerate(session_rows):
        conn.execute(
            """
            INSERT INTO thread_sessions (thread_id, session_id, position)
            VALUES (?, ?, ?)
            """,
            (root_session_id, row[0], position),
        )
    conn.execute(
        """
        UPDATE threads
        SET session_count = ?,
            depth = ?
        WHERE thread_id = ?
        """,
        (len(session_rows), max(len(session_rows) - 1, 0), root_session_id),
    )


def _root_ids(conn: sqlite3.Connection, session_ids: set[str]) -> set[str]:
    root_ids: set[str] = set()
    for session_id in session_ids:
        row = conn.execute(
            """
            SELECT COALESCE(root_session_id, session_id)
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is not None and row[0]:
            root_ids.add(str(row[0]))
    return root_ids


def _write_session_events(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    events: Iterable[ParsedSessionEvent],
    *,
    position_offset: int = 0,
) -> None:
    by_native_id = {
        message.provider_message_id: _message_id(
            session_id, message, fallback_position, position_offset=position_offset
        )
        for fallback_position, message in enumerate(messages)
        if message.provider_message_id
    }
    position = 0
    for event in events:
        if event.event_type == "compaction":
            conn.execute(
                """
                INSERT OR REPLACE INTO session_events (
                    session_id, source_message_id, position, event_type, summary, occurred_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    by_native_id.get(event.source_message_provider_id or ""),
                    position,
                    event.event_type,
                    _event_summary(event) or "",
                    _timestamp_ms(event.timestamp),
                ),
            )
            position += 1
        elif event.event_type == "agent_policy":
            conn.execute(
                """
                INSERT OR REPLACE INTO session_agent_policies (
                    session_id, source_message_id, position, approval_policy,
                    sandbox_policy, network_policy, observed_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    by_native_id.get(event.source_message_provider_id or ""),
                    position,
                    _payload_string(event.payload, "approval", "approval_policy"),
                    _payload_string(event.payload, "sandbox", "sandbox_policy"),
                    _payload_string(event.payload, "network", "network_policy"),
                    _timestamp_ms(event.timestamp),
                ),
            )
            position += 1


def _write_working_dirs(conn: sqlite3.Connection, session_id: str, working_directories: Iterable[str]) -> None:
    for position, path in enumerate(working_directories):
        conn.execute(
            """
            INSERT OR REPLACE INTO session_working_dirs (session_id, path, position)
            VALUES (?, ?, ?)
            """,
            (session_id, path, position),
        )


def _write_reported_costs(conn: sqlite3.Connection, session_id: str, session: ParsedSession) -> None:
    observed_at_ms = _timestamp_ms(session.updated_at) or _timestamp_ms(session.created_at)
    if session.reported_cost_usd is not None:
        conn.execute(
            """
            INSERT OR REPLACE INTO session_reported_costs (
                session_id, cost_kind, amount, source, observed_at_ms
            ) VALUES (?, 'usd', ?, 'origin_reported', ?)
            """,
            (session_id, session.reported_cost_usd, observed_at_ms),
        )
    model_names = {model_name.strip() for model_name in session.models_used if model_name.strip()}
    model_names.update(message.model_name.strip() for message in session.messages if message.model_name)
    for model_name in sorted(model_names):
        conn.execute(
            """
            INSERT OR REPLACE INTO session_model_usage (
                session_id, model_name, cost_provenance
            ) VALUES (?, ?, 'origin_reported')
            """,
            (session_id, model_name),
        )
    _aggregate_message_tokens_into_model_usage(conn, session_id)


def _aggregate_message_tokens_into_model_usage(conn: sqlite3.Connection, session_id: str) -> None:
    """Aggregate per-message token counts into session_model_usage and compute cost_usd.

    Called after messages are written (and after skeleton model-usage rows exist).
    Handles both full-write and merge-append paths: it always reads ALL messages
    currently in the DB for the session, so the token sums stay consistent with
    the full message set regardless of append ordering.

    Models with no messages carrying token data keep DEFAULT 0 token counts.
    Models with no catalog price entry get cost_usd = NULL / priced_with = NULL
    (no fabrication).

    Empty or NULL model_name values in the messages table are excluded from
    aggregation (the model is unknown so pricing is impossible).
    """
    import time

    from polylogue.archive.semantic.pricing import PRICING, _normalize_model, estimate_cost

    # Aggregate token counts from the messages table for all known models.
    token_rows = conn.execute(
        """
        SELECT model_name,
               SUM(input_tokens)        AS sum_input,
               SUM(output_tokens)       AS sum_output,
               SUM(cache_read_tokens)   AS sum_cache_read,
               SUM(cache_write_tokens)  AS sum_cache_write,
               COUNT(*)                 AS msg_count
        FROM messages
        WHERE session_id = ?
          AND model_name IS NOT NULL
          AND model_name != ''
        GROUP BY model_name
        """,
        (session_id,),
    ).fetchall()

    if not token_rows:
        return

    # Look up the active catalog_id once so FK can be set when we have a price.
    catalog_row = conn.execute("SELECT catalog_id FROM price_catalogs LIMIT 1").fetchone()
    active_catalog_id: str | None = str(catalog_row[0]) if catalog_row is not None else None
    priced_at_ms = int(time.time() * 1000)

    for row in token_rows:
        model_name: str = str(row[0])
        sum_input: int = int(row[1] or 0)
        sum_output: int = int(row[2] or 0)
        sum_cache_read: int = int(row[3] or 0)
        sum_cache_write: int = int(row[4] or 0)
        msg_count: int = int(row[5] or 0)

        # Compute cost_usd from the curated catalog when a price entry exists.
        # estimate_cost() reads the in-memory PRICING dict so the result always
        # matches the DB-backed model_prices rows seeded from the same source.
        normalized = _normalize_model(model_name)
        billable = sum_input + sum_output + sum_cache_read + sum_cache_write
        if normalized in PRICING and billable > 0:
            cost_usd: float | None = estimate_cost(sum_input, sum_output, model_name, sum_cache_read, sum_cache_write)
            priced_with: str | None = active_catalog_id
            row_priced_at: int | None = priced_at_ms
        else:
            cost_usd = None
            priced_with = None
            row_priced_at = None

        # UPSERT: the skeleton row was created by _write_reported_costs above.
        # For models that somehow landed in messages but not in models_used/
        # session.messages (edge case with merge_append + partial data), we
        # INSERT a fresh row.  For normal cases this is an UPDATE on the
        # existing skeleton row.
        conn.execute(
            """
            INSERT INTO session_model_usage (
                session_id, model_name,
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                message_count,
                priced_with, priced_at_ms, cost_usd,
                cost_provenance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'priced')
            ON CONFLICT(session_id, model_name) DO UPDATE SET
                input_tokens       = excluded.input_tokens,
                output_tokens      = excluded.output_tokens,
                cache_read_tokens  = excluded.cache_read_tokens,
                cache_write_tokens = excluded.cache_write_tokens,
                message_count      = excluded.message_count,
                priced_with        = excluded.priced_with,
                priced_at_ms       = excluded.priced_at_ms,
                cost_usd           = excluded.cost_usd,
                cost_provenance    = excluded.cost_provenance
            """,
            (
                session_id,
                model_name,
                sum_input,
                sum_output,
                sum_cache_read,
                sum_cache_write,
                msg_count,
                priced_with,
                row_priced_at,
                cost_usd,
            ),
        )


def _write_repo_edges(conn: sqlite3.Connection, session_id: str, session: ParsedSession) -> None:
    observed_at_ms = _timestamp_ms(session.updated_at) or _timestamp_ms(session.created_at)
    root_paths = tuple(dict.fromkeys(path.strip() for path in session.working_directories if path.strip()))
    if not root_paths and not session.git_repository_url:
        return

    origin_url = (session.git_repository_url or "").strip()
    for root_path in root_paths or ("",):
        repo_name = _repo_name(origin_url, root_path)
        repo_id = _repo_id(origin_url, root_path)
        conn.execute(
            """
            INSERT INTO repos (origin_url, root_path, repo_name, first_seen_at_ms, last_seen_at_ms)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(origin_url, root_path) DO UPDATE SET
                repo_name = COALESCE(excluded.repo_name, repos.repo_name),
                first_seen_at_ms = MIN(repos.first_seen_at_ms, excluded.first_seen_at_ms),
                last_seen_at_ms = MAX(repos.last_seen_at_ms, excluded.last_seen_at_ms)
            """,
            (origin_url, root_path, repo_name, observed_at_ms or 0, observed_at_ms or 0),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO session_repos (
                session_id, repo_id, branch_name, observed_at_ms
            ) VALUES (?, ?, ?, ?)
            """,
            (session_id, repo_id, session.git_branch or "", observed_at_ms or 0),
        )
        if session.git_commit_hash:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_commits (
                    session_id, commit_sha, repo_id, detection_type, method,
                    confidence, evidence_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    session.git_commit_hash,
                    repo_id,
                    "explicit_ref",
                    "parser-git-meta",
                    1.0,
                    _json_dumps(
                        {
                            "git_repository_url": origin_url or None,
                            "root_path": root_path or None,
                            "git_branch": session.git_branch,
                        }
                    ),
                    observed_at_ms or 0,
                ),
            )


def _normalized_messages(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    active_leaf_count = sum(1 for message in messages if message.is_active_leaf)
    if active_leaf_count == 1 or not messages:
        return messages
    active_leaf_message_id = messages[-1].provider_message_id
    return [
        message.model_copy(update={"is_active_leaf": message.provider_message_id == active_leaf_message_id})
        for message in messages
    ]


def _message_blocks(message: ParsedMessage) -> list[ParsedContentBlock]:
    if message.content_blocks:
        return list(message.content_blocks)
    if message.text:
        return [ParsedContentBlock(type=ContentBlockType.TEXT, text=message.text)]
    return []


def _active_leaf_message_id(
    session_id: str,
    messages: list[ParsedMessage],
    explicit_native_id: str | None,
    *,
    position_offset: int = 0,
) -> str | None:
    if explicit_native_id:
        for fallback_position, message in enumerate(messages):
            if message.provider_message_id == explicit_native_id:
                return _message_id(session_id, message, fallback_position, position_offset=position_offset)
    for fallback_position, message in enumerate(messages):
        if message.is_active_leaf:
            return _message_id(session_id, message, fallback_position, position_offset=position_offset)
    return (
        _message_id(session_id, messages[-1], len(messages) - 1, position_offset=position_offset) if messages else None
    )


def _message_id(
    session_id: str,
    message: ParsedMessage,
    fallback_position: int,
    *,
    position_offset: int = 0,
) -> str:
    position = position_offset + (message.position if message.position is not None else 0)
    variant_index = message.variant_index if message.variant_index is not None else 0
    return archive_message_id(
        session_id,
        message.provider_message_id,
        position=position if message.position is not None else position_offset + fallback_position,
        variant_index=variant_index,
    )


def _block_type(block: ParsedContentBlock) -> BlockType:
    value = _enum_value(block.type)
    if value == "thinking":
        return BlockType.THINKING
    if value == "tool_use":
        return BlockType.TOOL_USE
    if value == "tool_result":
        return BlockType.TOOL_RESULT
    if value == "image":
        return BlockType.IMAGE
    if value == "code":
        return BlockType.CODE
    if value == "document":
        return BlockType.DOCUMENT
    return BlockType.TEXT


def _block_language(block: ParsedContentBlock) -> str | None:
    metadata = block.metadata or {}
    value = metadata.get("language")
    return str(value) if value is not None else None


def _semantic_type(block: ParsedContentBlock) -> str | None:
    if _block_type(block) is not BlockType.TOOL_USE or not block.tool_name:
        return None
    tool_input = cast("Mapping[str, JSONValue]", block.tool_input or {})
    category = classify_tool(block.tool_name, tool_input)
    return None if category is ToolCategory.OTHER else category.value


def _has_block(message: ParsedMessage, block_type: ContentBlockType) -> int:
    return int(any(_enum_value(block.type) == block_type.value for block in message.content_blocks))


def _has_paste(message: ParsedMessage) -> int:
    return int(bool(message.paste_spans))


def _paste_boundary(message: ParsedMessage) -> str | None:
    """Message-level paste boundary state, taken from the first detected span."""
    if not message.paste_spans:
        return None
    return PasteBoundary(message.paste_spans[0].boundary_state).value


def _word_count(text: str | None) -> int:
    return len(text.split()) if text else 0


def _timestamp_ms(value: str | None) -> int | None:
    parsed = parse_timestamp(value) if value else None
    return int(parsed.timestamp() * 1000) if parsed is not None else None


def _event_summary(event: ParsedSessionEvent) -> str | None:
    summary = event.payload.get("summary") or event.payload.get("text")
    return str(summary) if summary is not None else None


def _payload_string(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return str(value)
    return None


def _repo_name(repository_url: str, root_path: str) -> str | None:
    candidate = repository_url.rstrip("/").rsplit("/", maxsplit=1)[-1] if repository_url else Path(root_path).name
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    return candidate or None


def _repo_id(origin_url: str, root_path: str) -> str:
    return f"{origin_url}\x1f{root_path}"


def _attachment_id(_session_id: str, attachment: ParsedAttachment) -> str:
    return _hash_bytes(
        "attachment",
        attachment.provider_attachment_id,
        attachment.provider_file_id or "",
        attachment.provider_drive_id or "",
        attachment.path or "",
        attachment.name or "",
        attachment.mime_type or "",
        str(attachment.size_bytes or 0),
    ).hex()


def _attachment_position(attachment: ParsedAttachment) -> int:
    digest = hashlib.sha256()
    digest.update(attachment.provider_attachment_id.encode("utf-8", errors="surrogatepass"))
    return int.from_bytes(digest.digest()[:4], "big")


def _attachment_blob_hash(attachment_id: str, attachment: ParsedAttachment) -> bytes:
    del attachment
    return bytes.fromhex(attachment_id)


def _attachment_source_url(attachment: ParsedAttachment) -> str | None:
    return attachment.source_url


def _attachment_caption(attachment: ParsedAttachment) -> str | None:
    return attachment.caption


def _write_attachment_native_ids(conn: sqlite3.Connection, ref_id: str, attachment: ParsedAttachment) -> None:
    native_values = (
        ("attachment", attachment.provider_attachment_id),
        ("file", attachment.provider_file_id),
        ("drive", attachment.provider_drive_id),
        ("url", _attachment_source_url(attachment)),
    )
    for id_kind, native_id in native_values:
        if native_id:
            conn.execute(
                """
                INSERT OR IGNORE INTO attachment_native_ids (ref_id, id_kind, native_id)
                VALUES (?, ?, ?)
                """,
                (ref_id, id_kind, native_id),
            )


def _hash_bytes(*parts: str) -> bytes:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return digest.digest()


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _iso_from_ms(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    parsed = parse_timestamp(int(value) / 1000)
    return parsed.isoformat() if parsed is not None else None


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


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)


__all__ = [
    "ArchiveAgentPolicy",
    "ArchiveBlockRow",
    "ArchiveInsightMaterialization",
    "ArchiveMessageRow",
    "ArchiveSessionPhase",
    "ArchiveSessionTag",
    "ArchiveSessionEnvelope",
    "ArchiveSessionWorkEvent",
    "read_insight_materialization",
    "read_session_agent_policies",
    "read_session_phases",
    "read_session_tags",
    "read_session_work_events",
    "rebuild_archive_messages_fts",
    "upsert_session_profile_costs",
    "apply_insight_materialization",
    "upsert_insight_materialization",
    "upsert_session_phase",
    "upsert_session_tag",
    "upsert_session_work_event",
    "read_archive_session_envelope",
    "search_archive_blocks",
    "write_parsed_session_to_archive",
]
