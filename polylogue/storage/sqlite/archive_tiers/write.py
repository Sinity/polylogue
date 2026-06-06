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
    ParsedProviderEvent,
    ParsedSession,
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


@dataclass(frozen=True, slots=True)
class ArchiveAttachmentRow:
    attachment_id: str
    message_id: str | None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    provider_meta: str | None = None


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
    parent_message_id: str | None = None
    attachments: tuple[ArchiveAttachmentRow, ...] = ()


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
    origin_meta: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
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

    with conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, branch_type, active_leaf_message_id,
                title, origin_meta, git_branch, git_repository_url, commit_hash,
                message_count, word_count, tool_use_count, thinking_count,
                paste_count, user_message_count, assistant_message_count, system_message_count,
                tool_message_count, user_word_count, assistant_word_count,
                content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(origin, native_id) DO UPDATE SET
                raw_id = excluded.raw_id,
                branch_type = excluded.branch_type,
                active_leaf_message_id = excluded.active_leaf_message_id,
                title = COALESCE(excluded.title, sessions.title),
                origin_meta = excluded.origin_meta,
                git_branch = COALESCE(excluded.git_branch, sessions.git_branch),
                git_repository_url = COALESCE(excluded.git_repository_url, sessions.git_repository_url),
                commit_hash = COALESCE(excluded.commit_hash, sessions.commit_hash),
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
                _json_dumps(session.provider_meta or {}),
                session.git_branch,
                session.git_repository_url,
                session.git_commit_hash,
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
                _hash_bytes("session", origin.value, native_id),
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
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        _write_messages(conn, session_id, messages, position_offset=position_offset)
        _write_blocks(conn, session_id, messages, position_offset=position_offset)
        _write_attachments(conn, session_id, messages, session.attachments, position_offset=position_offset)
        _write_paste_spans(conn, session_id, messages, position_offset=position_offset)
        _write_parent_links(conn, session_id, messages, position_offset=position_offset)
        _write_session_link(conn, session_id, session)
        _write_session_events(conn, session_id, messages, session.provider_events, position_offset=position_offset)
        _write_working_dirs(conn, session_id, session.working_directories)
        _write_repo_edges(conn, session_id, session)
        _refresh_session_counts(conn, session_id)
        _resolve_session_graph(conn, session_id, native_id, origin.value)
    return session_id


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
               parent_session_id, root_session_id, branch_type, origin_meta,
               created_at_ms, updated_at_ms
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if session is None:
        raise KeyError(session_id)

    attachment_rows = conn.execute(
        """
        SELECT r.message_id AS message_id, a.attachment_id AS attachment_id,
               a.mime_type AS mime_type, a.size_bytes AS size_bytes, a.path AS path,
               a.provider_meta AS provider_meta
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
                mime_type=attachment["mime_type"],
                size_bytes=attachment["size_bytes"],
                path=attachment["path"],
                provider_meta=attachment["provider_meta"],
            )
        )

    message_rows = conn.execute(
        """
        SELECT message_id, native_id, role, position, variant_index, is_active_path, is_active_leaf,
               message_type, word_count, has_tool_use, has_thinking, has_paste, occurred_at_ms,
               parent_message_id
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
                   tool_input, metadata
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
                        metadata=block["metadata"],
                    )
                    for block in block_rows
                ),
                message_type=message["message_type"],
                word_count=int(message["word_count"] or 0),
                has_tool_use=bool(message["has_tool_use"]),
                has_thinking=bool(message["has_thinking"]),
                has_paste=bool(message["has_paste"]),
                occurred_at=_iso_from_ms(message["occurred_at_ms"]),
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
        origin_meta=session["origin_meta"],
        created_at=_iso_from_ms(session["created_at_ms"]),
        updated_at=_iso_from_ms(session["updated_at_ms"]),
        orphan_attachments=tuple(attachments_by_message.get(None, ())),
    )


def search_archive_blocks(conn: sqlite3.Connection, query: str) -> list[str]:
    """Return block ids matched by the archive external-content FTS table."""
    match_query = normalize_fts5_query(query)
    if match_query is None:
        return []
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT block_id
        FROM blocks_fts
        WHERE blocks_fts MATCH ?
        ORDER BY rank
        """,
        (match_query,),
    ).fetchall()
    return [row["block_id"] for row in rows]


def rebuild_archive_blocks_fts(conn: sqlite3.Connection) -> int:
    """Rebuild the archive block FTS index from the canonical ``blocks`` table."""
    conn.execute("INSERT INTO blocks_fts(blocks_fts) VALUES('rebuild')")
    row = conn.execute("SELECT COUNT(*) FROM blocks_fts").fetchone()
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
                model_name, model_effort, has_tool_use, has_thinking, has_paste,
                variant_index, is_active_path, is_active_leaf, word_count,
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                duration_ms, content_hash, occurred_at_ms
            ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    tool_id, tool_input, semantic_type, media_type, metadata
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
                    _json_dumps(block.metadata or {}),
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
        conn.execute(
            """
            INSERT INTO attachments (
                attachment_id, mime_type, size_bytes, path, provider_meta,
                provider_attachment_id, provider_file_id, provider_drive_id, upload_origin, ref_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(attachment_id) DO UPDATE SET
                mime_type = COALESCE(excluded.mime_type, attachments.mime_type),
                size_bytes = COALESCE(excluded.size_bytes, attachments.size_bytes),
                path = COALESCE(excluded.path, attachments.path),
                provider_meta = excluded.provider_meta,
                provider_attachment_id = COALESCE(excluded.provider_attachment_id, attachments.provider_attachment_id),
                provider_file_id = COALESCE(excluded.provider_file_id, attachments.provider_file_id),
                provider_drive_id = COALESCE(excluded.provider_drive_id, attachments.provider_drive_id),
                upload_origin = COALESCE(excluded.upload_origin, attachments.upload_origin)
            """,
            (
                attachment_id,
                attachment.mime_type,
                attachment.size_bytes,
                attachment.path,
                _json_dumps(attachment.provider_meta or {}),
                attachment.provider_attachment_id,
                attachment.provider_file_id,
                attachment.provider_drive_id,
                attachment.upload_origin,
            ),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO attachment_refs (
                ref_id, attachment_id, session_id, message_id, provider_meta,
                provider_attachment_id, provider_file_id, provider_drive_id, upload_origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _attachment_ref_id(session_id, attachment),
                attachment_id,
                session_id,
                message_id,
                _json_dumps(attachment.provider_meta or {}),
                attachment.provider_attachment_id,
                attachment.provider_file_id,
                attachment.provider_drive_id,
                attachment.upload_origin,
            ),
        )
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
        boundary = PasteBoundary.WHOLE_MESSAGE_FALLBACK if text else PasteBoundary.HASH_ONLY
        conn.execute(
            """
            INSERT OR REPLACE INTO paste_spans (
                message_id, session_id, start_offset, end_offset, content_hash, boundary
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                0,
                len(text),
                _hash_bytes("paste", message_id, text),
                boundary.value,
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
            src_session_id, dst_session_native_id, link_type, status, method, confidence, evidence_json, observed_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            session.parent_session_provider_id,
            link_type,
            "unresolved",
            "parser-parent",
            1.0,
            _json_dumps({"parent_session_provider_id": session.parent_session_provider_id}),
            _timestamp_ms(session.updated_at) or _timestamp_ms(session.created_at),
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
        JOIN sessions src ON src.session_id = links.src_session_id
        WHERE links.dst_session_native_id = ?
          AND links.dst_session_id IS NULL
          AND src.origin = ?
        """,
        (native_id, origin),
    ).fetchall()
    for row in inbound_rows:
        conn.execute(
            """
            UPDATE session_links
            SET dst_session_id = ?, status = 'resolved'
            WHERE src_session_id = ?
              AND dst_session_native_id = ?
              AND dst_session_id IS NULL
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
        SET dst_session_id = (
                SELECT dst.session_id
                FROM sessions dst
                WHERE dst.native_id = session_links.dst_session_native_id
                  AND dst.origin = ?
                LIMIT 1
            ),
            status = 'resolved'
        WHERE src_session_id = ?
          AND dst_session_id IS NULL
          AND EXISTS (
                SELECT 1
                FROM sessions dst
                WHERE dst.native_id = session_links.dst_session_native_id
                  AND dst.origin = ?
          )
        """,
        (origin, session_id, origin),
    )


def _refresh_session_projection(conn: sqlite3.Connection, session_id: str, *, seen: set[str]) -> None:
    if session_id in seen:
        return
    seen.add(session_id)
    parent_link = conn.execute(
        """
        SELECT dst_session_id, link_type
        FROM session_links
        WHERE src_session_id = ? AND dst_session_id IS NOT NULL
        ORDER BY observed_at_ms IS NULL, observed_at_ms, dst_session_native_id, link_type
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    if parent_link is None:
        conn.execute(
            """
            UPDATE sessions
            SET parent_session_id = NULL,
                root_session_id = session_id,
                branch_type = NULL
            WHERE session_id = ?
            """,
            (session_id,),
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
        INSERT INTO threads (thread_id, root_session_id, origin, created_at_ms, updated_at_ms, session_count)
        VALUES (?, ?, ?, ?, ?, 0)
        ON CONFLICT(thread_id) DO UPDATE SET
            root_session_id = excluded.root_session_id,
            origin = excluded.origin,
            created_at_ms = excluded.created_at_ms,
            updated_at_ms = excluded.updated_at_ms
        """,
        (root_session_id, root_session_id, root[1], root[2], root[3]),
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
            updated_at_ms = (
                SELECT MAX(updated_at_ms)
                FROM sessions
                WHERE COALESCE(root_session_id, session_id) = ?
            )
        WHERE thread_id = ?
        """,
        (len(session_rows), root_session_id, root_session_id),
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
    events: Iterable[ParsedProviderEvent],
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
        if event.event_type not in {"compaction", "ghost_commit", "agent_policy"}:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO session_events (
                session_id, source_message_id, position, event_type, summary, payload, occurred_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                by_native_id.get(event.source_message_provider_id or ""),
                position,
                event.event_type,
                _event_summary(event),
                _json_dumps(event.payload),
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


def _write_repo_edges(conn: sqlite3.Connection, session_id: str, session: ParsedSession) -> None:
    observed_at_ms = _timestamp_ms(session.updated_at) or _timestamp_ms(session.created_at)
    root_paths = tuple(dict.fromkeys(path.strip() for path in session.working_directories if path.strip()))
    if not root_paths and not session.git_repository_url:
        return

    repository_url = (session.git_repository_url or "").strip()
    for root_path in root_paths or ("",):
        repo_name = _repo_name(repository_url, root_path)
        conn.execute(
            """
            INSERT INTO repos (repository_url, root_path, repo_name, first_seen_at_ms, last_seen_at_ms)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(repository_url, root_path) DO UPDATE SET
                repo_name = COALESCE(excluded.repo_name, repos.repo_name),
                first_seen_at_ms = MIN(repos.first_seen_at_ms, excluded.first_seen_at_ms),
                last_seen_at_ms = MAX(repos.last_seen_at_ms, excluded.last_seen_at_ms)
            """,
            (repository_url, root_path, repo_name, observed_at_ms or 0, observed_at_ms or 0),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO session_repos (
                session_id, repository_url, root_path, branch_name, observed_at_ms
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, repository_url, root_path, session.git_branch, observed_at_ms),
        )
        if session.git_commit_hash:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_commits (
                    session_id, repository_url, root_path, commit_hash, detection_method,
                    confidence, evidence_json, observed_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    repository_url,
                    root_path,
                    session.git_commit_hash,
                    "parser-git-meta",
                    1.0,
                    _json_dumps(
                        {
                            "git_repository_url": repository_url or None,
                            "root_path": root_path or None,
                            "git_branch": session.git_branch,
                        }
                    ),
                    observed_at_ms,
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


def _semantic_type(block: ParsedContentBlock) -> str | None:
    if _block_type(block) is not BlockType.TOOL_USE or not block.tool_name:
        return None
    tool_input = cast("Mapping[str, JSONValue]", block.tool_input or {})
    category = classify_tool(block.tool_name, tool_input)
    return None if category is ToolCategory.OTHER else category.value


def _has_block(message: ParsedMessage, block_type: ContentBlockType) -> int:
    return int(any(_enum_value(block.type) == block_type.value for block in message.content_blocks))


def _has_paste(message: ParsedMessage) -> int:
    meta = message.provider_meta or {}
    return int(bool(meta.get("claude_code_history_paste") or meta.get("has_paste")))


def _word_count(text: str | None) -> int:
    return len(text.split()) if text else 0


def _timestamp_ms(value: str | None) -> int | None:
    parsed = parse_timestamp(value) if value else None
    return int(parsed.timestamp() * 1000) if parsed is not None else None


def _event_summary(event: ParsedProviderEvent) -> str | None:
    summary = event.payload.get("summary") or event.payload.get("text")
    return str(summary) if summary is not None else None


def _repo_name(repository_url: str, root_path: str) -> str | None:
    candidate = repository_url.rstrip("/").rsplit("/", maxsplit=1)[-1] if repository_url else Path(root_path).name
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    return candidate or None


def _attachment_id(session_id: str, attachment: ParsedAttachment) -> str:
    return f"{session_id}:attachment:{attachment.provider_attachment_id}"


def _attachment_ref_id(session_id: str, attachment: ParsedAttachment) -> str:
    message_part = attachment.message_provider_id or "session"
    return f"{session_id}:attachment-ref:{message_part}:{attachment.provider_attachment_id}"


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
    "ArchiveBlockRow",
    "ArchiveInsightMaterialization",
    "ArchiveMessageRow",
    "ArchiveSessionPhase",
    "ArchiveSessionTag",
    "ArchiveSessionEnvelope",
    "ArchiveSessionWorkEvent",
    "read_insight_materialization",
    "read_session_phases",
    "read_session_tags",
    "read_session_work_events",
    "rebuild_archive_blocks_fts",
    "upsert_session_profile_costs",
    "upsert_insight_materialization",
    "upsert_session_phase",
    "upsert_session_tag",
    "upsert_session_work_event",
    "read_archive_session_envelope",
    "search_archive_blocks",
    "write_parsed_session_to_archive",
]
