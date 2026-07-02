"""Minimal archive index parsed-session writer/read helpers."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.topology.edge import TopologyEdgeType, branch_type_to_edge_type
from polylogue.archive.viewport.viewports import ToolCategory, classify_tool
from polylogue.core.enums import BlockType, PasteBoundary, SessionKind
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
from polylogue.storage.fts.fts_lifecycle import (
    message_fts_triggers_present_sync,
    restore_message_fts_triggers_sync,
    suspend_message_fts_triggers_sync,
)
from polylogue.storage.fts.sql import delete_session_rows_sql, insert_session_rows_sql
from polylogue.storage.search.query_support import normalize_fts5_query

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


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
    # Keystone structured tool-result outcome (schema v16). NULL = unknown.
    tool_result_is_error: int | None = None
    tool_result_exit_code: int | None = None


@dataclass(frozen=True, slots=True)
class ArchiveWebConstructRow:
    construct_id: str
    session_id: str
    message_id: str
    block_id: str
    position: int
    provider: str
    construct_type: str
    provider_key: str | None = None
    title: str | None = None
    url: str | None = None
    text: str | None = None
    source_id: str | None = None
    group_id: str | None = None
    group_title: str | None = None
    query: str | None = None
    asset_pointer: str | None = None
    mime_type: str | None = None
    status: str | None = None
    task_id: str | None = None
    task_type: str | None = None
    rank: int | None = None
    start_index: int | None = None
    end_index: int | None = None


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
    material_origin: str = "unknown"
    word_count: int = 0
    has_tool_use: bool = False
    has_thinking: bool = False
    has_paste: bool = False
    paste_boundary_state: str | None = None
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
    session_kind: str = SessionKind.STANDARD.value
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
    provider_project_ref: str | None = None
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
    input_high_water_mark_source: str | None = None


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


@dataclass(frozen=True, slots=True)
class SessionEventWriteResult:
    wrote_provider_usage_events: bool = False


def write_parsed_session_to_archive(
    conn: sqlite3.Connection,
    session: ParsedSession,
    *,
    content_hash: str | None = None,
    raw_id: str | None = None,
    merge_append: bool = False,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "append",
    signature_cache: dict[str, list[tuple[str, str]]] | None = None,
    manage_transaction: bool = True,
) -> str:
    """Write one parsed session into an initialized archive index DB.

    By default the whole write runs in its own transaction (``with conn:``)
    committed on success. A bulk caller that wants many sessions in one
    transaction — to amortize the per-commit fsync and WAL page churn that
    dominate re-ingest I/O — passes ``manage_transaction=False`` and owns the
    surrounding commit and any rollback-on-error itself.
    """
    t0 = time.perf_counter()

    def add_timing(name: str, started_at: float) -> None:
        _add_stage_timing(
            stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            name=name,
            started_at=started_at,
        )

    conn.execute("PRAGMA foreign_keys = ON")
    origin = origin_from_provider(session.source_name)
    native_id = session.provider_session_id
    session_id = archive_session_id(origin.value, native_id)
    incoming_freshness_ms = _timestamp_ms(session.updated_at) or _timestamp_ms(session.created_at)
    if not merge_append and incoming_freshness_ms is not None:
        row = conn.execute(
            "SELECT updated_at_ms FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        existing_updated_at_ms = int(row[0]) if row is not None and row[0] is not None else None
        if existing_updated_at_ms is not None and incoming_freshness_ms < existing_updated_at_ms:
            add_timing("index.skip_stale_replace", t0)
            return session_id
    # This session's own rows are about to be rewritten; drop any stale memoized
    # own-signatures so the batch cache never serves pre-write rows for it.
    if signature_cache is not None:
        signature_cache.pop(session_id, None)
    messages = _normalized_messages(session.messages)
    # Lineage normalization (#2467): when this is a prefix-sharing child whose
    # parent is already in the archive, drop the inherited prefix and keep only
    # the divergent tail. All downstream writes (messages, blocks, counts,
    # attachments, events) then operate on the tail, so each real message is
    # stored exactly once. Only applies to full-replace writes; merge-append is
    # an incremental extend of the same session.
    branch_point_message_id: str | None = None
    lineage_inheritance: str | None = None
    if not merge_append:
        parent_session_id = _existing_parent_session_id(conn, session, origin.value)
        if parent_session_id is not None and messages:
            branch_point_message_id, lineage_inheritance, messages = _extract_prefix_tail(
                conn, parent_session_id, messages, cache=signature_cache
            )
    duplicate_message_native_ids = _duplicate_message_native_ids(messages)
    active_leaf_message_id = _active_leaf_message_id(
        session_id,
        messages,
        session.active_leaf_message_provider_id,
        duplicate_native_ids=duplicate_message_native_ids,
    )
    session_content_hash = (
        bytes.fromhex(content_hash) if content_hash is not None else _hash_bytes("session", origin.value, native_id)
    )
    add_timing("index.prepare", t0)
    session_counts = _session_count_values(messages)

    # When the caller owns the transaction (bulk batching) we must not commit
    # per session; nullcontext leaves BEGIN/COMMIT to the caller.
    transaction = conn if manage_transaction else nullcontext()
    with transaction:
        t0 = time.perf_counter()
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, branch_type, active_leaf_message_id,
                title, session_kind, title_source, git_branch, git_repository_url, commit_hash,
                instructions_text, reported_duration_ms, provider_project_ref,
                message_count, word_count, tool_use_count, thinking_count,
                paste_count, user_message_count, authored_user_message_count,
                assistant_message_count, system_message_count,
                tool_message_count, user_word_count, authored_user_word_count, assistant_word_count,
                content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(origin, native_id) DO UPDATE SET
                raw_id = excluded.raw_id,
                branch_type = excluded.branch_type,
                active_leaf_message_id = excluded.active_leaf_message_id,
                title = COALESCE(excluded.title, sessions.title),
                session_kind = excluded.session_kind,
                title_source = COALESCE(excluded.title_source, sessions.title_source),
                git_branch = excluded.git_branch,
                git_repository_url = excluded.git_repository_url,
                commit_hash = excluded.commit_hash,
                provider_project_ref = excluded.provider_project_ref,
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
                _sqlite_text(session.title),
                _enum_value(session.session_kind) or SessionKind.STANDARD.value,
                _enum_value(session.title_source),
                _sqlite_text(session.git_branch),
                _sqlite_text(session.git_repository_url),
                _sqlite_text(session.git_commit_hash),
                _sqlite_text(session.instructions_text),
                session.reported_duration_ms,
                _sqlite_text(session.provider_project_ref),
                session_counts["message_count"],
                session_counts["word_count"],
                session_counts["tool_use_count"],
                session_counts["thinking_count"],
                session_counts["paste_count"],
                session_counts["user_message_count"],
                session_counts["authored_user_message_count"],
                session_counts["assistant_message_count"],
                session_counts["system_message_count"],
                session_counts["tool_message_count"],
                session_counts["user_word_count"],
                session_counts["authored_user_word_count"],
                session_counts["assistant_word_count"],
                session_content_hash,
                _timestamp_ms(session.created_at),
                _timestamp_ms(session.updated_at),
            ),
        )
        add_timing("index.session_upsert", t0)
        position_offset = 0
        stale_attachment_ids: set[str] = set()
        t0 = time.perf_counter()
        if merge_append:
            row = conn.execute(
                "SELECT COALESCE(MAX(position) + 1, 0) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            position_offset = int(row[0] or 0) if row is not None else 0
            conn.execute(
                """
                UPDATE messages
                SET is_active_leaf = 0
                WHERE session_id = ?
                  AND is_active_path = 1
                  AND is_active_leaf = 1
                """,
                (session_id,),
            )
            active_leaf_message_id = _active_leaf_message_id(
                session_id,
                messages,
                session.active_leaf_message_provider_id,
                position_offset=position_offset,
                duplicate_native_ids=duplicate_message_native_ids,
            )
            conn.execute(
                "UPDATE sessions SET active_leaf_message_id = ? WHERE session_id = ?",
                (active_leaf_message_id, session_id),
            )
            add_timing("index.merge_prepare", t0)
        else:
            stale_attachment_ids = _session_attachment_ids(conn, session_id)
            _replace_full_session_messages_and_blocks(
                conn,
                session,
                messages,
                duplicate_native_ids=duplicate_message_native_ids,
                stage_timings_s=stage_timings_s,
                stage_timing_prefix=stage_timing_prefix,
            )
            add_timing("index.full_replace", t0)
        if merge_append:
            t0 = time.perf_counter()
            _write_messages(
                conn,
                session_id,
                messages,
                position_offset=position_offset,
                duplicate_native_ids=duplicate_message_native_ids,
            )
            add_timing("index.messages", t0)
            t0 = time.perf_counter()
            _write_blocks(
                conn,
                session_id,
                messages,
                position_offset=position_offset,
                duplicate_native_ids=duplicate_message_native_ids,
            )
            add_timing("index.blocks", t0)
            t0 = time.perf_counter()
            _write_web_constructs(
                conn,
                session,
                messages,
                position_offset=position_offset,
                duplicate_native_ids=duplicate_message_native_ids,
                replace_session=False,
            )
            add_timing("index.web_constructs", t0)
        t0 = time.perf_counter()
        _write_attachments(
            conn,
            session_id,
            messages,
            session.attachments,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_message_native_ids,
            refresh_attachment_ids=stale_attachment_ids,
        )
        add_timing("index.attachments", t0)
        t0 = time.perf_counter()
        _write_paste_spans(
            conn,
            session_id,
            messages,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_message_native_ids,
        )
        add_timing("index.paste_spans", t0)
        t0 = time.perf_counter()
        _write_parent_links(
            conn,
            session_id,
            messages,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_message_native_ids,
        )
        add_timing("index.parent_links", t0)
        t0 = time.perf_counter()
        _write_session_link(
            conn,
            session_id,
            session,
            branch_point_message_id=branch_point_message_id,
            inheritance=lineage_inheritance,
        )
        add_timing("index.session_link", t0)
        t0 = time.perf_counter()
        event_position_offset = _next_session_event_position(conn, session_id)
        session_event_result = _write_session_events(
            conn,
            session_id,
            messages,
            session.session_events,
            position_offset=position_offset,
            event_position_offset=event_position_offset,
            duplicate_native_ids=duplicate_message_native_ids,
        )
        add_timing("index.session_events", t0)
        t0 = time.perf_counter()
        _write_working_dirs(conn, session_id, session.working_directories)
        add_timing("index.working_dirs", t0)
        t0 = time.perf_counter()
        _write_repo_edges(conn, session_id, session)
        add_timing("index.repo_edges", t0)
        t0 = time.perf_counter()
        _write_reported_costs(
            conn,
            session_id,
            session,
            replace_existing_model_rows=not merge_append,
            aggregate_message_tokens=not merge_append or _messages_have_token_counts(messages),
        )
        add_timing("index.reported_costs", t0)
        if merge_append and session_event_result.wrote_provider_usage_events:
            t0 = time.perf_counter()
            _aggregate_appended_provider_usage_into_model_usage(
                conn,
                session_id,
                start_position=event_position_offset,
            )
            add_timing("index.provider_usage_rollup", t0)
        elif not merge_append:
            t0 = time.perf_counter()
            _aggregate_provider_usage_into_model_usage(conn, session_id)
            add_timing("index.provider_usage_rollup", t0)
        t0 = time.perf_counter()
        if merge_append:
            _increment_session_counts_for_append(conn, session_id, session_counts)
        else:
            _refresh_session_counts(conn, session_id)
        add_timing("index.session_counts", t0)
        t0 = time.perf_counter()
        _resolve_session_graph(conn, session_id, native_id, origin.value, cache=signature_cache)
        add_timing("index.graph_resolve", t0)
        if session.ingest_flags:
            t0 = time.perf_counter()
            _write_ingest_flag_tags(conn, session_id, session.ingest_flags)
            add_timing("index.ingest_flags", t0)
    return session_id


def _add_stage_timing(
    stage_timings_s: dict[str, float] | None,
    *,
    stage_timing_prefix: str,
    name: str,
    started_at: float,
) -> None:
    if stage_timings_s is None:
        return
    key = f"{stage_timing_prefix}.{name}"
    stage_timings_s[key] = stage_timings_s.get(key, 0.0) + (time.perf_counter() - started_at)


def _write_ingest_flag_tags(conn: sqlite3.Connection, session_id: str, flags: list[str]) -> None:
    """Write parser-level ingest flags as auto-tags in the same transaction.

    Each flag is lowercased and written as ``(session_id, flag, 'auto')`` with
    ``method='parser'``.  Duplicate flags on re-ingest are silently skipped
    (``ON CONFLICT DO NOTHING``) so repeated ingest of the same session is
    idempotent.  Called from inside the ``with conn:`` block of
    ``write_parsed_session_to_archive`` so the tag rows are committed atomically
    with the session row they reference.
    """
    for raw_flag in flags:
        normalized = raw_flag.strip().lower()
        if not normalized:
            continue
        conn.execute(
            """
            INSERT INTO session_tags (session_id, tag, tag_source, method)
            VALUES (?, ?, 'auto', 'parser')
            ON CONFLICT(session_id, tag, tag_source) DO NOTHING
            """,
            (session_id, normalized),
        )


def upsert_parser_ingest_flag_tags(conn: sqlite3.Connection, session_id: str, flags: list[str]) -> None:
    """Upsert parser-owned ingest flag tags for an already-materialized session."""
    _write_ingest_flag_tags(conn, session_id, flags)


def _clear_session_projection_rows(conn: sqlite3.Connection, session_id: str) -> None:
    """Clear rows owned by parsed-session replacement before rewriting it."""
    conn.execute(
        """
        UPDATE messages
        SET parent_message_id = NULL
        WHERE parent_message_id IN (
            SELECT message_id FROM messages WHERE session_id = ?
        )
        """,
        (session_id,),
    )
    _purge_session_message_fts_when_delete_trigger_missing(conn, session_id)
    for table in (
        "blocks",
        "attachment_refs",
        "paste_spans",
        "session_events",
        "session_provider_usage_events",
        "session_agent_policies",
        "session_working_dirs",
        "session_repos",
        "session_commits",
        "session_reported_costs",
        "session_model_usage",
    ):
        conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM session_links WHERE src_session_id = ?", (session_id,))


def _purge_session_message_fts_when_delete_trigger_missing(conn: sqlite3.Connection, session_id: str) -> None:
    """Delete current session FTS rows before block deletion when triggers are suspended."""
    trigger_row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = 'messages_fts_ad'",
    ).fetchone()
    if trigger_row is not None:
        return
    table_rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table', 'virtual table')
          AND name IN ('messages_fts', 'messages_fts_docsize')
        """,
    ).fetchall()
    if {str(row[0]) for row in table_rows} != {"messages_fts", "messages_fts_docsize"}:
        return
    from polylogue.storage.fts.sql import delete_session_rows_sql

    conn.execute(delete_session_rows_sql(1), (session_id,))


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
    input_high_water_mark_source: str | None = None,
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
            source_updated_at_ms, source_sort_key_ms, input_high_water_mark_ms,
            input_high_water_mark_source, input_row_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(insight_type, session_id) DO UPDATE SET
            materializer_version = excluded.materializer_version,
            materialized_at_ms = excluded.materialized_at_ms,
            source_updated_at_ms = excluded.source_updated_at_ms,
            source_sort_key_ms = excluded.source_sort_key_ms,
            input_high_water_mark_ms = excluded.input_high_water_mark_ms,
            input_high_water_mark_source = excluded.input_high_water_mark_source,
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
            input_high_water_mark_source,
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
    input_high_water_mark_source: str | None = None,
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
            input_high_water_mark_source=input_high_water_mark_source,
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
            source_updated_at_ms, source_sort_key_ms, input_high_water_mark_ms,
            input_high_water_mark_source, input_row_count
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
        input_high_water_mark_source=row["input_high_water_mark_source"],
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


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
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


def read_archive_session_envelope(
    conn: sqlite3.Connection, session_id: str, *, _depth: int = 0
) -> ArchiveSessionEnvelope:
    """Read a compact archive envelope.

    For a prefix-sharing lineage child (#2467) the inherited prefix is not stored
    under this session; the returned ``messages`` compose the parent's transcript
    up to the branch point followed by this session's own messages, so reads see
    the full logical transcript while storage holds each message once.
    """
    conn.row_factory = sqlite3.Row
    session = conn.execute(
        """
        SELECT session_id, native_id, origin, title, session_kind, active_leaf_message_id,
               parent_session_id, root_session_id, branch_type,
               title_source, instructions_text,
               created_at_ms, updated_at_ms, git_branch, git_repository_url, provider_project_ref
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
               message_type, material_origin, word_count, has_tool_use, has_thinking, has_paste, occurred_at_ms,
               paste_boundary AS paste_boundary_state, duration_ms, parent_message_id
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
                   tool_input, language, tool_result_is_error, tool_result_exit_code
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
                        tool_result_is_error=block["tool_result_is_error"],
                        tool_result_exit_code=block["tool_result_exit_code"],
                    )
                    for block in block_rows
                ),
                message_type=message["message_type"],
                material_origin=message["material_origin"],
                word_count=int(message["word_count"] or 0),
                has_tool_use=bool(message["has_tool_use"]),
                has_thinking=bool(message["has_thinking"]),
                has_paste=bool(message["has_paste"]),
                paste_boundary_state=message["paste_boundary_state"],
                occurred_at=_iso_from_ms(message["occurred_at_ms"]),
                duration_ms=int(message["duration_ms"] or 0),
                parent_message_id=message["parent_message_id"],
                attachments=tuple(attachments_by_message.get(message["message_id"], ())),
            )
        )

    # Lineage composition (#2467): prepend the parent's composed transcript up to
    # and including the branch point. The parent envelope is itself composed via
    # this same recursion, so nested lineages resolve correctly.
    if _depth < _MAX_LINEAGE_DEPTH:
        edge = _prefix_sharing_edge_sync(conn, str(session["session_id"]))
        if edge is not None:
            parent_session_id, branch_point_message_id = edge
            parent_messages = read_archive_session_envelope(conn, parent_session_id, _depth=_depth + 1).messages
            prefix: list[ArchiveMessageRow] = []
            found = False
            for parent_message in parent_messages:
                prefix.append(parent_message)
                if parent_message.message_id == branch_point_message_id:
                    found = True
                    break
            # Dangling branch point (parent message hard-deleted): keep this
            # session's own tail rather than splice the entire parent (#2467 audit).
            if found:
                messages = prefix + messages

    return ArchiveSessionEnvelope(
        session_id=session["session_id"],
        native_id=session["native_id"],
        origin=session["origin"],
        title=session["title"],
        session_kind=session["session_kind"],
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
        provider_project_ref=session["provider_project_ref"],
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
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> None:
    def rows() -> Iterable[tuple[object, ...]]:
        for fallback_position, message in enumerate(messages):
            position = position_offset + (message.position if message.position is not None else fallback_position)
            variant_index = message.variant_index if message.variant_index is not None else 0
            yield (
                session_id,
                _sqlite_text(_effective_message_native_id(message, duplicate_native_ids)) or None,
                position,
                _enum_value(message.role),
                _enum_value(message.message_type),
                _enum_value(message.material_origin),
                _sqlite_text(message.model_name),
                _sqlite_text(message.model_effort),
                _sqlite_text(message.sender_name),
                _sqlite_text(message.recipient),
                _sqlite_text(message.delivery_status),
                None if message.end_turn is None else int(message.end_turn),
                _sqlite_text(message.user_context_text),
                _has_block(message, BlockType.TOOL_USE),
                _has_block(message, BlockType.THINKING),
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
            )

    conn.executemany(
        """
        INSERT OR REPLACE INTO messages (
            session_id, native_id, parent_message_id, position, role, message_type, material_origin,
            model_name, model_effort, sender_name, recipient, delivery_status, end_turn, user_context_text,
            has_tool_use, has_thinking, has_paste, paste_boundary,
            variant_index, is_active_path, is_active_leaf, word_count,
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            duration_ms, content_hash, occurred_at_ms
        ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows(),
    )


def _write_blocks(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> None:
    def rows() -> Iterable[tuple[object, ...]]:
        for fallback_position, message in enumerate(messages):
            message_id = _message_id(
                session_id,
                message,
                fallback_position,
                position_offset=position_offset,
                duplicate_native_ids=duplicate_native_ids,
            )
            blocks = _message_blocks(message)
            for position, block in enumerate(blocks):
                yield (
                    message_id,
                    session_id,
                    position,
                    _block_type(block).value,
                    _sqlite_text(block.text),
                    _sqlite_text(block.tool_name),
                    _sqlite_text(block.tool_id),
                    _json_dumps(block.tool_input) if block.tool_input is not None else None,
                    _sqlite_text(_semantic_type(block)),
                    _sqlite_text(block.media_type),
                    _sqlite_text(_block_language(block)),
                    _sqlite_bool(getattr(block, "is_error", None)),
                    getattr(block, "exit_code", None),
                )

    conn.executemany(
        """
        INSERT OR REPLACE INTO blocks (
            message_id, session_id, position, block_type, text, tool_name,
            tool_id, tool_input, semantic_type, media_type, language,
            tool_result_is_error, tool_result_exit_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows(),
    )


def _write_web_constructs(
    conn: sqlite3.Connection,
    session: ParsedSession,
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
    replace_session: bool = True,
) -> None:
    origin = origin_from_provider(session.source_name)
    session_id = archive_session_id(origin.value, session.provider_session_id)
    provider = _enum_value(session.source_name)
    rows: list[tuple[object, ...]] = []
    block_ids: list[str] = []
    # Iterate the (possibly lineage-sliced) tail messages, not session.messages —
    # a web construct on an inherited-prefix message would FK-violate against rows
    # that were never written under this session (#2467 audit).
    for fallback_position, message in enumerate(messages):
        message_id = _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        blocks = _message_blocks(message)
        for block_position, block in enumerate(blocks):
            block_id = f"{message_id}:{block_position}"
            if not replace_session:
                block_ids.append(block_id)
            for construct_position, construct in enumerate(block.web_constructs):
                rows.append(
                    (
                        session_id,
                        message_id,
                        block_id,
                        construct_position,
                        provider,
                        _enum_value(construct.construct_type),
                        _sqlite_text(construct.provider_key),
                        _sqlite_text(construct.title),
                        _sqlite_text(construct.url),
                        _sqlite_text(construct.text),
                        _sqlite_text(construct.source_id),
                        _sqlite_text(construct.group_id),
                        _sqlite_text(construct.group_title),
                        _sqlite_text(construct.query),
                        _sqlite_text(construct.asset_pointer),
                        _sqlite_text(construct.mime_type),
                        _sqlite_text(construct.status),
                        _sqlite_text(construct.task_id),
                        _sqlite_text(construct.task_type),
                        construct.rank,
                        construct.start_index,
                        construct.end_index,
                    )
                )

    if replace_session:
        conn.execute("DELETE FROM web_content_constructs WHERE session_id = ?", (session_id,))
    else:
        conn.executemany(
            "DELETE FROM web_content_constructs WHERE block_id = ?", ((block_id,) for block_id in block_ids)
        )

    conn.executemany(
        """
        INSERT OR REPLACE INTO web_content_constructs (
            session_id, message_id, block_id, position, provider, construct_type,
            provider_key, title, url, text, source_id, group_id, group_title,
            query, asset_pointer, mime_type, status, task_id, task_type,
            rank, start_index, end_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _replace_full_session_messages_and_blocks(
    conn: sqlite3.Connection,
    session: ParsedSession,
    messages: list[ParsedMessage],
    *,
    duplicate_native_ids: frozenset[str],
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "append",
) -> None:
    def add_timing(name: str, started_at: float) -> None:
        _add_stage_timing(
            stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            name=f"index.full_replace.{name}",
            started_at=started_at,
        )

    origin = origin_from_provider(session.source_name)
    session_id = archive_session_id(origin.value, session.provider_session_id)
    t0 = time.perf_counter()
    use_scoped_fts_rebuild = message_fts_triggers_present_sync(conn)
    add_timing("fts_probe", t0)
    if use_scoped_fts_rebuild:
        t0 = time.perf_counter()
        conn.execute(delete_session_rows_sql(1), (session_id,))
        add_timing("fts_delete", t0)
        t0 = time.perf_counter()
        suspend_message_fts_triggers_sync(conn)
        add_timing("fts_suspend", t0)
    try:
        t0 = time.perf_counter()
        _clear_session_projection_rows(conn, session_id)
        add_timing("clear_projection_rows", t0)
        t0 = time.perf_counter()
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        add_timing("delete_messages", t0)
        t0 = time.perf_counter()
        _write_messages(
            conn,
            session_id,
            messages,
            duplicate_native_ids=duplicate_native_ids,
        )
        add_timing("messages", t0)
        t0 = time.perf_counter()
        _write_blocks(
            conn,
            session_id,
            messages,
            duplicate_native_ids=duplicate_native_ids,
        )
        add_timing("blocks", t0)
        t0 = time.perf_counter()
        _write_web_constructs(
            conn,
            session,
            messages,
            duplicate_native_ids=duplicate_native_ids,
        )
        add_timing("web_constructs", t0)
        if use_scoped_fts_rebuild:
            t0 = time.perf_counter()
            conn.execute(insert_session_rows_sql(1), (session_id,))
            add_timing("fts_insert", t0)
    finally:
        if use_scoped_fts_rebuild:
            t0 = time.perf_counter()
            restore_message_fts_triggers_sync(conn)
            add_timing("fts_restore", t0)


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
            authored_user_message_count = (
                SELECT COUNT(*) FROM messages
                WHERE session_id = sessions.session_id AND material_origin = 'human_authored'
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
            authored_user_word_count = COALESCE((
                SELECT SUM(word_count) FROM messages
                WHERE session_id = sessions.session_id AND material_origin = 'human_authored'
            ), 0),
            assistant_word_count = COALESCE((
                SELECT SUM(word_count) FROM messages WHERE session_id = sessions.session_id AND role = 'assistant'
            ), 0)
        WHERE session_id = ?
        """,
        (session_id,),
    )


def _session_count_values(messages: list[ParsedMessage]) -> dict[str, int]:
    counts = {
        "message_count": 0,
        "word_count": 0,
        "tool_use_count": 0,
        "thinking_count": 0,
        "paste_count": 0,
        "user_message_count": 0,
        "authored_user_message_count": 0,
        "assistant_message_count": 0,
        "system_message_count": 0,
        "tool_message_count": 0,
        "user_word_count": 0,
        "authored_user_word_count": 0,
        "assistant_word_count": 0,
    }
    for message in messages:
        role = _enum_value(message.role)
        material_origin = _enum_value(message.material_origin)
        word_count = _word_count(message.text)
        counts["message_count"] += 1
        counts["word_count"] += word_count
        counts["tool_use_count"] += _has_block(message, BlockType.TOOL_USE)
        counts["thinking_count"] += _has_block(message, BlockType.THINKING)
        counts["paste_count"] += _has_paste(message)
        if role == "user":
            counts["user_message_count"] += 1
            counts["user_word_count"] += word_count
        elif role == "assistant":
            counts["assistant_message_count"] += 1
            counts["assistant_word_count"] += word_count
        elif role == "system":
            counts["system_message_count"] += 1
        elif role == "tool":
            counts["tool_message_count"] += 1
        if material_origin == "human_authored":
            counts["authored_user_message_count"] += 1
            counts["authored_user_word_count"] += word_count
    return counts


def _messages_have_token_counts(messages: Sequence[ParsedMessage]) -> bool:
    return any(
        message.input_tokens or message.output_tokens or message.cache_read_tokens or message.cache_write_tokens
        for message in messages
    )


def _increment_session_counts_for_append(
    conn: sqlite3.Connection,
    session_id: str,
    counts: dict[str, int],
) -> None:
    conn.execute(
        """
        UPDATE sessions
        SET message_count = COALESCE(message_count, 0) + ?,
            word_count = COALESCE(word_count, 0) + ?,
            tool_use_count = COALESCE(tool_use_count, 0) + ?,
            thinking_count = COALESCE(thinking_count, 0) + ?,
            paste_count = COALESCE(paste_count, 0) + ?,
            user_message_count = COALESCE(user_message_count, 0) + ?,
            authored_user_message_count = COALESCE(authored_user_message_count, 0) + ?,
            assistant_message_count = COALESCE(assistant_message_count, 0) + ?,
            system_message_count = COALESCE(system_message_count, 0) + ?,
            tool_message_count = COALESCE(tool_message_count, 0) + ?,
            user_word_count = COALESCE(user_word_count, 0) + ?,
            authored_user_word_count = COALESCE(authored_user_word_count, 0) + ?,
            assistant_word_count = COALESCE(assistant_word_count, 0) + ?
        WHERE session_id = ?
        """,
        (
            counts["message_count"],
            counts["word_count"],
            counts["tool_use_count"],
            counts["thinking_count"],
            counts["paste_count"],
            counts["user_message_count"],
            counts["authored_user_message_count"],
            counts["assistant_message_count"],
            counts["system_message_count"],
            counts["tool_message_count"],
            counts["user_word_count"],
            counts["authored_user_word_count"],
            counts["assistant_word_count"],
            session_id,
        ),
    )


def _write_attachments(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    attachments: Iterable[ParsedAttachment],
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
    refresh_attachment_ids: set[str] | None = None,
) -> None:
    by_native_message_id = {
        message.provider_message_id: _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        for fallback_position, message in enumerate(messages)
        if message.provider_message_id and message.provider_message_id not in duplicate_native_ids
    }
    touched_attachment_ids: set[str] = set()
    for attachment in attachments:
        attachment_id = _attachment_id(session_id, attachment)
        message_id = (
            by_native_message_id.get(attachment.message_provider_id) if attachment.message_provider_id else None
        )
        if message_id is None:
            continue
        touched_attachment_ids.add(attachment_id)
        blob_hash, byte_count, acquisition_status = _acquire_attachment_blob(attachment)
        conn.execute(
            """
            INSERT INTO attachments (
                attachment_id, display_name, media_type, byte_count, blob_hash, acquisition_status, ref_count
            ) VALUES (?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(attachment_id) DO UPDATE SET
                display_name = COALESCE(excluded.display_name, attachments.display_name),
                media_type = COALESCE(excluded.media_type, attachments.media_type),
                byte_count = excluded.byte_count,
                blob_hash = COALESCE(excluded.blob_hash, attachments.blob_hash),
                acquisition_status =
                    CASE WHEN excluded.acquisition_status = 'acquired'
                         THEN 'acquired' ELSE attachments.acquisition_status END
            """,
            (
                attachment_id,
                _sqlite_text(attachment.name),
                _sqlite_text(attachment.mime_type),
                byte_count,
                blob_hash,
                acquisition_status,
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
                _sqlite_text(attachment.upload_origin),
                _sqlite_text(_attachment_source_url(attachment)),
                _sqlite_text(_attachment_caption(attachment)),
            ),
        )
        ref_id = f"{message_id}:attachment:{ref_position}"
        _write_attachment_native_ids(conn, ref_id, attachment)
    affected_attachment_ids = touched_attachment_ids | (refresh_attachment_ids or set())
    if not affected_attachment_ids:
        return
    placeholders = ",".join("?" for _ in affected_attachment_ids)
    conn.execute(
        f"""
        UPDATE attachments
        SET ref_count = (
            SELECT COUNT(*) FROM attachment_refs WHERE attachment_refs.attachment_id = attachments.attachment_id
        )
        WHERE attachment_id IN ({placeholders})
        """,
        tuple(sorted(affected_attachment_ids)),
    )


def _session_attachment_ids(conn: sqlite3.Connection, session_id: str) -> set[str]:
    rows = conn.execute(
        "SELECT DISTINCT attachment_id FROM attachment_refs WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    return {str(row[0]) for row in rows}


def _write_paste_spans(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> None:
    for fallback_position, message in enumerate(messages):
        if not _has_paste(message):
            continue
        message_id = _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
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
                    _sqlite_text(evidence.source_event_id),
                    _sqlite_text(evidence.source_marker),
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
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> None:
    by_native_id = {
        message.provider_message_id: _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        for fallback_position, message in enumerate(messages)
        if message.provider_message_id and message.provider_message_id not in duplicate_native_ids
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
                _message_id(
                    session_id,
                    message,
                    fallback_position,
                    position_offset=position_offset,
                    duplicate_native_ids=duplicate_native_ids,
                ),
            ),
        )


def _write_session_link(
    conn: sqlite3.Connection,
    session_id: str,
    session: ParsedSession,
    *,
    branch_point_message_id: str | None = None,
    inheritance: str | None = None,
) -> None:
    if not session.parent_session_provider_id:
        return
    link_type = branch_type_to_edge_type(session.branch_type, default=TopologyEdgeType.BRANCH).value
    conn.execute(
        """
        INSERT OR REPLACE INTO session_links (
            src_session_id, dst_origin, dst_native_id, link_type,
            branch_point_message_id, inheritance,
            status, method, confidence, evidence_json, observed_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
        """,
        (
            session_id,
            origin_from_provider(session.source_name).value,
            _sqlite_text(session.parent_session_provider_id),
            link_type,
            branch_point_message_id,
            inheritance,
            "parser-parent",
            1.0,
            _json_dumps({"parent_session_provider_id": session.parent_session_provider_id}),
            _timestamp_ms(session.updated_at) or _timestamp_ms(session.created_at) or 0,
        ),
    )


def _branch_type_from_link_type(link_type: object) -> str | None:
    try:
        return BranchType(str(link_type)).value
    except ValueError:
        return None


def _resolve_session_graph(
    conn: sqlite3.Connection,
    session_id: str,
    native_id: str,
    origin: str,
    *,
    cache: dict[str, list[tuple[str, str]]] | None = None,
) -> None:
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
        # Deferred tail extraction (#2467): a child ingested before its parent was
        # stored whole (the inherited prefix could not be aligned yet). Now that
        # the parent exists, normalize the child the same way the parent-known
        # write path does — drop the inherited prefix rows and record the edge.
        _reextract_prefix_tail_db(conn, str(row[0]), session_id, cache=cache)

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
            branch_type = _branch_type_from_link_type(unresolved_link[0])
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
        (parent_session_id, parent_root_id, _branch_type_from_link_type(parent_link[1]), session_id),
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
        WHERE root_session_id = ? OR session_id = ?
        ORDER BY sort_key_ms IS NULL, sort_key_ms, session_id
        """,
        (root_session_id, root_session_id),
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


def _next_session_event_position(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        """
        SELECT MAX(position) + 1
        FROM (
            SELECT position FROM session_events WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_agent_policies WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_provider_usage_events WHERE session_id = ?
        )
        """,
        (session_id, session_id, session_id),
    ).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _write_session_events(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    events: Iterable[ParsedSessionEvent],
    *,
    position_offset: int = 0,
    event_position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> SessionEventWriteResult:
    by_native_id = {
        message.provider_message_id: _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        for fallback_position, message in enumerate(messages)
        if message.provider_message_id and message.provider_message_id not in duplicate_native_ids
    }
    wrote_provider_usage_events = False
    position = event_position_offset
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
                    _sqlite_text(event.event_type),
                    _sqlite_text(_event_summary(event) or ""),
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
                    _sqlite_text(_payload_string(event.payload, "approval", "approval_policy")),
                    _sqlite_text(_payload_string(event.payload, "sandbox", "sandbox_policy")),
                    _sqlite_text(_payload_string(event.payload, "network", "network_policy")),
                    _timestamp_ms(event.timestamp),
                ),
            )
            position += 1
        elif event.event_type in {"token_count", "message_usage"}:
            _write_provider_usage_event(
                conn,
                session_id,
                by_native_id.get(event.source_message_provider_id or ""),
                position,
                event,
            )
            wrote_provider_usage_events = True
            position += 1
    return SessionEventWriteResult(wrote_provider_usage_events=wrote_provider_usage_events)


def _write_provider_usage_event(
    conn: sqlite3.Connection,
    session_id: str,
    source_message_id: str | None,
    position: int,
    event: ParsedSessionEvent,
) -> None:
    last_usage = _payload_mapping(event.payload, "last_token_usage")
    total_usage = _payload_mapping(event.payload, "total_token_usage")
    conn.execute(
        """
        INSERT OR REPLACE INTO session_provider_usage_events (
            session_id, source_message_id, position, provider_event_type, model_name,
            last_input_tokens, last_output_tokens, last_cached_input_tokens,
            last_cache_write_tokens, last_reasoning_output_tokens, last_total_tokens,
            total_input_tokens, total_output_tokens, total_cached_input_tokens,
            total_cache_write_tokens, total_reasoning_output_tokens, total_tokens, model_context_window,
            payload_json, occurred_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            source_message_id,
            position,
            _sqlite_text(event.event_type),
            _sqlite_text(_payload_string(event.payload, "model", "model_name")),
            _payload_int(last_usage, "input_tokens"),
            _payload_int(last_usage, "output_tokens"),
            _payload_int(last_usage, "cached_input_tokens"),
            _payload_int(last_usage, "cache_write_tokens"),
            _payload_int(last_usage, "reasoning_output_tokens"),
            _payload_int(last_usage, "total_tokens"),
            _payload_int(total_usage, "input_tokens"),
            _payload_int(total_usage, "output_tokens"),
            _payload_int(total_usage, "cached_input_tokens"),
            _payload_int(total_usage, "cache_write_tokens"),
            _payload_int(total_usage, "reasoning_output_tokens"),
            _payload_int(total_usage, "total_tokens"),
            _payload_optional_int(event.payload, "model_context_window"),
            _json_dumps(event.payload),
            _timestamp_ms(event.timestamp),
        ),
    )


def _provider_usage_disjoint_lanes(
    input_with_cached: int,
    output_with_reasoning: int,
    cache_read: int,
    cache_write: int,
) -> tuple[int, int, int, int]:
    """Map Codex ``token_count`` totals onto disjoint billing lanes.

    Codex (OpenAI) reports ``input_tokens`` *inclusive* of
    ``cached_input_tokens`` and ``output_tokens`` *inclusive* of
    ``reasoning_output_tokens``. Verified across the full real corpus
    (1.84M token_count events): ``cached <= input`` on 100% of rows, and
    ``total == input + output`` on 98.9% (reasoning is a subset of output,
    not an additional term).

    The cost model (`archive/semantic/pricing.py:_cost_components`) bills
    ``input`` and ``cache_read`` as *separate additive lanes* — the Anthropic
    convention where ``input`` means fresh/uncached input. So the cached
    portion must be subtracted out of ``input`` or it is billed twice: once at
    the full input rate and again at the discounted cache-read rate. On the
    real archive cached is ~96% of Codex input, so the double-count inflated
    Codex input cost by roughly 8x. Likewise ``reasoning`` is already inside
    ``output``; adding it again over-counts output.

    Returns ``(fresh_input, output, cache_read, cache_write)`` with fresh input
    clamped at zero (defensive; ``input >= cached`` holds on every observed row).
    """
    fresh_input = max(input_with_cached - cache_read, 0)
    return fresh_input, output_with_reasoning, cache_read, cache_write


def _aggregate_provider_usage_into_model_usage(conn: sqlite3.Connection, session_id: str) -> None:
    """Fold provider-reported token-count totals into model usage rows.

    Codex ``token_count`` rows carry a *session-global* cumulative running total
    in their ``total_*`` columns — the counter spans the whole session, not a
    single model. So the cumulative is taken as one session-wide latest value
    (the highest-position ``token_count`` row that carries any ``total_*``),
    attributed to the model named on that row, and written as a single rollup.
    Partitioning the cumulative by model and summing would double-count, because
    each model's "latest cumulative" already includes every prior model's
    tokens (#2472).

    Older/simple token-count rows only expose request-scoped ``last_token_usage``
    (Claude-style per-message per-model deltas); when no cumulative ``total_*``
    appears at all, those are summed per model. Unknown-model events only fall
    back to a session model when exactly one model row exists, keeping
    multi-model sessions auditable rather than guessed.
    """

    rows = conn.execute(
        """
        SELECT provider_event_type, model_name, position,
               last_input_tokens, last_output_tokens, last_cached_input_tokens,
               last_cache_write_tokens, last_reasoning_output_tokens, last_total_tokens,
               total_input_tokens, total_output_tokens, total_cached_input_tokens,
               total_cache_write_tokens, total_reasoning_output_tokens, total_tokens
        FROM session_provider_usage_events
        WHERE session_id = ?
          AND provider_event_type = 'token_count'
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    if not rows:
        return

    existing_models = [
        str(row[0]).strip()
        for row in conn.execute(
            "SELECT model_name FROM session_model_usage WHERE session_id = ? ORDER BY model_name",
            (session_id,),
        ).fetchall()
        if row[0] and str(row[0]).strip()
    ]

    # The cumulative is session-global, so we keep a single latest cumulative
    # for the whole session (rows are ordered by position, so the last row that
    # carries any total_* wins = highest position) attributed to the model named
    # on that row. summed_last_* stays per-model for Claude-style per-message
    # reporting, and is only used when no cumulative total appears at all.
    latest_total: tuple[int, int, int, int, int, int] | None = None
    latest_total_model = ""
    summed_last_by_model: dict[str, list[int]] = {}

    for row in rows:
        model_name = str(row[1]).strip() if row[1] else ""
        if not model_name:
            model_name = existing_models[0] if len(existing_models) == 1 else ""
        if not model_name:
            continue

        last_input = int(row[3] or 0)
        last_output = int(row[4] or 0)
        last_cache_read = int(row[5] or 0)
        last_cache_write = int(row[6] or 0)
        last_reasoning = int(row[7] or 0)
        last_total = int(row[8] or 0)
        total_input = int(row[9] or 0)
        total_output = int(row[10] or 0)
        total_cache_read = int(row[11] or 0)
        total_cache_write = int(row[12] or 0)
        total_reasoning = int(row[13] or 0)
        total_tokens = int(row[14] or 0)

        if total_input or total_output or total_cache_read or total_cache_write or total_reasoning or total_tokens:
            latest_total = (
                total_input,
                total_output,
                total_cache_read,
                total_cache_write,
                total_reasoning,
                total_tokens,
            )
            latest_total_model = model_name
            continue

        if last_input or last_output or last_cache_read or last_cache_write or last_reasoning or last_total:
            bucket = summed_last_by_model.setdefault(model_name, [0, 0, 0, 0, 0])
            bucket[0] += last_input
            bucket[1] += last_output
            bucket[2] += last_cache_read
            bucket[3] += last_cache_write
            bucket[4] += last_reasoning

    if latest_total is not None:
        # Session-global cumulative: one rollup for the latest model. The
        # cumulative already subsumes every per-request last_*, so summed_last
        # rows are intentionally not written (writing them too double-counts).
        lane_input, lane_output, lane_cache_read, lane_cache_write = _provider_usage_disjoint_lanes(
            latest_total[0], latest_total[1], latest_total[2], latest_total[3]
        )
        _upsert_provider_usage_model_rollup(
            conn,
            session_id,
            latest_total_model,
            input_tokens=lane_input,
            output_tokens=lane_output,
            cache_read_tokens=lane_cache_read,
            cache_write_tokens=lane_cache_write,
        )
        return

    for model_name, summed_totals in summed_last_by_model.items():
        lane_input, lane_output, lane_cache_read, lane_cache_write = _provider_usage_disjoint_lanes(
            summed_totals[0], summed_totals[1], summed_totals[2], summed_totals[3]
        )
        _upsert_provider_usage_model_rollup(
            conn,
            session_id,
            model_name,
            input_tokens=lane_input,
            output_tokens=lane_output,
            cache_read_tokens=lane_cache_read,
            cache_write_tokens=lane_cache_write,
        )


def _aggregate_appended_provider_usage_into_model_usage(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    start_position: int,
) -> None:
    """Fold only newly appended provider usage events into model usage rows."""

    rows = conn.execute(
        """
        SELECT model_name, position,
               last_input_tokens, last_output_tokens, last_cached_input_tokens,
               last_cache_write_tokens, last_reasoning_output_tokens, last_total_tokens,
               total_input_tokens, total_output_tokens, total_cached_input_tokens,
               total_cache_write_tokens, total_reasoning_output_tokens, total_tokens
        FROM session_provider_usage_events
        WHERE session_id = ?
          AND provider_event_type = 'token_count'
          AND position >= ?
        ORDER BY position
        """,
        (session_id, start_position),
    ).fetchall()
    if not rows:
        return

    existing_models = _provider_usage_existing_models(conn, session_id)
    # The cumulative total_* is session-global (see the full-write aggregator).
    # The highest-position appended row that carries any total_* therefore holds
    # the authoritative running total for the *whole* session, including rows
    # before start_position, so we keep one session-wide latest cumulative
    # rather than partitioning it per model (#2472).
    latest_total: tuple[int, int, int, int, int, int] | None = None
    latest_total_model = ""
    summed_last_by_model: dict[str, list[int]] = {}

    for row in rows:
        model_name = _provider_usage_model_name(row[0], existing_models)
        if not model_name:
            continue

        last_input = int(row[2] or 0)
        last_output = int(row[3] or 0)
        last_cache_read = int(row[4] or 0)
        last_cache_write = int(row[5] or 0)
        last_reasoning = int(row[6] or 0)
        last_total = int(row[7] or 0)
        total_input = int(row[8] or 0)
        total_output = int(row[9] or 0)
        total_cache_read = int(row[10] or 0)
        total_cache_write = int(row[11] or 0)
        total_reasoning = int(row[12] or 0)
        total_tokens = int(row[13] or 0)

        if total_input or total_output or total_cache_read or total_cache_write or total_reasoning or total_tokens:
            latest_total = (
                total_input,
                total_output,
                total_cache_read,
                total_cache_write,
                total_reasoning,
                total_tokens,
            )
            latest_total_model = model_name
            continue

        if last_input or last_output or last_cache_read or last_cache_write or last_reasoning or last_total:
            bucket = summed_last_by_model.setdefault(model_name, [0, 0, 0, 0, 0])
            bucket[0] += last_input
            bucket[1] += last_output
            bucket[2] += last_cache_read
            bucket[3] += last_cache_write
            bucket[4] += last_reasoning

    if latest_total is not None:
        # Overwrite the single session-global cumulative rollup. If the model
        # switched since a prior append window, the earlier model's cumulative
        # rollup is now stale (the new cumulative already subsumes it); clear
        # those stale origin_reported cumulative rows so they are not summed
        # back in alongside the new latest.
        lane_input, lane_output, lane_cache_read, lane_cache_write = _provider_usage_disjoint_lanes(
            latest_total[0], latest_total[1], latest_total[2], latest_total[3]
        )
        _upsert_provider_usage_model_rollup(
            conn,
            session_id,
            latest_total_model,
            input_tokens=lane_input,
            output_tokens=lane_output,
            cache_read_tokens=lane_cache_read,
            cache_write_tokens=lane_cache_write,
        )
        _clear_stale_cumulative_rollups(conn, session_id, keep_model=latest_total_model)
        return

    for model_name, summed_totals in summed_last_by_model.items():
        if _provider_usage_has_cumulative_total(conn, session_id, model_name):
            continue
        lane_input, lane_output, lane_cache_read, lane_cache_write = _provider_usage_disjoint_lanes(
            summed_totals[0], summed_totals[1], summed_totals[2], summed_totals[3]
        )
        _increment_provider_usage_model_rollup(
            conn,
            session_id,
            model_name,
            input_tokens=lane_input,
            output_tokens=lane_output,
            cache_read_tokens=lane_cache_read,
            cache_write_tokens=lane_cache_write,
        )


def _provider_usage_existing_models(conn: sqlite3.Connection, session_id: str) -> list[str]:
    return [
        str(row[0]).strip()
        for row in conn.execute(
            "SELECT model_name FROM session_model_usage WHERE session_id = ? ORDER BY model_name",
            (session_id,),
        ).fetchall()
        if row[0] and str(row[0]).strip()
    ]


def _provider_usage_model_name(model_name: object, existing_models: Sequence[str]) -> str:
    resolved = str(model_name).strip() if model_name else ""
    if resolved:
        return resolved
    return existing_models[0] if len(existing_models) == 1 else ""


def _provider_usage_has_cumulative_total(conn: sqlite3.Connection, session_id: str, model_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM session_provider_usage_events
        WHERE session_id = ?
          AND provider_event_type = 'token_count'
          AND model_name = ?
          AND (
            total_input_tokens != 0
            OR total_output_tokens != 0
            OR total_cached_input_tokens != 0
            OR total_cache_write_tokens != 0
            OR total_reasoning_output_tokens != 0
            OR total_tokens != 0
          )
        LIMIT 1
        """,
        (session_id, model_name),
    ).fetchone()
    return row is not None


def _clear_stale_cumulative_rollups(conn: sqlite3.Connection, session_id: str, *, keep_model: str) -> None:
    """Zero origin-reported token rollups for all models except ``keep_model``.

    The Codex cumulative total is session-global, so exactly one rollup row
    should carry it. When an append window's latest cumulative is attributed to
    a different model than a previous window, the earlier model's rollup still
    holds a (now-subsumed) cumulative; left in place it would be summed back in
    on read. This resets those stale token counts to zero while keeping the
    model row itself (#2472).
    """
    conn.execute(
        """
        UPDATE session_model_usage
        SET input_tokens = 0,
            output_tokens = 0,
            cache_read_tokens = 0,
            cache_write_tokens = 0,
            cost_usd = NULL,
            priced_with = NULL,
            priced_at_ms = NULL
        WHERE session_id = ?
          AND model_name != ?
          AND cost_provenance = 'origin_reported'
        """,
        (session_id, keep_model),
    )


def _upsert_provider_usage_model_rollup(
    conn: sqlite3.Connection,
    session_id: str,
    model_name: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> None:
    conn.execute(
        """
        INSERT INTO session_model_usage (
            session_id, model_name,
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            cost_provenance
        ) VALUES (?, ?, ?, ?, ?, ?, 'origin_reported')
        ON CONFLICT(session_id, model_name) DO UPDATE SET
            input_tokens       = excluded.input_tokens,
            output_tokens      = excluded.output_tokens,
            cache_read_tokens  = excluded.cache_read_tokens,
            cache_write_tokens = excluded.cache_write_tokens,
            cost_provenance    = excluded.cost_provenance,
            cost_usd           = NULL,
            priced_with        = NULL,
            priced_at_ms       = NULL
        """,
        (
            session_id,
            model_name,
            max(int(input_tokens), 0),
            max(int(output_tokens), 0),
            max(int(cache_read_tokens), 0),
            max(int(cache_write_tokens), 0),
        ),
    )


def _increment_provider_usage_model_rollup(
    conn: sqlite3.Connection,
    session_id: str,
    model_name: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> None:
    conn.execute(
        """
        INSERT INTO session_model_usage (
            session_id, model_name,
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            cost_provenance
        ) VALUES (?, ?, ?, ?, ?, ?, 'origin_reported')
        ON CONFLICT(session_id, model_name) DO UPDATE SET
            input_tokens       = COALESCE(session_model_usage.input_tokens, 0) + excluded.input_tokens,
            output_tokens      = COALESCE(session_model_usage.output_tokens, 0) + excluded.output_tokens,
            cache_read_tokens  = COALESCE(session_model_usage.cache_read_tokens, 0) + excluded.cache_read_tokens,
            cache_write_tokens = COALESCE(session_model_usage.cache_write_tokens, 0) + excluded.cache_write_tokens,
            cost_provenance    = excluded.cost_provenance,
            cost_usd           = NULL,
            priced_with        = NULL,
            priced_at_ms       = NULL
        """,
        (
            session_id,
            model_name,
            max(int(input_tokens), 0),
            max(int(output_tokens), 0),
            max(int(cache_read_tokens), 0),
            max(int(cache_write_tokens), 0),
        ),
    )


def _write_working_dirs(conn: sqlite3.Connection, session_id: str, working_directories: Iterable[str]) -> None:
    for position, path in enumerate(working_directories):
        conn.execute(
            """
            INSERT OR REPLACE INTO session_working_dirs (session_id, path, position)
            VALUES (?, ?, ?)
            """,
            (session_id, _sqlite_text(path), position),
        )


def _write_reported_costs(
    conn: sqlite3.Connection,
    session_id: str,
    session: ParsedSession,
    *,
    replace_existing_model_rows: bool = True,
    aggregate_message_tokens: bool = True,
) -> None:
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
    model_usage_sql = (
        """
        INSERT OR REPLACE INTO session_model_usage (
            session_id, model_name, cost_provenance
        ) VALUES (?, ?, 'origin_reported')
        """
        if replace_existing_model_rows
        else """
        INSERT INTO session_model_usage (
            session_id, model_name, cost_provenance
        ) VALUES (?, ?, 'origin_reported')
        ON CONFLICT(session_id, model_name) DO NOTHING
        """
    )
    for model_name in sorted(model_names):
        conn.execute(model_usage_sql, (session_id, _sqlite_text(model_name)))
    if aggregate_message_tokens:
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
            WHERE NOT (
                session_model_usage.cost_provenance = 'origin_reported'
                AND (
                    COALESCE(session_model_usage.input_tokens, 0) != 0
                    OR COALESCE(session_model_usage.output_tokens, 0) != 0
                    OR COALESCE(session_model_usage.cache_read_tokens, 0) != 0
                    OR COALESCE(session_model_usage.cache_write_tokens, 0) != 0
                )
            )
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
                repo_name = COALESCE(NULLIF(excluded.repo_name, ''), repos.repo_name),
                first_seen_at_ms = MIN(repos.first_seen_at_ms, excluded.first_seen_at_ms),
                last_seen_at_ms = MAX(repos.last_seen_at_ms, excluded.last_seen_at_ms)
            """,
            # repos.repo_name is NOT NULL DEFAULT ''. _repo_name() returns None
            # when no name can be derived (e.g. a session whose cwd is "/" or
            # "."): insert the schema's empty-string sentinel instead of NULL so
            # the session is not dropped, while the NULLIF above keeps a later
            # re-ingest from clobbering a previously-derived name with ''.
            (
                _sqlite_text(origin_url),
                _sqlite_text(root_path),
                _sqlite_text(repo_name or ""),
                observed_at_ms or 0,
                observed_at_ms or 0,
            ),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO session_repos (
                session_id, repo_id, branch_name, observed_at_ms
            ) VALUES (?, ?, ?, ?)
            """,
            (session_id, repo_id, _sqlite_text(session.git_branch or ""), observed_at_ms or 0),
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
                    _sqlite_text(session.git_commit_hash),
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


def _message_blocks(message: ParsedMessage) -> Sequence[ParsedContentBlock]:
    if message.blocks:
        return message.blocks
    if message.text:
        return (ParsedContentBlock(type=BlockType.TEXT, text=message.text),)
    return ()


# --- Lineage normalization (#2467): prefix-inheritance tail extraction ---------
#
# A fork / resume / spawned subagent / auto-compaction child rollout physically
# copies the parent's context as a leading prefix. We store only the child's
# divergent tail plus a lineage edge with a branch point, so each real message is
# stored exactly once. The branch point is found by conservative contiguous
# prefix-alignment against the parent's *composed* transcript, using a per-message
# content signature (role + ordered block content). A message is treated as
# inherited only inside the matching leading run, so a genuinely-new block that
# happens to equal a parent block is never dropped.

_SIG_FIELD_SEP = "\x1f"
_SIG_BLOCK_SEP = "\x1e"
_MAX_LINEAGE_DEPTH = 64


def _canonical_json(value: object) -> str:
    """Stable JSON for signature comparison; accepts a value or a JSON string."""
    if value is None:
        return "null"
    parsed: object = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return value
    try:
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(parsed)


def _message_signature_from_blocks(role: str, block_fields: list[tuple[str, str, str, str]]) -> str:
    parts = [role]
    for block_type, text, tool_name, tool_input in block_fields:
        parts.append(_SIG_FIELD_SEP.join((block_type, text, tool_name, tool_input)))
    return hashlib.sha256(_SIG_BLOCK_SEP.join(parts).encode("utf-8", "surrogatepass")).hexdigest()


def _parsed_message_signature(message: ParsedMessage) -> str:
    role = _enum_value(message.role) or ""
    fields: list[tuple[str, str, str, str]] = []
    for block in _message_blocks(message):
        # Serialize tool_input through the same `_json_dumps` the writer uses to
        # store it, then canonicalize — so the parsed-side signature matches the
        # DB-side signature (which canonicalizes the stored JSON string). Calling
        # `_canonical_json` on the raw value would mis-handle scalar strings, which
        # it treats as JSON to re-parse (#2467 audit M6).
        tool_input = _canonical_json(_json_dumps(block.tool_input)) if block.tool_input is not None else "null"
        fields.append(
            (
                _block_type(block).value,
                block.text or "",
                block.tool_name or "",
                tool_input,
            )
        )
    return _message_signature_from_blocks(role, fields)


def _own_db_signatures(conn: sqlite3.Connection, session_id: str) -> list[tuple[str, str]]:
    """Return ``[(message_id, signature), ...]`` for ``session_id``'s OWN stored
    message rows (no inherited prefix). This is the expensive SQL+SHA-256 leg of
    composition; it depends only on the session's own rows, so it can be memoized
    per ingest batch and invalidated whenever those rows change."""
    own_rows = conn.execute(
        """
        SELECT m.message_id, m.position, m.role,
               b.block_type, b.text, b.tool_name, b.tool_input
        FROM messages m
        LEFT JOIN blocks b ON b.message_id = m.message_id
        WHERE m.session_id = ? AND m.variant_index = 0
        ORDER BY m.position, b.position
        """,
        (session_id,),
    ).fetchall()
    own: list[tuple[str, str]] = []
    cur_id: str | None = None
    cur_role = ""
    cur_blocks: list[tuple[str, str, str, str]] = []

    def flush() -> None:
        if cur_id is not None:
            own.append((cur_id, _message_signature_from_blocks(cur_role, cur_blocks)))

    for message_id, _position, role, block_type, text, tool_name, tool_input in own_rows:
        if message_id != cur_id:
            flush()
            cur_id = message_id
            cur_role = role or ""
            cur_blocks = []
        if block_type is not None:
            cur_blocks.append(
                (
                    block_type,
                    text or "",
                    tool_name or "",
                    _canonical_json(tool_input) if tool_input is not None else "null",
                )
            )
    flush()
    return own


def _composed_db_signatures(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    cache: dict[str, list[tuple[str, str]]] | None = None,
    _depth: int = 0,
) -> list[tuple[str, str]]:
    """Return ``[(message_id, signature), ...]`` for ``session_id``'s composed
    transcript (its inherited prefix + own tail), recursively resolving any
    prefix-sharing lineage edge. Mirrors the read-side composition.

    When ``cache`` is supplied, each session's OWN signatures are memoized by
    ``session_id`` for the life of one ingest batch. The composed (prefix+own)
    result is never cached because it embeds ancestor signatures that could be
    rewritten in the same batch; only the own-row leg is stable per session.
    """
    own = cache.get(session_id) if cache is not None else None
    if own is None:
        own = _own_db_signatures(conn, session_id)
        if cache is not None:
            cache[session_id] = own

    if _depth >= _MAX_LINEAGE_DEPTH:
        return own
    edge = conn.execute(
        """
        SELECT resolved_dst_session_id, branch_point_message_id
        FROM session_links
        WHERE src_session_id = ?
          AND inheritance = 'prefix-sharing'
          AND resolved_dst_session_id IS NOT NULL
          AND branch_point_message_id IS NOT NULL
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    if edge is None:
        return own
    parent_id, branch_point_message_id = edge
    parent_composed = _composed_db_signatures(conn, str(parent_id), cache=cache, _depth=_depth + 1)
    prefix: list[tuple[str, str]] = []
    for entry in parent_composed:
        prefix.append(entry)
        if entry[0] == branch_point_message_id:
            break
    return prefix + own


def _reextract_prefix_tail_db(
    conn: sqlite3.Connection,
    child_session_id: str,
    parent_session_id: str,
    *,
    cache: dict[str, list[tuple[str, str]]] | None = None,
) -> None:
    """Normalize a child that was stored whole because its parent was ingested
    later (#2467). Aligns the child's already-stored messages against the parent's
    composed transcript, deletes the inherited-prefix rows, and records the edge.
    Only runs while the lineage edge is still un-extracted (``inheritance`` NULL).
    """
    edge = conn.execute(
        """
        SELECT dst_origin, dst_native_id, link_type
        FROM session_links
        WHERE src_session_id = ?
          AND resolved_dst_session_id = ?
          AND inheritance IS NULL
        LIMIT 1
        """,
        (child_session_id, parent_session_id),
    ).fetchone()
    if edge is None:
        return
    dst_origin, dst_native_id, link_type = edge
    parent_composed = _composed_db_signatures(conn, parent_session_id, cache=cache)
    child_composed = _composed_db_signatures(conn, child_session_id, cache=cache)
    k = 0
    limit = min(len(parent_composed), len(child_composed))
    while k < limit and parent_composed[k][1] == child_composed[k][1]:
        k += 1

    def _set_edge(branch_point_message_id: str | None, inheritance: str) -> None:
        conn.execute(
            """
            UPDATE session_links
            SET branch_point_message_id = ?, inheritance = ?
            WHERE src_session_id = ? AND dst_origin = ? AND dst_native_id = ? AND link_type = ?
            """,
            (branch_point_message_id, inheritance, child_session_id, dst_origin, dst_native_id, link_type),
        )

    if k == 0:
        _set_edge(None, "spawned-fresh")
        return
    prefix_message_ids = [child_composed[i][0] for i in range(k)]
    placeholders = ",".join("?" for _ in prefix_message_ids)
    conn.execute(
        f"DELETE FROM messages WHERE message_id IN ({placeholders})",
        tuple(prefix_message_ids),
    )
    # The child's own rows just changed (inherited prefix deleted); drop its
    # memoized own-signatures so any later compose in this batch recomputes them.
    if cache is not None:
        cache.pop(child_session_id, None)
    _set_edge(parent_composed[k - 1][0], "prefix-sharing")
    _refresh_session_counts(conn, child_session_id)


def _extract_prefix_tail(
    conn: sqlite3.Connection,
    parent_session_id: str,
    messages: list[ParsedMessage],
    *,
    cache: dict[str, list[tuple[str, str]]] | None = None,
) -> tuple[str | None, str | None, list[ParsedMessage]]:
    """Align ``messages`` (the child's full parsed messages, which replay the
    parent's prefix) against the parent's composed transcript. Returns
    ``(branch_point_message_id, inheritance, tail_messages)``."""
    parent_composed = _composed_db_signatures(conn, parent_session_id, cache=cache)
    if not parent_composed:
        return (None, "spawned-fresh", messages)
    child_sigs = [_parsed_message_signature(m) for m in messages]
    k = 0
    limit = min(len(parent_composed), len(child_sigs))
    while k < limit and parent_composed[k][1] == child_sigs[k]:
        k += 1
    if k == 0:
        return (None, "spawned-fresh", messages)
    branch_point_message_id = parent_composed[k - 1][0]
    return (branch_point_message_id, "prefix-sharing", messages[k:])


def _prefix_sharing_edge_sync(conn: sqlite3.Connection, session_id: str) -> tuple[str, str] | None:
    """Return ``(parent_session_id, branch_point_message_id)`` for a resolved
    prefix-sharing lineage edge, else ``None``. Mirrors the async reader."""
    row = conn.execute(
        """
        SELECT resolved_dst_session_id, branch_point_message_id
        FROM session_links
        WHERE src_session_id = ?
          AND inheritance = 'prefix-sharing'
          AND resolved_dst_session_id IS NOT NULL
          AND branch_point_message_id IS NOT NULL
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return (str(row[0]), str(row[1]))


def _existing_parent_session_id(conn: sqlite3.Connection, session: ParsedSession, origin_value: str) -> str | None:
    parent_provider_id = session.parent_session_provider_id
    if not parent_provider_id:
        return None
    parent_session_id = archive_session_id(origin_value, parent_provider_id)
    row = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1",
        (parent_session_id,),
    ).fetchone()
    return parent_session_id if row is not None else None


def _active_leaf_message_id(
    session_id: str,
    messages: list[ParsedMessage],
    explicit_native_id: str | None,
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> str | None:
    if explicit_native_id:
        for fallback_position, message in enumerate(messages):
            if message.provider_message_id == explicit_native_id:
                return _message_id(
                    session_id,
                    message,
                    fallback_position,
                    position_offset=position_offset,
                    duplicate_native_ids=duplicate_native_ids,
                )
    for fallback_position, message in enumerate(messages):
        if message.is_active_leaf:
            return _message_id(
                session_id,
                message,
                fallback_position,
                position_offset=position_offset,
                duplicate_native_ids=duplicate_native_ids,
            )
    return (
        _message_id(
            session_id,
            messages[-1],
            len(messages) - 1,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        if messages
        else None
    )


def _message_id(
    session_id: str,
    message: ParsedMessage,
    fallback_position: int,
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> str:
    position = position_offset + (message.position if message.position is not None else 0)
    variant_index = message.variant_index if message.variant_index is not None else 0
    return archive_message_id(
        session_id,
        _effective_message_native_id(message, duplicate_native_ids),
        position=position if message.position is not None else position_offset + fallback_position,
        variant_index=variant_index,
    )


def _duplicate_message_native_ids(messages: Iterable[ParsedMessage]) -> frozenset[str]:
    counts = Counter(message.provider_message_id for message in messages if message.provider_message_id)
    return frozenset(native_id for native_id, count in counts.items() if count > 1)


def _effective_message_native_id(message: ParsedMessage, duplicate_native_ids: frozenset[str]) -> str | None:
    native_id = message.provider_message_id
    if native_id in duplicate_native_ids:
        return None
    return native_id


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


def _has_block(message: ParsedMessage, block_type: BlockType) -> int:
    return int(any(_enum_value(block.type) == block_type.value for block in message.blocks))


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


def _payload_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _payload_int(payload: Mapping[str, object], key: str) -> int:
    return _payload_optional_int(payload, key) or 0


def _payload_optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str) and value.strip():
        try:
            return max(int(float(value)), 0)
        except ValueError:
            return None
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


def _acquire_attachment_blob(attachment: ParsedAttachment) -> tuple[bytes | None, int, str]:
    """Acquire an attachment's bytes when available (#2468).

    Returns ``(blob_hash, byte_count, acquisition_status)``. Inline bytes present
    in the source export are written to the content-addressed blob store and the
    true SHA-256 is returned with status ``acquired``. Otherwise no blob is
    written: the hash is ``None`` and the status is ``unfetched`` (the bytes may
    be re-acquired later from ``source_url`` / provider file id). The former
    behavior — fabricating a 32-byte hash from the attachment id with no blob ever
    stored — is gone.
    """
    inline = attachment.inline_bytes
    if inline:
        from polylogue.storage.blob_store import get_blob_store

        hash_hex, size = get_blob_store().write_from_bytes(inline)
        return (bytes.fromhex(hash_hex), size, "acquired")
    return (None, attachment.size_bytes or 0, "unfetched")


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
                (ref_id, id_kind, _sqlite_text(native_id)),
            )


def _hash_bytes(*parts: str) -> bytes:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return digest.digest()


def _json_dumps(value: object) -> str:
    return json.dumps(_sqlite_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sqlite_text(value: str | None) -> str | None:
    if value is None:
        return None
    if not _SURROGATE_RE.search(value):
        return value
    return _SURROGATE_RE.sub("\ufffd", value)


def _sqlite_bool(value: bool | None) -> int | None:
    """Map an optional bool to SQLite 0/1, preserving None (unknown)."""
    if value is None:
        return None
    return 1 if value else 0


def _sqlite_json_value(value: object) -> object:
    if isinstance(value, str):
        return _sqlite_text(value)
    if isinstance(value, list):
        return [_sqlite_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sqlite_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(_sqlite_text(str(key))): _sqlite_json_value(item) for key, item in value.items()}
    return value


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
    "upsert_parser_ingest_flag_tags",
    "upsert_session_tag",
    "upsert_session_work_event",
    "read_archive_session_envelope",
    "search_archive_blocks",
    "write_parsed_session_to_archive",
]
