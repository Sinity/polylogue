"""Shared SQLite relations for run-projection reads.

These relations are the query substrate for ``run``, ``observed-event``, and
``context-snapshot`` reads. They synthesize source rows directly from
``sessions`` and ``blocks`` (polylogue-dab: there is no materialized cache
table to union anymore -- ``include_materialized=True`` raises).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Protocol

from polylogue.archive.query.predicate import QueryBoolPredicate, QueryFieldPredicate, QueryPredicate
from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.core.types import SessionId
from polylogue.insights.run_projection import ContextSnapshot, ObservedEvent, ProjectedRun
from polylogue.storage.runtime import (
    SessionContextSnapshotRecord,
    SessionObservedEventRecord,
    SessionRunRecord,
)


class RowLike(Protocol):
    def __getitem__(self, key: str) -> object: ...


def _tuple_from_json_array(value: object) -> tuple[str, ...]:
    loaded = json.loads(str(value or "[]"))
    if not isinstance(loaded, list):
        return ()
    return tuple(str(item) for item in loaded if item is not None)


def _int_value(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float | str | bytes | bytearray):
        return int(value)
    return default


def _in_or_equals_clause(column: str, values: tuple[str, ...], *, lower: bool = False) -> tuple[str, list[object]]:
    normalized = tuple(value.strip().lower() if lower else value.strip() for value in values if value.strip())
    if not normalized:
        return "1=1", []
    expr = f"LOWER({column})" if lower else column
    if len(normalized) == 1:
        return f"{expr} = ?", [normalized[0]]
    placeholders = ", ".join("?" for _ in normalized)
    return f"{expr} IN ({placeholders})", list(normalized)


def observed_event_source_pushdown(predicate: QueryPredicate) -> tuple[str, list[object]]:
    """Return source-block pushdown SQL for selective observed-event predicates."""

    clauses: list[str] = []
    params: list[object] = []
    selective = False

    def add_clause(clause: str, clause_params: list[object], *, is_selective: bool) -> None:
        nonlocal selective
        if clause:
            clauses.append(f"({clause})")
            params.extend(clause_params)
            selective = selective or is_selective

    def visit(current: QueryPredicate) -> bool:
        if isinstance(current, QueryBoolPredicate):
            if current.op != "and":
                return False
            return all(visit(child) for child in current.children)
        if isinstance(current, QueryFieldPredicate):
            field = current.bound_field_name(context="lowering observed-event source predicates")
            if field == "kind":
                # 'tool_finished' discriminates against session_started_base
                # (the other branch of the source_observed_events UNION), so
                # it is itself selective -- without this, "kind:tool_finished"
                # alone (no tool:/handler:/status: predicate alongside it)
                # falls through to the "not selective" 0=1 fallback below and
                # silently returns zero rows despite real matching evidence.
                add_clause("'tool_finished' = ?", ["tool_finished"], is_selective=True)
                return "tool_finished" in {value.strip().lower() for value in current.values}
            if field == "delivery_state":
                add_clause("'observed' = ?", ["observed"], is_selective=False)
                return "observed" in {value.strip().lower() for value in current.values}
            if field == "tool":
                normalized = tuple(value.strip().lower() for value in current.values if value.strip())
                if not normalized:
                    return True
                if len(normalized) == 1:
                    add_clause(
                        "COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown') = ?",
                        [normalized[0]],
                        is_selective=True,
                    )
                else:
                    placeholders = ", ".join("?" for _ in normalized)
                    add_clause(
                        f"COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown') IN ({placeholders})",
                        list(normalized),
                        is_selective=True,
                    )
                return True
            if field == "handler":
                normalized = tuple(value.strip().lower() for value in current.values if value.strip())
                if not normalized:
                    return True
                handler_clauses: list[str] = []
                handler_params: list[object] = []
                for value in normalized:
                    if value == "mcp":
                        handler_clauses.append(
                            "(COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown') >= ? "
                            "AND COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown') < ?)"
                        )
                        handler_params.extend(["mcp__", "mcp_`"])
                    elif value == "shell":
                        handler_clauses.append("COALESCE(u.tool_command, '') <> ''")
                    else:
                        handler_clauses.append("COALESCE(NULLIF(u.semantic_type, ''), 'tool_use') = ?")
                        handler_params.append(value)
                add_clause(" OR ".join(handler_clauses), handler_params, is_selective=True)
                return True
            if field == "status":
                status_expr = (
                    "CASE "
                    "WHEN r.tool_result_exit_code IS NOT NULL "
                    "THEN CASE WHEN r.tool_result_exit_code = 0 THEN 'ok' ELSE 'failed' END "
                    "WHEN r.tool_result_is_error = 1 THEN 'failed' "
                    "WHEN r.tool_result_is_error = 0 THEN 'ok' "
                    "ELSE 'unknown' END"
                )
                clause, clause_params = _in_or_equals_clause(status_expr, current.values, lower=True)
                add_clause(clause, clause_params, is_selective=True)
                return True
        return True

    supported = visit(predicate)
    if not supported or not selective:
        return "0=1", []
    return " AND ".join(clauses) if clauses else "1=1", params


def run_relation_sql(*, include_materialized: bool = False) -> str:
    """Construct run projection relation SQL.

    The materialized cache tables (session_runs) are no longer populated
    after polylogue-dab. Always use source-derived. This function
    must now return source-only queries; the materialized union code is
    dead and reserved for future deletion.
    """
    if include_materialized:
        raise ValueError(
            "session_runs materialized table is no longer written to (polylogue-dab). "
            "Pass include_materialized=False or omit the argument."
        )
    return """
WITH source_runs AS (
    SELECT
        'source' AS row_source,
        'run:' || s0.session_id AS run_ref,
        s0.session_id AS session_id,
        0 AS position,
        printf('%016d', COALESCE(s0.updated_at_ms, s0.created_at_ms, 0)) AS source_updated_at,
        s0.session_id AS native_session_id,
        s0.parent_session_id AS native_parent_session_id,
        CASE WHEN s0.parent_session_id IS NOT NULL THEN 'run:' || s0.parent_session_id ELSE NULL END AS parent_run_ref,
        'agent:' ||
            CASE
                WHEN s0.origin = 'codex-session' THEN 'codex'
                WHEN s0.origin = 'claude-code-session' THEN 'claude-code'
                WHEN s0.origin = 'chatgpt-export' THEN 'chatgpt'
                WHEN s0.origin IN ('gemini-cli-session', 'hermes-session', 'antigravity-session') THEN 'local'
                ELSE 'unknown'
            END || CASE WHEN s0.branch_type = 'subagent' THEN '/subagent' ELSE '/main' END AS agent_ref,
        json_array('run:' || s0.session_id) AS lineage_refs_json,
        s0.origin AS provider_origin,
        CASE
            WHEN s0.origin = 'codex-session' THEN 'codex'
            WHEN s0.origin = 'claude-code-session' THEN 'claude-code'
            WHEN s0.origin = 'chatgpt-export' THEN 'chatgpt'
            WHEN s0.origin IN ('gemini-cli-session', 'hermes-session', 'antigravity-session') THEN 'local'
            ELSE 'unknown'
        END AS harness,
        -- branch_type='subagent' (BranchType.SUBAGENT, core/enums.py) is the
        -- session-level flag a parsed subagent session already carries; no
        -- Task-tool-report parsing needed here (unlike the pre-polylogue-dab
        -- Python writer in insights/run_projection.py, which additionally
        -- derived agent_ref's subagent_type segment from the parent's Task
        -- tool_input -- that finer distinction is not reconstructed here).
        CASE WHEN s0.branch_type = 'subagent' THEN 'subagent' ELSE 'main' END AS role,
        COALESCE(NULLIF(s0.title, ''), s0.session_id) AS title,
        NULL AS cwd,
        s0.git_branch AS git_branch,
        CASE WHEN s0.message_count > 0 OR s0.tool_use_count > 0 THEN 'completed' ELSE 'unknown' END AS status,
        'raw' AS confidence,
        s0.session_id AS transcript_ref,
        json_array(s0.session_id) AS evidence_refs_json,
        'context-snapshot:' || s0.session_id || ':' ||
            CASE
                WHEN s0.branch_type = 'subagent' THEN 'subagent_start'
                WHEN s0.branch_type = 'continuation' THEN 'resume'
                ELSE 'session_start'
            END AS context_snapshot_ref,
        trim(COALESCE(s0.title, '') || ' ' || COALESCE(s0.native_id, '') || ' ' || COALESCE(s0.git_branch, '')) AS search_text,
        NULL AS payload_json,
        1 AS materializer_version,
        '' AS materialized_at
    FROM sessions s0
),
runs AS (
    SELECT * FROM source_runs
)
"""


def observed_event_relation_sql(*, source_where: str, include_materialized: bool = False) -> str:
    """Construct observed-event projection relation SQL.

    The materialized cache tables (session_observed_events) are no longer populated
    after polylogue-dab. Always use source-derived.
    """
    if include_materialized:
        raise ValueError(
            "session_observed_events materialized table is no longer written to (polylogue-dab). "
            "Pass include_materialized=False or omit the argument."
        )
    return f"""
WITH session_started_base AS (
    SELECT
        'source' AS row_source,
        'observed-event:' || s0.session_id || ':session_started' AS event_ref,
        s0.session_id AS session_id,
        'run:' || s0.session_id AS run_ref,
        0 AS position,
        printf('%016d', COALESCE(s0.created_at_ms, s0.updated_at_ms, 0)) AS source_updated_at,
        'session_started' AS kind,
        COALESCE(NULLIF(s0.title, ''), s0.session_id) AS summary,
        'observed' AS delivery_state,
        'session:' || s0.session_id AS subject_ref,
        json_array('session:' || s0.session_id) AS object_refs_json,
        json_array(s0.session_id) AS evidence_refs_json,
        json_object('origin', s0.origin, 'native_id', s0.native_id) AS payload_json,
        trim(COALESCE(s0.title, '') || ' ' || COALESCE(s0.native_id, '') || ' ' || COALESCE(s0.origin, '')) AS search_text,
        NULL AS subject_message_id,
        NULL AS tool_use_position,
        NULL AS result_message_id,
        NULL AS result_position,
        1 AS materializer_version,
        '' AS materialized_at
    FROM sessions s0
),
tool_finished_base AS (
    SELECT
        'source' AS row_source,
        'observed-event:' || u.block_id || ':tool_finished' AS event_ref,
        u.session_id AS session_id,
        'run:' || u.session_id AS run_ref,
        u.rowid AS position,
        NULL AS source_updated_at,
        'tool_finished' AS kind,
        COALESCE(NULLIF(u.tool_name, ''), 'unknown') AS tool_name,
        u.tool_id AS tool_id,
        u.tool_command AS command,
        u.message_id AS subject_message_id,
        u.position AS tool_use_position,
        r.message_id AS result_message_id,
        r.position AS result_position,
        CASE
            WHEN COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown') >= 'mcp__'
                AND COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown') < 'mcp_`'
                THEN 'mcp'
            WHEN COALESCE(u.tool_command, '') <> '' THEN 'shell'
            ELSE COALESCE(NULLIF(u.semantic_type, ''), 'tool_use')
        END AS handler_kind,
        CASE
            WHEN r.tool_result_exit_code IS NOT NULL
                THEN CASE WHEN r.tool_result_exit_code = 0 THEN 'ok' ELSE 'failed' END
            WHEN r.tool_result_is_error = 1 THEN 'failed'
            WHEN r.tool_result_is_error = 0 THEN 'ok'
            ELSE 'unknown'
        END AS status,
        CASE
            WHEN u.tool_id IS NOT NULL AND u.tool_id <> ''
                THEN json_array('tool-call:' || u.session_id || ':' || u.tool_id)
            ELSE '[]'
        END AS object_refs_json,
        json_array(
            u.session_id || '::' || u.message_id || '::' || u.position,
            r.session_id || '::' || r.message_id || '::' || r.position
        ) AS evidence_refs_json,
        trim(COALESCE(u.search_text, '') || ' ' || COALESCE(r.search_text, '')) AS search_text
    FROM blocks u
    JOIN blocks r
        ON r.session_id = u.session_id
        AND r.tool_id = u.tool_id
        AND r.block_type = 'tool_result'
    WHERE u.block_type = 'tool_use'
        AND u.tool_id IS NOT NULL
        AND u.tool_id <> ''
        AND ({source_where})
),
source_observed_events AS (
    SELECT * FROM session_started_base
    UNION ALL
    SELECT
        row_source,
        event_ref,
        session_id,
        run_ref,
        position,
        source_updated_at,
        kind,
        tool_name || ' [' || handler_kind || '] (' || status || ')'
            || CASE WHEN COALESCE(command, '') <> '' THEN ' - ' || command ELSE '' END AS summary,
        'observed' AS delivery_state,
        'message:' || subject_message_id AS subject_ref,
        object_refs_json,
        evidence_refs_json,
        json_object(
            'tool_name', tool_name,
            'tool_id', tool_id,
            'command', command,
            'handler_kind', handler_kind,
            'status', status
        ) AS payload_json,
        search_text,
        subject_message_id,
        tool_use_position,
        result_message_id,
        result_position,
        1 AS materializer_version,
        '' AS materialized_at
    FROM tool_finished_base
),
observed_events AS (
    SELECT * FROM source_observed_events
)
"""


def context_snapshot_relation_sql(*, include_materialized: bool = False) -> str:
    """Construct context-snapshot projection relation SQL.

    The materialized cache tables (session_context_snapshots) are no longer populated
    after polylogue-dab. Always use source-derived.
    """
    if include_materialized:
        raise ValueError(
            "session_context_snapshots materialized table is no longer written to (polylogue-dab). "
            "Pass include_materialized=False or omit the argument."
        )
    return """
WITH source_context_snapshots AS (
    SELECT
        'source' AS row_source,
        'context-snapshot:' || s0.session_id || ':' ||
            CASE
                WHEN s0.branch_type = 'subagent' THEN 'subagent_start'
                WHEN s0.branch_type = 'continuation' THEN 'resume'
                ELSE 'session_start'
            END AS snapshot_ref,
        s0.session_id AS session_id,
        'run:' || s0.session_id AS run_ref,
        0 AS position,
        printf('%016d', COALESCE(s0.updated_at_ms, s0.created_at_ms, 0)) AS source_updated_at,
        CASE
            WHEN s0.branch_type = 'subagent' THEN 'subagent_start'
            WHEN s0.branch_type = 'continuation' THEN 'resume'
            ELSE 'session_start'
        END AS boundary,
        'unknown' AS inheritance_mode,
        json_array('session:' || s0.session_id) AS segment_refs_json,
        json_array(s0.session_id) AS evidence_refs_json,
        json_object('source', 'archive-session') AS metadata_json,
        trim(COALESCE(s0.title, '') || ' ' || COALESCE(s0.native_id, '')) AS search_text,
        NULL AS payload_json,
        1 AS materializer_version,
        '' AS materialized_at
    FROM sessions s0
),
context_snapshots AS (
    SELECT * FROM source_context_snapshots
)
"""


def projected_run_from_row(row: RowLike) -> ProjectedRun:
    """Hydrate ProjectedRun from row.

    The materialized cache path (payload_json deserialization) was removed
    in polylogue-dab. Always use source-derived construction.
    """
    # Regression guard: ensure no stray materialized rows slipped through
    if str(row["row_source"]) == "materialized":
        raise ValueError(
            "Unexpected materialized row in projected_run_from_row; "
            "session_runs table should not be populated after polylogue-dab."
        )
    return ProjectedRun(
        run_ref=ObjectRef.parse(str(row["run_ref"])),
        native_session_id=str(row["native_session_id"]) if row["native_session_id"] is not None else None,
        native_parent_session_id=str(row["native_parent_session_id"])
        if row["native_parent_session_id"] is not None
        else None,
        parent_run_ref=ObjectRef.parse(str(row["parent_run_ref"])) if row["parent_run_ref"] is not None else None,
        agent_ref=ObjectRef.parse(str(row["agent_ref"])) if row["agent_ref"] is not None else None,
        lineage_refs=tuple(ObjectRef.parse(ref) for ref in _tuple_from_json_array(row["lineage_refs_json"])),
        provider_origin=str(row["provider_origin"]),
        harness=str(row["harness"]),  # type: ignore[arg-type]
        role=str(row["role"]),  # type: ignore[arg-type]
        title=str(row["title"]),
        cwd=str(row["cwd"]) if row["cwd"] is not None else None,
        git_branch=str(row["git_branch"]) if row["git_branch"] is not None else None,
        status=str(row["status"]),  # type: ignore[arg-type]
        confidence="raw",
        transcript_ref=EvidenceRef.parse(str(row["transcript_ref"])) if row["transcript_ref"] is not None else None,
        evidence_refs=tuple(EvidenceRef.parse(ref) for ref in _tuple_from_json_array(row["evidence_refs_json"])),
        context_snapshot_ref=ObjectRef.parse(str(row["context_snapshot_ref"]))
        if row["context_snapshot_ref"] is not None
        else None,
    )


def observed_event_from_row(row: RowLike) -> ObservedEvent:
    """Hydrate ObservedEvent from row.

    The materialized cache path (payload_json deserialization) was removed
    in polylogue-dab. Always use source-derived construction.
    """
    # Regression guard: ensure no stray materialized rows slipped through
    if str(row["row_source"]) == "materialized":
        raise ValueError(
            "Unexpected materialized row in observed_event_from_row; "
            "session_observed_events table should not be populated after polylogue-dab."
        )
    payload = json.loads(str(row["payload_json"] or "{}"))
    subject_ref = row["subject_ref"]
    return ObservedEvent(
        event_ref=ObjectRef.parse(str(row["event_ref"])),
        kind=str(row["kind"]),  # type: ignore[arg-type]
        run_ref=ObjectRef.parse(str(row["run_ref"])),
        summary=str(row["summary"]),
        delivery_state="observed",
        subject_ref=ObjectRef.parse(str(subject_ref)) if subject_ref is not None else None,
        object_refs=tuple(ObjectRef.parse(ref) for ref in _tuple_from_json_array(row["object_refs_json"])),
        evidence_refs=tuple(EvidenceRef.parse(ref) for ref in _tuple_from_json_array(row["evidence_refs_json"])),
        tool_name=str(payload["tool_name"]) if payload.get("tool_name") is not None else None,
        tool_id=str(payload["tool_id"]) if payload.get("tool_id") is not None else None,
        command=str(payload["command"]) if payload.get("command") is not None else None,
        handler_kind=str(payload["handler_kind"]) if payload.get("handler_kind") is not None else None,
        status=str(payload["status"]) if payload.get("status") is not None else None,
    )


def context_snapshot_from_row(row: RowLike) -> ContextSnapshot:
    """Hydrate ContextSnapshot from row.

    The materialized cache path (payload_json deserialization) was removed
    in polylogue-dab. Always use source-derived construction.
    """
    # Regression guard: ensure no stray materialized rows slipped through
    if str(row["row_source"]) == "materialized":
        raise ValueError(
            "Unexpected materialized row in context_snapshot_from_row; "
            "session_context_snapshots table should not be populated after polylogue-dab."
        )
    return ContextSnapshot(
        snapshot_ref=ObjectRef.parse(str(row["snapshot_ref"])),
        run_ref=ObjectRef.parse(str(row["run_ref"])),
        boundary=str(row["boundary"]),  # type: ignore[arg-type]
        inheritance_mode="unknown",
        segment_refs=tuple(ObjectRef.parse(ref) for ref in _tuple_from_json_array(row["segment_refs_json"])),
        evidence_refs=tuple(EvidenceRef.parse(ref) for ref in _tuple_from_json_array(row["evidence_refs_json"])),
        metadata=dict(json.loads(str(row["metadata_json"] or "{}"))),
    )


def row_to_session_run_record(row: RowLike) -> SessionRunRecord:
    return SessionRunRecord(
        session_id=SessionId(str(row["session_id"])),
        position=_int_value(row["position"], default=0),
        materializer_version=_int_value(row["materializer_version"], default=1),
        materialized_at=str(row["materialized_at"] or ""),
        source_updated_at=str(row["source_updated_at"]) if row["source_updated_at"] is not None else None,
        run=projected_run_from_row(row),
        search_text=str(row["search_text"] or ""),
    )


def row_to_session_observed_event_record(row: RowLike) -> SessionObservedEventRecord:
    return SessionObservedEventRecord(
        session_id=SessionId(str(row["session_id"])),
        position=_int_value(row["position"], default=0),
        materializer_version=_int_value(row["materializer_version"], default=1),
        materialized_at=str(row["materialized_at"] or ""),
        source_updated_at=str(row["source_updated_at"]) if row["source_updated_at"] is not None else None,
        event=observed_event_from_row(row),
        search_text=str(row["search_text"] or ""),
    )


def row_to_session_context_snapshot_record(row: RowLike) -> SessionContextSnapshotRecord:
    return SessionContextSnapshotRecord(
        session_id=SessionId(str(row["session_id"])),
        position=_int_value(row["position"], default=0),
        materializer_version=_int_value(row["materializer_version"], default=1),
        materialized_at=str(row["materialized_at"] or ""),
        source_updated_at=str(row["source_updated_at"]) if row["source_updated_at"] is not None else None,
        snapshot=context_snapshot_from_row(row),
        search_text=str(row["search_text"] or ""),
    )


def table_exists_sync(conn: sqlite3.Connection, table_name: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
    )
