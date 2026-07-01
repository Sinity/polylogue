"""Small archive-root façade over archive source/index/user tiers."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TypedDict, cast

from polylogue.archive.query.metadata import COUNT_QUERY_FIELD_REGISTRY, NUMERIC_QUERY_FIELD_REGISTRY
from polylogue.archive.query.path_prefix import escaped_sql_path_prefix_patterns
from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryExistsPredicate,
    QueryFieldPredicate,
    QueryFieldRef,
    QueryLineagePredicate,
    QueryNotPredicate,
    QueryPredicate,
    QuerySequencePredicate,
    QueryTextPredicate,
)
from polylogue.archive.semantic.pricing import (
    CostBasisPayload,
    CostEstimatePayload,
    CostEstimateStatus,
    CostUnavailableReason,
    _normalize_model,
)
from polylogue.archive.stats import ArchiveStats
from polylogue.core.dates import parse_date
from polylogue.core.enums import Provider
from polylogue.core.json import JSONValue, require_json_value
from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.core.sources import origin_from_provider
from polylogue.insights.affordance_usage import (
    clean_patterns as _clean_affordance_patterns,
)
from polylogue.insights.affordance_usage import (
    evidence_kind_for_row as _affordance_evidence_kind,
)
from polylogue.insights.affordance_usage import (
    matched_by_row as _affordance_matched_by,
)
from polylogue.insights.affordance_usage import (
    normalized_tool_name_for_row as _affordance_normalized_tool_name,
)
from polylogue.insights.archive import (
    ArchiveCoverageInsight,
    ArchiveDebtInsight,
    ArchiveEnrichmentProvenance,
    ArchiveInferenceProvenance,
    ArchiveInsightProvenance,
    CostRollupInsight,
    SessionCostInsight,
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
    SessionLatencyProfileInsight,
    SessionLatencyProfilePayload,
    SessionPhaseEvidencePayload,
    SessionPhaseInsight,
    SessionProfileInsight,
    SessionTagRollupInsight,
    SessionWorkEventInsight,
    ThreadInsight,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
)
from polylogue.insights.archive_models import ThreadMemberEvidencePayload, ThreadPayload
from polylogue.insights.archive_rollups import aggregate_cost_rollup_insights
from polylogue.insights.audit import InsightRigorAuditQuery, InsightRigorAuditReport, _audit_one
from polylogue.insights.confidence import ConfidenceBand
from polylogue.insights.confidence import from_score as confidence_from_score
from polylogue.insights.feedback import LearningCorrection, parse_correction_kind
from polylogue.insights.readiness import (
    InsightProviderCoverage,
    InsightReadinessEntry,
    InsightReadinessQuery,
    InsightReadinessReport,
    InsightReadinessVerdict,
    InsightStorageArtifact,
    InsightVersionCoverage,
    known_insight_readiness_names,
    normalize_insight_readiness_name,
)
from polylogue.insights.rigor import list_rigor_contracts
from polylogue.insights.run_projection import ContextSnapshot, ObservedEvent, ProjectedRun
from polylogue.insights.tool_usage import ToolUsageInsight, ToolUsageInsightQuery, build_tool_usage_insight
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.insights.session.records import SessionProfileRecord
from polylogue.storage.insights.session.runtime import (
    SESSION_INSIGHT_MATERIALIZATION_TYPES,
    SessionInsightStatusSnapshot,
)
from polylogue.storage.insights.session.status import session_insight_status_sync
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.search.query_support import normalize_fts5_query
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    archive_tier_spec,
    initialize_active_archive_root,
    initialize_archive_database,
)
from polylogue.storage.sqlite.archive_tiers.source_write import (
    write_source_raw_session,
    write_source_raw_session_blob_ref,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ASSERTION_DEFAULT_AUTHOR_KIND,
    ASSERTION_DEFAULT_AUTHOR_REF,
    ASSERTION_DEFAULT_CONTEXT_POLICY,
    ASSERTION_DEFAULT_STATUS,
    ASSERTION_DEFAULT_VISIBILITY,
    ArchiveAssertionEnvelope,
    ArchiveBlackboardNoteEnvelope,
    AssertionKind,
    assertion_id_for_annotation,
    assertion_id_for_correction,
    assertion_id_for_mark,
    assertion_id_for_recall_pack,
    assertion_id_for_saved_view,
    assertion_id_for_session_metadata,
    assertion_id_for_session_tag,
    assertion_id_for_workspace,
    correction_id_for,
    list_archive_blackboard_note_envelopes,
    list_assertions_by_kind,
    list_assertions_for_target,
    mark_assertion_status,
    read_assertion_envelope,
    upsert_annotation,
    upsert_blackboard_note,
    upsert_correction,
    upsert_mark,
    upsert_recall_pack,
    upsert_saved_view,
    upsert_session_metadata_assertion,
    upsert_session_tag_assertion,
    upsert_workspace,
)
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveInsightMaterialization,
    ArchiveSessionEnvelope,
    ArchiveSessionPhase,
    ArchiveSessionWorkEvent,
    read_archive_session_envelope,
    read_insight_materialization,
    read_session_phases,
    read_session_work_events,
    rebuild_archive_messages_fts,
    search_archive_blocks,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.connection_profile import (
    READ_CONNECTION_PRAGMA_STATEMENTS,
    WRITE_CONNECTION_PRAGMA_STATEMENTS,
)
from polylogue.storage.sqlite.queries.project_refs import expand_project_refs
from polylogue.storage.sqlite.queries.sessions_identity import session_id_prefix_bounds
from polylogue.storage.sqlite.queries.tool_usage import ToolUsageProviderCoverageRow, ToolUsageRow
from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync
from polylogue.types import SessionId


class IndexStatus(TypedDict):
    """block-FTS index existence and indexed-document count."""

    exists: bool
    count: int


@dataclass(frozen=True, slots=True)
class ArchiveSessionSummary:
    """archive summary projection over archive sessions."""

    session_id: str
    native_id: str
    origin: str
    provider: Provider
    title: str | None
    created_at: str | None
    updated_at: str | None
    message_count: int
    word_count: int
    tags: tuple[str, ...]
    session_kind: str = "standard"
    reported_duration_ms: int | None = None
    tool_use_count: int = 0
    thinking_count: int = 0
    paste_count: int = 0
    user_message_count: int = 0
    authored_user_message_count: int = 0
    assistant_message_count: int = 0
    system_message_count: int = 0
    tool_message_count: int = 0
    user_word_count: int = 0
    authored_user_word_count: int = 0
    assistant_word_count: int = 0
    working_directories: tuple[str, ...] = ()
    git_branch: str | None = None
    git_repository_url: str | None = None
    provider_project_ref: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveSessionSearchHit:
    """Search hit projection over archive block FTS."""

    rank: int
    session_id: str
    block_id: str
    message_id: str
    origin: str
    provider: Provider
    title: str | None
    snippet: str


@dataclass(frozen=True, slots=True)
class ArchiveMessageQueryRow:
    """Terminal query projection over archive messages."""

    message_id: str
    session_id: str
    origin: str
    title: str | None
    role: str
    message_type: str
    material_origin: str
    occurred_at_ms: int | None
    position: int
    word_count: int
    text: str


@dataclass(frozen=True, slots=True)
class ArchiveActionQueryRow:
    """Terminal query projection over normalized tool/action rows."""

    session_id: str
    message_id: str
    origin: str
    title: str | None
    tool_use_block_id: str
    tool_result_block_id: str | None
    tool_name: str | None
    semantic_type: str | None
    tool_command: str | None
    tool_path: str | None
    occurred_at_ms: int | None
    output_text: str | None


def _archive_action_query_row(row: sqlite3.Row) -> ArchiveActionQueryRow:
    return ArchiveActionQueryRow(
        session_id=str(row["session_id"]),
        message_id=str(row["message_id"]),
        origin=str(row["origin"]),
        title=str(row["title"]) if row["title"] is not None else None,
        tool_use_block_id=str(row["tool_use_block_id"]),
        tool_result_block_id=str(row["tool_result_block_id"]) if row["tool_result_block_id"] is not None else None,
        tool_name=str(row["tool_name"]) if row["tool_name"] is not None else None,
        semantic_type=str(row["semantic_type"]) if row["semantic_type"] is not None else None,
        tool_command=str(row["tool_command"]) if row["tool_command"] is not None else None,
        tool_path=str(row["tool_path"]) if row["tool_path"] is not None else None,
        occurred_at_ms=int(row["occurred_at_ms"]) if row["occurred_at_ms"] is not None else None,
        output_text=str(row["output_text"]) if row["output_text"] is not None else None,
    )


@dataclass(frozen=True, slots=True)
class ArchiveFileQueryRow:
    """Terminal query projection over affected file-path evidence."""

    session_id: str
    origin: str
    title: str | None
    path: str
    action_count: int
    first_message_id: str | None
    first_tool_use_block_id: str | None
    last_tool_use_block_id: str | None
    first_seen_ms: int | None
    last_seen_ms: int | None


@dataclass(frozen=True, slots=True)
class ArchiveBlockQueryRow:
    """Terminal query projection over archive content blocks."""

    block_id: str
    message_id: str
    session_id: str
    origin: str
    title: str | None
    block_type: str
    position: int
    text: str | None
    tool_name: str | None
    semantic_type: str | None
    tool_command: str | None
    tool_path: str | None


@dataclass(frozen=True, slots=True)
class ArchiveAssertionQueryRow:
    """Terminal query projection over user-tier assertion rows."""

    assertion_id: str
    target_ref: str
    scope_ref: str | None
    kind: str
    key: str | None
    body_text: str | None
    value: JSONValue
    author_ref: str
    author_kind: str
    status: str
    visibility: str
    evidence_refs: tuple[str, ...]
    staleness: JSONValue
    context_policy: JSONValue
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class ArchiveRunQueryRow:
    """Terminal query projection over materialized ``session_runs`` rows."""

    session_id: str
    origin: str
    title: str | None
    run: ProjectedRun


@dataclass(frozen=True, slots=True)
class ArchiveObservedEventQueryRow:
    """Terminal query projection over observed events from materialized or source rows."""

    session_id: str
    origin: str
    title: str | None
    event: ObservedEvent


@dataclass(frozen=True, slots=True)
class ArchiveContextSnapshotQueryRow:
    """Terminal query projection over materialized ``session_context_snapshots`` rows."""

    session_id: str
    origin: str
    title: str | None
    snapshot: ContextSnapshot


@dataclass(frozen=True, slots=True)
class ArchiveQueryUnitAggregateRow:
    """Aggregate count row over a terminal query-unit result set."""

    unit: str
    group_by: str | None
    group_key: str | None
    count: int


_OBSERVED_EVENT_RELATION_SQL = """
WITH tool_finished_base AS (
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
        result_position
    FROM tool_finished_base
),
materialized_observed_events AS (
    SELECT
        'materialized' AS row_source,
        e.event_ref,
        e.session_id,
        e.run_ref,
        e.position,
        e.source_updated_at,
        e.kind,
        e.summary,
        e.delivery_state,
        e.subject_ref,
        e.object_refs_json,
        e.evidence_refs_json,
        e.payload_json,
        e.search_text,
        NULL AS subject_message_id,
        NULL AS tool_use_position,
        NULL AS result_message_id,
        NULL AS result_position
    FROM session_observed_events e
    WHERE e.kind <> 'tool_finished'
        OR NOT EXISTS (
            SELECT 1
            FROM source_observed_events source
            WHERE source.session_id = e.session_id
        )
),
observed_events AS (
    SELECT * FROM source_observed_events
    UNION ALL
    SELECT * FROM materialized_observed_events
)
"""


def _observed_event_source_pushdown(predicate: QueryPredicate) -> tuple[str, list[object]]:
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
                add_clause("'tool_finished' = ?", ["tool_finished"], is_selective=False)
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


def _tuple_from_json_array(value: object) -> tuple[str, ...]:
    loaded = json.loads(str(value or "[]"))
    if not isinstance(loaded, list):
        return ()
    return tuple(str(item) for item in loaded if item is not None)


def _observed_event_from_query_row(row: sqlite3.Row) -> ObservedEvent:
    if str(row["row_source"]) == "materialized":
        return ObservedEvent.model_validate(json.loads(row["payload_json"]))
    payload = json.loads(str(row["payload_json"] or "{}"))
    return ObservedEvent(
        event_ref=ObjectRef.parse(str(row["event_ref"])),
        kind="tool_finished",
        run_ref=ObjectRef.parse(str(row["run_ref"])),
        summary=str(row["summary"]),
        delivery_state="observed",
        subject_ref=ObjectRef.parse(str(row["subject_ref"])),
        object_refs=tuple(ObjectRef.parse(ref) for ref in _tuple_from_json_array(row["object_refs_json"])),
        evidence_refs=tuple(EvidenceRef.parse(ref) for ref in _tuple_from_json_array(row["evidence_refs_json"])),
        tool_name=str(payload["tool_name"]) if payload.get("tool_name") is not None else None,
        tool_id=str(payload["tool_id"]) if payload.get("tool_id") is not None else None,
        command=str(payload["command"]) if payload.get("command") is not None else None,
        handler_kind=str(payload["handler_kind"]) if payload.get("handler_kind") is not None else None,
        status=str(payload["status"]) if payload.get("status") is not None else None,
    )


def _query_unit_order_direction(direction: Literal["asc", "desc"]) -> Literal["ASC", "DESC"]:
    """Return a closed SQL direction token for terminal row ordering."""

    return "DESC" if direction == "desc" else "ASC"


def _query_unit_aggregate_order(
    sort: Literal["count", "key"] | None,
    direction: Literal["asc", "desc"],
) -> str:
    """Return a closed SQL order clause for terminal aggregate rows."""

    sql_direction = _query_unit_order_direction(direction)
    if sort == "key":
        return f"group_key {sql_direction}, count DESC"
    return f"count {sql_direction}, group_key"


def _query_unit_group_expression(unit: str, row_alias: str, group_by: str | None) -> str:
    """Return the SQL expression for a supported terminal aggregate group."""

    if group_by is None:
        return "'all'"
    normalized = group_by.removeprefix("session.")
    session_fields = {
        "origin": "COALESCE(NULLIF(s.origin, ''), 'unknown')",
        "repo": "COALESCE(NULLIF(s.git_repository_url, ''), 'unknown')",
    }
    if group_by.startswith("session.") or group_by in {"origin", "repo"}:
        try:
            return session_fields[normalized]
        except KeyError as exc:
            raise ValueError(f"unsupported {unit} aggregate group field: {group_by}") from exc
    unit_fields = {
        "message": {
            "role": f"COALESCE(NULLIF({row_alias}.role, ''), 'unknown')",
            "type": f"COALESCE(NULLIF({row_alias}.message_type, ''), 'unknown')",
        },
        "action": {
            "tool": f"COALESCE(NULLIF({row_alias}.tool_name, ''), 'unknown')",
            "action": f"COALESCE(NULLIF({row_alias}.semantic_type, ''), 'unknown')",
            "type": f"COALESCE(NULLIF({row_alias}.semantic_type, ''), 'unknown')",
        },
        "file": {
            "path": f"COALESCE(NULLIF({row_alias}.path, ''), 'unknown')",
        },
        "block": {
            "type": f"COALESCE(NULLIF({row_alias}.block_type, ''), 'unknown')",
            "tool": f"COALESCE(NULLIF({row_alias}.tool_name, ''), 'unknown')",
            "action": f"COALESCE(NULLIF({row_alias}.semantic_type, ''), 'unknown')",
        },
        "assertion": {
            "kind": f"COALESCE(NULLIF({row_alias}.kind, ''), 'unknown')",
            "status": f"COALESCE(NULLIF({row_alias}.status, ''), '{ASSERTION_DEFAULT_STATUS}')",
            "visibility": f"COALESCE(NULLIF({row_alias}.visibility, ''), '{ASSERTION_DEFAULT_VISIBILITY}')",
            "author_kind": f"COALESCE(NULLIF({row_alias}.author_kind, ''), '{ASSERTION_DEFAULT_AUTHOR_KIND}')",
        },
        "observed-event": {
            "kind": f"COALESCE(NULLIF({row_alias}.kind, ''), 'unknown')",
            "delivery_state": f"COALESCE(NULLIF({row_alias}.delivery_state, ''), 'unknown')",
            "tool": f"COALESCE(NULLIF(json_extract({row_alias}.payload_json, '$.tool_name'), ''), 'unknown')",
            "handler": f"COALESCE(NULLIF(json_extract({row_alias}.payload_json, '$.handler_kind'), ''), 'unknown')",
            "status": f"COALESCE(NULLIF(json_extract({row_alias}.payload_json, '$.status'), ''), 'unknown')",
        },
    }
    try:
        return unit_fields[unit][group_by]
    except KeyError as exc:
        raise ValueError(f"unsupported {unit} aggregate group field: {group_by}") from exc


def _query_unit_group_uses_session(group_by: str | None) -> bool:
    """Return whether an aggregate group expression needs the sessions alias."""

    if group_by is None:
        return False
    return group_by.startswith("session.") or group_by in {"origin", "repo"}


def _session_filter_is_active(session_filters: Mapping[str, object] | None) -> bool:
    """Return whether normalized session filters contain a real constraint."""

    if not session_filters:
        return False
    for value in session_filters.values():
        if value is None or value is False or value == "":
            continue
        if isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) == 0:
            continue
        return True
    return False


class ArchiveStore:
    """Minimal archive-root façade for archive source/index/user tiers."""

    def __init__(
        self,
        archive_root: Path,
        *,
        initialize: bool = True,
        read_only: bool = False,
        read_timeout: float = 5.0,
    ) -> None:
        self.archive_root = archive_root
        self.source_db_path = archive_root / "source.db"
        self.index_db_path = archive_root / "index.db"
        self.embeddings_db_path = archive_root / "embeddings.db"
        self.user_db_path = archive_root / "user.db"
        self.ops_db_path = archive_root / "ops.db"
        self._read_only = read_only
        if initialize:
            initialize_active_archive_root(archive_root)
        if read_only:
            self._ensure_read_runtime_indexes()
        if read_only:
            self._conn = sqlite3.connect(f"file:{self.index_db_path}?mode=ro", uri=True, timeout=read_timeout)
            pragma_statements = READ_CONNECTION_PRAGMA_STATEMENTS
        else:
            self._conn = sqlite3.connect(self.index_db_path)
            pragma_statements = WRITE_CONNECTION_PRAGMA_STATEMENTS
        self._conn.row_factory = sqlite3.Row
        # Apply the canonical connection profile rather than relying on sqlite3
        # defaults. Daemon read routes may opt into a shorter busy timeout so
        # write-heavy catch-up surfaces as a typed degraded response instead of
        # a client-side route timeout.
        for statement in pragma_statements:
            self._conn.execute(statement)
        if read_only:
            self._conn.execute(f"PRAGMA busy_timeout = {max(0, int(read_timeout * 1000))}")
        self._user_tier_attached = False
        self._tags_relation = "session_tags"
        # Lazily opened persistent write connection to source.db. Keeping it open
        # across writes lets a bulk caller batch raw-session commits (it is the
        # twin of the index.db commit batched on self._conn) instead of paying a
        # source-tier fsync per session. Opened on first raw write only.
        self._source_conn: sqlite3.Connection | None = None
        self._attach_user_tier_if_present()

    @classmethod
    def open_existing(cls, archive_root: Path, *, read_only: bool = True, read_timeout: float = 5.0) -> ArchiveStore:
        """Open archive tier files.

        Read-only opens never bootstrap missing tiers; read/status surfaces must
        not create an empty archive and then report it as usable. Writers opt
        into bootstrap by passing ``read_only=False``.
        """
        initialize = not read_only
        return cls(archive_root, initialize=initialize, read_only=read_only, read_timeout=read_timeout)

    @staticmethod
    def _needs_tier_bootstrap(archive_root: Path) -> bool:
        return any(
            not (archive_root / filename).exists()
            for filename in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db")
        )

    def _ensure_read_runtime_indexes(self) -> None:
        """Best-effort performance-index ensure before opening the read connection."""
        if not self.index_db_path.exists():
            return
        try:
            with sqlite3.connect(self.index_db_path) as conn:
                current_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                if current_version != archive_tier_spec(ArchiveTier.INDEX).version:
                    return
                for statement in WRITE_CONNECTION_PRAGMA_STATEMENTS:
                    conn.execute(statement)
                ensure_runtime_indexes_sync(conn)
                conn.commit()
        except sqlite3.Error:
            return

    def _ensure_source_conn(self) -> sqlite3.Connection:
        """Return the persistent source.db write connection, opening it lazily."""
        if self._source_conn is None:
            conn = sqlite3.connect(self.source_db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            self._source_conn = conn
        return self._source_conn

    def commit(self) -> None:
        """Commit the index.db and (if open) source.db write connections.

        Bulk callers that pass ``manage_transaction=False`` to the write methods
        own commit cadence; this flushes both tiers' in-flight batch.
        """
        self._conn.commit()
        if self._source_conn is not None:
            self._source_conn.commit()

    def rollback(self) -> None:
        """Roll back the index.db and (if open) source.db write connections.

        Used by a bulk caller to discard an uncommitted, half-applied batch when
        a write raises, before propagating the error.
        """
        self._conn.rollback()
        if self._source_conn is not None:
            self._source_conn.rollback()

    def close(self) -> None:
        if self._source_conn is not None:
            self._source_conn.close()
            self._source_conn = None
        self._conn.close()

    def write_parsed(self, session: ParsedSession, *, content_hash: str | None = None) -> str:
        """Write a parsed session to index.db."""
        return write_parsed_session_to_archive(self._conn, session, content_hash=content_hash)

    def write_raw_and_parsed(
        self,
        session: ParsedSession,
        *,
        payload: bytes,
        source_path: str,
        acquired_at_ms: int,
        source_index: int = 0,
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "append",
        manage_transaction: bool = True,
    ) -> tuple[str, str]:
        """Write raw acquisition bytes and the parsed session they produced.

        With the default ``manage_transaction=True`` each tier write commits on
        its own. A bulk caller passes ``manage_transaction=False`` and owns
        commit cadence via :meth:`commit` / :meth:`rollback` so the source.db and
        index.db writes are batched into shared transactions.
        """

        def add_timing(name: str, started_at: float) -> None:
            if stage_timings_s is not None:
                key = f"{stage_timing_prefix}.{name}"
                stage_timings_s[key] = stage_timings_s.get(key, 0.0) + (time.perf_counter() - started_at)

        t0 = time.perf_counter()
        source_conn = self._ensure_source_conn()
        add_timing("source_connect", t0)
        t0 = time.perf_counter()
        raw_id = write_source_raw_session(
            source_conn,
            origin=origin_from_provider(session.source_name),
            source_path=source_path,
            source_index=source_index,
            native_id=session.provider_session_id,
            payload=payload,
            acquired_at_ms=acquired_at_ms,
            manage_transaction=manage_transaction,
        )
        add_timing("source_raw_write", t0)
        t0 = time.perf_counter()
        session_id = write_parsed_session_to_archive(
            self._conn,
            session,
            raw_id=raw_id,
            merge_append=source_index < 0,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=manage_transaction,
        )
        add_timing("index_parsed_write", t0)
        return raw_id, session_id

    def write_raw_blob_and_parsed(
        self,
        session: ParsedSession,
        *,
        blob_hash_hex: str,
        blob_size: int,
        source_path: str,
        acquired_at_ms: int,
        source_index: int = 0,
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "full",
        manage_transaction: bool = True,
    ) -> tuple[str, str]:
        """Write parsed session metadata for an already-materialized raw blob.

        See :meth:`write_raw_and_parsed` for the ``manage_transaction`` contract.
        """

        def add_timing(name: str, started_at: float) -> None:
            if stage_timings_s is not None:
                key = f"{stage_timing_prefix}.{name}"
                stage_timings_s[key] = stage_timings_s.get(key, 0.0) + (time.perf_counter() - started_at)

        t0 = time.perf_counter()
        source_conn = self._ensure_source_conn()
        add_timing("source_connect", t0)
        t0 = time.perf_counter()
        raw_id = write_source_raw_session_blob_ref(
            source_conn,
            origin=origin_from_provider(session.source_name),
            source_path=source_path,
            source_index=source_index,
            native_id=session.provider_session_id,
            blob_hash=bytes.fromhex(blob_hash_hex),
            blob_size=blob_size,
            acquired_at_ms=acquired_at_ms,
            manage_transaction=manage_transaction,
        )
        add_timing("source_raw_blob_ref_write", t0)
        t0 = time.perf_counter()
        session_id = write_parsed_session_to_archive(
            self._conn,
            session,
            raw_id=raw_id,
            merge_append=source_index < 0,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=manage_transaction,
        )
        add_timing("index_parsed_write", t0)
        return raw_id, session_id

    def read_session(self, session_id: str) -> ArchiveSessionEnvelope:
        """Read a session envelope from index.db."""
        return read_archive_session_envelope(self._conn, session_id)

    def get_session_tree(self, session_id: str) -> list[ArchiveSessionEnvelope]:
        """Return the rooted archive session tree containing ``session_id``."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return []
        root_session_id = self._root_session_id_for_tree(resolved_session_id)
        rows = self._conn.execute(
            """
            SELECT session_id
            FROM sessions
            WHERE session_id = ?
               OR root_session_id = ?
            ORDER BY
                CASE WHEN session_id = ? THEN 0 ELSE 1 END,
                COALESCE(sort_key_ms, created_at_ms, updated_at_ms, 0),
                session_id
            """,
            (root_session_id, root_session_id, root_session_id),
        ).fetchall()
        return [read_archive_session_envelope(self._conn, str(row["session_id"])) for row in rows]

    def _root_session_id_for_tree(self, session_id: str) -> str:
        row = self._conn.execute(
            "SELECT root_session_id, parent_session_id FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        if row["root_session_id"]:
            return str(row["root_session_id"])

        current_id = session_id
        seen: set[str] = set()
        while current_id not in seen:
            seen.add(current_id)
            parent_row = self._conn.execute(
                "SELECT parent_session_id FROM sessions WHERE session_id = ?",
                (current_id,),
            ).fetchone()
            if parent_row is None or not parent_row["parent_session_id"]:
                return current_id
            current_id = str(parent_row["parent_session_id"])
        return session_id

    def raw_artifacts_for_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, object]], int]:
        """Return raw acquisition surface rows for one archive session."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return [], 0
        raw_row = self._conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            (resolved_session_id,),
        ).fetchone()
        if raw_row is None or raw_row["raw_id"] is None or not self.source_db_path.exists():
            return [], 0
        raw_id = str(raw_row["raw_id"])
        source_conn = sqlite3.connect(f"file:{self.source_db_path}?mode=ro", uri=True)
        source_conn.row_factory = sqlite3.Row
        try:
            total = int(
                source_conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
            )
            rows = source_conn.execute(
                """
                SELECT raw_id, origin, source_path, blob_size, acquired_at_ms,
                       parsed_at_ms, validation_status
                FROM raw_sessions
                WHERE raw_id = ?
                ORDER BY acquired_at_ms DESC, raw_id
                LIMIT ? OFFSET ?
                """,
                (raw_id, max(limit, 0), max(offset, 0)),
            ).fetchall()
        finally:
            source_conn.close()
        return [
            {
                "raw_id": str(row["raw_id"]),
                "source_name": _provider_for_origin(str(row["origin"])).value,
                "source_path": str(row["source_path"]),
                "blob_size": int(row["blob_size"] or 0),
                "acquired_at": _iso_from_ms(row["acquired_at_ms"]),
                "parsed_at": _iso_from_ms(row["parsed_at_ms"]),
                "validation_status": row["validation_status"],
            }
            for row in rows
        ], total

    def get_session_work_event_insights(self, session_id: str) -> list[SessionWorkEventInsight]:
        """Read archive work-event insights for one session."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return []
        return self.list_session_work_event_insights(session_id=resolved_session_id)

    def list_session_work_event_insights(
        self,
        *,
        session_id: str | None = None,
        provider: str | None = None,
        heuristic_label: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionWorkEventInsight]:
        """List archive work-event insights with the public insight contract."""
        where: list[str] = []
        params: list[object] = []
        if session_id is not None:
            where.append("we.session_id = ?")
            params.append(self.resolve_session_id(session_id))
        origin = _origin_for_provider_value(provider)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if heuristic_label is not None:
            where.append("we.work_event_type = ?")
            params.append(heuristic_label)
        if since_ms is not None:
            where.append("COALESCE(we.started_at_ms, s.sort_key_ms) >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("COALESCE(we.started_at_ms, s.sort_key_ms) <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT we.session_id, we.position
            FROM session_work_events we
            JOIN sessions s ON s.session_id = we.session_id
            {clause}
            ORDER BY COALESCE(we.started_at_ms, s.sort_key_ms) DESC, we.session_id, we.position
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        events_by_session = {str(row["session_id"]) for row in rows}
        indexed: dict[tuple[str, int], SessionWorkEventInsight] = {}
        for event_session_id in events_by_session:
            materialization = _read_archive_materialization(self._conn, "work_events", event_session_id)
            session_origin = _session_origin(self._conn, event_session_id)
            for event in read_session_work_events(self._conn, session_id=event_session_id).values():
                if heuristic_label is None or event.work_event_type == heuristic_label:
                    indexed[(event.session_id, event.position)] = _work_event_insight_from_archive_row(
                        event,
                        origin=session_origin,
                        materialization=materialization,
                    )
        return [indexed[(str(row["session_id"]), int(row["position"]))] for row in rows]

    def get_session_phase_insights(self, session_id: str) -> list[SessionPhaseInsight]:
        """Read archive phase insights for one session."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return []
        return self.list_session_phase_insights(session_id=resolved_session_id)

    def list_session_phase_insights(
        self,
        *,
        session_id: str | None = None,
        provider: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionPhaseInsight]:
        """List archive phase insights with the public insight contract."""
        where: list[str] = []
        params: list[object] = []
        if session_id is not None:
            where.append("sp.session_id = ?")
            params.append(self.resolve_session_id(session_id))
        origin = _origin_for_provider_value(provider)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("COALESCE(sp.started_at_ms, s.sort_key_ms) >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("COALESCE(sp.started_at_ms, s.sort_key_ms) <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT sp.session_id, sp.position
            FROM session_phases sp
            JOIN sessions s ON s.session_id = sp.session_id
            {clause}
            ORDER BY COALESCE(sp.started_at_ms, s.sort_key_ms) DESC, sp.session_id, sp.position
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        phases_by_session = {str(row["session_id"]) for row in rows}
        indexed: dict[tuple[str, int], SessionPhaseInsight] = {}
        for phase_session_id in phases_by_session:
            materialization = _read_archive_materialization(self._conn, "phases", phase_session_id)
            session_origin = _session_origin(self._conn, phase_session_id)
            for phase in read_session_phases(self._conn, session_id=phase_session_id).values():
                indexed[(phase.session_id, phase.position)] = _phase_insight_from_archive_row(
                    phase,
                    origin=session_origin,
                    materialization=materialization,
                )
        return [indexed[(str(row["session_id"]), int(row["position"]))] for row in rows]

    def get_thread_insight(self, thread_id: str) -> ThreadInsight | None:
        """Read one archive thread projection as a public thread insight."""
        row = self._conn.execute(
            "SELECT thread_id FROM threads WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        return self._thread_insight_from_id(str(row["thread_id"]))

    def list_thread_insights(
        self,
        *,
        query: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[ThreadInsight]:
        """List threads as public thread insights."""
        where: list[str] = []
        params: list[object] = []
        if query:
            like = f"%{query.strip().lower()}%"
            where.append(
                """
                (
                    lower(t.thread_id) LIKE ?
                    OR EXISTS (
                        SELECT 1
                        FROM thread_sessions qts
                        JOIN sessions qs ON qs.session_id = qts.session_id
                        WHERE qts.thread_id = t.thread_id
                          AND (
                            lower(qs.session_id) LIKE ?
                            OR lower(COALESCE(qs.title, '')) LIKE ?
                            OR lower(COALESCE(qs.git_repository_url, '')) LIKE ?
                            OR lower(COALESCE(qs.git_branch, '')) LIKE ?
                          )
                    )
                )
                """.strip()
            )
            params.extend([like, like, like, like, like])
        if since_ms is not None:
            where.append("t.created_at_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("t.created_at_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT t.thread_id
            FROM threads t
            {clause}
            ORDER BY t.created_at_ms DESC, t.thread_id
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [insight for row in rows if (insight := self._thread_insight_from_id(str(row["thread_id"]))) is not None]

    def _thread_insight_from_id(self, thread_id: str) -> ThreadInsight | None:
        row = self._conn.execute(
            """
            SELECT thread_id, created_at_ms, session_count
            FROM threads
            WHERE thread_id = ?
            """,
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        session_rows = self._conn.execute(
            """
            SELECT s.session_id, s.parent_session_id, s.origin, s.title,
                   s.message_count, s.word_count, s.tool_use_count,
                   s.created_at_ms, s.updated_at_ms, s.git_repository_url,
                   s.git_branch, sp.first_message_at, sp.last_message_at,
                   sp.total_cost_usd AS profile_total_cost_usd
            FROM thread_sessions ts
            JOIN sessions s ON s.session_id = ts.session_id
            LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
            WHERE ts.thread_id = ?
            ORDER BY ts.position, s.sort_key_ms, s.session_id
            """,
            (thread_id,),
        ).fetchall()
        session_ids = tuple(str(session["session_id"]) for session in session_rows)
        origin_breakdown: dict[str, int] = {}
        for session in session_rows:
            session_origin = str(session["origin"])
            origin_breakdown[session_origin] = origin_breakdown.get(session_origin, 0) + 1
        start_ms = min(
            (
                timestamp_ms
                for session in session_rows
                if (
                    timestamp_ms := _profile_or_session_timestamp_ms(
                        session,
                        profile_column="first_message_at",
                        session_column="created_at_ms",
                    )
                )
                is not None
            ),
            default=None,
        )
        end_ms = max(
            (
                timestamp_ms
                for session in session_rows
                if (
                    timestamp_ms := _profile_or_session_timestamp_ms(
                        session,
                        profile_column="last_message_at",
                        session_column="updated_at_ms",
                    )
                )
                is not None
            ),
            default=None,
        )
        dominant_repo = _dominant_repo(session_rows)
        member_evidence = tuple(
            ThreadMemberEvidencePayload(
                session_id=str(session["session_id"]),
                parent_id=str(session["parent_session_id"]) if session["parent_session_id"] else None,
                role=_archive_thread_member_role(session, str(row["thread_id"])),
                depth=_thread_member_depth(session_rows, str(session["session_id"])),
                confidence=1.0,
                support_signals=_archive_thread_member_support_signals(session),
                evidence=_archive_thread_member_evidence(session, str(row["thread_id"]), index),
            )
            for index, session in enumerate(session_rows)
        )
        lineage_signals: tuple[str, ...] = ("archive_threads", "archive_thread_sessions")
        if any(session["parent_session_id"] is not None for session in session_rows):
            lineage_signals = (*lineage_signals, "explicit_lineage")
        payload = ThreadPayload(
            start_time=_iso_from_ms(start_ms),
            end_time=_iso_from_ms(end_ms),
            dominant_repo=dominant_repo,
            session_ids=session_ids,
            session_count=len(session_ids),
            depth=max((member.depth for member in member_evidence), default=0),
            branch_count=sum(1 for session in session_rows if session["parent_session_id"] is not None),
            total_messages=sum(int(session["message_count"] or 0) for session in session_rows),
            total_cost_usd=sum(float(session["profile_total_cost_usd"] or 0.0) for session in session_rows),
            wall_duration_ms=max(end_ms - start_ms, 0) if start_ms is not None and end_ms is not None else 0,
            origin_breakdown=origin_breakdown,
            confidence=1.0 if session_rows else 0.0,
            support_level=ConfidenceBand.STRONG if len(session_rows) > 1 else ConfidenceBand.MODERATE,
            support_signals=lineage_signals,
            member_evidence=member_evidence,
        )
        materialization = _read_archive_materialization(self._conn, "thread", thread_id)
        return ThreadInsight(
            thread_id=str(row["thread_id"]),
            root_id=str(row["thread_id"]),
            dominant_repo=dominant_repo,
            provenance=_archive_provenance(materialization),
            thread=payload,
        )

    def list_session_cost_insights(
        self,
        *,
        session_id: str | None = None,
        provider: str | None = None,
        status: str | None = None,
        model: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionCostInsight]:
        """List archive session cost insights from sessions plus session_profiles."""
        if model is not None:
            return []
        where: list[str] = []
        params: list[object] = []
        if session_id is not None:
            try:
                resolved_session_id = self.resolve_session_id(session_id)
            except KeyError:
                # Unknown session id: no cost insight exists. Returning [] lets
                # the daemon cost endpoint run its existence check and answer
                # 404 instead of surfacing this as an opaque 500.
                return []
            where.append("s.session_id = ?")
            params.append(resolved_session_id)
        origin = _origin_for_provider_value(provider)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT s.session_id, s.origin, s.title, s.created_at_ms, s.updated_at_ms,
                   s.sort_key_ms, sp.cost_credits, sp.cost_usd, sp.cost_is_estimated,
                   sp.cost_provenance,
                   (
                       SELECT smu.model_name
                       FROM session_model_usage smu
                       WHERE smu.session_id = s.session_id
                       ORDER BY smu.input_tokens + smu.output_tokens DESC, smu.model_name
                       LIMIT 1
                   ) AS model_name
            FROM sessions s
            LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
            {clause}
            ORDER BY s.sort_key_ms DESC, s.session_id
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        insights = [_session_cost_insight_from_archive_row(self._conn, row) for row in rows]
        if status is not None:
            insights = [insight for insight in insights if insight.estimate.status == status]
        return insights

    def list_cost_rollup_insights(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[CostRollupInsight]:
        """Aggregate archive session-cost insights into public cost rollups."""
        session_costs = self.list_session_cost_insights(
            provider=provider,
            model=model,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=None,
            offset=0,
        )
        rollups = aggregate_cost_rollup_insights(session_costs, materialized_at=datetime.now(UTC).isoformat())
        if offset:
            rollups = rollups[offset:]
        if limit is not None:
            rollups = rollups[: max(int(limit), 0)]
        return rollups

    def list_archive_debt_insights(
        self,
        *,
        category: str | None = None,
        only_actionable: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ArchiveDebtInsight]:
        """Report consistency debt."""
        insights = [
            _archive_messages_fts_debt(self._conn),
            _archive_profile_rows_debt(self._conn),
            _archive_profile_counts_debt(self._conn),
            _archive_materialization_debt(self._conn),
            _archive_source_raw_link_debt(self._conn, self.source_db_path),
            _archive_user_overlay_debt(self._conn, self.user_db_path),
        ]
        insights.sort(key=lambda insight: (insight.category, insight.debt_name))
        if category is not None:
            insights = [insight for insight in insights if insight.category == category]
        if only_actionable:
            insights = [insight for insight in insights if not insight.healthy]
        if offset:
            insights = insights[offset:]
        if limit is not None:
            insights = insights[: max(int(limit), 0)]
        return insights

    def get_session_latency_profile_insight(self, session_id: str) -> SessionLatencyProfileInsight | None:
        """Project one latency profile from timestamped messages."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return None
        row = self._conn.execute(
            """
            SELECT session_id, origin, title, sort_key_ms
            FROM sessions
            WHERE session_id = ?
            """,
            (resolved_session_id,),
        ).fetchone()
        return None if row is None else _session_latency_profile_from_archive_row(self._conn, row)

    def list_session_latency_profile_insights(
        self,
        *,
        session_id: str | None = None,
        provider: str | None = None,
        only_stuck: bool = False,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionLatencyProfileInsight]:
        """Project archive latency profiles from sessions plus timestamped messages."""
        where: list[str] = []
        params: list[object] = []
        if session_id is not None:
            where.append("s.session_id = ?")
            params.append(self.resolve_session_id(session_id))
        origin = _origin_for_provider_value(provider)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT s.session_id, s.origin, s.title, s.sort_key_ms
            FROM sessions s
            {clause}
            ORDER BY s.sort_key_ms DESC, s.session_id
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        insights = [_session_latency_profile_from_archive_row(self._conn, row) for row in rows]
        if only_stuck:
            insights = [insight for insight in insights if insight.latency.stuck_tool_count > 0]
        return insights

    def find_stuck_session_latency_profile_insights(
        self,
        *,
        provider: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
    ) -> list[SessionLatencyProfileInsight]:
        """Return archive latency profiles with stuck tools.

        currently lacks session event start/end pairs, so stuck
        tool detection remains conservative and this returns only profiles
        whose projected stuck count is non-zero.
        """
        return self.list_session_latency_profile_insights(
            provider=provider,
            only_stuck=True,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            offset=0,
        )

    def _fetch_session_profile_row(self, session_id: str) -> sqlite3.Row | None:
        """Resolve *session_id* and fetch its joined session/profile row, or None."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return None
        rows = self._conn.execute(
            """
            SELECT s.session_id, s.origin, s.root_session_id, s.title, s.created_at_ms, s.updated_at_ms,
                   s.message_count, s.word_count, s.tool_use_count, s.thinking_count,
                   sp.workflow_shape, sp.workflow_shape_confidence, sp.terminal_state,
                   sp.terminal_state_confidence, sp.duration_ms, sp.substantive_count,
                   sp.attachment_count, sp.work_event_count, sp.phase_count,
                   sp.tool_calls_per_minute, sp.cost_usd, sp.cost_is_estimated,
                   sp.cost_provenance,
                   sp.total_cost_usd, sp.total_duration_ms,
                   sp.evidence_payload_json, sp.inference_payload_json, sp.enrichment_payload_json
            FROM session_profiles sp
            JOIN sessions s ON s.session_id = sp.session_id
            WHERE sp.session_id = ?
            """,
            (resolved_session_id,),
        ).fetchall()
        return rows[0] if rows else None

    def get_session_profile_insight(self, session_id: str, *, tier: str = "merged") -> SessionProfileInsight | None:
        """Read one archive session profile insight."""
        row = self._fetch_session_profile_row(session_id)
        if row is None:
            return None
        return _session_profile_insight_from_archive_row(self._conn, row, tier=tier)

    def get_session_profile_record(self, session_id: str) -> SessionProfileRecord | None:
        """Read one archive session profile as a domain :class:`SessionProfileRecord`.

        Mirrors :meth:`get_session_profile_insight` but rehydrates the full
        record needed by ``hydrate_session_profile`` (domain ``SessionProfile``)
        and the provenance-based staleness check. The materialization HWM
        provenance is pulled from ``read_insight_materialization`` so the
        downstream ``is_stale`` comparison is grounded in the same source the
        daemon's ``/insights`` profile panel consumes.

        Returns ``None`` when the session id does not resolve or has no
        materialized profile.
        """
        row = self._fetch_session_profile_row(session_id)
        if row is None:
            return None
        return _session_profile_record_from_archive_row(self._conn, row)

    def list_session_profile_insights(
        self,
        *,
        provider: str | None = None,
        workflow_shape: str | None = None,
        terminal_state: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        tier: str = "merged",
        limit: int | None = 50,
        offset: int = 0,
        min_wallclock_seconds: float | None = None,
        max_wallclock_seconds: float | None = None,
        sort: str | None = None,
    ) -> list[SessionProfileInsight]:
        """List archive session profile insights.

        ``min_wallclock_seconds`` / ``max_wallclock_seconds`` filter on the
        session's message-timestamp span (last minus first message), and
        ``sort='wallclock'`` orders by that span descending.
        """
        # Wallclock span = newest minus oldest message timestamp for the session.
        wall_expr = (
            "(SELECT MAX(m.occurred_at_ms) - MIN(m.occurred_at_ms) "
            "FROM messages m WHERE m.session_id = s.session_id AND m.occurred_at_ms IS NOT NULL)"
        )
        where: list[str] = []
        params: list[object] = []
        origin = _origin_for_provider_value(provider)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if workflow_shape is not None:
            where.append("sp.workflow_shape = ?")
            params.append(workflow_shape)
        if terminal_state is not None:
            where.append("sp.terminal_state = ?")
            params.append(terminal_state)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        if min_wallclock_seconds is not None:
            where.append(f"COALESCE({wall_expr}, 0) >= ?")
            params.append(int(min_wallclock_seconds * 1000))
        if max_wallclock_seconds is not None:
            where.append(f"COALESCE({wall_expr}, 0) <= ?")
            params.append(int(max_wallclock_seconds * 1000))
        clause = "WHERE " + " AND ".join(where) if where else ""
        order_by = f"{wall_expr} DESC, s.session_id" if sort == "wallclock" else "s.sort_key_ms DESC, s.session_id"
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT s.session_id, s.origin, s.root_session_id, s.title, s.created_at_ms, s.updated_at_ms,
                   s.message_count, s.word_count, s.tool_use_count, s.thinking_count,
                   sp.workflow_shape, sp.workflow_shape_confidence, sp.terminal_state,
                   sp.terminal_state_confidence, sp.duration_ms, sp.substantive_count,
                   sp.attachment_count, sp.work_event_count, sp.phase_count,
                   sp.tool_calls_per_minute, sp.cost_usd, sp.cost_is_estimated,
                   sp.cost_provenance,
                   sp.total_cost_usd, sp.total_duration_ms,
                   sp.evidence_payload_json, sp.inference_payload_json, sp.enrichment_payload_json
            FROM session_profiles sp
            JOIN sessions s ON s.session_id = sp.session_id
            {clause}
            ORDER BY {order_by}
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [_session_profile_insight_from_archive_row(self._conn, row, tier=tier) for row in rows]

    def read_summary(self, session_id: str) -> ArchiveSessionSummary:
        """Read one session summary by exact session id."""
        row = self._conn.execute(
            f"""
            SELECT s.session_id, s.native_id, s.origin, s.title, s.created_at_ms, s.updated_at_ms,
                   s.session_kind,
                   s.message_count, s.word_count, s.reported_duration_ms,
                   s.tool_use_count, s.thinking_count, s.paste_count,
                   s.user_message_count, s.authored_user_message_count,
                   s.assistant_message_count, s.system_message_count,
                   s.tool_message_count, s.user_word_count, s.authored_user_word_count,
                   s.assistant_word_count,
                   s.git_branch, s.git_repository_url, s.provider_project_ref,
                   COALESCE(
                       (
                           SELECT json_group_array(swd.path)
                           FROM session_working_dirs swd
                           WHERE swd.session_id = s.session_id
                           ORDER BY swd.position, swd.path
                       ),
                       '[]'
                   ) AS working_directories_json,
                   COALESCE(
                       json_group_array(st.tag) FILTER (WHERE st.tag IS NOT NULL),
                       '[]'
                   ) AS tags_json
            FROM sessions s
            LEFT JOIN {self._tags_relation} st
              ON st.session_id = s.session_id
             AND st.tag_source = 'user'
            WHERE s.session_id = ?
            GROUP BY s.session_id
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return _summary_from_row(row)

    def resolve_session_id(self, token: str) -> str:
        """Resolve an exact or prefix session id token."""
        exact = self._conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?",
            (token,),
        ).fetchone()
        if exact is not None:
            return str(exact["session_id"])
        if ":" in token:
            provider_token, native_id = token.split(":", 1)
            origin_id = f"{origin_from_provider(Provider.from_string(provider_token)).value}:{native_id}"
            exact = self._conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (origin_id,),
            ).fetchone()
            if exact is not None:
                return str(exact["session_id"])
        lower_bound, upper_bound = session_id_prefix_bounds(token)
        where = "session_id >= ?"
        params: list[str] = [lower_bound]
        if upper_bound is not None:
            where = f"{where} AND session_id < ?"
            params.append(upper_bound)
        rows = self._conn.execute(
            f"""
            SELECT session_id
            FROM sessions
            WHERE {where}
            ORDER BY session_id
            LIMIT 2
            """,
            tuple(params),
        ).fetchall()
        if not rows:
            # Suffix fallback: a bare native id (e.g. the UUID that appears as
            # the session's source filename, ``1944721d-...``) resolves to the
            # stored ``<origin>:<native_id>``. Provider native ids are globally
            # unique, so a single suffix match is unambiguous; multiple matches
            # raise just like the prefix path rather than guessing.
            if ":" not in token:
                like_token = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                suffix_rows = self._conn.execute(
                    """
                    SELECT session_id
                    FROM sessions
                    WHERE session_id LIKE '%:' || ? ESCAPE '\\'
                    ORDER BY session_id
                    LIMIT 2
                    """,
                    (like_token,),
                ).fetchall()
                if len(suffix_rows) == 1:
                    return str(suffix_rows[0]["session_id"])
                if len(suffix_rows) > 1:
                    raise ValueError(f"session id suffix {token!r} is ambiguous")
            raise KeyError(token)
        if len(rows) > 1:
            raise ValueError(f"session id prefix {token!r} is ambiguous")
        return str(rows[0]["session_id"])

    def search_blocks(self, query: str) -> list[str]:
        """Search indexed block text and return block ids."""
        return search_archive_blocks(self._conn, query)

    def rebuild_index(self) -> int:
        """Rebuild the block FTS index from index.db blocks."""
        rebuilt_rows = rebuild_archive_messages_fts(self._conn)
        self._conn.commit()
        return rebuilt_rows

    def index_status(self) -> IndexStatus:
        """Return ``{exists, count}`` for the archive block FTS index.

        The block FTS index (``messages_fts`` over ``blocks``) is trigger-maintained, so a
        missing table means it was never built and the count is the
        indexed-block total.
        """
        if not _table_exists(self._conn, "messages_fts"):
            return IndexStatus(exists=False, count=0)
        return IndexStatus(exists=True, count=_count_scalar(self._conn, "SELECT COUNT(*) FROM messages_fts"))

    def add_user_tags(self, session_ids: tuple[str, ...], tags: tuple[str, ...]) -> int:
        """Add user tag assertions to archive user.db and return changed count."""
        user_db_path = self.user_db_path
        initialize_archive_database(user_db_path, ArchiveTier.USER)
        changed = 0
        user_conn = sqlite3.connect(user_db_path)
        user_conn.row_factory = sqlite3.Row
        try:
            with user_conn:
                for session_id in tuple(
                    dict.fromkeys(self.resolve_session_id(session_id) for session_id in session_ids)
                ):
                    for tag in tags:
                        normalized_tag = tag.strip().lower()
                        if not normalized_tag:
                            raise ValueError("tag cannot be empty")
                        existing = read_assertion_envelope(
                            user_conn,
                            assertion_id_for_session_tag(session_id, normalized_tag, "user"),
                        )
                        if existing is None or existing.status == "deleted":
                            changed += 1
                        upsert_session_tag_assertion(
                            user_conn,
                            session_id=session_id,
                            tag=normalized_tag,
                            tag_source="user",
                            method="cli",
                            evidence={"source": "archive_query"},
                        )
        finally:
            user_conn.close()
        self._attach_user_tier_if_present()
        return changed

    def remove_user_tags(self, session_ids: tuple[str, ...], tags: tuple[str, ...]) -> int:
        """Mark user tag assertions deleted and return deleted row count."""
        resolved_session_ids = tuple(dict.fromkeys(self.resolve_session_id(session_id) for session_id in session_ids))
        if not resolved_session_ids or not self.user_db_path.exists():
            return 0
        removed = 0
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                for session_id in resolved_session_ids:
                    for tag in tags:
                        normalized_tag = tag.strip().lower()
                        if not normalized_tag:
                            raise ValueError("tag cannot be empty")
                        assertion_id = assertion_id_for_session_tag(session_id, normalized_tag, "user")
                        assertion = read_assertion_envelope(user_conn, assertion_id)
                        if assertion is None or assertion.status == "deleted":
                            continue
                        if mark_assertion_status(user_conn, assertion_id, "deleted"):
                            removed += 1
        finally:
            user_conn.close()
        self._attach_user_tier_if_present()
        return removed

    def list_user_tags(self, *, origin: str | None = None) -> dict[str, int]:
        """Return user tag counts over archive sessions."""
        where = "WHERE st.tag_source = 'user'"
        params: list[object] = []
        if origin is not None:
            where += " AND s.origin = ?"
            params.append(origin)
        rows = self._conn.execute(
            f"""
            SELECT st.tag, COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN {self._tags_relation} st
              ON st.session_id = s.session_id
            {where}
            GROUP BY st.tag
            ORDER BY count DESC, st.tag
            """,
            tuple(params),
        ).fetchall()
        return {str(row["tag"]): int(row["count"] or 0) for row in rows}

    def list_session_tag_rollup_insights(
        self,
        *,
        provider: str | None = None,
        query: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[SessionTagRollupInsight]:
        """Aggregate archive session tags into public tag-rollup insights."""
        where: list[str] = []
        params: list[object] = []
        origin = _origin_for_provider_value(provider)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if query:
            where.append("lower(st.tag) LIKE ?")
            params.append(f"%{query.strip().lower()}%")
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        filter_params = tuple(params)
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT st.tag,
                   COUNT(DISTINCT s.session_id) AS session_count,
                   COUNT(DISTINCT COALESCE(s.root_session_id, s.session_id)) AS logical_session_count,
                   COUNT(DISTINCT CASE WHEN st.tag_source = 'user' THEN s.session_id END) AS explicit_count,
                   COUNT(DISTINCT CASE WHEN st.tag_source = 'auto' THEN s.session_id END) AS auto_count,
                   MAX(s.sort_key_ms) AS source_sort_key_ms
            FROM sessions s
            JOIN {self._tags_relation} st ON st.session_id = s.session_id
            {clause}
            GROUP BY st.tag
            ORDER BY session_count DESC, st.tag
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [
            SessionTagRollupInsight(
                tag=str(row["tag"]),
                session_count=int(row["session_count"] or 0),
                logical_session_count=int(row["logical_session_count"] or 0),
                explicit_count=int(row["explicit_count"] or 0),
                auto_count=int(row["auto_count"] or 0),
                origin_breakdown=_tag_origin_breakdown(
                    self._conn, str(row["tag"]), clause, filter_params, self._tags_relation
                ),
                repo_breakdown=_tag_repo_breakdown(
                    self._conn, str(row["tag"]), clause, filter_params, self._tags_relation
                ),
                provenance=ArchiveInsightProvenance(
                    materializer_version=1,
                    materialized_at=_iso_from_ms(row["source_sort_key_ms"]) or "1970-01-01T00:00:00Z",
                    source_updated_at=_iso_from_ms(row["source_sort_key_ms"]),
                    source_sort_key=(
                        float(row["source_sort_key_ms"]) / 1000.0 if row["source_sort_key_ms"] is not None else None
                    ),
                ),
            )
            for row in rows
        ]

    def list_tool_usage_insights(self, query: ToolUsageInsightQuery | None = None) -> list[ToolUsageInsight]:
        """Aggregate tool-usage insights from action rows."""
        request = query or ToolUsageInsightQuery()
        builder_request = _tool_usage_builder_query(request)
        insight = build_tool_usage_insight(
            rows=self._tool_usage_rows(request),
            coverage_rows=self._tool_usage_provider_coverage_rows(),
            query=builder_request,
            materialized_at=datetime.now(UTC).isoformat(),
        )
        return [insight]

    def list_tool_call_count_rows(self, query: ToolUsageInsightQuery | None = None) -> list[dict[str, object]]:
        """Fast call-count-only tool rollups from tool-use blocks."""
        request = query or ToolUsageInsightQuery()
        where = ["b.block_type = 'tool_use'"]
        params: list[object] = []
        origin = _origin_for_tool_usage_filter(request.provider)
        if origin:
            where.append("s.origin = ?")
            params.append(origin)
        tool_expr = "COALESCE(NULLIF(LOWER(b.tool_name), ''), 'unknown')"
        if request.tool:
            where.append(f"{tool_expr} = LOWER(?)")
            params.append(request.tool)
        if request.mcp_server:
            mcp_prefix = f"mcp__{request.mcp_server.lower()}__"
            where.append(f"{tool_expr} >= ?")
            where.append(f"{tool_expr} < ?")
            params.append(mcp_prefix)
            params.append(f"{mcp_prefix}\U0010ffff")
        if request.action_kind:
            where.append("COALESCE(NULLIF(b.semantic_type, ''), 'tool_use') = ?")
            params.append(request.action_kind)
        if request.since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(request.since_ms)
        if request.limit is not None:
            limit_clause = "LIMIT ? OFFSET ?"
            params.extend((request.limit, request.offset))
        elif request.offset:
            limit_clause = "LIMIT -1 OFFSET ?"
            params.append(request.offset)
        else:
            limit_clause = ""
        rows = self._conn.execute(
            f"""
            SELECT
                s.origin AS origin,
                {tool_expr} AS normalized_tool_name,
                COALESCE(NULLIF(b.semantic_type, ''), 'tool_use') AS action_kind,
                COUNT(*) AS call_count
            FROM blocks b
            JOIN sessions s ON s.session_id = b.session_id
            WHERE {" AND ".join(where)}
            GROUP BY s.origin, normalized_tool_name, action_kind
            ORDER BY call_count DESC, s.origin ASC, normalized_tool_name ASC
            {limit_clause}
            """,
            tuple(params),
        ).fetchall()
        return [
            {
                "source_name": _provider_for_origin(str(row["origin"])).value,
                "origin": str(row["origin"] or "unknown-export"),
                "normalized_tool_name": str(row["normalized_tool_name"] or "unknown"),
                "action_kind": str(row["action_kind"] or "tool_use"),
                "call_count": int(row["call_count"] or 0),
            }
            for row in rows
        ]

    def list_tool_observed_event_count_rows(
        self, query: ToolUsageInsightQuery | None = None
    ) -> list[dict[str, object]]:
        """Tool outcome rollups from canonical tool-use/result block evidence."""
        request = query or ToolUsageInsightQuery()
        where = ["u.block_type = 'tool_use'"]
        params: list[object] = []
        origin = _origin_for_tool_usage_filter(request.provider)
        if origin:
            where.append("s.origin = ?")
            params.append(origin)
        tool_expr = "COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown')"
        handler_expr = (
            "CASE "
            f"WHEN {tool_expr} >= 'mcp__' AND {tool_expr} < 'mcp__\U0010ffff' THEN 'mcp' "
            "WHEN NULLIF(u.tool_command, '') IS NOT NULL THEN 'shell' "
            "ELSE COALESCE(NULLIF(u.semantic_type, ''), 'tool_use') "
            "END"
        )
        status_expr = (
            "CASE "
            "WHEN r.tool_result_exit_code IS NOT NULL "
            "THEN CASE WHEN r.tool_result_exit_code = 0 THEN 'ok' ELSE 'failed' END "
            "WHEN r.tool_result_is_error IS NOT NULL "
            "THEN CASE WHEN r.tool_result_is_error = 1 THEN 'failed' ELSE 'ok' END "
            "ELSE 'unknown' "
            "END"
        )
        where.append("r.rowid IS NOT NULL")
        if request.tool:
            where.append(f"{tool_expr} = LOWER(?)")
            params.append(request.tool)
        if request.mcp_server:
            mcp_prefix = f"mcp__{request.mcp_server.lower()}__"
            where.append(f"{tool_expr} >= ?")
            where.append(f"{tool_expr} < ?")
            params.append(mcp_prefix)
            params.append(f"{mcp_prefix}\U0010ffff")
        if request.action_kind:
            where.append(f"{handler_expr} = ?")
            params.append(request.action_kind)
        if request.since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(request.since_ms)
        if request.limit is not None:
            limit_clause = "LIMIT ? OFFSET ?"
            params.extend((request.limit, request.offset))
        elif request.offset:
            limit_clause = "LIMIT -1 OFFSET ?"
            params.append(request.offset)
        else:
            limit_clause = ""
        rows = self._conn.execute(
            f"""
            SELECT
                s.origin AS origin,
                {tool_expr} AS normalized_tool_name,
                {handler_expr} AS action_kind,
                {status_expr} AS status,
                COUNT(*) AS event_count
            FROM blocks u
            JOIN sessions s ON s.session_id = u.session_id
            LEFT JOIN blocks r
                ON r.tool_id = u.tool_id
               AND r.session_id = u.session_id
               AND r.block_type = 'tool_result'
            WHERE {" AND ".join(where)}
            GROUP BY s.origin, normalized_tool_name, action_kind, status
            ORDER BY event_count DESC, s.origin ASC, normalized_tool_name ASC, status ASC
            {limit_clause}
            """,
            tuple(params),
        ).fetchall()
        return [
            {
                "source_name": _provider_for_origin(str(row["origin"])).value,
                "origin": str(row["origin"] or "unknown-export"),
                "normalized_tool_name": str(row["normalized_tool_name"] or "unknown"),
                "action_kind": str(row["action_kind"] or "unknown"),
                "status": str(row["status"] or "unknown"),
                "event_count": int(row["event_count"] or 0),
            }
            for row in rows
        ]

    def list_tool_action_evidence_count_rows(
        self,
        query: ToolUsageInsightQuery | None = None,
        *,
        detail_patterns: tuple[str, ...] = (),
        since_ms: int | None = None,
    ) -> list[dict[str, object]]:
        """Tool/affordance rollups from the canonical ``actions`` projection.

        Unlike raw tool-use block counts, this basis can match command/path/input
        details and then normalize generic shell rows into families such as
        ``codebase-memory/command-detail``. The normalized grouping is a read
        projection; raw tool names remain folded into the evidence kind and
        matched-by fields rather than replacing source evidence.
        """

        request = query or ToolUsageInsightQuery()
        where: list[str] = ["u.block_type = 'tool_use'"]
        params: list[object] = []
        origin = _origin_for_tool_usage_filter(request.provider)
        if origin:
            where.append("s.origin = ?")
            params.append(origin)
        tool_expr = "COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown')"
        if request.tool:
            where.append(f"{tool_expr} = LOWER(?)")
            params.append(request.tool)
        tool_patterns: tuple[str, ...] = ()
        if request.mcp_server:
            tool_patterns = (f"mcp__{request.mcp_server.lower()}__",)
            where.append(f"{tool_expr} >= ?")
            where.append(f"{tool_expr} < ?")
            params.append(tool_patterns[0])
            params.append(f"{tool_patterns[0]}\U0010ffff")
        if request.action_kind:
            where.append("COALESCE(NULLIF(u.semantic_type, ''), 'tool_use') = ?")
            params.append(request.action_kind)
        effective_since_ms = since_ms if since_ms is not None else request.since_ms
        if effective_since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(effective_since_ms)
        cleaned_details = _clean_affordance_patterns(detail_patterns)
        fts_queries = tuple(
            fts_query for pattern in cleaned_details if (fts_query := normalize_fts5_query(pattern)) is not None
        )
        if cleaned_details:
            if not fts_queries:
                return []
            where.append(
                "("
                + " OR ".join("u.rowid IN (SELECT rowid FROM messages_fts WHERE text MATCH ?)" for _ in fts_queries)
                + ")"
            )
            params.extend(fts_queries)

        def fetch_rows() -> list[sqlite3.Row]:
            return list(
                self._conn.execute(
                    f"""
            SELECT
                s.origin AS origin,
                u.tool_name AS tool_name,
                COALESCE(NULLIF(u.semantic_type, ''), 'tool_use') AS action_kind,
                u.session_id AS session_id,
                u.message_id AS message_id,
                COALESCE(u.tool_command, '') || ' ' ||
                    COALESCE(u.tool_path, '') || ' ' ||
                    COALESCE(u.tool_input, '') AS match_detail,
                r.tool_result_is_error AS is_error,
                r.tool_result_exit_code AS exit_code
            FROM blocks u
            JOIN sessions s ON s.session_id = u.session_id
            LEFT JOIN blocks r
                ON r.tool_id = u.tool_id
               AND r.session_id = u.session_id
               AND r.block_type = 'tool_result'
            {"WHERE " + " AND ".join(where) if where else ""}
            """,
                    tuple(params),
                ).fetchall()
            )

        rows = fetch_rows()

        buckets: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
        sessions: dict[tuple[str, str, str, str, str], set[str]] = {}
        for row in rows:
            source_name = _provider_for_origin(str(row["origin"])).value
            public_row = {
                "tool_name": str(row["tool_name"] or ""),
                "match_detail": str(row["match_detail"] or ""),
            }
            if cleaned_details and not any(
                pattern in str(public_row["match_detail"]).lower() for pattern in cleaned_details
            ):
                continue
            normalized_tool_name = _affordance_normalized_tool_name(public_row)
            evidence_kind = _affordance_evidence_kind(public_row)
            matched_by = _affordance_matched_by(
                public_row,
                tool_patterns=tool_patterns,
                detail_patterns=cleaned_details,
            )
            key = (
                source_name,
                str(row["origin"] or "unknown-export"),
                normalized_tool_name,
                str(row["action_kind"] or "tool_use"),
                evidence_kind,
            )
            bucket = buckets.setdefault(
                key,
                {
                    "source_name": source_name,
                    "origin": str(row["origin"] or "unknown-export"),
                    "normalized_tool_name": normalized_tool_name,
                    "action_kind": str(row["action_kind"] or "tool_use"),
                    "evidence_kind": evidence_kind,
                    "matched_by": matched_by,
                    "call_count": 0,
                    "session_count": 0,
                    "error_count": 0,
                    "nonzero_exit_count": 0,
                },
            )
            sessions.setdefault(key, set()).add(str(row["session_id"]))
            bucket["call_count"] = int(str(bucket["call_count"])) + 1
            bucket["error_count"] = int(str(bucket["error_count"])) + (1 if int(row["is_error"] or 0) == 1 else 0)
            bucket["nonzero_exit_count"] = int(str(bucket["nonzero_exit_count"])) + (
                1 if row["exit_code"] is not None and int(row["exit_code"] or 0) != 0 else 0
            )
        for key, bucket in buckets.items():
            bucket["session_count"] = len(sessions.get(key, set()))
        ordered = sorted(
            buckets.values(),
            key=lambda item: (
                -int(str(item["call_count"])),
                str(item["origin"]),
                str(item["normalized_tool_name"]),
                str(item["evidence_kind"]),
            ),
        )
        offset = request.offset or 0
        if request.limit is not None:
            ordered = ordered[offset : offset + request.limit]
        elif offset:
            ordered = ordered[offset:]
        return ordered

    def _tool_usage_rows(self, query: ToolUsageInsightQuery | None = None) -> list[ToolUsageRow]:
        request = query or ToolUsageInsightQuery()
        where: list[str] = []
        params: list[object] = []
        origin = _origin_for_tool_usage_filter(request.provider)
        if origin:
            where.append("s.origin = ?")
            params.append(origin)
        tool_expr = "COALESCE(NULLIF(LOWER(a.tool_name), ''), 'unknown')"
        if request.tool:
            where.append(f"{tool_expr} = LOWER(?)")
            params.append(request.tool)
        if request.mcp_server:
            mcp_prefix = f"mcp__{request.mcp_server.lower()}__"
            where.append(f"{tool_expr} >= ?")
            where.append(f"{tool_expr} < ?")
            params.append(mcp_prefix)
            params.append(f"{mcp_prefix}\U0010ffff")
        if request.action_kind:
            where.append("COALESCE(NULLIF(a.semantic_type, ''), 'tool_use') = ?")
            params.append(request.action_kind)
        if request.since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(request.since_ms)

        sql = """
            SELECT
                s.origin AS origin,
                {tool_expr} AS normalized_tool_name,
                COALESCE(NULLIF(a.semantic_type, ''), 'tool_use') AS action_kind,
                COUNT(*) AS call_count,
                COUNT(DISTINCT s.session_id) AS session_count,
                COUNT(DISTINCT a.message_id) AS message_count,
                COUNT(DISTINCT a.tool_use_block_id) AS distinct_tool_ids,
                SUM(CASE WHEN a.tool_path IS NOT NULL AND a.tool_path != '' THEN 1 ELSE 0 END) AS affected_path_calls,
                SUM(CASE WHEN a.output_text IS NOT NULL AND a.output_text != '' THEN 1 ELSE 0 END) AS output_text_calls
            FROM actions a
            JOIN sessions s ON s.session_id = a.session_id
            {where_clause}
            GROUP BY s.origin, normalized_tool_name, action_kind
            ORDER BY call_count DESC, s.origin ASC, normalized_tool_name ASC
            {limit_clause}
            """
        if request.limit is not None:
            limit_clause = "LIMIT ? OFFSET ?"
            params.extend((request.limit, request.offset))
        elif request.offset:
            limit_clause = "LIMIT -1 OFFSET ?"
            params.append(request.offset)
        else:
            limit_clause = ""
        rows = self._conn.execute(
            sql.format(
                tool_expr=tool_expr,
                where_clause=("WHERE " + " AND ".join(where)) if where else "",
                limit_clause=limit_clause,
            ),
            tuple(params),
        ).fetchall()
        return [
            {
                "source_name": _provider_for_origin(str(row["origin"])).value,
                "normalized_tool_name": str(row["normalized_tool_name"] or "unknown"),
                "action_kind": str(row["action_kind"] or "tool_use"),
                "call_count": int(row["call_count"] or 0),
                "session_count": int(row["session_count"] or 0),
                "message_count": int(row["message_count"] or 0),
                "distinct_tool_ids": int(row["distinct_tool_ids"] or 0),
                "affected_path_calls": int(row["affected_path_calls"] or 0),
                "output_text_calls": int(row["output_text_calls"] or 0),
            }
            for row in rows
        ]

    def _tool_usage_provider_coverage_rows(self) -> list[ToolUsageProviderCoverageRow]:
        rows = self._conn.execute(
            """
            SELECT
                s.origin AS origin,
                COUNT(DISTINCT s.session_id) AS session_count,
                COUNT(a.tool_use_block_id) AS action_count,
                COUNT(DISTINCT COALESCE(NULLIF(LOWER(a.tool_name), ''), 'unknown')) AS distinct_tool_count,
                COUNT(DISTINCT COALESCE(NULLIF(a.semantic_type, ''), 'tool_use')) AS distinct_action_kind_count,
                COUNT(a.tool_use_block_id) AS has_tool_id_signal,
                SUM(CASE WHEN a.tool_path IS NOT NULL AND a.tool_path != '' THEN 1 ELSE 0 END) AS has_affected_paths_signal,
                SUM(CASE WHEN a.output_text IS NOT NULL AND a.output_text != '' THEN 1 ELSE 0 END) AS has_output_text_signal
            FROM sessions s
            LEFT JOIN actions a ON a.session_id = s.session_id
            GROUP BY s.origin
            ORDER BY action_count DESC, session_count DESC, s.origin ASC
            """
        ).fetchall()
        return [
            {
                "source_name": _provider_for_origin(str(row["origin"])).value,
                "session_count": int(row["session_count"] or 0),
                "action_count": int(row["action_count"] or 0),
                "distinct_tool_count": int(row["distinct_tool_count"] or 0),
                "distinct_action_kind_count": int(row["distinct_action_kind_count"] or 0),
                "has_tool_id_signal": int(row["has_tool_id_signal"] or 0),
                "has_affected_paths_signal": int(row["has_affected_paths_signal"] or 0),
                "has_output_text_signal": int(row["has_output_text_signal"] or 0),
            }
            for row in rows
        ]

    def list_archive_coverage_insights(
        self,
        *,
        group_by: str = "provider",
        provider: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ArchiveCoverageInsight]:
        """Aggregate archive coverage from index tables."""
        origin = _origin_for_provider_value(provider)
        if group_by == "provider":
            return self._provider_coverage_insights(origin=origin, limit=limit, offset=offset)
        if group_by == "day":
            return self._time_bucket_coverage_insights(
                bucket_format="%Y-%m-%d",
                group_by="day",
                origin=origin,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=limit,
                offset=offset,
            )
        if group_by == "week":
            return self._time_bucket_coverage_insights(
                bucket_format="%Y-W%W",
                group_by="week",
                origin=origin,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=limit,
                offset=offset,
            )
        raise ValueError("archive coverage group_by must be one of: provider, day, week")

    def _provider_coverage_insights(
        self,
        *,
        origin: str | None,
        limit: int | None,
        offset: int,
    ) -> list[ArchiveCoverageInsight]:
        where = ""
        params: list[object] = []
        if origin is not None:
            where = "WHERE s.origin = ?"
            params.append(origin)
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT
                s.origin,
                COUNT(*) AS session_count,
                SUM(s.message_count) AS message_count,
                SUM(s.user_message_count) AS user_message_count,
                SUM(s.authored_user_message_count) AS authored_user_message_count,
                SUM(s.assistant_message_count) AS assistant_message_count,
                SUM(s.user_word_count) AS user_word_sum,
                SUM(s.authored_user_word_count) AS authored_user_word_sum,
                SUM(s.assistant_word_count) AS assistant_word_sum,
                SUM(s.tool_use_count) AS tool_use_count,
                SUM(s.thinking_count) AS thinking_count,
                SUM(CASE WHEN s.tool_use_count > 0 THEN 1 ELSE 0 END) AS sessions_with_tools,
                SUM(CASE WHEN s.thinking_count > 0 THEN 1 ELSE 0 END) AS sessions_with_thinking
            FROM sessions s
            {where}
            GROUP BY s.origin
            ORDER BY session_count DESC, s.origin
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [_provider_coverage_from_archive_row(row) for row in rows]

    def _time_bucket_coverage_insights(
        self,
        *,
        bucket_format: str,
        group_by: str,
        origin: str | None,
        since_ms: int | None,
        until_ms: int | None,
        limit: int | None,
        offset: int,
    ) -> list[ArchiveCoverageInsight]:
        where: list[str] = []
        params: list[object] = []
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT
                strftime('{bucket_format}', s.sort_key_ms / 1000, 'unixepoch') AS bucket,
                COUNT(DISTINCT s.session_id) AS session_count,
                COUNT(DISTINCT COALESCE(s.root_session_id, s.session_id)) AS logical_session_count,
                SUM(s.message_count) AS message_count,
                SUM(s.word_count) AS total_words,
                SUM(COALESCE(sp.cost_usd, 0.0)) AS total_cost_usd,
                SUM(COALESCE(sp.duration_ms, 0)) AS total_duration_ms,
                SUM(COALESCE(sp.duration_ms, 0)) AS total_wall_duration_ms,
                MAX(s.sort_key_ms) AS source_sort_key_ms
            FROM sessions s
            LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
            {clause}
            GROUP BY bucket
            HAVING bucket IS NOT NULL
            ORDER BY bucket DESC
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [
            ArchiveCoverageInsight(
                group_by=group_by,
                bucket=str(row["bucket"]),
                session_count=int(row["session_count"] or 0),
                logical_session_count=int(row["logical_session_count"] or 0),
                message_count=int(row["message_count"] or 0),
                total_cost_usd=float(row["total_cost_usd"] or 0.0),
                total_duration_ms=int(row["total_duration_ms"] or 0),
                total_wall_duration_ms=int(row["total_wall_duration_ms"] or 0),
                total_words=int(row["total_words"] or 0),
                work_event_breakdown=_coverage_work_event_breakdown(
                    self._conn,
                    str(row["bucket"]),
                    bucket_format,
                    origin=origin,
                    since_ms=since_ms,
                    until_ms=until_ms,
                ),
                repos_active=_coverage_repos_active(
                    self._conn,
                    str(row["bucket"]),
                    bucket_format,
                    origin=origin,
                    since_ms=since_ms,
                    until_ms=until_ms,
                ),
                origin_breakdown=_coverage_origin_breakdown(
                    self._conn,
                    str(row["bucket"]),
                    bucket_format,
                    origin=origin,
                    since_ms=since_ms,
                    until_ms=until_ms,
                ),
                provenance=ArchiveInsightProvenance(
                    materializer_version=1,
                    materialized_at=_iso_from_ms(row["source_sort_key_ms"]) or "1970-01-01T00:00:00Z",
                    source_updated_at=_iso_from_ms(row["source_sort_key_ms"]),
                    source_sort_key=(
                        float(row["source_sort_key_ms"]) / 1000.0 if row["source_sort_key_ms"] is not None else None
                    ),
                ),
            )
            for row in rows
        ]

    def set_user_metadata(self, session_ids: tuple[str, ...], pairs: tuple[tuple[str, object], ...]) -> int:
        """Set human-owned metadata as archive user.db assertions."""
        user_db_path = self.user_db_path
        initialize_archive_database(user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(user_db_path)
        user_conn.row_factory = sqlite3.Row
        try:
            changed = 0
            with user_conn:
                for session_id in tuple(
                    dict.fromkeys(self.resolve_session_id(session_id) for session_id in session_ids)
                ):
                    for key, value in pairs:
                        normalized_key = key.strip()
                        if not normalized_key:
                            raise ValueError("metadata key cannot be empty")
                        existing = read_assertion_envelope(
                            user_conn,
                            assertion_id_for_session_metadata(session_id, normalized_key),
                        )
                        if (
                            existing is not None
                            and existing.status != "deleted"
                            and _canonical_json_text(existing.value) == _canonical_json_text(value)
                        ):
                            continue
                        upsert_session_metadata_assertion(
                            user_conn,
                            session_id=session_id,
                            key=normalized_key,
                            value=value,
                        )
                        changed += 1
        finally:
            user_conn.close()
        return changed

    def read_user_metadata(self, session_id: str) -> dict[str, object]:
        """Read human-owned metadata assertions for one archive session."""
        resolved_session_id = self.resolve_session_id(session_id)
        if not self.user_db_path.exists():
            return {}
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        user_conn.row_factory = sqlite3.Row
        try:
            rows = list_assertions_for_target(user_conn, f"session:{resolved_session_id}", kind=AssertionKind.METADATA)
        finally:
            user_conn.close()
        decoded: dict[str, object] = {}
        for assertion in rows:
            if assertion.status == "deleted" or assertion.key is None:
                continue
            decoded[str(assertion.key)] = assertion.value
        return decoded

    def delete_user_metadata(self, session_id: str, key: str) -> int:
        """Mark one user metadata assertion deleted."""
        resolved_session_id = self.resolve_session_id(session_id)
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("metadata key cannot be empty")
        if not self.user_db_path.exists():
            return 0
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                assertion_id = assertion_id_for_session_metadata(resolved_session_id, normalized_key)
                assertion = read_assertion_envelope(user_conn, assertion_id)
                if assertion is None or assertion.status == "deleted":
                    return 0
                return 1 if mark_assertion_status(user_conn, assertion_id, "deleted") else 0
        finally:
            user_conn.close()

    def add_mark(self, target_type: str, target_id: str, mark_type: str) -> bool:
        """Add one user mark to archive user.db."""
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            assertion = read_assertion_envelope(user_conn, assertion_id_for_mark(target_type, target_id, mark_type))
            exists = assertion is not None and assertion.status != "deleted"
            with user_conn:
                upsert_mark(user_conn, target_type, target_id, mark_type)
            return not exists
        finally:
            user_conn.close()

    def remove_mark(self, target_type: str, target_id: str, mark_type: str) -> bool:
        """Remove one user mark from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                return mark_assertion_status(
                    user_conn,
                    assertion_id_for_mark(target_type, target_id, mark_type),
                    "deleted",
                )
        finally:
            user_conn.close()

    def list_marks(
        self,
        *,
        mark_type: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List user marks from archive user.db."""
        if not self.user_db_path.exists():
            return []
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            assertions = list_assertions_by_kind(user_conn, AssertionKind.MARK)
        finally:
            user_conn.close()
        out: list[dict[str, str]] = []
        for assertion in assertions:
            found_target_type, found_target_id = _split_user_target_ref(assertion.target_ref)
            if mark_type and assertion.key != mark_type:
                continue
            if target_type and found_target_type != target_type:
                continue
            if target_id and found_target_id != target_id:
                continue
            out.append(
                {
                    "target_type": found_target_type,
                    "target_id": found_target_id,
                    "session_id": _user_mark_session_id(found_target_type, found_target_id),
                    "message_id": found_target_id if found_target_type == "message" else "",
                    "mark_type": str(assertion.key or ""),
                    "created_at": str(assertion.created_at_ms),
                }
            )
        return out

    def save_annotation(self, annotation_id: str, target_type: str, target_id: str, note_text: str) -> bool:
        """Create or update one annotation in archive user.db."""
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            assertion = read_assertion_envelope(user_conn, assertion_id_for_annotation(annotation_id))
            exists = assertion is not None and assertion.status != "deleted"
            with user_conn:
                upsert_annotation(
                    user_conn,
                    target_type,
                    target_id,
                    note_text,
                    annotation_id=annotation_id,
                )
            return not exists
        finally:
            user_conn.close()

    def get_annotation(self, annotation_id: str) -> dict[str, str] | None:
        """Read one annotation from archive user.db."""
        rows = self.list_annotations(annotation_id=annotation_id)
        return rows[0] if rows else None

    def list_annotations(
        self,
        *,
        annotation_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List annotations from archive user.db.

        When ``session_id`` is supplied (and no explicit target filter),
        the result includes both the session-target annotation and every
        message-target annotation whose native message id is prefixed by the
        session id (``session_id:message_native_id``). This mirrors the read model
        contract where annotations on messages belonging to a session were
        listed under that session.
        """
        if not self.user_db_path.exists():
            return []
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            assertions = list_assertions_by_kind(user_conn, AssertionKind.ANNOTATION)
        finally:
            user_conn.close()
        out: list[dict[str, str]] = []
        for assertion in assertions:
            found_annotation_id = str(assertion.key or "")
            found_target_type, found_target_id = _split_user_target_ref(assertion.target_ref)
            if annotation_id and found_annotation_id != annotation_id:
                continue
            if session_id and target_id is None:
                belongs_to_session = (found_target_type == "session" and found_target_id == session_id) or (
                    found_target_type == "message"
                    and (found_target_id == session_id or found_target_id.startswith(f"{session_id}:"))
                )
                if not belongs_to_session:
                    continue
            if target_type and found_target_type != target_type:
                continue
            if target_id and found_target_id != target_id:
                continue
            out.append(
                {
                    "annotation_id": found_annotation_id,
                    "target_type": found_target_type,
                    "target_id": found_target_id,
                    "session_id": _user_mark_session_id(found_target_type, found_target_id),
                    "message_id": found_target_id if found_target_type == "message" else "",
                    "note_text": assertion.body_text or "",
                    "created_at": str(assertion.created_at_ms),
                    "updated_at": str(assertion.updated_at_ms),
                }
            )
        return out

    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete one annotation from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                return mark_assertion_status(user_conn, assertion_id_for_annotation(annotation_id), "deleted")
        finally:
            user_conn.close()

    def save_view(self, view_id: str, name: str, query_json: str) -> bool:
        """Create or update one saved view in archive user.db."""
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("name must not be empty")
        query = json.loads(query_json)
        if not isinstance(query, dict):
            raise ValueError("query_json must encode an object")
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            assertion_id = assertion_id_for_saved_view(view_id)
            assertion = read_assertion_envelope(user_conn, assertion_id)
            name_assertion = _active_assertion_by_kind_key(user_conn, AssertionKind.SAVED_QUERY, normalized_name)
            exists = (assertion is not None and assertion.status != "deleted") or name_assertion is not None
            with user_conn:
                if name_assertion is not None and name_assertion.assertion_id != assertion_id:
                    mark_assertion_status(user_conn, name_assertion.assertion_id, "deleted")
                upsert_saved_view(user_conn, normalized_name, query, view_id=view_id)
            return not exists
        finally:
            user_conn.close()

    def get_view(self, view_id: str) -> dict[str, str] | None:
        """Get one saved view by id from archive user.db."""
        return next((row for row in self.list_views() if row["view_id"] == view_id), None)

    def get_view_by_name(self, name: str) -> dict[str, str] | None:
        """Get one saved view by name from archive user.db."""
        return next((row for row in self.list_views() if row["name"] == name), None)

    def list_views(self) -> list[dict[str, str]]:
        """List saved views from archive user.db."""
        return self._list_views()

    def _list_views(self, *, where: str = "", params: tuple[object, ...] = ()) -> list[dict[str, str]]:
        del where, params
        if not self.user_db_path.exists():
            return []
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            assertions = list_assertions_by_kind(user_conn, AssertionKind.SAVED_QUERY)
        finally:
            user_conn.close()
        return [
            {
                "view_id": _id_from_target_ref(assertion.target_ref, "saved_view:"),
                "name": str(assertion.key or ""),
                "query_json": json.dumps(
                    assertion.value if isinstance(assertion.value, dict) else {},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "created_at": str(assertion.created_at_ms),
            }
            for assertion in assertions
        ]

    def delete_view(self, view_id: str) -> bool:
        """Delete one saved view from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                return mark_assertion_status(user_conn, assertion_id_for_saved_view(view_id), "deleted")
        finally:
            user_conn.close()

    def save_recall_pack(
        self,
        pack_id: str,
        label: str,
        session_ids_json: str,
        payload_json: str,
    ) -> bool:
        """Create or update one recall pack in archive user.db."""
        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise ValueError("payload_json must encode an object")
        payload = dict(payload)
        payload["session_ids_json"] = session_ids_json
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            assertion = read_assertion_envelope(user_conn, assertion_id_for_recall_pack(pack_id))
            exists = assertion is not None and assertion.status != "deleted"
            with user_conn:
                upsert_recall_pack(user_conn, label, payload, recall_pack_id=pack_id)
            return not exists
        finally:
            user_conn.close()

    def get_recall_pack(self, pack_id: str) -> dict[str, str] | None:
        """Get one recall pack by id from archive user.db."""
        return next((row for row in self.list_recall_packs() if row["pack_id"] == pack_id), None)

    def list_recall_packs(self) -> list[dict[str, str]]:
        """List recall packs from archive user.db."""
        return self._list_recall_packs()

    def _list_recall_packs(self, *, where: str = "", params: tuple[object, ...] = ()) -> list[dict[str, str]]:
        del where, params
        if not self.user_db_path.exists():
            return []
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            assertions = list_assertions_by_kind(user_conn, AssertionKind.RECALL_PACK)
        finally:
            user_conn.close()
        out: list[dict[str, str]] = []
        for assertion in assertions:
            payload = assertion.value if isinstance(assertion.value, dict) else {}
            if not isinstance(payload, dict):
                payload = {}
            session_ids_json = payload.pop("session_ids_json", "[]")
            out.append(
                {
                    "pack_id": _id_from_target_ref(assertion.target_ref, "recall_pack:"),
                    "label": str(assertion.key or ""),
                    "session_ids_json": str(session_ids_json),
                    "payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    "created_at": str(assertion.created_at_ms),
                }
            )
        return out

    def delete_recall_pack(self, pack_id: str) -> bool:
        """Delete one recall pack from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                return mark_assertion_status(user_conn, assertion_id_for_recall_pack(pack_id), "deleted")
        finally:
            user_conn.close()

    def save_workspace(
        self,
        *,
        workspace_id: str,
        name: str,
        mode: str,
        open_targets_json: str,
        layout_json: str,
        active_target_json: str,
    ) -> bool:
        """Create or update one reader workspace in archive user.db."""
        settings: dict[str, object] = {
            "mode": mode,
            "open_targets_json": open_targets_json,
            "layout_json": layout_json,
            "active_target_json": active_target_json,
        }
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            assertion_id = assertion_id_for_workspace(workspace_id)
            assertion = read_assertion_envelope(user_conn, assertion_id)
            name_assertion = _active_assertion_by_kind_key(user_conn, AssertionKind.WORKSPACE_NOTE, name)
            exists = (assertion is not None and assertion.status != "deleted") or name_assertion is not None
            with user_conn:
                if name_assertion is not None and name_assertion.assertion_id != assertion_id:
                    mark_assertion_status(user_conn, name_assertion.assertion_id, "deleted")
                upsert_workspace(user_conn, name, settings, workspace_id=workspace_id)
            return not exists
        finally:
            user_conn.close()

    def get_workspace(self, workspace_id: str) -> dict[str, str] | None:
        """Get one workspace by id from archive user.db."""
        return next((row for row in self.list_workspaces() if row["workspace_id"] == workspace_id), None)

    def list_workspaces(self) -> list[dict[str, str]]:
        """List workspaces from archive user.db."""
        return self._list_workspaces()

    def _list_workspaces(self, *, where: str = "", params: tuple[object, ...] = ()) -> list[dict[str, str]]:
        del where, params
        if not self.user_db_path.exists():
            return []
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            assertions = list_assertions_by_kind(user_conn, AssertionKind.WORKSPACE_NOTE)
        finally:
            user_conn.close()
        out: list[dict[str, str]] = []
        for assertion in assertions:
            settings = assertion.value if isinstance(assertion.value, dict) else {}
            if not isinstance(settings, dict):
                settings = {}
            out.append(
                {
                    "workspace_id": _id_from_target_ref(assertion.target_ref, "workspace:"),
                    "name": str(assertion.key or ""),
                    "mode": str(settings.get("mode") or ""),
                    "open_targets_json": str(settings.get("open_targets_json") or "[]"),
                    "layout_json": str(settings.get("layout_json") or "{}"),
                    "active_target_json": str(settings.get("active_target_json") or "{}"),
                    "created_at": str(assertion.created_at_ms),
                    "updated_at": str(assertion.updated_at_ms),
                }
            )
        return out

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete one workspace from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                return mark_assertion_status(user_conn, assertion_id_for_workspace(workspace_id), "deleted")
        finally:
            user_conn.close()

    def record_correction(
        self,
        session_id: str,
        kind: str,
        payload: dict[str, str],
        *,
        note: str | None = None,
    ) -> LearningCorrection:
        """Record one learning correction in archive user.db."""
        resolved_session_id = self.resolve_session_id(session_id)
        correction_kind = parse_correction_kind(kind)
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        stored_payload: dict[str, object] = {"payload": dict(payload), "note": note}
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                upsert_correction(
                    user_conn,
                    "insight",
                    resolved_session_id,
                    correction_kind.value,
                    stored_payload,
                )
        finally:
            user_conn.close()
        listed = self.list_corrections(session_id=resolved_session_id, kind=correction_kind.value)
        if not listed:
            raise KeyError((resolved_session_id, correction_kind.value))
        return listed[0]

    def list_corrections(self, *, session_id: str | None = None, kind: str | None = None) -> list[LearningCorrection]:
        """List learning corrections from archive user.db."""
        if not self.user_db_path.exists():
            return []
        resolved_session_id = self.resolve_session_id(session_id) if session_id else None
        correction_kind = parse_correction_kind(kind).value if kind is not None else None
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            assertions = list_assertions_by_kind(user_conn, AssertionKind.CORRECTION)
        finally:
            user_conn.close()
        out: list[LearningCorrection] = []
        for assertion in assertions:
            target_type, target_id = _split_user_target_ref(assertion.target_ref)
            if target_type != "insight":
                continue
            if resolved_session_id is not None and target_id != resolved_session_id:
                continue
            if correction_kind is not None and assertion.key != correction_kind:
                continue
            payload_json = json.dumps(assertion.value if isinstance(assertion.value, dict) else {}, sort_keys=True)
            out.append(
                _learning_correction_from_archive_row(
                    (target_id, str(assertion.key or ""), payload_json, assertion.updated_at_ms)
                )
            )
        return out

    def delete_correction(self, session_id: str, kind: str) -> bool:
        """Delete one learning correction from archive user.db."""
        resolved_session_id = self.resolve_session_id(session_id)
        correction_kind = parse_correction_kind(kind)
        if not self.user_db_path.exists():
            return False
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                correction_id = correction_id_for("insight", resolved_session_id, correction_kind.value)
                return mark_assertion_status(user_conn, assertion_id_for_correction(correction_id), "deleted")
        finally:
            user_conn.close()

    def clear_corrections(self, session_id: str) -> int:
        """Delete all learning corrections for one archive session."""
        resolved_session_id = self.resolve_session_id(session_id)
        if not self.user_db_path.exists():
            return 0
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                deleted_count = 0
                for assertion in list_assertions_by_kind(user_conn, AssertionKind.CORRECTION):
                    target_type, target_id = _split_user_target_ref(assertion.target_ref)
                    if (
                        target_type == "insight"
                        and target_id == resolved_session_id
                        and mark_assertion_status(user_conn, assertion.assertion_id, "deleted")
                    ):
                        deleted_count += 1
                return deleted_count
        finally:
            user_conn.close()

    def post_blackboard_note(
        self,
        body: str,
        *,
        target_type: str | None = None,
        target_id: str | None = None,
        note_id: str | None = None,
        author_ref: str | None = None,
        author_kind: str = "user",
        evidence_refs: tuple[str, ...] = (),
        staleness: dict[str, object] | None = None,
        context_policy: dict[str, object] | None = None,
    ) -> ArchiveBlackboardNoteEnvelope:
        """Insert-or-update one blackboard note in archive user.db."""
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            envelope = upsert_blackboard_note(
                user_conn,
                body,
                target_type=target_type,
                target_id=target_id,
                note_id=note_id,
                author_ref=author_ref,
                author_kind=author_kind,
                evidence_refs=evidence_refs,
                staleness=staleness,
                context_policy=context_policy,
            )
            user_conn.commit()
            return envelope
        finally:
            user_conn.close()

    def list_blackboard_notes(self, *, limit: int | None = None) -> list[ArchiveBlackboardNoteEnvelope]:
        """List blackboard notes from archive user.db, newest first.

        Assertion rows own note ids, targets, body text, and timestamps.
        Structured-field decoding (kind/title/scope) is a presentation concern
        handled by ``polylogue.archive.blackboard``.
        """
        if not self.user_db_path.exists():
            return []
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            return list_archive_blackboard_note_envelopes(user_conn, limit=limit)
        finally:
            user_conn.close()

    def delete_sessions(self, session_ids: tuple[str, ...]) -> int:
        """Delete rebuildable archive sessions by id.

        User-tier overlays are intentionally left in ``user.db``; the user
        overlay orphan checker owns follow-up visibility for those durable rows.
        """
        resolved_session_ids = tuple(dict.fromkeys(self.resolve_session_id(session_id) for session_id in session_ids))
        if not resolved_session_ids:
            return 0
        conn = sqlite3.connect(self.index_db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        deleted = 0
        try:
            with conn:
                for session_id in resolved_session_ids:
                    cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                    deleted += max(int(cursor.rowcount), 0)
        finally:
            conn.close()
        return deleted

    def _attach_user_tier_if_present(self) -> None:
        if self._user_tier_attached or not self.user_db_path.exists():
            return
        user_db_uri = f"file:{self.user_db_path}?mode=ro" if self._read_only else str(self.user_db_path)
        self._conn.execute("ATTACH DATABASE ? AS user_tier", (user_db_uri,))
        self._user_tier_attached = True
        self._tags_relation = _all_session_tags_sql()

    def count_sessions(
        self,
        *,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        session_id: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
    ) -> int:
        """Count sessions in the archive index."""
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        if session_id is not None:
            try:
                resolved_id = self.resolve_session_id(session_id)
            except KeyError:
                return 0
            where = f"{where} AND s.session_id = ?" if where else "WHERE s.session_id = ?"
            params.append(resolved_id)
        return int(self._conn.execute(f"SELECT COUNT(*) FROM sessions s {where}", params).fetchone()[0])

    def session_insight_status(self) -> SessionInsightStatusSnapshot:
        """Return readiness for session insight tables."""
        return session_insight_status_sync(self._conn)

    def insight_readiness_report(self, query: InsightReadinessQuery | None = None) -> InsightReadinessReport:
        """Return public insight readiness from tables."""
        request = query or InsightReadinessQuery()
        selected = (
            tuple(normalize_insight_readiness_name(insight) for insight in request.insights)
            if request.insights
            else known_insight_readiness_names()
        )
        status = self.session_insight_status()
        origin_filter = _origin_for_provider_value(request.provider)
        since_ms = _epoch_ms_from_iso(request.since)
        until_ms = _epoch_ms_from_iso(request.until)
        total_sessions = self.count_sessions(origin=origin_filter, since_ms=since_ms, until_ms=until_ms)
        coverage = self._archive_session_provider_coverage(origin=origin_filter, since_ms=since_ms, until_ms=until_ms)
        entries = tuple(
            entry
            for name in selected
            if (
                entry := self._insight_readiness_entry(
                    name,
                    status=status,
                    total_sessions=total_sessions,
                    provider_coverage=coverage,
                    origin=origin_filter,
                    since_ms=since_ms,
                    until_ms=until_ms,
                )
            )
            is not None
        )
        return InsightReadinessReport(
            checked_at=datetime.now(UTC).isoformat(),
            aggregate_verdict=_insight_readiness_aggregate_verdict(entries),
            total_sessions=total_sessions,
            provider=request.provider,
            since=request.since,
            until=request.until,
            insights=entries,
        )

    def audit_insight_rigor(self, query: InsightRigorAuditQuery | None = None) -> InsightRigorAuditReport:
        """Audit insight rigor over read models."""
        request = query or InsightRigorAuditQuery()
        targeted = set(request.insights) if request.insights else None
        entries = []
        for contract in list_rigor_contracts():
            if targeted is not None and contract.insight_name not in targeted:
                continue
            rows = self._rigor_audit_rows(contract.insight_name, limit=max(request.sample_limit, 0))
            entries.append(_audit_one(rows, contract))
        return InsightRigorAuditReport(sample_limit=request.sample_limit, entries=tuple(entries))

    def _rigor_audit_rows(self, insight_name: str, *, limit: int) -> list[object]:
        if insight_name == "session_profiles":
            return list(self.list_session_profile_insights(limit=limit))
        if insight_name == "session_work_events":
            return list(self.list_session_work_event_insights(limit=limit))
        if insight_name == "session_phases":
            return list(self.list_session_phase_insights(limit=limit))
        if insight_name == "threads":
            return list(self.list_thread_insights(limit=limit))
        if insight_name == "session_tag_rollups":
            return list(self.list_session_tag_rollup_insights(limit=limit))
        return []

    def _archive_session_provider_coverage(
        self, *, origin: str | None, since_ms: int | None, until_ms: int | None
    ) -> tuple[InsightProviderCoverage, ...]:
        """Per-provider session distribution for insight readiness coverage."""
        where: list[str] = []
        params: list[object] = []
        if origin is not None:
            where.append("origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("COALESCE(updated_at_ms, created_at_ms) >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("COALESCE(updated_at_ms, created_at_ms) <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        rows = self._conn.execute(
            f"SELECT origin, COUNT(*) AS n, MIN(created_at_ms) AS lo, MAX(updated_at_ms) AS hi "
            f"FROM sessions {clause} GROUP BY origin ORDER BY n DESC, origin",
            tuple(params),
        ).fetchall()
        return tuple(
            InsightProviderCoverage(
                source_name=_provider_for_origin(str(row["origin"])).value,
                row_count=int(row["n"]),
                min_time=_iso_from_ms(row["lo"]),
                max_time=_iso_from_ms(row["hi"]),
            )
            for row in rows
        )

    def _readiness_session_filter(
        self, *, origin: str | None, since_ms: int | None, until_ms: int | None
    ) -> tuple[str, list[object]]:
        """Build a ``WHERE`` fragment over the joined ``sessions`` (alias ``s``)."""
        clauses: list[str] = []
        params: list[object] = []
        if origin is not None:
            clauses.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            clauses.append("COALESCE(s.updated_at_ms, s.created_at_ms) >= ?")
            params.append(since_ms)
        if until_ms is not None:
            clauses.append("COALESCE(s.updated_at_ms, s.created_at_ms) <= ?")
            params.append(until_ms)
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    def _archive_materialization_signals(
        self,
        insight_type: str,
        *,
        origin: str | None,
        since_ms: int | None,
        until_ms: int | None,
    ) -> tuple[tuple[InsightVersionCoverage, ...], int, int]:
        """Derive version coverage, incompatible count, and native staleness.

        Reads the ``insight_materialization`` high-water marks for ``insight_type``
        joined to ``sessions``. A row is *incompatible* (legacy) when its
        ``materializer_version`` is below ``SESSION_INSIGHT_MATERIALIZER_VERSION``;
        it is *stale* when its captured ``source_sort_key_ms`` no longer matches the
        live session ``sort_key_ms`` (the native source high-water mark). The
        ``session_profiles.materializer_version``/``source_sort_key`` columns are not
        used here: they are not reliably populated by the canonical rebuild path,
        so the materialization ledger is the authoritative provenance source.
        """
        if not _table_exists(self._conn, "insight_materialization"):
            return ((), 0, 0)
        clause, params = self._readiness_session_filter(origin=origin, since_ms=since_ms, until_ms=until_ms)
        version_rows = self._conn.execute(
            "SELECT im.materializer_version AS version, COUNT(*) AS n "
            "FROM insight_materialization AS im "
            "JOIN sessions AS s ON s.session_id = im.session_id "
            f"WHERE im.insight_type = ?{clause} "
            "GROUP BY im.materializer_version ORDER BY im.materializer_version",
            (insight_type, *params),
        ).fetchall()
        versions = {str(int(row["version"])): int(row["n"]) for row in version_rows}
        incompatible_count = sum(
            count for version, count in versions.items() if int(version) < SESSION_INSIGHT_MATERIALIZER_VERSION
        )
        version_coverage = (
            (
                InsightVersionCoverage(
                    field="materializer_version",
                    current_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
                    versions=versions,
                    incompatible_count=incompatible_count,
                ),
            )
            if versions
            else ()
        )
        stale_row = self._conn.execute(
            "SELECT COUNT(*) AS n "
            "FROM insight_materialization AS im "
            "JOIN sessions AS s ON s.session_id = im.session_id "
            f"WHERE im.insight_type = ?{clause} "
            "AND COALESCE(im.source_sort_key_ms, -1) != COALESCE(s.sort_key_ms, -1)",
            (insight_type, *params),
        ).fetchone()
        stale_count = int(stale_row["n"]) if stale_row is not None else 0
        return (version_coverage, incompatible_count, stale_count)

    def _archive_fallback_coverage(
        self,
        table_name: str,
        column_paths: tuple[tuple[str, str], ...],
        *,
        origin: str | None,
        since_ms: int | None,
        until_ms: int | None,
    ) -> tuple[int, dict[str, int]]:
        """Count rows whose enrichment provenance carries fallback reasons.

        Each insight row stores its fallback markers as JSON arrays under
        ``$.fallback_reasons`` inside one or more payload columns (e.g.
        ``inference_payload_json`` and ``enrichment_payload_json`` on
        ``session_profiles``). A row is *degraded* when any declared
        ``(column, path)`` holds a non-empty array; the row is counted at most
        once regardless of how many columns flag it. ``reason_totals`` sums
        occurrences per reason across every inspected column.
        """
        if not _table_exists(self._conn, table_name):
            return (0, {})
        clause, params = self._readiness_session_filter(origin=origin, since_ms=since_ms, until_ms=until_ms)
        any_terms = " OR ".join(
            f"json_array_length(COALESCE(json_extract(t.{column}, '{path}'), '[]')) > 0"
            for column, path in column_paths
        )
        degraded_row = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM {table_name} AS t "
            "JOIN sessions AS s ON s.session_id = t.session_id "
            f"WHERE ({any_terms}){clause}",
            tuple(params),
        ).fetchone()
        degraded_count = int(degraded_row["n"]) if degraded_row is not None else 0
        reason_totals: dict[str, int] = {}
        for column, path in column_paths:
            rows = self._conn.execute(
                "SELECT value AS reason, COUNT(*) AS occurrences "
                f"FROM {table_name} AS t "
                "JOIN sessions AS s ON s.session_id = t.session_id, "
                f"json_each(COALESCE(json_extract(t.{column}, '{path}'), '[]')) "
                f"WHERE 1=1{clause} GROUP BY value",
                tuple(params),
            ).fetchall()
            for row in rows:
                reason = str(row["reason"])
                reason_totals[reason] = reason_totals.get(reason, 0) + int(row["occurrences"])
        return (degraded_count, dict(sorted(reason_totals.items())))

    def _insight_readiness_entry(
        self,
        name: str,
        *,
        status: SessionInsightStatusSnapshot,
        total_sessions: int,
        provider_coverage: tuple[InsightProviderCoverage, ...] = (),
        origin: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
    ) -> InsightReadinessEntry | None:
        specs = {
            "session_profiles": (
                "Session Profiles",
                "session_profiles",
                status.profile_row_count,
                total_sessions,
                status.missing_profile_row_count,
                0,
                status.orphan_profile_row_count,
                {"profile_rows_ready": status.profile_rows_ready},
                ("session_profiles",),
            ),
            "session_work_events": (
                "Work Events",
                "session_work_events",
                status.work_event_inference_count,
                status.expected_work_event_inference_count,
                0,
                status.stale_work_event_inference_count,
                status.orphan_work_event_inference_count,
                {"work_event_inference_rows_ready": status.work_event_inference_rows_ready},
                ("session_work_events",),
            ),
            "session_phases": (
                "Session Phases",
                "session_phases",
                status.phase_count,
                status.expected_phase_count,
                0,
                status.stale_phase_count,
                status.orphan_phase_count,
                {"phase_rows_ready": status.phase_rows_ready},
                ("session_phases",),
            ),
            "threads": (
                "Threads",
                "threads",
                status.thread_count,
                status.root_threads,
                0,
                status.stale_thread_count,
                status.orphan_thread_count,
                {"threads_ready": status.threads_ready},
                ("threads", "thread_sessions"),
            ),
            "session_tag_rollups": (
                "Session Tag Rollups",
                "session_tags",
                status.tag_rollup_count,
                status.expected_tag_rollup_count,
                0,
                status.stale_tag_rollup_count,
                0,
                {"tag_rollups_ready": status.tag_rollups_ready},
                ("session_tags",),
            ),
            "archive_coverage": (
                "Archive Coverage",
                "sessions",
                total_sessions,
                total_sessions,
                0,
                0,
                0,
                {},
                ("sessions",),
            ),
        }
        spec = specs.get(name)
        if spec is None:
            return None
        (
            display_name,
            table_name,
            row_count,
            expected_row_count,
            missing_count,
            stale_count,
            orphan_count,
            ready_flags,
            artifact_names,
        ) = spec
        table_present = _table_exists(self._conn, table_name)
        artifacts = tuple(
            InsightStorageArtifact(
                name=artifact,
                present=_table_exists(self._conn, artifact),
                ready=ready_flags[next(iter(ready_flags))] if len(ready_flags) == 1 else None,
            )
            for artifact in artifact_names
        )
        # Provenance-backed insights (profiles, work events, phases) carry their
        # materializer version and source high-water mark in the
        # ``insight_materialization`` ledger; the #1278 fallback taxonomy lives in
        # each session profile's ``provenance_json``. Threads/tags/coverage have no
        # such ledger entry and keep the status-derived staleness only.
        version_coverage: tuple[InsightVersionCoverage, ...] = ()
        incompatible_count = 0
        materialization_type = _INSIGHT_MATERIALIZATION_TYPE.get(name)
        if materialization_type is not None and table_present:
            version_coverage, incompatible_count, native_stale = self._archive_materialization_signals(
                materialization_type, origin=origin, since_ms=since_ms, until_ms=until_ms
            )
            stale_count = native_stale
        degraded_count = 0
        fallback_reason_counts: dict[str, int] = {}
        fallback = _INSIGHT_FALLBACK_PAYLOAD.get(name)
        if fallback is not None and table_present:
            fallback_table, fallback_column_paths = fallback
            degraded_count, fallback_reason_counts = self._archive_fallback_coverage(
                fallback_table,
                fallback_column_paths,
                origin=origin,
                since_ms=since_ms,
                until_ms=until_ms,
            )
        verdict = _archive_insight_readiness_verdict(
            table_present=table_present,
            row_count=row_count,
            expected_row_count=expected_row_count,
            missing_count=missing_count,
            stale_count=stale_count,
            orphan_count=orphan_count,
            incompatible_count=incompatible_count,
            degraded_count=degraded_count,
            ready_flags=ready_flags,
            total_sessions=total_sessions,
        )
        return InsightReadinessEntry(
            insight_name=name,
            display_name=display_name,
            verdict=verdict,
            row_count=row_count,
            expected_row_count=expected_row_count,
            missing_count=missing_count,
            stale_count=stale_count,
            orphan_count=orphan_count,
            incompatible_count=incompatible_count,
            degraded_count=degraded_count,
            fallback_reason_counts=fallback_reason_counts,
            storage_artifacts=artifacts,
            ready_flags=ready_flags,
            provider_coverage=provider_coverage,
            version_coverage=version_coverage,
            evidence=_archive_insight_readiness_evidence(
                row_count=row_count,
                expected_row_count=expected_row_count,
                missing_count=missing_count,
                stale_count=stale_count,
                orphan_count=orphan_count,
                incompatible_count=incompatible_count,
                degraded_count=degraded_count,
                fallback_reason_counts=fallback_reason_counts,
                ready_flags=ready_flags,
            ),
        )

    def list_summaries(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        session_id: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
        sample: bool = False,
        sort: str | None = None,
        reverse: bool = False,
    ) -> list[ArchiveSessionSummary]:
        """List session summaries ordered like the normal archive recency view."""
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        if session_id is not None:
            try:
                resolved_id = self.resolve_session_id(session_id)
            except KeyError:
                return []
            where = f"{where} AND s.session_id = ?" if where else "WHERE s.session_id = ?"
            params.append(resolved_id)
        order_by = _summary_order_by(sample=sample, sort=sort, reverse=reverse)
        params.extend([limit, 0 if sample else offset])
        rows = self._conn.execute(
            f"""
            SELECT s.session_id, s.native_id, s.origin, s.title, s.created_at_ms, s.updated_at_ms,
                   s.session_kind,
                   s.message_count, s.word_count, s.reported_duration_ms,
                   s.tool_use_count, s.thinking_count, s.paste_count,
                   s.user_message_count, s.authored_user_message_count,
                   s.assistant_message_count, s.system_message_count,
                   s.tool_message_count, s.user_word_count, s.authored_user_word_count,
                   s.assistant_word_count,
                   s.git_branch, s.git_repository_url, s.provider_project_ref,
                   COALESCE(
                       (
                           SELECT json_group_array(swd.path)
                           FROM session_working_dirs swd
                           WHERE swd.session_id = s.session_id
                           ORDER BY swd.position, swd.path
                       ),
                       '[]'
                   ) AS working_directories_json,
                   COALESCE(
                       json_group_array(st.tag) FILTER (WHERE st.tag IS NOT NULL),
                       '[]'
                   ) AS tags_json
            FROM sessions s
            LEFT JOIN {self._tags_relation} st
              ON st.session_id = s.session_id
             AND st.tag_source = 'user'
            {where}
            GROUP BY s.session_id
            {order_by}
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [_summary_from_row(row) for row in rows]

    def search_summaries(
        self,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
        sort: str | None = None,
        reverse: bool = False,
        session_id: str | None = None,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
    ) -> list[ArchiveSessionSearchHit]:
        """Search archive block text and return session-level hits with snippets."""
        match_query = normalize_fts5_query(query)
        if match_query is None:
            # Empty / whitespace / asterisk-only query: no FTS expression to
            # run. Mirror the read model lexical path and return no hits rather
            # than raising ``fts5: syntax error``.
            return []
        # A real query needs the block FTS index. Surface a degraded index as a
        # sanitized DatabaseError (→ 503 "Search index") instead of a raw
        # ``no such table`` 500 or a misleading empty-result 200.
        _ensure_messages_fts_ready(self._conn)
        where, filter_params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            tags_relation=self._tags_relation,
            prefix="AND",
        )
        where, filter_params = _with_since_session_filter(
            self._conn,
            where,
            filter_params,
            "s",
            since_session_id=since_session_id,
            prefix="AND",
        )
        if session_id is not None:
            where = f"{where} AND s.session_id = ?"
            filter_params.append(session_id)
        order_by = _search_order_by(sort=sort, reverse=reverse)
        params: list[object] = [match_query, *filter_params]
        params.extend([limit, offset])
        rows = self._conn.execute(
            f"""
            SELECT b.block_id, b.message_id, b.session_id, s.origin, s.native_id, s.title,
                   b.search_text AS fallback_text,
                   snippet(messages_fts, 4, '[', ']', '...', 12) AS snippet,
                   rank
            FROM messages_fts
            JOIN blocks b ON b.rowid = messages_fts.rowid
            JOIN sessions s ON s.session_id = b.session_id
            WHERE messages_fts MATCH ?
            {where}
            {order_by}
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [
            ArchiveSessionSearchHit(
                rank=index,
                session_id=str(row["session_id"]),
                block_id=str(row["block_id"]),
                message_id=str(row["message_id"]),
                origin=str(row["origin"]),
                provider=_provider_for_origin(str(row["origin"])),
                title=str(row["title"]) if row["title"] is not None else None,
                snippet=_highlight_search_snippet(
                    str(row["snippet"] or ""),
                    fallback=str(row["fallback_text"] or ""),
                    query=match_query,
                ),
            )
            for index, row in enumerate(rows, start=offset + 1)
        ]

    def count_search_sessions(
        self,
        query: str,
        *,
        session_id: str | None = None,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
    ) -> int:
        """Count distinct sessions matching the archive block FTS search."""
        match_query = normalize_fts5_query(query)
        if match_query is None:
            return 0
        where, filter_params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            tags_relation=self._tags_relation,
            prefix="AND",
        )
        where, filter_params = _with_since_session_filter(
            self._conn,
            where,
            filter_params,
            "s",
            since_session_id=since_session_id,
            prefix="AND",
        )
        if session_id is not None:
            where = f"{where} AND s.session_id = ?"
            filter_params.append(session_id)
        row = self._conn.execute(
            f"""
            SELECT COUNT(DISTINCT b.session_id)
            FROM messages_fts
            JOIN blocks b ON b.rowid = messages_fts.rowid
            JOIN sessions s ON s.session_id = b.session_id
            WHERE messages_fts MATCH ?
            {where}
            """,
            [match_query, *filter_params],
        ).fetchone()
        return int(row[0] if row is not None else 0)

    def semantic_summaries(
        self,
        scored_message_ids: list[tuple[str, float]],
        *,
        limit: int = 20,
        offset: int = 0,
        session_id: str | None = None,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
    ) -> list[ArchiveSessionSearchHit]:
        """Resolve vector-ranked message ids into filtered session-level hits."""
        if not scored_message_ids:
            return []
        message_ids = tuple(message_id for message_id, _score in scored_message_ids)
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        placeholders = ", ".join("?" for _ in message_ids)
        where = f"{where} AND m.message_id IN ({placeholders})" if where else f"WHERE m.message_id IN ({placeholders})"
        params.extend(message_ids)
        if session_id is not None:
            where = f"{where} AND s.session_id = ?"
            params.append(session_id)
        rows = self._conn.execute(
            f"""
            SELECT m.message_id, m.session_id, s.origin, s.native_id, s.title,
                   b.block_id, b.text
            FROM messages m
            JOIN sessions s ON s.session_id = m.session_id
            LEFT JOIN blocks b
              ON b.message_id = m.message_id
             AND b.position = (
                 SELECT MIN(position)
                 FROM blocks
                 WHERE message_id = m.message_id
                   AND text IS NOT NULL
             )
            {where}
            """,
            params,
        ).fetchall()
        rows_by_message_id = {str(row["message_id"]): row for row in rows}
        deduped: list[ArchiveSessionSearchHit] = []
        seen_sessions: set[str] = set()
        for message_id, _score in scored_message_ids:
            row = rows_by_message_id.get(message_id)
            if row is None:
                continue
            session_id = str(row["session_id"])
            if session_id in seen_sessions:
                continue
            seen_sessions.add(session_id)
            text = str(row["text"] or "")
            deduped.append(
                ArchiveSessionSearchHit(
                    rank=len(deduped) + 1,
                    session_id=session_id,
                    block_id=str(row["block_id"] or message_id),
                    message_id=message_id,
                    origin=str(row["origin"]),
                    provider=_provider_for_origin(str(row["origin"])),
                    title=str(row["title"]) if row["title"] is not None else None,
                    snippet=text[:160],
                )
            )
        page = deduped[offset : offset + limit]
        return [replace(hit, rank=offset + index) for index, hit in enumerate(page, start=1)]

    def query_messages(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveMessageQueryRow]:
        """Return message rows matching a unit-scoped query predicate."""

        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        order_direction = _query_unit_order_direction(sort_direction)
        if sort == "time":
            order_by = f"COALESCE(m.occurred_at_ms, s.sort_key_ms, 0) {order_direction}, m.message_id {order_direction}"
        else:
            order_by = "COALESCE(m.occurred_at_ms, s.sort_key_ms, 0), m.message_id"
        clause, params = _structural_predicate_clause("message", "m", predicate, session_alias="s")
        session_clause = ""
        session_params: list[object] = []
        if session_filters:
            session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
        rows = self._conn.execute(
            f"""
            SELECT
                m.message_id,
                m.session_id,
                s.origin,
                s.title,
                m.role,
                m.message_type,
                m.material_origin,
                m.occurred_at_ms,
                m.position,
                m.word_count,
                COALESCE((
                    SELECT group_concat(ordered.search_text, char(10))
                    FROM (
                        SELECT b.search_text
                        FROM blocks b
                        WHERE b.message_id = m.message_id
                          AND b.search_text IS NOT NULL
                        ORDER BY b.position, b.block_id
                    ) AS ordered
                ), '') AS text
            FROM messages m
            JOIN sessions s ON s.session_id = m.session_id
            WHERE {clause}
            {session_clause}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
            """,
            [*params, *session_params, normalized_limit, normalized_offset],
        ).fetchall()
        return [
            ArchiveMessageQueryRow(
                message_id=str(row["message_id"]),
                session_id=str(row["session_id"]),
                origin=str(row["origin"]),
                title=str(row["title"]) if row["title"] is not None else None,
                role=str(row["role"]),
                message_type=str(row["message_type"]),
                material_origin=str(row["material_origin"]),
                occurred_at_ms=int(row["occurred_at_ms"]) if row["occurred_at_ms"] is not None else None,
                position=int(row["position"]),
                word_count=int(row["word_count"]),
                text=str(row["text"] or ""),
            )
            for row in rows
        ]

    def query_unit_counts(
        self,
        unit: str,
        predicate: QueryPredicate,
        *,
        group_by: str | None = None,
        sort: Literal["count", "key"] | None = None,
        sort_direction: Literal["asc", "desc"] = "desc",
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
    ) -> list[ArchiveQueryUnitAggregateRow]:
        """Return exact grouped counts for SQL-backed terminal query units."""

        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        if unit == "assertion" and not self.user_db_path.exists():
            return []
        if unit == "assertion":
            self._attach_user_tier_if_present()

        row_alias = {
            "message": "m",
            "action": "a",
            "block": "b",
            "file": "f",
            "assertion": "a",
            "observed-event": "e",
        }.get(unit)
        if row_alias is None:
            raise ValueError(f"Query unit {unit!r} is not wired to SQL aggregate counts")
        if unit == "file":
            return self._query_file_counts(
                predicate,
                group_by=group_by,
                sort=sort,
                sort_direction=sort_direction,
                limit=normalized_limit,
                offset=normalized_offset,
                session_filters=session_filters,
            )
        active_session_filters = _session_filter_is_active(session_filters)
        needs_session = (
            unit != "observed-event"
            or active_session_filters
            or _query_unit_group_uses_session(group_by)
            or _predicate_uses_session_scope(predicate)
        )
        session_alias = "s" if needs_session else None
        group_expr = _query_unit_group_expression(unit, row_alias, group_by)
        clause, params = _structural_predicate_clause(unit, row_alias, predicate, session_alias=session_alias)
        where_clause = clause or "1=1"
        session_clause = ""
        session_params: list[object] = []
        if needs_session and active_session_filters and session_filters is not None:
            session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
        from_sql_by_unit = {
            "message": "messages m JOIN sessions s ON s.session_id = m.session_id",
            "action": "actions a JOIN sessions s ON s.session_id = a.session_id",
            "block": "blocks b JOIN sessions s ON s.session_id = b.session_id",
            "assertion": "user_tier.assertions a LEFT JOIN sessions s ON a.target_ref = 'session:' || s.session_id",
            "observed-event": "observed_events e JOIN sessions s ON s.session_id = e.session_id",
        }
        from_sql = "observed_events e" if unit == "observed-event" and not needs_session else from_sql_by_unit[unit]
        order_clause = _query_unit_aggregate_order(sort, sort_direction)
        source_where = "0=1"
        source_params: list[object] = []
        if unit == "observed-event":
            source_where, source_params = _observed_event_source_pushdown(predicate)
        prefix_sql = _OBSERVED_EVENT_RELATION_SQL.format(source_where=source_where) if unit == "observed-event" else ""
        rows = self._conn.execute(
            f"""
            {prefix_sql}
            SELECT {group_expr} AS group_key, COUNT(*) AS count
            FROM {from_sql}
            WHERE {where_clause}
            {session_clause}
            GROUP BY group_key
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?
            """,
            [*source_params, *params, *session_params, normalized_limit, normalized_offset],
        ).fetchall()
        return [
            ArchiveQueryUnitAggregateRow(
                unit=unit,
                group_by=group_by,
                group_key=str(row["group_key"]) if row["group_key"] is not None else None,
                count=int(row["count"]),
            )
            for row in rows
        ]

    def query_actions(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveActionQueryRow]:
        """Return action rows matching a unit-scoped query predicate."""

        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        order_direction = _query_unit_order_direction(sort_direction)
        if sort == "time":
            order_by = (
                f"COALESCE(m.occurred_at_ms, s.sort_key_ms, 0) {order_direction}, a.tool_use_block_id {order_direction}"
            )
        else:
            order_by = "COALESCE(m.occurred_at_ms, s.sort_key_ms, 0), a.tool_use_block_id"
        clause, params = _structural_predicate_clause("action", "a", predicate, session_alias="s")
        session_clause = ""
        session_params: list[object] = []
        if session_filters:
            session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
        rows = self._conn.execute(
            f"""
            SELECT
                a.session_id,
                a.message_id,
                s.origin,
                s.title,
                a.tool_use_block_id,
                a.tool_result_block_id,
                a.tool_name,
                a.semantic_type,
                a.tool_command,
                a.tool_path,
                m.occurred_at_ms,
                a.output_text
            FROM actions a
            JOIN sessions s ON s.session_id = a.session_id
            JOIN messages m ON m.message_id = a.message_id
            WHERE {clause}
            {session_clause}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
            """,
            [*params, *session_params, normalized_limit, normalized_offset],
        ).fetchall()
        return [_archive_action_query_row(row) for row in rows]

    def query_session_actions(
        self,
        session_ids: Sequence[str],
        *,
        limit: int = 50,
        offset: int = 0,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveActionQueryRow]:
        """Return action rows for known sessions using the session-position block index."""

        normalized_session_ids = tuple(
            dict.fromkeys(session_id.strip() for session_id in session_ids if session_id.strip())
        )
        if not normalized_session_ids:
            return []
        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        order_direction = _query_unit_order_direction(sort_direction)
        placeholders = ", ".join("?" for _ in normalized_session_ids)
        rows = self._conn.execute(
            f"""
            SELECT
                u.session_id,
                u.message_id,
                s.origin,
                s.title,
                u.block_id AS tool_use_block_id,
                r.block_id AS tool_result_block_id,
                u.tool_name,
                u.semantic_type,
                u.tool_command,
                u.tool_path,
                m.occurred_at_ms,
                r.text AS output_text
            FROM blocks u INDEXED BY idx_blocks_session_position
            JOIN sessions s ON s.session_id = u.session_id
            JOIN messages m ON m.message_id = u.message_id
            LEFT JOIN blocks r
                ON r.tool_id = u.tool_id
               AND r.session_id = u.session_id
               AND r.block_type = 'tool_result'
            WHERE u.session_id IN ({placeholders})
              AND u.block_type = 'tool_use'
            ORDER BY COALESCE(m.occurred_at_ms, s.sort_key_ms, 0) {order_direction},
                     u.block_id {order_direction}
            LIMIT ? OFFSET ?
            """,
            [*normalized_session_ids, normalized_limit, normalized_offset],
        ).fetchall()
        return [_archive_action_query_row(row) for row in rows]

    def query_files(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveFileQueryRow]:
        """Return affected file-path rows matching a unit-scoped query predicate."""

        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        order_direction = _query_unit_order_direction(sort_direction)
        if sort == "time":
            order_by = f"COALESCE(f.first_seen_ms, 0) {order_direction}, f.path {order_direction}"
        else:
            order_by = "f.path, COALESCE(f.first_seen_ms, 0)"
        clause, params = _structural_predicate_clause("file", "a", predicate, session_alias="s")
        session_clause = ""
        session_params: list[object] = []
        if session_filters:
            session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
        rows = self._conn.execute(
            f"""
            SELECT
                f.session_id,
                s.origin,
                s.title,
                f.path,
                f.action_count,
                f.first_message_id,
                f.first_tool_use_block_id,
                f.last_tool_use_block_id,
                f.first_seen_ms,
                f.last_seen_ms
            FROM (
                SELECT
                    a.session_id,
                    REPLACE(a.tool_path, char(92), '/') AS path,
                    COUNT(*) AS action_count,
                    MIN(a.message_id) AS first_message_id,
                    MIN(a.tool_use_block_id) AS first_tool_use_block_id,
                    MAX(a.tool_use_block_id) AS last_tool_use_block_id,
                    MIN(COALESCE(m.occurred_at_ms, s.sort_key_ms, 0)) AS first_seen_ms,
                    MAX(COALESCE(m.occurred_at_ms, s.sort_key_ms, 0)) AS last_seen_ms
                FROM actions a
                JOIN sessions s ON s.session_id = a.session_id
                JOIN messages m ON m.message_id = a.message_id
                WHERE a.tool_path IS NOT NULL
                AND a.tool_path != ''
                AND {clause}
                {session_clause}
                GROUP BY a.session_id, path
            ) f
            JOIN sessions s ON s.session_id = f.session_id
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
            """,
            [*params, *session_params, normalized_limit, normalized_offset],
        ).fetchall()
        return [
            ArchiveFileQueryRow(
                session_id=str(row["session_id"]),
                origin=str(row["origin"]),
                title=str(row["title"]) if row["title"] is not None else None,
                path=str(row["path"]),
                action_count=int(row["action_count"]),
                first_message_id=str(row["first_message_id"]) if row["first_message_id"] is not None else None,
                first_tool_use_block_id=str(row["first_tool_use_block_id"])
                if row["first_tool_use_block_id"] is not None
                else None,
                last_tool_use_block_id=str(row["last_tool_use_block_id"])
                if row["last_tool_use_block_id"] is not None
                else None,
                first_seen_ms=int(row["first_seen_ms"]) if row["first_seen_ms"] is not None else None,
                last_seen_ms=int(row["last_seen_ms"]) if row["last_seen_ms"] is not None else None,
            )
            for row in rows
        ]

    def _query_file_counts(
        self,
        predicate: QueryPredicate,
        *,
        group_by: str | None,
        sort: Literal["count", "key"] | None,
        sort_direction: Literal["asc", "desc"],
        limit: int,
        offset: int,
        session_filters: Mapping[str, object] | None,
    ) -> list[ArchiveQueryUnitAggregateRow]:
        group_expr = _query_unit_group_expression("file", "f", group_by)
        clause, params = _structural_predicate_clause("file", "a", predicate, session_alias="s")
        session_clause = ""
        session_params: list[object] = []
        if session_filters:
            session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
        order_clause = _query_unit_aggregate_order(sort, sort_direction)
        rows = self._conn.execute(
            f"""
            SELECT {group_expr} AS group_key, COUNT(*) AS count
            FROM (
                SELECT
                    a.session_id,
                    REPLACE(a.tool_path, char(92), '/') AS path
                FROM actions a
                JOIN sessions s ON s.session_id = a.session_id
                JOIN messages m ON m.message_id = a.message_id
                WHERE a.tool_path IS NOT NULL
                AND a.tool_path != ''
                AND {clause}
                {session_clause}
                GROUP BY a.session_id, path
            ) f
            JOIN sessions s ON s.session_id = f.session_id
            GROUP BY group_key
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?
            """,
            [*params, *session_params, limit, offset],
        ).fetchall()
        return [
            ArchiveQueryUnitAggregateRow(
                unit="file",
                group_by=group_by,
                group_key=str(row["group_key"]) if row["group_key"] is not None else None,
                count=int(row["count"]),
            )
            for row in rows
        ]

    def query_blocks(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveBlockQueryRow]:
        """Return content-block rows matching a unit-scoped query predicate."""

        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        order_direction = _query_unit_order_direction(sort_direction)
        if sort == "time":
            order_by = f"COALESCE(m.occurred_at_ms, s.sort_key_ms, 0) {order_direction}, b.block_id {order_direction}"
        else:
            order_by = "COALESCE(m.occurred_at_ms, s.sort_key_ms, 0), b.block_id"
        clause, params = _structural_predicate_clause("block", "b", predicate, session_alias="s")
        session_clause = ""
        session_params: list[object] = []
        if session_filters:
            session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
        rows = self._conn.execute(
            f"""
            SELECT
                b.block_id,
                b.message_id,
                b.session_id,
                s.origin,
                s.title,
                b.block_type,
                b.position,
                b.text,
                b.tool_name,
                b.semantic_type,
                b.tool_command,
                b.tool_path
            FROM blocks b
            JOIN sessions s ON s.session_id = b.session_id
            JOIN messages m ON m.message_id = b.message_id
            WHERE {clause}
            {session_clause}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
            """,
            [*params, *session_params, normalized_limit, normalized_offset],
        ).fetchall()
        return [
            ArchiveBlockQueryRow(
                block_id=str(row["block_id"]),
                message_id=str(row["message_id"]),
                session_id=str(row["session_id"]),
                origin=str(row["origin"]),
                title=str(row["title"]) if row["title"] is not None else None,
                block_type=str(row["block_type"]),
                position=int(row["position"]),
                text=str(row["text"]) if row["text"] is not None else None,
                tool_name=str(row["tool_name"]) if row["tool_name"] is not None else None,
                semantic_type=str(row["semantic_type"]) if row["semantic_type"] is not None else None,
                tool_command=str(row["tool_command"]) if row["tool_command"] is not None else None,
                tool_path=str(row["tool_path"]) if row["tool_path"] is not None else None,
            )
            for row in rows
        ]

    def query_assertions(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveAssertionQueryRow]:
        """Return user-tier assertion rows matching a unit-scoped predicate."""

        if not self.user_db_path.exists():
            return []
        self._attach_user_tier_if_present()
        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        order_direction = _query_unit_order_direction(sort_direction)
        if sort == "time":
            order_by = (
                f"COALESCE(a.updated_at_ms, a.created_at_ms, 0) {order_direction}, a.assertion_id {order_direction}"
            )
        else:
            order_by = "a.updated_at_ms DESC, a.assertion_id"
        clause, params = _structural_predicate_clause("assertion", "a", predicate, session_alias="s")
        session_clause = ""
        session_params: list[object] = []
        if session_filters:
            session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
        rows = self._conn.execute(
            f"""
            SELECT
                a.assertion_id,
                a.target_ref,
                a.scope_ref,
                a.kind,
                a.key,
                a.body_text,
                a.value_json,
                a.author_ref,
                a.author_kind,
                a.status,
                a.visibility,
                a.evidence_refs_json,
                a.staleness_json,
                a.context_policy_json,
                a.created_at_ms,
                a.updated_at_ms
            FROM user_tier.assertions a
            LEFT JOIN sessions s ON a.target_ref = 'session:' || s.session_id
            WHERE {clause}
            {session_clause}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
            """,
            [*params, *session_params, normalized_limit, normalized_offset],
        ).fetchall()
        return [
            ArchiveAssertionQueryRow(
                assertion_id=str(row["assertion_id"]),
                target_ref=str(row["target_ref"]),
                scope_ref=str(row["scope_ref"]) if row["scope_ref"] is not None else None,
                kind=str(row["kind"]),
                key=str(row["key"]) if row["key"] is not None else None,
                body_text=str(row["body_text"]) if row["body_text"] is not None else None,
                value=_json_value(row["value_json"], default={}),
                author_ref=str(row["author_ref"] if row["author_ref"] is not None else ASSERTION_DEFAULT_AUTHOR_REF),
                author_kind=str(
                    row["author_kind"] if row["author_kind"] is not None else ASSERTION_DEFAULT_AUTHOR_KIND
                ),
                status=str(row["status"] if row["status"] is not None else ASSERTION_DEFAULT_STATUS),
                visibility=str(row["visibility"] if row["visibility"] is not None else ASSERTION_DEFAULT_VISIBILITY),
                evidence_refs=_json_str_tuple(row["evidence_refs_json"]),
                staleness=_json_value(row["staleness_json"], default={}),
                context_policy=_json_value(row["context_policy_json"], default=ASSERTION_DEFAULT_CONTEXT_POLICY),
                created_at_ms=int(row["created_at_ms"]),
                updated_at_ms=int(row["updated_at_ms"]),
            )
            for row in rows
        ]

    def query_runs(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveRunQueryRow]:
        """Return materialized run rows matching a unit-scoped predicate."""

        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        order_direction = _query_unit_order_direction(sort_direction)
        if sort == "time":
            order_by = (
                f"COALESCE(r.source_updated_at, '') {order_direction}, "
                f"r.session_id {order_direction}, r.position {order_direction}, r.run_ref {order_direction}"
            )
        else:
            order_by = "r.session_id, r.position, r.run_ref"
        clause, params = _structural_predicate_clause("run", "r", predicate, session_alias="s")
        session_clause = ""
        session_params: list[object] = []
        if session_filters:
            session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
        rows = self._conn.execute(
            f"""
            SELECT r.session_id, s.origin, s.title, r.payload_json
            FROM session_runs r
            JOIN sessions s ON r.session_id = s.session_id
            WHERE {clause}
            {session_clause}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
            """,
            [*params, *session_params, normalized_limit, normalized_offset],
        ).fetchall()
        return [
            ArchiveRunQueryRow(
                session_id=str(row["session_id"]),
                origin=str(row["origin"]),
                title=str(row["title"]) if row["title"] is not None else None,
                run=ProjectedRun.model_validate(json.loads(row["payload_json"])),
            )
            for row in rows
        ]

    def query_observed_events(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveObservedEventQueryRow]:
        """Return observed-event rows matching a unit-scoped predicate."""

        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        order_direction = _query_unit_order_direction(sort_direction)
        if sort == "time":
            order_by = (
                f"COALESCE(e.source_updated_at, '') {order_direction}, "
                f"e.session_id {order_direction}, e.position {order_direction}, e.event_ref {order_direction}"
            )
        else:
            order_by = "e.session_id, e.position, e.event_ref"
        source_where, source_params = _observed_event_source_pushdown(predicate)
        clause, params = _structural_predicate_clause("observed-event", "e", predicate, session_alias="s")
        session_clause = ""
        session_params: list[object] = []
        if session_filters:
            session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
        rows = self._conn.execute(
            f"""
            {_OBSERVED_EVENT_RELATION_SQL.format(source_where=source_where)}
            SELECT e.*, s.origin, s.title
            FROM observed_events e
            JOIN sessions s ON e.session_id = s.session_id
            WHERE {clause}
            {session_clause}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
            """,
            [*source_params, *params, *session_params, normalized_limit, normalized_offset],
        ).fetchall()
        return [
            ArchiveObservedEventQueryRow(
                session_id=str(row["session_id"]),
                origin=str(row["origin"]),
                title=str(row["title"]) if row["title"] is not None else None,
                event=_observed_event_from_query_row(row),
            )
            for row in rows
        ]

    def query_context_snapshots(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveContextSnapshotQueryRow]:
        """Return materialized context-snapshot rows matching a unit-scoped predicate."""

        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        order_direction = _query_unit_order_direction(sort_direction)
        if sort == "time":
            order_by = (
                f"COALESCE(c.source_updated_at, '') {order_direction}, "
                f"c.session_id {order_direction}, c.position {order_direction}, c.snapshot_ref {order_direction}"
            )
        else:
            order_by = "c.session_id, c.position, c.snapshot_ref"
        clause, params = _structural_predicate_clause("context-snapshot", "c", predicate, session_alias="s")
        session_clause = ""
        session_params: list[object] = []
        if session_filters:
            session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
        rows = self._conn.execute(
            f"""
            SELECT c.session_id, s.origin, s.title, c.payload_json
            FROM session_context_snapshots c
            JOIN sessions s ON c.session_id = s.session_id
            WHERE {clause}
            {session_clause}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
            """,
            [*params, *session_params, normalized_limit, normalized_offset],
        ).fetchall()
        return [
            ArchiveContextSnapshotQueryRow(
                session_id=str(row["session_id"]),
                origin=str(row["origin"]),
                title=str(row["title"]) if row["title"] is not None else None,
                snapshot=ContextSnapshot.model_validate(json.loads(row["payload_json"])),
            )
            for row in rows
        ]

    def stats(
        self,
        *,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        session_ids: tuple[str, ...] = (),
    ) -> ArchiveStats:
        """Return archive-level stats from filtered archive index sessions."""
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        where, params = _with_session_id_filter(where, params, "s", session_ids=session_ids)
        row = self._conn.execute(
            f"""
            SELECT COUNT(*) AS total_sessions,
                   COALESCE(SUM(s.message_count), 0) AS total_messages
            FROM sessions s
            {where}
            """,
            params,
        ).fetchone()
        provider_rows = self._conn.execute(
            f"""
            SELECT s.origin, COUNT(*) AS count
            FROM sessions s
            {where}
            GROUP BY s.origin
            ORDER BY count DESC, s.origin
            """,
            params,
        ).fetchall()
        attachment_row = self._conn.execute(
            f"""
            SELECT COUNT(DISTINCT ar.attachment_id) AS total_attachments
            FROM sessions s
            JOIN attachment_refs ar ON ar.session_id = s.session_id
            {where}
            """,
            params,
        ).fetchone()
        role_row = self._conn.execute(
            f"""
            SELECT COALESCE(SUM(s.user_message_count), 0) AS user_count,
                   COALESCE(SUM(s.assistant_message_count), 0) AS assistant_count,
                   COALESCE(SUM(s.system_message_count), 0) AS system_count,
                   COALESCE(SUM(s.tool_message_count), 0) AS tool_count,
                   COALESCE(SUM(s.message_count), 0)
                     - COALESCE(SUM(s.user_message_count), 0)
                     - COALESCE(SUM(s.assistant_message_count), 0)
                     - COALESCE(SUM(s.system_message_count), 0)
                     - COALESCE(SUM(s.tool_message_count), 0) AS unknown_count
            FROM sessions s
            {where}
            """,
            params,
        ).fetchone()
        if where:
            message_type_rows = self._conn.execute(
                f"""
                SELECT m.message_type AS group_key,
                       COUNT(*) AS count
                FROM sessions s
                JOIN messages m ON m.session_id = s.session_id
                {where}
                GROUP BY m.message_type
                ORDER BY count DESC, m.message_type
                """,
                params,
            ).fetchall()
            material_origin_rows = self._conn.execute(
                f"""
                SELECT m.material_origin AS group_key,
                       COUNT(*) AS count
                FROM sessions s
                JOIN messages m ON m.session_id = s.session_id
                {where}
                GROUP BY m.material_origin
                ORDER BY count DESC, m.material_origin
                """,
                params,
            ).fetchall()
        else:
            message_type_rows = self._conn.execute(
                """
                SELECT m.message_type AS group_key,
                       COUNT(*) AS count
                FROM messages m
                GROUP BY m.message_type
                ORDER BY count DESC, m.message_type
                """
            ).fetchall()
            material_origin_rows = self._conn.execute(
                """
                SELECT m.material_origin AS group_key,
                       COUNT(*) AS count
                FROM messages m
                GROUP BY m.material_origin
                ORDER BY count DESC, m.material_origin
                """
            ).fetchall()
        return ArchiveStats(
            total_sessions=int(row["total_sessions"] or 0) if row is not None else 0,
            total_messages=int(row["total_messages"] or 0) if row is not None else 0,
            total_attachments=int(attachment_row["total_attachments"] or 0) if attachment_row is not None else 0,
            origins={str(provider_row["origin"]): int(provider_row["count"] or 0) for provider_row in provider_rows},
            role_counts={
                key: count
                for key, count in (
                    ("tool", int(role_row["tool_count"] or 0) if role_row is not None else 0),
                    ("assistant", int(role_row["assistant_count"] or 0) if role_row is not None else 0),
                    ("user", int(role_row["user_count"] or 0) if role_row is not None else 0),
                    ("system", int(role_row["system_count"] or 0) if role_row is not None else 0),
                    ("unknown", int(role_row["unknown_count"] or 0) if role_row is not None else 0),
                )
                if count > 0
            },
            message_types={str(item["group_key"] or "unknown"): int(item["count"] or 0) for item in message_type_rows},
            material_origins={
                str(item["group_key"] or "unknown"): int(item["count"] or 0) for item in material_origin_rows
            },
            db_size_bytes=self.index_db_path.stat().st_size if self.index_db_path.exists() else 0,
        )

    def stats_by(
        self,
        group_by: str,
        *,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        session_ids: tuple[str, ...] = (),
    ) -> dict[str, int]:
        """Return filtered session counts grouped by a archive dimension."""
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        where, params = _with_session_id_filter(where, params, "s", session_ids=session_ids)
        rows = self._conn.execute(_stats_by_sql(group_by, where, tags_relation=self._tags_relation), params).fetchall()
        results = {str(row["group_key"]): int(row["count"] or 0) for row in rows if row["group_key"] is not None}
        return results

    def __enter__(self) -> ArchiveStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _summary_from_row(row: sqlite3.Row) -> ArchiveSessionSummary:
    import json

    def row_int(key: str) -> int:
        try:
            value = row[key]
        except IndexError:
            return 0
        return int(value or 0)

    raw_tags = json.loads(str(row["tags_json"] or "[]"))
    tags = tuple(str(tag) for tag in raw_tags if tag is not None)
    raw_working_dirs = json.loads(str(row["working_directories_json"] or "[]"))
    working_directories = tuple(str(path) for path in raw_working_dirs if path)
    origin = str(row["origin"])
    return ArchiveSessionSummary(
        session_id=str(row["session_id"]),
        native_id=str(row["native_id"]),
        origin=origin,
        provider=_provider_for_origin(origin),
        title=str(row["title"]) if row["title"] is not None else None,
        session_kind=str(row["session_kind"] or "standard"),
        created_at=_iso_from_ms(row["created_at_ms"]),
        updated_at=_iso_from_ms(row["updated_at_ms"]),
        message_count=int(row["message_count"] or 0),
        word_count=int(row["word_count"] or 0),
        tags=tags,
        reported_duration_ms=(int(row["reported_duration_ms"]) if row["reported_duration_ms"] is not None else None),
        tool_use_count=row_int("tool_use_count"),
        thinking_count=row_int("thinking_count"),
        paste_count=row_int("paste_count"),
        user_message_count=row_int("user_message_count"),
        authored_user_message_count=row_int("authored_user_message_count"),
        assistant_message_count=row_int("assistant_message_count"),
        system_message_count=row_int("system_message_count"),
        tool_message_count=row_int("tool_message_count"),
        user_word_count=row_int("user_word_count"),
        authored_user_word_count=row_int("authored_user_word_count"),
        assistant_word_count=row_int("assistant_word_count"),
        working_directories=working_directories,
        git_branch=str(row["git_branch"]) if row["git_branch"] is not None else None,
        git_repository_url=str(row["git_repository_url"]) if row["git_repository_url"] is not None else None,
        provider_project_ref=(str(row["provider_project_ref"]) if row["provider_project_ref"] is not None else None),
    )


def _highlight_search_snippet(snippet: str, *, fallback: str, query: str) -> str:
    """Return bracket-highlighted text when contentless FTS omits markers."""
    import re

    text = snippet or fallback
    if "[" in text and "]" in text:
        return text
    terms = [term.strip('"') for term in re.findall(r'"[^"]+"|[\w.-]+', query) if term.strip('"')]
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(text):
            return str(pattern.sub(lambda match: f"[{match.group(0)}]", text, count=1))
    return text


def _summary_order_by(*, sample: bool, sort: str | None, reverse: bool) -> str:
    if sample or sort == "random":
        return "ORDER BY RANDOM()"
    direction = "ASC" if reverse else "DESC"
    if sort in {None, "date"}:
        return f"ORDER BY s.sort_key_ms IS NULL, s.sort_key_ms {direction}, s.session_id {direction}"
    if sort == "messages":
        return f"ORDER BY s.message_count {direction}, s.sort_key_ms {direction}, s.session_id {direction}"
    if sort == "words":
        return f"ORDER BY s.word_count {direction}, s.sort_key_ms {direction}, s.session_id {direction}"
    if sort == "longest":
        return f"""
            ORDER BY (
                SELECT COALESCE(MAX(m.word_count), 0)
                FROM messages m
                WHERE m.session_id = s.session_id
            ) {direction}, s.sort_key_ms {direction}, s.session_id {direction}
        """
    if sort == "tokens":
        return f"""
            ORDER BY (
                SELECT COALESCE(SUM(m.input_tokens + m.output_tokens + m.cache_read_tokens + m.cache_write_tokens), 0)
                FROM messages m
                WHERE m.session_id = s.session_id
            ) {direction}, s.sort_key_ms {direction}, s.session_id {direction}
        """
    raise ValueError("archive root query sort must be one of date, messages, words, longest, tokens, random.")


def _search_order_by(*, sort: str | None, reverse: bool) -> str:
    if sort is None:
        return "ORDER BY rank DESC" if reverse else "ORDER BY rank"
    return _summary_order_by(sample=False, sort=sort, reverse=reverse)


def _with_session_id_filter(
    where: str,
    params: list[object],
    table_alias: str,
    *,
    session_ids: tuple[str, ...],
) -> tuple[str, list[object]]:
    if not session_ids:
        return where, params
    placeholders = ", ".join("?" for _ in session_ids)
    clause = f"{table_alias}.session_id IN ({placeholders})"
    merged_params = [*params, *session_ids]
    if where:
        return f"{where} AND {clause}", merged_params
    return f"WHERE {clause}", merged_params


def _with_since_session_filter(
    conn: sqlite3.Connection,
    where: str,
    params: list[object],
    table_alias: str,
    *,
    since_session_id: str | None,
    prefix: str = "WHERE",
) -> tuple[str, list[object]]:
    if since_session_id is None:
        return where, params
    reference = _since_session_reference(conn, since_session_id)
    if reference is None:
        clause = "0 = 1"
        if where:
            return f"{where} AND {clause}", params
        return f"{prefix} {clause}", params
    ref_session_id, ref_sort_key_ms, ref_paths = reference
    clauses = [f"{table_alias}.session_id != ?"]
    merged_params: list[object] = [*params, ref_session_id]
    if ref_sort_key_ms is not None:
        clauses.append(f"{table_alias}.sort_key_ms > ?")
        merged_params.append(ref_sort_key_ms)
    if ref_paths:
        path_clauses: list[str] = []
        for ref_path in ref_paths:
            exact_prefix, child_prefix = escaped_sql_path_prefix_patterns(ref_path)
            path_clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM session_working_dirs since_cwd
                    WHERE since_cwd.session_id = {table_alias}.session_id
                      AND (
                        REPLACE(since_cwd.path, char(92), '/') = ?
                        OR REPLACE(since_cwd.path, char(92), '/') LIKE ? ESCAPE '\\'
                      )
                )
                """.strip()
            )
            merged_params.extend([exact_prefix, child_prefix])
        clauses.append("(" + " OR ".join(path_clauses) + ")")
    clause = " AND ".join(clauses)
    if where:
        return f"{where} AND {clause}", merged_params
    return f"{prefix} {clause}", merged_params


def _since_session_reference(
    conn: sqlite3.Connection,
    token: str,
) -> tuple[str, int | None, tuple[str, ...]] | None:
    lower_bound, upper_bound = session_id_prefix_bounds(token)
    prefix_clause = "s.session_id >= ?"
    prefix_params: list[str] = [lower_bound]
    if upper_bound is not None:
        prefix_clause = f"{prefix_clause} AND s.session_id < ?"
        prefix_params.append(upper_bound)
    rows = conn.execute(
        f"""
        SELECT s.session_id,
               COALESCE(
                   (SELECT MAX(m.occurred_at_ms) FROM messages m WHERE m.session_id = s.session_id),
                   s.sort_key_ms
               ) AS anchor_ms
        FROM sessions s
        WHERE s.session_id = ? OR ({prefix_clause})
        ORDER BY CASE WHEN s.session_id = ? THEN 0 ELSE 1 END, s.session_id
        LIMIT 2
        """,
        (token, *prefix_params, token),
    ).fetchall()
    if not rows:
        return None
    row = rows[0]
    session_id = str(row["session_id"])
    path_rows = conn.execute(
        """
        SELECT path
        FROM session_working_dirs
        WHERE session_id = ?
        ORDER BY position, path
        """,
        (session_id,),
    ).fetchall()
    paths = tuple(str(path_row["path"]) for path_row in path_rows if path_row["path"])
    anchor_value = row["anchor_ms"]
    return session_id, int(anchor_value) if anchor_value is not None else None, paths


def _all_session_tags_sql() -> str:
    return """
        (
            SELECT session_id, tag, tag_source, method, confidence, evidence_json
            FROM session_tags
            WHERE tag_source = 'auto'
            UNION ALL
            SELECT
                substr(target_ref, 9) AS session_id,
                COALESCE(key, body_text) AS tag,
                'user' AS tag_source,
                json_extract(value_json, '$.method') AS method,
                confidence,
                json_extract(value_json, '$.evidence') AS evidence_json
            FROM user_tier.assertions
            WHERE kind = 'tag'
              AND target_ref LIKE 'session:%'
              AND COALESCE(status, 'active') != 'deleted'
              AND COALESCE(key, body_text) IS NOT NULL
        )
    """


def _split_user_target_ref(target_ref: str) -> tuple[str, str]:
    target_type, sep, target_id = target_ref.partition(":")
    if not sep:
        return "", target_ref
    return target_type, target_id


def _id_from_target_ref(target_ref: str, prefix: str) -> str:
    return target_ref[len(prefix) :] if target_ref.startswith(prefix) else target_ref


def _active_assertion_by_kind_key(
    conn: sqlite3.Connection,
    kind: str,
    key: str,
) -> ArchiveAssertionEnvelope | None:
    for assertion in list_assertions_by_kind(conn, kind):
        if assertion.key == key:
            return assertion
    return None


def _user_mark_session_id(target_type: str, target_id: str) -> str:
    if target_type == "session":
        return target_id
    if target_type == "message":
        session_id, _sep, _message_native_id = target_id.rpartition(":")
        return session_id
    return ""


def _learning_correction_from_archive_row(row: sqlite3.Row | tuple[object, ...]) -> LearningCorrection:
    session_id = str(row[0])
    kind = parse_correction_kind(str(row[1]))
    try:
        stored = json.loads(str(row[2]))
    except json.JSONDecodeError:
        stored = {}
    if isinstance(stored, dict) and isinstance(stored.get("payload"), dict):
        payload = {str(key): str(value) for key, value in dict(stored["payload"]).items()}
        note_raw = stored.get("note")
        note = str(note_raw) if note_raw is not None else None
    elif isinstance(stored, dict):
        payload = {str(key): str(value) for key, value in stored.items()}
        note = None
    else:
        payload = {}
        note = None
    raw_updated_at_ms = row[3]
    updated_at_ms = int(str(raw_updated_at_ms or 0))
    return LearningCorrection(
        session_id=session_id,
        kind=kind,
        payload=payload,
        note=note,
        created_at=datetime.fromtimestamp(updated_at_ms / 1000.0, tz=UTC),
    )


def _origin_for_provider_value(provider: str | None) -> str | None:
    if provider is None:
        return None
    return origin_from_provider(Provider.from_string(provider)).value


def _origin_for_tool_usage_filter(provider_or_origin: str | None) -> str | None:
    if provider_or_origin is None:
        return None
    known_origin = {
        "claude-code-session",
        "codex-session",
        "gemini-cli-session",
        "hermes-session",
        "antigravity-session",
        "chatgpt-export",
        "claude-ai-export",
        "aistudio-drive",
        "unknown-export",
    }
    if provider_or_origin in known_origin:
        return provider_or_origin
    return _origin_for_provider_value(provider_or_origin)


def _tool_usage_builder_query(query: ToolUsageInsightQuery) -> ToolUsageInsightQuery:
    origin = _origin_for_tool_usage_filter(query.provider)
    updates: dict[str, object] = {"limit": None, "offset": 0}
    if origin is None:
        return query.model_copy(update=updates)
    updates["provider"] = _provider_for_origin(origin).value
    return query.model_copy(update=updates)


def _session_origin(conn: sqlite3.Connection, session_id: str) -> str:
    row = conn.execute("SELECT origin FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return str(row["origin"]) if row is not None else "unknown-export"


def _read_archive_materialization(
    conn: sqlite3.Connection,
    insight_type: str,
    session_id: str,
) -> ArchiveInsightMaterialization:
    try:
        return read_insight_materialization(conn, insight_type, session_id)
    except KeyError:
        return ArchiveInsightMaterialization(
            insight_type=insight_type,
            session_id=session_id,
            materializer_version=1,
            materialized_at_ms=0,
            source_updated_at_ms=None,
            source_sort_key_ms=None,
            input_high_water_mark_ms=None,
            input_row_count=0,
        )


def _archive_provenance(materialization: ArchiveInsightMaterialization) -> ArchiveInsightProvenance:
    return ArchiveInsightProvenance(
        materializer_version=materialization.materializer_version,
        materialized_at=_iso_from_ms(materialization.materialized_at_ms) or "1970-01-01T00:00:00Z",
        source_updated_at=_iso_from_ms(materialization.source_updated_at_ms),
        source_sort_key=(
            materialization.source_sort_key_ms / 1000.0 if materialization.source_sort_key_ms is not None else None
        ),
    )


def _archive_inference_provenance(materialization: ArchiveInsightMaterialization) -> ArchiveInferenceProvenance:
    base = _archive_provenance(materialization)
    return ArchiveInferenceProvenance(
        materializer_version=base.materializer_version,
        materialized_at=base.materialized_at,
        source_updated_at=base.source_updated_at,
        source_sort_key=base.source_sort_key,
        inference_version=materialization.materializer_version,
        inference_family="archive",
    )


def _archive_enrichment_provenance(materialization: ArchiveInsightMaterialization) -> ArchiveEnrichmentProvenance:
    base = _archive_provenance(materialization)
    return ArchiveEnrichmentProvenance(
        materializer_version=base.materializer_version,
        materialized_at=base.materialized_at,
        source_updated_at=base.source_updated_at,
        source_sort_key=base.source_sort_key,
        enrichment_version=materialization.materializer_version,
        enrichment_family="archive",
    )


def _work_event_insight_from_archive_row(
    event: ArchiveSessionWorkEvent,
    *,
    origin: str,
    materialization: ArchiveInsightMaterialization,
) -> SessionWorkEventInsight:
    evidence_payload = {
        **event.evidence,
        "start_index": event.start_index,
        "end_index": event.end_index,
        "start_time": _iso_from_ms(event.started_at_ms),
        "end_time": _iso_from_ms(event.ended_at_ms),
        "duration_ms": event.duration_ms,
        "file_paths": event.file_paths,
        "tools_used": event.tools_used,
    }
    inference_payload = {
        **event.inference,
        "heuristic_label": event.work_event_type,
        "summary": event.summary,
        "confidence": event.confidence,
        "support_level": confidence_from_score(event.confidence),
    }
    return SessionWorkEventInsight(
        event_id=event.event_id,
        session_id=event.session_id,
        source_name=_provider_for_origin(origin).value,
        event_index=event.position,
        provenance=_archive_provenance(materialization),
        inference_provenance=_archive_inference_provenance(materialization),
        evidence=WorkEventEvidencePayload.model_validate(evidence_payload),
        inference=WorkEventInferencePayload.model_validate(inference_payload),
    )


def _phase_insight_from_archive_row(
    phase: ArchiveSessionPhase,
    *,
    origin: str,
    materialization: ArchiveInsightMaterialization,
) -> SessionPhaseInsight:
    evidence_payload = {
        **phase.evidence,
        "start_time": _iso_from_ms(phase.started_at_ms),
        "end_time": _iso_from_ms(phase.ended_at_ms),
        "message_range": (phase.start_index, phase.end_index),
        "duration_ms": phase.duration_ms,
        "tool_counts": phase.tool_counts,
        "word_count": phase.word_count,
    }
    return SessionPhaseInsight(
        phase_id=phase.phase_id,
        session_id=phase.session_id,
        source_name=_provider_for_origin(origin).value,
        phase_index=phase.position,
        provenance=_archive_provenance(materialization),
        evidence=SessionPhaseEvidencePayload.model_validate(evidence_payload),
    )


@dataclass(frozen=True)
class _SessionProfileComponents:
    """Extracted session-profile payloads shared by the insight and record builders."""

    materialization: ArchiveInsightMaterialization
    evidence: SessionEvidencePayload
    inference: SessionInferencePayload
    enrichment: SessionEnrichmentPayload | None


def _session_profile_components_from_archive_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> _SessionProfileComponents:
    """Build the evidence/inference/enrichment payloads from a session profile row.

    This is the shared extraction used by both
    :func:`_session_profile_insight_from_archive_row` (tier-gated insight projection)
    and :meth:`ArchiveStore.get_session_profile_record` (full domain-record
    hydration). All three payloads are always materialized here; the insight
    builder applies tier gating on top.

    Reads the typed *_payload_json columns written by the canonical
    session-profile writer (replace_session_profiles_bulk_sync).  The legacy
    provenance_json column has been dropped from the DDL.
    """
    from polylogue.storage.sqlite.queries.mappers_insight_fallback import parse_payload_model

    session_id = str(row["session_id"])
    materialization = _read_archive_materialization(conn, "session_profile", session_id)
    workflow_shape = str(row["workflow_shape"] or "unknown")
    workflow_confidence = float(row["workflow_shape_confidence"] or 0.0)
    terminal_state = str(row["terminal_state"] or "unknown")
    terminal_confidence = float(row["terminal_state_confidence"] or 0.0)

    evidence = parse_payload_model(row, "evidence_payload_json", record_id=session_id, model=SessionEvidencePayload)
    if evidence is None:
        # Fallback for rows written before the typed-column migration: build
        # a minimal payload from the direct session/profile row columns.
        evidence = SessionEvidencePayload.model_validate(
            {
                "created_at": _iso_from_ms(row["created_at_ms"]),
                "updated_at": _iso_from_ms(row["updated_at_ms"]),
                "message_count": int(row["message_count"] or 0),
                "substantive_count": int(row["substantive_count"] or 0),
                "attachment_count": int(row["attachment_count"] or 0),
                "tool_use_count": int(row["tool_use_count"] or 0),
                "thinking_count": int(row["thinking_count"] or 0),
                "word_count": int(row["word_count"] or 0),
                "total_cost_usd": float(row["total_cost_usd"] or row["cost_usd"] or 0.0),
                "total_duration_ms": int(row["total_duration_ms"] or row["duration_ms"] or 0),
                "workflow_shape": workflow_shape,
                "workflow_shape_confidence": workflow_confidence,
                "terminal_state": terminal_state,
                "terminal_state_confidence": terminal_confidence,
                "cost_is_estimated": bool(row["cost_is_estimated"]),
                "cost_provenance": str(row["cost_provenance"] or "unknown"),
                "logical_session_id": str(row["root_session_id"] or session_id),
                "tool_calls_per_minute": float(row["tool_calls_per_minute"] or 0.0),
            }
        )

    inference = parse_payload_model(row, "inference_payload_json", record_id=session_id, model=SessionInferencePayload)
    if inference is None:
        inference = SessionInferencePayload.model_validate(
            {
                "work_event_count": int(row["work_event_count"] or 0),
                "phase_count": int(row["phase_count"] or 0),
                "engaged_duration_ms": int(row["total_duration_ms"] or row["duration_ms"] or 0),
                "engaged_minutes": float(row["total_duration_ms"] or row["duration_ms"] or 0) / 60000.0,
                "workflow_shape": workflow_shape,
                "workflow_shape_confidence": workflow_confidence,
                "terminal_state": terminal_state,
                "terminal_state_confidence": terminal_confidence,
                "support_level": confidence_from_score(max(workflow_confidence, terminal_confidence)),
            }
        )
    else:
        # The denormalized native session_profiles columns are the authoritative
        # ranking signals; reconcile the JSON-derived payload onto them so resume
        # ranking and aggregation read the queryable native columns rather than a
        # divergent payload copy.
        inference = inference.model_copy(
            update={
                "workflow_shape": workflow_shape,
                "workflow_shape_confidence": workflow_confidence,
                "terminal_state": terminal_state,
                "terminal_state_confidence": terminal_confidence,
            }
        )

    enrichment = parse_payload_model(
        row, "enrichment_payload_json", record_id=session_id, model=SessionEnrichmentPayload
    )
    return _SessionProfileComponents(
        materialization=materialization,
        evidence=evidence,
        inference=inference,
        enrichment=enrichment,
    )


def _session_profile_insight_from_archive_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    tier: str,
) -> SessionProfileInsight:
    session_id = str(row["session_id"])
    components = _session_profile_components_from_archive_row(conn, row)
    materialization = components.materialization
    include_evidence = tier in {"merged", "evidence"}
    include_inference = tier in {"merged", "inference"}
    include_enrichment = tier == "merged"
    evidence = components.evidence if include_evidence else None
    inference = components.inference if include_inference else None
    enrichment = None
    enrichment_provenance = None
    if include_enrichment and components.enrichment is not None:
        enrichment = components.enrichment
        enrichment_provenance = _archive_enrichment_provenance(materialization)
    return SessionProfileInsight(
        semantic_tier=tier,
        session_id=session_id,
        logical_session_id=str(row["root_session_id"] or session_id),
        source_name=_provider_for_origin(str(row["origin"])).value,
        title=str(row["title"]) if row["title"] is not None else None,
        provenance=_archive_provenance(materialization),
        evidence=evidence,
        inference_provenance=_archive_inference_provenance(materialization) if include_inference else None,
        inference=inference,
        enrichment_provenance=enrichment_provenance,
        enrichment=enrichment,
    )


def _session_profile_record_from_archive_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> SessionProfileRecord:
    """Build the full domain :class:`SessionProfileRecord` from a session profile row.

    Reuses the same payload extraction as the insight projection and pulls the
    materialization HWM provenance from ``read_insight_materialization`` so the
    record carries the fields ``hydrate_session_profile`` and the
    ``is_stale`` staleness check expect. The FTS-only ``*_search_text`` fields
    are not stored in the archive ``session_profiles`` row and are not read by
    ``hydrate_session_profile``; they are synthesized as non-empty strings here
    purely to satisfy the model's required-non-empty validators.
    """
    session_id = str(row["session_id"])
    components = _session_profile_components_from_archive_row(conn, row)
    materialization = components.materialization
    evidence = components.evidence
    inference = components.inference
    enrichment = components.enrichment if components.enrichment is not None else SessionEnrichmentPayload()
    logical_session_id = str(row["root_session_id"] or session_id)
    source_name = _provider_for_origin(str(row["origin"])).value
    title = str(row["title"]) if row["title"] is not None else None
    workflow_shape = str(row["workflow_shape"] or "unknown")
    materialized_at = _iso_from_ms(materialization.materialized_at_ms) or "1970-01-01T00:00:00Z"
    # search_text* are FTS-only and not consumed by hydrate_session_profile;
    # synthesize a stable non-empty string so the record validates.
    search_text = title or workflow_shape or session_id
    return SessionProfileRecord(
        session_id=SessionId(session_id),
        logical_session_id=SessionId(logical_session_id),
        materializer_version=materialization.materializer_version,
        materialized_at=materialized_at,
        source_updated_at=_iso_from_ms(materialization.source_updated_at_ms),
        source_sort_key=(
            materialization.source_sort_key_ms / 1000.0 if materialization.source_sort_key_ms is not None else None
        ),
        input_high_water_mark=_iso_from_ms(materialization.input_high_water_mark_ms),
        input_high_water_mark_source=None,
        input_row_count=materialization.input_row_count,
        source_name=source_name,
        title=title,
        first_message_at=evidence.first_message_at,
        last_message_at=evidence.last_message_at,
        canonical_session_date=evidence.canonical_session_date,
        repo_paths=evidence.repo_paths,
        repo_names=inference.repo_names,
        tags=evidence.tags,
        auto_tags=inference.auto_tags,
        message_count=int(row["message_count"] or 0),
        substantive_count=int(row["substantive_count"] or 0),
        attachment_count=int(row["attachment_count"] or 0),
        work_event_count=int(row["work_event_count"] or 0),
        phase_count=int(row["phase_count"] or 0),
        word_count=int(row["word_count"] or 0),
        tool_use_count=int(row["tool_use_count"] or 0),
        thinking_count=int(row["thinking_count"] or 0),
        total_cost_usd=evidence.total_cost_usd,
        total_duration_ms=evidence.total_duration_ms,
        engaged_duration_ms=inference.engaged_duration_ms,
        tool_active_duration_ms=evidence.tool_active_duration_ms,
        wall_duration_ms=evidence.wall_duration_ms,
        workflow_shape=workflow_shape,
        workflow_shape_confidence=float(row["workflow_shape_confidence"] or 0.0),
        terminal_state=str(row["terminal_state"] or "unknown"),
        terminal_state_confidence=float(row["terminal_state_confidence"] or 0.0),
        cost_is_estimated=bool(row["cost_is_estimated"]),
        thinking_duration_ms=evidence.thinking_duration_ms,
        output_duration_ms=evidence.output_duration_ms,
        tool_duration_ms=evidence.tool_duration_ms,
        tool_calls_per_minute=float(row["tool_calls_per_minute"] or 0.0),
        timing_provenance=evidence.timing_provenance,
        total_input_tokens=evidence.total_input_tokens,
        total_output_tokens=evidence.total_output_tokens,
        total_cache_read_tokens=evidence.total_cache_read_tokens,
        total_cache_write_tokens=evidence.total_cache_write_tokens,
        total_credit_cost=evidence.total_credit_cost,
        cost_provenance=str(row["cost_provenance"] or "unknown"),
        evidence_payload=evidence,
        inference_payload=inference,
        search_text=search_text,
        evidence_search_text=search_text,
        inference_search_text=search_text,
        enrichment_payload=enrichment,
        enrichment_search_text=search_text,
    )


def _session_cost_insight_from_archive_row(conn: sqlite3.Connection, row: sqlite3.Row) -> SessionCostInsight:
    session_id = str(row["session_id"])
    source_name = _provider_for_origin(str(row["origin"])).value
    total_usd = float(row["cost_usd"] or 0.0)
    cost_provenance = str(row["cost_provenance"] or "")
    try:
        raw_model_name = row["model_name"]
    except (IndexError, KeyError):
        raw_model_name = None
    model_name = str(raw_model_name) if raw_model_name is not None else None
    normalized_model = _normalize_model(model_name) if model_name else None
    status: CostEstimateStatus
    unavailable_reason: CostUnavailableReason | None
    provenance: tuple[str, ...]
    if total_usd > 0:
        status = "exact" if cost_provenance == "exact" else "priced"
        confidence = 1.0 if status == "exact" else (0.7 if row["cost_is_estimated"] else 0.9)
        basis = (
            CostBasisPayload(provider_reported_usd=total_usd)
            if status == "exact"
            else CostBasisPayload(catalog_priced_usd=total_usd)
        )
        missing_reasons: tuple[str, ...] = ()
        unavailable_reason = None
        provenance = ("archive_session_profiles", cost_provenance or status)
    else:
        status = "unavailable"
        confidence = 0.0
        basis = CostBasisPayload()
        missing_reasons = ("archive_profile_no_cost",)
        unavailable_reason = "no_tokens"
        provenance = ("archive_session_profiles",)
    materialization = _read_archive_materialization(conn, "session_profile", session_id)
    return SessionCostInsight(
        session_id=session_id,
        source_name=source_name,
        title=str(row["title"]) if row["title"] is not None else None,
        created_at=_iso_from_ms(row["created_at_ms"]),
        updated_at=_iso_from_ms(row["updated_at_ms"]),
        estimate=CostEstimatePayload(
            source_name=source_name,
            session_id=session_id,
            model_name=model_name,
            normalized_model=normalized_model,
            status=status,
            confidence=confidence,
            total_usd=total_usd,
            basis=basis,
            missing_reasons=missing_reasons,
            unavailable_reason=unavailable_reason,
            provenance=provenance,
        ),
        provenance=_archive_provenance(materialization),
    )


def _json_object_from_text(value: object) -> dict[str, object]:
    try:
        decoded = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _json_value(value: object, *, default: JSONValue) -> JSONValue:
    try:
        decoded = json.loads(str(value or json.dumps(default)))
    except json.JSONDecodeError:
        return default
    try:
        return require_json_value(decoded)
    except TypeError:
        return default


def _canonical_json_text(value: object) -> str:
    return json.dumps(require_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_str_tuple(value: object) -> tuple[str, ...]:
    decoded = _json_value(value, default=[])
    if not isinstance(decoded, list):
        return ()
    return tuple(str(item) for item in decoded)


def _stats_by_sql(group_by: str, where: str, *, tags_relation: str = "session_tags") -> str:
    if group_by in {"provider", "origin"}:
        return f"""
            SELECT s.origin AS group_key, COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            {where}
            GROUP BY s.origin
            ORDER BY count DESC, group_key
        """
    if group_by in {"day", "month", "year"}:
        formats = {"day": "%Y-%m-%d", "month": "%Y-%m", "year": "%Y"}
        return f"""
            SELECT strftime('{formats[group_by]}', s.sort_key_ms / 1000, 'unixepoch') AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            {where}
            GROUP BY group_key
            HAVING group_key IS NOT NULL
            ORDER BY group_key DESC
        """
    if group_by == "tag":
        return f"""
            SELECT st.tag AS group_key, COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN {tags_relation} st ON st.session_id = s.session_id
            {where}
            GROUP BY st.tag
            ORDER BY count DESC, group_key
        """
    if group_by == "role":
        return f"""
            SELECT COALESCE(NULLIF(m.role, ''), 'unknown') AS group_key,
                   COUNT(*) AS count
            FROM sessions s
            JOIN messages m ON m.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    if group_by == "message_type":
        return f"""
            SELECT COALESCE(NULLIF(m.message_type, ''), 'unknown') AS group_key,
                   COUNT(*) AS count
            FROM sessions s
            JOIN messages m ON m.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    if group_by == "material_origin":
        return f"""
            SELECT COALESCE(NULLIF(m.material_origin, ''), 'unknown') AS group_key,
                   COUNT(*) AS count
            FROM sessions s
            JOIN messages m ON m.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    if group_by == "repo":
        return f"""
            SELECT COALESCE(NULLIF(r.repo_name, ''), NULLIF(r.root_path, ''), NULLIF(r.origin_url, '')) AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN session_repos sr ON sr.session_id = s.session_id
            JOIN repos r ON r.repo_id = sr.repo_id
            {where}
            GROUP BY group_key
            HAVING group_key IS NOT NULL
            ORDER BY count DESC, group_key
        """
    if group_by == "tool":
        return f"""
            SELECT COALESCE(NULLIF(LOWER(a.tool_name), ''), 'unknown') AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN actions a ON a.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    if group_by == "action":
        return f"""
            SELECT COALESCE(NULLIF(a.semantic_type, ''), 'unknown') AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN actions a ON a.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    if group_by == "work-kind":
        return f"""
            SELECT COALESCE(NULLIF(sp.workflow_shape, ''), 'unknown') AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN session_profiles sp ON sp.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    raise ValueError(
        "Unknown group_by "
        f"{group_by!r}; expected one of: provider, origin, day, month, year, tag, role, "
        "message_type, material_origin, repo, tool, action, work-kind"
    )


def _clause_without_prefix(where: str, *, prefix: str) -> str:
    stripped = where.strip()
    marker = f"{prefix} "
    if stripped.startswith(marker):
        return stripped[len(marker) :].strip()
    return stripped


def _date_ms(value: str, *, field: str) -> int:
    parsed = parse_date(value)
    if parsed is None:
        raise ValueError(f"invalid {field}: {value}")
    return int(parsed.timestamp() * 1000)


def _field_predicate_clause(
    table_alias: str,
    predicate: QueryFieldPredicate,
    *,
    tags_relation: str,
) -> tuple[str, list[object]]:
    field = predicate.bound_field_name(context="lowering session Boolean predicates")
    values = predicate.values
    kwargs: dict[str, Any] = {}
    if field in {"id", "session"}:
        if not values:
            return "", []
        return f"{table_alias}.session_id = ?", [values[-1]]
    if field == "repo":
        kwargs["repo_names"] = values
    elif field == "project":
        kwargs["project_refs"] = values
    elif field == "origin":
        kwargs["origins"] = values
    elif field == "tag":
        kwargs["tags"] = values
    elif field == "path":
        kwargs["referenced_paths"] = values
    elif field == "cwd":
        kwargs["cwd_prefix"] = values[-1] if values else None
    elif field == "tool":
        kwargs["tool_terms"] = values
    elif field == "action":
        kwargs["action_terms"] = values
    elif field == "has":
        has_types: list[str] = []
        for value in values:
            if value == "paste":
                kwargs["has_paste"] = True
            elif value == "tools":
                kwargs["has_tool_use"] = True
            elif value == "thinking":
                kwargs["has_thinking"] = True
            else:
                has_types.append(value)
        kwargs["has_types"] = tuple(has_types)
    elif field == "title":
        kwargs["title"] = " ".join(values)
    elif field == "date":
        if values:
            session_time_expr = f"COALESCE({table_alias}.updated_at_ms, {table_alias}.created_at_ms)"
            if predicate.op == ">=":
                kwargs["since_ms"] = _date_ms(values[-1], field="date")
            elif predicate.op == ">":
                return f"{session_time_expr} > ?", [_date_ms(values[-1], field="date")]
            elif predicate.op == "<=":
                kwargs["until_ms"] = _date_ms(values[-1], field="date")
            elif predicate.op == "<":
                return f"{session_time_expr} < ?", [_date_ms(values[-1], field="date")]
            else:
                raise ValueError("unsupported Boolean query operator for date")
    elif field == "since":
        if values:
            kwargs["since_ms"] = _date_ms(values[-1], field="since")
    elif field == "until":
        if values:
            kwargs["until_ms"] = _date_ms(values[-1], field="until")
    elif count_info := COUNT_QUERY_FIELD_REGISTRY.get(field):
        return _count_predicate_clause(f"{table_alias}.{count_info.session_column}", predicate)
    elif numeric_info := NUMERIC_QUERY_FIELD_REGISTRY.get(field):
        column = numeric_info.unit_columns.get("session")
        if column is None:
            raise ValueError(f"unsupported Boolean query field: {field}")
        return _numeric_predicate_clause(f"{table_alias}.{column}", predicate)
    else:
        raise ValueError(f"unsupported Boolean query field: {field}")
    where, params = _session_filter_clause(table_alias, tags_relation=tags_relation, prefix="WHERE", **kwargs)
    return _clause_without_prefix(where, prefix="WHERE"), params


def _scoped_session_field(field: str) -> str | None:
    prefix = "session."
    if not field.startswith(prefix):
        return None
    scoped = field[len(prefix) :]
    return scoped or None


def _predicate_session_field(predicate: QueryFieldPredicate) -> str | None:
    if predicate.field_ref is not None and predicate.field_ref.scope == "session":
        return predicate.field_ref.name
    if _scoped_session_field(predicate.field) is not None:
        raise ValueError(
            f"unbound session-scoped query field predicate {predicate.field!r}; "
            "bind query predicate context before lowering structural predicates"
        )
    return None


def _predicate_uses_session_scope(predicate: QueryPredicate) -> bool:
    """Return whether a structural predicate needs the sessions alias."""

    if isinstance(predicate, QueryFieldPredicate):
        return _predicate_session_field(predicate) is not None
    if isinstance(predicate, QueryNotPredicate):
        return _predicate_uses_session_scope(predicate.child)
    if isinstance(predicate, QueryBoolPredicate):
        return any(_predicate_uses_session_scope(child) for child in predicate.children)
    return isinstance(
        predicate, QueryTextPredicate | QueryExistsPredicate | QuerySequencePredicate | QueryLineagePredicate
    )


def _in_or_equals_clause(column: str, values: tuple[str, ...], *, lower: bool = False) -> tuple[str, list[object]]:
    normalized = tuple(value.strip().lower() if lower else value.strip() for value in values if value.strip())
    if not normalized:
        return "", []
    expression = f"lower({column})" if lower else column
    if len(normalized) == 1:
        return f"{expression} = ?", [normalized[0]]
    placeholders = ", ".join("?" for _ in normalized)
    return f"{expression} IN ({placeholders})", list(normalized)


def _count_predicate_clause(column: str, predicate: QueryFieldPredicate) -> tuple[str, list[object]]:
    if not predicate.values:
        return "", []
    value = int(predicate.values[-1])
    if predicate.op == ">":
        return f"{column} > ?", [value]
    if predicate.op == ">=":
        return f"{column} >= ?", [value]
    if predicate.op == "<":
        return f"{column} < ?", [value]
    if predicate.op == "<=":
        return f"{column} <= ?", [value]
    return f"{column} = ?", [value]


def _numeric_predicate_clause(
    column: str,
    predicate: QueryFieldPredicate,
) -> tuple[str, list[object]]:
    if not predicate.values:
        return "", []
    value = int(predicate.values[-1])
    if predicate.op == ">":
        return f"{column} > ?", [value]
    if predicate.op == ">=":
        return f"{column} >= ?", [value]
    if predicate.op == "<":
        return f"{column} < ?", [value]
    if predicate.op == "<=":
        return f"{column} <= ?", [value]
    return f"{column} = ?", [value]


def _time_predicate_clause(expression: str, predicate: QueryFieldPredicate) -> tuple[str, list[object]]:
    if not predicate.values:
        return "", []
    value_ms = _date_ms(predicate.values[-1], field="time")
    if predicate.op == ">":
        return f"{expression} > ?", [value_ms]
    if predicate.op == ">=":
        return f"{expression} >= ?", [value_ms]
    if predicate.op == "<":
        return f"{expression} < ?", [value_ms]
    if predicate.op == "<=":
        return f"{expression} <= ?", [value_ms]
    raise ValueError("unsupported Boolean query operator for time")


def _query_unit_time_expression(unit: str, row_alias: str) -> str:
    if unit == "message":
        return (
            f"COALESCE({row_alias}.occurred_at_ms, "
            f"(SELECT time_sessions.sort_key_ms FROM sessions time_sessions "
            f"WHERE time_sessions.session_id = {row_alias}.session_id), 0)"
        )
    if unit in {"action", "block"}:
        return (
            f"(SELECT COALESCE(time_messages.occurred_at_ms, time_sessions.sort_key_ms, 0) "
            f"FROM messages time_messages "
            f"JOIN sessions time_sessions ON time_sessions.session_id = time_messages.session_id "
            f"WHERE time_messages.message_id = {row_alias}.message_id "
            f"LIMIT 1)"
        )
    if unit == "file":
        return f"COALESCE({row_alias}.first_seen_ms, 0)"
    if unit == "assertion":
        return f"COALESCE({row_alias}.updated_at_ms, {row_alias}.created_at_ms, 0)"
    raise ValueError(f"unsupported time predicate unit: {unit}")


def _like_clause(
    expression: str,
    values: tuple[str, ...],
    *,
    joiner: Literal["AND", "OR"] = "OR",
) -> tuple[str, list[object]]:
    normalized = tuple(value.strip().lower() for value in values if value.strip())
    if not normalized:
        return "", []
    clauses = [f"lower({expression}) LIKE ?" for _ in normalized]
    joined = f" {joiner} ".join(clauses)
    return (f"({joined})" if len(clauses) > 1 else joined), [f"%{value}%" for value in normalized]


def _message_field_predicate_clause(message_alias: str, predicate: QueryFieldPredicate) -> tuple[str, list[object]]:
    field = predicate.bound_field_name(context="lowering message predicates")
    if field == "role":
        return _in_or_equals_clause(f"{message_alias}.role", predicate.values, lower=True)
    if field == "type":
        return _in_or_equals_clause(f"{message_alias}.message_type", predicate.values, lower=True)
    if field == "words":
        return _count_predicate_clause(f"{message_alias}.word_count", predicate)
    if numeric_info := NUMERIC_QUERY_FIELD_REGISTRY.get(field):
        column = numeric_info.unit_columns.get("message")
        if column is None:
            raise ValueError(f"unsupported message predicate field: {field}")
        return _numeric_predicate_clause(f"{message_alias}.{column}", predicate)
    if field == "time":
        return _time_predicate_clause(_query_unit_time_expression("message", message_alias), predicate)
    if field in {"text", "command", "path", "output", "tool", "action"}:
        action_clause = ""
        params: list[object] = []
        if field == "text":
            block_clause, params = _like_clause("COALESCE(filter_blocks.search_text, '')", predicate.values)
            action_clause = f"""
                EXISTS (
                    SELECT 1
                    FROM blocks filter_blocks
                    WHERE filter_blocks.message_id = {message_alias}.message_id
                      AND {block_clause}
                )
            """.strip()
        elif field == "tool":
            inner_clause, params = _in_or_equals_clause("filter_actions.tool_name", predicate.values, lower=True)
            action_clause = f"""
                EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.message_id = {message_alias}.message_id
                      AND {inner_clause}
                )
            """.strip()
        elif field == "action":
            inner_clause, params = _in_or_equals_clause("filter_actions.semantic_type", predicate.values, lower=True)
            action_clause = f"""
                EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.message_id = {message_alias}.message_id
                      AND {inner_clause}
                )
            """.strip()
        else:
            action_column = {
                "command": "COALESCE(filter_actions.tool_command, '')",
                "path": "REPLACE(COALESCE(filter_actions.tool_path, ''), char(92), '/')",
                "output": "COALESCE(filter_actions.output_text, '')",
            }[field]
            inner_clause, params = _like_clause(action_column, predicate.values)
            action_clause = f"""
                EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.message_id = {message_alias}.message_id
                      AND {inner_clause}
                )
            """.strip()
        return action_clause, params
    raise ValueError(f"unsupported message predicate field: {field}")


def _action_field_predicate_clause(action_alias: str, predicate: QueryFieldPredicate) -> tuple[str, list[object]]:
    field = predicate.bound_field_name(context="lowering action predicates")
    if field == "tool":
        return _in_or_equals_clause(f"{action_alias}.tool_name", predicate.values, lower=True)
    if field in {"action", "type"}:
        return _in_or_equals_clause(f"{action_alias}.semantic_type", predicate.values, lower=True)
    if field == "time":
        return _time_predicate_clause(_query_unit_time_expression("action", action_alias), predicate)
    if field == "command":
        return _like_clause(f"COALESCE({action_alias}.tool_command, '')", predicate.values)
    if field == "path":
        return _like_clause(f"REPLACE(COALESCE({action_alias}.tool_path, ''), char(92), '/')", predicate.values)
    if field == "output":
        return _like_clause(f"COALESCE({action_alias}.output_text, '')", predicate.values)
    if field == "text":
        return _like_clause(
            f"""
            COALESCE({action_alias}.tool_name, '') || ' ' ||
            COALESCE({action_alias}.semantic_type, '') || ' ' ||
            COALESCE({action_alias}.tool_command, '') || ' ' ||
            COALESCE({action_alias}.tool_path, '') || ' ' ||
            COALESCE({action_alias}.tool_input, '') || ' ' ||
            COALESCE({action_alias}.output_text, '')
            """.strip(),
            predicate.values,
        )
    raise ValueError(f"unsupported action predicate field: {field}")


def _file_field_predicate_clause(action_alias: str, predicate: QueryFieldPredicate) -> tuple[str, list[object]]:
    field = predicate.bound_field_name(context="lowering file predicates")
    if field in {"tool", "action", "type", "command", "time"}:
        return _action_field_predicate_clause(action_alias, predicate)
    if field == "path":
        return _like_clause(f"REPLACE(COALESCE({action_alias}.tool_path, ''), char(92), '/')", predicate.values)
    if field == "text":
        return _like_clause(
            f"""
            REPLACE(COALESCE({action_alias}.tool_path, ''), char(92), '/') || ' ' ||
            COALESCE({action_alias}.tool_name, '') || ' ' ||
            COALESCE({action_alias}.semantic_type, '') || ' ' ||
            COALESCE({action_alias}.tool_command, '')
            """.strip(),
            predicate.values,
        )
    raise ValueError(f"unsupported file predicate field: {field}")


def _block_field_predicate_clause(block_alias: str, predicate: QueryFieldPredicate) -> tuple[str, list[object]]:
    field = predicate.bound_field_name(context="lowering block predicates")
    if field == "type":
        return _in_or_equals_clause(f"{block_alias}.block_type", predicate.values, lower=True)
    if field == "time":
        return _time_predicate_clause(_query_unit_time_expression("block", block_alias), predicate)
    if field == "text":
        return _like_clause(f"COALESCE({block_alias}.search_text, '')", predicate.values)
    if field == "tool":
        return _in_or_equals_clause(f"{block_alias}.tool_name", predicate.values, lower=True)
    if field in {"action", "command", "path"}:
        column = {
            "action": f"{block_alias}.semantic_type",
            "command": f"COALESCE({block_alias}.tool_command, '')",
            "path": f"REPLACE(COALESCE({block_alias}.tool_path, ''), char(92), '/')",
        }[field]
        if field == "action":
            return _in_or_equals_clause(column, predicate.values, lower=True)
        return _like_clause(column, predicate.values)
    raise ValueError(f"unsupported block predicate field: {field}")


def _assertion_field_predicate_clause(assertion_alias: str, predicate: QueryFieldPredicate) -> tuple[str, list[object]]:
    field = predicate.bound_field_name(context="lowering assertion predicates")
    if field == "time":
        return _time_predicate_clause(_query_unit_time_expression("assertion", assertion_alias), predicate)
    if field == "status":
        clause, params = _in_or_equals_clause(f"COALESCE({assertion_alias}.status, ?)", predicate.values, lower=True)
        return clause, [ASSERTION_DEFAULT_STATUS, *params]
    if field == "visibility":
        clause, params = _in_or_equals_clause(
            f"COALESCE({assertion_alias}.visibility, ?)", predicate.values, lower=True
        )
        return clause, [ASSERTION_DEFAULT_VISIBILITY, *params]
    if field == "author_kind":
        clause, params = _in_or_equals_clause(
            f"COALESCE({assertion_alias}.author_kind, ?)", predicate.values, lower=True
        )
        return clause, [ASSERTION_DEFAULT_AUTHOR_KIND, *params]
    if field in {"kind", "key"}:
        return _in_or_equals_clause(f"{assertion_alias}.{field}", predicate.values, lower=True)
    if field in {"target", "target_ref"}:
        return _like_clause(f"{assertion_alias}.target_ref", predicate.values)
    if field in {"scope", "scope_ref"}:
        return _like_clause(f"{assertion_alias}.scope_ref", predicate.values)
    if field in {"author", "author_ref"}:
        clause, params = _like_clause(f"COALESCE({assertion_alias}.author_ref, ?)", predicate.values)
        return clause, [ASSERTION_DEFAULT_AUTHOR_REF, *params]
    if field in {"text", "body"}:
        return _like_clause(f"{assertion_alias}.body_text", predicate.values)
    if field == "value":
        return _like_clause(f"{assertion_alias}.value_json", predicate.values)
    if field == "evidence":
        return _like_clause(f"{assertion_alias}.evidence_refs_json", predicate.values)
    if field == "context":
        default_context_json = json.dumps(ASSERTION_DEFAULT_CONTEXT_POLICY, sort_keys=True, separators=(",", ":"))
        clause, params = _like_clause(f"COALESCE({assertion_alias}.context_policy_json, ?)", predicate.values)
        return clause, [default_context_json, *params]
    raise ValueError(f"unsupported assertion predicate field: {field}")


def _run_field_predicate_clause(run_alias: str, predicate: QueryFieldPredicate) -> tuple[str, list[object]]:
    field = predicate.bound_field_name(context="lowering run predicates")
    if field in {"harness", "role", "status", "confidence"}:
        return _in_or_equals_clause(f"{run_alias}.{field}", predicate.values, lower=True)
    if field in {"origin", "provider_origin"}:
        return _in_or_equals_clause(f"{run_alias}.provider_origin", predicate.values, lower=True)
    if field in {"run", "run_ref"}:
        return _like_clause(f"{run_alias}.run_ref", predicate.values)
    if field in {"parent", "parent_run_ref"}:
        return _like_clause(f"{run_alias}.parent_run_ref", predicate.values)
    if field in {"agent", "agent_ref"}:
        return _like_clause(f"{run_alias}.agent_ref", predicate.values)
    if field in {"context_snapshot", "context_snapshot_ref"}:
        return _like_clause(f"{run_alias}.context_snapshot_ref", predicate.values)
    if field in {"transcript", "transcript_ref"}:
        return _like_clause(f"{run_alias}.transcript_ref", predicate.values)
    if field in {"lineage", "lineage_ref"}:
        return _like_clause(f"{run_alias}.lineage_refs_json", predicate.values)
    if field == "evidence":
        return _like_clause(f"{run_alias}.evidence_refs_json", predicate.values)
    if field == "native_session_id":
        return _like_clause(f"{run_alias}.native_session_id", predicate.values)
    if field == "native_parent_session_id":
        return _like_clause(f"{run_alias}.native_parent_session_id", predicate.values)
    if field == "cwd":
        return _like_clause(f"{run_alias}.cwd", predicate.values)
    if field in {"branch", "git_branch"}:
        return _like_clause(f"{run_alias}.git_branch", predicate.values)
    if field == "title":
        return _like_clause(f"{run_alias}.title", predicate.values)
    if field == "text":
        return _like_clause(f"{run_alias}.search_text", predicate.values)
    raise ValueError(f"unsupported run predicate field: {field}")


def _observed_event_field_predicate_clause(
    event_alias: str, predicate: QueryFieldPredicate
) -> tuple[str, list[object]]:
    field = predicate.bound_field_name(context="lowering observed-event predicates")
    if field in {"kind", "delivery_state"}:
        return _in_or_equals_clause(f"{event_alias}.{field}", predicate.values)
    if field == "tool":
        return _in_or_equals_clause(
            f"json_extract({event_alias}.payload_json, '$.tool_name')",
            predicate.values,
            lower=True,
        )
    if field == "handler":
        return _in_or_equals_clause(
            f"json_extract({event_alias}.payload_json, '$.handler_kind')",
            predicate.values,
            lower=True,
        )
    if field == "status":
        return _in_or_equals_clause(
            f"json_extract({event_alias}.payload_json, '$.status')",
            predicate.values,
            lower=True,
        )
    if field == "summary":
        return _like_clause(f"{event_alias}.summary", predicate.values)
    if field in {"subject", "subject_ref"}:
        return _like_clause(f"{event_alias}.subject_ref", predicate.values)
    if field in {"object", "object_ref"}:
        return _like_clause(f"{event_alias}.object_refs_json", predicate.values)
    if field == "evidence":
        return _like_clause(f"{event_alias}.evidence_refs_json", predicate.values)
    if field == "text":
        return _like_clause(f"{event_alias}.search_text", predicate.values)
    raise ValueError(f"unsupported observed-event predicate field: {field}")


def _context_snapshot_field_predicate_clause(
    snapshot_alias: str, predicate: QueryFieldPredicate
) -> tuple[str, list[object]]:
    field = predicate.bound_field_name(context="lowering context-snapshot predicates")
    if field in {"boundary", "inheritance_mode"}:
        return _in_or_equals_clause(f"{snapshot_alias}.{field}", predicate.values, lower=True)
    if field in {"run", "run_ref"}:
        return _like_clause(f"{snapshot_alias}.run_ref", predicate.values)
    if field in {"segment", "segment_ref"}:
        return _like_clause(f"{snapshot_alias}.segment_refs_json", predicate.values)
    if field == "evidence":
        return _like_clause(f"{snapshot_alias}.evidence_refs_json", predicate.values)
    if field == "metadata":
        return _like_clause(f"{snapshot_alias}.metadata_json", predicate.values)
    if field == "text":
        return _like_clause(f"{snapshot_alias}.search_text", predicate.values)
    raise ValueError(f"unsupported context-snapshot predicate field: {field}")


def _structural_predicate_clause(
    unit: str,
    row_alias: str,
    predicate: QueryPredicate,
    *,
    session_alias: str | None = None,
) -> tuple[str, list[object]]:
    if isinstance(predicate, QueryFieldPredicate):
        session_field = _predicate_session_field(predicate)
        if session_field is not None:
            if session_alias is None:
                raise ValueError(f"session-scoped {unit} predicate requires a session alias")
            return _field_predicate_clause(
                session_alias,
                predicate,
                tags_relation="session_tags",
            )
        if unit == "message":
            return _message_field_predicate_clause(row_alias, predicate)
        if unit == "action":
            return _action_field_predicate_clause(row_alias, predicate)
        if unit == "file":
            return _file_field_predicate_clause(row_alias, predicate)
        if unit == "block":
            return _block_field_predicate_clause(row_alias, predicate)
        if unit == "assertion":
            return _assertion_field_predicate_clause(row_alias, predicate)
        if unit == "run":
            return _run_field_predicate_clause(row_alias, predicate)
        if unit == "observed-event":
            return _observed_event_field_predicate_clause(row_alias, predicate)
        if unit == "context-snapshot":
            return _context_snapshot_field_predicate_clause(row_alias, predicate)
    if isinstance(predicate, QueryNotPredicate):
        clause, params = _structural_predicate_clause(unit, row_alias, predicate.child, session_alias=session_alias)
        return (f"NOT ({clause})" if clause else "", params)
    if isinstance(predicate, QueryBoolPredicate):
        child_clauses: list[str] = []
        merged_params: list[object] = []
        for child in predicate.children:
            clause, child_params = _structural_predicate_clause(unit, row_alias, child, session_alias=session_alias)
            if clause:
                child_clauses.append(f"({clause})")
                merged_params.extend(child_params)
        if not child_clauses:
            return "", merged_params
        joiner = " OR " if predicate.op == "or" else " AND "
        return joiner.join(child_clauses), merged_params
    if isinstance(
        predicate, QueryTextPredicate | QueryExistsPredicate | QuerySequencePredicate | QueryLineagePredicate
    ):
        if session_alias is None:
            raise ValueError(f"session-scoped {unit} predicate requires a session alias")
        return _boolean_predicate_clause(session_alias, predicate, tags_relation="session_tags")
    raise ValueError(f"unsupported nested structural predicate for {unit}: {predicate!r}")


def _exists_predicate_clause(table_alias: str, predicate: QueryExistsPredicate) -> tuple[str, list[object]]:
    if predicate.unit == "message":
        row_alias = "exists_messages"
        child_clause, params = _structural_predicate_clause(
            predicate.unit,
            row_alias,
            predicate.child,
            session_alias=table_alias,
        )
        return (
            f"""
            EXISTS (
                SELECT 1
                FROM messages {row_alias}
                WHERE {row_alias}.session_id = {table_alias}.session_id
                  AND {child_clause}
            )
            """.strip(),
            params,
        )
    if predicate.unit == "action":
        row_alias = "exists_actions"
        child_clause, params = _structural_predicate_clause(
            predicate.unit,
            row_alias,
            predicate.child,
            session_alias=table_alias,
        )
        return (
            f"""
            EXISTS (
                SELECT 1
                FROM actions {row_alias}
                WHERE {row_alias}.session_id = {table_alias}.session_id
                  AND {child_clause}
            )
            """.strip(),
            params,
        )
    if predicate.unit == "file":
        row_alias = "exists_files"
        child_clause, params = _structural_predicate_clause(
            predicate.unit,
            row_alias,
            predicate.child,
            session_alias=table_alias,
        )
        return (
            f"""
            EXISTS (
                SELECT 1
                FROM actions {row_alias}
                WHERE {row_alias}.session_id = {table_alias}.session_id
                  AND {row_alias}.tool_path IS NOT NULL
                  AND {row_alias}.tool_path != ''
                  AND {child_clause}
            )
            """.strip(),
            params,
        )
    if predicate.unit == "block":
        row_alias = "exists_blocks"
        child_clause, params = _structural_predicate_clause(
            predicate.unit,
            row_alias,
            predicate.child,
            session_alias=table_alias,
        )
        return (
            f"""
            EXISTS (
                SELECT 1
                FROM blocks {row_alias}
                WHERE {row_alias}.session_id = {table_alias}.session_id
                  AND {child_clause}
            )
            """.strip(),
            params,
        )
    if predicate.unit == "assertion":
        row_alias = "exists_assertions"
        child_clause, params = _structural_predicate_clause(
            predicate.unit,
            row_alias,
            predicate.child,
            session_alias=table_alias,
        )
        return (
            f"""
            EXISTS (
                SELECT 1
                FROM user_tier.assertions {row_alias}
                WHERE {row_alias}.target_ref = 'session:' || {table_alias}.session_id
                  AND {child_clause}
            )
            """.strip(),
            params,
        )
    if predicate.unit in {"run", "observed-event", "context-snapshot"}:
        run_projection_tables = {
            "run": ("session_runs", "exists_runs"),
            "observed-event": ("session_observed_events", "exists_observed_events"),
            "context-snapshot": ("session_context_snapshots", "exists_context_snapshots"),
        }
        table_name, row_alias = run_projection_tables[predicate.unit]
        child_clause, params = _structural_predicate_clause(
            predicate.unit,
            row_alias,
            predicate.child,
            session_alias=table_alias,
        )
        return (
            f"""
            EXISTS (
                SELECT 1
                FROM {table_name} {row_alias}
                WHERE {row_alias}.session_id = {table_alias}.session_id
                  AND {child_clause}
            )
            """.strip(),
            params,
        )
    raise ValueError(f"unsupported structural query unit: {predicate.unit}")


def _fts_predicate_clause(table_alias: str, predicate: QueryTextPredicate) -> tuple[str, list[object]]:
    match_query = normalize_fts5_query(predicate.text)
    if match_query is None:
        raise ValueError("FTS predicate requires non-empty text")
    return (
        f"""
        EXISTS (
            SELECT 1
            FROM messages_fts
            JOIN blocks filter_fts_blocks
              ON filter_fts_blocks.rowid = messages_fts.rowid
            WHERE filter_fts_blocks.session_id = {table_alias}.session_id
              AND messages_fts MATCH ?
        )
        """.strip(),
        [match_query],
    )


def _lineage_predicate_clause(table_alias: str, predicate: QueryLineagePredicate) -> tuple[str, list[object]]:
    seed_session_id = predicate.seed_session_id.strip()
    if not seed_session_id:
        raise ValueError("lineage predicate requires a session id")
    return (
        f"""
        COALESCE({table_alias}.root_session_id, {table_alias}.session_id) = (
            SELECT COALESCE(seed.root_session_id, seed.session_id)
            FROM sessions seed
            WHERE seed.session_id = ?
        )
        """.strip(),
        [seed_session_id],
    )


def _boolean_predicate_clause(
    table_alias: str,
    predicate: QueryPredicate,
    *,
    tags_relation: str,
) -> tuple[str, list[object]]:
    if isinstance(predicate, QueryFieldPredicate):
        return _field_predicate_clause(table_alias, predicate, tags_relation=tags_relation)
    if isinstance(predicate, QueryExistsPredicate):
        return _exists_predicate_clause(table_alias, predicate)
    if isinstance(predicate, QuerySequencePredicate):
        if len(predicate.steps) < 2:
            raise ValueError("action sequence predicates require at least two steps")
        return _action_sequence_steps_clause(table_alias, predicate.steps)
    if isinstance(predicate, QueryTextPredicate):
        return _fts_predicate_clause(table_alias, predicate)
    if isinstance(predicate, QueryLineagePredicate):
        return _lineage_predicate_clause(table_alias, predicate)
    if isinstance(predicate, QueryNotPredicate):
        clause, params = _boolean_predicate_clause(table_alias, predicate.child, tags_relation=tags_relation)
        return (f"NOT ({clause})" if clause else "", params)
    if isinstance(predicate, QueryBoolPredicate):
        child_clauses: list[str] = []
        merged_params: list[object] = []
        for child in predicate.children:
            clause, child_params = _boolean_predicate_clause(table_alias, child, tags_relation=tags_relation)
            if clause:
                child_clauses.append(f"({clause})")
                merged_params.extend(child_params)
        if not child_clauses:
            return "", merged_params
        joiner = " OR " if predicate.op == "or" else " AND "
        return joiner.join(child_clauses), merged_params
    raise TypeError(f"unsupported Boolean query predicate: {predicate!r}")


def _session_filter_clause(
    table_alias: str,
    *,
    origin: str | None = None,
    origins: tuple[str, ...] = (),
    excluded_origins: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    excluded_tags: tuple[str, ...] = (),
    repo_names: tuple[str, ...] = (),
    project_refs: tuple[str, ...] = (),
    has_types: tuple[str, ...] = (),
    has_tool_use: bool = False,
    has_thinking: bool = False,
    has_paste: bool = False,
    tool_terms: tuple[str, ...] = (),
    excluded_tool_terms: tuple[str, ...] = (),
    action_terms: tuple[str, ...] = (),
    excluded_action_terms: tuple[str, ...] = (),
    action_sequence: tuple[str, ...] = (),
    action_text_terms: tuple[str, ...] = (),
    referenced_paths: tuple[str, ...] = (),
    cwd_prefix: str | None = None,
    typed_only: bool = False,
    message_type: str | None = None,
    title: str | None = None,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
    max_words: int | None = None,
    since_ms: int | None = None,
    until_ms: int | None = None,
    boolean_predicate: QueryPredicate | None = None,
    tags_relation: str = "session_tags",
    prefix: str = "WHERE",
) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if origin is not None:
        clauses.append(f"{table_alias}.origin = ?")
        params.append(origin)
    if origins:
        placeholders = ", ".join("?" for _ in origins)
        clauses.append(f"{table_alias}.origin IN ({placeholders})")
        params.extend(origins)
    if excluded_origins:
        placeholders = ", ".join("?" for _ in excluded_origins)
        clauses.append(f"{table_alias}.origin NOT IN ({placeholders})")
        params.extend(excluded_origins)
    if tags:
        placeholders = ", ".join("?" for _ in tags)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM {tags_relation} filter_tags
                WHERE filter_tags.session_id = {table_alias}.session_id
                  AND filter_tags.tag IN ({placeholders})
            )
            """.strip()
        )
        params.extend(tags)
    if excluded_tags:
        placeholders = ", ".join("?" for _ in excluded_tags)
        clauses.append(
            f"""
            NOT EXISTS (
                SELECT 1
                FROM {tags_relation} excluded_filter_tags
                WHERE excluded_filter_tags.session_id = {table_alias}.session_id
                  AND excluded_filter_tags.tag IN ({placeholders})
            )
            """.strip()
        )
        params.extend(excluded_tags)
    if repo_names:
        placeholders = ", ".join("?" for _ in repo_names)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM session_repos filter_session_repos
                JOIN repos filter_repos
                  ON filter_repos.repo_id = filter_session_repos.repo_id
                WHERE filter_session_repos.session_id = {table_alias}.session_id
                  AND filter_repos.repo_name IN ({placeholders})
            )
            """.strip()
        )
        params.extend(repo_names)
    if project_refs:
        project_refs = expand_project_refs(project_refs)
        placeholders = ", ".join("?" for _ in project_refs)
        clauses.append(f"{table_alias}.provider_project_ref IN ({placeholders})")
        params.extend(project_refs)
    if has_types:
        placeholders = ", ".join("?" for _ in has_types)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM blocks filter_blocks
                WHERE filter_blocks.session_id = {table_alias}.session_id
                  AND filter_blocks.block_type IN ({placeholders})
            )
            """.strip()
        )
        params.extend(has_types)
    if has_tool_use:
        clauses.append(f"{table_alias}.tool_use_count > 0")
    if has_thinking:
        clauses.append(f"{table_alias}.thinking_count > 0")
    if has_paste:
        clauses.append(f"{table_alias}.paste_count > 0")
    if typed_only:
        clauses.append(f"{table_alias}.paste_count = 0")
    for term in tool_terms:
        normalized = term.strip().lower()
        if not normalized:
            continue
        if normalized == "none":
            clauses.append(
                f"NOT EXISTS (SELECT 1 FROM actions filter_actions WHERE filter_actions.session_id = {table_alias}.session_id)"
            )
        else:
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.session_id = {table_alias}.session_id
                      AND lower(filter_actions.tool_name) = ?
                )
                """.strip()
            )
            params.append(normalized)
    for term in excluded_tool_terms:
        normalized = term.strip().lower()
        if not normalized:
            continue
        if normalized == "none":
            clauses.append(
                f"EXISTS (SELECT 1 FROM actions filter_actions WHERE filter_actions.session_id = {table_alias}.session_id)"
            )
        else:
            clauses.append(
                f"""
                NOT EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.session_id = {table_alias}.session_id
                      AND lower(filter_actions.tool_name) = ?
                )
                """.strip()
            )
            params.append(normalized)
    for term in action_terms:
        normalized = term.strip().lower()
        if not normalized:
            continue
        if normalized == "none":
            clauses.append(
                f"NOT EXISTS (SELECT 1 FROM actions filter_actions WHERE filter_actions.session_id = {table_alias}.session_id)"
            )
        else:
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.session_id = {table_alias}.session_id
                      AND filter_actions.semantic_type = ?
                )
                """.strip()
            )
            params.append(normalized)
    for term in excluded_action_terms:
        normalized = term.strip().lower()
        if not normalized:
            continue
        if normalized == "none":
            clauses.append(
                f"EXISTS (SELECT 1 FROM actions filter_actions WHERE filter_actions.session_id = {table_alias}.session_id)"
            )
        else:
            clauses.append(
                f"""
                NOT EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.session_id = {table_alias}.session_id
                      AND filter_actions.semantic_type = ?
                )
                """.strip()
            )
            params.append(normalized)
    if action_sequence:
        sequence_clause, sequence_params = _action_sequence_clause(table_alias, action_sequence)
        clauses.append(sequence_clause)
        params.extend(sequence_params)
    for term in action_text_terms:
        normalized = term.strip().lower()
        if not normalized:
            continue
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM actions filter_actions
                WHERE filter_actions.session_id = {table_alias}.session_id
                  AND lower(
                      COALESCE(filter_actions.tool_name, '') || ' ' ||
                      COALESCE(filter_actions.semantic_type, '') || ' ' ||
                      COALESCE(filter_actions.tool_command, '') || ' ' ||
                      COALESCE(filter_actions.tool_path, '') || ' ' ||
                      COALESCE(filter_actions.tool_input, '') || ' ' ||
                      COALESCE(filter_actions.output_text, '')
                  ) LIKE ?
            )
            """.strip()
        )
        params.append(f"%{normalized}%")
    for term in referenced_paths:
        normalized = term.strip().replace("\\", "/").lower()
        if not normalized:
            continue
        escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM actions filter_actions
                WHERE filter_actions.session_id = {table_alias}.session_id
                  AND REPLACE(LOWER(COALESCE(filter_actions.tool_path, '')), char(92), '/') LIKE ? ESCAPE '\\'
            )
            """.strip()
        )
        params.append(f"%{escaped}%")
    if cwd_prefix:
        exact_prefix, child_prefix = escaped_sql_path_prefix_patterns(cwd_prefix)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM session_working_dirs filter_cwd
                WHERE filter_cwd.session_id = {table_alias}.session_id
                  AND (
                    REPLACE(filter_cwd.path, char(92), '/') = ?
                    OR REPLACE(filter_cwd.path, char(92), '/') LIKE ? ESCAPE '\\'
                  )
            )
            """.strip()
        )
        params.extend([exact_prefix, child_prefix])
    if message_type:
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM messages filter_messages
                WHERE filter_messages.session_id = {table_alias}.session_id
                  AND filter_messages.message_type = ?
            )
            """.strip()
        )
        params.append(message_type)
    if title:
        clauses.append(f"{table_alias}.title LIKE ? ESCAPE '\\'")
        escaped_title = title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params.append(f"%{escaped_title}%")
    if min_messages is not None:
        clauses.append(f"{table_alias}.message_count >= ?")
        params.append(min_messages)
    if max_messages is not None:
        clauses.append(f"{table_alias}.message_count <= ?")
        params.append(max_messages)
    if min_words is not None:
        clauses.append(f"{table_alias}.word_count >= ?")
        params.append(min_words)
    if max_words is not None:
        clauses.append(f"{table_alias}.word_count <= ?")
        params.append(max_words)
    if since_ms is not None:
        clauses.append(f"COALESCE({table_alias}.updated_at_ms, {table_alias}.created_at_ms) >= ?")
        params.append(since_ms)
    if until_ms is not None:
        clauses.append(f"COALESCE({table_alias}.updated_at_ms, {table_alias}.created_at_ms) <= ?")
        params.append(until_ms)
    if boolean_predicate is not None:
        boolean_clause, boolean_params = _boolean_predicate_clause(
            table_alias,
            boolean_predicate,
            tags_relation=tags_relation,
        )
        if boolean_clause:
            clauses.append(f"({boolean_clause})")
            params.extend(boolean_params)
    if not clauses:
        return "", params
    return f"{prefix} " + " AND ".join(clauses), params


def _action_sequence_clause(table_alias: str, action_sequence: tuple[str, ...]) -> tuple[str, list[object]]:
    steps = tuple(
        QueryFieldPredicate(field="action", values=(term,), op="=").with_field_ref(
            QueryFieldRef(scope="unit", name="action", source_name="action", unit="action")
        )
        for term in action_sequence
        if term.strip()
    )
    return _action_sequence_steps_clause(table_alias, steps)


def _action_sequence_steps_clause(table_alias: str, steps: tuple[QueryPredicate, ...]) -> tuple[str, list[object]]:
    joins: list[str] = []
    predicates: list[str] = []
    params: list[object] = []
    for index, step in enumerate(steps):
        action_alias = f"seq_a{index}"
        message_alias = f"seq_m{index}"
        block_alias = f"seq_b{index}"
        joins.append(
            f"""
            JOIN actions {action_alias}
              ON {action_alias}.session_id = {table_alias}.session_id
            JOIN messages {message_alias}
              ON {message_alias}.message_id = {action_alias}.message_id
            JOIN blocks {block_alias}
              ON {block_alias}.block_id = {action_alias}.tool_use_block_id
            """.strip()
        )
        step_clause, step_params = _structural_predicate_clause("action", action_alias, step)
        if step_clause:
            predicates.append(f"({step_clause})")
            params.extend(step_params)
        if index > 0:
            predicates.append(_action_after_predicate(index - 1, index))
    sql = (
        "EXISTS ("
        "SELECT 1 FROM sessions sequence_root "
        f"{' '.join(joins)} "
        f"WHERE sequence_root.session_id = {table_alias}.session_id "
        f"AND {' AND '.join(predicates)}"
        ")"
    )
    return sql, params


def _action_after_predicate(previous: int, current: int) -> str:
    prev_message = f"seq_m{previous}"
    curr_message = f"seq_m{current}"
    prev_block = f"seq_b{previous}"
    curr_block = f"seq_b{current}"
    return (
        "("
        f"{curr_message}.position > {prev_message}.position "
        f"OR ({curr_message}.position = {prev_message}.position "
        f"AND {curr_message}.variant_index > {prev_message}.variant_index) "
        f"OR ({curr_message}.position = {prev_message}.position "
        f"AND {curr_message}.variant_index = {prev_message}.variant_index "
        f"AND {curr_block}.position > {prev_block}.position)"
        ")"
    )


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    return _count_scalar(conn, f"SELECT COUNT(*) FROM {table}")


def _count_scalar(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _ensure_messages_fts_ready(conn: sqlite3.Connection) -> None:
    """Raise ``DatabaseError`` unless message FTS is built and complete.

    Mirrors the archive FTS readiness contract for the split-
    file archive: a missing ``messages_fts`` virtual table means the search index
    was never built, and an FTS row count below the text-bearing block count
    means a bulk write suspended the triggers and never restored them. Both are
    reported as a sanitized ``DatabaseError`` so the reader degrades to a 503
    "Search index" response instead of surfacing a raw ``no such table`` /
    empty-result 200.
    """
    from polylogue.maintenance.targets import build_maintenance_target_catalog
    from polylogue.storage.fts.fts_lifecycle import check_fts_readiness, message_fts_search_readiness_sync

    repair_hint = build_maintenance_target_catalog().repair_hint(("dangling_fts",), include_run_all=True)
    check_fts_readiness(message_fts_search_readiness_sync(conn), repair_hint)


def _epoch_ms_from_iso(value: object) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


# Insights whose provenance is tracked in the ``insight_materialization`` ledger
# (materializer version + source high-water mark). Threads use a separate version
# namespace and are intentionally excluded from the version-compatibility check.
_INSIGHT_MATERIALIZATION_TYPE: dict[str, str] = {
    "session_profiles": "session_profile",
    "session_work_events": "work_events",
    "session_phases": "phases",
}

# Insights whose #1278 fallback markers are stored as JSON arrays inside payload
# columns: (table_name, ((column, json_path), ...)). Session profiles carry the
# inference and enrichment fallback reasons under ``$.fallback_reasons`` in their
# respective ``inference_payload_json`` / ``enrichment_payload_json`` columns.
_INSIGHT_FALLBACK_PAYLOAD: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "session_profiles": (
        "session_profiles",
        (
            ("inference_payload_json", "$.fallback_reasons"),
            ("enrichment_payload_json", "$.fallback_reasons"),
        ),
    ),
}


def _archive_insight_readiness_verdict(
    *,
    table_present: bool,
    row_count: int,
    expected_row_count: int | None,
    missing_count: int,
    stale_count: int,
    orphan_count: int,
    incompatible_count: int,
    degraded_count: int,
    ready_flags: dict[str, bool],
    total_sessions: int,
) -> InsightReadinessVerdict:
    if not table_present:
        return "missing"
    if incompatible_count:
        return "incompatible"
    if stale_count or orphan_count:
        return "stale"
    if missing_count or (expected_row_count is not None and row_count < expected_row_count):
        return "partial"
    if row_count == 0:
        # An empty archive (no sessions at all) reports every surface as empty.
        # In a populated archive a surface with 0 expected rows is vacuously
        # ready (e.g. no tags to roll up); a surface that should hold rows was
        # already caught by the partial branch above.
        if total_sessions > 0 and expected_row_count == 0:
            return "ready"
        return "empty"
    if degraded_count:
        return "degraded"
    if ready_flags and all(ready_flags.values()):
        return "ready"
    if not ready_flags:
        return "ready"
    return "unknown"


def _insight_readiness_aggregate_verdict(entries: tuple[InsightReadinessEntry, ...]) -> InsightReadinessVerdict:
    verdicts = {entry.verdict for entry in entries}
    for verdict in ("incompatible", "stale", "partial", "missing", "degraded", "unknown", "empty"):
        if verdict in verdicts:
            return verdict
    return "ready"


def _archive_insight_readiness_evidence(
    *,
    row_count: int,
    expected_row_count: int | None,
    missing_count: int,
    stale_count: int,
    orphan_count: int,
    incompatible_count: int,
    degraded_count: int,
    fallback_reason_counts: dict[str, int],
    ready_flags: dict[str, bool],
) -> tuple[str, ...]:
    values = [f"rows={row_count}"]
    if expected_row_count is not None:
        values.append(f"expected={expected_row_count}")
    if missing_count:
        values.append(f"missing={missing_count}")
    if stale_count:
        values.append(f"stale={stale_count}")
    if orphan_count:
        values.append(f"orphan={orphan_count}")
    if incompatible_count:
        values.append(f"incompatible={incompatible_count}")
    if degraded_count:
        values.append(f"degraded={degraded_count}")
    values.extend(f"fallback_reason={reason}={count}" for reason, count in fallback_reason_counts.items())
    values.extend(f"{key}={value}" for key, value in sorted(ready_flags.items()))
    return tuple(values)


def _provider_coverage_from_archive_row(row: sqlite3.Row) -> ArchiveCoverageInsight:
    session_count = int(row["session_count"] or 0)
    message_count = int(row["message_count"] or 0)
    user_message_count = int(row["user_message_count"] or 0)
    authored_user_message_count = int(row["authored_user_message_count"] or 0)
    assistant_message_count = int(row["assistant_message_count"] or 0)
    user_word_sum = int(row["user_word_sum"] or 0)
    authored_user_word_sum = int(row["authored_user_word_sum"] or 0)
    assistant_word_sum = int(row["assistant_word_sum"] or 0)
    sessions_with_tools = int(row["sessions_with_tools"] or 0)
    sessions_with_thinking = int(row["sessions_with_thinking"] or 0)
    origin = str(row["origin"])
    source_name = _provider_for_origin(origin).value
    return ArchiveCoverageInsight(
        group_by="provider",
        bucket=source_name,
        source_name=source_name,
        session_count=session_count,
        message_count=message_count,
        user_message_count=user_message_count,
        authored_user_message_count=authored_user_message_count,
        assistant_message_count=assistant_message_count,
        avg_messages_per_session=(message_count / session_count if session_count else 0.0),
        avg_user_words=(user_word_sum / user_message_count if user_message_count else 0.0),
        avg_authored_user_words=(
            authored_user_word_sum / authored_user_message_count if authored_user_message_count else 0.0
        ),
        avg_assistant_words=(assistant_word_sum / assistant_message_count if assistant_message_count else 0.0),
        tool_use_count=int(row["tool_use_count"] or 0),
        thinking_count=int(row["thinking_count"] or 0),
        total_sessions_with_tools=sessions_with_tools,
        total_sessions_with_thinking=sessions_with_thinking,
        tool_use_percentage=(sessions_with_tools / session_count) * 100 if session_count else 0.0,
        thinking_percentage=((sessions_with_thinking / session_count) * 100 if session_count else 0.0),
    )


def _archive_debt(
    *,
    name: str,
    category: str,
    issue_count: int,
    detail: str,
    destructive: bool = False,
) -> ArchiveDebtInsight:
    return ArchiveDebtInsight(
        debt_name=name,
        category=category,
        maintenance_target=name,
        destructive=destructive,
        issue_count=issue_count,
        healthy=issue_count == 0,
        detail=detail,
    )


def _archive_messages_fts_debt(conn: sqlite3.Connection) -> ArchiveDebtInsight:
    text_blocks = _count_scalar(conn, "SELECT COUNT(*) FROM blocks WHERE search_text != ''")
    fts_rows = _count_scalar(conn, "SELECT COUNT(*) FROM messages_fts")
    issue_count = abs(text_blocks - fts_rows)
    detail = "archive message FTS synchronized" if issue_count == 0 else f"{issue_count:,} message FTS row mismatch"
    return _archive_debt(
        name="archive_messages_fts",
        category="derived_repair",
        issue_count=issue_count,
        detail=detail,
    )


def _archive_profile_rows_debt(conn: sqlite3.Connection) -> ArchiveDebtInsight:
    missing = _count_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM sessions AS s
        WHERE NOT EXISTS (
            SELECT 1 FROM session_profiles AS p WHERE p.session_id = s.session_id
        )
        """,
    )
    orphaned = _count_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM session_profiles AS p
        WHERE NOT EXISTS (
            SELECT 1 FROM sessions AS s WHERE s.session_id = p.session_id
        )
        """,
    )
    issue_count = missing + orphaned
    detail = (
        "archive session profile rows complete"
        if issue_count == 0
        else f"{missing:,} missing and {orphaned:,} orphaned archive session profile rows"
    )
    return _archive_debt(
        name="archive_session_profile_rows",
        category="derived_repair",
        issue_count=issue_count,
        detail=detail,
    )


def _archive_profile_counts_debt(conn: sqlite3.Connection) -> ArchiveDebtInsight:
    work_event_mismatch = _count_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM session_profiles AS p
        WHERE p.work_event_count != (
            SELECT COUNT(*) FROM session_work_events AS e WHERE e.session_id = p.session_id
        )
        """,
    )
    phase_mismatch = _count_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM session_profiles AS p
        WHERE p.phase_count != (
            SELECT COUNT(*) FROM session_phases AS ph WHERE ph.session_id = p.session_id
        )
        """,
    )
    issue_count = work_event_mismatch + phase_mismatch
    detail = (
        "archive profile derived counts match timeline rows"
        if issue_count == 0
        else f"{work_event_mismatch:,} work-event and {phase_mismatch:,} phase count mismatches"
    )
    return _archive_debt(
        name="archive_profile_counts",
        category="derived_repair",
        issue_count=issue_count,
        detail=detail,
    )


def _archive_materialization_debt(conn: sqlite3.Connection) -> ArchiveDebtInsight:
    missing = _archive_missing_materialization_counts(conn)
    issue_count = sum(missing.values())
    detail = (
        "archive insight materialization rows complete"
        if issue_count == 0
        else "missing archive materialization rows: "
        + ", ".join(f"{key}={value}" for key, value in sorted(missing.items()) if value)
    )
    return _archive_debt(
        name="archive_insight_materialization",
        category="derived_repair",
        issue_count=issue_count,
        detail=detail,
    )


def _archive_source_raw_link_debt(conn: sqlite3.Connection, source_db_path: Path) -> ArchiveDebtInsight:
    raw_links = _count_scalar(conn, "SELECT COUNT(*) FROM sessions WHERE raw_id IS NOT NULL")
    if not source_db_path.exists():
        issue_count = raw_links
        detail = (
            "archive sessions have no source raw links"
            if raw_links == 0
            else f"source.db missing while {raw_links:,} sessions carry raw_id links"
        )
        return _archive_debt(
            name="archive_source_raw_links",
            category="source_ingest",
            issue_count=issue_count,
            detail=detail,
        )
    source_uri = f"file:{source_db_path}?mode=ro"
    conn.execute("ATTACH DATABASE ? AS source_debt", (source_uri,))
    try:
        missing = _count_scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM sessions AS s
            WHERE s.raw_id IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM source_debt.raw_sessions AS r WHERE r.raw_id = s.raw_id
              )
            """,
        )
    finally:
        conn.execute("DETACH DATABASE source_debt")
    detail = "archive source raw links resolve" if missing == 0 else f"{missing:,} sessions reference missing raw rows"
    return _archive_debt(
        name="archive_source_raw_links",
        category="source_ingest",
        issue_count=missing,
        detail=detail,
    )


def _archive_user_overlay_debt(conn: sqlite3.Connection, user_db_path: Path) -> ArchiveDebtInsight:
    if not user_db_path.exists():
        return _archive_debt(
            name="archive_user_overlay_orphans",
            category="archive_cleanup",
            issue_count=0,
            detail="archive user tier absent; no overlay orphan check needed",
        )
    conn.execute("ATTACH DATABASE ? AS user_debt", (f"file:{user_db_path}?mode=ro",))
    try:
        checks = (
            "SELECT COUNT(*) FROM user_debt.assertions u "
            "WHERE u.target_ref LIKE 'session:%' "
            "AND COALESCE(u.status, '') != 'deleted' "
            "AND NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = substr(u.target_ref, 9))",
            "SELECT COUNT(*) FROM user_debt.assertions u "
            "WHERE u.kind IN ('mark', 'annotation', 'note', 'suppression') "
            "AND u.target_ref LIKE 'message:%' "
            "AND COALESCE(u.status, '') != 'deleted' "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM sessions s "
            "  WHERE substr(u.target_ref, 9) = s.session_id "
            "     OR (substr(substr(u.target_ref, 9), 1, length(s.session_id)) = s.session_id "
            "         AND substr(substr(u.target_ref, 9), length(s.session_id) + 1, 1) = ':')"
            ")",
            "SELECT COUNT(*) FROM user_debt.assertions u "
            "WHERE u.kind = 'correction' "
            "AND u.target_ref LIKE 'insight:%' "
            "AND COALESCE(u.status, '') != 'deleted' "
            "AND NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = substr(u.target_ref, 9))",
        )
        issue_count = sum(_count_scalar(conn, sql) for sql in checks)
    finally:
        conn.execute("DETACH DATABASE user_debt")
    detail = (
        "archive user overlays resolve to index sessions"
        if issue_count == 0
        else f"{issue_count:,} archive user overlay rows reference missing sessions"
    )
    return _archive_debt(
        name="archive_user_overlay_orphans",
        category="archive_cleanup",
        issue_count=issue_count,
        detail=detail,
    )


def _session_latency_profile_from_archive_row(
    conn: sqlite3.Connection, row: sqlite3.Row
) -> SessionLatencyProfileInsight:
    session_id = str(row["session_id"])
    response_rows = conn.execute(
        """
        SELECT role, occurred_at_ms
        FROM messages
        WHERE session_id = ?
          AND occurred_at_ms IS NOT NULL
          AND role IN ('user', 'assistant')
        ORDER BY position, variant_index
        """,
        (session_id,),
    ).fetchall()
    agent_response_ms: list[int] = []
    user_response_ms: list[int] = []
    previous_role: str | None = None
    previous_at: int | None = None
    for message in response_rows:
        role = str(message["role"])
        occurred_at = int(message["occurred_at_ms"])
        if previous_role is not None and previous_at is not None:
            delta_ms = max(occurred_at - previous_at, 0)
            if previous_role == "user" and role == "assistant":
                agent_response_ms.append(delta_ms)
            elif previous_role == "assistant" and role == "user" and delta_ms <= 1_800_000:
                user_response_ms.append(delta_ms)
        previous_role = role
        previous_at = occurred_at
    tool_counts = _latency_tool_category_counts(conn, session_id)
    materialization = _read_archive_materialization(conn, "latency", session_id)
    return SessionLatencyProfileInsight(
        session_id=session_id,
        source_name=_provider_for_origin(str(row["origin"])).value,
        title=str(row["title"]) if row["title"] is not None else None,
        provenance=_archive_provenance(materialization),
        latency=SessionLatencyProfilePayload(
            median_tool_call_ms=0,
            p90_tool_call_ms=0,
            max_tool_call_ms=0,
            stuck_tool_count=0,
            median_agent_response_ms=_median_ms(agent_response_ms),
            median_user_response_ms=_median_ms(user_response_ms),
            tool_call_count_by_category=tool_counts,
        ),
    )


def _latency_tool_category_counts(conn: sqlite3.Connection, session_id: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT COALESCE(NULLIF(semantic_type, ''), 'unknown') AS category, COUNT(*) AS count
        FROM actions
        WHERE session_id = ?
        GROUP BY category
        ORDER BY count DESC, category
        """,
        (session_id,),
    ).fetchall()
    return {str(row["category"]): int(row["count"] or 0) for row in rows}


def _median_ms(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return int((ordered[middle - 1] + ordered[middle]) / 2)


def _coverage_bucket_filter(
    bucket: str,
    bucket_format: str,
    *,
    origin: str | None,
    since_ms: int | None,
    until_ms: int | None,
) -> tuple[str, tuple[object, ...]]:
    clauses = ["strftime(?, s.sort_key_ms / 1000, 'unixepoch') = ?"]
    params: list[object] = [bucket_format, bucket]
    if origin is not None:
        clauses.append("s.origin = ?")
        params.append(origin)
    if since_ms is not None:
        clauses.append("s.sort_key_ms >= ?")
        params.append(since_ms)
    if until_ms is not None:
        clauses.append("s.sort_key_ms <= ?")
        params.append(until_ms)
    return "WHERE " + " AND ".join(clauses), tuple(params)


def _coverage_work_event_breakdown(
    conn: sqlite3.Connection,
    bucket: str,
    bucket_format: str,
    *,
    origin: str | None,
    since_ms: int | None,
    until_ms: int | None,
) -> dict[str, int]:
    where, params = _coverage_bucket_filter(
        bucket,
        bucket_format,
        origin=origin,
        since_ms=since_ms,
        until_ms=until_ms,
    )
    rows = conn.execute(
        f"""
        SELECT e.work_event_type, COUNT(*) AS count
        FROM sessions s
        JOIN session_work_events e ON e.session_id = s.session_id
        {where}
        GROUP BY e.work_event_type
        ORDER BY count DESC, e.work_event_type
        """,
        params,
    ).fetchall()
    return {str(row["work_event_type"]): int(row["count"] or 0) for row in rows}


def _coverage_repos_active(
    conn: sqlite3.Connection,
    bucket: str,
    bucket_format: str,
    *,
    origin: str | None,
    since_ms: int | None,
    until_ms: int | None,
) -> tuple[str, ...]:
    where, params = _coverage_bucket_filter(
        bucket,
        bucket_format,
        origin=origin,
        since_ms=since_ms,
        until_ms=until_ms,
    )
    rows = conn.execute(
        f"""
        SELECT DISTINCT COALESCE(NULLIF(r.repo_name, ''), NULLIF(r.root_path, ''), NULLIF(r.origin_url, '')) AS repo
        FROM sessions s
        JOIN session_repos sr ON sr.session_id = s.session_id
        JOIN repos r ON r.repo_id = sr.repo_id
        {where}
        ORDER BY repo
        """,
        params,
    ).fetchall()
    return tuple(str(row["repo"]) for row in rows if row["repo"])


def _coverage_origin_breakdown(
    conn: sqlite3.Connection,
    bucket: str,
    bucket_format: str,
    *,
    origin: str | None,
    since_ms: int | None,
    until_ms: int | None,
) -> dict[str, int]:
    where, params = _coverage_bucket_filter(
        bucket,
        bucket_format,
        origin=origin,
        since_ms=since_ms,
        until_ms=until_ms,
    )
    rows = conn.execute(
        f"""
        SELECT s.origin, COUNT(DISTINCT s.session_id) AS count
        FROM sessions s
        {where}
        GROUP BY s.origin
        ORDER BY count DESC, s.origin
        """,
        params,
    ).fetchall()
    return {str(row["origin"]): int(row["count"] or 0) for row in rows}


def _archive_missing_materialization_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        insight_type: _count_scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM sessions AS s
            WHERE NOT EXISTS (
                SELECT 1
                FROM insight_materialization AS m
                WHERE m.insight_type = ? AND m.session_id = s.session_id
            )
            """,
            (insight_type,),
        )
        for insight_type in SESSION_INSIGHT_MATERIALIZATION_TYPES
    }


def _dominant_repo(rows: list[sqlite3.Row]) -> str | None:
    counts: dict[str, int] = {}
    for row in rows:
        repo = row["git_repository_url"]
        if not isinstance(repo, str) or not repo:
            continue
        counts[repo] = counts.get(repo, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _thread_member_depth(rows: list[sqlite3.Row], session_id: str) -> int:
    parents = {
        str(row["session_id"]): str(row["parent_session_id"]) for row in rows if row["parent_session_id"] is not None
    }
    depth = 0
    current = session_id
    seen: set[str] = set()
    while current in parents and current not in seen:
        seen.add(current)
        current = parents[current]
        depth += 1
    return depth


def _archive_thread_member_role(row: sqlite3.Row, thread_id: str) -> str:
    if str(row["session_id"]) == thread_id:
        return "root"
    if row["parent_session_id"] is not None:
        return "parent_continuation"
    return "member"


def _archive_thread_member_support_signals(row: sqlite3.Row) -> tuple[str, ...]:
    signals = ["archive_thread_sessions"]
    if row["parent_session_id"] is not None:
        signals.append("parent_session_id")
    return tuple(signals)


def _archive_thread_member_evidence(row: sqlite3.Row, thread_id: str, position: int) -> tuple[str, ...]:
    evidence = [f"position={position}"]
    if row["parent_session_id"] is not None:
        evidence.append(f"parent_id={row['parent_session_id']}")
        evidence.append(f"root_id={thread_id}")
    return tuple(evidence)


def _profile_or_session_timestamp_ms(row: sqlite3.Row, *, profile_column: str, session_column: str) -> int | None:
    profile_timestamp = row[profile_column]
    if isinstance(profile_timestamp, str) and profile_timestamp.strip():
        parsed = _epoch_ms_from_iso(profile_timestamp)
        if parsed is not None:
            return parsed
    session_timestamp = row[session_column]
    return int(session_timestamp) if isinstance(session_timestamp, int) else None


def _tag_origin_breakdown(
    conn: sqlite3.Connection,
    tag: str,
    clause: str,
    params: tuple[object, ...],
    tags_relation: str,
) -> dict[str, int]:
    tag_clause, tag_params = _with_exact_tag_filter(clause, params, tag)
    rows = conn.execute(
        f"""
        SELECT s.origin, COUNT(DISTINCT s.session_id) AS count
        FROM sessions s
        JOIN {tags_relation} st ON st.session_id = s.session_id
        {tag_clause}
        GROUP BY s.origin
        ORDER BY count DESC, s.origin
        """,
        tag_params,
    ).fetchall()
    return {str(row["origin"]): int(row["count"] or 0) for row in rows}


def _tag_repo_breakdown(
    conn: sqlite3.Connection,
    tag: str,
    clause: str,
    params: tuple[object, ...],
    tags_relation: str,
) -> dict[str, int]:
    tag_clause, tag_params = _with_exact_tag_filter(clause, params, tag)
    rows = conn.execute(
        f"""
        SELECT s.git_repository_url AS repo, COUNT(DISTINCT s.session_id) AS count
        FROM sessions s
        JOIN {tags_relation} st ON st.session_id = s.session_id
        {tag_clause}
          AND s.git_repository_url IS NOT NULL
          AND s.git_repository_url != ''
        GROUP BY s.git_repository_url
        ORDER BY count DESC, s.git_repository_url
        """,
        tag_params,
    ).fetchall()
    return {str(row["repo"]): int(row["count"] or 0) for row in rows}


def _with_exact_tag_filter(clause: str, params: tuple[object, ...], tag: str) -> tuple[str, tuple[object, ...]]:
    if clause:
        return f"{clause} AND st.tag = ?", (*params, tag)
    return "WHERE st.tag = ?", (tag,)


def _iso_from_ms(value: object) -> str | None:
    if not isinstance(value, int):
        return None
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def _provider_for_origin(origin: str) -> Provider:
    return {
        "claude-code-session": Provider.CLAUDE_CODE,
        "codex-session": Provider.CODEX,
        "gemini-cli-session": Provider.GEMINI_CLI,
        "hermes-session": Provider.HERMES,
        "antigravity-session": Provider.ANTIGRAVITY,
        "chatgpt-export": Provider.CHATGPT,
        "claude-ai-export": Provider.CLAUDE_AI,
        "aistudio-drive": Provider.GEMINI,
        "unknown-export": Provider.UNKNOWN,
    }.get(origin, Provider.UNKNOWN)


__all__ = ["ArchiveFileQueryRow", "ArchiveStore", "ArchiveSessionSearchHit", "ArchiveSessionSummary"]
