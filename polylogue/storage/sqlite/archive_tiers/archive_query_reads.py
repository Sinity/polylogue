"""Terminal archive query reads and their SQL projections."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from polylogue.analysis.run_projection import ContextSnapshot, ObservedEvent, ProjectedRun
from polylogue.archive.actions.followup import ACKNOWLEDGMENT_MARKERS
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
    QuerySequenceConstraint,
    QuerySequencePredicate,
    QueryTextPredicate,
)
from polylogue.archive.topology.edge import topology_status_composes_sql
from polylogue.core.dates import parse_date
from polylogue.core.enums import ActionResultState, ToolOutcome
from polylogue.core.json import JSONValue, require_json_value
from polylogue.core.refs import delegation_edge_object_id
from polylogue.storage.search.query_support import normalize_fts5_query
from polylogue.storage.sqlite.action_relation import bounded_action_relation_cte
from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import BLOCKS_SPEC
from polylogue.storage.sqlite.archive_tiers.types import (
    DelegationMappingState,
    DelegationResultStatus,
)
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ASSERTION_DEFAULT_AUTHOR_KIND,
    ASSERTION_DEFAULT_AUTHOR_REF,
    ASSERTION_DEFAULT_CONTEXT_POLICY,
    ASSERTION_DEFAULT_STATUS,
    ASSERTION_DEFAULT_VISIBILITY,
)
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveBlockRow,
)
from polylogue.storage.sqlite.queries.project_refs import expand_project_refs
from polylogue.storage.sqlite.run_projection_relations import (
    context_snapshot_from_row,
    context_snapshot_relation_sql,
    observed_event_from_row,
    observed_event_relation_sql,
    observed_event_source_pushdown,
    projected_run_from_row,
    run_relation_sql,
)


class _ArchiveQueryReadsHost(Protocol):
    _conn: sqlite3.Connection
    user_db_path: Path

    def _attach_user_tier_if_present(self) -> None: ...

    def require_user_tier(self) -> None: ...


_UNIT_SESSION_ID_EXPRESSION: dict[str, str] = {
    "message": "{alias}.session_id",
    "action": "{alias}.session_id",
    "file": "{alias}.session_id",
    "block": "{alias}.session_id",
    "assertion": "substr({alias}.target_ref, 9)",
    "run": "{alias}.session_id",
    "observed-event": "{alias}.session_id",
    "context-snapshot": "{alias}.session_id",
    "delegation": "{alias}.parent_session_id",
}


@dataclass(frozen=True, slots=True)
class ArchiveMessageQueryRow:
    """Terminal query projection over archive messages."""

    message_id: str
    session_id: str
    origin: str
    title: str | None
    repo: str | None
    role: str
    message_type: str
    material_origin: str
    occurred_at_ms: int | None
    position: int
    word_count: int
    text: str
    blocks: tuple[ArchiveBlockRow, ...] = ()


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
    is_error: int | None
    exit_code: int | None
    result_state: ActionResultState
    followup_class: str | None
    followup_message_ref: str | None


def _archive_action_query_row(row: sqlite3.Row) -> ArchiveActionQueryRow:
    tool_result_block_id = str(row["tool_result_block_id"]) if row["tool_result_block_id"] is not None else None
    is_error = int(row["is_error"]) if row["is_error"] is not None else None
    exit_code = int(row["exit_code"]) if row["exit_code"] is not None else None
    return ArchiveActionQueryRow(
        session_id=str(row["session_id"]),
        message_id=str(row["message_id"]),
        origin=str(row["origin"]),
        title=str(row["title"]) if row["title"] is not None else None,
        tool_use_block_id=str(row["tool_use_block_id"]),
        tool_result_block_id=tool_result_block_id,
        tool_name=str(row["tool_name"]) if row["tool_name"] is not None else None,
        semantic_type=str(row["semantic_type"]) if row["semantic_type"] is not None else None,
        tool_command=str(row["tool_command"]) if row["tool_command"] is not None else None,
        tool_path=str(row["tool_path"]) if row["tool_path"] is not None else None,
        occurred_at_ms=int(row["occurred_at_ms"]) if row["occurred_at_ms"] is not None else None,
        output_text=str(row["output_text"]) if row["output_text"] is not None else None,
        is_error=is_error,
        exit_code=exit_code,
        result_state=ActionResultState(str(row["result_state"])),
        followup_class=str(row["followup_class"]) if row["followup_class"] is not None else None,
        followup_message_ref=str(row["followup_message_ref"]) if row["followup_message_ref"] is not None else None,
    )


@dataclass(frozen=True, slots=True)
class ArchiveDelegationQueryRow:
    """Terminal query projection over one `delegations` view row
    (polylogue-y964). ``mapping_state`` is the view's own vocabulary --
    resolved/unresolved/edge_only/quarantined -- never reinterpreted here
    (polylogue-1vpm.7 retired 'ambiguous': with a provider-asserted content
    or trivial-cohort join key, a cardinality mismatch is not a reachable
    state). Action-observed rows (resolved/unresolved) always carry
    ``instruction_tool_use_block_id``; edge-only rows (edge_only/quarantined)
    never fabricate one."""

    parent_session_id: str
    child_session_id: str | None
    mapping_state: DelegationMappingState
    link_confidence: float | None
    link_method: str | None
    inheritance: str | None
    branch_point_message_id: str | None
    instruction_message_id: str | None
    instruction_tool_use_block_id: str | None
    instruction_payload: str | None
    dispatch_turn_model: str | None
    requested_model: str | None
    artifact_block_id: str | None
    artifact_text: str | None
    result_is_error: int | None
    result_exit_code: int | None
    result_status: DelegationResultStatus
    parent_origin: str
    parent_session_dominant_model: str | None
    parent_session_dominant_model_family: str | None
    parent_terminal_state: str | None
    child_session_dominant_model: str | None
    child_session_dominant_model_family: str | None
    child_cost_usd: float | None
    child_cost_is_estimated: int | None
    child_tokens: int | None
    child_wall_ms: int | None
    child_terminal_state: str | None


@dataclass(frozen=True, slots=True)
class ArchiveDelegationContextRow:
    """One bounded message excerpt surrounding a delegation dispatch."""

    message_id: str
    role: str
    text: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class ArchiveDelegationCard:
    """Explicit bounded evidence card for one delegation attempt."""

    attempt: ArchiveDelegationQueryRow
    delegation_ref: str
    parent_session_title: str | None
    child_session_title: str | None
    run_ref: str | None
    run_title: str | None
    instruction: str | None
    parent_context: tuple[ArchiveDelegationContextRow, ...]
    parent_context_truncated: bool
    dispatch_result: str | None
    dispatch_result_truncated: bool
    child_excerpt: str | None
    child_excerpt_truncated: bool
    parent_followup: tuple[ArchiveDelegationContextRow, ...]
    parent_followup_truncated: bool
    annotation_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]


def _archive_delegation_query_row(row: sqlite3.Row) -> ArchiveDelegationQueryRow:
    return ArchiveDelegationQueryRow(
        parent_session_id=str(row["parent_session_id"]),
        child_session_id=str(row["child_session_id"]) if row["child_session_id"] is not None else None,
        mapping_state=cast(DelegationMappingState, str(row["mapping_state"])),
        link_confidence=float(row["link_confidence"]) if row["link_confidence"] is not None else None,
        link_method=str(row["link_method"]) if row["link_method"] is not None else None,
        inheritance=str(row["inheritance"]) if row["inheritance"] is not None else None,
        branch_point_message_id=(
            str(row["branch_point_message_id"]) if row["branch_point_message_id"] is not None else None
        ),
        instruction_message_id=(
            str(row["instruction_message_id"]) if row["instruction_message_id"] is not None else None
        ),
        instruction_tool_use_block_id=(
            str(row["instruction_tool_use_block_id"]) if row["instruction_tool_use_block_id"] is not None else None
        ),
        instruction_payload=str(row["instruction_payload"]) if row["instruction_payload"] is not None else None,
        dispatch_turn_model=str(row["dispatch_turn_model"]) if row["dispatch_turn_model"] is not None else None,
        requested_model=str(row["requested_model"]) if row["requested_model"] is not None else None,
        artifact_block_id=str(row["artifact_block_id"]) if row["artifact_block_id"] is not None else None,
        artifact_text=str(row["artifact_text"]) if row["artifact_text"] is not None else None,
        result_is_error=int(row["result_is_error"]) if row["result_is_error"] is not None else None,
        result_exit_code=int(row["result_exit_code"]) if row["result_exit_code"] is not None else None,
        result_status=cast(DelegationResultStatus, str(row["result_status"])),
        parent_origin=str(row["parent_origin"]),
        parent_session_dominant_model=(
            str(row["parent_session_dominant_model"]) if row["parent_session_dominant_model"] is not None else None
        ),
        parent_session_dominant_model_family=(
            str(row["parent_session_dominant_model_family"])
            if row["parent_session_dominant_model_family"] is not None
            else None
        ),
        parent_terminal_state=(str(row["parent_terminal_state"]) if row["parent_terminal_state"] is not None else None),
        child_session_dominant_model=(
            str(row["child_session_dominant_model"]) if row["child_session_dominant_model"] is not None else None
        ),
        child_session_dominant_model_family=(
            str(row["child_session_dominant_model_family"])
            if row["child_session_dominant_model_family"] is not None
            else None
        ),
        child_cost_usd=float(row["child_cost_usd"]) if row["child_cost_usd"] is not None else None,
        child_cost_is_estimated=(
            int(row["child_cost_is_estimated"]) if row["child_cost_is_estimated"] is not None else None
        ),
        child_tokens=int(row["child_tokens"]) if row["child_tokens"] is not None else None,
        child_wall_ms=int(row["child_wall_ms"]) if row["child_wall_ms"] is not None else None,
        child_terminal_state=str(row["child_terminal_state"]) if row["child_terminal_state"] is not None else None,
    )


@dataclass(frozen=True, slots=True)
class ArchiveDelegationAncestryRow:
    """One node in a depth-annotated delegation ancestry chain (polylogue-qsb4),
    root-to-node ordered. ``depth`` is 0 at the queried session itself and
    increases toward the root. ``child_session_id`` is the next node down the
    chain (the one dispatched by this node, one step closer to the queried
    session) -- ``None`` at depth 0, which has no outgoing edge in this
    traversal. ``mapping_state``/``instruction_tool_use_block_id``/
    ``link_confidence``/``link_method`` describe that edge (this node ->
    ``child_session_id``)."""

    session_id: str
    depth: int
    child_session_id: str | None
    mapping_state: DelegationMappingState | None
    instruction_tool_use_block_id: str | None
    link_confidence: float | None
    link_method: str | None


@dataclass(frozen=True, slots=True)
class ArchiveDelegationSubtreeRow:
    """One node in a depth-annotated delegation subtree (polylogue-qsb4),
    the queried session plus all its transitive dispatch descendants.
    ``depth`` is 0 at the queried session itself (the subtree root) and
    increases toward descendants. ``parent_session_id`` is this node's own
    dispatcher -- ``None`` at depth 0, which is out of scope for this
    traversal. ``mapping_state``/``instruction_tool_use_block_id``/
    ``link_confidence``/``link_method`` describe the edge
    (``parent_session_id`` -> this node)."""

    session_id: str
    depth: int
    parent_session_id: str | None
    mapping_state: DelegationMappingState | None
    instruction_tool_use_block_id: str | None
    link_confidence: float | None
    link_method: str | None


def _archive_delegation_ancestry_row(row: sqlite3.Row) -> ArchiveDelegationAncestryRow:
    return ArchiveDelegationAncestryRow(
        session_id=str(row["session_id"]),
        depth=int(row["depth"]),
        child_session_id=str(row["child_session_id"]) if row["child_session_id"] is not None else None,
        mapping_state=(
            cast(DelegationMappingState, str(row["mapping_state"])) if row["mapping_state"] is not None else None
        ),
        instruction_tool_use_block_id=(
            str(row["instruction_tool_use_block_id"]) if row["instruction_tool_use_block_id"] is not None else None
        ),
        link_confidence=float(row["link_confidence"]) if row["link_confidence"] is not None else None,
        link_method=str(row["link_method"]) if row["link_method"] is not None else None,
    )


def _archive_delegation_subtree_row(row: sqlite3.Row) -> ArchiveDelegationSubtreeRow:
    return ArchiveDelegationSubtreeRow(
        session_id=str(row["session_id"]),
        depth=int(row["depth"]),
        parent_session_id=str(row["parent_session_id"]) if row["parent_session_id"] is not None else None,
        mapping_state=(
            cast(DelegationMappingState, str(row["mapping_state"])) if row["mapping_state"] is not None else None
        ),
        instruction_tool_use_block_id=(
            str(row["instruction_tool_use_block_id"]) if row["instruction_tool_use_block_id"] is not None else None
        ),
        link_confidence=float(row["link_confidence"]) if row["link_confidence"] is not None else None,
        link_method=str(row["link_method"]) if row["link_method"] is not None else None,
    )


_DELEGATION_ANCESTRY_SQL = f"""
WITH RECURSIVE ancestry(session_id, depth, child_session_id, mapping_state,
                         instruction_tool_use_block_id, link_confidence, link_method, path) AS (
    SELECT ?, 0, NULL, NULL, NULL, NULL, NULL, '/' || ? || '/'
    UNION ALL
    SELECT
        d.parent_session_id,
        a.depth + 1,
        a.session_id,
        d.mapping_state,
        d.instruction_tool_use_block_id,
        d.link_confidence,
        d.link_method,
        a.path || d.parent_session_id || '/'
    FROM delegations d
    JOIN ancestry a ON d.child_session_id = a.session_id
    WHERE {topology_status_composes_sql("d.mapping_state")}
      AND instr(a.path, '/' || d.parent_session_id || '/') = 0
)
SELECT session_id, depth, child_session_id, mapping_state, instruction_tool_use_block_id, link_confidence, link_method
FROM ancestry
ORDER BY depth DESC
"""

_DELEGATION_SUBTREE_SQL = f"""
WITH RECURSIVE subtree(session_id, depth, parent_session_id, mapping_state,
                        instruction_tool_use_block_id, link_confidence, link_method, path) AS (
    SELECT ?, 0, NULL, NULL, NULL, NULL, NULL, '/' || ? || '/'
    UNION ALL
    SELECT
        d.child_session_id,
        s.depth + 1,
        s.session_id,
        d.mapping_state,
        d.instruction_tool_use_block_id,
        d.link_confidence,
        d.link_method,
        s.path || d.child_session_id || '/'
    FROM delegations d
    JOIN subtree s ON d.parent_session_id = s.session_id
    WHERE {topology_status_composes_sql("d.mapping_state")}
      AND d.child_session_id IS NOT NULL
      AND instr(s.path, '/' || d.child_session_id || '/') = 0
)
SELECT session_id, depth, parent_session_id, mapping_state, instruction_tool_use_block_id, link_confidence, link_method
FROM subtree
ORDER BY depth, session_id
"""


def _delegation_instruction(payload: str | None) -> str | None:
    if payload is None:
        return None
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return payload
    if isinstance(decoded, dict):
        for key in ("prompt", "description", "instruction", "task"):
            value = decoded.get(key)
            if isinstance(value, str) and value:
                return value
        return None
    return None


def _bounded_delegation_card_text(value: str | None, *, limit: int) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    if len(value) <= limit:
        return value, False
    return value[:limit], True


def _delegation_message_window(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    anchor_position: int,
    before: bool,
    limit: int = 3,
    text_limit: int = 1000,
) -> tuple[tuple[ArchiveDelegationContextRow, ...], bool]:
    operator = "<" if before else ">"
    direction = "DESC" if before else "ASC"
    rows = conn.execute(
        f"""
        SELECT
            m.message_id,
            m.role,
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
        WHERE m.session_id = ? AND m.position {operator} ?
        ORDER BY m.position {direction}, m.message_id {direction}
        LIMIT ?
        """,
        (session_id, anchor_position, limit + 1),
    ).fetchall()
    window_truncated = len(rows) > limit
    rows = rows[:limit]
    if before:
        rows = list(reversed(rows))
    projected: list[ArchiveDelegationContextRow] = []
    for row in rows:
        text, text_truncated = _bounded_delegation_card_text(str(row["text"] or ""), limit=text_limit)
        projected.append(
            ArchiveDelegationContextRow(
                message_id=str(row["message_id"]),
                role=str(row["role"]),
                text=text or "",
                truncated=text_truncated,
            )
        )
    return tuple(projected), window_truncated


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
    """Terminal query projection over source-derived or materialized run rows."""

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
    """Terminal query projection over source-derived or materialized context rows."""

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


@dataclass(frozen=True, slots=True)
class ArchiveQueryUnitMultiAggregateRow:
    """One losslessly addressable row from a multi-field aggregate page."""

    unit: str
    group_by: tuple[str, ...]
    group_values: tuple[str, ...]
    count: int


@dataclass(frozen=True, slots=True)
class ArchiveQueryUnitMultiAggregatePage:
    """A bounded multi-field aggregate page plus exact full-result facts.

    ``rows`` is bounded by the caller's page limit. ``denominator`` and the
    per-field quality counters describe the complete matching row set rather
    than merely the returned page, so page boundaries never change aggregate
    meaning.
    """

    rows: tuple[ArchiveQueryUnitMultiAggregateRow, ...]
    denominator: int
    missing_counts: tuple[int, ...]
    unknown_counts: tuple[int, ...]


def _sql_string_literal(value: str) -> str:
    """Return a SQL single-quoted literal for a static in-repo token."""

    return "'" + value.replace("'", "''") + "'"


_LEGACY_EXECUTION_TOOL_NAMES = (
    "bash",
    "exec",
    "exec_command",
    "functions.exec",
    "functions.exec_command",
    "local_shell_call",
    "run",
    "shell",
    "shell_command",
    "terminal",
)


def _action_command_expression(row_alias: str) -> str:
    """Return canonical command text without rewriting historical evidence.

    Current parsers persist ``command`` alongside provider-native fields. Older
    Codex rows may instead carry ``cmd`` or free-form ``arguments``. Restrict
    those fallbacks to execution tools so unrelated argument strings never
    become shell commands merely because they share the same JSON key.
    """

    execution_tools = ", ".join(_sql_string_literal(name) for name in _LEGACY_EXECUTION_TOOL_NAMES)
    return f"""
        COALESCE(
            NULLIF({row_alias}.tool_command, ''),
            CASE
                WHEN LOWER(COALESCE({row_alias}.tool_name, '')) IN ({execution_tools}) THEN
                    COALESCE(
                        NULLIF(json_extract({row_alias}.tool_input, '$.cmd'), ''),
                        CASE
                            WHEN json_type({row_alias}.tool_input, '$.arguments') = 'text'
                                THEN NULLIF(json_extract({row_alias}.tool_input, '$.arguments'), '')
                            ELSE NULL
                        END
                    )
                ELSE NULL
            END
        )
    """.strip()


_ACTION_FOLLOWUP_ACK_CONDITION = " OR ".join(
    f"followup_text_lower LIKE '%' || {_sql_string_literal(marker)} || '%'" for marker in ACKNOWLEDGMENT_MARKERS
)

_ACTION_FOLLOWUP_RELATION_SQL = f"""
WITH action_followup_base AS (
    SELECT
        a.*,
        CASE
            WHEN COALESCE(a.is_error, 0) = 1 OR COALESCE(a.exit_code, 0) != 0
                THEN (
                    SELECT nm.message_id
                    FROM messages nm
                    WHERE nm.session_id = a.session_id
                      AND nm.role = 'assistant'
                      AND nm.position > COALESCE(
                          (
                              SELECT rm.position
                              FROM blocks rb
                              JOIN messages rm ON rm.message_id = rb.message_id
                              WHERE rb.block_id = a.tool_result_block_id
                              LIMIT 1
                          ),
                          (
                              SELECT um.position
                              FROM messages um
                              WHERE um.message_id = a.message_id
                              LIMIT 1
                          ),
                          -1
                      )
                    ORDER BY nm.position, nm.message_id
                    LIMIT 1
                )
            ELSE NULL
        END AS followup_message_id
    FROM actions a
),
action_followup_text AS (
    SELECT
        afb.*,
        COALESCE((
            SELECT group_concat(ordered.search_text, char(10))
            FROM (
                SELECT b.search_text
                FROM blocks b
                WHERE b.message_id = afb.followup_message_id
                  AND b.search_text IS NOT NULL
                ORDER BY b.position, b.block_id
            ) ordered
        ), '') AS followup_text,
        EXISTS (
            SELECT 1
            FROM blocks tool_block
            WHERE tool_block.message_id = afb.followup_message_id
              AND tool_block.block_type = 'tool_use'
        ) AS followup_has_tool_use,
        COALESCE((
            SELECT SUM(LENGTH(COALESCE(text_block.search_text, '')))
            FROM blocks text_block
            WHERE text_block.message_id = afb.followup_message_id
              AND text_block.block_type = 'text'
              AND text_block.position < COALESCE(
                  (
                      SELECT MIN(first_tool.position)
                      FROM blocks first_tool
                      WHERE first_tool.message_id = afb.followup_message_id
                        AND first_tool.block_type = 'tool_use'
                  ),
                  9223372036854775807
              )
        ), 0) AS followup_pre_tool_text_chars
    FROM action_followup_base afb
),
action_rows AS (
    SELECT
        aft.*,
        CASE
            WHEN NOT (COALESCE(aft.is_error, 0) = 1 OR COALESCE(aft.exit_code, 0) != 0) THEN NULL
            WHEN aft.followup_message_id IS NULL THEN 'ambiguous'
            WHEN {_ACTION_FOLLOWUP_ACK_CONDITION} THEN 'acknowledged'
            WHEN TRIM(aft.followup_text_lower) GLOB '<thinking>*</thinking>'
              OR TRIM(aft.followup_text_lower) GLOB '<analysis>*</analysis>'
              OR TRIM(aft.followup_text_lower) GLOB '<reasoning>*</reasoning>' THEN 'ambiguous'
            WHEN aft.followup_has_tool_use = 1
             AND aft.followup_pre_tool_text_chars <= 40 THEN 'wordless_continuation'
            WHEN LENGTH(TRIM(aft.followup_text)) < 20 THEN 'ambiguous'
            ELSE 'silent_proceed'
        END AS followup_class,
        CASE
            WHEN aft.followup_message_id IS NOT NULL THEN 'message:' || aft.followup_message_id
            ELSE NULL
        END AS followup_message_ref
    FROM (
        SELECT
            action_followup_text.*,
            LOWER(' ' || action_followup_text.followup_text || ' ') AS followup_text_lower
        FROM action_followup_text
    ) aft
)
"""


def _exact_session_ids_from_predicate(predicate: QueryPredicate) -> tuple[str, ...] | None:
    """Return a safe owning-session bound implied by a predicate subtree."""
    if isinstance(predicate, QueryFieldPredicate):
        session_field = _predicate_session_field(predicate)
        if session_field not in {"id", "session"} or not predicate.values:
            return None
        # Session identity equality follows ordinary predicate lowering and uses
        # the final parsed value.  Do not broaden the physical relation beyond
        # the predicate's actual semantics.
        return tuple(value for value in predicate.values[-1:] if value)
    if isinstance(predicate, QueryBoolPredicate):
        child_bounds = [
            bound for child in predicate.children if (bound := _exact_session_ids_from_predicate(child)) is not None
        ]
        if not child_bounds:
            return None
        if predicate.op == "or":
            if len(child_bounds) != len(predicate.children):
                return None
            return tuple(dict.fromkeys(session_id for bound in child_bounds for session_id in bound))
        intersection = set(child_bounds[0])
        for bound in child_bounds[1:]:
            intersection.intersection_update(bound)
        return tuple(session_id for session_id in child_bounds[0] if session_id in intersection)
    return None


def _action_relation_for_query(
    *,
    predicate: QueryPredicate | None = None,
    session_ids: Sequence[str] = (),
    include_followup: bool,
) -> tuple[str, str, list[object]]:
    """Select the canonical action relation, physically bounded when safe."""
    explicit_ids = tuple(dict.fromkeys(session_id for session_id in session_ids if session_id))
    predicate_ids = _exact_session_ids_from_predicate(predicate) if predicate is not None else None
    normalized_ids = explicit_ids if explicit_ids else predicate_ids
    if normalized_ids is None:
        return (
            (_ACTION_FOLLOWUP_RELATION_SQL if include_followup else ""),
            ("action_rows" if include_followup else "actions"),
            [],
        )
    bounded_cte = bounded_action_relation_cte(
        relation_name="bounded_actions",
        session_count=len(normalized_ids),
    )
    relation_params: list[object] = [*normalized_ids, *normalized_ids, *normalized_ids]
    if not include_followup:
        return f"WITH {bounded_cte}", "bounded_actions", relation_params
    followup_ctes = _ACTION_FOLLOWUP_RELATION_SQL.strip().removeprefix("WITH ")
    followup_ctes = followup_ctes.replace("FROM actions a", "FROM bounded_actions a", 1)
    return f"WITH {bounded_cte},\n{followup_ctes}", "action_rows", relation_params


def _query_unit_order_direction(direction: Literal["asc", "desc"]) -> Literal["ASC", "DESC"]:
    """Return a closed SQL direction token for terminal row ordering."""

    return "DESC" if direction == "desc" else "ASC"


# ArchiveBlockRow is intentionally a compact read model rather than a full
# blocks-table row.  Keep its curated projection tied to the canonical table
# declaration: adding or renaming a storage column cannot leave this SELECT's
# spelling silently stale.  The order follows BLOCKS_SPEC; sqlite3.Row
# hydration is name-based, so this does not impose a positional contract.
_ARCHIVE_BLOCK_QUERY_ROW_FIELDS: frozenset[str] = frozenset(
    {
        "block_id",
        "message_id",
        "block_type",
        "text",
        "tool_name",
        "tool_id",
        "semantic_type",
        "tool_input",
        "language",
        "tool_result_is_error",
        "tool_result_exit_code",
        "tool_outcome",
    }
)
_ARCHIVE_BLOCK_QUERY_COLUMNS: tuple[str, ...] = tuple(
    column.name for column in BLOCKS_SPEC.all_columns if column.name in _ARCHIVE_BLOCK_QUERY_ROW_FIELDS
)
if set(_ARCHIVE_BLOCK_QUERY_COLUMNS) != _ARCHIVE_BLOCK_QUERY_ROW_FIELDS:
    raise RuntimeError("ArchiveBlockRow projection is not covered by BLOCKS_SPEC")


def _hydrate_archive_block_row(row: sqlite3.Row) -> ArchiveBlockRow:
    """Build an ``ArchiveBlockRow`` from a row selected via ``_ARCHIVE_BLOCK_QUERY_COLUMNS``."""

    return ArchiveBlockRow(
        block_id=str(row["block_id"]),
        message_id=str(row["message_id"]),
        block_type=str(row["block_type"]),
        text=str(row["text"]) if row["text"] is not None else None,
        tool_name=str(row["tool_name"]) if row["tool_name"] is not None else None,
        tool_id=str(row["tool_id"]) if row["tool_id"] is not None else None,
        semantic_type=str(row["semantic_type"]) if row["semantic_type"] is not None else None,
        tool_input=str(row["tool_input"]) if row["tool_input"] is not None else None,
        language=str(row["language"]) if row["language"] is not None else None,
        tool_result_is_error=(int(row["tool_result_is_error"]) if row["tool_result_is_error"] is not None else None),
        tool_result_exit_code=(int(row["tool_result_exit_code"]) if row["tool_result_exit_code"] is not None else None),
        tool_outcome=ToolOutcome(row["tool_outcome"]) if row["tool_outcome"] is not None else None,
    )


def _fetch_blocks_for_messages(
    conn: sqlite3.Connection, message_ids: tuple[str, ...]
) -> dict[str, list[ArchiveBlockRow]]:
    """Fetch and hydrate every block for ``message_ids``, keyed by message_id.

    Shared by ``query_messages`` and ``query_session_messages`` -- see
    ``_ARCHIVE_BLOCK_QUERY_COLUMNS`` docstring above for why this used to be
    two independently-maintained copies of the same query.
    """

    blocks_by_message: dict[str, list[ArchiveBlockRow]] = {message_id: [] for message_id in message_ids}
    if not message_ids:
        return blocks_by_message
    block_placeholders = ", ".join("?" for _ in message_ids)
    columns_sql = ", ".join(_ARCHIVE_BLOCK_QUERY_COLUMNS)
    block_rows = conn.execute(
        f"""
        SELECT {columns_sql}
        FROM blocks
        WHERE message_id IN ({block_placeholders})
        ORDER BY message_id, position, block_id
        """,
        message_ids,
    ).fetchall()
    for block in block_rows:
        blocks_by_message[str(block["message_id"])].append(_hydrate_archive_block_row(block))
    return blocks_by_message


_ARCHIVE_FILE_QUERY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("session_id", "f.session_id"),
    ("origin", "s.origin"),
    ("title", "s.title"),
    ("path", "f.path"),
    ("action_count", "f.action_count"),
    ("first_message_id", "f.first_message_id"),
    ("first_tool_use_block_id", "f.first_tool_use_block_id"),
    ("last_tool_use_block_id", "f.last_tool_use_block_id"),
    ("first_seen_ms", "f.first_seen_ms"),
    ("last_seen_ms", "f.last_seen_ms"),
)

_ARCHIVE_FILE_QUERY_SELECT_SQL = ",\n                ".join(
    expr if expr.endswith(f".{name}") else f"{expr} AS {name}" for name, expr in _ARCHIVE_FILE_QUERY_COLUMNS
)


def _hydrate_archive_file_query_row(row: sqlite3.Row) -> ArchiveFileQueryRow:
    """Build an ``ArchiveFileQueryRow`` from a row selected via ``_ARCHIVE_FILE_QUERY_SELECT_SQL``."""

    return ArchiveFileQueryRow(
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


_ARCHIVE_ACTION_QUERY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("session_id", "a.session_id"),
    ("message_id", "a.message_id"),
    ("origin", "s.origin"),
    ("title", "s.title"),
    ("tool_use_block_id", "a.tool_use_block_id"),
    ("tool_result_block_id", "a.tool_result_block_id"),
    ("tool_name", "a.tool_name"),
    ("semantic_type", "a.semantic_type"),
    ("tool_command", _action_command_expression("a")),
    ("tool_path", "a.tool_path"),
    ("occurred_at_ms", "m.occurred_at_ms"),
    ("output_text", "a.output_text"),
    ("is_error", "a.is_error"),
    ("exit_code", "a.exit_code"),
    ("result_state", "a.result_state"),
    ("followup_class", "a.followup_class"),
    ("followup_message_ref", "a.followup_message_ref"),
)

_ARCHIVE_ACTION_QUERY_SELECT_SQL = ",\n                ".join(
    expr if expr.endswith(f".{name}") else f"{expr} AS {name}" for name, expr in _ARCHIVE_ACTION_QUERY_COLUMNS
)

_QUERY_UNIT_ROW_ALIAS: dict[str, str] = {
    "message": "m",
    "action": "a",
    "block": "b",
    "file": "f",
    "assertion": "a",
    "observed-event": "e",
    "delegation": "d",
}


def _query_unit_from_sql_by_unit(action_relation_name: str) -> dict[str, str]:
    """Return the unit -> FROM-clause map shared by both aggregate-count methods."""

    return {
        "message": "messages m JOIN sessions s ON s.session_id = m.session_id",
        "action": f"{action_relation_name} a JOIN sessions s ON s.session_id = a.session_id",
        "block": "blocks b JOIN sessions s ON s.session_id = b.session_id",
        "assertion": "user_tier.assertions a LEFT JOIN sessions s ON a.target_ref = 'session:' || s.session_id",
        "observed-event": "observed_events e JOIN sessions s ON s.session_id = e.session_id",
        "delegation": "delegations d JOIN sessions s ON s.session_id = d.parent_session_id",
    }


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
    fields = tuple(field.strip() for field in group_by.split(","))
    if len(fields) > 1:
        # Keep multi-dimensional grouping in SQLite.  The old Python fallback
        # fetched every matching action/delegation row before grouping, which
        # made a bounded page an archive-wide materialization request.
        components = ", ".join(f"'{field}', {_query_unit_group_expression(unit, row_alias, field)}" for field in fields)
        return f"json_object({components})"
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
            "is_error": f"COALESCE(CAST({row_alias}.is_error AS TEXT), 'unknown')",
            "exit_code": f"COALESCE(CAST({row_alias}.exit_code AS TEXT), 'unknown')",
            "followup_class": f"COALESCE(NULLIF({row_alias}.followup_class, ''), 'unknown')",
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
        "delegation": {
            "basis": (f"CASE WHEN {row_alias}.instruction_tool_use_block_id IS NULL THEN 'edge' ELSE 'action' END"),
            "mapping_state": f"COALESCE(NULLIF({row_alias}.mapping_state, ''), 'unknown')",
            "result_status": f"COALESCE(NULLIF({row_alias}.result_status, ''), 'unknown')",
            "requested_model": f"COALESCE(NULLIF({row_alias}.requested_model, ''), 'unknown')",
            "dispatch_model": f"COALESCE(NULLIF({row_alias}.dispatch_turn_model, ''), 'unknown')",
            "child_model": f"COALESCE(NULLIF({row_alias}.child_session_dominant_model, ''), 'unknown')",
        },
    }
    try:
        return unit_fields[unit][group_by]
    except KeyError as exc:
        raise ValueError(f"unsupported {unit} aggregate group field: {group_by}") from exc


@dataclass(frozen=True, slots=True)
class _MultiAggregateFieldSQL:
    """Closed SQL expressions for one multi-field aggregate dimension."""

    value: str
    missing: str
    unknown: str


def _query_unit_multi_group_field_sql(unit: str, row_alias: str, field: str) -> _MultiAggregateFieldSQL:
    """Return lossless value and quality expressions for one group field.

    Multi-field aggregation historically converted terminal rows back into
    Python objects, where ``None`` became ``[missing]`` and literal ``unknown``
    values remained distinguishable. Keep that contract while moving the work
    into SQLite; do not conflate an empty string, a missing value, and an
    explicit unknown token.
    """

    if field == "session.origin":
        raw = "s.origin"
    elif field == "session.repo":
        raw = "s.git_repository_url"
    else:
        raw_fields = {
            "message": {
                "role": f"{row_alias}.role",
                "type": f"{row_alias}.message_type",
            },
            "action": {
                "tool": f"{row_alias}.tool_name",
                "action": f"{row_alias}.semantic_type",
                "type": f"{row_alias}.semantic_type",
                "is_error": f"{row_alias}.is_error",
                "exit_code": f"{row_alias}.exit_code",
                "followup_class": f"{row_alias}.followup_class",
            },
            "block": {
                "type": f"{row_alias}.block_type",
                "tool": f"{row_alias}.tool_name",
                "action": f"{row_alias}.semantic_type",
            },
            "file": {"path": f"{row_alias}.path"},
            "assertion": {
                "kind": f"{row_alias}.kind",
                "status": f"{row_alias}.status",
                "visibility": f"{row_alias}.visibility",
                "author_kind": f"{row_alias}.author_kind",
            },
            "observed-event": {
                "kind": f"{row_alias}.kind",
                "delivery_state": f"{row_alias}.delivery_state",
                "tool": f"json_extract({row_alias}.payload_json, '$.tool_name')",
                "handler": f"json_extract({row_alias}.payload_json, '$.handler_kind')",
                "status": f"json_extract({row_alias}.payload_json, '$.status')",
            },
            "delegation": {
                "mapping_state": f"{row_alias}.mapping_state",
                "result_status": f"{row_alias}.result_status",
                "requested_model": f"{row_alias}.requested_model",
                "dispatch_model": f"{row_alias}.dispatch_turn_model",
                "child_model": f"{row_alias}.child_session_dominant_model",
            },
        }
        if unit == "delegation" and field == "basis":
            return _MultiAggregateFieldSQL(
                value=(f"CASE WHEN {row_alias}.instruction_tool_use_block_id IS NULL THEN 'edge' ELSE 'action' END"),
                missing="0",
                unknown="0",
            )
        try:
            raw = raw_fields[unit][field]
        except KeyError as exc:
            raise ValueError(f"unsupported {unit} multi-aggregate group field: {field}") from exc

    assertion_defaults = {
        "status": ASSERTION_DEFAULT_STATUS,
        "visibility": ASSERTION_DEFAULT_VISIBILITY,
        "author_kind": ASSERTION_DEFAULT_AUTHOR_KIND,
    }
    if unit == "assertion" and field in assertion_defaults:
        value = f"CAST(COALESCE(NULLIF({raw}, ''), {_sql_string_literal(assertion_defaults[field])}) AS TEXT)"
        missing = "0"
    else:
        value = f"CASE WHEN {raw} IS NULL THEN '[missing]' ELSE CAST({raw} AS TEXT) END"
        missing = f"CASE WHEN {raw} IS NULL THEN 1 ELSE 0 END"
    unknown = f"CASE WHEN {raw} IS NOT NULL AND LOWER(CAST({raw} AS TEXT)) = 'unknown' THEN 1 ELSE 0 END"
    return _MultiAggregateFieldSQL(value=value, missing=missing, unknown=unknown)


def _query_unit_multi_aggregate_order(
    group_columns: Sequence[str],
    sort: Literal["count", "key"] | None,
    direction: Literal["asc", "desc"],
) -> str:
    """Return the stable ordering used by the former Python tuple sort."""

    sql_direction = _query_unit_order_direction(direction)
    key_order = ", ".join(f"{column} {sql_direction}" for column in group_columns)
    if sort == "key":
        return key_order
    return f"count {sql_direction}, {key_order}"


def _append_query_ctes(prefix_sql: str, *ctes: str) -> str:
    """Append CTEs to an optional relation prefix that already starts WITH."""

    body = ",\n".join(cte.strip() for cte in ctes)
    prefix = prefix_sql.strip()
    if not prefix:
        return f"WITH {body}"
    if not prefix.upper().startswith("WITH "):
        raise ValueError("query relation prefix must start with WITH")
    return f"{prefix},\n{body}"


def _predicate_uses_unit_field(predicate: QueryPredicate, field_name: str, *, unit: str | None = None) -> bool:
    """Return whether a predicate subtree targets a unit-scoped field."""

    if isinstance(predicate, QueryFieldPredicate):
        if predicate.field_ref is not None:
            if predicate.field_ref.name != field_name:
                return False
            return predicate.field_ref.scope == "unit" and (unit is None or predicate.field_ref.unit == unit)
        return predicate.field.removeprefix("session.") == field_name
    if isinstance(predicate, QueryNotPredicate):
        return _predicate_uses_unit_field(predicate.child, field_name, unit=unit)
    if isinstance(predicate, QueryBoolPredicate):
        return any(_predicate_uses_unit_field(child, field_name, unit=unit) for child in predicate.children)
    if isinstance(predicate, QueryExistsPredicate):
        return _predicate_uses_unit_field(predicate.child, field_name, unit=predicate.unit)
    if isinstance(predicate, QuerySequencePredicate):
        return any(_predicate_uses_unit_field(step, field_name, unit="action") for step in predicate.steps)
    return False


def _action_query_needs_followup_relation(predicate: QueryPredicate, *, group_by: str | None = None) -> bool:
    """Return whether an action query needs the derived follow-up relation."""

    return group_by == "followup_class" or _predicate_uses_unit_field(predicate, "followup_class", unit="action")


def _query_unit_group_uses_session(group_by: str | None) -> bool:
    """Return whether an aggregate group expression needs the sessions alias."""

    if group_by is None:
        return False
    return any(
        field.strip().startswith("session.") or field.strip() in {"origin", "repo"} for field in group_by.split(",")
    )


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


def _json_value(value: object, *, default: JSONValue) -> JSONValue:
    try:
        decoded = json.loads(str(value or json.dumps(default)))
    except json.JSONDecodeError:
        return default
    try:
        return require_json_value(decoded)
    except TypeError:
        return default


def _json_str_tuple(value: object) -> tuple[str, ...]:
    decoded = _json_value(value, default=[])
    if not isinstance(decoded, list):
        return ()
    return tuple(str(item) for item in decoded)


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


def _unit_owned_session_identity_clause(
    unit: str,
    row_alias: str,
    predicate: QueryFieldPredicate,
    session_field: str,
) -> tuple[str, list[object]] | None:
    """Push exact owning-session identity onto the unit relation itself.

    A semantically equivalent predicate on a joined ``sessions`` alias is not
    planner-equivalent for views that rank both sides before joining. Keeping
    the bound on the owning relation lets SQLite push it into those branches.
    """
    if session_field not in {"id", "session"} or not predicate.values:
        return None
    expression_template = _UNIT_SESSION_ID_EXPRESSION.get(unit)
    if expression_template is None:
        return None
    return f"{expression_template.format(alias=row_alias)} = ?", [predicate.values[-1]]


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
    """Build a ``time`` field predicate clause.

    A row with no resolvable timestamp (``expression`` evaluates to NULL --
    see :func:`_query_unit_time_expression`) is included rather than
    silently excluded: an unknown time is not evidence the row falls
    outside the requested range (polylogue-z29t, sort_key_ms COALESCE
    audit, .agent/reports/sort-key-ms-coalesce-audit-2026-07-08.md). Before
    this, the expression coalesced to epoch 0, which always failed a
    ``>``/``>=`` comparison (silent exclusion) and always passed a
    ``<``/``<=`` comparison (silent false-inclusion as "old").
    """
    if not predicate.values:
        return "", []
    value_ms = _date_ms(predicate.values[-1], field="time")
    if predicate.op == ">":
        return f"({expression} IS NULL OR {expression} > ?)", [value_ms]
    if predicate.op == ">=":
        return f"({expression} IS NULL OR {expression} >= ?)", [value_ms]
    if predicate.op == "<":
        return f"({expression} IS NULL OR {expression} < ?)", [value_ms]
    if predicate.op == "<=":
        return f"({expression} IS NULL OR {expression} <= ?)", [value_ms]
    raise ValueError("unsupported Boolean query operator for time")


def _query_unit_time_expression(unit: str, row_alias: str) -> str:
    if unit == "message":
        return (
            f"COALESCE({row_alias}.occurred_at_ms, "
            f"(SELECT time_sessions.sort_key_ms FROM sessions time_sessions "
            f"WHERE time_sessions.session_id = {row_alias}.session_id))"
        )
    if unit in {"action", "block"}:
        return (
            f"(SELECT COALESCE(time_messages.occurred_at_ms, time_sessions.sort_key_ms) "
            f"FROM messages time_messages "
            f"JOIN sessions time_sessions ON time_sessions.session_id = time_messages.session_id "
            f"WHERE time_messages.message_id = {row_alias}.message_id "
            f"LIMIT 1)"
        )
    if unit == "file":
        return f"{row_alias}.first_seen_ms"
    if unit == "assertion":
        return f"COALESCE({row_alias}.updated_at_ms, {row_alias}.created_at_ms)"
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
                "command": f"COALESCE({_action_command_expression('filter_actions')}, '')",
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
        return _like_clause(f"COALESCE({_action_command_expression(action_alias)}, '')", predicate.values)
    if field == "path":
        return _like_clause(f"REPLACE(COALESCE({action_alias}.tool_path, ''), char(92), '/')", predicate.values)
    if field == "output":
        return _like_clause(f"COALESCE({action_alias}.output_text, '')", predicate.values)
    if field == "is_error":
        normalized = {value.strip().lower() for value in predicate.values if value.strip()}
        if not normalized:
            return "", []
        truthy = normalized & {"1", "true", "yes", "y", "error", "failed", "failure"}
        falsy = normalized & {"0", "false", "no", "n", "ok", "success", "passed"}
        if truthy and falsy:
            return f"{action_alias}.is_error IN (0, 1)", []
        if truthy:
            return f"{action_alias}.is_error = 1", []
        if falsy:
            return f"{action_alias}.is_error = 0", []
        return "0=1", []
    if field == "exit_code":
        return _numeric_predicate_clause(f"{action_alias}.exit_code", predicate)
    if field == "followup_class":
        return _in_or_equals_clause(f"{action_alias}.followup_class", predicate.values, lower=True)
    if field == "text":
        return _like_clause(
            f"""
            COALESCE({action_alias}.tool_name, '') || ' ' ||
            COALESCE({action_alias}.semantic_type, '') || ' ' ||
            COALESCE({_action_command_expression(action_alias)}, '') || ' ' ||
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
            COALESCE({_action_command_expression(action_alias)}, '')
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
            "command": f"COALESCE({_action_command_expression(block_alias)}, '')",
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
    if field.startswith("value.") and len(field) > len("value."):
        return _assertion_value_path_predicate_clause(assertion_alias, field[len("value.") :], predicate)
    if field == "evidence":
        return _like_clause(f"{assertion_alias}.evidence_refs_json", predicate.values)
    if field == "context":
        default_context_json = json.dumps(ASSERTION_DEFAULT_CONTEXT_POLICY, sort_keys=True, separators=(",", ":"))
        clause, params = _like_clause(f"COALESCE({assertion_alias}.context_policy_json, ?)", predicate.values)
        return clause, [default_context_json, *params]
    raise ValueError(f"unsupported assertion predicate field: {field}")


def _assertion_value_path_predicate_clause(
    assertion_alias: str, path: str, predicate: QueryFieldPredicate
) -> tuple[str, list[object]]:
    """Build a typed JSON-path predicate clause over ``assertions.value_json``.

    ``path`` is a dot-separated JSON-object path below the assertion value
    root (``value.score`` lowers to ``json_extract(value_json, '$.score')``).
    The DSL layer (``_is_assertion_value_path_field``) only accepts plain
    identifier segments, so ``path`` cannot carry SQLite JSON-path
    metacharacters; it is still passed as a bound parameter rather than
    interpolated, so this holds even if that upstream guarantee ever weakens.
    Comparison operators (``>``, ``>=``, ``<``, ``<=``) require both the
    stored JSON scalar and right-hand side to be numeric. Equality preserves
    JSON scalar type, including the distinction between strings such as
    ``"4"``/``"true"``/``"null"`` and their numeric/boolean/null peers.
    """

    if not predicate.values:
        return "", []
    json_path = f"$.{path}"
    extract_expr = f"json_extract({assertion_alias}.value_json, ?)"
    if predicate.op == "=":
        clauses: list[str] = []
        params: list[object] = []
        for raw_value in predicate.values:
            json_types, decoded = _decode_assertion_value_path_literal(raw_value)
            type_placeholders = ", ".join("?" for _ in json_types)
            clauses.append(
                f"(json_type({assertion_alias}.value_json, ?) IN ({type_placeholders}) AND {extract_expr} IS ?)"
            )
            params.extend((json_path, *json_types, json_path, decoded))
        return "(" + " OR ".join(clauses) + ")", params
    op_sql = {">": ">", ">=": ">=", "<": "<", "<=": "<="}[predicate.op]
    raw_value = predicate.values[0]
    return (
        f"(json_type({assertion_alias}.value_json, ?) IN ('integer', 'real') "
        f"AND CAST({extract_expr} AS REAL) {op_sql} ?)",
        [json_path, json_path, float(raw_value)],
    )


def _decode_assertion_value_path_literal(text: str) -> tuple[tuple[str, ...], object]:
    """Decode one scalar DSL literal into accepted SQLite JSON types and value."""

    stripped = text.strip()
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        return ("text",), stripped
    if decoded is True:
        return ("true",), 1
    if decoded is False:
        return ("false",), 0
    if decoded is None:
        return ("null",), None
    if isinstance(decoded, str):
        return ("text",), decoded
    if isinstance(decoded, int):
        if -(2**63) <= decoded <= 2**63 - 1:
            return ("integer", "real"), decoded
        numeric_value = float(decoded)
        if not math.isfinite(numeric_value):
            raise ValueError("assertion value-path equality requires a finite JSON number")
        return ("integer", "real"), numeric_value
    if isinstance(decoded, float):
        if not math.isfinite(decoded):
            raise ValueError("assertion value-path equality requires a finite JSON number")
        return ("integer", "real"), decoded
    raise ValueError("assertion value-path equality requires a JSON scalar")


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


def _delegation_instruction_sql_expression(delegation_alias: str) -> str:
    payload = f"{delegation_alias}.instruction_payload"
    candidates = ", ".join(
        f"NULLIF(CASE WHEN json_type({payload}, '$.{key}') = 'text' THEN json_extract({payload}, '$.{key}') END, '')"
        for key in ("prompt", "description", "instruction", "task")
    )
    return (
        f"CASE WHEN NOT json_valid({payload}) THEN COALESCE({payload}, '') "
        f"WHEN json_type({payload}) = 'object' THEN COALESCE({candidates}, '') "
        "ELSE '' END"
    )


def _delegation_field_predicate_clause(
    delegation_alias: str, predicate: QueryFieldPredicate
) -> tuple[str, list[object]]:
    field = predicate.bound_field_name(context="lowering delegation predicates")
    if field in {"mapping_state", "result_status", "inheritance", "link_method"}:
        return _in_or_equals_clause(f"{delegation_alias}.{field}", predicate.values, lower=True)
    if field == "basis":
        normalized = {value.strip().lower() for value in predicate.values if value.strip()}
        clauses: list[str] = []
        if "action" in normalized:
            clauses.append(f"{delegation_alias}.instruction_tool_use_block_id IS NOT NULL")
        if "edge" in normalized:
            clauses.append(f"{delegation_alias}.instruction_tool_use_block_id IS NULL")
        return ("(" + " OR ".join(clauses) + ")" if clauses else "0=1"), []
    if field in {"parent", "child"}:
        column = "parent_session_id" if field == "parent" else "child_session_id"
        return _like_clause(f"COALESCE({delegation_alias}.{column}, '')", predicate.values)
    if field == "instruction":
        instruction_expr = _delegation_instruction_sql_expression(delegation_alias)
        return _like_clause(instruction_expr, predicate.values)
    if field == "requested_model":
        return _like_clause(f"COALESCE({delegation_alias}.requested_model, '')", predicate.values)
    if field == "dispatch_model":
        return _like_clause(f"COALESCE({delegation_alias}.dispatch_turn_model, '')", predicate.values)
    if field == "child_model":
        return _like_clause(f"COALESCE({delegation_alias}.child_session_dominant_model, '')", predicate.values)
    if field == "is_error":
        normalized = {value.strip().lower() for value in predicate.values if value.strip()}
        truthy = normalized & {"1", "true", "yes", "y", "error", "failed", "failure"}
        falsy = normalized & {"0", "false", "no", "n", "ok", "passed"}
        clauses = []
        if truthy:
            clauses.append(f"{delegation_alias}.result_is_error = 1")
        if falsy:
            clauses.append(f"{delegation_alias}.result_is_error = 0")
        return ("(" + " OR ".join(clauses) + ")" if clauses else "0=1"), []
    if field == "exit_code":
        return _numeric_predicate_clause(f"{delegation_alias}.result_exit_code", predicate)
    if field == "text":
        return _like_clause(
            f"""
            COALESCE({delegation_alias}.parent_session_id, '') || ' ' ||
            COALESCE({delegation_alias}.child_session_id, '') || ' ' ||
            COALESCE({delegation_alias}.instruction_payload, '') || ' ' ||
            COALESCE({delegation_alias}.artifact_text, '') || ' ' ||
            COALESCE({delegation_alias}.dispatch_turn_model, '') || ' ' ||
            COALESCE({delegation_alias}.requested_model, '')
            """.strip(),
            predicate.values,
        )
    raise ValueError(f"unsupported delegation predicate field: {field}")


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
            owned_identity_clause = _unit_owned_session_identity_clause(
                unit,
                row_alias,
                predicate,
                session_field,
            )
            if owned_identity_clause is not None:
                return owned_identity_clause
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
        if unit == "delegation":
            return _delegation_field_predicate_clause(row_alias, predicate)
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
        needs_followup = _action_query_needs_followup_relation(predicate.child)
        child_clause, params = _structural_predicate_clause(
            predicate.unit,
            row_alias,
            predicate.child,
            session_alias=table_alias,
        )
        relation_sql = _ACTION_FOLLOWUP_RELATION_SQL if needs_followup else ""
        relation_name = "action_rows" if needs_followup else "actions"
        return (
            f"""
            EXISTS (
                {relation_sql}
                SELECT 1
                FROM {relation_name} {row_alias}
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
        if predicate.unit == "observed-event":
            source_where, source_params = observed_event_source_pushdown(predicate.child)
            prefix_sql = observed_event_relation_sql(source_where=source_where)
            relation_name = "observed_events"
            row_alias = "exists_observed_events"
            relation_params = source_params
        elif predicate.unit == "run":
            prefix_sql = run_relation_sql()
            relation_name = "runs"
            row_alias = "exists_runs"
            relation_params = []
        else:
            prefix_sql = context_snapshot_relation_sql()
            relation_name = "context_snapshots"
            row_alias = "exists_context_snapshots"
            relation_params = []
        child_clause, params = _structural_predicate_clause(
            predicate.unit,
            row_alias,
            predicate.child,
            session_alias=table_alias,
        )
        return (
            f"""
            EXISTS (
                {prefix_sql}
                SELECT 1
                FROM {relation_name} {row_alias}
                WHERE {row_alias}.session_id = {table_alias}.session_id
                  AND {child_clause}
            )
            """.strip(),
            [*relation_params, *params],
        )
    if predicate.unit == "delegation":
        row_alias = "exists_delegations"
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
                FROM delegations {row_alias}
                WHERE {row_alias}.parent_session_id = {table_alias}.session_id
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


def _logical_predicate_clause(table_alias: str, predicate: QueryLineagePredicate) -> tuple[str, list[object]]:
    """Match every physical session in the seed's materialized logical family."""
    seed_session_id = predicate.seed_session_id.strip()
    if not seed_session_id:
        raise ValueError("logical predicate requires a session id")
    return (
        f"""
        COALESCE(
            (SELECT logical_session_id FROM session_profiles
             WHERE session_id = {table_alias}.session_id),
            {table_alias}.session_id
        ) = (
            SELECT COALESCE(seed_profile.logical_session_id, seed.session_id)
            FROM sessions seed
            LEFT JOIN session_profiles seed_profile ON seed_profile.session_id = seed.session_id
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
        return _action_sequence_steps_clause(table_alias, predicate.steps, predicate.constraints)
    if isinstance(predicate, QueryTextPredicate):
        return _fts_predicate_clause(table_alias, predicate)
    if isinstance(predicate, QueryLineagePredicate):
        if predicate.logical:
            return _logical_predicate_clause(table_alias, predicate)
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
    root: bool | None = None,
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
                      COALESCE({_action_command_expression("filter_actions")}, '') || ' ' ||
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
    if root is True:
        clauses.append(f"{table_alias}.parent_session_id IS NULL")
    elif root is False:
        clauses.append(f"{table_alias}.parent_session_id IS NOT NULL")
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


def _action_sequence_steps_clause(
    table_alias: str,
    steps: tuple[QueryPredicate, ...],
    constraints: tuple[QuerySequenceConstraint, ...] = (),
) -> tuple[str, list[object]]:
    needs_followup = any(_predicate_uses_unit_field(step, "followup_class", unit="action") for step in steps)
    relation_sql = _ACTION_FOLLOWUP_RELATION_SQL if needs_followup else ""
    action_relation = "action_rows" if needs_followup else "actions"
    joins: list[str] = []
    predicates: list[str] = []
    params: list[object] = []
    edge_constraints = constraints or tuple(QuerySequenceConstraint() for _ in range(len(steps) - 1))
    for index, step in enumerate(steps):
        action_alias = f"seq_a{index}"
        message_alias = f"seq_m{index}"
        block_alias = f"seq_b{index}"
        joins.append(
            f"""
            JOIN {action_relation} {action_alias}
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
            constraint = edge_constraints[index - 1]
            if constraint.kind == "next":
                predicates.append(_no_action_between_predicate(index - 1, index, action_relation))
            elif constraint.kind == "within":
                predicates.append(
                    f"{message_alias}.occurred_at_ms IS NOT NULL AND seq_m{index - 1}.occurred_at_ms IS NOT NULL "
                    f"AND {message_alias}.occurred_at_ms >= seq_m{index - 1}.occurred_at_ms "
                    f"AND {message_alias}.occurred_at_ms - seq_m{index - 1}.occurred_at_ms <= ?"
                )
                params.append(constraint.within_ms)
    sql = (
        "EXISTS ("
        f"{relation_sql} "
        "SELECT 1 FROM sessions sequence_root "
        f"{' '.join(joins)} "
        f"WHERE sequence_root.session_id = {table_alias}.session_id "
        f"AND {' AND '.join(predicates)}"
        ")"
    )
    return sql, params


def _action_after_predicate(previous: int | str, current: int | str) -> str:
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


def _no_action_between_predicate(previous: int, current: int, action_relation: str) -> str:
    after_previous = _action_after_predicate(previous, "between")
    before_current = _action_after_predicate("between", current)
    return (
        "NOT EXISTS ("
        f"SELECT 1 FROM {action_relation} seq_abetween "
        "JOIN messages seq_mbetween ON seq_mbetween.message_id = seq_abetween.message_id "
        "JOIN blocks seq_bbetween ON seq_bbetween.block_id = seq_abetween.tool_use_block_id "
        f"WHERE seq_abetween.session_id = seq_a{previous}.session_id "
        f"AND {after_previous} AND {before_current}"
        ")"
    )


def query_messages(
    self: _ArchiveQueryReadsHost,
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
        order_by = f"COALESCE(m.occurred_at_ms, s.sort_key_ms) {order_direction}, m.message_id {order_direction}"
    else:
        order_by = "COALESCE(m.occurred_at_ms, s.sort_key_ms), m.message_id"
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
            (SELECT group_concat(repo_name, ', ')
             FROM (
                 SELECT r.repo_name
                 FROM session_repos sr
                 JOIN repos r ON r.repo_id = sr.repo_id
                 WHERE sr.session_id = m.session_id
                 ORDER BY r.repo_name
             )) AS repo,
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
    message_ids = tuple(str(row["message_id"]) for row in rows)
    blocks_by_message = _fetch_blocks_for_messages(self._conn, message_ids)
    return [
        ArchiveMessageQueryRow(
            message_id=str(row["message_id"]),
            session_id=str(row["session_id"]),
            origin=str(row["origin"]),
            title=str(row["title"]) if row["title"] is not None else None,
            repo=str(row["repo"]) if row["repo"] is not None else None,
            role=str(row["role"]),
            message_type=str(row["message_type"]),
            material_origin=str(row["material_origin"]),
            occurred_at_ms=int(row["occurred_at_ms"]) if row["occurred_at_ms"] is not None else None,
            position=int(row["position"]),
            word_count=int(row["word_count"]),
            text=str(row["text"] or ""),
            blocks=tuple(blocks_by_message[str(row["message_id"])]),
        )
        for row in rows
    ]


def query_session_messages(
    self: _ArchiveQueryReadsHost,
    session_ids: Sequence[str],
    *,
    limit: int = 50,
    offset: int = 0,
    sort_direction: Literal["asc", "desc"] = "asc",
    roles: Sequence[str] = (),
    message_type: str | None = None,
    material_origins: Sequence[str] = (),
) -> list[ArchiveMessageQueryRow]:
    """Return message rows for known sessions using the session sort-key index."""

    normalized_session_ids = tuple(
        dict.fromkeys(session_id.strip() for session_id in session_ids if session_id.strip())
    )
    if not normalized_session_ids:
        return []
    normalized_limit = max(int(limit), 0)
    normalized_offset = max(int(offset), 0)
    order_direction = _query_unit_order_direction(sort_direction)
    placeholders = ", ".join("?" for _ in normalized_session_ids)
    predicates = [f"m.session_id IN ({placeholders})"]
    filter_params: list[object] = [*normalized_session_ids]
    normalized_roles = tuple(dict.fromkeys(str(role) for role in roles if str(role)))
    if normalized_roles:
        role_placeholders = ", ".join("?" for _ in normalized_roles)
        predicates.append(f"m.role IN ({role_placeholders})")
        filter_params.extend(normalized_roles)
    if message_type is not None:
        predicates.append("m.message_type = ?")
        filter_params.append(str(message_type))
    normalized_origins = tuple(dict.fromkeys(str(origin) for origin in material_origins if str(origin)))
    if normalized_origins:
        origin_placeholders = ", ".join("?" for _ in normalized_origins)
        predicates.append(f"m.material_origin IN ({origin_placeholders})")
        filter_params.extend(normalized_origins)
    rows = self._conn.execute(
        f"""
        SELECT
            m.message_id,
            m.session_id,
            s.origin,
            s.title,
            (SELECT group_concat(repo_name, ', ')
             FROM (
                 SELECT r.repo_name
                 FROM session_repos sr
                 JOIN repos r ON r.repo_id = sr.repo_id
                 WHERE sr.session_id = m.session_id
                 ORDER BY r.repo_name
             )) AS repo,
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
        FROM messages m INDEXED BY idx_messages_session_sortkey
        JOIN sessions s ON s.session_id = m.session_id
        WHERE {" AND ".join(predicates)}
        ORDER BY m.position {order_direction}, m.variant_index {order_direction}, m.message_id {order_direction}
        LIMIT ? OFFSET ?
        """,
        [*filter_params, normalized_limit, normalized_offset],
    ).fetchall()
    message_ids = tuple(str(row["message_id"]) for row in rows)
    blocks_by_message = _fetch_blocks_for_messages(self._conn, message_ids)
    return [
        ArchiveMessageQueryRow(
            message_id=str(row["message_id"]),
            session_id=str(row["session_id"]),
            origin=str(row["origin"]),
            title=str(row["title"]) if row["title"] is not None else None,
            repo=str(row["repo"]) if row["repo"] is not None else None,
            role=str(row["role"]),
            message_type=str(row["message_type"]),
            material_origin=str(row["material_origin"]),
            occurred_at_ms=int(row["occurred_at_ms"]) if row["occurred_at_ms"] is not None else None,
            position=int(row["position"]),
            word_count=int(row["word_count"]),
            text=str(row["text"] or ""),
            blocks=tuple(blocks_by_message[str(row["message_id"])]),
        )
        for row in rows
    ]


def count_session_messages(
    self: _ArchiveQueryReadsHost,
    session_ids: Sequence[str],
    *,
    roles: Sequence[str] = (),
    message_type: str | None = None,
    material_origins: Sequence[str] = (),
) -> int:
    """Return an exact count for a bounded session-message projection."""

    normalized_session_ids = tuple(
        dict.fromkeys(session_id.strip() for session_id in session_ids if session_id.strip())
    )
    if not normalized_session_ids:
        return 0
    placeholders = ", ".join("?" for _ in normalized_session_ids)
    predicates = [f"session_id IN ({placeholders})"]
    params: list[object] = [*normalized_session_ids]
    normalized_roles = tuple(dict.fromkeys(str(role) for role in roles if str(role)))
    if normalized_roles:
        role_placeholders = ", ".join("?" for _ in normalized_roles)
        predicates.append(f"role IN ({role_placeholders})")
        params.extend(normalized_roles)
    if message_type is not None:
        predicates.append("message_type = ?")
        params.append(str(message_type))
    normalized_origins = tuple(dict.fromkeys(str(origin) for origin in material_origins if str(origin)))
    if normalized_origins:
        origin_placeholders = ", ".join("?" for _ in normalized_origins)
        predicates.append(f"material_origin IN ({origin_placeholders})")
        params.extend(normalized_origins)
    row = self._conn.execute(
        f"SELECT COUNT(*) AS count FROM messages WHERE {' AND '.join(predicates)}", params
    ).fetchone()
    return int(row["count"]) if row is not None else 0


def query_unit_counts(
    self: _ArchiveQueryReadsHost,
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
    if unit == "assertion":
        self.require_user_tier()

    row_alias = _QUERY_UNIT_ROW_ALIAS.get(unit)
    if row_alias is None:
        raise ValueError(f"Query unit {unit!r} is not wired to SQL aggregate counts")
    if unit == "file":
        return _query_file_counts(
            self,
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
    action_needs_followup = unit == "action" and _action_query_needs_followup_relation(predicate, group_by=group_by)
    action_prefix_sql = ""
    action_relation_name = "actions"
    action_relation_params: list[object] = []
    if unit == "action":
        action_prefix_sql, action_relation_name, action_relation_params = _action_relation_for_query(
            predicate=predicate,
            include_followup=action_needs_followup,
        )
    from_sql_by_unit = _query_unit_from_sql_by_unit(action_relation_name)
    from_sql = "observed_events e" if unit == "observed-event" and not needs_session else from_sql_by_unit[unit]
    order_clause = _query_unit_aggregate_order(sort, sort_direction)
    source_where = "0=1"
    source_params: list[object] = []
    if unit == "observed-event":
        source_where, source_params = observed_event_source_pushdown(predicate)
    if unit == "observed-event":
        prefix_sql = observed_event_relation_sql(source_where=source_where)
    elif unit == "action":
        prefix_sql = action_prefix_sql
    else:
        prefix_sql = ""
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
        [
            *action_relation_params,
            *source_params,
            *params,
            *session_params,
            normalized_limit,
            normalized_offset,
        ],
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


def query_unit_multi_counts(
    self: _ArchiveQueryReadsHost,
    unit: str,
    predicate: QueryPredicate,
    *,
    group_by: Sequence[str],
    sort: Literal["count", "key"] | None = None,
    sort_direction: Literal["asc", "desc"] = "desc",
    limit: int = 50,
    offset: int = 0,
    session_filters: Mapping[str, object] | None = None,
) -> ArchiveQueryUnitMultiAggregatePage:
    """Return one bounded multi-field count page with exact full-set facts.

    The source relation is filtered and grouped inside one SQLite statement.
    Python retains at most the requested aggregate page; it never pages the
    selected row relation with OFFSET and never reconstructs the query
    algorithm in application memory.
    """

    fields = tuple(str(field).strip() for field in group_by if str(field).strip())
    if len(fields) < 2:
        raise ValueError("multi-field aggregate counts require at least two group fields")
    normalized_limit = max(int(limit), 0)
    normalized_offset = max(int(offset), 0)
    empty_page = ArchiveQueryUnitMultiAggregatePage(
        rows=(),
        denominator=0,
        missing_counts=tuple(0 for _ in fields),
        unknown_counts=tuple(0 for _ in fields),
    )
    if unit == "assertion":
        self.require_user_tier()

    row_alias = _QUERY_UNIT_ROW_ALIAS.get(unit)
    if row_alias is None:
        raise ValueError(f"Query unit {unit!r} is not wired to SQL multi-aggregate counts")

    active_session_filters = _session_filter_is_active(session_filters)
    needs_session = (
        unit != "observed-event"
        or active_session_filters
        or any(_query_unit_group_uses_session(field) for field in fields)
        or _predicate_uses_session_scope(predicate)
    )
    session_alias = "s" if needs_session else None
    clause, predicate_params = _structural_predicate_clause(
        unit,
        "a" if unit == "file" else row_alias,
        predicate,
        session_alias=session_alias,
    )
    where_clause = clause or "1=1"
    session_clause = ""
    session_params: list[object] = []
    if needs_session and active_session_filters and session_filters is not None:
        session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)

    relation_params: list[object] = []
    source_params: list[object] = []
    prefix_sql = ""
    selected_where_clause = where_clause
    selected_session_clause = session_clause
    if unit == "file":
        file_rows_cte = f"""
            file_rows AS (
                SELECT
                    a.session_id,
                    REPLACE(a.tool_path, char(92), '/') AS path
                FROM actions a
                JOIN sessions s ON s.session_id = a.session_id
                JOIN messages m ON m.message_id = a.message_id
                WHERE a.tool_path IS NOT NULL
                  AND a.tool_path != ''
                  AND {where_clause}
                  {session_clause}
                GROUP BY a.session_id, path
            )
        """
        from_sql = "file_rows f JOIN sessions s ON s.session_id = f.session_id"
        source_ctes = [file_rows_cte]
        selected_where_clause = "1=1"
        selected_session_clause = ""
    else:
        source_ctes = []
        action_needs_followup = unit == "action" and (
            "followup_class" in fields or _action_query_needs_followup_relation(predicate)
        )
        action_relation_name = "actions"
        if unit == "action":
            prefix_sql, action_relation_name, relation_params = _action_relation_for_query(
                predicate=predicate,
                include_followup=action_needs_followup,
            )
        from_sql_by_unit = _query_unit_from_sql_by_unit(action_relation_name)
        from_sql = "observed_events e" if unit == "observed-event" and not needs_session else from_sql_by_unit[unit]
        if unit == "observed-event":
            source_where, source_params = observed_event_source_pushdown(predicate)
            prefix_sql = observed_event_relation_sql(source_where=source_where)

    field_sql = tuple(_query_unit_multi_group_field_sql(unit, row_alias, field) for field in fields)
    group_columns = tuple(f"group_{index}" for index in range(len(fields)))
    selected_columns = ",\n".join(
        expression
        for index, spec in enumerate(field_sql)
        for expression in (
            f"{spec.value} AS group_{index}",
            f"{spec.missing} AS missing_{index}",
            f"{spec.unknown} AS unknown_{index}",
        )
    )
    grouped_quality_columns = ",\n".join(
        expression
        for index in range(len(fields))
        for expression in (
            f"SUM(missing_{index}) AS group_missing_{index}",
            f"SUM(unknown_{index}) AS group_unknown_{index}",
        )
    )
    ranked_quality_columns = ",\n".join(
        expression
        for index in range(len(fields))
        for expression in (
            f"SUM(group_missing_{index}) OVER () AS total_missing_{index}",
            f"SUM(group_unknown_{index}) OVER () AS total_unknown_{index}",
        )
    )
    stats_columns = ",\n".join(
        expression
        for index in range(len(fields))
        for expression in (
            f"COALESCE(MAX(total_missing_{index}), 0) AS missing_{index}",
            f"COALESCE(MAX(total_unknown_{index}), 0) AS unknown_{index}",
        )
    )
    final_stats_columns = ",\n".join(
        expression
        for index in range(len(fields))
        for expression in (
            f"stats.missing_{index}",
            f"stats.unknown_{index}",
        )
    )
    final_group_columns = ",\n".join(f"page.group_{index}" for index in range(len(fields)))
    group_column_list = ", ".join(group_columns)
    order_clause = _query_unit_multi_aggregate_order(group_columns, sort, sort_direction)

    selected_cte = f"""
        selected AS (
            SELECT
                {selected_columns}
            FROM {from_sql}
            WHERE {selected_where_clause}
            {selected_session_clause}
        )
    """
    grouped_cte = f"""
        grouped AS (
            SELECT
                {group_column_list},
                COUNT(*) AS count,
                {grouped_quality_columns}
            FROM selected
            GROUP BY {group_column_list}
        )
    """
    ranked_cte = f"""
        ranked AS (
            SELECT
                grouped.*,
                SUM(count) OVER () AS denominator,
                {ranked_quality_columns},
                ROW_NUMBER() OVER (ORDER BY {order_clause}) AS ordinal
            FROM grouped
        )
    """
    page_cte = """
        page AS (
            SELECT *
            FROM ranked
            WHERE ordinal > ? AND ordinal <= ?
        )
    """
    stats_cte = f"""
        stats AS (
            SELECT
                COALESCE(MAX(denominator), 0) AS denominator,
                {stats_columns}
            FROM ranked
        )
    """
    cte_sql = _append_query_ctes(
        prefix_sql,
        *source_ctes,
        selected_cte,
        grouped_cte,
        ranked_cte,
        page_cte,
        stats_cte,
    )
    rows = self._conn.execute(
        f"""
        {cte_sql}
        SELECT
            stats.denominator,
            {final_stats_columns},
            {final_group_columns},
            page.count,
            page.ordinal
        FROM stats
        LEFT JOIN page ON 1 = 1
        ORDER BY page.ordinal
        """,
        [
            *relation_params,
            *source_params,
            *predicate_params,
            *session_params,
            normalized_offset,
            normalized_offset + normalized_limit,
        ],
    ).fetchall()
    if not rows:
        return empty_page
    stats_row = rows[0]
    aggregate_rows = tuple(
        ArchiveQueryUnitMultiAggregateRow(
            unit=unit,
            group_by=fields,
            group_values=tuple(str(row[f"group_{index}"]) for index in range(len(fields))),
            count=int(row["count"]),
        )
        for row in rows
        if row["count"] is not None
    )
    return ArchiveQueryUnitMultiAggregatePage(
        rows=aggregate_rows,
        denominator=int(stats_row["denominator"]),
        missing_counts=tuple(int(stats_row[f"missing_{index}"]) for index in range(len(fields))),
        unknown_counts=tuple(int(stats_row[f"unknown_{index}"]) for index in range(len(fields))),
    )


def query_actions(
    self: _ArchiveQueryReadsHost,
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
        order_by = f"COALESCE(m.occurred_at_ms, s.sort_key_ms) {order_direction}, a.tool_use_block_id {order_direction}"
    else:
        order_by = "COALESCE(m.occurred_at_ms, s.sort_key_ms), a.tool_use_block_id"
    clause, params = _structural_predicate_clause("action", "a", predicate, session_alias="s")
    clause = clause or "1=1"
    session_clause = ""
    session_params: list[object] = []
    if session_filters:
        session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
    if _action_query_needs_followup_relation(predicate):
        # A follow-up-class predicate needs the derived relation to decide
        # membership. Keep that compatibility path until the derived
        # relation has its own selective index.
        prefix_sql, action_relation_name, relation_params = _action_relation_for_query(
            predicate=predicate,
            include_followup=True,
        )
        outer_clause = clause
        outer_session_clause = session_clause
        outer_paging = "LIMIT ? OFFSET ?"
        query_params = [*relation_params, *params, *session_params, normalized_limit, normalized_offset]
    else:
        # Follow-up detail is part of the row projection, but ordinary
        # action predicates do not need it to choose the page. Select the
        # ordered page from the indexed base relation first, then derive
        # follow-up detail for only those rows. The old shape ran the
        # correlated follow-up CTE over every archive action before this
        # LIMIT/OFFSET, turning a selective MCP read into an archive-wide
        # materialization.
        selected_actions_cte = f"""
            selected_actions AS (
                SELECT a.*
                FROM actions a
                JOIN sessions s ON s.session_id = a.session_id
                JOIN messages m ON m.message_id = a.message_id
                WHERE {clause}
                {session_clause}
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
            )
        """
        followup_ctes = _ACTION_FOLLOWUP_RELATION_SQL.strip().removeprefix("WITH ")
        prefix_sql = (
            f"WITH {selected_actions_cte},\n{followup_ctes.replace('FROM actions a', 'FROM selected_actions a', 1)}"
        )
        action_relation_name = "action_rows"
        outer_clause = "1=1"
        outer_session_clause = ""
        outer_paging = ""
        query_params = [*params, *session_params, normalized_limit, normalized_offset]
    rows = self._conn.execute(
        f"""
        {prefix_sql}
        SELECT
            {_ARCHIVE_ACTION_QUERY_SELECT_SQL}
        FROM {action_relation_name} a
        JOIN sessions s ON s.session_id = a.session_id
        JOIN messages m ON m.message_id = a.message_id
        WHERE {outer_clause}
        {outer_session_clause}
        ORDER BY {order_by}
        {outer_paging}
        """,
        query_params,
    ).fetchall()
    return [_archive_action_query_row(row) for row in rows]


def query_session_actions(
    self: _ArchiveQueryReadsHost,
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
    prefix_sql, action_relation_name, relation_params = _action_relation_for_query(
        session_ids=normalized_session_ids,
        include_followup=True,
    )
    rows = self._conn.execute(
        f"""
        {prefix_sql}
        SELECT
            {_ARCHIVE_ACTION_QUERY_SELECT_SQL}
        FROM {action_relation_name} a
        JOIN sessions s ON s.session_id = a.session_id
        JOIN messages m ON m.message_id = a.message_id
        WHERE a.session_id IN ({placeholders})
        ORDER BY COALESCE(m.occurred_at_ms, s.sort_key_ms) {order_direction},
                 a.tool_use_block_id {order_direction}
        LIMIT ? OFFSET ?
        """,
        [*relation_params, *normalized_session_ids, normalized_limit, normalized_offset],
    ).fetchall()
    return [_archive_action_query_row(row) for row in rows]


def query_session_action_occurrences(
    self: _ArchiveQueryReadsHost,
    session_ids: Sequence[str],
    *,
    limit: int = 50,
    offset: int = 0,
    sort_direction: Literal["asc", "desc"] = "asc",
) -> list[ArchiveActionQueryRow]:
    """Return lightweight action occurrence rows for known sessions.

    This intentionally skips follow-up classification. Temporal read views
    only need occurrence evidence, and the full follow-up relation can be
    expensive on very large sessions.
    """

    normalized_session_ids = tuple(
        dict.fromkeys(session_id.strip() for session_id in session_ids if session_id.strip())
    )
    if not normalized_session_ids:
        return []
    normalized_limit = max(int(limit), 0)
    normalized_offset = max(int(offset), 0)
    order_direction = _query_unit_order_direction(sort_direction)
    placeholders = ", ".join("?" for _ in normalized_session_ids)
    prefix_sql, action_relation_name, relation_params = _action_relation_for_query(
        session_ids=normalized_session_ids,
        include_followup=False,
    )
    rows = self._conn.execute(
        f"""
        {prefix_sql}
        SELECT
            a.session_id,
            a.message_id,
            s.origin,
            s.title,
            a.tool_use_block_id,
            a.tool_result_block_id,
            a.tool_name,
            a.semantic_type,
            {_action_command_expression("a")} AS tool_command,
            a.tool_path,
            m.occurred_at_ms,
            a.output_text,
            a.is_error,
            a.exit_code,
            a.result_state,
            NULL AS followup_class,
            NULL AS followup_message_ref
        FROM {action_relation_name} a
        JOIN sessions s ON s.session_id = a.session_id
        JOIN messages m ON m.message_id = a.message_id
        WHERE a.session_id IN ({placeholders})
        ORDER BY COALESCE(m.occurred_at_ms, s.sort_key_ms) {order_direction},
                 a.tool_use_block_id {order_direction}
        LIMIT ? OFFSET ?
        """,
        [*relation_params, *normalized_session_ids, normalized_limit, normalized_offset],
    ).fetchall()
    return [_archive_action_query_row(row) for row in rows]


def get_delegation_attempt(
    self: _ArchiveQueryReadsHost,
    *,
    instruction_tool_use_block_id: str | None = None,
    parent_session_id: str | None = None,
    child_session_id: str | None = None,
) -> ArchiveDelegationQueryRow | None:
    """Resolve one `delegations` row (polylogue-y964) by its ref identity.

    Action-observed identity (resolved/unresolved): pass only
    ``instruction_tool_use_block_id``. Edge identity (edge_only,
    quarantined, or authority-contradicted, all with no parent-side
    dispatch action to key off): pass both ``parent_session_id`` and
    ``child_session_id``. Only rows with no instruction are eligible, so
    this path never shadows an action-observed row for the same pair.
    """

    if instruction_tool_use_block_id is not None:
        row = self._conn.execute(
            "SELECT * FROM delegations WHERE instruction_tool_use_block_id = ? LIMIT 1",
            (instruction_tool_use_block_id,),
        ).fetchone()
    elif parent_session_id is not None and child_session_id is not None:
        row = self._conn.execute(
            """
            SELECT * FROM delegations
            WHERE parent_session_id = ? AND child_session_id = ?
              AND mapping_state IN ('edge_only', 'quarantined', 'authority-contradicted')
            LIMIT 1
            """,
            (parent_session_id, child_session_id),
        ).fetchone()
    else:
        raise ValueError(
            "get_delegation_attempt requires either instruction_tool_use_block_id or both "
            "parent_session_id and child_session_id"
        )
    return None if row is None else _archive_delegation_query_row(row)


def get_delegation_card(
    self: _ArchiveQueryReadsHost,
    *,
    instruction_tool_use_block_id: str | None = None,
    parent_session_id: str | None = None,
    child_session_id: str | None = None,
) -> ArchiveDelegationCard | None:
    """Return the explicit bounded evidence card for one delegation."""

    attempt = get_delegation_attempt(
        self,
        instruction_tool_use_block_id=instruction_tool_use_block_id,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
    )
    if attempt is None:
        return None
    if attempt.instruction_tool_use_block_id is not None:
        delegation_ref = f"delegation:{attempt.instruction_tool_use_block_id}"
    else:
        if attempt.child_session_id is None:
            raise ValueError("edge-identified delegation card requires a child session id")
        delegation_ref = "delegation:" + delegation_edge_object_id(attempt.parent_session_id, attempt.child_session_id)

    title_row = self._conn.execute(
        """
        SELECT p.title AS parent_title, c.title AS child_title
        FROM sessions p
        LEFT JOIN sessions c ON c.session_id = ?
        WHERE p.session_id = ?
        """,
        (attempt.child_session_id, attempt.parent_session_id),
    ).fetchone()
    parent_title = (
        str(title_row["parent_title"]) if title_row is not None and title_row["parent_title"] is not None else None
    )
    child_title = (
        str(title_row["child_title"]) if title_row is not None and title_row["child_title"] is not None else None
    )

    run_ref: str | None = None
    run_title: str | None = None
    if attempt.child_session_id is not None:
        # source-derived run_relation_sql() (polylogue-dab) keys a
        # subagent run's own `session_id`/`native_session_id` to the
        # subagent's own session, not the parent -- unlike the old
        # materialized writer, which grouped subagent run rows under
        # the parent's session_id. Matching directly on
        # native_session_id = child_session_id is therefore both
        # sufficient and simpler than the old parent+native_session_id
        # pairing. The old evidence_refs_json block-id fallback (for
        # when only tool-use/artifact block ids are known, no resolved
        # child_session_id) has no equivalent here: the source-derived
        # CTE's evidence_refs_json carries only the owning session id,
        # not per-block references, so that fallback path is not
        # reconstructed -- native_session_id matching covers the
        # common case where child_session_id is already resolved.
        run_row = self._conn.execute(
            f"""
            {run_relation_sql()}
            SELECT run_ref, title
            FROM runs
            WHERE native_session_id = ? AND role = 'subagent'
            ORDER BY position, run_ref
            LIMIT 1
            """,
            (attempt.child_session_id,),
        ).fetchone()
        if run_row is not None:
            run_ref = str(run_row["run_ref"])
            if run_row["title"]:
                run_title = str(run_row["title"])

    instruction_position: int | None = None
    if attempt.instruction_message_id is not None:
        position_row = self._conn.execute(
            "SELECT position FROM messages WHERE message_id = ?",
            (attempt.instruction_message_id,),
        ).fetchone()
        if position_row is not None:
            instruction_position = int(position_row["position"])

    artifact_position: int | None = None
    if attempt.artifact_block_id is not None:
        artifact_row = self._conn.execute(
            """
            SELECT m.position
            FROM blocks b
            JOIN messages m ON m.message_id = b.message_id
            WHERE b.block_id = ?
            """,
            (attempt.artifact_block_id,),
        ).fetchone()
        if artifact_row is not None:
            artifact_position = int(artifact_row["position"])

    if instruction_position is not None:
        parent_context, parent_context_truncated = _delegation_message_window(
            self._conn,
            session_id=attempt.parent_session_id,
            anchor_position=instruction_position,
            before=True,
        )
    else:
        parent_context, parent_context_truncated = (), False
    followup_anchor = artifact_position if artifact_position is not None else instruction_position
    if followup_anchor is not None:
        parent_followup, parent_followup_truncated = _delegation_message_window(
            self._conn,
            session_id=attempt.parent_session_id,
            anchor_position=followup_anchor,
            before=False,
        )
    else:
        parent_followup, parent_followup_truncated = (), False

    dispatch_result, dispatch_result_truncated = _bounded_delegation_card_text(
        attempt.artifact_text,
        limit=4000,
    )
    child_excerpt_source: str | None = None
    child_excerpt_message_id: str | None = None
    if attempt.child_session_id is not None:
        child_row = self._conn.execute(
            """
            SELECT
                m.message_id,
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
            WHERE m.session_id = ? AND m.role = 'assistant'
            ORDER BY m.position DESC, m.message_id DESC
            LIMIT 1
            """,
            (attempt.child_session_id,),
        ).fetchone()
        if child_row is not None:
            child_excerpt_source = str(child_row["text"] or "")
            child_excerpt_message_id = str(child_row["message_id"])
    child_excerpt, child_excerpt_truncated = _bounded_delegation_card_text(
        child_excerpt_source,
        limit=4000,
    )

    annotation_refs: tuple[str, ...] = ()
    if self.user_db_path.exists():
        self._attach_user_tier_if_present()
        assertion_rows = self._conn.execute(
            """
            SELECT assertion_id
            FROM user_tier.assertions
            WHERE target_ref = ?
            ORDER BY updated_at_ms DESC, assertion_id
            LIMIT 20
            """,
            (delegation_ref,),
        ).fetchall()
        annotation_refs = tuple(f"assertion:{row['assertion_id']}" for row in assertion_rows)

    evidence_refs: list[str] = []
    if attempt.instruction_tool_use_block_id is not None:
        evidence_refs.append(f"block:{attempt.instruction_tool_use_block_id}")
    elif attempt.instruction_message_id is not None:
        evidence_refs.append(f"message:{attempt.instruction_message_id}")
    if attempt.artifact_block_id is not None:
        evidence_refs.append(f"block:{attempt.artifact_block_id}")
    if child_excerpt_message_id is not None:
        evidence_refs.append(f"message:{child_excerpt_message_id}")
    evidence_refs.extend(f"message:{row.message_id}" for row in parent_context)
    evidence_refs.extend(f"message:{row.message_id}" for row in parent_followup)

    return ArchiveDelegationCard(
        attempt=attempt,
        delegation_ref=delegation_ref,
        parent_session_title=parent_title,
        child_session_title=child_title,
        run_ref=run_ref,
        run_title=run_title,
        instruction=_delegation_instruction(attempt.instruction_payload),
        parent_context=parent_context,
        parent_context_truncated=parent_context_truncated,
        dispatch_result=dispatch_result,
        dispatch_result_truncated=dispatch_result_truncated,
        child_excerpt=child_excerpt,
        child_excerpt_truncated=child_excerpt_truncated,
        parent_followup=parent_followup,
        parent_followup_truncated=parent_followup_truncated,
        annotation_refs=annotation_refs,
        evidence_refs=tuple(dict.fromkeys(evidence_refs)),
    )


def query_delegations(
    self: _ArchiveQueryReadsHost,
    predicate: QueryPredicate,
    *,
    limit: int = 50,
    offset: int = 0,
    session_filters: Mapping[str, object] | None = None,
    sort: None = None,
    sort_direction: Literal["asc", "desc"] = "asc",
) -> list[ArchiveDelegationQueryRow]:
    """Return delegation attempts without inferring child utility or success."""

    if sort is not None:
        raise ValueError("delegation rows do not expose an honest time sort")
    normalized_limit = max(int(limit), 0)
    normalized_offset = max(int(offset), 0)
    order_direction = _query_unit_order_direction(sort_direction)
    clause, params = _structural_predicate_clause("delegation", "d", predicate, session_alias="s")
    where_clause = f"WHERE {clause}" if clause else ""
    session_clause = ""
    session_params: list[object] = []
    if session_filters:
        session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
    rows = self._conn.execute(
        f"""
        SELECT d.*
        FROM delegations d
        JOIN sessions s ON s.session_id = d.parent_session_id
        {where_clause}
        {session_clause}
        ORDER BY d.parent_session_id {order_direction},
                 COALESCE(d.instruction_tool_use_block_id, d.child_session_id) {order_direction}
        LIMIT ? OFFSET ?
        """,
        [*params, *session_params, normalized_limit, normalized_offset],
    ).fetchall()
    return [_archive_delegation_query_row(row) for row in rows]


def get_delegation_ancestry(self: _ArchiveQueryReadsHost, session_id: str) -> list[ArchiveDelegationAncestryRow]:
    """Return the full root-to-node ancestry chain for ``session_id`` in
    one recursive-CTE call, depth-annotated (polylogue-qsb4). The queried
    session is always the last row (``depth=0``); its dispatchers follow
    at increasing depth, ordered root-first. Quarantined (cycle-break)
    edges are never traversed. Returns a single-row list (just the
    origin) when ``session_id`` was never dispatched by anything. Quarantined
    and authority-contradicted edges are never composed into ancestry."""

    rows = self._conn.execute(_DELEGATION_ANCESTRY_SQL, (session_id, session_id)).fetchall()
    return [_archive_delegation_ancestry_row(row) for row in rows]


def get_delegation_subtree(self: _ArchiveQueryReadsHost, session_id: str) -> list[ArchiveDelegationSubtreeRow]:
    """Return the full subtree (``session_id`` plus all transitive
    dispatch descendants) in one recursive-CTE call, depth-annotated
    (polylogue-qsb4). The queried session is always the first row
    (``depth=0``), ordered breadth-first thereafter. Quarantined
    (cycle-break) and authority-contradicted edges are never traversed.
    Returns a single-row list (just the root) when ``session_id`` never
    dispatched anything."""

    rows = self._conn.execute(_DELEGATION_SUBTREE_SQL, (session_id, session_id)).fetchall()
    return [_archive_delegation_subtree_row(row) for row in rows]


def query_files(
    self: _ArchiveQueryReadsHost,
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
        order_by = f"f.first_seen_ms {order_direction}, f.path {order_direction}"
    else:
        order_by = "f.path, f.first_seen_ms"
    clause, params = _structural_predicate_clause("file", "a", predicate, session_alias="s")
    session_clause = ""
    session_params: list[object] = []
    if session_filters:
        session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
    rows = self._conn.execute(
        f"""
        SELECT
            {_ARCHIVE_FILE_QUERY_SELECT_SQL}
        FROM (
            SELECT
                a.session_id,
                REPLACE(a.tool_path, char(92), '/') AS path,
                COUNT(*) AS action_count,
                MIN(a.message_id) AS first_message_id,
                MIN(a.tool_use_block_id) AS first_tool_use_block_id,
                MAX(a.tool_use_block_id) AS last_tool_use_block_id,
                MIN(COALESCE(m.occurred_at_ms, s.sort_key_ms)) AS first_seen_ms,
                MAX(COALESCE(m.occurred_at_ms, s.sort_key_ms)) AS last_seen_ms
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
    return [_hydrate_archive_file_query_row(row) for row in rows]


def query_session_files(
    self: _ArchiveQueryReadsHost,
    session_ids: Sequence[str],
    *,
    limit: int = 50,
    offset: int = 0,
    sort_direction: Literal["asc", "desc"] = "asc",
) -> list[ArchiveFileQueryRow]:
    """Return affected file-path rows for known sessions using indexed tool-use blocks."""

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
            {_ARCHIVE_FILE_QUERY_SELECT_SQL}
        FROM (
            SELECT
                u.session_id,
                REPLACE(u.tool_path, char(92), '/') AS path,
                COUNT(*) AS action_count,
                MIN(u.message_id) AS first_message_id,
                MIN(u.block_id) AS first_tool_use_block_id,
                MAX(u.block_id) AS last_tool_use_block_id,
                MIN(COALESCE(m.occurred_at_ms, s.sort_key_ms)) AS first_seen_ms,
                MAX(COALESCE(m.occurred_at_ms, s.sort_key_ms)) AS last_seen_ms
            FROM blocks u INDEXED BY idx_blocks_session_position
            JOIN sessions s ON s.session_id = u.session_id
            JOIN messages m ON m.message_id = u.message_id
            WHERE u.session_id IN ({placeholders})
              AND u.block_type = 'tool_use'
              AND u.tool_path IS NOT NULL
              AND u.tool_path != ''
            GROUP BY u.session_id, path
        ) f
        JOIN sessions s ON s.session_id = f.session_id
        ORDER BY f.first_seen_ms {order_direction},
                 f.path {order_direction}
        LIMIT ? OFFSET ?
        """,
        [*normalized_session_ids, normalized_limit, normalized_offset],
    ).fetchall()
    return [_hydrate_archive_file_query_row(row) for row in rows]


def _query_file_counts(
    self: _ArchiveQueryReadsHost,
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
    self: _ArchiveQueryReadsHost,
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
        order_by = f"COALESCE(m.occurred_at_ms, s.sort_key_ms) {order_direction}, b.block_id {order_direction}"
    else:
        order_by = "COALESCE(m.occurred_at_ms, s.sort_key_ms), b.block_id"
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
            {_action_command_expression("b")} AS tool_command,
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
    self: _ArchiveQueryReadsHost,
    predicate: QueryPredicate,
    *,
    limit: int = 50,
    offset: int = 0,
    session_filters: Mapping[str, object] | None = None,
    sort: Literal["time"] | None = None,
    sort_direction: Literal["asc", "desc"] = "asc",
) -> list[ArchiveAssertionQueryRow]:
    """Return user-tier assertion rows matching a unit-scoped predicate."""

    self.require_user_tier()
    normalized_limit = max(int(limit), 0)
    normalized_offset = max(int(offset), 0)
    order_direction = _query_unit_order_direction(sort_direction)
    if sort == "time":
        order_by = f"COALESCE(a.updated_at_ms, a.created_at_ms, 0) {order_direction}, a.assertion_id {order_direction}"
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
            author_kind=str(row["author_kind"] if row["author_kind"] is not None else ASSERTION_DEFAULT_AUTHOR_KIND),
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
    self: _ArchiveQueryReadsHost,
    predicate: QueryPredicate,
    *,
    limit: int = 50,
    offset: int = 0,
    session_filters: Mapping[str, object] | None = None,
    sort: Literal["time"] | None = None,
    sort_direction: Literal["asc", "desc"] = "asc",
) -> list[ArchiveRunQueryRow]:
    """Return run rows matching a unit-scoped predicate."""

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
        {run_relation_sql()}
        SELECT r.*, s.origin, s.title AS session_title
        FROM runs r
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
            title=str(row["session_title"]) if row["session_title"] is not None else None,
            run=projected_run_from_row(row),
        )
        for row in rows
    ]


def query_observed_events(
    self: _ArchiveQueryReadsHost,
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
    source_where, source_params = observed_event_source_pushdown(predicate)
    clause, params = _structural_predicate_clause("observed-event", "e", predicate, session_alias="s")
    session_clause = ""
    session_params: list[object] = []
    if session_filters:
        session_clause, session_params = cast(Any, _session_filter_clause)("s", prefix="AND", **session_filters)
    rows = self._conn.execute(
        f"""
        {observed_event_relation_sql(source_where=source_where)}
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
            event=observed_event_from_row(row),
        )
        for row in rows
    ]


def query_context_snapshots(
    self: _ArchiveQueryReadsHost,
    predicate: QueryPredicate,
    *,
    limit: int = 50,
    offset: int = 0,
    session_filters: Mapping[str, object] | None = None,
    sort: Literal["time"] | None = None,
    sort_direction: Literal["asc", "desc"] = "asc",
) -> list[ArchiveContextSnapshotQueryRow]:
    """Return context-snapshot rows matching a unit-scoped predicate."""

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
        {context_snapshot_relation_sql()}
        SELECT c.*, s.origin, s.title AS session_title
        FROM context_snapshots c
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
            title=str(row["session_title"]) if row["session_title"] is not None else None,
            snapshot=context_snapshot_from_row(row),
        )
        for row in rows
    ]
