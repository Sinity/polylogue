"""Concrete column specifications for archive_tiers tables.

Defines the single source of truth for:
  - messages table structure (29 writable columns + 1 GENERATED message_id)
  - blocks table structure (16 writable columns + 1 GENERATED block_id)
  - Other key tables (sessions, session_events, etc.)

Each spec drives INSERT/SELECT generation and typed row extraction.

Extractors are typically defined at module level or passed as callables at spec
construction time. This approach consolidates the column triplicates (column list,
placeholder string, tuple order) into a single spec that drives INSERT/SELECT.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from operator import itemgetter

from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.core.enums import (
    BlockType,
    BranchType,
    LinkType,
    MaterialOrigin,
    Origin,
    PasteBoundary,
    SessionKind,
    SessionRefKind,
    StopReason,
    TitleSource,
    ToolResultUnknownReason,
    TopologyEdgeStatus,
    WebConstructType,
)
from polylogue.storage.sqlite.archive_tiers.column_spec import ColumnSpec, TableColumnSpec
from polylogue.storage.sqlite.archive_tiers.common import (
    CONTENT_HASH_CHECK,
    check,
    json_array_check,
    json_object_check,
    literal_check,
    nullable_check,
)


def _value(name: str) -> Callable[[dict[str, object]], object]:
    return itemgetter(name)


def _none_to_zero(value: object) -> int:
    if isinstance(value, (int, float, str)):
        return int(value)
    return 0


def _epoch_seconds_to_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float, str)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        return None
    except (TypeError, ValueError, OSError):
        return None


def _bool_value(value: object) -> bool:
    return bool(value)


def _optional_bool_value(value: object) -> bool | None:
    return None if value is None else bool(value)


def _ddl(name: str, sql: str) -> ColumnSpec:
    return ColumnSpec(name=name, is_generated="GENERATED ALWAYS" in sql, ddl_sql=f"{name} {sql}")


def _make_messages_spec() -> TableColumnSpec:
    """Create the messages table column specification.

    The messages table structure (from schema):
      session_id, native_id, identity_source, parent_message_id, position, role, message_type,
      material_origin, model_name, model_effort, sender_name, recipient,
      delivery_status, end_turn, user_context_text, has_tool_use, has_thinking,
      has_paste, paste_boundary, variant_index, is_active_path, is_active_leaf,
      word_count, input_tokens, output_tokens, cache_read_tokens,
      cache_write_tokens, duration_ms, content_address, content_hash, occurred_at_ms

    GENERATED (not writable): message_id

    Special handling: parent_message_id is always NULL on INSERT (no tuple value).
    """
    all_columns: tuple[ColumnSpec, ...] = (
        _ddl(
            "message_id",
            "TEXT GENERATED ALWAYS AS (session_id || ':' || CASE WHEN native_id IS NULL THEN 'p:' || position || '.' || variant_index ELSE 'n:' || native_id END) STORED UNIQUE",
        ),
        _ddl("session_id", "TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"),
        _ddl("native_id", "TEXT"),
        ColumnSpec(
            "identity_source",
            "TEXT",
            ddl_sql="identity_source TEXT NOT NULL DEFAULT 'positional' CHECK(identity_source IN ('native', 'positional'))",
        ),
        ColumnSpec(
            "parent_message_id",
            "TEXT",
            extract_placeholder="NULL",
            ddl_sql="parent_message_id TEXT REFERENCES messages(message_id) ON DELETE SET NULL",
        ),
        _ddl("position", "INTEGER NOT NULL CHECK(position >= 0)"),
        ColumnSpec("role", "TEXT", ddl_sql=f"role TEXT NOT NULL CHECK ({check('role', Role)})"),
        ColumnSpec(
            "message_type",
            "TEXT",
            ddl_sql=f"message_type TEXT NOT NULL DEFAULT 'message' CHECK ({check('message_type', MessageType)})",
        ),
        ColumnSpec(
            "material_origin",
            "TEXT",
            ddl_sql=f"material_origin TEXT NOT NULL DEFAULT 'unknown' CHECK ({check('material_origin', MaterialOrigin)})",
        ),
        _ddl("model_name", "TEXT"),
        _ddl("model_effort", "TEXT"),
        _ddl("sender_name", "TEXT"),
        _ddl("recipient", "TEXT"),
        _ddl("delivery_status", "TEXT"),
        _ddl("end_turn", "INTEGER CHECK(end_turn IN (0, 1) OR end_turn IS NULL)"),
        _ddl("user_context_text", "TEXT"),
        _ddl("has_tool_use", "INTEGER NOT NULL DEFAULT 0 CHECK(has_tool_use IN (0, 1))"),
        _ddl("has_thinking", "INTEGER NOT NULL DEFAULT 0 CHECK(has_thinking IN (0, 1))"),
        _ddl("has_paste", "INTEGER NOT NULL DEFAULT 0 CHECK(has_paste IN (0, 1))"),
        ColumnSpec(
            "paste_boundary",
            "TEXT",
            ddl_sql=f"paste_boundary TEXT CHECK ({nullable_check('paste_boundary', PasteBoundary)})",
        ),
        _ddl("variant_index", "INTEGER NOT NULL DEFAULT 0 CHECK(variant_index >= 0)"),
        _ddl("is_active_path", "INTEGER NOT NULL DEFAULT 1 CHECK(is_active_path IN (0, 1))"),
        _ddl("is_active_leaf", "INTEGER NOT NULL DEFAULT 0 CHECK(is_active_leaf IN (0, 1))"),
        _ddl("word_count", "INTEGER NOT NULL DEFAULT 0 CHECK(word_count >= 0)"),
        _ddl("input_tokens", "INTEGER NOT NULL DEFAULT 0 CHECK(input_tokens >= 0)"),
        _ddl("output_tokens", "INTEGER NOT NULL DEFAULT 0 CHECK(output_tokens >= 0)"),
        _ddl("cache_read_tokens", "INTEGER NOT NULL DEFAULT 0 CHECK(cache_read_tokens >= 0)"),
        _ddl("cache_write_tokens", "INTEGER NOT NULL DEFAULT 0 CHECK(cache_write_tokens >= 0)"),
        _ddl("duration_ms", "INTEGER CHECK(duration_ms IS NULL OR duration_ms >= 0)"),
        _ddl("content_address", "BLOB CHECK(content_address IS NULL OR length(content_address) = 32)"),
        _ddl("content_hash", f"BLOB NOT NULL {CONTENT_HASH_CHECK}"),
        _ddl("occurred_at_ms", "INTEGER"),
        ColumnSpec(
            "stop_reason", "TEXT", ddl_sql=f"stop_reason TEXT CHECK ({nullable_check('stop_reason', StopReason)})"
        ),
    )

    record_columns = (
        ColumnSpec(
            "text",
            record_name="text",
            domain_name="text",
            select_expression="COALESCE((SELECT group_concat(b.text, char(10)) FROM blocks b WHERE b.message_id = {alias}.message_id AND b.text IS NOT NULL), '')",
        ),
        ColumnSpec("source_name", record_name="source_name", select_expression="s.origin"),
        ColumnSpec("version", record_name="version", select_expression="1"),
    )

    def record(
        name: str,
        *,
        record_name: str | None = None,
        domain_name: str | None = None,
        select_expression: str | None = None,
        record_transform: Callable[[object], object] | None = None,
        domain_transform: Callable[[object], object] | None = None,
    ) -> ColumnSpec:
        return ColumnSpec(
            name,
            record_name=record_name or name,
            domain_name=domain_name,
            select_expression=select_expression,
            record_transform=record_transform,
            domain_transform=domain_transform,
        )

    mappings = {
        "message_id": record("message_id", domain_name="id"),
        "session_id": record("session_id"),
        "native_id": record("native_id", record_name="provider_message_id"),
        "identity_source": record("identity_source", domain_name="identity_source"),
        "parent_message_id": record("parent_message_id", domain_name="parent_id"),
        "content_address": record("content_address", select_expression="lower(hex({alias}.content_address))"),
        "position": record("position", domain_name="position", record_transform=_none_to_zero),
        "role": record("role", domain_name="role"),
        "message_type": record("message_type", domain_name="message_type"),
        "material_origin": record("material_origin", domain_name="material_origin"),
        "model_name": record("model_name", domain_name="model_name"),
        "has_tool_use": record(
            "has_tool_use", domain_name="has_tool_use", record_transform=_none_to_zero, domain_transform=_bool_value
        ),
        "has_thinking": record(
            "has_thinking", domain_name="has_thinking", record_transform=_none_to_zero, domain_transform=_bool_value
        ),
        "has_paste": record(
            "has_paste", domain_name="has_paste", record_transform=_none_to_zero, domain_transform=_bool_value
        ),
        "paste_boundary": record(
            "paste_boundary", record_name="paste_boundary_state", domain_name="paste_boundary_state"
        ),
        "variant_index": record(
            "variant_index", record_name="branch_index", domain_name="branch_index", record_transform=_none_to_zero
        ),
        "is_active_path": record(
            "is_active_path",
            domain_name="is_active_path",
            record_transform=_optional_bool_value,
            domain_transform=lambda value: value,
        ),
        "is_active_leaf": record(
            "is_active_leaf", domain_name="is_active_leaf", record_transform=_bool_value, domain_transform=_bool_value
        ),
        "word_count": record("word_count", record_transform=_none_to_zero),
        "input_tokens": record("input_tokens", domain_name="input_tokens", record_transform=_none_to_zero),
        "output_tokens": record("output_tokens", domain_name="output_tokens", record_transform=_none_to_zero),
        "cache_read_tokens": record(
            "cache_read_tokens", domain_name="cache_read_tokens", record_transform=_none_to_zero
        ),
        "cache_write_tokens": record(
            "cache_write_tokens", domain_name="cache_write_tokens", record_transform=_none_to_zero
        ),
        "duration_ms": record("duration_ms", domain_name="duration_ms", domain_transform=_none_to_zero),
        "content_hash": record("content_hash", select_expression="lower(hex({alias}.content_hash))"),
        "occurred_at_ms": record(
            "occurred_at_ms",
            record_name="sort_key",
            domain_name="timestamp",
            select_expression="{alias}.occurred_at_ms / 1000.0",
            domain_transform=_epoch_seconds_to_datetime,
        ),
        "stop_reason": record("stop_reason", domain_name="stop_reason"),
    }
    all_columns = tuple(
        replace(
            col,
            extract=_value(col.name) if col.extract_placeholder == "?" else None,
            record_name=mappings[col.name].record_name,
            select_expression=mappings[col.name].select_expression,
            record_transform=mappings[col.name].record_transform,
            domain_name=mappings[col.name].domain_name,
            domain_transform=mappings[col.name].domain_transform,
        )
        if col.name in mappings
        else replace(col, extract=_value(col.name) if col.extract_placeholder == "?" else None)
        for col in all_columns
    )

    writable_columns = tuple(col for col in all_columns if not col.is_generated)

    return TableColumnSpec(
        table_name="messages",
        all_columns=all_columns,
        writable_columns=writable_columns,
        record_only_columns=record_columns,
        table_constraints=("PRIMARY KEY(session_id, position, variant_index)",),
    )


def _make_blocks_spec() -> TableColumnSpec:
    """Create the blocks table column specification.

    The blocks table structure (from schema):
      session_id, message_id, position, block_type, text, tool_name, tool_id,
      tool_input, semantic_type, media_type, language, tool_result_is_error,
      tool_result_exit_code, tool_result_outcome_unknown_reason, signature,
      content_hash

    GENERATED (not writable):
      block_id, tool_command, tool_path, search_text, tool_detail_text
    """
    all_columns: tuple[ColumnSpec, ...] = (
        _ddl("block_id", "TEXT GENERATED ALWAYS AS (message_id || ':' || position) STORED UNIQUE"),
        _ddl("message_id", "TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE"),
        _ddl("session_id", "TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"),
        _ddl("position", "INTEGER NOT NULL CHECK(position >= 0)"),
        ColumnSpec("block_type", "TEXT", ddl_sql=f"block_type TEXT NOT NULL CHECK ({check('block_type', BlockType)})"),
        _ddl("text", "TEXT"),
        _ddl("tool_name", "TEXT"),
        _ddl("tool_id", "TEXT"),
        ColumnSpec(
            "tool_input", "TEXT", ddl_sql=f"tool_input TEXT CHECK ({json_object_check('tool_input', nullable=True)})"
        ),
        _ddl("semantic_type", "TEXT"),
        _ddl("media_type", "TEXT"),
        _ddl("language", "TEXT"),
        _ddl("tool_result_is_error", "INTEGER CHECK (tool_result_is_error IN (0, 1))"),
        _ddl("tool_result_exit_code", "INTEGER"),
        ColumnSpec(
            "tool_result_outcome_unknown_reason",
            "TEXT",
            ddl_sql=f"tool_result_outcome_unknown_reason TEXT CHECK ({nullable_check('tool_result_outcome_unknown_reason', ToolResultUnknownReason)})",
        ),
        _ddl("signature", "TEXT"),
        _ddl("content_hash", "BLOB CHECK(content_hash IS NULL OR length(content_hash) = 32)"),
        _ddl("tool_command", "TEXT GENERATED ALWAYS AS (json_extract(tool_input, '$.command')) VIRTUAL"),
        _ddl(
            "tool_path",
            "TEXT GENERATED ALWAYS AS (COALESCE(json_extract(tool_input, '$.file_path'), json_extract(tool_input, '$.path'))) VIRTUAL",
        ),
        _ddl(
            "search_text",
            "TEXT GENERATED ALWAYS AS (trim(COALESCE(text, '') || ' ' || COALESCE(tool_name, '') || ' ' || COALESCE(json_extract(tool_input, '$.command'), '') || ' ' || COALESCE(json_extract(tool_input, '$.file_path'), '') || ' ' || COALESCE(json_extract(tool_input, '$.path'), ''))) VIRTUAL",
        ),
        _ddl(
            "tool_detail_text",
            "TEXT GENERATED ALWAYS AS (lower(COALESCE(tool_command, '') || ' ' || COALESCE(tool_path, ''))) VIRTUAL",
        ),
    )

    record_columns = (ColumnSpec("metadata", record_name="metadata", select_expression="NULL"),)
    mappings = {
        "block_id": ("block_id", None),
        "message_id": ("message_id", None),
        "session_id": ("session_id", None),
        "position": ("block_index", None),
        "block_type": ("type", None),
        "text": ("text", None),
        "tool_name": ("tool_name", None),
        "tool_id": ("tool_id", None),
        "tool_input": ("tool_input", None),
        "semantic_type": ("semantic_type", None),
        "tool_result_is_error": ("tool_result_is_error", None),
        "tool_result_exit_code": ("tool_result_exit_code", None),
        "tool_result_outcome_unknown_reason": ("tool_result_outcome_unknown_reason", None),
        "signature": ("signature", None),
    }
    all_columns = tuple(
        replace(
            col,
            extract=_value(col.name) if col.extract_placeholder == "?" else None,
            record_name=mappings[col.name][0],
        )
        if col.name in mappings
        else replace(col, extract=_value(col.name) if col.extract_placeholder == "?" else None)
        for col in all_columns
    )

    writable_columns = tuple(col for col in all_columns if not col.is_generated)

    return TableColumnSpec(
        table_name="blocks",
        all_columns=all_columns,
        writable_columns=writable_columns,
        record_only_columns=record_columns,
        table_constraints=("PRIMARY KEY(message_id, position)",),
    )


# Index-tier table specs. Each rendered table body is sourced from these
# column definitions plus its table-level constraints; indexes, triggers,
# and virtual tables remain in index.py because they are not row schemas.
def _raw_column(name: str, ddl_sql: str) -> ColumnSpec:
    return ColumnSpec(name=name, is_generated="GENERATED ALWAYS" in ddl_sql, ddl_sql=ddl_sql)


def _make_table_spec(
    table_name: str,
    columns: tuple[ColumnSpec, ...],
    *,
    table_constraints: tuple[str, ...] = (),
) -> TableColumnSpec:
    return TableColumnSpec(
        table_name=table_name,
        all_columns=columns,
        writable_columns=tuple(column for column in columns if not column.is_generated),
        table_constraints=table_constraints,
    )


FTS_FRESHNESS_STATE_SPEC = _make_table_spec(
    "fts_freshness_state",
    (
        _raw_column("surface", """surface TEXT PRIMARY KEY"""),
        _raw_column("state", f"""state TEXT NOT NULL CHECK ({literal_check("state", "ready", "stale", "unknown")})"""),
        _raw_column("checked_at", """checked_at TEXT NOT NULL"""),
        _raw_column("source_rows", """source_rows INTEGER NOT NULL DEFAULT 0"""),
        _raw_column("indexed_rows", """indexed_rows INTEGER NOT NULL DEFAULT 0"""),
        _raw_column("missing_rows", """missing_rows INTEGER NOT NULL DEFAULT 0"""),
        _raw_column("excess_rows", """excess_rows INTEGER NOT NULL DEFAULT 0"""),
        _raw_column("duplicate_rows", """duplicate_rows INTEGER NOT NULL DEFAULT 0"""),
        _raw_column("identity_mismatch_rows", """identity_mismatch_rows INTEGER NOT NULL DEFAULT 0"""),
        _raw_column(
            "verification_kind",
            f"""verification_kind TEXT NOT NULL DEFAULT 'unknown' CHECK ({literal_check("verification_kind", "unknown", "bounded", "exact")})""",
        ),
        _raw_column("exact_checked_at", """exact_checked_at TEXT"""),
        _raw_column("exact_generation", """exact_generation INTEGER"""),
        _raw_column("detail", """detail TEXT"""),
    ),
    table_constraints=(
        """-- polylogue-rlvj (v52): a 'ready' verdict must be backed by an exact
    -- count that actually balances -- a scoped/targeted repair's correct
    -- answer to a narrower question must not be written here as a global
    -- 'ready'. See INDEX_SCHEMA_VERSION's v52 comment above for the live
    -- incident (messages_fts reported ready/missing_rows=0 while 12,659
    -- blocks were unindexed).
    CHECK (
        state != 'ready'
        OR (
            missing_rows = 0
            AND excess_rows = 0
            AND duplicate_rows = 0
            AND identity_mismatch_rows = 0
            AND source_rows = indexed_rows
            AND verification_kind = 'exact'
            AND exact_checked_at IS NOT NULL
            AND exact_generation IS NOT NULL
        )
    )""",
    ),
)

QUERY_UNIT_FRAME_STATE_SPEC = _make_table_spec(
    "query_unit_frame_state",
    (
        _raw_column("singleton", """singleton INTEGER PRIMARY KEY CHECK (singleton = 1)"""),
        _raw_column("epoch", """epoch INTEGER NOT NULL DEFAULT 0 CHECK (epoch >= 0)"""),
    ),
)

RAW_REVISION_APPLICATIONS_SPEC = _make_table_spec(
    "raw_revision_applications",
    (
        _raw_column("decision_id", """decision_id              TEXT PRIMARY KEY"""),
        _raw_column("raw_id", """raw_id                   TEXT NOT NULL"""),
        _raw_column("session_id", """session_id               TEXT NOT NULL"""),
        _raw_column("logical_source_key", """logical_source_key       TEXT NOT NULL"""),
        _raw_column("source_revision", """source_revision          TEXT NOT NULL"""),
        _raw_column(
            "acquisition_generation", """acquisition_generation  INTEGER NOT NULL CHECK(acquisition_generation >= 0)"""
        ),
        _raw_column(
            "decision", f"""decision                 TEXT NOT NULL CHECK ({check("decision", ApplicationDecision)})"""
        ),
        _raw_column("accepted_raw_id", """accepted_raw_id          TEXT"""),
        _raw_column("accepted_source_revision", """accepted_source_revision TEXT"""),
        _raw_column(
            "accepted_content_hash",
            """accepted_content_hash    BLOB CHECK(
                                 accepted_content_hash IS NULL OR length(accepted_content_hash) = 32
                             )""",
        ),
        _raw_column(
            "accepted_frontier_kind",
            """accepted_frontier_kind   TEXT CHECK(
                                 accepted_frontier_kind IS NULL
                                 OR accepted_frontier_kind IN ('byte', 'semantic')
                             )""",
        ),
        _raw_column(
            "accepted_frontier",
            """accepted_frontier        INTEGER CHECK(
                                 accepted_frontier IS NULL OR accepted_frontier >= 0
                             )""",
        ),
        _raw_column("baseline_raw_id", """baseline_raw_id          TEXT"""),
        _raw_column("predecessor_raw_id", """predecessor_raw_id       TEXT"""),
        _raw_column(
            "append_end_offset",
            """append_end_offset        INTEGER CHECK(append_end_offset IS NULL OR append_end_offset >= 0)""",
        ),
        _raw_column("detail", """detail                   TEXT NOT NULL"""),
        _raw_column("decided_at_ms", """decided_at_ms            INTEGER NOT NULL CHECK(decided_at_ms >= 0)"""),
    ),
    table_constraints=(
        """CHECK(
        (
            accepted_raw_id IS NULL
            AND accepted_source_revision IS NULL
            AND accepted_content_hash IS NULL
            AND accepted_frontier_kind IS NULL
            AND accepted_frontier IS NULL
        )
        OR
        (
            accepted_raw_id IS NOT NULL
            AND accepted_source_revision IS NOT NULL
            AND accepted_content_hash IS NOT NULL
            AND accepted_frontier_kind IS NOT NULL
            AND accepted_frontier IS NOT NULL
        )
    )""",
    ),
)

RAW_REVISION_HEADS_SPEC = _make_table_spec(
    "raw_revision_heads",
    (
        _raw_column("logical_source_key", """logical_source_key       TEXT PRIMARY KEY"""),
        _raw_column("session_id", """session_id               TEXT NOT NULL"""),
        _raw_column("accepted_raw_id", """accepted_raw_id          TEXT NOT NULL"""),
        _raw_column("accepted_source_revision", """accepted_source_revision TEXT NOT NULL"""),
        _raw_column(
            "accepted_content_hash",
            """accepted_content_hash    BLOB NOT NULL CHECK(length(accepted_content_hash) = 32)""",
        ),
        _raw_column(
            "accepted_frontier_kind",
            f"""accepted_frontier_kind   TEXT NOT NULL CHECK({literal_check("accepted_frontier_kind", "byte", "semantic")})""",
        ),
        _raw_column("accepted_frontier", """accepted_frontier        INTEGER NOT NULL CHECK(accepted_frontier >= 0)"""),
        _raw_column(
            "acquisition_generation", """acquisition_generation  INTEGER NOT NULL CHECK(acquisition_generation >= 0)"""
        ),
        _raw_column(
            "append_end_offset",
            """append_end_offset        INTEGER CHECK(append_end_offset IS NULL OR append_end_offset >= 0)""",
        ),
        _raw_column("decided_at_ms", """decided_at_ms            INTEGER NOT NULL CHECK(decided_at_ms >= 0)"""),
    ),
)

SESSIONS_SPEC = _make_table_spec(
    "sessions",
    (
        _raw_column(
            "session_id",
            """session_id              TEXT GENERATED ALWAYS AS (origin || ':' || native_id) STORED UNIQUE""",
        ),
        _raw_column("native_id", """native_id               TEXT NOT NULL"""),
        _raw_column("origin", f"""origin                  TEXT NOT NULL CHECK ({check("origin", Origin)})"""),
        _raw_column(
            "parent_session_id", """parent_session_id       TEXT REFERENCES sessions(session_id) ON DELETE SET NULL"""
        ),
        _raw_column(
            "root_session_id", """root_session_id         TEXT REFERENCES sessions(session_id) ON DELETE SET NULL"""
        ),
        _raw_column("raw_id", """raw_id                  TEXT"""),
        _raw_column(
            "parser_fingerprint",
            """-- Written by the parsed-session chokepoint in the same transaction as
    -- this row. Pre-v64 generations are intentionally nullable until replay.
    parser_fingerprint      TEXT""",
        ),
        _raw_column("lowering_fingerprint", """lowering_fingerprint    TEXT"""),
        _raw_column(
            "branch_type", f"""branch_type             TEXT CHECK ({nullable_check("branch_type", BranchType)})"""
        ),
        _raw_column("active_leaf_message_id", """active_leaf_message_id  TEXT"""),
        _raw_column("title", """title                   TEXT"""),
        _raw_column(
            "session_kind",
            f"""session_kind            TEXT NOT NULL DEFAULT 'standard' CHECK ({check("session_kind", SessionKind)})""",
        ),
        _raw_column(
            "title_source",
            f"""-- polylogue-5dfu: NULL is already the "no title evidence" state for a
    -- nullable column; TitleSource.UNKNOWN was a redundant second spelling of
    -- the same fact (every read site that branches on title_source treats
    -- NULL and 'unknown' identically -- see archive.py's has_real_title
    -- check), so the enum was collapsed to only the members a producer
    -- actually assigns and this CHECK is now generated from it like the
    -- other enum-backed columns instead of hand-listing the values.
    title_source            TEXT CHECK({nullable_check("title_source", TitleSource)})""",
        ),
        _raw_column(
            "title_ref",
            """-- Specific provenance beyond TitleSource's coarse strategy label: which
    -- exact evidence row won (e.g. "codex-thread-name:<id>",
    -- "codex-history:<id>", "message:<provider_message_id>") plus a 0..1
    -- confidence signal for that resolution (polylogue-ih67 AC#5, ref/
    -- confidence slice). Both derived/rebuildable, never hand-edited.
    title_ref               TEXT""",
        ),
        _raw_column(
            "title_confidence",
            """title_confidence        REAL CHECK(title_confidence IS NULL OR (title_confidence >= 0 AND title_confidence <= 1))""",
        ),
        _raw_column(
            "display_name",
            """-- polylogue-2qx.4 (v46): the human-readable name behind an opaque
    -- native/slug id -- e.g. Claude Code Task-tool subagent slugs
    -- ("greedy-squishing-hamming") so subagent rows read a name instead of
    -- "5ecdb160-...:agent-af4e". Distinct from `title` (the session's own
    -- resolved title): this is a display label for the session's identity,
    -- not its content.
    display_name            TEXT""",
        ),
        _raw_column(
            "run_settings_json",
            f"""-- polylogue-2qx.4 (v46): per-session provider run configuration
    -- (aistudio-drive runSettings: temperature/topP/topK/maxOutputTokens/
    -- thinkingLevel/safetySettings/enable* flags). A JSON column by
    -- deliberate decision -- decomposing a provider-specific settings bag
    -- into typed columns would couple this schema to one provider for no
    -- query benefit; nothing here is queried across origins today.
    run_settings_json       TEXT CHECK ({json_object_check("run_settings_json", nullable=True)})""",
        ),
        _raw_column(
            "pending_drafts_json",
            f"""-- polylogue-o4j2 (v47): non-blank chunkedPrompt.pendingInputs entries --
    -- the operator's not-yet-submitted textbox draft(s) -- verbatim as a
    -- JSON array of {{text, role, token_count}} objects. Deliberately a
    -- session-row field, NOT a session_event: a draft is CURRENT mutable
    -- UI state (edited in place, then disappears entirely on submit), not
    -- an append-only historical fact, so it must stay outside
    -- session_revision_projection's message/attachment/event comparison
    -- axes (polylogue-aggz Invariant 1) -- exactly the shape polylogue-bu1i
    -- and polylogue-nuec were fixed for, on a third axis (mutable session
    -- state rather than acquisition state or provider-remeasurement).
    pending_drafts_json      TEXT CHECK ({json_array_check("pending_drafts_json", nullable=True)})""",
        ),
        _raw_column("git_branch", """git_branch              TEXT"""),
        _raw_column("git_repository_url", """git_repository_url      TEXT"""),
        _raw_column("provider_project_ref", """provider_project_ref    TEXT"""),
        _raw_column("commit_hash", """commit_hash             TEXT"""),
        _raw_column("instructions_text", """instructions_text       TEXT"""),
        _raw_column(
            "reported_duration_ms",
            """reported_duration_ms    INTEGER CHECK(reported_duration_ms IS NULL OR reported_duration_ms >= 0)""",
        ),
        _raw_column(
            "reported_cost_usd",
            """-- polylogue-gt1z (v49): exact provider-reported session cost total, when
    -- the origin's own export carries one (claude-code-session costUSD,
    -- hermes-session state.db) -- ParsedSession.reported_cost_usd verbatim.
    -- NULL means the origin never reports a session-level total, not a
    -- measured zero (a genuine $0 total is preserved by the parser same as
    -- any other reported value). Feeds `_session_level_estimate`'s
    -- ``status == "exact"`` cost path; per-model catalog pricing in
    -- session_model_usage remains the token-total authority (see that
    -- table's header) -- this column is a parallel exact-dollar figure, not
    -- a token source.
    reported_cost_usd       REAL CHECK(reported_cost_usd IS NULL OR reported_cost_usd >= 0)""",
        ),
        _raw_column(
            "message_count", """message_count           INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0)"""
        ),
        _raw_column("word_count", """word_count              INTEGER NOT NULL DEFAULT 0 CHECK(word_count >= 0)"""),
        _raw_column(
            "tool_use_count", """tool_use_count          INTEGER NOT NULL DEFAULT 0 CHECK(tool_use_count >= 0)"""
        ),
        _raw_column(
            "thinking_count", """thinking_count          INTEGER NOT NULL DEFAULT 0 CHECK(thinking_count >= 0)"""
        ),
        _raw_column("paste_count", """paste_count             INTEGER NOT NULL DEFAULT 0 CHECK(paste_count >= 0)"""),
        _raw_column(
            "user_message_count",
            """user_message_count      INTEGER NOT NULL DEFAULT 0 CHECK(user_message_count >= 0)""",
        ),
        _raw_column(
            "authored_user_message_count",
            """authored_user_message_count INTEGER NOT NULL DEFAULT 0 CHECK(authored_user_message_count >= 0)""",
        ),
        _raw_column(
            "assistant_message_count",
            """assistant_message_count INTEGER NOT NULL DEFAULT 0 CHECK(assistant_message_count >= 0)""",
        ),
        _raw_column(
            "system_message_count",
            """system_message_count    INTEGER NOT NULL DEFAULT 0 CHECK(system_message_count >= 0)""",
        ),
        _raw_column(
            "tool_message_count",
            """tool_message_count      INTEGER NOT NULL DEFAULT 0 CHECK(tool_message_count >= 0)""",
        ),
        _raw_column(
            "user_word_count", """user_word_count         INTEGER NOT NULL DEFAULT 0 CHECK(user_word_count >= 0)"""
        ),
        _raw_column(
            "authored_user_word_count",
            """authored_user_word_count INTEGER NOT NULL DEFAULT 0 CHECK(authored_user_word_count >= 0)""",
        ),
        _raw_column(
            "assistant_word_count",
            """assistant_word_count    INTEGER NOT NULL DEFAULT 0 CHECK(assistant_word_count >= 0)""",
        ),
        _raw_column("content_hash", f"""content_hash            BLOB NOT NULL {CONTENT_HASH_CHECK}"""),
        _raw_column("created_at_ms", """created_at_ms           INTEGER"""),
        _raw_column("updated_at_ms", """updated_at_ms           INTEGER"""),
        _raw_column(
            "sort_key_ms",
            """sort_key_ms             INTEGER GENERATED ALWAYS AS (COALESCE(updated_at_ms, created_at_ms)) STORED""",
        ),
    ),
    table_constraints=("""PRIMARY KEY(origin, native_id)""",),
)

WEB_CONTENT_CONSTRUCTS_SPEC = _make_table_spec(
    "web_content_constructs",
    (
        _raw_column(
            "construct_id", """construct_id    TEXT GENERATED ALWAYS AS (block_id || ':' || position) STORED UNIQUE"""
        ),
        _raw_column(
            "session_id", """session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"""
        ),
        _raw_column(
            "message_id", """message_id      TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE"""
        ),
        _raw_column("block_id", """block_id        TEXT NOT NULL REFERENCES blocks(block_id) ON DELETE CASCADE"""),
        _raw_column("position", """position        INTEGER NOT NULL CHECK(position >= 0)"""),
        _raw_column("provider", """provider        TEXT NOT NULL"""),
        _raw_column(
            "construct_type", f"""construct_type  TEXT NOT NULL CHECK ({check("construct_type", WebConstructType)})"""
        ),
        _raw_column("provider_key", """provider_key    TEXT"""),
        _raw_column("title", """title           TEXT"""),
        _raw_column("url", """url             TEXT"""),
        _raw_column("text", """text            TEXT"""),
        _raw_column("source_id", """source_id       TEXT"""),
        _raw_column("group_id", """group_id        TEXT"""),
        _raw_column("group_title", """group_title     TEXT"""),
        _raw_column("query", """query           TEXT"""),
        _raw_column("asset_pointer", """asset_pointer   TEXT"""),
        _raw_column("mime_type", """mime_type       TEXT"""),
        _raw_column("status", """status          TEXT"""),
        _raw_column("task_id", """task_id         TEXT"""),
        _raw_column("task_type", """task_type       TEXT"""),
        _raw_column("rank", """rank            INTEGER"""),
        _raw_column("start_index", """start_index     INTEGER"""),
        _raw_column("end_index", """end_index       INTEGER"""),
    ),
    table_constraints=("""PRIMARY KEY(block_id, position)""",),
)

FILE_EDITS_SPEC = _make_table_spec(
    "file_edits",
    (
        _raw_column(
            "tool_use_block_id",
            """tool_use_block_id   TEXT PRIMARY KEY REFERENCES blocks(block_id) ON DELETE CASCADE""",
        ),
        _raw_column(
            "session_id", """session_id          TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"""
        ),
        _raw_column(
            "message_id", """message_id          TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE"""
        ),
        _raw_column("file_path", """file_path           TEXT"""),
        _raw_column(
            "structured_patch_json",
            f"""structured_patch_json TEXT CHECK ({json_array_check("structured_patch_json", nullable=True)})""",
        ),
        _raw_column("original_file", """original_file       TEXT"""),
        _raw_column("old_string", """old_string          TEXT"""),
        _raw_column("new_string", """new_string          TEXT"""),
        _raw_column(
            "replace_all", """replace_all         INTEGER CHECK(replace_all IN (0, 1) OR replace_all IS NULL)"""
        ),
        _raw_column(
            "user_modified", """user_modified       INTEGER CHECK(user_modified IN (0, 1) OR user_modified IS NULL)"""
        ),
        _raw_column("observed_at_ms", """observed_at_ms      INTEGER"""),
    ),
)

SESSION_REFS_SPEC = _make_table_spec(
    "session_refs",
    (
        _raw_column(
            "ref_id", """ref_id          TEXT GENERATED ALWAYS AS (session_id || ':' || position) STORED UNIQUE"""
        ),
        _raw_column(
            "session_id", """session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"""
        ),
        _raw_column("position", """position        INTEGER NOT NULL CHECK(position >= 0)"""),
        _raw_column("kind", f"""kind            TEXT NOT NULL CHECK ({check("kind", SessionRefKind)})"""),
        _raw_column("repo", """repo            TEXT"""),
        _raw_column("ref_number", """ref_number      INTEGER"""),
        _raw_column("url", """url             TEXT NOT NULL"""),
        _raw_column("observed_at_ms", """observed_at_ms  INTEGER"""),
    ),
    table_constraints=("""PRIMARY KEY(session_id, position)""",),
)

SESSION_EVENTS_SPEC = _make_table_spec(
    "session_events",
    (
        _raw_column(
            "event_id",
            """event_id                   TEXT GENERATED ALWAYS AS (session_id || ':' || position) STORED UNIQUE""",
        ),
        _raw_column(
            "session_id",
            """session_id                 TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE""",
        ),
        _raw_column(
            "source_message_id",
            """source_message_id          TEXT REFERENCES messages(message_id) ON DELETE SET NULL""",
        ),
        _raw_column("source_message_provider_id", """source_message_provider_id TEXT"""),
        _raw_column("position", """position                   INTEGER NOT NULL CHECK(position >= 0)"""),
        _raw_column("event_type", """event_type                 TEXT NOT NULL CHECK(length(trim(event_type)) > 0)"""),
        _raw_column("summary", """summary                    TEXT NOT NULL"""),
        _raw_column(
            "payload_json",
            f"""payload_json               TEXT NOT NULL DEFAULT '{{}}' CHECK ({json_object_check("payload_json")})""",
        ),
        _raw_column("occurred_at_ms", """occurred_at_ms             INTEGER"""),
        _raw_column("boundary_start_position", """boundary_start_position  INTEGER"""),
        _raw_column("boundary_end_position", """boundary_end_position    INTEGER"""),
        _raw_column("boundary_message_id", """boundary_message_id      TEXT"""),
    ),
    table_constraints=("""PRIMARY KEY(session_id, position)""",),
)

SESSION_AGENT_POLICIES_SPEC = _make_table_spec(
    "session_agent_policies",
    (
        _raw_column(
            "policy_id", """policy_id         TEXT GENERATED ALWAYS AS (session_id || ':' || position) STORED UNIQUE"""
        ),
        _raw_column(
            "session_id", """session_id        TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"""
        ),
        _raw_column(
            "source_message_id", """source_message_id TEXT REFERENCES messages(message_id) ON DELETE SET NULL"""
        ),
        _raw_column("position", """position          INTEGER NOT NULL CHECK(position >= 0)"""),
        _raw_column("approval_policy", """approval_policy   TEXT"""),
        _raw_column("sandbox_policy", """sandbox_policy    TEXT"""),
        _raw_column("network_policy", """network_policy    TEXT"""),
        _raw_column("observed_at_ms", """observed_at_ms    INTEGER"""),
    ),
    table_constraints=("""PRIMARY KEY(session_id, position)""",),
)

SESSION_LINKS_SPEC = _make_table_spec(
    "session_links",
    (
        _raw_column(
            "src_session_id",
            """src_session_id          TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE""",
        ),
        _raw_column("dst_origin", f"""dst_origin              TEXT NOT NULL CHECK ({check("dst_origin", Origin)})"""),
        _raw_column("dst_native_id", """dst_native_id           TEXT NOT NULL"""),
        _raw_column("link_type", f"""link_type               TEXT NOT NULL CHECK ({check("link_type", LinkType)})"""),
        _raw_column(
            "resolved_dst_session_id",
            """resolved_dst_session_id TEXT REFERENCES sessions(session_id) ON DELETE SET NULL""",
        ),
        _raw_column(
            "branch_point_message_id",
            """-- Lineage normalization (#2467): for a prefix-sharing child (fork / resume /
    -- spawned subagent / auto-compaction copy) the child stores only its own
    -- divergent tail; `branch_point_message_id` is the last parent message the
    -- child inherited, and `inheritance` records whether the child shares the
    -- parent's leading prefix ('prefix-sharing') or is a fresh spawn that merely
    -- references the parent ('spawned-fresh'). NULL until the parent is resolved.
    -- Deliberately NOT a FK: message_id is deterministic, so a parent full-replace
    -- re-ingest re-creates the same id. An `ON DELETE SET NULL` FK would instead
    -- null this during the parent's DELETE step and permanently break the child's
    -- composition (the cascade fires before the re-INSERT) — see #2467 audit.
    branch_point_message_id TEXT""",
        ),
        _raw_column(
            "branch_point_content_address",
            """branch_point_content_address BLOB CHECK(
                branch_point_content_address IS NULL OR length(branch_point_content_address) = 32
            )""",
        ),
        _raw_column(
            "inheritance",
            f"""inheritance             TEXT CHECK({literal_check("inheritance", "prefix-sharing", "spawned-fresh")} OR inheritance IS NULL)""",
        ),
        _raw_column(
            "status",
            f"""-- polylogue-5dfu: TopologyEdgeStatus used to declare 4 members
    -- (unresolved/resolved/repaired/quarantined) while only 2 were ever
    -- storable here -- unresolved/resolved is already carried by
    -- resolved_dst_session_id IS NOT NULL, so those two members were never
    -- assigned anywhere outside a Pydantic field default. Narrowed the enum
    -- to the two exception markers a resolver actually writes and generated
    -- this CHECK from it, replacing the hand-written literal list that had
    -- silently drifted out of sync with `_status_value`'s narrower runtime
    -- projection.
    status                  TEXT CHECK({nullable_check("status", TopologyEdgeStatus)})""",
        ),
        _raw_column(
            "parent_tool_use_block_id",
            """-- polylogue-2qx.4 (v46): the parent-session tool_use block that
    -- dispatched this child (Claude Code parentToolUseID, 842,819 records /
    -- 185,982 distinct dispatch ids on the wire). This IS the delegation
    -- edge's join key; `method` records how the edge was derived (e.g.
    -- 'parent-tool-use-id') once a resolver populates this column. Replaces
    -- delegation_facts' cardinality-gated ordinal dispatch<->child pairing,
    -- which only resolved 12.8% of cases. Not a FK to sessions -- it is a
    -- block within the PARENT session, resolvable independently of whether
    -- src/dst session identity has resolved yet.
    parent_tool_use_block_id TEXT REFERENCES blocks(block_id) ON DELETE SET NULL""",
        ),
        _raw_column("method", """method                  TEXT"""),
        _raw_column(
            "confidence", """confidence              REAL NOT NULL DEFAULT 1.0 CHECK(confidence BETWEEN 0 AND 1)"""
        ),
        _raw_column("evidence_json", """evidence_json           TEXT NOT NULL DEFAULT '[]'"""),
        _raw_column("observed_at_ms", """observed_at_ms          INTEGER NOT NULL"""),
        _raw_column("resolved_at_ms", """resolved_at_ms          INTEGER"""),
    ),
    table_constraints=("""PRIMARY KEY(src_session_id, dst_origin, dst_native_id, link_type)""",),
)

SESSION_WORKING_DIRS_SPEC = _make_table_spec(
    "session_working_dirs",
    (
        _raw_column("session_id", """session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"""),
        _raw_column("path", """path        TEXT NOT NULL"""),
        _raw_column("position", """position    INTEGER NOT NULL CHECK(position >= 0)"""),
    ),
    table_constraints=("""PRIMARY KEY(session_id, path)""",),
)

REPOS_SPEC = _make_table_spec(
    "repos",
    (
        _raw_column("repo_id", """repo_id           TEXT PRIMARY KEY"""),
        _raw_column("origin_url", """origin_url        TEXT NOT NULL DEFAULT ''"""),
        _raw_column("root_path", """root_path         TEXT NOT NULL DEFAULT ''"""),
        _raw_column("repo_name", """repo_name         TEXT NOT NULL DEFAULT ''"""),
        _raw_column("first_seen_at_ms", """first_seen_at_ms  INTEGER NOT NULL"""),
        _raw_column("last_seen_at_ms", """last_seen_at_ms   INTEGER NOT NULL"""),
    ),
)

REPO_CHECKOUTS_SPEC = _make_table_spec(
    "repo_checkouts",
    (
        _raw_column("repo_id", """repo_id           TEXT NOT NULL REFERENCES repos(repo_id) ON DELETE CASCADE"""),
        _raw_column("root_path", """root_path         TEXT NOT NULL"""),
        _raw_column("first_seen_at_ms", """first_seen_at_ms  INTEGER NOT NULL"""),
        _raw_column("last_seen_at_ms", """last_seen_at_ms   INTEGER NOT NULL"""),
    ),
    table_constraints=("""PRIMARY KEY(repo_id, root_path)""",),
)

SESSION_REPOS_SPEC = _make_table_spec(
    "session_repos",
    (
        _raw_column(
            "session_id", """session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"""
        ),
        _raw_column("repo_id", """repo_id         TEXT NOT NULL REFERENCES repos(repo_id) ON DELETE CASCADE"""),
        _raw_column(
            "root_path",
            """-- This session's own observed checkout root (may differ from
    -- repos.root_path's representative value when other sessions checked
    -- out the same repo identity elsewhere) -- needed so repo-relative path
    -- projection (decision 2) strips the *correct* prefix for this session
    -- regardless of which checkout wrote the shared repos row first.
    root_path       TEXT NOT NULL DEFAULT ''""",
        ),
        _raw_column("branch_name", """branch_name     TEXT NOT NULL DEFAULT ''"""),
        _raw_column("observed_at_ms", """observed_at_ms  INTEGER NOT NULL"""),
    ),
    table_constraints=("""PRIMARY KEY(session_id, repo_id)""",),
)

SESSION_COMMITS_SPEC = _make_table_spec(
    "session_commits",
    (
        _raw_column(
            "session_id", """session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"""
        ),
        _raw_column("commit_sha", """commit_sha      TEXT NOT NULL"""),
        _raw_column("repo_id", """repo_id         TEXT REFERENCES repos(repo_id) ON DELETE CASCADE"""),
        _raw_column(
            "detection_type",
            f"""detection_type  TEXT NOT NULL CHECK({literal_check("detection_type", "time_window", "file_overlap", "explicit_ref", "origin_reported")})""",
        ),
        _raw_column("method", """method          TEXT"""),
        _raw_column("confidence", """confidence      REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1)"""),
        _raw_column("evidence_json", """evidence_json   TEXT NOT NULL DEFAULT '{}'"""),
        _raw_column("created_at_ms", """created_at_ms   INTEGER NOT NULL"""),
    ),
    table_constraints=("""PRIMARY KEY(session_id, commit_sha)""",),
)

ATTACHMENTS_SPEC = _make_table_spec(
    "attachments",
    (
        _raw_column("attachment_id", """attachment_id          TEXT PRIMARY KEY"""),
        _raw_column("display_name", """display_name           TEXT"""),
        _raw_column("media_type", """media_type             TEXT"""),
        _raw_column("byte_count", """byte_count             INTEGER NOT NULL DEFAULT 0 CHECK(byte_count >= 0)"""),
        _raw_column(
            "blob_hash",
            """-- #2468: real SHA-256 of the stored bytes when acquired, else NULL. Previously
    -- a synthetic hash of attachment metadata was written here, falsely implying a
    -- blob existed (0 blobs were ever stored). `acquisition_status` records whether
    -- the bytes were fetched ('acquired'), are known unrecoverable from the source
    -- polylogue holds ('unavailable'), or have not yet been fetched ('unfetched').
    blob_hash              BLOB CHECK(blob_hash IS NULL OR length(blob_hash) = 32)""",
        ),
        _raw_column(
            "acquisition_status",
            f"""acquisition_status     TEXT NOT NULL DEFAULT 'unfetched'
                               CHECK({literal_check("acquisition_status", "acquired", "unavailable", "unfetched")})""",
        ),
        _raw_column("ref_count", """ref_count              INTEGER NOT NULL DEFAULT 0 CHECK(ref_count >= 0)"""),
    ),
)

ATTACHMENT_REFS_SPEC = _make_table_spec(
    "attachment_refs",
    (
        _raw_column(
            "ref_id",
            """ref_id                 TEXT GENERATED ALWAYS AS (message_id || ':attachment:' || position) STORED UNIQUE""",
        ),
        _raw_column(
            "attachment_id",
            """attachment_id          TEXT NOT NULL REFERENCES attachments(attachment_id) ON DELETE CASCADE""",
        ),
        _raw_column(
            "session_id", """session_id             TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"""
        ),
        _raw_column(
            "message_id", """message_id             TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE"""
        ),
        _raw_column("position", """position               INTEGER NOT NULL CHECK(position >= 0)"""),
        _raw_column(
            "upload_origin",
            f"""upload_origin          TEXT CHECK({literal_check("upload_origin", "drive", "paste", "url", "oauth")} OR upload_origin IS NULL)""",
        ),
        _raw_column("source_url", """source_url             TEXT"""),
        _raw_column("caption", """caption                TEXT"""),
    ),
    table_constraints=("""PRIMARY KEY(message_id, position)""",),
)

ATTACHMENT_NATIVE_IDS_SPEC = _make_table_spec(
    "attachment_native_ids",
    (
        _raw_column("ref_id", """ref_id     TEXT NOT NULL REFERENCES attachment_refs(ref_id) ON DELETE CASCADE"""),
        _raw_column(
            "id_kind",
            f"""id_kind    TEXT NOT NULL CHECK({literal_check("id_kind", "attachment", "file", "drive", "url")})""",
        ),
        _raw_column("native_id", """native_id  TEXT NOT NULL"""),
    ),
    table_constraints=("""PRIMARY KEY(ref_id, id_kind, native_id)""",),
)

PASTE_SPANS_SPEC = _make_table_spec(
    "paste_spans",
    (
        _raw_column(
            "paste_id", """paste_id        TEXT GENERATED ALWAYS AS (message_id || ':' || position) STORED UNIQUE"""
        ),
        _raw_column(
            "message_id", """message_id      TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE"""
        ),
        _raw_column(
            "session_id", """session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"""
        ),
        _raw_column("position", """position        INTEGER NOT NULL CHECK(position >= 0)"""),
        _raw_column("start_offset", """start_offset    INTEGER CHECK(start_offset IS NULL OR start_offset >= 0)"""),
        _raw_column(
            "end_offset", """end_offset      INTEGER CHECK(end_offset IS NULL OR end_offset >= start_offset)"""
        ),
        _raw_column(
            "boundary_state", f"""boundary_state  TEXT NOT NULL CHECK ({check("boundary_state", PasteBoundary)})"""
        ),
        _raw_column("source_event_id", """source_event_id TEXT"""),
        _raw_column("source_marker", """source_marker   TEXT"""),
        _raw_column("content_hash", """content_hash    BLOB NOT NULL CHECK(length(content_hash) = 32)"""),
        _raw_column("observed_at_ms", """observed_at_ms  INTEGER"""),
    ),
    table_constraints=("""PRIMARY KEY(message_id, position)""",),
)

SESSION_MODEL_USAGE_SPEC = _make_table_spec(
    "session_model_usage",
    (
        _raw_column(
            "session_id", """session_id              TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"""
        ),
        _raw_column("model_name", """model_name              TEXT NOT NULL"""),
        _raw_column("input_tokens", """input_tokens            INTEGER NOT NULL DEFAULT 0 CHECK(input_tokens >= 0)"""),
        _raw_column(
            "output_tokens", """output_tokens           INTEGER NOT NULL DEFAULT 0 CHECK(output_tokens >= 0)"""
        ),
        _raw_column(
            "cache_read_tokens", """cache_read_tokens       INTEGER NOT NULL DEFAULT 0 CHECK(cache_read_tokens >= 0)"""
        ),
        _raw_column(
            "cache_write_tokens",
            """cache_write_tokens      INTEGER NOT NULL DEFAULT 0 CHECK(cache_write_tokens >= 0)""",
        ),
        _raw_column(
            "message_count", """message_count           INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0)"""
        ),
        _raw_column("provider_cost_usd", """provider_cost_usd       REAL"""),
        _raw_column("catalog_cost_usd", """catalog_cost_usd        REAL"""),
        _raw_column("cost_credits", """cost_credits            REAL"""),
    ),
    table_constraints=(
        """CHECK (provider_cost_usd IS NULL OR provider_cost_usd >= 0)""",
        """CHECK (catalog_cost_usd IS NULL OR catalog_cost_usd >= 0)""",
        """PRIMARY KEY(session_id, model_name)""",
    ),
)

SESSION_PROVIDER_USAGE_EVENTS_SPEC = _make_table_spec(
    "session_provider_usage_events",
    (
        _raw_column(
            "usage_event_id",
            """usage_event_id                 TEXT GENERATED ALWAYS AS (session_id || ':usage:' || position) STORED UNIQUE""",
        ),
        _raw_column(
            "session_id",
            """session_id                     TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE""",
        ),
        _raw_column(
            "source_message_id",
            """source_message_id              TEXT REFERENCES messages(message_id) ON DELETE SET NULL""",
        ),
        _raw_column("position", """position                       INTEGER NOT NULL CHECK(position >= 0)"""),
        _raw_column(
            "provider_event_type",
            f"""provider_event_type            TEXT NOT NULL CHECK({literal_check("provider_event_type", "token_count", "message_usage")})""",
        ),
        _raw_column("model_name", """model_name                     TEXT"""),
        _raw_column(
            "last_input_tokens",
            """last_input_tokens              INTEGER NOT NULL DEFAULT 0 CHECK(last_input_tokens >= 0)""",
        ),
        _raw_column(
            "last_output_tokens",
            """last_output_tokens             INTEGER NOT NULL DEFAULT 0 CHECK(last_output_tokens >= 0)""",
        ),
        _raw_column(
            "last_cached_input_tokens",
            """last_cached_input_tokens       INTEGER NOT NULL DEFAULT 0 CHECK(last_cached_input_tokens >= 0)""",
        ),
        _raw_column(
            "last_cache_write_tokens",
            """last_cache_write_tokens        INTEGER NOT NULL DEFAULT 0 CHECK(last_cache_write_tokens >= 0)""",
        ),
        _raw_column(
            "last_reasoning_output_tokens",
            """last_reasoning_output_tokens   INTEGER NOT NULL DEFAULT 0 CHECK(last_reasoning_output_tokens >= 0)""",
        ),
        _raw_column(
            "last_total_tokens",
            """last_total_tokens              INTEGER CHECK(last_total_tokens IS NULL OR last_total_tokens >= 0)""",
        ),
        _raw_column(
            "total_input_tokens",
            """total_input_tokens             INTEGER NOT NULL DEFAULT 0 CHECK(total_input_tokens >= 0)""",
        ),
        _raw_column(
            "total_output_tokens",
            """total_output_tokens            INTEGER NOT NULL DEFAULT 0 CHECK(total_output_tokens >= 0)""",
        ),
        _raw_column(
            "total_cached_input_tokens",
            """total_cached_input_tokens      INTEGER NOT NULL DEFAULT 0 CHECK(total_cached_input_tokens >= 0)""",
        ),
        _raw_column(
            "total_cache_write_tokens",
            """total_cache_write_tokens       INTEGER NOT NULL DEFAULT 0 CHECK(total_cache_write_tokens >= 0)""",
        ),
        _raw_column(
            "total_reasoning_output_tokens",
            """total_reasoning_output_tokens  INTEGER NOT NULL DEFAULT 0 CHECK(total_reasoning_output_tokens >= 0)""",
        ),
        _raw_column(
            "total_tokens",
            """total_tokens                   INTEGER CHECK(total_tokens IS NULL OR total_tokens >= 0)""",
        ),
        _raw_column("occurred_at_ms", """occurred_at_ms                 INTEGER"""),
    ),
    table_constraints=("""PRIMARY KEY(session_id, position)""",),
)

SESSION_TAGS_SPEC = _make_table_spec(
    "session_tags",
    (
        _raw_column("session_id", """session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"""),
        _raw_column("tag", """tag           TEXT NOT NULL"""),
        _raw_column(
            "tag_source", f"""tag_source    TEXT NOT NULL CHECK({literal_check("tag_source", "user", "auto")})"""
        ),
        _raw_column("method", """method        TEXT"""),
        _raw_column("confidence", """confidence    REAL CHECK(confidence IS NULL OR confidence BETWEEN 0 AND 1)"""),
        _raw_column("evidence_json", """evidence_json TEXT"""),
    ),
    table_constraints=("""PRIMARY KEY(session_id, tag, tag_source)""",),
)

INSIGHT_MATERIALIZATION_SPEC = _make_table_spec(
    "insight_materialization",
    (
        _raw_column(
            "insight_type",
            """insight_type                 TEXT NOT NULL CHECK(insight_type IN (
                                    'session_profile', 'work_events', 'phases', 'latency', 'thread',
                                    'runs', 'observed_events', 'context_snapshots', 'provider_usage'))""",
        ),
        _raw_column(
            "session_id",
            """session_id                   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE""",
        ),
        _raw_column("materializer_version", """materializer_version         INTEGER NOT NULL"""),
        _raw_column("materialized_at_ms", """materialized_at_ms           INTEGER NOT NULL"""),
        _raw_column("source_updated_at_ms", """source_updated_at_ms         INTEGER"""),
        _raw_column("source_sort_key_ms", """source_sort_key_ms           INTEGER"""),
        _raw_column("input_high_water_mark_ms", """input_high_water_mark_ms     INTEGER"""),
        _raw_column("input_high_water_mark_source", """input_high_water_mark_source TEXT"""),
        _raw_column(
            "input_row_count",
            """input_row_count              INTEGER NOT NULL DEFAULT 0 CHECK(input_row_count >= 0)""",
        ),
    ),
    table_constraints=("""PRIMARY KEY(insight_type, session_id)""",),
)

SESSION_WORK_EVENTS_SPEC = _make_table_spec(
    "session_work_events",
    (
        _raw_column(
            "event_id",
            """event_id           TEXT GENERATED ALWAYS AS (session_id || ':work_event:' || position) STORED UNIQUE""",
        ),
        _raw_column(
            "session_id", """session_id         TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"""
        ),
        _raw_column("position", """position           INTEGER NOT NULL CHECK(position >= 0)"""),
        _raw_column("work_event_type", """work_event_type    TEXT NOT NULL"""),
        _raw_column("summary", """summary            TEXT NOT NULL"""),
        _raw_column("confidence", """confidence         REAL NOT NULL DEFAULT 0.0 CHECK(confidence BETWEEN 0 AND 1)"""),
        _raw_column("start_index", """start_index        INTEGER NOT NULL DEFAULT 0 CHECK(start_index >= 0)"""),
        _raw_column("end_index", """end_index          INTEGER NOT NULL DEFAULT 0 CHECK(end_index >= start_index)"""),
        _raw_column("started_at_ms", """started_at_ms      INTEGER"""),
        _raw_column("ended_at_ms", """ended_at_ms        INTEGER"""),
        _raw_column("duration_ms", """duration_ms        INTEGER NOT NULL DEFAULT 0 CHECK(duration_ms >= 0)"""),
        _raw_column("file_paths_json", """file_paths_json    TEXT NOT NULL DEFAULT '[]'"""),
        _raw_column("tools_used_json", """tools_used_json    TEXT NOT NULL DEFAULT '[]'"""),
        _raw_column("input_high_water_mark", """input_high_water_mark        TEXT"""),
        _raw_column("input_high_water_mark_source", """input_high_water_mark_source TEXT"""),
        _raw_column("evidence_json", """evidence_json      TEXT NOT NULL DEFAULT '{}'"""),
        _raw_column("inference_json", """inference_json     TEXT NOT NULL DEFAULT '{}'"""),
        _raw_column("search_text", """search_text        TEXT NOT NULL DEFAULT ''"""),
    ),
    table_constraints=("""PRIMARY KEY(session_id, position)""",),
)

SESSION_PHASES_SPEC = _make_table_spec(
    "session_phases",
    (
        _raw_column(
            "phase_id",
            """phase_id        TEXT GENERATED ALWAYS AS (session_id || ':phase:' || position) STORED UNIQUE""",
        ),
        _raw_column(
            "session_id", """session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"""
        ),
        _raw_column("position", """position        INTEGER NOT NULL CHECK(position >= 0)"""),
        _raw_column("start_index", """start_index     INTEGER NOT NULL DEFAULT 0 CHECK(start_index >= 0)"""),
        _raw_column("end_index", """end_index       INTEGER NOT NULL DEFAULT 0 CHECK(end_index >= start_index)"""),
        _raw_column("started_at_ms", """started_at_ms   INTEGER"""),
        _raw_column("ended_at_ms", """ended_at_ms     INTEGER"""),
        _raw_column("duration_ms", """duration_ms     INTEGER NOT NULL DEFAULT 0 CHECK(duration_ms >= 0)"""),
        _raw_column("tool_counts_json", """tool_counts_json TEXT NOT NULL DEFAULT '{}'"""),
        _raw_column("word_count", """word_count      INTEGER NOT NULL DEFAULT 0 CHECK(word_count >= 0)"""),
        _raw_column("input_high_water_mark", """input_high_water_mark        TEXT"""),
        _raw_column("input_high_water_mark_source", """input_high_water_mark_source TEXT"""),
        _raw_column("evidence_json", """evidence_json   TEXT NOT NULL DEFAULT '{}'"""),
        _raw_column("inference_json", """inference_json  TEXT NOT NULL DEFAULT '{}'"""),
        _raw_column("search_text", """search_text     TEXT NOT NULL DEFAULT ''"""),
    ),
    table_constraints=("""PRIMARY KEY(session_id, position)""",),
)

SESSION_LATENCY_PROFILES_SPEC = _make_table_spec(
    "session_latency_profiles",
    (
        _raw_column(
            "session_id",
            """session_id                       TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE""",
        ),
        _raw_column("materializer_version", """materializer_version             INTEGER NOT NULL DEFAULT 5"""),
        _raw_column("materialized_at", """materialized_at                  TEXT NOT NULL"""),
        _raw_column("source_updated_at", """source_updated_at                TEXT"""),
        _raw_column("source_sort_key", """source_sort_key                  REAL"""),
        _raw_column("input_high_water_mark", """input_high_water_mark            TEXT"""),
        _raw_column("input_high_water_mark_source", """input_high_water_mark_source     TEXT"""),
        _raw_column(
            "input_row_count",
            """input_row_count                  INTEGER NOT NULL DEFAULT 0 CHECK(input_row_count >= 0)""",
        ),
        _raw_column("source_name", """source_name                      TEXT NOT NULL"""),
        _raw_column("title", """title                            TEXT"""),
        _raw_column("first_message_at", """first_message_at                 TEXT"""),
        _raw_column("last_message_at", """last_message_at                  TEXT"""),
        _raw_column("canonical_session_date", """canonical_session_date           TEXT"""),
        _raw_column(
            "median_tool_call_ms",
            """median_tool_call_ms              INTEGER NOT NULL DEFAULT 0 CHECK(median_tool_call_ms >= 0)""",
        ),
        _raw_column(
            "p90_tool_call_ms",
            """p90_tool_call_ms                 INTEGER NOT NULL DEFAULT 0 CHECK(p90_tool_call_ms >= 0)""",
        ),
        _raw_column(
            "max_tool_call_ms",
            """max_tool_call_ms                 INTEGER NOT NULL DEFAULT 0 CHECK(max_tool_call_ms >= 0)""",
        ),
        _raw_column(
            "stuck_tool_count",
            """stuck_tool_count                 INTEGER NOT NULL DEFAULT 0 CHECK(stuck_tool_count >= 0)""",
        ),
        _raw_column(
            "median_agent_response_ms",
            """median_agent_response_ms         INTEGER NOT NULL DEFAULT 0 CHECK(median_agent_response_ms >= 0)""",
        ),
        _raw_column(
            "median_user_response_ms",
            """median_user_response_ms          INTEGER NOT NULL DEFAULT 0 CHECK(median_user_response_ms >= 0)""",
        ),
        _raw_column(
            "tool_call_count_by_category_json", """tool_call_count_by_category_json TEXT NOT NULL DEFAULT '{}'"""
        ),
        _raw_column("evidence_payload_json", """evidence_payload_json            TEXT NOT NULL DEFAULT '{}'"""),
        _raw_column("search_text", """search_text                      TEXT NOT NULL DEFAULT ''"""),
    ),
)

SESSION_PROFILES_SPEC = _make_table_spec(
    "session_profiles",
    (
        _raw_column(
            "session_id",
            """session_id                      TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE""",
        ),
        _raw_column("logical_session_id", """logical_session_id              TEXT"""),
        _raw_column("materializer_version", """materializer_version            INTEGER NOT NULL DEFAULT 5"""),
        _raw_column("materialized_at", """materialized_at                 TEXT NOT NULL DEFAULT ''"""),
        _raw_column("source_updated_at", """source_updated_at               TEXT"""),
        _raw_column("source_sort_key", """source_sort_key                 REAL"""),
        _raw_column("input_high_water_mark", """input_high_water_mark           TEXT"""),
        _raw_column("input_high_water_mark_source", """input_high_water_mark_source    TEXT"""),
        _raw_column("input_content_hash", """input_content_hash              TEXT"""),
        _raw_column(
            "input_row_count",
            """input_row_count                 INTEGER NOT NULL DEFAULT 0 CHECK(input_row_count >= 0)""",
        ),
        _raw_column("source_name", """source_name                     TEXT NOT NULL DEFAULT ''"""),
        _raw_column("title", """title                           TEXT"""),
        _raw_column("first_message_at", """first_message_at                TEXT"""),
        _raw_column("last_message_at", """last_message_at                 TEXT"""),
        _raw_column("canonical_session_date", """canonical_session_date          TEXT"""),
        _raw_column("repo_paths_json", """repo_paths_json                 TEXT"""),
        _raw_column("repo_names_json", """repo_names_json                 TEXT"""),
        _raw_column("tags_json", """tags_json                       TEXT"""),
        _raw_column("auto_tags_json", """auto_tags_json                  TEXT"""),
        _raw_column(
            "message_count", """message_count                   INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0)"""
        ),
        _raw_column(
            "substantive_count",
            """substantive_count               INTEGER NOT NULL DEFAULT 0 CHECK(substantive_count >= 0)""",
        ),
        _raw_column(
            "attachment_count",
            """attachment_count                INTEGER NOT NULL DEFAULT 0 CHECK(attachment_count >= 0)""",
        ),
        _raw_column(
            "work_event_count",
            """work_event_count                INTEGER NOT NULL DEFAULT 0 CHECK(work_event_count >= 0)""",
        ),
        _raw_column(
            "phase_count", """phase_count                     INTEGER NOT NULL DEFAULT 0 CHECK(phase_count >= 0)"""
        ),
        _raw_column(
            "word_count", """word_count                      INTEGER NOT NULL DEFAULT 0 CHECK(word_count >= 0)"""
        ),
        _raw_column(
            "tool_use_count",
            """tool_use_count                  INTEGER NOT NULL DEFAULT 0 CHECK(tool_use_count >= 0)""",
        ),
        _raw_column(
            "thinking_count",
            """thinking_count                  INTEGER NOT NULL DEFAULT 0 CHECK(thinking_count >= 0)""",
        ),
        _raw_column(
            "total_duration_ms",
            """total_duration_ms               INTEGER NOT NULL DEFAULT 0 CHECK(total_duration_ms >= 0)""",
        ),
        _raw_column(
            "engaged_duration_ms",
            """engaged_duration_ms             INTEGER NOT NULL DEFAULT 0 CHECK(engaged_duration_ms >= 0)""",
        ),
        _raw_column(
            "tool_active_duration_ms",
            """tool_active_duration_ms         INTEGER NOT NULL DEFAULT 0 CHECK(tool_active_duration_ms >= 0)""",
        ),
        _raw_column(
            "wall_duration_ms",
            """wall_duration_ms                INTEGER NOT NULL DEFAULT 0 CHECK(wall_duration_ms >= 0)""",
        ),
        _raw_column("workflow_shape", """workflow_shape                  TEXT"""),
        _raw_column("workflow_shape_method", """workflow_shape_method           TEXT"""),
        _raw_column(
            "workflow_shape_confidence",
            """workflow_shape_confidence       REAL CHECK(workflow_shape_confidence BETWEEN 0 AND 1 OR workflow_shape_confidence IS NULL)""",
        ),
        _raw_column("workflow_shape_features_json", """workflow_shape_features_json    TEXT NOT NULL DEFAULT '{}'"""),
        _raw_column("terminal_state", """terminal_state                  TEXT"""),
        _raw_column("terminal_state_method", """terminal_state_method           TEXT"""),
        _raw_column(
            "terminal_state_confidence",
            """terminal_state_confidence       REAL CHECK(terminal_state_confidence BETWEEN 0 AND 1 OR terminal_state_confidence IS NULL)""",
        ),
        _raw_column("terminal_state_evidence_json", """terminal_state_evidence_json    TEXT NOT NULL DEFAULT '{}'"""),
        _raw_column(
            "thinking_duration_ms",
            """thinking_duration_ms            INTEGER NOT NULL DEFAULT 0 CHECK(thinking_duration_ms >= 0)""",
        ),
        _raw_column(
            "output_duration_ms",
            """output_duration_ms              INTEGER NOT NULL DEFAULT 0 CHECK(output_duration_ms >= 0)""",
        ),
        _raw_column(
            "tool_duration_ms",
            """tool_duration_ms                INTEGER NOT NULL DEFAULT 0 CHECK(tool_duration_ms >= 0)""",
        ),
        _raw_column("latency_percentiles_ms_json", """latency_percentiles_ms_json     TEXT NOT NULL DEFAULT '{}'"""),
        _raw_column("tool_calls_per_minute", """tool_calls_per_minute           REAL"""),
        _raw_column(
            "timing_provenance", """timing_provenance               TEXT NOT NULL DEFAULT 'sort_key_estimated'"""
        ),
        _raw_column("evidence_payload_json", """evidence_payload_json           TEXT NOT NULL DEFAULT '{}'"""),
        _raw_column("inference_payload_json", """inference_payload_json          TEXT NOT NULL DEFAULT '{}'"""),
        _raw_column("enrichment_payload_json", """enrichment_payload_json         TEXT NOT NULL DEFAULT '{}'"""),
        _raw_column("evidence_search_text", """evidence_search_text            TEXT NOT NULL DEFAULT ''"""),
        _raw_column("inference_search_text", """inference_search_text           TEXT NOT NULL DEFAULT ''"""),
        _raw_column("enrichment_search_text", """enrichment_search_text          TEXT NOT NULL DEFAULT ''"""),
        _raw_column("enrichment_version", """enrichment_version              INTEGER NOT NULL DEFAULT 1"""),
        _raw_column(
            "enrichment_family",
            """enrichment_family               TEXT NOT NULL DEFAULT 'scored_session_enrichment'""",
        ),
        _raw_column("inference_version", """inference_version               INTEGER NOT NULL DEFAULT 1"""),
        _raw_column(
            "inference_family",
            """inference_family                TEXT NOT NULL DEFAULT 'heuristic_session_semantics'""",
        ),
        _raw_column("search_text", """search_text                     TEXT NOT NULL DEFAULT ''"""),
        _raw_column(
            "duration_ms", """duration_ms                     INTEGER CHECK(duration_ms IS NULL OR duration_ms >= 0)"""
        ),
        _raw_column(
            "primary_model_name",
            """-- 1vpm.1: dominant model by assistant output-token share + its canonical
    -- family (anthropic/openai/deepseek/...) -- the enabling primitive for
    -- the `delegations` view's orchestrator/subagent model identity.
    primary_model_name              TEXT""",
        ),
        _raw_column("primary_model_family", """primary_model_family            TEXT"""),
    ),
)

DERIVED_REFRESH_GUARD_SPEC = _make_table_spec(
    "derived_refresh_guard",
    (_raw_column("guard_name", """guard_name TEXT PRIMARY KEY"""),),
)

WORK_EVIDENCE_GRAPHS_SPEC = _make_table_spec(
    "work_evidence_graphs",
    (
        _raw_column("graph_id", """graph_id              TEXT PRIMARY KEY"""),
        _raw_column("corpus_snapshot_ref", """corpus_snapshot_ref   TEXT NOT NULL"""),
    ),
)

WORK_EVIDENCE_NODES_SPEC = _make_table_spec(
    "work_evidence_nodes",
    (
        _raw_column(
            "graph_id",
            """graph_id                       TEXT NOT NULL REFERENCES work_evidence_graphs(graph_id) ON DELETE CASCADE""",
        ),
        _raw_column("node_ref", """node_ref                       TEXT NOT NULL"""),
        _raw_column("node_kind", """node_kind                      TEXT NOT NULL"""),
        _raw_column("label", """label                          TEXT NOT NULL"""),
        _raw_column("evidence_refs_json", """evidence_refs_json             TEXT NOT NULL"""),
        _raw_column("corpus_snapshot_ref", """corpus_snapshot_ref            TEXT NOT NULL"""),
        _raw_column("authority", """authority                      TEXT NOT NULL"""),
        _raw_column("confidence", """confidence                     REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1)"""),
        _raw_column(
            "occurred_at_ms",
            """occurred_at_ms                 INTEGER CHECK(occurred_at_ms IS NULL OR occurred_at_ms >= 0)""",
        ),
        _raw_column("actor_ref", """actor_ref                      TEXT"""),
        _raw_column("execution_context_id", """execution_context_id           TEXT"""),
        _raw_column("role", """role                           TEXT NOT NULL DEFAULT 'unknown'"""),
        _raw_column("execution_context_known_json", """execution_context_known_json   TEXT NOT NULL DEFAULT '[]'"""),
        _raw_column("execution_context_unknown_json", """execution_context_unknown_json TEXT NOT NULL DEFAULT '[]'"""),
        _raw_column(
            "execution_context_addressed",
            """execution_context_addressed    INTEGER CHECK(execution_context_addressed IN (0, 1) OR execution_context_addressed IS NULL)""",
        ),
        _raw_column("association_state", """association_state              TEXT NOT NULL"""),
        _raw_column("claim_text", """claim_text                     TEXT"""),
    ),
    table_constraints=("""PRIMARY KEY(graph_id, node_ref)""",),
)

WORK_EVIDENCE_EDGES_SPEC = _make_table_spec(
    "work_evidence_edges",
    (
        _raw_column(
            "graph_id",
            """graph_id             TEXT NOT NULL REFERENCES work_evidence_graphs(graph_id) ON DELETE CASCADE""",
        ),
        _raw_column("edge_ref", """edge_ref             TEXT NOT NULL"""),
        _raw_column("edge_kind", """edge_kind            TEXT NOT NULL"""),
        _raw_column("source_ref", """source_ref           TEXT NOT NULL"""),
        _raw_column("target_ref", """target_ref           TEXT NOT NULL"""),
        _raw_column("evidence_refs_json", """evidence_refs_json   TEXT NOT NULL"""),
        _raw_column("corpus_snapshot_ref", """corpus_snapshot_ref  TEXT NOT NULL"""),
        _raw_column("authority", """authority            TEXT NOT NULL"""),
        _raw_column("confidence", """confidence           REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1)"""),
        _raw_column(
            "occurred_at_ms", """occurred_at_ms       INTEGER CHECK(occurred_at_ms IS NULL OR occurred_at_ms >= 0)"""
        ),
        _raw_column("association_state", """association_state    TEXT NOT NULL"""),
    ),
    table_constraints=(
        """PRIMARY KEY(graph_id, edge_ref)""",
        """FOREIGN KEY(graph_id, source_ref) REFERENCES work_evidence_nodes(graph_id, node_ref) ON DELETE CASCADE""",
        """FOREIGN KEY(graph_id, target_ref) REFERENCES work_evidence_nodes(graph_id, node_ref) ON DELETE CASCADE""",
    ),
)

AGENT_META_SIDECAR_PURGE_RECEIPTS_SPEC = _make_table_spec(
    "agent_meta_sidecar_purge_receipts",
    (
        _raw_column("session_id", """session_id           TEXT PRIMARY KEY"""),
        _raw_column("origin", """origin               TEXT NOT NULL"""),
        _raw_column("native_id", """native_id            TEXT NOT NULL"""),
        _raw_column("raw_id", """raw_id               TEXT NOT NULL"""),
        _raw_column("source_path", """source_path          TEXT NOT NULL"""),
        _raw_column("purged_at_ms", """purged_at_ms         INTEGER NOT NULL CHECK(purged_at_ms >= 0)"""),
        _raw_column("tool_version", """tool_version         TEXT NOT NULL"""),
        _raw_column("backup_manifest_path", """backup_manifest_path TEXT NOT NULL"""),
        _raw_column("detail", """detail               TEXT NOT NULL DEFAULT ''"""),
    ),
)

INDEX_TABLE_SPECS = {
    "fts_freshness_state": FTS_FRESHNESS_STATE_SPEC,
    "query_unit_frame_state": QUERY_UNIT_FRAME_STATE_SPEC,
    "raw_revision_applications": RAW_REVISION_APPLICATIONS_SPEC,
    "raw_revision_heads": RAW_REVISION_HEADS_SPEC,
    "sessions": SESSIONS_SPEC,
    "web_content_constructs": WEB_CONTENT_CONSTRUCTS_SPEC,
    "file_edits": FILE_EDITS_SPEC,
    "session_refs": SESSION_REFS_SPEC,
    "session_events": SESSION_EVENTS_SPEC,
    "session_agent_policies": SESSION_AGENT_POLICIES_SPEC,
    "session_links": SESSION_LINKS_SPEC,
    "session_working_dirs": SESSION_WORKING_DIRS_SPEC,
    "repos": REPOS_SPEC,
    "repo_checkouts": REPO_CHECKOUTS_SPEC,
    "session_repos": SESSION_REPOS_SPEC,
    "session_commits": SESSION_COMMITS_SPEC,
    "attachments": ATTACHMENTS_SPEC,
    "attachment_refs": ATTACHMENT_REFS_SPEC,
    "attachment_native_ids": ATTACHMENT_NATIVE_IDS_SPEC,
    "paste_spans": PASTE_SPANS_SPEC,
    "session_model_usage": SESSION_MODEL_USAGE_SPEC,
    "session_provider_usage_events": SESSION_PROVIDER_USAGE_EVENTS_SPEC,
    "session_tags": SESSION_TAGS_SPEC,
    "insight_materialization": INSIGHT_MATERIALIZATION_SPEC,
    "session_work_events": SESSION_WORK_EVENTS_SPEC,
    "session_phases": SESSION_PHASES_SPEC,
    "session_latency_profiles": SESSION_LATENCY_PROFILES_SPEC,
    "session_profiles": SESSION_PROFILES_SPEC,
    "derived_refresh_guard": DERIVED_REFRESH_GUARD_SPEC,
    "work_evidence_graphs": WORK_EVIDENCE_GRAPHS_SPEC,
    "work_evidence_nodes": WORK_EVIDENCE_NODES_SPEC,
    "work_evidence_edges": WORK_EVIDENCE_EDGES_SPEC,
    "agent_meta_sidecar_purge_receipts": AGENT_META_SIDECAR_PURGE_RECEIPTS_SPEC,
}


# Global registry of table specs
MESSAGES_SPEC = _make_messages_spec()
BLOCKS_SPEC = _make_blocks_spec()

TABLE_SPECS = {
    "messages": MESSAGES_SPEC,
    "blocks": BLOCKS_SPEC,
    **INDEX_TABLE_SPECS,
}
