"""Shared schema bootstrap planning for sync and async SQLite backends."""

from __future__ import annotations

import re
import sqlite3
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, TypeAlias

import aiosqlite

from polylogue.errors import DatabaseError
from polylogue.storage.backends.schema_ddl import (
    _ACTION_EVENT_DDL,
    _ACTION_FTS_DDL,
    _ARTIFACT_OBSERVATION_DDL,
    _PUBLICATION_DDL,
    _SESSION_PRODUCT_DDL,
    _VEC0_DDL,
    SCHEMA_DDL,
    SCHEMA_VERSION,
)

_RAW_SOURCE_MTIME_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_raw_conv_source_mtime "
    "ON raw_conversations(source_path, file_mtime) "
    "WHERE file_mtime IS NOT NULL"
)


@dataclass(frozen=True)
class SchemaSnapshot:
    """Minimal database state needed to plan bootstrap extensions."""

    current_version: int
    table_columns: dict[str, frozenset[str]]
    index_sql: dict[str, str | None]

    def has_table(self, table_name: str) -> bool:
        return table_name in self.table_columns

    def columns(self, table_name: str) -> frozenset[str]:
        return self.table_columns.get(table_name, frozenset())

    def sql_for_index(self, index_name: str) -> str | None:
        return self.index_sql.get(index_name)


@dataclass(frozen=True)
class SchemaExtensionPlan:
    """Ordered SQL needed to bring a current-version DB up to current extensions."""

    statements: tuple[str, ...]
    scripts: tuple[str, ...]


@dataclass(frozen=True)
class SchemaBootstrapDecision:
    """Shared schema bootstrap branch chosen from a schema snapshot."""

    action: Literal["create_fresh", "apply_current_extensions"]
    extension_plan: SchemaExtensionPlan | None = None


@dataclass(frozen=True)
class SchemaColumnExtensionDescriptor:
    """Declarative check for a column added after the base schema version."""

    table_name: str
    column_name: str
    ddl: str

    def snapshot_tables(self) -> tuple[str, ...]:
        return (self.table_name,)

    def snapshot_indexes(self) -> tuple[str, ...]:
        return ()

    def statements(self, snapshot: SchemaSnapshot) -> tuple[str, ...]:
        if not snapshot.has_table(self.table_name):
            return ()
        if self.column_name in snapshot.columns(self.table_name):
            return ()
        return (self.ddl,)

    def scripts(self, _snapshot: SchemaSnapshot) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class SchemaIndexExtensionDescriptor:
    """Declarative check for an index extension and optional drift repair."""

    table_name: str
    index_name: str
    ddl: str
    replace_on_drift: bool = False

    def snapshot_tables(self) -> tuple[str, ...]:
        return (self.table_name,)

    def snapshot_indexes(self) -> tuple[str, ...]:
        return (self.index_name,)

    def statements(self, snapshot: SchemaSnapshot) -> tuple[str, ...]:
        if not snapshot.has_table(self.table_name):
            return ()

        existing_sql = snapshot.sql_for_index(self.index_name)
        if existing_sql is None:
            return (self.ddl,)

        if self.replace_on_drift and _normalize_sql(existing_sql) != _normalize_sql(self.ddl):
            return (f"DROP INDEX IF EXISTS {self.index_name}", self.ddl)

        return ()

    def scripts(self, _snapshot: SchemaSnapshot) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class SchemaScriptExtensionDescriptor:
    """Reviewable DDL script that is safe to run for current schemas."""

    ddl: str

    def snapshot_tables(self) -> tuple[str, ...]:
        return ()

    def snapshot_indexes(self) -> tuple[str, ...]:
        return ()

    def statements(self, _snapshot: SchemaSnapshot) -> tuple[str, ...]:
        return ()

    def scripts(self, _snapshot: SchemaSnapshot) -> tuple[str, ...]:
        return (self.ddl,)


@dataclass(frozen=True)
class SchemaBackfillDescriptor:
    """Data repair statement gated by a table's presence."""

    table_name: str
    ddl: str

    def snapshot_tables(self) -> tuple[str, ...]:
        return (self.table_name,)

    def snapshot_indexes(self) -> tuple[str, ...]:
        return ()

    def statements(self, snapshot: SchemaSnapshot) -> tuple[str, ...]:
        if not snapshot.has_table(self.table_name):
            return ()
        return (self.ddl,)

    def scripts(self, _snapshot: SchemaSnapshot) -> tuple[str, ...]:
        return ()


SchemaExtensionDescriptor: TypeAlias = (
    SchemaColumnExtensionDescriptor
    | SchemaIndexExtensionDescriptor
    | SchemaScriptExtensionDescriptor
    | SchemaBackfillDescriptor
)


def _normalize_sql(sql: str) -> str:
    normalized = " ".join(sql.replace(";", " ").split())
    normalized = re.sub(r"\bIF\s+NOT\s+EXISTS\b", "", normalized, flags=re.IGNORECASE)
    return " ".join(normalized.split())


_SCHEMA_EXTENSION_DESCRIPTORS: tuple[SchemaExtensionDescriptor, ...] = (
    SchemaColumnExtensionDescriptor(
        table_name="raw_conversations",
        column_name="blob_size",
        ddl="ALTER TABLE raw_conversations ADD COLUMN blob_size INTEGER NOT NULL DEFAULT 0",
    ),
    SchemaIndexExtensionDescriptor(
        table_name="raw_conversations",
        index_name="idx_raw_conv_source_mtime",
        ddl=_RAW_SOURCE_MTIME_INDEX_SQL,
        replace_on_drift=True,
    ),
    SchemaIndexExtensionDescriptor(
        table_name="content_blocks",
        index_name="idx_content_blocks_tool_use_conversation",
        ddl="""
            CREATE INDEX IF NOT EXISTS idx_content_blocks_tool_use_conversation
            ON content_blocks(conversation_id)
            WHERE type = 'tool_use'
            """,
    ),
    SchemaIndexExtensionDescriptor(
        table_name="attachments",
        index_name="idx_attachments_provider_meta_id",
        ddl="""
                CREATE INDEX IF NOT EXISTS idx_attachments_provider_meta_id
                ON attachments(json_extract(provider_meta, '$.id'))
                WHERE provider_meta IS NOT NULL
                """,
    ),
    SchemaIndexExtensionDescriptor(
        table_name="attachments",
        index_name="idx_attachments_provider_meta_provider_id",
        ddl="""
                CREATE INDEX IF NOT EXISTS idx_attachments_provider_meta_provider_id
                ON attachments(json_extract(provider_meta, '$.provider_id'))
                WHERE provider_meta IS NOT NULL
                """,
    ),
    SchemaIndexExtensionDescriptor(
        table_name="attachments",
        index_name="idx_attachments_provider_meta_file_id",
        ddl="""
                CREATE INDEX IF NOT EXISTS idx_attachments_provider_meta_file_id
                ON attachments(json_extract(provider_meta, '$.fileId'))
                WHERE provider_meta IS NOT NULL
                """,
    ),
    SchemaIndexExtensionDescriptor(
        table_name="attachments",
        index_name="idx_attachments_provider_meta_drive_id",
        ddl="""
                CREATE INDEX IF NOT EXISTS idx_attachments_provider_meta_drive_id
                ON attachments(json_extract(provider_meta, '$.driveId'))
                WHERE provider_meta IS NOT NULL
                """,
    ),
    SchemaIndexExtensionDescriptor(
        table_name="attachment_refs",
        index_name="idx_attachment_refs_provider_meta_id",
        ddl="""
                CREATE INDEX IF NOT EXISTS idx_attachment_refs_provider_meta_id
                ON attachment_refs(json_extract(provider_meta, '$.id'))
                WHERE provider_meta IS NOT NULL
                """,
    ),
    SchemaIndexExtensionDescriptor(
        table_name="attachment_refs",
        index_name="idx_attachment_refs_provider_meta_provider_id",
        ddl="""
                CREATE INDEX IF NOT EXISTS idx_attachment_refs_provider_meta_provider_id
                ON attachment_refs(json_extract(provider_meta, '$.provider_id'))
                WHERE provider_meta IS NOT NULL
                """,
    ),
    SchemaIndexExtensionDescriptor(
        table_name="attachment_refs",
        index_name="idx_attachment_refs_provider_meta_file_id",
        ddl="""
                CREATE INDEX IF NOT EXISTS idx_attachment_refs_provider_meta_file_id
                ON attachment_refs(json_extract(provider_meta, '$.fileId'))
                WHERE provider_meta IS NOT NULL
                """,
    ),
    SchemaIndexExtensionDescriptor(
        table_name="attachment_refs",
        index_name="idx_attachment_refs_provider_meta_drive_id",
        ddl="""
                CREATE INDEX IF NOT EXISTS idx_attachment_refs_provider_meta_drive_id
                ON attachment_refs(json_extract(provider_meta, '$.driveId'))
                WHERE provider_meta IS NOT NULL
                """,
    ),
    SchemaScriptExtensionDescriptor(_ARTIFACT_OBSERVATION_DDL),
    SchemaScriptExtensionDescriptor(_PUBLICATION_DDL),
    SchemaScriptExtensionDescriptor(_ACTION_EVENT_DDL),
    SchemaColumnExtensionDescriptor(
        table_name="action_events",
        column_name="materializer_version",
        ddl="ALTER TABLE action_events ADD COLUMN materializer_version INTEGER NOT NULL DEFAULT 1",
    ),
    SchemaScriptExtensionDescriptor(_ACTION_FTS_DDL),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="canonical_session_date",
        ddl="ALTER TABLE session_profiles ADD COLUMN canonical_session_date TEXT",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="last_message_at",
        ddl="ALTER TABLE session_profiles ADD COLUMN last_message_at TEXT",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="substantive_count",
        ddl="ALTER TABLE session_profiles ADD COLUMN substantive_count INTEGER NOT NULL DEFAULT 0",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="attachment_count",
        ddl="ALTER TABLE session_profiles ADD COLUMN attachment_count INTEGER NOT NULL DEFAULT 0",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="phase_count",
        ddl="ALTER TABLE session_profiles ADD COLUMN phase_count INTEGER NOT NULL DEFAULT 0",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="engaged_duration_ms",
        ddl="ALTER TABLE session_profiles ADD COLUMN engaged_duration_ms INTEGER NOT NULL DEFAULT 0",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="cost_is_estimated",
        ddl="ALTER TABLE session_profiles ADD COLUMN cost_is_estimated INTEGER NOT NULL DEFAULT 0",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="evidence_payload_json",
        ddl="ALTER TABLE session_profiles ADD COLUMN evidence_payload_json TEXT NOT NULL DEFAULT '{}'",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="inference_payload_json",
        ddl="ALTER TABLE session_profiles ADD COLUMN inference_payload_json TEXT NOT NULL DEFAULT '{}'",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="evidence_search_text",
        ddl="ALTER TABLE session_profiles ADD COLUMN evidence_search_text TEXT NOT NULL DEFAULT ''",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="inference_search_text",
        ddl="ALTER TABLE session_profiles ADD COLUMN inference_search_text TEXT NOT NULL DEFAULT ''",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="enrichment_payload_json",
        ddl="ALTER TABLE session_profiles ADD COLUMN enrichment_payload_json TEXT NOT NULL DEFAULT '{}'",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="enrichment_search_text",
        ddl="ALTER TABLE session_profiles ADD COLUMN enrichment_search_text TEXT NOT NULL DEFAULT ''",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="enrichment_version",
        ddl="ALTER TABLE session_profiles ADD COLUMN enrichment_version INTEGER NOT NULL DEFAULT 1",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="enrichment_family",
        ddl=(
            "ALTER TABLE session_profiles ADD COLUMN enrichment_family "
            "TEXT NOT NULL DEFAULT 'scored_session_enrichment'"
        ),
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="inference_version",
        ddl="ALTER TABLE session_profiles ADD COLUMN inference_version INTEGER NOT NULL DEFAULT 1",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_profiles",
        column_name="inference_family",
        ddl=(
            "ALTER TABLE session_profiles ADD COLUMN inference_family "
            "TEXT NOT NULL DEFAULT 'heuristic_session_semantics'"
        ),
    ),
    SchemaBackfillDescriptor(
        table_name="session_profiles",
        ddl="""
                UPDATE session_profiles
                SET evidence_search_text = search_text
                WHERE TRIM(COALESCE(evidence_search_text, '')) = ''
                """,
    ),
    SchemaBackfillDescriptor(
        table_name="session_profiles",
        ddl="""
                UPDATE session_profiles
                SET inference_search_text = search_text
                WHERE TRIM(COALESCE(inference_search_text, '')) = ''
                """,
    ),
    SchemaBackfillDescriptor(
        table_name="session_profiles",
        ddl="""
                UPDATE session_profiles
                SET enrichment_search_text = inference_search_text
                WHERE TRIM(COALESCE(enrichment_search_text, '')) = ''
                """,
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_work_events",
        column_name="start_time",
        ddl="ALTER TABLE session_work_events ADD COLUMN start_time TEXT",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_work_events",
        column_name="end_time",
        ddl="ALTER TABLE session_work_events ADD COLUMN end_time TEXT",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_work_events",
        column_name="duration_ms",
        ddl="ALTER TABLE session_work_events ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_work_events",
        column_name="canonical_session_date",
        ddl="ALTER TABLE session_work_events ADD COLUMN canonical_session_date TEXT",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_work_events",
        column_name="evidence_payload_json",
        ddl="ALTER TABLE session_work_events ADD COLUMN evidence_payload_json TEXT NOT NULL DEFAULT '{}'",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_work_events",
        column_name="inference_payload_json",
        ddl="ALTER TABLE session_work_events ADD COLUMN inference_payload_json TEXT NOT NULL DEFAULT '{}'",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_work_events",
        column_name="inference_version",
        ddl="ALTER TABLE session_work_events ADD COLUMN inference_version INTEGER NOT NULL DEFAULT 1",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_work_events",
        column_name="inference_family",
        ddl=(
            "ALTER TABLE session_work_events ADD COLUMN inference_family "
            "TEXT NOT NULL DEFAULT 'heuristic_session_semantics'"
        ),
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_phases",
        column_name="confidence",
        ddl="ALTER TABLE session_phases ADD COLUMN confidence REAL NOT NULL DEFAULT 0",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_phases",
        column_name="evidence_reasons_json",
        ddl="ALTER TABLE session_phases ADD COLUMN evidence_reasons_json TEXT NOT NULL DEFAULT '[]'",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_phases",
        column_name="evidence_payload_json",
        ddl="ALTER TABLE session_phases ADD COLUMN evidence_payload_json TEXT NOT NULL DEFAULT '{}'",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_phases",
        column_name="inference_payload_json",
        ddl="ALTER TABLE session_phases ADD COLUMN inference_payload_json TEXT NOT NULL DEFAULT '{}'",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_phases",
        column_name="inference_version",
        ddl="ALTER TABLE session_phases ADD COLUMN inference_version INTEGER NOT NULL DEFAULT 1",
    ),
    SchemaColumnExtensionDescriptor(
        table_name="session_phases",
        column_name="inference_family",
        ddl=(
            "ALTER TABLE session_phases ADD COLUMN inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'"
        ),
    ),
    SchemaScriptExtensionDescriptor(_SESSION_PRODUCT_DDL),
)


def schema_extension_snapshot_tables() -> tuple[str, ...]:
    """Tables that current-schema extension planning must inspect."""
    table_names: dict[str, None] = {}
    for descriptor in _SCHEMA_EXTENSION_DESCRIPTORS:
        for table_name in descriptor.snapshot_tables():
            table_names.setdefault(table_name, None)
    return tuple(table_names)


def schema_extension_snapshot_indexes() -> tuple[str, ...]:
    """Indexes that current-schema extension planning must inspect."""
    index_names: dict[str, None] = {}
    for descriptor in _SCHEMA_EXTENSION_DESCRIPTORS:
        for index_name in descriptor.snapshot_indexes():
            index_names.setdefault(index_name, None)
    return tuple(index_names)


def schema_version_mismatch_message(current_version: int) -> str:
    return (
        f"Database schema version {current_version} is incompatible with expected version {SCHEMA_VERSION}. "
        "Delete the database and re-import: `polylogue reset --database && polylogue run`."
    )


def assert_supported_archive_layout_snapshot(snapshot: SchemaSnapshot) -> None:
    """Reject legacy archive layouts that the current runtime cannot write safely."""
    if not snapshot.has_table("raw_conversations"):
        return

    if "raw_content" in snapshot.columns("raw_conversations"):
        raise DatabaseError(
            "Database uses the legacy inline raw-content layout and is incompatible with the current blob-store "
            "archive format. Move the database aside or run `polylogue reset --database` before re-importing."
        )


def build_current_schema_extension_plan(snapshot: SchemaSnapshot) -> SchemaExtensionPlan:
    """Build the canonical extension sequence for current-version databases."""
    assert_supported_archive_layout_snapshot(snapshot)

    statements: list[str] = []
    scripts: list[str] = []

    for descriptor in _SCHEMA_EXTENSION_DESCRIPTORS:
        statements.extend(descriptor.statements(snapshot))
        scripts.extend(descriptor.scripts(snapshot))

    return SchemaExtensionPlan(statements=tuple(statements), scripts=tuple(scripts))


def decide_schema_bootstrap(snapshot: SchemaSnapshot) -> SchemaBootstrapDecision:
    """Choose the canonical schema bootstrap path for sync and async backends."""
    if snapshot.current_version == 0:
        return SchemaBootstrapDecision(action="create_fresh")

    assert_supported_archive_layout_snapshot(snapshot)

    if snapshot.current_version != SCHEMA_VERSION:
        raise DatabaseError(schema_version_mismatch_message(snapshot.current_version))

    return SchemaBootstrapDecision(
        action="apply_current_extensions",
        extension_plan=build_current_schema_extension_plan(snapshot),
    )


def capture_schema_snapshot(conn: sqlite3.Connection) -> SchemaSnapshot:
    current_version = conn.execute("PRAGMA user_version").fetchone()[0]
    table_columns: dict[str, frozenset[str]] = {}
    index_sql: dict[str, str | None] = {}

    for table_name in schema_extension_snapshot_tables():
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        if row is None:
            continue
        columns = {item[1] for item in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        table_columns[table_name] = frozenset(columns)

    for index_name in schema_extension_snapshot_indexes():
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (index_name,)).fetchone()
        index_sql[index_name] = row[0] if row and isinstance(row[0], str) else None
    return SchemaSnapshot(current_version=current_version, table_columns=table_columns, index_sql=index_sql)


async def capture_schema_snapshot_async(conn: aiosqlite.Connection) -> SchemaSnapshot:
    cursor = await conn.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    current_version = row[0] if row else 0
    table_columns: dict[str, frozenset[str]] = {}
    index_sql: dict[str, str | None] = {}

    for table_name in schema_extension_snapshot_tables():
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        row = await cursor.fetchone()
        if row is None:
            continue
        cursor = await conn.execute(f"PRAGMA table_info({table_name})")
        columns = {item[1] for item in await cursor.fetchall()}
        table_columns[table_name] = frozenset(columns)

    for index_name in schema_extension_snapshot_indexes():
        cursor = await conn.execute("SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (index_name,))
        row = await cursor.fetchone()
        index_sql[index_name] = row[0] if row and isinstance(row[0], str) else None
    return SchemaSnapshot(current_version=current_version, table_columns=table_columns, index_sql=index_sql)


def apply_schema_extension_plan(conn: sqlite3.Connection, plan: SchemaExtensionPlan) -> None:
    for statement in plan.statements:
        conn.execute(statement)
    for script in plan.scripts:
        conn.executescript(script)


async def apply_schema_extension_plan_async(conn: aiosqlite.Connection, plan: SchemaExtensionPlan) -> None:
    for statement in plan.statements:
        await conn.execute(statement)
    for script in plan.scripts:
        await conn.executescript(script)


def ensure_vec0_table(conn: sqlite3.Connection) -> None:
    with suppress(Exception):
        conn.execute("SELECT vec_version()")
        conn.execute(_VEC0_DDL)


async def ensure_vec0_table_async(conn: aiosqlite.Connection) -> None:
    with suppress(Exception):
        await conn.execute("SELECT vec_version()")
        await conn.execute(_VEC0_DDL)


__all__ = [
    "SCHEMA_DDL",
    "SCHEMA_VERSION",
    "SchemaBackfillDescriptor",
    "SchemaBootstrapDecision",
    "SchemaColumnExtensionDescriptor",
    "SchemaExtensionPlan",
    "SchemaIndexExtensionDescriptor",
    "SchemaScriptExtensionDescriptor",
    "SchemaSnapshot",
    "apply_schema_extension_plan",
    "apply_schema_extension_plan_async",
    "assert_supported_archive_layout_snapshot",
    "build_current_schema_extension_plan",
    "capture_schema_snapshot",
    "capture_schema_snapshot_async",
    "decide_schema_bootstrap",
    "ensure_vec0_table",
    "ensure_vec0_table_async",
    "schema_extension_snapshot_indexes",
    "schema_extension_snapshot_tables",
    "schema_version_mismatch_message",
]
