"""Shared schema bootstrap planning for sync and async SQLite backends."""

from __future__ import annotations

import re
import sqlite3
from contextlib import suppress
from dataclasses import dataclass

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


def _normalize_sql(sql: str) -> str:
    normalized = " ".join(sql.replace(";", " ").split())
    normalized = re.sub(r"\bIF\s+NOT\s+EXISTS\b", "", normalized, flags=re.IGNORECASE)
    return " ".join(normalized.split())


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

    if snapshot.has_table("raw_conversations") and "blob_size" not in snapshot.columns("raw_conversations"):
        statements.append("ALTER TABLE raw_conversations ADD COLUMN blob_size INTEGER NOT NULL DEFAULT 0")

    if snapshot.has_table("raw_conversations"):
        existing_index_sql = snapshot.sql_for_index("idx_raw_conv_source_mtime")
        if existing_index_sql is None or _normalize_sql(existing_index_sql) != _normalize_sql(
            _RAW_SOURCE_MTIME_INDEX_SQL
        ):
            if existing_index_sql is not None:
                statements.append("DROP INDEX IF EXISTS idx_raw_conv_source_mtime")
            statements.append(_RAW_SOURCE_MTIME_INDEX_SQL)

    if snapshot.has_table("content_blocks"):
        statements.append(
            """
            CREATE INDEX IF NOT EXISTS idx_content_blocks_tool_use_conversation
            ON content_blocks(conversation_id)
            WHERE type = 'tool_use'
            """
        )

    scripts.extend((_ARTIFACT_OBSERVATION_DDL, _PUBLICATION_DDL, _ACTION_EVENT_DDL))
    if snapshot.has_table("action_events") and "materializer_version" not in snapshot.columns("action_events"):
        statements.append("ALTER TABLE action_events ADD COLUMN materializer_version INTEGER NOT NULL DEFAULT 1")
    scripts.append(_ACTION_FTS_DDL)

    if snapshot.has_table("session_profiles"):
        session_profile_columns = snapshot.columns("session_profiles")
        if "canonical_session_date" not in session_profile_columns:
            statements.append("ALTER TABLE session_profiles ADD COLUMN canonical_session_date TEXT")
        if "last_message_at" not in session_profile_columns:
            statements.append("ALTER TABLE session_profiles ADD COLUMN last_message_at TEXT")
        if "substantive_count" not in session_profile_columns:
            statements.append("ALTER TABLE session_profiles ADD COLUMN substantive_count INTEGER NOT NULL DEFAULT 0")
        if "attachment_count" not in session_profile_columns:
            statements.append("ALTER TABLE session_profiles ADD COLUMN attachment_count INTEGER NOT NULL DEFAULT 0")
        if "phase_count" not in session_profile_columns:
            statements.append("ALTER TABLE session_profiles ADD COLUMN phase_count INTEGER NOT NULL DEFAULT 0")
        if "engaged_duration_ms" not in session_profile_columns:
            statements.append("ALTER TABLE session_profiles ADD COLUMN engaged_duration_ms INTEGER NOT NULL DEFAULT 0")
        if "cost_is_estimated" not in session_profile_columns:
            statements.append("ALTER TABLE session_profiles ADD COLUMN cost_is_estimated INTEGER NOT NULL DEFAULT 0")
        if "evidence_payload_json" not in session_profile_columns:
            statements.append(
                "ALTER TABLE session_profiles ADD COLUMN evidence_payload_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "inference_payload_json" not in session_profile_columns:
            statements.append(
                "ALTER TABLE session_profiles ADD COLUMN inference_payload_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "evidence_search_text" not in session_profile_columns:
            statements.append("ALTER TABLE session_profiles ADD COLUMN evidence_search_text TEXT NOT NULL DEFAULT ''")
        if "inference_search_text" not in session_profile_columns:
            statements.append("ALTER TABLE session_profiles ADD COLUMN inference_search_text TEXT NOT NULL DEFAULT ''")
        if "enrichment_payload_json" not in session_profile_columns:
            statements.append(
                "ALTER TABLE session_profiles ADD COLUMN enrichment_payload_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "enrichment_search_text" not in session_profile_columns:
            statements.append("ALTER TABLE session_profiles ADD COLUMN enrichment_search_text TEXT NOT NULL DEFAULT ''")
        if "enrichment_version" not in session_profile_columns:
            statements.append("ALTER TABLE session_profiles ADD COLUMN enrichment_version INTEGER NOT NULL DEFAULT 1")
        if "enrichment_family" not in session_profile_columns:
            statements.append(
                "ALTER TABLE session_profiles ADD COLUMN enrichment_family TEXT NOT NULL DEFAULT 'scored_session_enrichment'"
            )
        if "inference_version" not in session_profile_columns:
            statements.append("ALTER TABLE session_profiles ADD COLUMN inference_version INTEGER NOT NULL DEFAULT 1")
        if "inference_family" not in session_profile_columns:
            statements.append(
                "ALTER TABLE session_profiles ADD COLUMN inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'"
            )
        statements.extend(
            (
                """
                UPDATE session_profiles
                SET evidence_search_text = search_text
                WHERE TRIM(COALESCE(evidence_search_text, '')) = ''
                """,
                """
                UPDATE session_profiles
                SET inference_search_text = search_text
                WHERE TRIM(COALESCE(inference_search_text, '')) = ''
                """,
                """
                UPDATE session_profiles
                SET enrichment_search_text = inference_search_text
                WHERE TRIM(COALESCE(enrichment_search_text, '')) = ''
                """,
            )
        )

    if snapshot.has_table("session_work_events"):
        session_work_event_columns = snapshot.columns("session_work_events")
        if "start_time" not in session_work_event_columns:
            statements.append("ALTER TABLE session_work_events ADD COLUMN start_time TEXT")
        if "end_time" not in session_work_event_columns:
            statements.append("ALTER TABLE session_work_events ADD COLUMN end_time TEXT")
        if "duration_ms" not in session_work_event_columns:
            statements.append("ALTER TABLE session_work_events ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0")
        if "canonical_session_date" not in session_work_event_columns:
            statements.append("ALTER TABLE session_work_events ADD COLUMN canonical_session_date TEXT")
        if "evidence_payload_json" not in session_work_event_columns:
            statements.append(
                "ALTER TABLE session_work_events ADD COLUMN evidence_payload_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "inference_payload_json" not in session_work_event_columns:
            statements.append(
                "ALTER TABLE session_work_events ADD COLUMN inference_payload_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "inference_version" not in session_work_event_columns:
            statements.append("ALTER TABLE session_work_events ADD COLUMN inference_version INTEGER NOT NULL DEFAULT 1")
        if "inference_family" not in session_work_event_columns:
            statements.append(
                "ALTER TABLE session_work_events ADD COLUMN inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'"
            )

    if snapshot.has_table("session_phases"):
        session_phase_columns = snapshot.columns("session_phases")
        if "confidence" not in session_phase_columns:
            statements.append("ALTER TABLE session_phases ADD COLUMN confidence REAL NOT NULL DEFAULT 0")
        if "evidence_reasons_json" not in session_phase_columns:
            statements.append("ALTER TABLE session_phases ADD COLUMN evidence_reasons_json TEXT NOT NULL DEFAULT '[]'")
        if "evidence_payload_json" not in session_phase_columns:
            statements.append("ALTER TABLE session_phases ADD COLUMN evidence_payload_json TEXT NOT NULL DEFAULT '{}'")
        if "inference_payload_json" not in session_phase_columns:
            statements.append("ALTER TABLE session_phases ADD COLUMN inference_payload_json TEXT NOT NULL DEFAULT '{}'")
        if "inference_version" not in session_phase_columns:
            statements.append("ALTER TABLE session_phases ADD COLUMN inference_version INTEGER NOT NULL DEFAULT 1")
        if "inference_family" not in session_phase_columns:
            statements.append(
                "ALTER TABLE session_phases ADD COLUMN inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'"
            )

    scripts.append(_SESSION_PRODUCT_DDL)
    return SchemaExtensionPlan(statements=tuple(statements), scripts=tuple(scripts))


def capture_schema_snapshot(conn: sqlite3.Connection) -> SchemaSnapshot:
    current_version = conn.execute("PRAGMA user_version").fetchone()[0]
    table_columns: dict[str, frozenset[str]] = {}
    index_sql: dict[str, str | None] = {}

    for table_name in (
        "raw_conversations",
        "content_blocks",
        "action_events",
        "session_profiles",
        "session_work_events",
        "session_phases",
    ):
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        if row is None:
            continue
        columns = {item[1] for item in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        table_columns[table_name] = frozenset(columns)

    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_raw_conv_source_mtime'"
    ).fetchone()
    index_sql["idx_raw_conv_source_mtime"] = row[0] if row and isinstance(row[0], str) else None
    return SchemaSnapshot(current_version=current_version, table_columns=table_columns, index_sql=index_sql)


async def capture_schema_snapshot_async(conn: aiosqlite.Connection) -> SchemaSnapshot:
    cursor = await conn.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    current_version = row[0] if row else 0
    table_columns: dict[str, frozenset[str]] = {}
    index_sql: dict[str, str | None] = {}

    for table_name in (
        "raw_conversations",
        "content_blocks",
        "action_events",
        "session_profiles",
        "session_work_events",
        "session_phases",
    ):
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

    cursor = await conn.execute("SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_raw_conv_source_mtime'")
    row = await cursor.fetchone()
    index_sql["idx_raw_conv_source_mtime"] = row[0] if row and isinstance(row[0], str) else None
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
    "SchemaExtensionPlan",
    "SchemaSnapshot",
    "apply_schema_extension_plan",
    "apply_schema_extension_plan_async",
    "assert_supported_archive_layout_snapshot",
    "build_current_schema_extension_plan",
    "capture_schema_snapshot",
    "capture_schema_snapshot_async",
    "ensure_vec0_table",
    "ensure_vec0_table_async",
    "schema_version_mismatch_message",
]
