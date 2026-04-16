"""Schema upgrade helpers for the fixed archive schema."""

from __future__ import annotations

import re
import sqlite3
from contextlib import suppress

from polylogue.errors import DatabaseError
from polylogue.logging import get_logger
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

logger = get_logger(__name__)


def ensure_vec0_table(conn: sqlite3.Connection) -> None:
    with suppress(Exception):
        conn.execute("SELECT vec_version()")
        conn.execute(_VEC0_DDL)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return bool(
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
    )


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}


def _index_sql(conn: sqlite3.Connection, index_name: str) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,),
    ).fetchone()
    if row is None:
        return None
    sql = row[0]
    return sql if isinstance(sql, str) else None


def _normalize_sql(sql: str) -> str:
    normalized = " ".join(sql.replace(";", " ").split())
    normalized = re.sub(r"\bIF\s+NOT\s+EXISTS\b", "", normalized, flags=re.IGNORECASE)
    return " ".join(normalized.split())


def _ensure_raw_source_mtime_index(conn: sqlite3.Connection) -> None:
    desired_sql = (
        "CREATE INDEX IF NOT EXISTS idx_raw_conv_source_mtime "
        "ON raw_conversations(source_path, file_mtime) "
        "WHERE file_mtime IS NOT NULL"
    )
    existing_sql = _index_sql(conn, "idx_raw_conv_source_mtime")
    if existing_sql is not None and _normalize_sql(existing_sql) == _normalize_sql(desired_sql):
        return
    if existing_sql is not None:
        logger.info("Replacing idx_raw_conv_source_mtime with partial covering definition")
        conn.execute("DROP INDEX IF EXISTS idx_raw_conv_source_mtime")
    conn.execute(desired_sql)


def _ensure_tool_use_conversation_index(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_content_blocks_tool_use_conversation
        ON content_blocks(conversation_id)
        WHERE type = 'tool_use'
        """
    )


def assert_supported_archive_layout(conn: sqlite3.Connection) -> None:
    """Reject legacy archive layouts that the current runtime cannot write safely."""
    if not _table_exists(conn, "raw_conversations"):
        return

    raw_columns = _table_columns(conn, "raw_conversations")
    if "raw_content" in raw_columns:
        raise DatabaseError(
            "Database uses the legacy inline raw-content layout and is incompatible with the current blob-store "
            "archive format. Move the database aside or run `polylogue reset --database` before re-importing."
        )


def apply_current_schema_extensions(conn: sqlite3.Connection) -> None:
    assert_supported_archive_layout(conn)
    if _table_exists(conn, "raw_conversations"):
        raw_columns = _table_columns(conn, "raw_conversations")
        if "blob_size" not in raw_columns:
            conn.execute("ALTER TABLE raw_conversations ADD COLUMN blob_size INTEGER NOT NULL DEFAULT 0")

    # Covering index for mtime-skip queries (avoids full table scan of raw payload storage)
    _ensure_raw_source_mtime_index(conn)
    if _table_exists(conn, "content_blocks"):
        _ensure_tool_use_conversation_index(conn)
    conn.executescript(_ARTIFACT_OBSERVATION_DDL)
    conn.executescript(_PUBLICATION_DDL)
    conn.executescript(_ACTION_EVENT_DDL)
    action_event_columns = _table_columns(conn, "action_events")
    if "materializer_version" not in action_event_columns:
        conn.execute("ALTER TABLE action_events ADD COLUMN materializer_version INTEGER NOT NULL DEFAULT 1")
    conn.executescript(_ACTION_FTS_DDL)

    if _table_exists(conn, "session_profiles"):
        session_profile_columns = _table_columns(conn, "session_profiles")
        if "canonical_session_date" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN canonical_session_date TEXT")
        if "last_message_at" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN last_message_at TEXT")
        if "substantive_count" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN substantive_count INTEGER NOT NULL DEFAULT 0")
        if "attachment_count" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN attachment_count INTEGER NOT NULL DEFAULT 0")
        if "phase_count" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN phase_count INTEGER NOT NULL DEFAULT 0")
        if "engaged_duration_ms" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN engaged_duration_ms INTEGER NOT NULL DEFAULT 0")
        if "cost_is_estimated" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN cost_is_estimated INTEGER NOT NULL DEFAULT 0")
        if "evidence_payload_json" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN evidence_payload_json TEXT NOT NULL DEFAULT '{}'")
        if "inference_payload_json" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN inference_payload_json TEXT NOT NULL DEFAULT '{}'")
        if "evidence_search_text" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN evidence_search_text TEXT NOT NULL DEFAULT ''")
        if "inference_search_text" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN inference_search_text TEXT NOT NULL DEFAULT ''")
        if "enrichment_payload_json" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN enrichment_payload_json TEXT NOT NULL DEFAULT '{}'")
        if "enrichment_search_text" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN enrichment_search_text TEXT NOT NULL DEFAULT ''")
        if "enrichment_version" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN enrichment_version INTEGER NOT NULL DEFAULT 1")
        if "enrichment_family" not in session_profile_columns:
            conn.execute(
                "ALTER TABLE session_profiles ADD COLUMN enrichment_family TEXT NOT NULL DEFAULT 'scored_session_enrichment'"
            )
        if "inference_version" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN inference_version INTEGER NOT NULL DEFAULT 1")
        if "inference_family" not in session_profile_columns:
            conn.execute(
                "ALTER TABLE session_profiles ADD COLUMN inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'"
            )
        conn.execute(
            """
            UPDATE session_profiles
            SET evidence_search_text = search_text
            WHERE TRIM(COALESCE(evidence_search_text, '')) = ''
            """
        )
        conn.execute(
            """
            UPDATE session_profiles
            SET inference_search_text = search_text
            WHERE TRIM(COALESCE(inference_search_text, '')) = ''
            """
        )
        conn.execute(
            """
            UPDATE session_profiles
            SET enrichment_search_text = inference_search_text
            WHERE TRIM(COALESCE(enrichment_search_text, '')) = ''
            """
        )

    if _table_exists(conn, "session_work_events"):
        session_work_event_columns = _table_columns(conn, "session_work_events")
        if "start_time" not in session_work_event_columns:
            conn.execute("ALTER TABLE session_work_events ADD COLUMN start_time TEXT")
        if "end_time" not in session_work_event_columns:
            conn.execute("ALTER TABLE session_work_events ADD COLUMN end_time TEXT")
        if "duration_ms" not in session_work_event_columns:
            conn.execute("ALTER TABLE session_work_events ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0")
        if "canonical_session_date" not in session_work_event_columns:
            conn.execute("ALTER TABLE session_work_events ADD COLUMN canonical_session_date TEXT")
        if "evidence_payload_json" not in session_work_event_columns:
            conn.execute("ALTER TABLE session_work_events ADD COLUMN evidence_payload_json TEXT NOT NULL DEFAULT '{}'")
        if "inference_payload_json" not in session_work_event_columns:
            conn.execute("ALTER TABLE session_work_events ADD COLUMN inference_payload_json TEXT NOT NULL DEFAULT '{}'")
        if "inference_version" not in session_work_event_columns:
            conn.execute("ALTER TABLE session_work_events ADD COLUMN inference_version INTEGER NOT NULL DEFAULT 1")
        if "inference_family" not in session_work_event_columns:
            conn.execute(
                "ALTER TABLE session_work_events ADD COLUMN inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'"
            )

    if _table_exists(conn, "session_phases"):
        session_phase_columns = _table_columns(conn, "session_phases")
        if "confidence" not in session_phase_columns:
            conn.execute("ALTER TABLE session_phases ADD COLUMN confidence REAL NOT NULL DEFAULT 0")
        if "evidence_reasons_json" not in session_phase_columns:
            conn.execute("ALTER TABLE session_phases ADD COLUMN evidence_reasons_json TEXT NOT NULL DEFAULT '[]'")
        if "evidence_payload_json" not in session_phase_columns:
            conn.execute("ALTER TABLE session_phases ADD COLUMN evidence_payload_json TEXT NOT NULL DEFAULT '{}'")
        if "inference_payload_json" not in session_phase_columns:
            conn.execute("ALTER TABLE session_phases ADD COLUMN inference_payload_json TEXT NOT NULL DEFAULT '{}'")
        if "inference_version" not in session_phase_columns:
            conn.execute("ALTER TABLE session_phases ADD COLUMN inference_version INTEGER NOT NULL DEFAULT 1")
        if "inference_family" not in session_phase_columns:
            conn.execute(
                "ALTER TABLE session_phases ADD COLUMN inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'"
            )

    conn.executescript(_SESSION_PRODUCT_DDL)
    ensure_vec0_table(conn)
    conn.commit()


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the database is at the current schema version."""
    current_version = conn.execute("PRAGMA user_version").fetchone()[0]

    if current_version == 0:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_DDL)
        ensure_vec0_table(conn)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        logger.debug("Created fresh schema v%s", SCHEMA_VERSION)
        return

    assert_supported_archive_layout(conn)

    if current_version == SCHEMA_VERSION:
        apply_current_schema_extensions(conn)
        return

    raise DatabaseError(
        f"Database schema version {current_version} is incompatible with expected version {SCHEMA_VERSION}. "
        "Delete the database and re-import: `polylogue reset --database && polylogue run`."
    )


__all__ = ["apply_current_schema_extensions", "assert_supported_archive_layout", "ensure_schema", "ensure_vec0_table"]
