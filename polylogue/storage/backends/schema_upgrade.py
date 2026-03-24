"""Schema upgrade helpers for the fixed v1 archive schema."""

from __future__ import annotations

import sqlite3
from contextlib import suppress

from polylogue.errors import DatabaseError
from polylogue.logging import get_logger
from polylogue.storage.backends.schema_ddl import (
    _ACTION_EVENT_DDL,
    _ACTION_FTS_DDL,
    _ARTIFACT_OBSERVATION_DDL,
    _MAINTENANCE_RUN_DDL,
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


def apply_current_schema_extensions(conn: sqlite3.Connection) -> None:
    conn.executescript(_ARTIFACT_OBSERVATION_DDL)
    conn.executescript(_PUBLICATION_DDL)
    conn.executescript(_MAINTENANCE_RUN_DDL)
    conn.executescript(_ACTION_EVENT_DDL)
    action_event_columns = _table_columns(conn, "action_events")
    if "materializer_version" not in action_event_columns:
        conn.execute(
            "ALTER TABLE action_events ADD COLUMN materializer_version INTEGER NOT NULL DEFAULT 1"
        )
    conn.executescript(_ACTION_FTS_DDL)

    if _table_exists(conn, "session_profiles"):
        session_profile_columns = _table_columns(conn, "session_profiles")
        if "canonical_session_date" not in session_profile_columns:
            conn.execute("ALTER TABLE session_profiles ADD COLUMN canonical_session_date TEXT")
        if "phase_count" not in session_profile_columns:
            conn.execute(
                "ALTER TABLE session_profiles ADD COLUMN phase_count INTEGER NOT NULL DEFAULT 0"
            )
        if "engaged_duration_ms" not in session_profile_columns:
            conn.execute(
                "ALTER TABLE session_profiles ADD COLUMN engaged_duration_ms INTEGER NOT NULL DEFAULT 0"
            )

    if _table_exists(conn, "session_work_events"):
        session_work_event_columns = _table_columns(conn, "session_work_events")
        if "start_time" not in session_work_event_columns:
            conn.execute("ALTER TABLE session_work_events ADD COLUMN start_time TEXT")
        if "end_time" not in session_work_event_columns:
            conn.execute("ALTER TABLE session_work_events ADD COLUMN end_time TEXT")
        if "duration_ms" not in session_work_event_columns:
            conn.execute(
                "ALTER TABLE session_work_events ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0"
            )
        if "canonical_session_date" not in session_work_event_columns:
            conn.execute("ALTER TABLE session_work_events ADD COLUMN canonical_session_date TEXT")

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

    if current_version == SCHEMA_VERSION:
        apply_current_schema_extensions(conn)
        return

    raise DatabaseError(
        f"Database schema version {current_version} is incompatible with expected version {SCHEMA_VERSION}. "
        "This database was created with a different schema. Recreate the database "
        f"or update its user_version to match v{SCHEMA_VERSION} after verifying the schema."
    )


__all__ = ["apply_current_schema_extensions", "ensure_schema", "ensure_vec0_table"]
