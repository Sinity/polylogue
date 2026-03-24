"""Schema bootstrap helpers for the async SQLite backend."""

from __future__ import annotations

import aiosqlite


async def ensure_schema(conn: aiosqlite.Connection) -> None:
    """Ensure database schema exists and is at the current schema version."""
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

    cursor = await conn.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    current_version = row[0] if row else 0

    if current_version == 0:
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.executescript(SCHEMA_DDL)
        try:
            await conn.execute("SELECT vec_version()")
            await conn.execute(_VEC0_DDL)
        except Exception:
            pass
        await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        await conn.commit()
        return

    if current_version != SCHEMA_VERSION:
        from polylogue.errors import DatabaseError

        raise DatabaseError(
            f"Database schema version {current_version} is incompatible with expected version {SCHEMA_VERSION}. "
            f"Delete the database file and re-run polylogue to create a fresh v{SCHEMA_VERSION} schema."
        )

    await conn.executescript(_ARTIFACT_OBSERVATION_DDL)
    await conn.executescript(_PUBLICATION_DDL)
    await conn.executescript(_MAINTENANCE_RUN_DDL)
    await conn.executescript(_ACTION_EVENT_DDL)
    cursor = await conn.execute("PRAGMA table_info(action_events)")
    action_event_columns = {row[1] for row in await cursor.fetchall()}
    if "materializer_version" not in action_event_columns:
        await conn.execute(
            "ALTER TABLE action_events ADD COLUMN materializer_version INTEGER NOT NULL DEFAULT 1"
        )
    await conn.executescript(_ACTION_FTS_DDL)

    session_profiles_exists = bool(
        await (
            await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
            )
        ).fetchone()
    )
    if session_profiles_exists:
        cursor = await conn.execute("PRAGMA table_info(session_profiles)")
        session_profile_columns = {row[1] for row in await cursor.fetchall()}
        if "canonical_session_date" not in session_profile_columns:
            await conn.execute("ALTER TABLE session_profiles ADD COLUMN canonical_session_date TEXT")
        if "phase_count" not in session_profile_columns:
            await conn.execute(
                "ALTER TABLE session_profiles ADD COLUMN phase_count INTEGER NOT NULL DEFAULT 0"
            )
        if "engaged_duration_ms" not in session_profile_columns:
            await conn.execute(
                "ALTER TABLE session_profiles ADD COLUMN engaged_duration_ms INTEGER NOT NULL DEFAULT 0"
            )

    session_work_events_exists = bool(
        await (
            await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_work_events'"
            )
        ).fetchone()
    )
    if session_work_events_exists:
        cursor = await conn.execute("PRAGMA table_info(session_work_events)")
        session_work_event_columns = {row[1] for row in await cursor.fetchall()}
        if "start_time" not in session_work_event_columns:
            await conn.execute("ALTER TABLE session_work_events ADD COLUMN start_time TEXT")
        if "end_time" not in session_work_event_columns:
            await conn.execute("ALTER TABLE session_work_events ADD COLUMN end_time TEXT")
        if "duration_ms" not in session_work_event_columns:
            await conn.execute(
                "ALTER TABLE session_work_events ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0"
            )
        if "canonical_session_date" not in session_work_event_columns:
            await conn.execute("ALTER TABLE session_work_events ADD COLUMN canonical_session_date TEXT")

    await conn.executescript(_SESSION_PRODUCT_DDL)
    await conn.commit()


__all__ = ["ensure_schema"]
