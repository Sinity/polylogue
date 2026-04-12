"""Schema bootstrap helpers for the async SQLite backend."""

from __future__ import annotations

import aiosqlite


async def ensure_schema(conn: aiosqlite.Connection) -> None:
    """Ensure database schema exists and is at the current schema version."""
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

    cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='raw_conversations'")
    raw_table_exists = await cursor.fetchone() is not None
    if raw_table_exists:
        cursor = await conn.execute("PRAGMA table_info(raw_conversations)")
        raw_columns = {row[1] for row in await cursor.fetchall()}
        if "raw_content" in raw_columns:
            from polylogue.errors import DatabaseError

            raise DatabaseError(
                "Database uses the legacy inline raw-content layout and is incompatible with the current blob-store "
                "archive format. Move the database aside or run `polylogue reset --database` before re-importing."
            )
    else:
        raw_columns = set()

    if current_version != SCHEMA_VERSION:
        from polylogue.errors import DatabaseError

        raise DatabaseError(
            f"Database schema version {current_version} is incompatible with expected version {SCHEMA_VERSION}. "
            f"Delete the database file and re-run polylogue to create a fresh v{SCHEMA_VERSION} schema."
        )

    if "blob_size" not in raw_columns:
        await conn.execute("ALTER TABLE raw_conversations ADD COLUMN blob_size INTEGER NOT NULL DEFAULT 0")

    content_blocks_exists = bool(
        await (await conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='content_blocks'")).fetchone()
    )
    if content_blocks_exists:
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_content_blocks_tool_use_conversation
            ON content_blocks(conversation_id)
            WHERE type = 'tool_use'
            """
        )
    await conn.executescript(_ARTIFACT_OBSERVATION_DDL)
    await conn.executescript(_PUBLICATION_DDL)
    await conn.executescript(_ACTION_EVENT_DDL)
    cursor = await conn.execute("PRAGMA table_info(action_events)")
    action_event_columns = {row[1] for row in await cursor.fetchall()}
    if "materializer_version" not in action_event_columns:
        await conn.execute("ALTER TABLE action_events ADD COLUMN materializer_version INTEGER NOT NULL DEFAULT 1")
    await conn.executescript(_ACTION_FTS_DDL)

    session_profiles_exists = bool(
        await (
            await conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'")
        ).fetchone()
    )
    if session_profiles_exists:
        cursor = await conn.execute("PRAGMA table_info(session_profiles)")
        session_profile_columns = {row[1] for row in await cursor.fetchall()}
        if "canonical_session_date" not in session_profile_columns:
            await conn.execute("ALTER TABLE session_profiles ADD COLUMN canonical_session_date TEXT")
        if "last_message_at" not in session_profile_columns:
            await conn.execute("ALTER TABLE session_profiles ADD COLUMN last_message_at TEXT")
        if "substantive_count" not in session_profile_columns:
            await conn.execute("ALTER TABLE session_profiles ADD COLUMN substantive_count INTEGER NOT NULL DEFAULT 0")
        if "attachment_count" not in session_profile_columns:
            await conn.execute("ALTER TABLE session_profiles ADD COLUMN attachment_count INTEGER NOT NULL DEFAULT 0")
        if "phase_count" not in session_profile_columns:
            await conn.execute("ALTER TABLE session_profiles ADD COLUMN phase_count INTEGER NOT NULL DEFAULT 0")
        if "engaged_duration_ms" not in session_profile_columns:
            await conn.execute("ALTER TABLE session_profiles ADD COLUMN engaged_duration_ms INTEGER NOT NULL DEFAULT 0")
        if "cost_is_estimated" not in session_profile_columns:
            await conn.execute("ALTER TABLE session_profiles ADD COLUMN cost_is_estimated INTEGER NOT NULL DEFAULT 0")
        if "evidence_payload_json" not in session_profile_columns:
            await conn.execute(
                "ALTER TABLE session_profiles ADD COLUMN evidence_payload_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "inference_payload_json" not in session_profile_columns:
            await conn.execute(
                "ALTER TABLE session_profiles ADD COLUMN inference_payload_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "evidence_search_text" not in session_profile_columns:
            await conn.execute("ALTER TABLE session_profiles ADD COLUMN evidence_search_text TEXT NOT NULL DEFAULT ''")
        if "inference_search_text" not in session_profile_columns:
            await conn.execute("ALTER TABLE session_profiles ADD COLUMN inference_search_text TEXT NOT NULL DEFAULT ''")
        if "inference_version" not in session_profile_columns:
            await conn.execute("ALTER TABLE session_profiles ADD COLUMN inference_version INTEGER NOT NULL DEFAULT 1")
        if "inference_family" not in session_profile_columns:
            await conn.execute(
                "ALTER TABLE session_profiles ADD COLUMN inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'"
            )

    session_work_events_exists = bool(
        await (
            await conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_work_events'")
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
            await conn.execute("ALTER TABLE session_work_events ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0")
        if "canonical_session_date" not in session_work_event_columns:
            await conn.execute("ALTER TABLE session_work_events ADD COLUMN canonical_session_date TEXT")
        if "evidence_payload_json" not in session_work_event_columns:
            await conn.execute(
                "ALTER TABLE session_work_events ADD COLUMN evidence_payload_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "inference_payload_json" not in session_work_event_columns:
            await conn.execute(
                "ALTER TABLE session_work_events ADD COLUMN inference_payload_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "inference_version" not in session_work_event_columns:
            await conn.execute(
                "ALTER TABLE session_work_events ADD COLUMN inference_version INTEGER NOT NULL DEFAULT 1"
            )
        if "inference_family" not in session_work_event_columns:
            await conn.execute(
                "ALTER TABLE session_work_events ADD COLUMN inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'"
            )

    session_phases_exists = bool(
        await (
            await conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_phases'")
        ).fetchone()
    )
    if session_phases_exists:
        cursor = await conn.execute("PRAGMA table_info(session_phases)")
        session_phase_columns = {row[1] for row in await cursor.fetchall()}
        if "confidence" not in session_phase_columns:
            await conn.execute("ALTER TABLE session_phases ADD COLUMN confidence REAL NOT NULL DEFAULT 0")
        if "evidence_reasons_json" not in session_phase_columns:
            await conn.execute("ALTER TABLE session_phases ADD COLUMN evidence_reasons_json TEXT NOT NULL DEFAULT '[]'")
        if "evidence_payload_json" not in session_phase_columns:
            await conn.execute("ALTER TABLE session_phases ADD COLUMN evidence_payload_json TEXT NOT NULL DEFAULT '{}'")
        if "inference_payload_json" not in session_phase_columns:
            await conn.execute(
                "ALTER TABLE session_phases ADD COLUMN inference_payload_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "inference_version" not in session_phase_columns:
            await conn.execute("ALTER TABLE session_phases ADD COLUMN inference_version INTEGER NOT NULL DEFAULT 1")
        if "inference_family" not in session_phase_columns:
            await conn.execute(
                "ALTER TABLE session_phases ADD COLUMN inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'"
            )

    await conn.executescript(_SESSION_PRODUCT_DDL)
    await conn.commit()


__all__ = ["ensure_schema"]
