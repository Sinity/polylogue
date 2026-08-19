"""One Hermes snapshot holding many sessions must import as one shared raw.

polylogue-1fijp. ``hermes_profile_raw_id`` (``sources/sqlite_snapshot.py``) is
keyed per SNAPSHOT -- profile directory + source_index + blob_hash -- and
deliberately excludes the session id, because Hermes session ids are only
unique within a profile. One ``state.db`` therefore yields N ParsedSessions
that all carry the SAME acquisition raw id and DIFFERENT native ids.

That is the grouped-capture shape, and the one-shot importer
(``parse_sources_archive``) has to treat it as such. When it instead admitted
each session as its own BASELINE observation against that shared raw id, the
second session collided: the raw row already existed, so
``_assert_existing_raw_observation_identity`` refused it as a substitution
hazard, and ``write_pair`` -- which catches only ``ContentExcisedError`` --
let the exception abort and roll back the whole import batch.

The regression could not surface in unit coverage of
``admit_raw_observation`` itself (a single admission never collides) nor in
the existing single-session Hermes fixtures. It needs a genuinely
multi-session snapshot driven end to end, which is what this does.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

SESSION_IDS = ("hermes-alpha", "hermes-beta", "hermes-gamma")


def _write_multi_session_state_db(path: Path) -> None:
    """Write a minimal but real Hermes state.db carrying three sessions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    with conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version (version) VALUES (7)")
        # Column set matches hermes_state._REQUIRED_SESSION_COLUMNS plus the
        # _HERMES_SIGNATURE_SESSION_COLUMNS the detector requires.
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                model TEXT,
                model_config TEXT,
                source TEXT,
                system_prompt TEXT,
                started_at INTEGER,
                parent_session_id TEXT
            )
            """
        )
        # Likewise _REQUIRED_MESSAGE_COLUMNS + _HERMES_SIGNATURE_MESSAGE_COLUMNS.
        conn.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp INTEGER,
                tool_calls TEXT,
                observed INTEGER,
                active INTEGER,
                compacted INTEGER
            )
            """
        )
        for index, session_id in enumerate(SESSION_IDS):
            conn.execute(
                "INSERT INTO sessions (id, title, model, model_config, source, system_prompt,"
                " started_at, parent_session_id) VALUES (?, ?, ?, '{}', 'cli', NULL, ?, NULL)",
                (session_id, f"session {session_id}", "hermes-test-model", 1_767_000_000 + index),
            )
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp, tool_calls,"
                " observed, active, compacted) VALUES (?, ?, ?, ?, ?, NULL, 0, 1, 0)",
                (index * 2 + 1, session_id, "user", f"question from {session_id}", 1_767_000_000 + index),
            )
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp, tool_calls,"
                " observed, active, compacted) VALUES (?, ?, ?, ?, ?, NULL, 0, 1, 0)",
                (index * 2 + 2, session_id, "assistant", f"answer for {session_id}", 1_767_000_001 + index),
            )
    conn.close()


@pytest.mark.usefixtures("workspace_env")
def test_multi_session_hermes_snapshot_imports_as_one_shared_raw(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    state_db = tmp_path / "hermes-profile" / "state.db"
    _write_multi_session_state_db(state_db)

    asyncio.run(parse_sources_archive(archive_root, [Source(name="hermes", path=state_db)]))

    with sqlite3.connect(archive_root / "index.db") as index_conn:
        session_ids = sorted(str(row[0]) for row in index_conn.execute("SELECT session_id FROM sessions"))
    with sqlite3.connect(archive_root / "source.db") as source_conn:
        raw_ids = [str(row[0]) for row in source_conn.execute("SELECT raw_id FROM raw_sessions")]
        native_ids = [row[0] for row in source_conn.execute("SELECT native_id FROM raw_sessions")]

    # Every session in the snapshot is indexed -- the batch must not abort.
    assert len(session_ids) == len(SESSION_IDS), session_ids
    # One physical snapshot is one acquisition, so exactly one raw row, keyed
    # by the snapshot coordinate rather than by any one session's id.
    assert len(raw_ids) == 1, raw_ids
    assert native_ids == [None], native_ids
