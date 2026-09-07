"""A retained logical export replays into the same derived rows live ingest wrote.

The retained material for a mutable SQLite member is its canonical logical
export, so the reindex route must reproduce from that export exactly what the
live acquire -> parse -> write route produced from the database itself. If the
two disagree, the archive's derived tier is not rebuildable from its durable
evidence, which is the whole contract the split-tier design rests on.

Both declared shapes are covered: a Hermes ``state.db``, whose export becomes
sessions through ``write_parsed_session_to_archive``, and a Codex
``state_5.sqlite``, whose export becomes the ``codex_thread_state`` /
``codex_thread_spawn_edges`` projection and never a session.

Anti-vacuity: retain the page image instead of the export (or drop the
export-aware routing in ``_parse_one`` / ``classify_codex_sqlite_path``) and
the replayed archive comes back empty while the live one is populated.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

_THREAD_ID = "8c1d2e3f-4a5b-4c6d-8e7f-901234567890"
_CHILD_THREAD_ID = "9d2e3f4a-5b6c-4d7e-9f80-123456789012"


def _write_hermes_state_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (19);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT, model_config TEXT,
                parent_session_id TEXT, started_at REAL, ended_at REAL,
                end_reason TEXT, title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT, timestamp REAL NOT NULL, tool_calls TEXT,
                observed INTEGER DEFAULT 0, active INTEGER DEFAULT 1, compacted INTEGER DEFAULT 0
            );
            """
        )
        conn.execute(
            "INSERT INTO sessions (id, source, model_config, started_at, ended_at, end_reason, title) "
            "VALUES ('hermes-session-1', 'cli', '{}', 1.0, 8.0, 'completed', 'A retained conversation')"
        )
        conn.executemany(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            [
                (1, "hermes-session-1", "user", "first user turn", 2.0),
                (2, "hermes-session-1", "assistant", "first assistant turn", 3.0),
            ],
        )
        conn.commit()


def _write_codex_state_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY, title TEXT, cwd TEXT, created_at_ms INTEGER,
                updated_at_ms INTEGER, source TEXT, model TEXT, agent_nickname TEXT,
                agent_role TEXT, archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE thread_spawn_edges (
                parent_thread_id TEXT, child_thread_id TEXT, status TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO threads (id, title, cwd, created_at_ms, updated_at_ms, source, archived) "
            "VALUES (?, 'A curated thread title', '/repo', 1000, 2000, 'cli', 0)",
            (_THREAD_ID,),
        )
        conn.execute(
            "INSERT INTO thread_spawn_edges VALUES (?, ?, 'closed')",
            (_THREAD_ID, _CHILD_THREAD_ID),
        )
        conn.commit()


def _derived_rows(index_db: Path, statements: dict[str, str]) -> dict[str, list[tuple[Any, ...]]]:
    with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True) as conn:
        return {name: [tuple(row) for row in conn.execute(sql)] for name, sql in statements.items()}


_HERMES_ROWS = {
    "sessions": (
        "SELECT origin, native_id, title, message_count, word_count, content_hash FROM sessions ORDER BY session_id"
    ),
    "messages": (
        "SELECT session_id, position, role, material_origin, word_count, content_hash FROM messages ORDER BY message_id"
    ),
    "blocks": "SELECT message_id, position, block_type, text, content_hash FROM blocks ORDER BY block_id",
}

_CODEX_ROWS = {
    "codex_thread_state": (
        "SELECT thread_id, title, cwd, created_at_ms, updated_at_ms, source, model, "
        "agent_nickname, agent_role, archived FROM codex_thread_state ORDER BY thread_id"
    ),
    "codex_thread_spawn_edges": (
        "SELECT parent_thread_id, child_thread_id, status FROM codex_thread_spawn_edges "
        "ORDER BY parent_thread_id, child_thread_id"
    ),
}


async def _ingest(archive_root: Path, data_root: Path, source: WatchSource, paths: list[Path]) -> None:
    archive = Polylogue(archive_root=archive_root, db_path=data_root / "cursor.db")
    processor = LiveBatchProcessor(
        archive,
        (source,),
        cursor=CursorStore(data_root / "cursor.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        metrics = await processor.ingest_files(paths, emit_event=False)
        assert metrics.failed_file_count == 0
    finally:
        await archive.close()


def _replay_from_source_tier(archive_root: Path) -> None:
    """Discard the derived tier and rebuild it from the retained exports alone."""
    index_db = archive_root / "index.db"
    index_db.unlink()
    for sidecar in archive_root.glob("index.db-*"):
        sidecar.unlink()
    initialize_active_archive_root(archive_root)
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET parsed_at_ms = NULL, parse_error = NULL")
        conn.execute("DELETE FROM raw_membership_census")
        conn.execute("DELETE FROM raw_authority_parser_census")
        conn.commit()
    backfill_historical_revision_evidence(archive_root)


@pytest.mark.asyncio
async def test_hermes_state_export_replays_into_the_same_sessions(workspace_env: dict[str, Path]) -> None:
    archive_root = workspace_env["archive_root"]
    root = workspace_env["data_root"] / "hermes"
    state_db = root / "state.db"
    _write_hermes_state_db(state_db)
    source = WatchSource(name="hermes", root=root, suffixes=(".db", ".sqlite", ".json", ".jsonl"))

    await _ingest(archive_root, workspace_env["data_root"], source, [state_db])
    live = _derived_rows(archive_root / "index.db", _HERMES_ROWS)
    assert live["sessions"], "sanity: live ingest produced a session"
    assert live["messages"], "sanity: live ingest produced messages"

    _replay_from_source_tier(archive_root)
    replayed = _derived_rows(archive_root / "index.db", _HERMES_ROWS)

    assert replayed == live


@pytest.mark.asyncio
async def test_codex_state_export_replays_into_the_same_projection(workspace_env: dict[str, Path]) -> None:
    archive_root = workspace_env["archive_root"]
    root = workspace_env["data_root"] / "codex-state"
    state_db = root / "state_5.sqlite"
    _write_codex_state_db(state_db)
    source = WatchSource(name="codex-state", root=root, suffixes=(".sqlite", ".db"))

    await _ingest(archive_root, workspace_env["data_root"], source, [state_db])
    live = _derived_rows(archive_root / "index.db", _CODEX_ROWS)
    assert live["codex_thread_state"] == [
        (_THREAD_ID, "A curated thread title", "/repo", 1000, 2000, "cli", None, None, None, 0)
    ]
    assert live["codex_thread_spawn_edges"] == [(_THREAD_ID, _CHILD_THREAD_ID, "closed")]

    _replay_from_source_tier(archive_root)
    replayed = _derived_rows(archive_root / "index.db", _CODEX_ROWS)

    assert replayed == live
