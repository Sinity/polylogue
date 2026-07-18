"""Tests for the Hermes verification-coverage report section (fs1.4 Phase 4).

Uses the real live-ingestion path (LiveBatchProcessor over a real
verification_evidence.db, same pattern as
tests/unit/sources/test_hermes_source_freshness_integration.py) to fetch
authentic ``SessionEventRecord`` rows via the same
``get_session_events`` reader the archive's own materializer/rebuild path
uses, then exercises ``correlate_verification_coverage`` -- the pure
aggregator -- against them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.insights.hermes_verification_coverage import correlate_verification_coverage
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers import hermes_verification
from polylogue.storage.runtime.archive.records import SessionEventRecord
from polylogue.storage.sqlite.queries.session_events import get_session_events


async def _fetch_verification_events(archive_root: Path, hermes_session_id: str) -> list[SessionEventRecord]:
    verification_session_id = hermes_verification.observer_session_provider_id(hermes_session_id)
    qualified_session_id = f"hermes-session:{verification_session_id}"
    async with aiosqlite.connect(archive_root / "index.db") as conn:
        conn.row_factory = aiosqlite.Row
        return await get_session_events(conn, qualified_session_id)


_VERIFICATION_DB_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE verification_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    session_id TEXT NOT NULL,
    cwd TEXT NOT NULL,
    root TEXT NOT NULL,
    command TEXT NOT NULL,
    canonical_command TEXT NOT NULL,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    status TEXT NOT NULL,
    exit_code INTEGER NOT NULL,
    output_summary TEXT NOT NULL
);
CREATE TABLE verification_state (
    session_id TEXT NOT NULL,
    root TEXT NOT NULL,
    last_event_id INTEGER,
    last_edit_at TEXT,
    changed_paths_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (session_id, root)
);
INSERT INTO meta(key, value) VALUES ('schema_version', '1');
"""


async def _ingest_verification_db(
    workspace_env: dict[str, Path], rows: list[tuple[str, str, str, str, str, str, str, str, str, int, str]]
) -> Path:
    root = workspace_env["data_root"] / "hermes-home"
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "verification_evidence.db"
    db_path = workspace_env["data_root"] / "verification-coverage.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="hermes", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        with sqlite3.connect(source_path) as conn:
            conn.executescript(_VERIFICATION_DB_SCHEMA)
            conn.executemany(
                "INSERT INTO verification_events "
                "(created_at, session_id, cwd, root, command, canonical_command, kind, scope, status, "
                "exit_code, output_summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        metrics = await processor.ingest_files([source_path], emit_event=False)
        assert metrics.failed_file_count == 0
    finally:
        await archive.close()
    return workspace_env["archive_root"]


@pytest.mark.asyncio
async def test_correlate_verification_coverage_summarizes_passed_and_failed_events(
    workspace_env: dict[str, Path],
) -> None:
    hermes_session_id = "coverage-session-1"
    archive_root = await _ingest_verification_db(
        workspace_env,
        [
            (
                "2026-07-18T00:00:00Z",
                hermes_session_id,
                "/repo",
                "/repo",
                "ruff check .",
                "ruff",
                "lint",
                "full",
                "passed",
                0,
                "ok",
            ),
            (
                "2026-07-18T00:01:00Z",
                hermes_session_id,
                "/repo",
                "/repo",
                "pytest -k foo",
                "pytest",
                "test",
                "targeted",
                "failed",
                1,
                "1 failed",
            ),
        ],
    )
    archive = Polylogue(archive_root=archive_root, db_path=workspace_env["data_root"] / "verification-coverage.db")
    try:
        events = await _fetch_verification_events(archive_root, hermes_session_id)
        assert events

        coverage = correlate_verification_coverage(hermes_session_id, events)

        assert coverage.available is True
        assert len(coverage.events) == 2
        assert [event.status for event in coverage.events] == ["passed", "failed"]
        assert coverage.final_status == "failed"
        assert coverage.events[1].canonical_command == "pytest"
        assert coverage.events[1].exit_code == 1
        assert not coverage.caveats
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_correlate_verification_coverage_surfaces_changed_paths(workspace_env: dict[str, Path]) -> None:
    hermes_session_id = "coverage-session-paths"
    root = workspace_env["data_root"] / "hermes-home"
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "verification_evidence.db"
    db_path = workspace_env["data_root"] / "verification-coverage-paths.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="hermes", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        with sqlite3.connect(source_path) as conn:
            conn.executescript(_VERIFICATION_DB_SCHEMA)
            conn.execute(
                "INSERT INTO verification_events "
                "(created_at, session_id, cwd, root, command, canonical_command, kind, scope, status, "
                "exit_code, output_summary) VALUES "
                "('2026-07-18T00:00:00Z', ?, '/repo', '/repo', 'pytest', 'pytest', 'test', 'targeted', 'passed', 0, 'ok')",
                (hermes_session_id,),
            )
            conn.execute(
                "INSERT INTO verification_state (session_id, root, last_event_id, last_edit_at, changed_paths_json) "
                "VALUES (?, '/repo', 1, '2026-07-18T00:00:01Z', '[\"src/a.py\", \"src/b.py\"]')",
                (hermes_session_id,),
            )
        metrics = await processor.ingest_files([source_path], emit_event=False)
        assert metrics.failed_file_count == 0

        events = await _fetch_verification_events(workspace_env["archive_root"], hermes_session_id)
        coverage = correlate_verification_coverage(hermes_session_id, events)
        assert set(coverage.changed_paths) == {"src/a.py", "src/b.py"}
    finally:
        await archive.close()


def test_correlate_verification_coverage_is_absent_not_fabricated_when_no_session() -> None:
    coverage = correlate_verification_coverage("no-such-session", None)

    assert coverage.available is False
    assert coverage.events == ()
    assert coverage.final_status is None
    assert coverage.caveats
