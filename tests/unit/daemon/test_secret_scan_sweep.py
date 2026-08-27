"""Periodic bounded archive-wide secret-candidate sweep (polylogue-layg.1).

Exercises ``run_secret_scan_sweep_once_sync`` -- the sync body the periodic
daemon loop (``periodic_secret_scan_sweep``) schedules on a quiet cadence --
directly against a real archive fixture. Mirrors
``tests/unit/daemon/test_fts_orphan_audit.py``'s shape for the sibling
"missing feeder" sweep.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import AssertionKind
from polylogue.daemon.secret_scan_sweep import (
    SecretScanSweepResult,
    _record_secret_scan_sweep_event,
    periodic_secret_scan_sweep,
    run_secret_scan_sweep_once_sync,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _seed_archive_session(archive_root: Path, *, native_id: str, text: str) -> str:
    archive_root.mkdir(parents=True, exist_ok=True)
    index_db = archive_root / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    conn = sqlite3.connect(index_db)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms) "
            "VALUES (?, 'codex-session', ?, zeroblob(32), 1000, 2000)",
            (native_id, f"Session {native_id}"),
        )
        session_id = conn.execute("SELECT session_id FROM sessions WHERE native_id = ?", (native_id,)).fetchone()[0]
        conn.execute(
            "INSERT INTO messages (session_id, native_id, position, role, content_hash) "
            "VALUES (?, 'm1', 0, 'user', zeroblob(32))",
            (session_id,),
        )
        message_id = conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
        conn.execute(
            "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, 0, 'text', ?)",
            (message_id, session_id, text),
        )
        conn.commit()
    finally:
        conn.close()
    return str(session_id)


class TestRunSecretScanSweepOnceSync:
    def test_missing_archive_is_a_bounded_noop(self, tmp_path: Path) -> None:
        result = run_secret_scan_sweep_once_sync(tmp_path / "does-not-exist")
        assert result == SecretScanSweepResult()

    def test_no_pending_sessions_is_a_real_run_with_nothing_to_do(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        initialize_archive_database(archive_root / "index.db", ArchiveTier.INDEX)

        result = run_secret_scan_sweep_once_sync(archive_root)

        assert result.ran is True
        assert result.sessions_scanned == 0
        assert result.remaining_pending == 0

    def test_finds_and_records_a_planted_secret_candidate(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_archive_session(
            archive_root, native_id="sweep-1", text="AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP"
        )

        result = run_secret_scan_sweep_once_sync(archive_root)

        assert result.ran is True
        assert result.sessions_scanned == 1
        assert result.candidates_found == 1
        assert result.remaining_pending == 0

        user_conn = sqlite3.connect(archive_root / "user.db")
        try:
            row = user_conn.execute(
                "SELECT target_ref FROM assertions WHERE kind = ?", (AssertionKind.SECRET_CANDIDATE.value,)
            ).fetchone()
        finally:
            user_conn.close()
        assert row is not None
        assert row[0].startswith("block:")

        ops_conn = sqlite3.connect(archive_root / "ops.db")
        try:
            covered = ops_conn.execute(
                "SELECT 1 FROM secret_scan_status WHERE session_id = ?", (session_id,)
            ).fetchone()
        finally:
            ops_conn.close()
        assert covered is not None, "coverage row must be committed so a later tick does not re-scan"

    def test_bounded_window_leaves_remaining_pending_for_next_tick(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        for i in range(4):
            _seed_archive_session(archive_root, native_id=f"sweep-page-{i}", text=f"AWS_ACCESS_KEY_ID=AKIA{i:016X}")

        result = run_secret_scan_sweep_once_sync(archive_root, max_sessions=2)

        assert result.sessions_scanned == 2
        assert result.remaining_pending == 2

        # A second tick drains the rest -- proves the bounded window is
        # genuinely incremental, not a one-shot full scan disguised as bounded.
        second = run_secret_scan_sweep_once_sync(archive_root, max_sessions=2)
        assert second.sessions_scanned == 2
        assert second.remaining_pending == 0


def test_failed_sweep_outcome_is_recorded_for_health_readers(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _record_secret_scan_sweep_event(
        archive_root,
        status="failed",
        error=sqlite3.OperationalError("schema drift"),
    )

    with sqlite3.connect(archive_root / "ops.db") as conn:
        row = conn.execute("SELECT stage, status, payload_json FROM daemon_stage_events").fetchone()

    assert row[0] == "maintenance.secret_scan_sweep"
    assert row[1] == "failed"
    assert '"retryable":true' in row[2]
    assert '"error_type":"OperationalError"' in row[2]


class _StopPeriodicLoopError(Exception):
    """Stop the infinite periodic loop after one completed tick."""


@pytest.mark.asyncio
async def test_periodic_sweep_records_partial_failure_for_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"

    class Coordinator:
        async def run_sync(self, *_args: object, **_kwargs: object) -> SecretScanSweepResult:
            return SecretScanSweepResult(ran=True, errors=1, remaining_pending=1)

    sleep_calls = 0

    async def stop_after_one_tick(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise _StopPeriodicLoopError

    monkeypatch.setattr("polylogue.daemon.write_coordinator.daemon_write_coordinator", Coordinator)
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: archive_root)
    monkeypatch.setattr("polylogue.daemon.secret_scan_sweep.asyncio.sleep", stop_after_one_tick)

    with pytest.raises(_StopPeriodicLoopError):
        await periodic_secret_scan_sweep()

    with sqlite3.connect(archive_root / "ops.db") as conn:
        row = conn.execute(
            "SELECT status, payload_json FROM daemon_stage_events WHERE stage = 'maintenance.secret_scan_sweep'"
        ).fetchone()

    assert row is not None
    assert row[0] == "failed"
    assert '"errors":1' in row[1]
    assert '"retryable":true' in row[1]
