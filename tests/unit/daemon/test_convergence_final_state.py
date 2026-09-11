"""Convergence final-state contract test.

Verifies that after daemon live-ingest completes on a synthetic corpus:
  1. All files succeed — no failures, no silently dropped sessions
  2. Sessions and messages appear in the archive database
  3. Post-ingest convergence stages complete without error
  4. Convergence debt is empty (no pending retry items)

This is a behavioral contract, not a benchmark. It proves that
convergence leads to a consistent final archive state.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_default_convergence_stages
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.session_profile_composition import compose_session_profile_callback
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import LiveWatcher, WatchSource


def _write_claude_code_session(path: Path, session_id: str, n_messages: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        records.append(
            {
                "parentUuid": None if i == 0 else f"msg-{session_id}-{i - 1:03d}",
                "sessionId": session_id,
                "type": role,
                "message": {
                    "role": role,
                    "content": f"Message {i}: {'The quick brown fox. ' * 5}",
                },
                "uuid": f"msg-{session_id}-{i:03d}",
                "timestamp": f"2026-05-16T00:{i // 60:02d}:{i % 60:02d}.000Z",
                "cwd": "/realm/project/polylogue",
                "version": "1.0.6",
                "isSidechain": False,
                "userType": "external",
            }
        )
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class _MinimalPolylogue:
    """Minimal polylogue double sufficient for LiveBatchProcessor."""

    def __init__(self, archive_root: Path, db_path: Path) -> None:
        self.archive_root = archive_root
        self.backend = SimpleNamespace(db_path=db_path)


def test_convergence_produces_consistent_final_archive_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After live ingest of a synthetic corpus, archive state must be consistent.

    Checks:
    - All files succeed (no failures)
    - Sessions and messages are in the archive DB
    - Convergence stages complete without error
    - No convergence debt remains
    """
    corpus_root = tmp_path / "corpus" / "proj"
    db_path = tmp_path / "index.db"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")

    # Generate 3 sessions with 10 messages each.
    sessions = [
        ("aaaa0000-0000-0000-0000-000000000001", 10),
        ("aaaa0000-0000-0000-0000-000000000002", 10),
        ("aaaa0000-0000-0000-0000-000000000003", 10),
    ]
    files: list[Path] = []
    for session_id, n_msgs in sessions:
        p = corpus_root / f"{session_id}.jsonl"
        _write_claude_code_session(p, session_id, n_msgs)
        files.append(p)

    converger = DaemonConverger(stages=make_default_convergence_stages(db_path))
    polylogue = _MinimalPolylogue(tmp_path, db_path)
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="test", root=corpus_root.parent),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-v1",
        converger=converger,
    )

    metrics = asyncio.run(processor.ingest_files(files, emit_event=False))

    # ── Ingest completeness ──────────────────────────────────────────
    assert metrics.failed_file_count == 0, f"Unexpected ingest failures: {metrics.failed_file_count}"
    assert metrics.succeeded_file_count == len(files), (
        f"Expected {len(files)} successes, got {metrics.succeeded_file_count}"
    )

    # ── Archive state correctness ────────────────────────────────────
    # The archive writes the parsed session/message tree into ``index.db``
    # (``sessions`` + ``messages``), not the single-file shape.
    index_db = tmp_path / "index.db"
    with sqlite3.connect(index_db) as conn:
        (session_count,) = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        (msg_count,) = conn.execute("SELECT COUNT(*) FROM messages").fetchone()

    # Each session (3) should produce one archive session row.
    assert session_count >= len(sessions), f"Expected >= {len(sessions)} sessions, got {session_count}"
    # Each session has 10 messages.
    assert msg_count >= len(sessions) * 10, f"Expected >= {len(sessions) * 10} messages, got {msg_count}"

    # ── Convergence debt ─────────────────────────────────────────────
    # Native daemon telemetry (convergence debt) lives in the disposable
    # ``ops.db`` tier.
    ops_db = tmp_path / "ops.db"
    if ops_db.exists():
        with sqlite3.connect(ops_db) as conn:
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='live_convergence_debt'"
            ).fetchone()
            if table_exists:
                (debt_count,) = conn.execute("SELECT COUNT(*) FROM live_convergence_debt").fetchone()
                assert debt_count == 0, f"Expected no convergence debt, found {debt_count} pending items"


@pytest.mark.asyncio
async def test_fresh_configured_source_catch_up_recovers_profiles_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normal watcher route catches up a fresh root before a restarted sweep repairs profiles.

    The first daemon instance uses its configured source root and the real
    catch-up implementation to establish durable session material. A fresh
    shared session-profile composition then receives the normal no-hint scope
    used by periodic maintenance after restart. No manual ingest receipt or
    reindex route participates.

    Anti-vacuity: bypass configured-source catch-up, retain only an in-memory
    changed-session list, or replace the restart sweep with a live-source
    lookup and the persisted session has no recoverable profile.
    """
    archive_root = tmp_path / "archive"
    corpus_root = tmp_path / "configured-source"
    source = corpus_root / "fresh-session.jsonl"
    session_id = "aaaa0000-0000-0000-0000-000000000001"
    _write_claude_code_session(source, session_id, n_messages=2)
    # The production quiet predicate defers files still being written. This
    # synthetic startup input is deliberately old enough to be a catch-up
    # candidate, while the restart sweep receives a controlled later clock.
    os.utime(source, (1.0, 1.0))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")

    polylogue = _MinimalPolylogue(archive_root, archive_root / "index.db")
    watcher = LiveWatcher(
        cast(Any, polylogue),
        (WatchSource(name="configured", root=corpus_root),),
        cursor=CursorStore(archive_root / "ops.db"),
    )
    try:
        await watcher._catch_up([corpus_root])
    finally:
        watcher.stop()

    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (f"claude-code:{session_id}",)).fetchone()
        assert conn.execute("SELECT 1 FROM session_profiles").fetchone() is None

    # This is a new process-equivalent owner: it has no watcher state and can
    # only recover from durable source/index evidence through the no-hint pass.
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    try:
        profiles = compose_session_profile_callback(
            archive_root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 100.0,
        )
        report = await profiles(None)
        assert report.failed == 0
        assert report.pending == 0
        assert report.cursor.position("session_profile").swept
    finally:
        compute.shutdown(wait=True)
        assert await coordinator.shutdown(timeout=1.0)

    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute(
            "SELECT 1 FROM session_profiles WHERE session_id = ?", (f"claude-code:{session_id}",)
        ).fetchone()
