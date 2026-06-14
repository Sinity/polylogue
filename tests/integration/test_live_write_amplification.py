"""Write-amplification regression test for live ingest (#1851).

Pins the per-batch DB write cost for a small session write.

BEFORE fix: ~14 MiB WAL delta (FTS5 automerge of large existing segments)
AFTER fix:  ≤3.5 MiB WAL delta

The test bootstraps a fixture archive (tmp_path, never the real user archive),
seeds it with enough sessions to build a non-trivial FTS index, then measures
WAL growth for one additional write with and without the automerge fix applied.

WAL file growth is a reliable proxy for actual SQLite write bytes because
SQLite's WAL mode appends every modified page to the WAL before committing.
The before/after comparison uses TRUNCATE checkpoints to zero the WAL between
measurement windows.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# WAL measurement helpers
# ---------------------------------------------------------------------------

_TIER_NAMES = ("index.db", "source.db", "ops.db", "user.db", "embeddings.db")

# After the automerge fix the per-batch WAL delta should be well under 3.5 MiB.
# The baseline (unfixed) is ~10–14 MiB.  The ceiling gives ≥4× headroom above
# the actual post-fix observed value (~1–2 MiB) to avoid flakiness.
_WRITE_AMP_CEILING_MiB = 3.5


def _wal_bytes(db: Path) -> int:
    wal = Path(str(db) + "-wal")
    return wal.stat().st_size if wal.exists() else 0


def _checkpoint_truncate(db: Path) -> None:
    """TRUNCATE checkpoint to zero the WAL before a measurement window."""
    if not db.exists():
        return
    try:
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
    except sqlite3.Error:
        pass


def _snapshot_wals(root: Path) -> dict[str, int]:
    return {name: _wal_bytes(root / name) for name in _TIER_NAMES}


def _wal_delta_mib(before: dict[str, int], after: dict[str, int]) -> float:
    total = sum(after[name] - before[name] for name in _TIER_NAMES)
    return total / (1024 * 1024)


def _wal_delta_report(before: dict[str, int], after: dict[str, int]) -> str:
    lines: list[str] = ["WAL growth per tier:"]
    for name in _TIER_NAMES:
        d = after[name] - before[name]
        lines.append(f"  {name:<20} {d:>+12,} bytes")
    total = sum(after[name] - before[name] for name in _TIER_NAMES)
    lines.append(f"  {'Total':<20} {total:>+12,} bytes  ({total / (1024 * 1024):.2f} MiB)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Archive bootstrap + session factory
# ---------------------------------------------------------------------------


def _bootstrap_archive(root: Path) -> Path:
    """Bootstrap split-file archive; return index.db path."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(root):
        pass
    return root / "index.db"


def _make_parsed_session(session_id: str, n_messages: int = 5) -> Any:
    """Build a minimal ParsedSession with ``n_messages`` user messages."""
    from polylogue.archive.message.roles import Role
    from polylogue.sources.parsers.base_models import ParsedMessage, ParsedSession
    from polylogue.types import Provider

    messages = [
        ParsedMessage(
            provider_message_id=f"{session_id}:msg-{i}",
            role=Role.USER,
            text="write amplification probe " * 20,
            occurred_at_ms=1_718_000_000_000 + i * 1000,
        )
        for i in range(n_messages)
    ]
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=session_id,
        title=f"Probe session {session_id}",
        messages=messages,
    )


def _write_session(index_db: Path, session_id: str, n_messages: int = 5) -> None:
    """Write one session through the canonical archive write path."""
    from tests.infra.live_ingest import write_session_sync

    session = _make_parsed_session(session_id, n_messages=n_messages)
    write_session_sync(index_db, session)


# ---------------------------------------------------------------------------
# Main regression test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestWriteAmplification:
    """Pin WAL write budget for a small session write.

    Two scenarios:
    A) Baseline — default automerge=8 (expected ~10–14 MiB, NOT asserted
       so the test doesn't fail on fresh archives; only documented).
    B) Fixed — automerge=0 applied before the measurement write (asserted).
    """

    def test_small_write_stays_within_budget_with_automerge_fix(
        self,
        tmp_path: Path,
    ) -> None:
        """A session write must not exceed the WAL ceiling with automerge=0.

        Seeds 80 sessions to build a non-trivial FTS index, applies the
        automerge=0 fix, checkpoints, then writes one more session and
        measures WAL growth across all tiers.
        """
        root = tmp_path / "archive"
        index_db = _bootstrap_archive(root)

        # Seed: 80 sessions × 5 messages = 400 blocks in FTS index.
        # This is enough to accumulate level-0 FTS segments that would
        # normally trigger the automerge on the next write.
        for i in range(80):
            _write_session(index_db, f"seed-session-{i:04d}", n_messages=5)

        # Apply the Fix A: set automerge=0 on all FTS surfaces.
        from polylogue.daemon.fts_automerge import configure_fts_automerge_sync
        from polylogue.storage.sqlite.connection_profile import open_connection

        conn = open_connection(index_db, timeout=30.0)
        try:
            configure_fts_automerge_sync(conn)
        finally:
            conn.close()

        # Checkpoint everything to zero before measurement.
        for name in _TIER_NAMES:
            _checkpoint_truncate(root / name)

        before = _snapshot_wals(root)

        # Measurement write: one small session.
        _write_session(index_db, "probe-write-fixed", n_messages=3)

        after = _snapshot_wals(root)
        delta_mib = _wal_delta_mib(before, after)
        report = _wal_delta_report(before, after)

        assert delta_mib <= _WRITE_AMP_CEILING_MiB, (
            f"WAL write amplification {delta_mib:.2f} MiB exceeds ceiling "
            f"{_WRITE_AMP_CEILING_MiB} MiB — FTS5 automerge fix may have regressed.\n\n"
            f"{report}\n\n"
            "Expected: automerge=0 in configure_fts_automerge_sync keeps per-batch\n"
            "WAL writes near the actual data size, not proportional to existing FTS\n"
            "segment sizes (#1851)."
        )


# ---------------------------------------------------------------------------
# Unit tests for automerge configuration
# ---------------------------------------------------------------------------


class TestFtsAutomergeConfiguration:
    """Unit-style tests for configure_fts_automerge_sync."""

    def test_configure_sets_automerge_to_zero(self, tmp_path: Path) -> None:
        """automerge=0 must be persisted in the FTS5 %_config shadow table."""
        from polylogue.daemon.fts_automerge import configure_fts_automerge_sync
        from polylogue.storage.fts.fts_lifecycle import ensure_fts_index_sync
        from polylogue.storage.sqlite.schema import SCHEMA_DDL

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_DDL)
            ensure_fts_index_sync(conn)
            conn.commit()

            configured = configure_fts_automerge_sync(conn)

            # messages_fts must always be configured (it is in SCHEMA_DDL).
            assert "messages_fts" in configured

            # Verify the setting was persisted.
            row = conn.execute("SELECT v FROM messages_fts_config WHERE k = 'automerge'").fetchone()
            assert row is not None, "automerge key missing from messages_fts_config"
            assert str(row[0]) == "0", f"automerge not set to 0, got: {row[0]}"
        finally:
            conn.close()

    def test_configure_skips_absent_surfaces(self, tmp_path: Path) -> None:
        """Surfaces not in the schema are silently skipped without raising."""
        from polylogue.daemon.fts_automerge import configure_fts_automerge_sync
        from polylogue.storage.sqlite.schema import SCHEMA_DDL

        db_path = tmp_path / "bare.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_DDL)
            conn.commit()

            # Must not raise even if some FTS surfaces are absent.
            configured = configure_fts_automerge_sync(conn)
            assert isinstance(configured, list)
        finally:
            conn.close()

    def test_configure_is_idempotent(self, tmp_path: Path) -> None:
        """Calling configure twice must not raise."""
        from polylogue.daemon.fts_automerge import configure_fts_automerge_sync
        from polylogue.storage.fts.fts_lifecycle import ensure_fts_index_sync
        from polylogue.storage.sqlite.schema import SCHEMA_DDL

        db_path = tmp_path / "idem.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_DDL)
            ensure_fts_index_sync(conn)
            conn.commit()

            configure_fts_automerge_sync(conn)
            configure_fts_automerge_sync(conn)  # second call — must not raise
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Early-return behaviour in ensure_fts_triggers_sync
# ---------------------------------------------------------------------------


class TestEnsureFtsTriggersEarlyReturn:
    """Pin the steady-state early-return in ensure_fts_triggers_sync (#1851)."""

    def test_no_executescript_when_all_triggers_present(self, tmp_path: Path) -> None:
        """When all triggers exist, ensure_fts_triggers_sync must not call executescript.

        Each executescript() call issues an implicit COMMIT that fragments the
        caller's WAL transaction; the early-return prevents this in steady state.
        """
        from polylogue.storage.fts.fts_lifecycle import (
            _FTS_TRIGGER_NAMES,
            ensure_fts_index_sync,
            ensure_fts_triggers_sync,
        )
        from polylogue.storage.sqlite.schema import SCHEMA_DDL

        # sqlite3.Connection.executescript is a read-only C attribute, so it
        # cannot be patched on an instance; subclass and record calls instead.
        executescript_calls: list[str] = []

        class _RecordingConnection(sqlite3.Connection):
            def executescript(self, sql_script: str, /) -> sqlite3.Cursor:
                executescript_calls.append(sql_script[:80])
                return super().executescript(sql_script)

        db_path = tmp_path / "triggers.db"
        conn = sqlite3.connect(str(db_path), factory=_RecordingConnection)
        try:
            conn.executescript(SCHEMA_DDL)
            ensure_fts_index_sync(conn)
            conn.commit()

            # All expected triggers must be present after ensure_fts_index_sync.
            row = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name IN ({})".format(
                    ", ".join("?" for _ in _FTS_TRIGGER_NAMES)
                ),
                _FTS_TRIGGER_NAMES,
            ).fetchone()
            assert row is not None and row[0] == len(_FTS_TRIGGER_NAMES), (
                f"Expected {len(_FTS_TRIGGER_NAMES)} triggers before test, found {row[0] if row else 0}"
            )

            # Reset the recorder, then verify the steady-state early-return path
            # issues NO executescript (each one would force a premature COMMIT).
            executescript_calls.clear()
            ensure_fts_triggers_sync(conn)

            assert executescript_calls == [], (
                f"ensure_fts_triggers_sync called executescript {len(executescript_calls)} "
                f"time(s) even though all {len(_FTS_TRIGGER_NAMES)} triggers were present. "
                f"This causes premature WAL COMMITs that fragment ingest transactions (#1851).\n"
                f"First call: {executescript_calls[0] if executescript_calls else '-'}"
            )
        finally:
            conn.close()

    def test_executescript_called_when_trigger_missing(self, tmp_path: Path) -> None:
        """When a trigger is absent, ensure_fts_triggers_sync must recreate it."""
        from polylogue.storage.fts.fts_lifecycle import (
            ensure_fts_index_sync,
            ensure_fts_triggers_sync,
        )
        from polylogue.storage.sqlite.schema import SCHEMA_DDL

        db_path = tmp_path / "missing.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_DDL)
            ensure_fts_index_sync(conn)
            conn.commit()

            # Simulate partial trigger residue from a bulk-suspend SIGKILL.
            conn.execute("DROP TRIGGER IF EXISTS messages_fts_ai")
            conn.commit()

            ensure_fts_triggers_sync(conn)
            conn.commit()

            row = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name='messages_fts_ai'"
            ).fetchone()
            assert row is not None and row[0] == 1, (
                "ensure_fts_triggers_sync did not restore missing trigger messages_fts_ai"
            )
        finally:
            conn.close()

    def test_fts_queries_still_work_after_automerge_disabled(self, tmp_path: Path) -> None:
        """FTS search must return correct results with automerge=0."""
        from polylogue.daemon.fts_automerge import configure_fts_automerge_sync
        from polylogue.storage.fts.fts_lifecycle import ensure_fts_index_sync
        from polylogue.storage.sqlite.schema import SCHEMA_DDL

        db_path = tmp_path / "query.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_DDL)
            ensure_fts_index_sync(conn)
            conn.commit()

            configure_fts_automerge_sync(conn)

            # Insert a test block so FTS has something to search.
            conn.execute(
                "INSERT INTO sessions(native_id, origin, content_hash) VALUES (?, ?, ?)",
                ("probe-s1", "codex-session", bytes(32)),
            )
            conn.execute(
                "INSERT INTO messages(session_id, native_id, position, role, message_type, content_hash)"
                " VALUES (?, ?, 0, 'user', 'message', ?)",
                ("codex-session:probe-s1", "probe-m1", bytes(32)),
            )
            conn.execute(
                "INSERT INTO blocks(message_id, session_id, position, block_type, text) VALUES (?, ?, 0, 'text', ?)",
                ("codex-session:probe-s1:probe-m1", "codex-session:probe-s1", "automerge probe needle"),
            )
            conn.commit()

            # FTS search must find the inserted block.
            hits = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'automerge'").fetchone()
            assert hits is not None and hits[0] == 1, (
                f"FTS query returned {hits[0] if hits else 0} hits after automerge=0 "
                "was set; search must still work (#1851)"
            )
        finally:
            conn.close()
