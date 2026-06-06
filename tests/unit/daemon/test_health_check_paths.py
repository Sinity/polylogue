"""Per-tier health-check evidence: OK, degraded, and recovery paths.

Each of the 13 registered checks in ``polylogue/daemon/health.py`` carries:

- an OK-path assertion on a healthy synthetic fixture, and
- at least one representative degraded-path assertion exercising the
  branch that escalates severity past OK.

Three checks (one per tier) additionally pin the recovery transition —
forcing failure, observing ``consecutive_failures`` climb, then forcing
success and confirming the counter resets to zero. Recovery coverage is
asserted on ``daemon_liveness`` (FAST), ``raw_failures`` (MEDIUM), and
``db_integrity`` (EXPENSIVE); together they exercise the three distinct
``_record_failure`` invocation patterns used across the module.

EXPENSIVE-tier tests are marked ``@pytest.mark.slow`` per acceptance
criteria so the default ``devtools verify`` test selection stays fast.

Synthetic-only: no real archive paths, no user data. Fixtures are scoped to
``workspace_env`` (isolated XDG roots + archive root under ``tmp_path``).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from polylogue.daemon import health as health_module
from polylogue.daemon.blob_integrity_alerts import blob_integrity_alerts_from_report
from polylogue.daemon.health import (
    HealthSeverity,
    _check_blob_integrity_expensive,
    _check_daemon_liveness_fast,
    _check_db_integrity_expensive,
    _check_disk_space_fast,
    _check_embedding_coverage_expensive,
    _check_fts_readiness_medium,
    _check_fts_trigger_drift_fast,
    _check_insight_freshness_medium,
    _check_raw_failures_medium,
    _check_repeated_stage_failures_medium,
    _check_schema_version_fast,
    _check_source_availability_fast,
    _check_stale_ingest_attempts_medium,
    _check_wal_size_fast,
)
from polylogue.daemon.live_ingest_attempt_models import LiveIngestAttemptSummary
from polylogue.paths import archive_root, index_db_path
from polylogue.storage.blob_integrity import BlobIntegrityFinding, BlobIntegrityReport
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.ops_write import record_ingest_attempt
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_failure_counts() -> Iterator[None]:
    """Each test starts with an empty ``_failure_counts`` map."""
    health_module._failure_counts.clear()
    yield
    health_module._failure_counts.clear()


def _init_messages_db(path: Path, *, fts_rows: int = 0, message_rows: int = 0) -> None:
    """Create a minimal DB with ``messages`` and ``messages_fts``.

    The FTS table is created without ``content=`` linkage so its row count
    can drift independently of ``messages`` — that drift is what the
    ``fts_readiness`` check measures.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, text TEXT)")
        conn.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(text)")
        conn.execute("CREATE TRIGGER messages_fts_ai AFTER INSERT ON messages BEGIN SELECT 1; END")
        conn.execute("CREATE TRIGGER messages_fts_ad AFTER DELETE ON messages BEGIN SELECT 1; END")
        conn.execute("CREATE TRIGGER messages_fts_au AFTER UPDATE ON messages BEGIN SELECT 1; END")
        for i in range(message_rows):
            conn.execute("INSERT INTO messages(id, text) VALUES (?, ?)", (i + 1, f"body {i}"))
        for i in range(fts_rows):
            conn.execute("INSERT INTO messages_fts(rowid, text) VALUES (?, ?)", (i + 1, f"body {i}"))
        conn.commit()
    finally:
        conn.close()


@dataclass(frozen=True)
class _FakeSource:
    name: str
    root: Path

    def exists(self) -> bool:
        return self.root.exists()


# ---------------------------------------------------------------------------
# FAST: daemon_liveness (OK + degraded + RECOVERY)
# ---------------------------------------------------------------------------


def test_daemon_liveness_ok_and_recovery(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polylogue.daemon.status as status_module

    # Degraded first: liveness probe says no process.
    monkeypatch.setattr(status_module, "_check_daemon_liveness", lambda: False)
    bad1 = _check_daemon_liveness_fast()
    assert bad1.severity == HealthSeverity.WARNING
    assert bad1.consecutive_failures == 1
    bad2 = _check_daemon_liveness_fast()
    assert bad2.consecutive_failures == 2

    # Recovery: probe now reports alive — counter resets.
    monkeypatch.setattr(status_module, "_check_daemon_liveness", lambda: True)
    good = _check_daemon_liveness_fast()
    assert good.severity == HealthSeverity.OK
    assert good.consecutive_failures == 0
    assert "running" in good.message


# ---------------------------------------------------------------------------
# FAST: disk_space
# ---------------------------------------------------------------------------


def test_disk_space_ok(
    workspace_env: dict[str, Path],
) -> None:
    archive_root().mkdir(parents=True, exist_ok=True)
    alert = _check_disk_space_fast()
    # tmpfs (/dev/shm) typically reports plenty of free space.
    assert alert.severity == HealthSeverity.OK
    assert alert.consecutive_failures == 0


def test_disk_space_critical_when_statvfs_reports_low(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root().mkdir(parents=True, exist_ok=True)

    class _Statvfs:
        f_frsize = 4096
        f_bavail = 1  # ~4 KB free — well under 100 MB critical threshold

    monkeypatch.setattr("polylogue.daemon.health.os.statvfs", lambda _path: _Statvfs())
    alert = _check_disk_space_fast()
    assert alert.severity == HealthSeverity.CRITICAL
    assert alert.consecutive_failures == 1
    assert "critically low" in alert.message


# ---------------------------------------------------------------------------
# FAST: wal_size
# ---------------------------------------------------------------------------


def test_wal_size_ok_when_no_wal_file(
    workspace_env: dict[str, Path],
) -> None:
    alert = _check_wal_size_fast()
    assert alert.severity == HealthSeverity.OK
    assert "not present" in alert.message


def test_wal_size_error_when_oversized(
    workspace_env: dict[str, Path],
) -> None:
    dbf = index_db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    wal = dbf.with_suffix(".db-wal")
    # Use sparse write to allocate > 200 MB cheaply on tmpfs.
    with wal.open("wb") as f:
        f.seek(250 * 1024 * 1024)
        f.write(b"\x00")

    alert = _check_wal_size_fast()
    assert alert.severity == HealthSeverity.ERROR
    assert alert.consecutive_failures == 1
    assert "too large" in alert.message


# ---------------------------------------------------------------------------
# FAST: source_availability
# ---------------------------------------------------------------------------


def test_source_availability_ok(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src_dir = workspace_env["archive_root"] / "src"
    src_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "polylogue.sources.live.watcher.default_sources",
        lambda: [_FakeSource(name="claude", root=src_dir)],
    )
    alert = _check_source_availability_fast()
    assert alert.severity == HealthSeverity.OK
    assert "available" in alert.message


def test_source_availability_warning_when_missing(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "polylogue.sources.live.watcher.default_sources",
        lambda: [_FakeSource(name="claude", root=workspace_env["archive_root"] / "missing")],
    )
    alert = _check_source_availability_fast()
    assert alert.severity == HealthSeverity.WARNING
    assert alert.consecutive_failures == 1
    assert "missing" in alert.message


# ---------------------------------------------------------------------------
# FAST: schema_version
# ---------------------------------------------------------------------------


def test_schema_version_ok_when_db_absent(
    workspace_env: dict[str, Path],
) -> None:
    # Fresh install: no DB yet — check should treat as OK (will bootstrap).
    alert = _check_schema_version_fast()
    assert alert.severity == HealthSeverity.OK
    assert "no database yet" in alert.message


def test_schema_version_critical_when_archive_version_differs(
    workspace_env: dict[str, Path],
) -> None:
    dbf = index_db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    # Write a DB whose user_version is far beyond runtime SCHEMA_VERSION.
    conn = sqlite3.connect(str(dbf))
    try:
        conn.execute("PRAGMA user_version = 99999")
        conn.commit()
    finally:
        conn.close()

    alert = _check_schema_version_fast()
    assert alert.severity == HealthSeverity.CRITICAL
    assert alert.consecutive_failures == 1
    assert "schema v99999 is not runtime" in alert.message.lower()


# ---------------------------------------------------------------------------
# MEDIUM: fts_readiness
# ---------------------------------------------------------------------------


def test_fts_readiness_ok(
    workspace_env: dict[str, Path],
) -> None:
    dbf = index_db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    _init_messages_db(dbf)
    alert = _check_fts_readiness_medium()
    assert alert.severity == HealthSeverity.OK
    assert "up to date" in alert.message


def test_fts_readiness_error_when_large_gap(
    workspace_env: dict[str, Path],
) -> None:
    dbf = index_db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    # 100 messages, 0 FTS rows → gap = 100 (100%), well above 10% ERROR threshold.
    _init_messages_db(dbf, message_rows=100, fts_rows=0)
    alert = _check_fts_readiness_medium()
    assert alert.severity == HealthSeverity.ERROR
    assert alert.consecutive_failures == 1
    assert "missing row" in alert.message


def test_fts_readiness_counts_docsize_not_virtual_table(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production status must not count the FTS5 virtual table directly."""
    dbf = index_db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    _init_messages_db(dbf, message_rows=3, fts_rows=3)

    original_connect = sqlite3.connect
    queries: list[str] = []

    class GuardedConnection:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
            normalized = " ".join(sql.split()).lower()
            queries.append(normalized)
            if "count(*) from messages_fts" in normalized and "messages_fts_docsize" not in normalized:
                raise AssertionError("health status must not COUNT(*) the FTS5 virtual table")
            return self._inner.execute(sql, *args, **kwargs)

        def close(self) -> None:
            self._inner.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    def guarded_connect(path: str) -> GuardedConnection:
        return GuardedConnection(original_connect(path))

    monkeypatch.setattr("polylogue.daemon.health.sqlite3.connect", guarded_connect)

    alert = _check_fts_readiness_medium()

    assert alert.severity == HealthSeverity.OK
    assert any("messages_fts_docsize" in query for query in queries)


def test_fts_trigger_drift_checks_archive_blocks_triggers(
    workspace_env: dict[str, Path],
) -> None:
    dbf = index_db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    initialize_archive_database(dbf, ArchiveTier.INDEX)
    with sqlite3.connect(dbf) as conn:
        conn.execute("DROP TRIGGER blocks_fts_ad")
        conn.commit()

    alert = _check_fts_trigger_drift_fast()

    assert alert.severity == HealthSeverity.CRITICAL
    assert "blocks_fts_ad" in alert.message


def test_fts_readiness_does_not_accept_stats_when_messages_drift(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dbf = index_db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    _init_messages_db(dbf, message_rows=999, fts_rows=3)
    conn = sqlite3.connect(str(dbf))
    try:
        conn.execute("CREATE TABLE session_stats (session_id TEXT PRIMARY KEY, message_count INTEGER)")
        conn.execute("INSERT INTO session_stats VALUES ('c1', 3)")
        conn.commit()
    finally:
        conn.close()

    original_connect = sqlite3.connect
    queries: list[str] = []

    class GuardedConnection:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
            normalized = " ".join(sql.split()).lower()
            queries.append(normalized)
            if normalized == "select count(*) from messages":
                raise AssertionError("health status must use session_stats when available")
            return self._inner.execute(sql, *args, **kwargs)

        def close(self) -> None:
            self._inner.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    def guarded_connect(path: str) -> GuardedConnection:
        return GuardedConnection(original_connect(path))

    monkeypatch.setattr("polylogue.daemon.health.sqlite3.connect", guarded_connect)

    alert = _check_fts_readiness_medium()

    assert alert.severity == HealthSeverity.ERROR
    assert "missing row" in alert.message
    assert all("sum(message_count)" not in query for query in queries)


def test_fts_readiness_flags_stale_extra_rows(
    workspace_env: dict[str, Path],
) -> None:
    dbf = index_db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    _init_messages_db(dbf, message_rows=3, fts_rows=5)
    conn = sqlite3.connect(str(dbf))
    try:
        conn.execute("CREATE TABLE session_stats (session_id TEXT PRIMARY KEY, message_count INTEGER)")
        conn.execute("INSERT INTO session_stats VALUES ('c1', 3)")
        conn.commit()
    finally:
        conn.close()

    alert = _check_fts_readiness_medium()

    assert alert.severity == HealthSeverity.ERROR
    assert "stale row" in alert.message


# ---------------------------------------------------------------------------
# MEDIUM: raw_failures (OK + degraded + RECOVERY)
# ---------------------------------------------------------------------------


def test_raw_failures_ok_degraded_and_recovery(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polylogue.daemon.status as status_module

    # Phase 1: degraded — over CRITICAL threshold.
    monkeypatch.setattr(
        status_module,
        "_raw_failure_info",
        lambda: {"parse_failures": 30, "validation_failures": 40, "quarantined": 5},
    )
    bad = _check_raw_failures_medium()
    assert bad.severity == HealthSeverity.CRITICAL
    assert bad.consecutive_failures == 1
    assert "investigation needed" in bad.message

    # Phase 2: recovery — counter resets.
    monkeypatch.setattr(
        status_module,
        "_raw_failure_info",
        lambda: {"parse_failures": 0, "validation_failures": 0, "quarantined": 0},
    )
    good = _check_raw_failures_medium()
    assert good.severity == HealthSeverity.OK
    assert good.consecutive_failures == 0
    assert good.message == "no raw failures"


# ---------------------------------------------------------------------------
# MEDIUM: stale_ingest_attempts
# ---------------------------------------------------------------------------


def test_stale_ingest_attempts_ok(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "polylogue.daemon.status._live_ingest_attempt_summary_info",
        lambda: LiveIngestAttemptSummary(running_count=0, stale_running_count=0),
    )
    alert = _check_stale_ingest_attempts_medium()
    assert alert.severity == HealthSeverity.OK
    assert "no running" in alert.message


def test_stale_ingest_attempts_warning_when_some_stale(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "polylogue.daemon.status._live_ingest_attempt_summary_info",
        lambda: LiveIngestAttemptSummary(running_count=5, stale_running_count=2),
    )
    alert = _check_stale_ingest_attempts_medium()
    assert alert.severity == HealthSeverity.WARNING
    assert alert.consecutive_failures == 1
    assert "stale" in alert.message


# ---------------------------------------------------------------------------
# MEDIUM: insight_freshness
# ---------------------------------------------------------------------------


def test_insight_freshness_ok_when_fully_profiled(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "polylogue.daemon.status._insight_freshness_info",
        lambda: {"total_sessions": 50, "sessions_with_profiles": 50},
    )
    alert = _check_insight_freshness_medium()
    assert alert.severity == HealthSeverity.OK
    assert "profiled" in alert.message


def test_insight_freshness_error_when_large_gap(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "polylogue.daemon.status._insight_freshness_info",
        lambda: {"total_sessions": 100, "sessions_with_profiles": 50},
    )
    alert = _check_insight_freshness_medium()
    assert alert.severity == HealthSeverity.ERROR
    assert alert.consecutive_failures == 1
    assert "stalled" in alert.message


# ---------------------------------------------------------------------------
# MEDIUM: repeated_stage_failures
# ---------------------------------------------------------------------------


def _init_live_ingest_attempt(dbf: Path, *, total: int, failed: int) -> None:
    conn = sqlite3.connect(str(dbf))
    try:
        conn.execute(
            "CREATE TABLE live_ingest_attempt ("
            "attempt_id TEXT PRIMARY KEY,"
            "started_at TEXT NOT NULL,"
            "status TEXT NOT NULL,"
            "phase TEXT,"
            "error TEXT"
            ")"
        )
        for i in range(total):
            status = "failed" if i < failed else "completed"
            error = "boom" if status == "failed" else None
            conn.execute(
                "INSERT INTO live_ingest_attempt(attempt_id, started_at, status, phase, error) VALUES (?, ?, ?, ?, ?)",
                (f"a{i}", f"2026-05-17T00:00:{i:02d}+00:00", status, "convergence", error),
            )
        conn.commit()
    finally:
        conn.close()


def test_repeated_stage_failures_ok(
    workspace_env: dict[str, Path],
) -> None:
    dbf = index_db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    _init_live_ingest_attempt(dbf, total=5, failed=0)
    alert = _check_repeated_stage_failures_medium()
    assert alert.severity == HealthSeverity.OK


def test_repeated_stage_failures_error_when_many_recent_failures(
    workspace_env: dict[str, Path],
) -> None:
    dbf = index_db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    _init_live_ingest_attempt(dbf, total=10, failed=5)
    alert = _check_repeated_stage_failures_medium()
    assert alert.severity == HealthSeverity.ERROR
    assert alert.consecutive_failures == 1
    assert "recent attempts failed" in alert.message


def test_repeated_stage_failures_reads_ops_tier_without_polylogue_db(
    workspace_env: dict[str, Path],
) -> None:
    dbf = index_db_path()
    ops_db = dbf.with_name("ops.db")
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        for i in range(5):
            record_ingest_attempt(
                conn,
                attempt_id=f"failed-{i}",
                status="failed",
                phase="convergence",
                started_at_ms=1_770_000_000_000 + i,
                error_message="boom",
            )

    alert = _check_repeated_stage_failures_medium()

    assert alert.severity == HealthSeverity.ERROR
    assert "5/5 recent attempts failed" in alert.message
    assert "phase=convergence: boom" in alert.message


def test_repeated_stage_failures_prefers_populated_archive_ops(
    workspace_env: dict[str, Path],
) -> None:
    dbf = index_db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    _init_live_ingest_attempt(dbf, total=5, failed=0)
    ops_db = dbf.with_name("ops.db")
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        for i in range(3):
            record_ingest_attempt(
                conn,
                attempt_id=f"failed-{i}",
                status="failed",
                phase="parse",
                started_at_ms=1_770_000_000_000 + i,
                error_message="v1 boom",
            )

    alert = _check_repeated_stage_failures_medium()

    assert alert.severity == HealthSeverity.ERROR
    assert "3/3 recent attempts failed" in alert.message
    assert "v1 boom" in alert.message


# ---------------------------------------------------------------------------
# EXPENSIVE: db_integrity (OK + degraded + RECOVERY)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_db_integrity_ok_degraded_and_recovery(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dbf = index_db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    # Create a valid empty DB so the real PRAGMA integrity_check executes.
    sqlite3.connect(str(dbf)).close()

    # Phase 1: simulate integrity failure by patching sqlite3.connect to
    # return a connection whose integrity_check yields errors.
    real_connect = sqlite3.connect

    class _BadConn:
        def execute(self, sql: str) -> object:
            if "integrity_check" in sql:

                class _Cur:
                    def fetchall(self) -> list[tuple[str]]:
                        return [("corruption found in page 7",)]

                return _Cur()
            raise AssertionError(f"unexpected sql {sql!r}")

        def close(self) -> None: ...

    monkeypatch.setattr("polylogue.daemon.health.sqlite3.connect", lambda *a, **kw: _BadConn())
    bad = _check_db_integrity_expensive()
    assert bad.severity == HealthSeverity.CRITICAL
    assert bad.consecutive_failures == 1
    assert "integrity errors" in bad.message

    # Phase 2: recovery — restore real connect, counter resets.
    monkeypatch.setattr("polylogue.daemon.health.sqlite3.connect", real_connect)
    good = _check_db_integrity_expensive()
    assert good.severity == HealthSeverity.OK
    assert good.consecutive_failures == 0
    assert "ok" in good.message


# ---------------------------------------------------------------------------
# EXPENSIVE: blob_integrity
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_blob_integrity_ok(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "polylogue.storage.blob_integrity.scan_blob_integrity",
        lambda *_args, **_kwargs: BlobIntegrityReport(
            full_scan=False,
            sample_size=100,
            scanned_blobs=10,
            scanned_references=10,
            total_blobs_seen=10,
            total_references_seen=10,
            active_lease_count=0,
            stale_lease_count=0,
            findings=(),
        ),
        raising=False,
    )
    alert = _check_blob_integrity_expensive()[0]
    assert alert.severity == HealthSeverity.OK
    assert "ok" in alert.message


@pytest.mark.slow
def test_blob_integrity_warning_when_failures_present(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "polylogue.storage.blob_integrity.scan_blob_integrity",
        lambda *_args, **_kwargs: BlobIntegrityReport(
            full_scan=False,
            sample_size=100,
            scanned_blobs=10,
            scanned_references=10,
            total_blobs_seen=10,
            total_references_seen=10,
            active_lease_count=0,
            stale_lease_count=0,
            findings=(
                BlobIntegrityFinding(
                    kind="hash_mismatch",
                    severity="critical",
                    count=1,
                    sample=("abc12345" * 8,),
                    suggested_action="restore from backup",
                ),
            ),
        ),
        raising=False,
    )
    alert = _check_blob_integrity_expensive()[0]
    assert alert.severity == HealthSeverity.CRITICAL
    assert alert.consecutive_failures == 1
    assert "hash_mismatch" in alert.message


def test_blob_integrity_alert_renderer_emits_one_alert_per_finding() -> None:
    report = BlobIntegrityReport(
        full_scan=False,
        sample_size=100,
        scanned_blobs=10,
        scanned_references=10,
        total_blobs_seen=10,
        total_references_seen=10,
        active_lease_count=0,
        stale_lease_count=0,
        findings=(
            BlobIntegrityFinding(
                kind="orphan_blobs",
                severity="warning",
                count=2,
                sample=("a" * 64,),
                suggested_action="preview cleanup",
                bytes_total=42,
            ),
        ),
    )

    alerts = blob_integrity_alerts_from_report(report, "2026-05-24T00:00:00+00:00", lambda _name, _ok: 7)

    assert [alert.check_name for alert in alerts] == ["blob_integrity.orphan_blobs"]
    assert alerts[0].severity == HealthSeverity.WARNING
    assert alerts[0].consecutive_failures == 7
    assert "bytes=42" in alerts[0].message


# ---------------------------------------------------------------------------
# EXPENSIVE: embedding_coverage
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_embedding_coverage_ok_when_disabled(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class _Cfg:
        embedding_enabled: bool = False
        voyage_api_key: str | None = None

    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda: _Cfg())
    alert = _check_embedding_coverage_expensive()
    assert alert.severity == HealthSeverity.OK
    assert alert.message == "embedding disabled"


@pytest.mark.slow
def test_embedding_coverage_error_when_enabled_with_failures(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class _Cfg:
        embedding_enabled: bool = True
        voyage_api_key: str | None = "vk-test"

    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda: _Cfg())
    monkeypatch.setattr(
        "polylogue.daemon.health.embedding_readiness_info",
        lambda _db_file: {"embedding_coverage_percent": 12.5, "embedding_failure_count": 7},
    )
    alert = _check_embedding_coverage_expensive()
    assert alert.severity == HealthSeverity.ERROR
    assert alert.consecutive_failures == 1
    assert "failures" in alert.message
