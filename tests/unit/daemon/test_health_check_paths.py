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

Every test records a ``daemon.health.<check_name>`` evidence artifact via
``record_contract_evidence`` with bounded fields:
``severity_before``, ``severity_after``, ``consecutive_failures``. EXPENSIVE-
tier tests are marked ``@pytest.mark.slow`` per acceptance criteria so the
default ``devtools verify`` test selection stays fast.

Synthetic-only: no real archive paths, no user data. Fixtures are scoped to
``workspace_env`` (isolated XDG roots + archive root under ``tmp_path``).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from polylogue.core.json import JSONValue
from polylogue.daemon import health as health_module
from polylogue.daemon.health import (
    HealthAlert,
    HealthSeverity,
    _check_blob_integrity_expensive,
    _check_daemon_liveness_fast,
    _check_db_integrity_expensive,
    _check_disk_space_fast,
    _check_embedding_coverage_expensive,
    _check_fts_readiness_medium,
    _check_insight_freshness_medium,
    _check_raw_failures_medium,
    _check_repeated_stage_failures_medium,
    _check_schema_version_fast,
    _check_source_availability_fast,
    _check_stale_ingest_attempts_medium,
    _check_wal_size_fast,
)
from polylogue.daemon.live_ingest_attempt_models import LiveIngestAttemptSummary
from polylogue.paths import archive_root, db_path
from polylogue.storage.blob_store import BlobVerifyAllResult, BlobVerifyFailure
from tests.infra.contract_evidence import ContractEvidenceRecorder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_failure_counts() -> Iterator[None]:
    """Each test starts with an empty ``_failure_counts`` map."""
    health_module._failure_counts.clear()
    yield
    health_module._failure_counts.clear()


def _record(
    recorder: ContractEvidenceRecorder,
    alert: HealthAlert,
    *,
    severity_before: HealthSeverity,
    severity_after: HealthSeverity,
    facts: dict[str, JSONValue] | None = None,
) -> None:
    payload_facts: dict[str, JSONValue] = {
        "severity_before": severity_before.value,
        "severity_after": severity_after.value,
        "consecutive_failures": alert.consecutive_failures,
        "tier": alert.tier.value,
    }
    if facts:
        payload_facts.update(facts)
    recorder.record(
        f"daemon.health.{alert.check_name}",
        surface="daemon",
        request={"check": alert.check_name},
        result={
            "severity": alert.severity.value,
            "consecutive_failures": alert.consecutive_failures,
        },
        facts=payload_facts,
    )


def _init_messages_db(path: Path, *, fts_rows: int = 0, message_rows: int = 0) -> None:
    """Create a minimal DB with ``messages`` and ``messages_fts``.

    The FTS table is created without ``content=`` linkage so its row count
    can drift independently of ``messages`` — that drift is what the
    ``fts_readiness`` check measures.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(body)")
        for i in range(message_rows):
            conn.execute("INSERT INTO messages(id, body) VALUES (?, ?)", (i + 1, f"body {i}"))
        for i in range(fts_rows):
            conn.execute("INSERT INTO messages_fts(rowid, body) VALUES (?, ?)", (i + 1, f"body {i}"))
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
    record_contract_evidence: ContractEvidenceRecorder,
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

    _record(
        record_contract_evidence,
        good,
        severity_before=HealthSeverity.WARNING,
        severity_after=HealthSeverity.OK,
        facts={"failures_before_recovery": bad2.consecutive_failures},
    )


# ---------------------------------------------------------------------------
# FAST: disk_space
# ---------------------------------------------------------------------------


def test_disk_space_ok(
    workspace_env: dict[str, Path],
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    archive_root().mkdir(parents=True, exist_ok=True)
    alert = _check_disk_space_fast()
    # tmpfs (/dev/shm) typically reports plenty of free space.
    assert alert.severity == HealthSeverity.OK
    assert alert.consecutive_failures == 0
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.OK,
    )


def test_disk_space_critical_when_statvfs_reports_low(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_contract_evidence: ContractEvidenceRecorder,
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
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.CRITICAL,
    )


# ---------------------------------------------------------------------------
# FAST: wal_size
# ---------------------------------------------------------------------------


def test_wal_size_ok_when_no_wal_file(
    workspace_env: dict[str, Path],
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    alert = _check_wal_size_fast()
    assert alert.severity == HealthSeverity.OK
    assert "not present" in alert.message
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.OK,
    )


def test_wal_size_error_when_oversized(
    workspace_env: dict[str, Path],
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    dbf = db_path()
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
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.ERROR,
    )


# ---------------------------------------------------------------------------
# FAST: source_availability
# ---------------------------------------------------------------------------


def test_source_availability_ok(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_contract_evidence: ContractEvidenceRecorder,
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
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.OK,
    )


def test_source_availability_warning_when_missing(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    monkeypatch.setattr(
        "polylogue.sources.live.watcher.default_sources",
        lambda: [_FakeSource(name="claude", root=workspace_env["archive_root"] / "missing")],
    )
    alert = _check_source_availability_fast()
    assert alert.severity == HealthSeverity.WARNING
    assert alert.consecutive_failures == 1
    assert "missing" in alert.message
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.WARNING,
    )


# ---------------------------------------------------------------------------
# FAST: schema_version
# ---------------------------------------------------------------------------


def test_schema_version_ok_when_db_absent(
    workspace_env: dict[str, Path],
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    # Fresh install: no DB yet — check should treat as OK (will bootstrap).
    alert = _check_schema_version_fast()
    assert alert.severity == HealthSeverity.OK
    assert "no database yet" in alert.message
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.OK,
    )


def test_schema_version_critical_on_incompatible_version(
    workspace_env: dict[str, Path],
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    dbf = db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    # Write a DB whose user_version is far beyond runtime SCHEMA_VERSION,
    # ensuring decide_schema_bootstrap classifies it as version_mismatch.
    conn = sqlite3.connect(str(dbf))
    try:
        conn.execute("PRAGMA user_version = 99999")
        conn.commit()
    finally:
        conn.close()

    alert = _check_schema_version_fast()
    assert alert.severity == HealthSeverity.CRITICAL
    assert alert.consecutive_failures == 1
    assert "incompatible" in alert.message.lower()
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.CRITICAL,
    )


# ---------------------------------------------------------------------------
# MEDIUM: fts_readiness
# ---------------------------------------------------------------------------


def test_fts_readiness_ok(
    workspace_env: dict[str, Path],
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    dbf = db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    _init_messages_db(dbf)
    alert = _check_fts_readiness_medium()
    assert alert.severity == HealthSeverity.OK
    assert "up to date" in alert.message
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.OK,
    )


def test_fts_readiness_error_when_large_gap(
    workspace_env: dict[str, Path],
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    dbf = db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    # 100 messages, 0 FTS rows → gap = 100 (100%), well above 10% ERROR threshold.
    _init_messages_db(dbf, message_rows=100, fts_rows=0)
    alert = _check_fts_readiness_medium()
    assert alert.severity == HealthSeverity.ERROR
    assert alert.consecutive_failures == 1
    assert "FTS gap" in alert.message
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.ERROR,
    )


# ---------------------------------------------------------------------------
# MEDIUM: raw_failures (OK + degraded + RECOVERY)
# ---------------------------------------------------------------------------


def test_raw_failures_ok_degraded_and_recovery(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_contract_evidence: ContractEvidenceRecorder,
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

    _record(
        record_contract_evidence,
        good,
        severity_before=HealthSeverity.CRITICAL,
        severity_after=HealthSeverity.OK,
        facts={"failures_before_recovery": bad.consecutive_failures},
    )


# ---------------------------------------------------------------------------
# MEDIUM: stale_ingest_attempts
# ---------------------------------------------------------------------------


def test_stale_ingest_attempts_ok(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    monkeypatch.setattr(
        "polylogue.daemon.status._live_ingest_attempt_summary_info",
        lambda: LiveIngestAttemptSummary(running_count=0, stale_running_count=0),
    )
    alert = _check_stale_ingest_attempts_medium()
    assert alert.severity == HealthSeverity.OK
    assert "no running" in alert.message
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.OK,
    )


def test_stale_ingest_attempts_warning_when_some_stale(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    monkeypatch.setattr(
        "polylogue.daemon.status._live_ingest_attempt_summary_info",
        lambda: LiveIngestAttemptSummary(running_count=5, stale_running_count=2),
    )
    alert = _check_stale_ingest_attempts_medium()
    assert alert.severity == HealthSeverity.WARNING
    assert alert.consecutive_failures == 1
    assert "stale" in alert.message
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.WARNING,
    )


# ---------------------------------------------------------------------------
# MEDIUM: insight_freshness
# ---------------------------------------------------------------------------


def test_insight_freshness_ok_when_fully_profiled(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    monkeypatch.setattr(
        "polylogue.daemon.status._insight_freshness_info",
        lambda: {"total_sessions": 50, "sessions_with_profiles": 50},
    )
    alert = _check_insight_freshness_medium()
    assert alert.severity == HealthSeverity.OK
    assert "profiled" in alert.message
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.OK,
    )


def test_insight_freshness_error_when_large_gap(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    monkeypatch.setattr(
        "polylogue.daemon.status._insight_freshness_info",
        lambda: {"total_sessions": 100, "sessions_with_profiles": 50},
    )
    alert = _check_insight_freshness_medium()
    assert alert.severity == HealthSeverity.ERROR
    assert alert.consecutive_failures == 1
    assert "stalled" in alert.message
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.ERROR,
    )


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
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    dbf = db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    _init_live_ingest_attempt(dbf, total=5, failed=0)
    alert = _check_repeated_stage_failures_medium()
    assert alert.severity == HealthSeverity.OK
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.OK,
    )


def test_repeated_stage_failures_error_when_many_recent_failures(
    workspace_env: dict[str, Path],
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    dbf = db_path()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    _init_live_ingest_attempt(dbf, total=10, failed=5)
    alert = _check_repeated_stage_failures_medium()
    assert alert.severity == HealthSeverity.ERROR
    assert alert.consecutive_failures == 1
    assert "recent attempts failed" in alert.message
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.ERROR,
    )


# ---------------------------------------------------------------------------
# EXPENSIVE: db_integrity (OK + degraded + RECOVERY)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_db_integrity_ok_degraded_and_recovery(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    dbf = db_path()
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

    _record(
        record_contract_evidence,
        good,
        severity_before=HealthSeverity.CRITICAL,
        severity_after=HealthSeverity.OK,
        facts={"failures_before_recovery": bad.consecutive_failures},
    )


# ---------------------------------------------------------------------------
# EXPENSIVE: blob_integrity
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_blob_integrity_ok(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    class _Store:
        def verify_all(self, *, max_failures: int = 5) -> BlobVerifyAllResult:
            return BlobVerifyAllResult(checked=10, checked_bytes=1024, failures=(), truncated=False)

    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: _Store())
    alert = _check_blob_integrity_expensive()
    assert alert.severity == HealthSeverity.OK
    assert "ok" in alert.message
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.OK,
    )


@pytest.mark.slow
def test_blob_integrity_warning_when_failures_present(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    class _Store:
        def verify_all(self, *, max_failures: int = 5) -> BlobVerifyAllResult:
            return BlobVerifyAllResult(
                checked=10,
                checked_bytes=1024,
                failures=(BlobVerifyFailure(hash="abc12345" * 8, reason="hash_mismatch"),),
                truncated=False,
            )

    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: _Store())
    alert = _check_blob_integrity_expensive()
    assert alert.severity == HealthSeverity.WARNING
    assert alert.consecutive_failures == 1
    assert "hash_mismatch" in alert.message
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.WARNING,
    )


# ---------------------------------------------------------------------------
# EXPENSIVE: embedding_coverage
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_embedding_coverage_ok_when_disabled(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    @dataclass(frozen=True)
    class _Cfg:
        embedding_enabled: bool = False
        voyage_api_key: str | None = None

    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda: _Cfg())
    alert = _check_embedding_coverage_expensive()
    assert alert.severity == HealthSeverity.OK
    assert alert.message == "embedding disabled"
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.OK,
    )


@pytest.mark.slow
def test_embedding_coverage_error_when_enabled_with_failures(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    @dataclass(frozen=True)
    class _Cfg:
        embedding_enabled: bool = True
        voyage_api_key: str | None = "vk-test"

    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda: _Cfg())
    monkeypatch.setattr(
        "polylogue.daemon.status._embedding_readiness_info",
        lambda: {"embedding_coverage_percent": 12.5, "embedding_failure_count": 7},
    )
    alert = _check_embedding_coverage_expensive()
    assert alert.severity == HealthSeverity.ERROR
    assert alert.consecutive_failures == 1
    assert "failures" in alert.message
    _record(
        record_contract_evidence,
        alert,
        severity_before=HealthSeverity.OK,
        severity_after=HealthSeverity.ERROR,
    )
