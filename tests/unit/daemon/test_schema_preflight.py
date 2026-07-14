"""Tests for archive preflight, degraded mode, and dedup limiter (#1003)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.core.degraded import (
    DegradedReason,
    clear_degraded,
    degraded_reason,
    is_degraded,
    set_degraded,
)
from polylogue.core.errors import DatabaseError, SchemaVersionMismatchError
from polylogue.daemon.health import (
    HealthSeverity,
    HealthTier,
    _check_schema_version_fast,
)
from polylogue.sources.live.dedup import (
    SCHEMA_MISMATCH_DEDUP_WINDOW_S,
    RateLimiter,
    handle_schema_version_mismatch,
    schema_warning_limiter,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.schema import _ensure_schema
from polylogue.storage.sqlite.schema_bootstrap import SCHEMA_VERSION


@pytest.fixture(autouse=True)
def _reset_degraded_state() -> Iterator[None]:
    """Each test starts from a clean process-local degraded flag and limiter."""
    clear_degraded()
    schema_warning_limiter.reset()
    yield
    clear_degraded()
    schema_warning_limiter.reset()


# ---------------------------------------------------------------------------
# Typed exception
# ---------------------------------------------------------------------------


def test_schema_version_mismatch_error_carries_versions(tmp_path: Path) -> None:
    """Both versions are attributes so callers don't have to parse strings."""
    db = tmp_path / "ahead.db"
    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()

    with pytest.raises(SchemaVersionMismatchError) as exc_info:
        _ensure_schema(conn)

    err = exc_info.value
    assert isinstance(err, DatabaseError)  # subclass relationship preserved
    assert err.current_version == SCHEMA_VERSION + 1
    assert err.expected_version == SCHEMA_VERSION
    assert "newer than this Polylogue runtime expects" in str(err)
    conn.close()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def _seed_db_at_version(path: Path, version: int) -> None:
    conn = sqlite3.connect(path)
    try:
        # Create at least one table so the file is a real bootstrapped DB,
        # not just an empty file (which the check tolerates).
        conn.execute("CREATE TABLE meta (k TEXT PRIMARY KEY)")
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()
    finally:
        conn.close()


def test_schema_version_health_ok_when_versions_match(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    db = workspace_env["archive_root"] / "index.db"

    monkeypatch.setattr("polylogue.daemon.health.index_db_path", lambda: db)

    alert = _check_schema_version_fast()

    assert alert.check_name == "schema_version"
    assert alert.tier == HealthTier.FAST
    assert alert.severity == HealthSeverity.OK
    assert "archive tier layout matches runtime" in alert.message


def test_schema_version_health_critical_when_db_ahead(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    db = workspace_env["archive_root"] / "index.db"
    _seed_db_at_version(db, ARCHIVE_TIER_SPECS[ArchiveTier.INDEX].version + 1)

    monkeypatch.setattr("polylogue.daemon.health.index_db_path", lambda: db)

    alert = _check_schema_version_fast()

    assert alert.tier == HealthTier.FAST
    assert alert.severity == HealthSeverity.CRITICAL
    assert "tier user_version mismatch" in alert.message
    assert "index.db" in alert.message


def test_schema_version_health_ok_when_no_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh-install case: bootstrap will create at the right version."""
    monkeypatch.setattr("polylogue.daemon.health.db_path", lambda: tmp_path / "missing.db")

    alert = _check_schema_version_fast()

    assert alert.severity == HealthSeverity.OK


# ---------------------------------------------------------------------------
# Degraded mode flag
# ---------------------------------------------------------------------------


def test_degraded_flag_round_trip() -> None:
    assert not is_degraded()
    assert degraded_reason() is None

    reason = DegradedReason(code="schema_version_mismatch", message="test")
    set_degraded(reason)

    assert is_degraded()
    snapshot = degraded_reason()
    assert snapshot is not None
    assert snapshot.code == "schema_version_mismatch"

    clear_degraded()
    assert not is_degraded()


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


def test_rate_limiter_admits_first_then_suppresses_within_window() -> None:
    fake_now = [0.0]

    def clock() -> float:
        return fake_now[0]

    limiter = RateLimiter(window_s=60.0, clock=clock)

    assert limiter.admit(("source-a", "sig-1")) is True
    # Within window — suppressed.
    fake_now[0] = 30.0
    assert limiter.admit(("source-a", "sig-1")) is False
    fake_now[0] = 59.99
    assert limiter.admit(("source-a", "sig-1")) is False
    # After window — admitted again.
    fake_now[0] = 60.5
    assert limiter.admit(("source-a", "sig-1")) is True


def test_rate_limiter_keys_independent() -> None:
    fake_now = [0.0]
    limiter = RateLimiter(window_s=60.0, clock=lambda: fake_now[0])

    assert limiter.admit(("source-a", "sig-1")) is True
    # Different source — independent admission.
    assert limiter.admit(("source-b", "sig-1")) is True
    # Different signature on same source — independent.
    assert limiter.admit(("source-a", "sig-2")) is True


def test_schema_dedup_window_is_one_minute() -> None:
    """AC item: ≤1 mismatch warning per minute per source."""
    assert SCHEMA_MISMATCH_DEDUP_WINDOW_S == 60.0


# ---------------------------------------------------------------------------
# Live batch behavior under schema mismatch
# ---------------------------------------------------------------------------


class _StubCursor:
    """Minimal cursor double for LiveBatchProcessor.ingest_files."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def begin_ingest_attempt(self, **kwargs: object) -> str:
        return "attempt-stub"

    def update_ingest_attempt(self, *args: object, **kwargs: object) -> None:
        pass

    def finish_ingest_attempt(self, *args: object, **kwargs: object) -> None:
        pass

    def get_records(self, paths: list[Path]) -> dict[Path, None]:
        return {}

    def get_record(self, path: Path) -> None:
        return None


@pytest.mark.asyncio
async def test_ingest_files_short_circuits_when_degraded(tmp_path: Path) -> None:
    """While degraded, ingest_files must not enter the full-parse path."""
    from polylogue.sources.live.batch import LiveBatchProcessor

    set_degraded(DegradedReason(code="schema_version_mismatch", message="test"))

    # Create some fake .jsonl files to simulate inotify events.
    files: list[Path] = []
    for i in range(120):
        f = tmp_path / f"event-{i}.jsonl"
        f.write_bytes(b'{"ok": true}\n')
        files.append(f)

    db_path = tmp_path / "index.db"
    cursor = _StubCursor(db_path)

    class _StubPolylogue:
        archive_root = tmp_path
        backend: object | None = None
        config: object | None = None

    sources_root = type(
        "SourceRoot",
        (),
        {
            "name": "claude-code",
            "root": tmp_path,
        },
    )()

    processor = LiveBatchProcessor(
        _StubPolylogue(),  # type: ignore[arg-type]
        [sources_root],
        cursor=cursor,  # type: ignore[arg-type]
        parser_fingerprint="test-fp",
    )

    metrics = await processor.ingest_files(files)

    # No bytes read, no full parse, no per-file failures (skipped, not failed).
    assert metrics.source_payload_read_bytes == 0
    assert metrics.full_file_count == 0
    assert metrics.succeeded_file_count == 0
    assert metrics.failed_file_count == 0
    assert metrics.skipped_file_count == len(files)


# ---------------------------------------------------------------------------
# Dedup behavior on real schema mismatch
# ---------------------------------------------------------------------------


def test_handle_schema_version_mismatch_dedups_and_marks_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated schema mismatches in the same window log once and pin degraded mode.

    This exercises the path that was previously a generic ``except Exception``:
    every inotify event would log the mismatch and try again on the next event.
    """
    exc = SchemaVersionMismatchError(
        "Database schema version 12 is newer than this Polylogue runtime expects (9).",
        current_version=12,
        expected_version=9,
    )

    # Capture warning calls by intercepting the module-level logger in dedup.
    warnings: list[tuple[str, tuple[object, ...]]] = []

    import polylogue.sources.live.dedup as dedup_module

    real_logger = dedup_module._logger

    class _RecordingLogger:
        def warning(self, message: str, *args: object, **kwargs: object) -> None:
            warnings.append((message, args))

        def __getattr__(self, name: str) -> object:
            return getattr(real_logger, name)

    monkeypatch.setattr(dedup_module, "_logger", _RecordingLogger())

    # First call: warns once and flips degraded.
    handle_schema_version_mismatch("claude-code", exc)
    assert is_degraded()
    reason = degraded_reason()
    assert reason is not None
    assert reason.code == "schema_version_mismatch"
    assert reason.detail == {"current_version": 12, "expected_version": 9}

    # Many subsequent attempts within the dedup window: still only one warning.
    for _ in range(150):
        handle_schema_version_mismatch("claude-code", exc)

    refusal_warnings = [w for w in warnings if "refusing further ingest" in w[0]]
    assert len(refusal_warnings) == 1, warnings
    args = refusal_warnings[0][1]
    # Args include source name, version suffix, and error text.
    assert args[0] == "claude-code"
    assert "db schema" in str(args[1])
    assert "v12" in str(args[1])
    assert "runtime expects v9" in str(args[1])


@pytest.mark.asyncio
async def test_ingest_files_short_circuits_under_burst_after_degraded(
    tmp_path: Path,
) -> None:
    """AC item: with degraded mode set, 100+ events do not enter full-parse.

    Pins the structural fix from #1003: once the daemon has classified the DB
    as schema-version-mismatch, the ingest entry path returns synthetic empty
    metrics rather than retrying every file.
    """
    from polylogue.sources.live.batch import LiveBatchProcessor

    set_degraded(DegradedReason(code="schema_version_mismatch", message="v12 vs v9"))

    files: list[Path] = []
    for i in range(150):
        f = tmp_path / f"event-{i}.jsonl"
        f.write_bytes(b'{"ok": true}\n')
        files.append(f)

    db_path = tmp_path / "index.db"
    db_path.touch()
    db_size_before = db_path.stat().st_size

    cursor = _StubCursor(db_path)

    class _StubPolylogue:
        archive_root = tmp_path
        backend: object | None = None
        config: object | None = None

    sources_root = type("SourceRoot", (), {"name": "claude-code", "root": tmp_path})()
    processor = LiveBatchProcessor(
        _StubPolylogue(),  # type: ignore[arg-type]
        [sources_root],
        cursor=cursor,  # type: ignore[arg-type]
        parser_fingerprint="test-fp",
    )

    # Simulate many bursts of events.
    total_read = 0
    for _ in range(5):
        metrics = await processor.ingest_files(files)
        total_read += metrics.source_payload_read_bytes

    # Zero source-payload reads — the file contents were never opened by the
    # full-parse path because the degraded gate fired first.
    assert total_read == 0
    assert db_path.stat().st_size == db_size_before


@pytest.mark.asyncio
async def test_live_batch_marks_structural_database_error_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A layout error stops future live batches instead of retrying per file."""
    from polylogue.sources.live.batch import LiveBatchProcessor

    f = tmp_path / "event.jsonl"
    f.write_bytes(b'{"ok": true}\n')

    db_path = tmp_path / "index.db"
    db_path.touch()
    cursor = _StubCursor(db_path)

    class _StubPolylogue:
        archive_root = tmp_path
        backend: object | None = None
        config: object | None = None

    sources_root = type("SourceRoot", (), {"name": "claude-code", "root": tmp_path})()
    processor = LiveBatchProcessor(
        _StubPolylogue(),  # type: ignore[arg-type]
        [sources_root],
        cursor=cursor,  # type: ignore[arg-type]
        parser_fingerprint="test-fp",
    )

    async def fail_full_parse(*args: object, **kwargs: object) -> object:
        raise DatabaseError("generated sessions.source_name layout")

    monkeypatch.setattr(processor, "_ingest_full_paths", fail_full_parse)

    first = await processor.ingest_files([f])

    assert first.failed_file_count == 1
    assert first.source_payload_read_bytes == 0
    assert is_degraded()
    reason = degraded_reason()
    assert reason is not None
    assert reason.code == "database_layout_mismatch"

    second = await processor.ingest_files([f])

    assert second.source_payload_read_bytes == 0
    assert second.full_file_count == 0
    assert second.skipped_file_count == 1
