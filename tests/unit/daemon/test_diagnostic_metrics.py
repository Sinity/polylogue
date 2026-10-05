"""Diagnostic delivery loss is visible on the metrics surface."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from polylogue.daemon.metrics import handle_metrics
from polylogue.operations import daemon_metrics as metrics
from polylogue.operations.storage_io_observation import IoPhaseObservation, StorageIoObservation
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _scrape(db: Path) -> str:
    responder = MagicMock()
    handle_metrics(responder, db)
    assert responder._send_text.call_args.args[0] == HTTPStatus.OK
    return cast(str, responder._send_text.call_args.args[1])


def test_metrics_expose_maintained_sink_loss_without_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Omitting the loss series or hiding drops as failures makes this red."""
    monkeypatch.setattr(
        metrics,
        "diagnostic_snapshot",
        lambda: {
            "queued": 3,
            "dropped": 7,
            "failures": 2,
            "delivered": 11,
            "undrained": 1,
            "high_water": 9,
            "backpressure": 8,
            "priority_evictions": 1,
        },
    )
    body = metrics.format_metrics(tmp_path / "missing.db")
    assert 'polylogue_diagnostic_delivery_total{outcome="dropped"} 7' in body
    assert 'polylogue_diagnostic_delivery_total{outcome="failures"} 2' in body
    assert 'polylogue_diagnostic_delivery_total{outcome="delivered"} 11' in body
    assert 'polylogue_diagnostic_delivery_total{outcome="undrained"} 1' in body
    assert "polylogue_diagnostic_queue_depth 3" in body
    assert "polylogue_diagnostic_backpressure_total 8" in body
    assert "polylogue_diagnostic_priority_evictions_total 1" in body


def test_metrics_distinguish_measured_io_phases_from_unavailable_sqlite_internals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An uninstrumentable fsync must not masquerade as a measured zero."""
    monkeypatch.setattr(
        metrics,
        "storage_io_observation",
        lambda: StorageIoObservation(
            samples=(
                IoPhaseObservation("ops", "commit", True, True, 2, 250_000_000),
                IoPhaseObservation("ops", "commit", True, False, 1, 10_000_000),
            ),
            unavailable_phases=("sqlite_file_fsync",),
        ),
    )
    body = metrics.format_metrics(tmp_path / "missing.db")
    assert (
        'polylogue_storage_io_phase_total{inside_writer_lease="true",phase="commit",succeeded="true",tier="ops"} 2'
    ) in body
    assert (
        'polylogue_storage_io_phase_seconds_total{inside_writer_lease="true",phase="commit",'
        'succeeded="true",tier="ops"} 0.25'
    ) in body
    assert 'polylogue_storage_io_phase_observable{phase="sqlite_file_fsync"} 0' in body
    assert 'polylogue_storage_io_phase_observable{phase="commit"} 1' in body


def test_late_archive_failure_retains_process_and_completed_storage_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, message_count INTEGER, raw_id TEXT)"
        )
    monkeypatch.setattr(
        metrics,
        "diagnostic_snapshot",
        lambda: {"queued": 3, "dropped": 7, "failures": 2, "delivered": 11, "undrained": 1},
    )

    def fail_after_prior_groups(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("unreadable /synthetic/private-input SELECT secret FROM rows")

    warnings: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(metrics, "emit", lambda event, **fields: warnings.append((event, fields)))
    monkeypatch.setattr(metrics, "_emit_archive_source_index_link_metrics", fail_after_prior_groups)
    body = _scrape(db)
    assert "polylogue_daemon_uptime_seconds " in body
    assert 'polylogue_diagnostic_delivery_total{outcome="dropped"} 7' in body
    assert 'polylogue_archive_tier_present{tier="index"} 1' in body
    assert 'polylogue_daemon_metrics_collection_available{group="archive_index"} 0' in body
    assert "polylogue_archive_sessions_total" not in body
    assert "/synthetic/private-input" not in body
    assert "SELECT secret" not in body
    assert any("/synthetic/private-input" in str(fields.get("error_detail")) for _, fields in warnings)


def test_outer_format_failure_keeps_process_counters_without_private_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        metrics,
        "diagnostic_snapshot",
        lambda: {"queued": 3, "dropped": 7, "failures": 2, "delivered": 11, "undrained": 1},
    )
    warnings: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(metrics, "emit", lambda event, **fields: warnings.append((event, fields)))
    monkeypatch.setattr(
        metrics,
        "format_metrics",
        lambda _db: (_ for _ in ()).throw(RuntimeError("unreadable /synthetic/private-input")),
    )
    body = _scrape(tmp_path / "index.db")
    assert "polylogue_daemon_uptime_seconds " in body
    assert 'polylogue_diagnostic_delivery_total{outcome="dropped"} 7' in body
    assert 'polylogue_daemon_metrics_collection_error{reason="collector_failed"} 1' in body
    assert "/synthetic/private-input" not in body
    assert any("/synthetic/private-input" in str(fields.get("error_detail")) for _, fields in warnings)


def test_exception_messages_cannot_create_metric_labels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, message_count INTEGER, raw_id TEXT)"
        )
    failure = ["first /synthetic/private-A", "second SELECT private FROM B"]
    monkeypatch.setattr(
        metrics,
        "_emit_archive_source_index_link_metrics",
        lambda *_args, **_kw: (_ for _ in ()).throw(RuntimeError(failure[0])),
    )
    first = _scrape(db)
    failure[0] = failure[1]
    second = _scrape(db)
    assert first == second or {
        line for line in first.splitlines() if line.startswith("polylogue_daemon_metrics_collection_available")
    } == {line for line in second.splitlines() if line.startswith("polylogue_daemon_metrics_collection_available")}
    assert all(marker not in first + second for marker in ("private-A", "SELECT private", "private FROM B"))


def test_collection_availability_series_survives_recovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: failure and recovery must publish the same group series."""
    db = tmp_path / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, message_count INTEGER, raw_id TEXT)"
        )
    fail = True

    def fail_once(*_args: object, **_kwargs: object) -> None:
        if fail:
            raise RuntimeError("synthetic collection failure")

    monkeypatch.setattr(metrics, "_emit_archive_source_index_link_metrics", fail_once)
    failed = _scrape(db)
    fail = False
    recovered = _scrape(db)
    available_prefix = 'polylogue_daemon_metrics_collection_available{group="archive_index"}'
    assert f"{available_prefix} 0" in failed
    assert f"{available_prefix} 1" in recovered
    assert 'polylogue_daemon_metrics_collection_available{group="archive_index",reason=' not in recovered
    assert 'polylogue_daemon_metrics_collection_reason{group="archive_index",reason="collector_failed"} 1' in failed


def test_missing_and_malformed_index_are_unavailable_not_measured_zero(tmp_path: Path) -> None:
    missing = _scrape(tmp_path / "index.db")
    assert 'polylogue_daemon_metrics_collection_available{group="archive_index"} 0' in missing
    assert "polylogue_archive_sessions_total " not in missing
    assert "polylogue_fts_triggers_all_present 0" not in missing
    (tmp_path / "index.db").write_bytes(b"not a sqlite database")
    malformed = _scrape(tmp_path / "index.db")
    assert 'polylogue_daemon_metrics_collection_available{group="archive_index"} 0' in malformed
    assert "polylogue_archive_sessions_total " not in malformed


def test_missing_source_is_reported_without_inventing_source_count(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, message_count INTEGER, raw_id TEXT)"
        )
    body = _scrape(db)
    assert 'polylogue_archive_tier_present{tier="source"} 0' in body
    assert "polylogue_archive_ready 0" in body
    assert 'polylogue_archive_source_index_links_total{source="unknown",state="source_db_missing"} 0' in body


def test_readable_empty_index_retains_measured_zero(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, message_count INTEGER, raw_id TEXT)"
        )
    body = _scrape(db)
    assert 'polylogue_daemon_metrics_collection_available{group="archive_index"} 1' in body
    assert "polylogue_archive_sessions_total 0" in body


def test_ops_debt_without_attempt_ledger_does_not_invent_attempt_zeros(tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.executescript(
            """
            CREATE TABLE convergence_debt (stage TEXT NOT NULL, status TEXT NOT NULL);
            INSERT INTO convergence_debt VALUES ('materialize', 'failed');
            """
        )
    body = _scrape(tmp_path / "index.db")
    assert 'polylogue_convergence_debt_count{stage="materialize",status="failed"} 1' not in body
    assert 'polylogue_daemon_metrics_collection_available{group="ops_attempts"} 0' in body
    assert "polylogue_live_ingest_attempts_total{status=" not in body
    assert "polylogue_live_ingest_attempts_in_flight 0" not in body
    assert "polylogue_stale_cursor_writes_total 0" not in body
    assert "polylogue_live_ingest_storage_route_total 0" not in body


def test_unreadable_ops_tier_is_not_reported_as_missing_schema(tmp_path: Path) -> None:
    """Anti-vacuity: corrupt ops bytes make both ops probes unavailable as unreadable."""
    (tmp_path / "ops.db").write_bytes(b"not a sqlite database")
    body = _scrape(tmp_path / "index.db")
    assert 'polylogue_daemon_metrics_collection_available{group="ops_or_discovery"} 0' in body
    assert 'polylogue_daemon_metrics_collection_reason{group="ops_or_discovery",reason="archive_unreadable"} 1' in body
    assert 'polylogue_daemon_metrics_collection_reason{group="ops_attempts",reason="archive_unreadable"} 1' in body


def test_ops_only_openability_probe_uses_a_closed_query_only_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Restoring the hand-built probe bypasses the query-only owner and fails."""
    from polylogue.storage.sqlite import connection_profile

    ops_db = tmp_path / "ops.db"
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with closing(sqlite3.connect(ops_db)) as conn:
        conn.execute("CREATE TABLE sentinel (value INTEGER)")
        conn.commit()
    opened: list[sqlite3.Connection] = []
    original_open = connection_profile.open_readonly_connection

    def observe_open(path: str | Path, *, validate_schema: bool = True, **kwargs: Any) -> sqlite3.Connection:
        conn = original_open(path, validate_schema=validate_schema, **kwargs)
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("INSERT INTO sentinel VALUES (1)")
        opened.append(conn)
        return conn

    monkeypatch.setattr(connection_profile, "open_readonly_connection", observe_open)
    assert metrics._format_ops_only_metrics([], ops_db) is True
    assert opened
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


def test_ops_only_missing_tier_is_not_created(tmp_path: Path) -> None:
    """Opening an absent ops tier as writable would create it and fail."""
    ops_db = tmp_path / "ops.db"
    assert metrics._format_ops_only_metrics([], ops_db) is None
    assert not ops_db.exists()


def test_readable_empty_ops_attempt_ledger_keeps_measured_zero(tmp_path: Path) -> None:
    initialize_archive_database(tmp_path / "ops.db", ArchiveTier.OPS)
    body = _scrape(tmp_path / "index.db")
    assert 'polylogue_daemon_metrics_collection_available{group="ops_attempts"} 1' in body
    assert 'polylogue_live_ingest_attempts_total{status="completed"} 0' in body
    assert 'polylogue_live_ingest_attempts_total{status="failed"} 0' in body
    assert "polylogue_live_ingest_attempts_in_flight 0" in body


def test_malformed_source_tier_does_not_publish_version_zero(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, message_count INTEGER, raw_id TEXT)"
        )
    (tmp_path / "source.db").write_bytes(b"not a sqlite database")
    body = _scrape(db)
    assert "polylogue_daemon_uptime_seconds " in body
    assert 'polylogue_daemon_metrics_collection_available{group="archive_storage"} 0' in body
    assert 'polylogue_archive_tier_user_version{tier="source"} 0' not in body
    assert "polylogue_archive_ready 0" not in body


def test_process_collection_failure_isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(metrics, "storage_io_observation", lambda: (_ for _ in ()).throw(RuntimeError("private")))
    body = _scrape(tmp_path / "index.db")
    assert "polylogue_daemon_uptime_seconds " in body
    assert "polylogue_diagnostic_delivery_total" in body
    assert "polylogue_storage_io_phase_observable" not in body
    assert 'polylogue_daemon_metrics_collection_available{group="process_io"} 0' in body


@pytest.mark.parametrize(
    ("reader_name", "fallback", "reason"),
    [
        ("_ops_attempt_counts", None, "attempt_counts_unreadable"),
        ("_ops_recent_attempt_durations", [], "attempt_durations_unreadable"),
        ("_ops_latest_ingest_memory", [], "ingest_memory_unreadable"),
        ("_ops_storage_route_counts", None, "storage_route_counts_unreadable"),
        ("_archive_latest_embedding_run_state", None, "latest_embedding_run_unreadable"),
    ],
)
def test_tier_query_fault_is_visible_and_closes_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader_name: str, fallback: object, reason: str
) -> None:
    """Removing typed refusal, its diagnostic or finally-close makes this red."""
    import polylogue.storage.sqlite.connection_profile as profiles

    database = tmp_path / "ops.db"
    conn = sqlite3.connect(database)
    conn.set_authorizer(
        lambda action, *_args: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_SELECT else sqlite3.SQLITE_OK
    )
    monkeypatch.setattr(profiles, "open_readonly_connection", lambda *_args, **_kwargs: conn)
    events: list[dict[str, object]] = []
    monkeypatch.setattr(metrics, "emit", lambda _event, **fields: events.append(fields))
    try:
        if reader_name == "_archive_latest_embedding_run_state":
            assert getattr(metrics, reader_name)(database) == fallback
        else:
            with pytest.raises(sqlite3.OperationalError):
                getattr(metrics, reader_name)(database)
        assert len(events) == 1
        assert events[0]["reason"] == reason
        assert events[0]["outcome"] == "degraded"
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("PRAGMA database_list")
    finally:
        conn.close()


def test_throughput_query_fault_does_not_append_partial_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed acquisition returns the existing refusal without claiming a rate."""
    import polylogue.storage.sqlite.connection_profile as profiles

    database = tmp_path / "ops.db"
    conn = sqlite3.connect(database)
    conn.set_authorizer(
        lambda action, *_args: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_SELECT else sqlite3.SQLITE_OK
    )
    monkeypatch.setattr(profiles, "open_readonly_connection", lambda *_args, **_kwargs: conn)
    lines: list[str] = []
    try:
        assert metrics._emit_ops_throughput_metrics(lines, database) is False
        assert lines == []
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("PRAGMA database_list")
    finally:
        conn.close()
