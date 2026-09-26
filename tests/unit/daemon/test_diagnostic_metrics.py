"""Diagnostic delivery loss is visible on the metrics surface."""

from __future__ import annotations

import sqlite3
from http import HTTPStatus
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from polylogue.daemon import metrics
from polylogue.operations.storage_io_observation import IoPhaseObservation, StorageIoObservation


def _scrape(db: Path) -> str:
    responder = MagicMock()
    metrics.handle_metrics(responder, db)
    assert responder._send_text.call_args.args[0] == HTTPStatus.OK
    return responder._send_text.call_args.args[1]


def test_metrics_expose_maintained_sink_loss_without_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Omitting the loss series or hiding drops as failures makes this red."""
    monkeypatch.setattr(
        metrics,
        "diagnostic_snapshot",
        lambda: {"queued": 3, "dropped": 7, "failures": 2, "delivered": 11, "undrained": 1, "high_water": 9},
    )
    body = metrics.format_metrics(tmp_path / "missing.db")
    assert 'polylogue_diagnostic_delivery_total{outcome="dropped"} 7' in body
    assert 'polylogue_diagnostic_delivery_total{outcome="failures"} 2' in body
    assert 'polylogue_diagnostic_delivery_total{outcome="delivered"} 11' in body
    assert 'polylogue_diagnostic_delivery_total{outcome="undrained"} 1' in body
    assert "polylogue_diagnostic_queue_depth 3" in body


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
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, message_count INTEGER)")
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
    assert 'polylogue_daemon_metrics_collection_available{group="archive_index",reason="collector_failed"} 0' in body
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
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
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


def test_missing_and_malformed_index_are_unavailable_not_measured_zero(tmp_path: Path) -> None:
    missing = _scrape(tmp_path / "index.db")
    assert (
        'polylogue_daemon_metrics_collection_available{group="archive_index",reason="schema_unavailable"} 0' in missing
    )
    assert "polylogue_archive_sessions_total " not in missing
    assert "polylogue_fts_triggers_all_present 0" not in missing
    (tmp_path / "index.db").write_bytes(b"not a sqlite database")
    malformed = _scrape(tmp_path / "index.db")
    assert (
        'polylogue_daemon_metrics_collection_available{group="archive_index",reason="archive_unreadable"} 0'
        in malformed
    )
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
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
    body = _scrape(db)
    assert 'polylogue_daemon_metrics_collection_available{group="archive_index",reason="none"} 1' in body
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
    assert 'polylogue_convergence_debt_count{stage="materialize",status="failed"} 1' in body
    assert 'polylogue_daemon_metrics_collection_available{group="ops_attempts",reason="schema_unavailable"} 0' in body
    assert "polylogue_live_ingest_attempts_total{status=" not in body
    assert "polylogue_live_ingest_attempts_in_flight 0" not in body
    assert "polylogue_stale_cursor_writes_total 0" not in body
    assert "polylogue_live_ingest_storage_route_total 0" not in body


def test_readable_empty_ops_attempt_ledger_keeps_measured_zero(tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute(
            "CREATE TABLE ingest_attempts (status TEXT NOT NULL, started_at_ms INTEGER, finished_at_ms INTEGER)"
        )
    body = _scrape(tmp_path / "index.db")
    assert 'polylogue_daemon_metrics_collection_available{group="ops_attempts",reason="none"} 1' in body
    assert 'polylogue_live_ingest_attempts_total{status="completed"} 0' in body
    assert 'polylogue_live_ingest_attempts_total{status="failed"} 0' in body
    assert "polylogue_live_ingest_attempts_in_flight 0" in body


def test_malformed_source_tier_does_not_publish_version_zero(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
    (tmp_path / "source.db").write_bytes(b"not a sqlite database")
    body = _scrape(db)
    assert "polylogue_daemon_uptime_seconds " in body
    assert (
        'polylogue_daemon_metrics_collection_available{group="archive_storage",reason="archive_unreadable"} 0' in body
    )
    assert 'polylogue_archive_tier_user_version{tier="source"} 0' not in body
    assert "polylogue_archive_ready 0" not in body


def test_process_collection_failure_isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(metrics, "storage_io_observation", lambda: (_ for _ in ()).throw(RuntimeError("private")))
    body = _scrape(tmp_path / "index.db")
    assert "polylogue_daemon_uptime_seconds " in body
    assert "polylogue_diagnostic_delivery_total" in body
    assert "polylogue_storage_io_phase_observable" not in body
    assert 'polylogue_daemon_metrics_collection_available{group="process_io",reason="collector_failed"} 0' in body
