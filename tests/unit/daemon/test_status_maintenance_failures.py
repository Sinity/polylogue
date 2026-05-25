"""Tests for daemon status integration of maintenance failure routing (#1198).

Pins the acceptance criteria:

* ``_raw_failure_info()`` surfaces maintenance failures with
  ``source="maintenance"`` and the originating ``operation_id``
  alongside live-ingest failures;
* ``DaemonStatus`` carries the maintenance count via
  ``raw_maintenance_failures`` and the merged sample list;
* ``_check_raw_failures_medium`` escalates when maintenance failures
  cross the existing thresholds, citing the operation id;
* the plain-text formatter emits the maintenance bucket and per-op
  hint when the sample list is non-empty.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

from polylogue.daemon.health import (
    HealthSeverity,
    _check_raw_failures_medium,
)
from polylogue.daemon.status import (
    RawFailureSample,
    _raw_failure_info,
    format_daemon_status_lines,
)
from polylogue.maintenance.failure_routing import route_failure_sample
from polylogue.maintenance.planner import FailureSample


def _seed_raw_table(db: Path, parse_error: str | None = None, validation_status: str | None = None) -> None:
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_conversations (
                raw_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                payload_provider TEXT,
                source_name TEXT,
                source_path TEXT NOT NULL,
                source_index INTEGER,
                blob_size INTEGER NOT NULL,
                acquired_at TEXT NOT NULL,
                file_mtime TEXT,
                parsed_at TEXT,
                parse_error TEXT,
                validated_at TEXT,
                validation_status TEXT,
                validation_error TEXT,
                validation_drift_count INTEGER DEFAULT 0,
                validation_provider TEXT,
                validation_mode TEXT,
                detection_warnings TEXT
            );
            """
        )
        if parse_error is not None or validation_status is not None:
            conn.execute(
                "INSERT INTO raw_conversations (raw_id, source_name, source_path, blob_size, acquired_at, parse_error, validation_status) "
                "VALUES (?,?,?,?,?,?,?)",
                ("raw-1", "claude-code", "/x/y", 1, "2026-01-01T00:00:00Z", parse_error, validation_status),
            )


def _route(archive_root: Path, op_id: str, kind: str = "RuntimeError", message: str = "boom") -> None:
    route_failure_sample(
        FailureSample(kind=kind, locator="target:session_insights", message=message),
        operation_id=op_id,
        archive_root=archive_root,
        target="session_insights",
    )


def test_raw_failure_info_surfaces_maintenance_with_no_db(tmp_path: Path) -> None:
    """Maintenance failures appear even when the archive DB doesn't exist yet."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    _route(archive_root, op_id="op-1")

    db = tmp_path / "missing.db"
    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status.archive_root", return_value=archive_root),
        patch("polylogue.daemon.status._maintenance_failure_info") as mock_mf,
    ):
        # Re-route through real reader.
        from polylogue.maintenance.failure_routing import (
            count_maintenance_failures,
            read_maintenance_failures,
        )

        def _real() -> tuple[list[RawFailureSample], int]:
            records = read_maintenance_failures(archive_root)
            samples = [
                RawFailureSample(
                    failure_kind="maintenance",
                    provider_hint=r.target or None,
                    redacted_error=f"{r.kind}: {r.message}",
                    source="maintenance",
                    operation_id=r.operation_id,
                    locator=r.locator,
                )
                for r in records
            ]
            return samples, count_maintenance_failures(archive_root)

        mock_mf.side_effect = _real
        info = _raw_failure_info()

    assert info["maintenance_failures"] == 1
    samples = info["samples"]
    assert isinstance(samples, list)
    assert len(samples) == 1
    assert samples[0].source == "maintenance"
    assert samples[0].operation_id == "op-1"
    assert samples[0].failure_kind == "maintenance"


def test_raw_failure_info_merges_ingest_and_maintenance(tmp_path: Path) -> None:
    db = tmp_path / "polylogue.db"
    _seed_raw_table(db, parse_error="JSONDecodeError at /tmp/foo")

    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    _route(archive_root, op_id="op-mix-1")
    _route(archive_root, op_id="op-mix-2", kind="ValueError", message="x")

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status.archive_root", return_value=archive_root),
    ):
        info = _raw_failure_info()

    assert info["parse_failures"] == 1
    assert info["maintenance_failures"] == 2

    samples = info["samples"]
    assert isinstance(samples, list)
    by_source = {s.source for s in samples}
    assert by_source == {"ingest", "maintenance"}

    # The maintenance samples carry their operation_id.
    maint = [s for s in samples if s.source == "maintenance"]
    assert {s.operation_id for s in maint} == {"op-mix-1", "op-mix-2"}


def test_check_raw_failures_medium_escalates_on_maintenance(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        _route(archive_root, op_id=f"op-batch-{i}")

    db = tmp_path / "polylogue.db"
    _seed_raw_table(db)

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status.archive_root", return_value=archive_root),
    ):
        alert = _check_raw_failures_medium()

    assert alert.severity == HealthSeverity.WARNING
    assert "5" in alert.message
    assert "maintenance" in alert.message
    assert "op=" in alert.message  # op-id hint included


def test_check_raw_failures_medium_critical_on_large_maintenance_backlog(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    for _ in range(60):
        _route(archive_root, op_id="op-backlog")

    db = tmp_path / "polylogue.db"
    _seed_raw_table(db)

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status.archive_root", return_value=archive_root),
    ):
        alert = _check_raw_failures_medium()

    assert alert.severity == HealthSeverity.CRITICAL
    assert "60" in alert.message


def test_format_daemon_status_lines_renders_maintenance_bucket() -> None:
    payload = {
        "raw_parse_failures": 0,
        "raw_validation_failures": 0,
        "raw_quarantined": 0,
        "raw_maintenance_failures": 3,
        "raw_failure_samples": [
            {
                "failure_kind": "maintenance",
                "provider_hint": "session_insights",
                "redacted_error": "RuntimeError: insight rebuild failed",
                "source": "maintenance",
                "operation_id": "op-12345678abcdef",
                "locator": "target:session_insights",
            },
        ],
    }
    lines = format_daemon_status_lines(payload)  # type: ignore[arg-type]
    rendered = "\n".join(lines)
    assert "3 maintenance" in rendered
    assert "[maintenance]" in rendered
    assert "op=op-12345" in rendered  # truncated id hint
