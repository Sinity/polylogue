"""Capture-completeness health alert tests (polylogue-3uw AC2).

Proves a seeded missed-session scenario (SessionStart hook evidence with no
matching archived session) trips the daemon health alert via the production
:func:`~polylogue.daemon.health._check_capture_coverage_medium` check, not a
parallel alerting path.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.daemon.health import HealthSeverity, _check_capture_coverage_medium
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.frozen_clock import DEFAULT_FROZEN_EPOCH

GRACE_MS = 15 * 60 * 1000
NOW_MS = int(DEFAULT_FROZEN_EPOCH * 1000)

pytestmark = pytest.mark.frozen_clock_modules("polylogue.daemon.health")


@pytest.fixture
def coverage_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: object) -> Path:
    archive = tmp_path / "archive"
    initialize_archive_database(archive / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(archive / "index.db", ArchiveTier.INDEX)
    monkeypatch.setattr("polylogue.daemon.health.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.daemon.health._active_health_db_path", lambda: archive / "index.db")
    return archive


def _insert_hook_event(archive: Path, native_id: str, *, observed_at_ms: int) -> None:
    payload = {"event_type": "SessionStart", "session_id": native_id}
    with sqlite3.connect(archive / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, session_native_id,
                source_path, event_type, payload_json, observed_at_ms
            ) VALUES (?, 'claude-code-session', ?, ?, ?, 'SessionStart', ?, ?)
            """,
            (
                f"hook-{native_id}",
                native_id,
                native_id,
                f"/hooks/{native_id}.jsonl",
                json.dumps(payload),
                observed_at_ms,
            ),
        )


def test_seeded_missed_session_trips_capture_coverage_alert(coverage_archive: Path) -> None:
    _insert_hook_event(coverage_archive, "session-lost", observed_at_ms=NOW_MS - GRACE_MS - 1_000)

    alert = _check_capture_coverage_medium()

    assert alert.check_name == "capture_coverage"
    assert alert.severity != HealthSeverity.OK
    assert "1 session(s)" in alert.message


def test_no_hook_evidence_is_ok(coverage_archive: Path) -> None:
    alert = _check_capture_coverage_medium()

    assert alert.check_name == "capture_coverage"
    assert alert.severity == HealthSeverity.OK


def test_matched_session_does_not_trip_alert(coverage_archive: Path) -> None:
    _insert_hook_event(coverage_archive, "session-ok", observed_at_ms=NOW_MS - GRACE_MS - 1_000)
    with sqlite3.connect(coverage_archive / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, content_hash, updated_at_ms)
            VALUES ('session-ok', 'claude-code-session', ?, ?)
            """,
            (bytes.fromhex("11" * 32), NOW_MS),
        )

    alert = _check_capture_coverage_medium()

    assert alert.severity == HealthSeverity.OK
