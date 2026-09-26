"""Current-run catch-up progress from attempt and stage receipts."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from polylogue.daemon.catchup_status import _cumulative_attempts, catchup_status_info
from polylogue.sources.live.cold_build import ColdBuildGeneration
from polylogue.sources.live.production_baseline import ProductionSourceBaseline, SourceDecision
from polylogue.storage.index_generation import IndexGeneration, IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_cumulative_progress_uses_latest_attempt_snapshots_and_no_guessed_eta(tmp_path: Path) -> None:
    """Summing progress events twice or deriving a page-sized ETA makes this red."""
    ops = tmp_path / "ops.db"
    with sqlite3.connect(ops) as conn:
        conn.executescript(
            """
            CREATE TABLE daemon_lifecycle(started_at_ms INTEGER);
            CREATE TABLE ingest_attempts(attempt_id TEXT, status TEXT, started_at_ms INTEGER,
                                         heartbeat_at_ms INTEGER, finished_at_ms INTEGER);
            CREATE TABLE daemon_stage_events(attempt_id TEXT, stage TEXT, observed_at_ms INTEGER,
                                             payload_json TEXT);
            """
        )
        conn.execute("INSERT INTO daemon_lifecycle VALUES (1000)")
        conn.executemany(
            "INSERT INTO ingest_attempts VALUES (?, ?, ?, ?, ?)",
            [
                ("old", "completed", 500, 900, 900),
                ("a", "completed_with_failures", 1100, 3000, 3000),
                ("b", "completed", 1200, 4000, 4000),
            ],
        )
        conn.executemany(
            "INSERT INTO daemon_stage_events VALUES (?, ?, ?, ?)",
            [
                ("old", "completed", 900, json.dumps({"succeeded_file_count": 100, "ingested_bytes": 100000000})),
                ("a", "started", 1500, json.dumps({"succeeded_file_count": 0, "needed_file_count": 2})),
                (
                    "a",
                    "completed",
                    3000,
                    json.dumps(
                        {
                            "succeeded_file_count": 1,
                            "failed_file_count": 1,
                            "needed_file_count": 2,
                            "ingested_bytes": 1000000,
                            "failed_bytes": 100,
                            "excluded_file_count": 0,
                        }
                    ),
                ),
                (
                    "b",
                    "completed",
                    4000,
                    json.dumps(
                        {
                            "succeeded_file_count": 1,
                            "needed_file_count": 2,
                            "deferred_file_count": 1,
                            "ingested_bytes": 2000000,
                            "excluded_file_count": 0,
                        }
                    ),
                ),
            ],
        )
    totals = _cumulative_attempts(ops, now=datetime.fromtimestamp(5, tz=UTC))
    assert totals["cumulative_succeeded_file_count"] == 2
    assert totals["cumulative_failed_file_attempts"] == 1
    assert totals["cumulative_failed_ingest_attempt_count"] == 1
    assert totals["cumulative_unmeasured_failed_ingest_attempt_count"] == 0
    assert totals["cumulative_deferred_file_count"] == 1
    assert totals["cumulative_ingested_bytes"] == 3000000
    rate = totals["running_mb_per_second"]
    assert isinstance(rate, float) and rate > 0
    assert totals["last_advanced_age_s"] == 1.0
    assert totals["planned_file_count"] is None
    assert totals["eta_s"] is None


def test_failed_attempt_without_chunk_summary_is_visible_after_terminal_update(tmp_path: Path) -> None:
    """A terminal attempt update must invalidate totals without a new stage event."""
    ops = tmp_path / "ops.db"
    with sqlite3.connect(ops) as conn:
        conn.executescript(
            "CREATE TABLE daemon_lifecycle(started_at_ms INTEGER);"
            "CREATE TABLE ingest_attempts(attempt_id TEXT, status TEXT, started_at_ms INTEGER,"
            "heartbeat_at_ms INTEGER, finished_at_ms INTEGER);"
            "CREATE TABLE daemon_stage_events(attempt_id TEXT, stage TEXT, observed_at_ms INTEGER,"
            "payload_json TEXT);"
            "INSERT INTO daemon_lifecycle VALUES (1000);"
            "INSERT INTO ingest_attempts VALUES ('crash', 'running', 1100, 1200, NULL);"
        )
    first = _cumulative_attempts(ops, now=datetime.fromtimestamp(2, tz=UTC))
    assert first["cumulative_failed_ingest_attempt_count"] == 0
    with sqlite3.connect(ops) as conn:
        conn.execute(
            "UPDATE ingest_attempts SET status = 'failed', heartbeat_at_ms = 2000, finished_at_ms = 2000 "
            "WHERE attempt_id = 'crash'"
        )
    second = _cumulative_attempts(ops, now=datetime.fromtimestamp(3, tz=UTC))
    assert second["cumulative_failed_ingest_attempt_count"] == 1
    assert second["cumulative_unmeasured_failed_ingest_attempt_count"] == 1
    assert second["cumulative_failed_file_attempts"] == 0


def test_missing_attempt_receipts_are_unavailable_not_measured_zero(tmp_path: Path) -> None:
    """An absent ops tier cannot assert that no input failed."""
    status = catchup_status_info(
        tmp_path / "index.db", latest_attempt=None, convergence=SimpleNamespace(), ops_db=tmp_path / "ops.db"
    )
    assert status.cumulative_available is False
    assert status.cumulative_succeeded_file_count is None
    assert status.cumulative_failed_file_attempts is None
    assert status.cumulative_failed_ingest_attempt_count is None


def test_cold_build_eta_counts_only_matching_applied_raw_revisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A duplicate application or a retained but unapplied raw must not advance ETA."""
    index = tmp_path / "candidate.db"
    source = tmp_path / "source.db"
    with sqlite3.connect(index) as conn:
        conn.execute("CREATE TABLE raw_revision_applications(raw_id TEXT, decision TEXT)")
        conn.executemany(
            "INSERT INTO raw_revision_applications VALUES (?, ?)",
            [
                ("one", "selected_baseline"),
                ("one", "selected_baseline"),
                ("two", "deferred"),
                ("other", "selected_baseline"),
            ],
        )
    with sqlite3.connect(source) as conn:
        conn.execute(
            "CREATE TABLE raw_sessions(raw_id TEXT PRIMARY KEY, source_path TEXT, source_index INTEGER, blob_hash BLOB)"
        )
        conn.executemany(
            "INSERT INTO raw_sessions VALUES (?, ?, ?, ?)",
            [
                ("one", "/export.zip:a.json", 3, bytes.fromhex("aa" * 32)),
                ("two", "/export.zip:b.json", 4, bytes.fromhex("bb" * 32)),
                ("other", "/other.json", 0, bytes.fromhex("cc" * 32)),
            ],
        )

    class Baseline:
        reads = 0

        @property
        def accepted(self) -> tuple[SourceDecision, ...]:
            self.reads += 1
            return (
                SourceDecision("s", "/export.zip:a.json", "accepted", "", "aa" * 32, 3),
                SourceDecision("s", "/export.zip:b.json", "accepted", "", "bb" * 32, 4),
            )

    baseline = Baseline()
    from polylogue.sources.live import cold_build
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection as real_open

    profiles: list[tuple[str, bool]] = []
    candidate_sql: list[str] = []

    def profile_reader(path: Path, **kwargs: object) -> sqlite3.Connection:
        profiles.append((path.name, kwargs.get("validate_schema", True) is True))
        # The miniature fixture has no production schema stamp. Preserve all
        # other named profile settings while skipping only that fixture check.
        tier = kwargs.get("tier")
        timeout_class = kwargs.get("timeout_class")
        assert isinstance(tier, ArchiveTier)
        assert isinstance(timeout_class, str)
        conn = real_open(path, tier=tier, timeout_class=timeout_class, validate_schema=False)
        if path.name == "candidate.db":
            conn.set_trace_callback(candidate_sql.append)
        return conn

    monkeypatch.setattr(cold_build, "open_readonly_connection", profile_reader)
    generation = ColdBuildGeneration(
        archive_root=tmp_path,
        generation=cast(IndexGeneration, SimpleNamespace(index_path=str(index), generation_id="candidate")),
        reason="test",
        operation_id="test",
        _store=cast(IndexGenerationStore, SimpleNamespace()),
        source_baseline=cast(ProductionSourceBaseline, baseline),
    )
    initial = generation.accepted_progress
    assert initial == (None, 2, None, None)
    generation.refresh_accepted_progress()
    assert profiles == [("candidate.db", False), ("source.db", True)]
    assert any(statement == "BEGIN" for statement in candidate_sql)
    assert any("rowid <=" in statement for statement in candidate_sql)
    completed, planned, rate, eta = generation.accepted_progress
    assert (completed, planned) == (1, 2)
    assert rate is not None and rate > 0
    assert eta is not None and eta > 0
    with sqlite3.connect(index) as conn:
        conn.execute("INSERT INTO raw_revision_applications VALUES ('two', 'applied_append')")
    generation.refresh_accepted_progress()
    assert generation.accepted_progress[0] == 2
    assert generation.accepted_progress[3] == 0
    assert baseline.reads == 1

    class NoWarmLookup(dict[tuple[str, int, str], int]):
        def __getitem__(self, key: tuple[str, int, str]) -> int:
            raise AssertionError(f"warm status traversed baseline weights: {key}")

    generation._accepted_progress_weights = NoWarmLookup(generation._accepted_progress_weights)
    assert generation.accepted_progress[0] == 2
