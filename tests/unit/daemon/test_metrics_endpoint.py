"""Contract tests for the daemon ``/metrics`` Prometheus endpoint (#1321).

Pins:

1. **Exposition format** — every metric carries ``# HELP`` and ``# TYPE``
   directives before its samples, and the response uses the documented
   ``text/plain; version=0.0.4`` content type that Prometheus expects.
2. **Series inventory** — the operator-facing series names are stable;
   adding or removing one is a contract change visible in this file.
3. **Graceful degradation** — when a backing table is missing (fresh
   archive, schema bump mid-rollout) the endpoint still emits the
   discovery skeleton with zero values rather than 5xx-ing.
4. **Unauthenticated** — same posture as ``/healthz/*``; scrapers do
   not carry credentials.
5. **State observation** — current ingest and debt counts come from ops.db;
   index FTS trigger state remains visible when ops is unavailable.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from polylogue.daemon.metrics import PROMETHEUS_CONTENT_TYPE
from polylogue.operations.daemon_metrics import (
    _collect_group,
    _emit_unmeasured_probe,
    format_metrics,
)
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture

pytestmark = pytest.mark.uses_real_clock(
    "Metrics endpoint test records a live rebuild-ingest heartbeat timestamp; production readiness and metrics intentionally compare it against the host clock."
)

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


# ---------------------------------------------------------------------------
# Series inventory — pinned contract
# ---------------------------------------------------------------------------

EXPECTED_SERIES: frozenset[str] = frozenset(
    {
        "polylogue_daemon_uptime_seconds",
        "polylogue_daemon_build_info",
        "polylogue_storage_io_phase_total",
        "polylogue_storage_io_phase_seconds_total",
        "polylogue_storage_io_phase_observable",
        "polylogue_daemon_result_cache_entries",
        "polylogue_daemon_result_cache_bytes",
        "polylogue_daemon_result_cache_hits_total",
        "polylogue_daemon_result_cache_misses_total",
        "polylogue_daemon_result_cache_evictions_total",
        "polylogue_live_ingest_attempts_total",
        "polylogue_live_ingest_attempts_in_flight",
        "polylogue_live_ingest_storage_route_total",
        "polylogue_live_ingest_attempt_duration_seconds",
        "polylogue_convergence_debt_count",
        "polylogue_fts_trigger_present",
        "polylogue_fts_triggers_all_present",
        "polylogue_fts_freshness_ready",
        "polylogue_fts_drift_rows",
        "polylogue_live_ingest_memory_mebibytes",
        "polylogue_stale_cursor_writes_total",
        "polylogue_embedding_sessions",
        "polylogue_embedding_messages",
        "polylogue_embedding_coverage_percent",
        "polylogue_embedding_status_state",
        "polylogue_embedding_retrieval_ready",
        "polylogue_embedding_latest_catchup_run_info",
        "polylogue_embedding_latest_catchup_sessions",
        "polylogue_embedding_latest_catchup_messages",
        "polylogue_embedding_latest_catchup_estimated_cost_usd",
        "polylogue_archive_tier_present",
        "polylogue_archive_tier_count",
        "polylogue_archive_tier_file_size_bytes",
        "polylogue_archive_tier_user_version",
        "polylogue_archive_storage_layout",
        "polylogue_archive_storage_ready",
        "polylogue_archive_active_store",
        "polylogue_archive_active_tier_role",
        "polylogue_archive_ready",
        "polylogue_archive_blocker_count",
        "polylogue_archive_blocker",
        "polylogue_hook_flow_healthy",
        "polylogue_hook_flow_state",
        "polylogue_hook_sessions",
        "polylogue_hook_event_observed_ratio",
    }
)


_HELP_RE = re.compile(r"^# HELP (\S+) ")
_TYPE_RE = re.compile(r"^# TYPE (\S+) (counter|gauge|histogram|summary|untyped)$")


def test_independent_unmeasured_collectors_share_one_metric_header() -> None:
    """Two failing probes still form valid shared Prometheus metadata.

    Anti-vacuity: remove global header deduplication from ``_collect_group``
    and both collectors append their own HELP/TYPE declarations.
    """
    lines: list[str] = []
    states: list[tuple[dict[str, str], int]] = []

    def collector(probe: str) -> Callable[[list[str]], None]:
        def collect(pending: list[str]) -> None:
            _emit_unmeasured_probe(pending, probe)

        return collect

    _collect_group(lines, states, "hooks", collector("hook_statuses"))
    _collect_group(lines, states, "index", collector("convergence_debt"))

    assert states == [
        ({"group": "hooks"}, 1),
        ({"group": "hooks", "reason": "none"}, -1),
        ({"group": "index"}, 1),
        ({"group": "index", "reason": "none"}, -1),
    ]
    assert sum(line.startswith("# HELP polylogue_probe_unmeasured ") for line in lines) == 1
    assert sum(line == "# TYPE polylogue_probe_unmeasured gauge" for line in lines) == 1
    assert 'polylogue_probe_unmeasured{probe="hook_statuses"} 1' in lines
    assert 'polylogue_probe_unmeasured{probe="convergence_debt"} 1' in lines


def _parse_exposition(body: str) -> dict[str, dict[str, object]]:
    """Parse Prometheus exposition body into ``{metric: {type, samples}}``."""
    metrics: dict[str, dict[str, object]] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if m := _HELP_RE.match(line):
            metrics.setdefault(m.group(1), {"type": None, "help": True, "samples": []})
        elif m := _TYPE_RE.match(line):
            entry = metrics.setdefault(m.group(1), {"type": None, "help": False, "samples": []})
            entry["type"] = m.group(2)
        elif line.startswith("#"):
            continue
        else:
            # sample line: "metric{labels} value"
            name = line.split("{", 1)[0].split(" ", 1)[0]
            entry = metrics.setdefault(name, {"type": None, "help": False, "samples": []})
            cast(list[str], entry["samples"]).append(line)
    return metrics


# ---------------------------------------------------------------------------
# format_metrics() unit tests
# ---------------------------------------------------------------------------


class TestFormatMetricsExpositionShape:
    """Hand-rolled exposition must satisfy the documented format."""

    def test_missing_db_emits_discovery_skeleton(self, tmp_path: Path) -> None:
        body = format_metrics(tmp_path / "missing.db")
        parsed = _parse_exposition(body)
        # Uptime and build_info always emit; remaining series emit as
        # zero-sample discovery placeholders when the DB is absent.
        for name in EXPECTED_SERIES:
            assert name in parsed, f"missing series for fresh archive: {name}"
            assert parsed[name]["type"] is not None, f"missing TYPE for {name}"

    def test_unreadable_status_frame_has_one_unavailable_metric(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from polylogue.daemon.status_snapshot import (
            get_status_snapshot_payload,
            refresh_status_snapshot,
            reset_status_snapshot,
        )

        missing_index = tmp_path / "missing-index.db"

        def unreadable_index(_root: Path) -> Path:
            raise PermissionError("synthetic unreadable index")

        monkeypatch.setattr("polylogue.daemon.status_snapshot.resolve_active_index_path", unreadable_index)
        try:
            refresh_status_snapshot(payload={"ok": True})
            status = get_status_snapshot_payload()
            snapshot = cast(dict[str, object], status["status_snapshot"])
            assert snapshot["state"] == "unavailable"
            assert snapshot["frame_error"] == "archive frame unavailable"

            body = format_metrics(missing_index)
            assert 'polylogue_status_snapshot_state{state="unavailable"} 1' in body
            for state in ("fresh", "stale", "missing"):
                assert f'polylogue_status_snapshot_state{{state="{state}"}} 0' in body
        finally:
            reset_status_snapshot()

    def test_every_sample_line_has_preceding_help_and_type(self, tmp_path: Path) -> None:
        body = format_metrics(tmp_path / "missing.db")
        parsed = _parse_exposition(body)
        for name, entry in parsed.items():
            assert entry["help"], f"{name} missing # HELP"
            assert entry["type"], f"{name} missing # TYPE"

    def test_build_info_exposes_version_label(self, tmp_path: Path) -> None:
        body = format_metrics(tmp_path / "missing.db")
        assert re.search(r'polylogue_daemon_build_info\{[^}]*\bversion="[^"]+"', body)

    def test_build_info_exposes_full_revision_and_dirty_labels(self, tmp_path: Path) -> None:
        """polylogue-6rvt: the metric must carry the full (untruncated) build
        revision plus an explicit dirty flag, not just the human version
        string — dashboards/attestation join on ``revision``, not
        ``version``."""
        from polylogue.version import VERSION_INFO

        body = format_metrics(tmp_path / "missing.db")
        match = re.search(r"polylogue_daemon_build_info\{([^}]*)\} 1", body)
        assert match is not None, body
        labels = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
        assert labels["version"] == VERSION_INFO.version
        assert labels["revision"] == (VERSION_INFO.commit or "unknown")
        assert labels["dirty"] in {"true", "false"}
        assert labels["dirty"] == ("true" if VERSION_INFO.dirty else "false")
        # Never truncated to a short/collision-prone prefix.
        if labels["revision"] != "unknown":
            assert len(labels["revision"]) == 40

    def test_archive_storage_metrics_report_archive_file_sets(self, tmp_path: Path) -> None:
        for spec in ARCHIVE_TIER_SPECS.values():
            if spec.tier is not ArchiveTier.EMBEDDINGS:
                if spec.tier is ArchiveTier.SOURCE:
                    initialize_runtime_source_fixture(tmp_path / spec.filename)
                else:
                    initialize_archive_database(tmp_path / spec.filename, spec.tier)
        with sqlite3.connect(tmp_path / "embeddings.db") as conn:
            conn.execute(f"PRAGMA user_version = {EMBEDDINGS_SCHEMA_VERSION}")
            conn.commit()

        body = format_metrics(tmp_path / "index.db")

        assert 'polylogue_archive_tier_present{tier="source"} 1' in body
        assert 'polylogue_archive_tier_present{tier="index"} 1' in body
        assert 'polylogue_archive_tier_present{tier="embeddings"} 1' in body
        assert f'polylogue_archive_tier_count{{state="present"}} {len(ARCHIVE_TIER_SPECS)}' in body
        assert 'polylogue_archive_tier_count{state="missing"} 0' in body
        assert (
            f'polylogue_archive_tier_user_version{{tier="source"}} {ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]}'
            in body
        )
        assert 'polylogue_archive_storage_layout{layout="archive_complete"} 1' in body
        assert 'polylogue_archive_storage_layout{layout="archive_partial"} 0' in body
        assert 'polylogue_archive_storage_ready{state="archive_runtime"} 1' in body
        assert 'polylogue_archive_storage_ready{state="final_shape"} 1' in body
        assert 'polylogue_archive_active_store{store="archive_file_set"} 1' in body
        assert 'polylogue_archive_active_store{store="empty"} 0' in body
        assert 'polylogue_archive_active_tier_role{role="index"} 1' in body
        assert 'polylogue_archive_active_tier_role{role="unknown"} 0' in body
        assert "polylogue_archive_ready 1" in body
        assert "polylogue_archive_blocker_count 0" in body
        assert 'polylogue_archive_blocker{blocker="missing_archive_tiers"} 0' in body
        assert 'polylogue_fts_trigger_present{trigger="messages_fts_ai"} 1' in body
        assert 'polylogue_fts_trigger_present{trigger="messages_fts_ad"} 1' in body
        assert 'polylogue_fts_trigger_present{trigger="messages_fts_au"} 1' in body

    def test_db_space_metrics_report_wal_and_planner_stats(self, tmp_path: Path) -> None:
        index_db = tmp_path / "index.db"
        initialize_archive_database(index_db, ArchiveTier.INDEX)
        with sqlite3.connect(index_db) as conn:
            conn.execute("ANALYZE")

        body = format_metrics(index_db)

        assert "polylogue_db_wal_file_size_bytes " in body
        assert "polylogue_db_sqlite_stat1_rows " in body

    def test_archive_storage_metrics_report_layout_blockers(self, tmp_path: Path) -> None:
        """Partial archive roots expose blockers without activating unrelated files."""
        initialize_archive_database(tmp_path / "ops.db", ArchiveTier.OPS)
        unrelated_db = tmp_path / "custom.sqlite"
        with sqlite3.connect(unrelated_db) as conn:
            conn.execute("CREATE TABLE unsupported_marker (id INTEGER)")
            conn.commit()

        body = format_metrics(unrelated_db)

        assert 'polylogue_archive_tier_present{tier="ops"} 1' in body
        assert 'polylogue_archive_tier_count{state="present"} 1' in body
        assert f'polylogue_archive_tier_count{{state="missing"}} {len(ARCHIVE_TIER_SPECS) - 1}' in body
        assert 'polylogue_archive_storage_layout{layout="archive_partial"} 1' in body
        assert 'polylogue_archive_storage_layout{layout="archive_missing"} 0' in body
        assert 'polylogue_archive_storage_ready{state="archive_runtime"} 0' in body
        assert 'polylogue_archive_storage_ready{state="final_shape"} 0' in body
        assert 'polylogue_archive_active_store{store="archive_file_set"} 0' in body
        assert 'polylogue_archive_active_store{store="empty"} 1' in body
        assert 'polylogue_archive_active_tier_role{role="unknown"} 1' in body
        assert "polylogue_archive_ready 0" in body
        assert (
            f"polylogue_archive_blocker_count {1 + sum(spec.backup_required for spec in ARCHIVE_TIER_SPECS.values())}"
            in body
        )
        assert 'polylogue_archive_blocker{blocker="missing_archive_tiers"} 1' in body
        assert 'polylogue_archive_blocker{blocker="missing_backup_required_tier:source"} 1' in body

    def test_archive_storage_metrics_gate_runtime_readiness_on_schema_match(self, tmp_path: Path) -> None:
        for spec in ARCHIVE_TIER_SPECS.values():
            if spec.tier is ArchiveTier.SOURCE:
                initialize_runtime_source_fixture(tmp_path / spec.filename)
            else:
                initialize_archive_database(tmp_path / spec.filename, spec.tier)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX] + 1}")

        body = format_metrics(tmp_path / "index.db")

        assert 'polylogue_archive_storage_ready{state="archive_runtime"} 0' in body
        assert 'polylogue_archive_storage_ready{state="final_shape"} 1' in body
        assert 'polylogue_archive_active_store{store="archive_file_set"} 1' in body
        assert "polylogue_archive_ready 0" in body
        assert "polylogue_archive_blocker_count 1" in body
        assert 'polylogue_archive_blocker{blocker="schema_mismatch:index"} 1' in body

    def test_no_unknown_label_escaping(self, tmp_path: Path) -> None:
        """Quotes/backslashes inside label values must be escaped."""
        from polylogue.operations.daemon_metrics import _escape_label_value

        assert _escape_label_value('a"b') == 'a\\"b'
        assert _escape_label_value("c\\d") == "c\\\\d"
        assert _escape_label_value("e\nf") == "e\\nf"


class TestFormatMetricsReadsArchiveState:
    """When the DB exists, series reflect live state."""

    def _make_db(self, tmp_path: Path) -> Path:
        db = tmp_path / "archive.db"
        conn = sqlite3.connect(db)
        try:
            conn.executescript(
                """
                CREATE TABLE blocks (
                    block_id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    text TEXT,
                    search_text TEXT
                );
                CREATE TABLE messages_fts (block_id TEXT PRIMARY KEY, text TEXT);
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    origin TEXT NOT NULL DEFAULT 'codex-session',
                    message_count INTEGER NOT NULL DEFAULT 0,
                    raw_id TEXT
                );
                CREATE TABLE messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    message_type TEXT NOT NULL DEFAULT 'message',
                    material_origin TEXT NOT NULL DEFAULT 'human_authored',
                    word_count INTEGER NOT NULL DEFAULT 8
                );
                CREATE TABLE embedding_status (
                    session_id TEXT PRIMARY KEY,
                    message_count_embedded INTEGER NOT NULL DEFAULT 0,
                    needs_reindex INTEGER NOT NULL,
                    error_message TEXT
                );
                CREATE TABLE message_embedding_refs (message_id TEXT PRIMARY KEY, vector_derivation_hash BLOB);
                CREATE TRIGGER messages_fts_ai AFTER INSERT ON blocks
                    BEGIN SELECT 1; END;
                CREATE TRIGGER messages_fts_ad AFTER DELETE ON blocks
                    BEGIN SELECT 1; END;
                CREATE TRIGGER messages_fts_au AFTER UPDATE ON blocks
                    BEGIN SELECT 1; END;
                """
            )
            conn.executemany(
                "INSERT INTO sessions (session_id, message_count) VALUES (?, ?)",
                [("conv-embedded", 1), ("conv-pending", 1), ("conv-missing-status", 1)],
            )
            conn.executemany(
                "INSERT INTO messages (message_id, session_id) VALUES (?, ?)",
                [
                    ("conv-embedded:m1", "conv-embedded"),
                    ("conv-pending:m1", "conv-pending"),
                    ("conv-missing-status:m1", "conv-missing-status"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO embedding_status (
                    session_id, message_count_embedded, needs_reindex, error_message
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    ("conv-embedded", 1, 0, None),
                    ("conv-pending", 0, 1, "provider timeout"),
                ],
            )
            conn.executemany(
                "INSERT INTO message_embedding_refs (message_id, vector_derivation_hash) VALUES (?, zeroblob(32))",
                [("msg-1",), ("msg-2",)],
            )
            conn.commit()
        finally:
            conn.close()
        return db

    @staticmethod
    def _seed_ops(ops_db: Path) -> None:
        from polylogue.storage.sqlite.archive_tiers.ops_write import (
            add_convergence_debt,
            record_daemon_stage_event,
            record_ingest_attempt,
        )

        initialize_archive_database(ops_db, ArchiveTier.OPS)
        with sqlite3.connect(ops_db) as conn:
            for attempt_id, status, started_at_ms, finished_at_ms in (
                ("a1", "completed", 1_770_000_000_000, 1_770_000_001_500),
                ("a2", "completed", 1_770_000_010_000, 1_770_000_012_500),
                ("a3", "failed", 1_770_000_020_000, 1_770_000_021_000),
                ("a4", "running", 1_770_000_030_000, None),
            ):
                record_ingest_attempt(
                    conn,
                    attempt_id=attempt_id,
                    status=status,
                    started_at_ms=started_at_ms,
                    finished_at_ms=finished_at_ms,
                )
            for debt_id, stage, status in (
                ("d1", "parse", "failed"),
                ("d2", "parse", "failed"),
                ("d3", "convergence", "failed"),
                ("d4", "convergence", "deferred"),
            ):
                add_convergence_debt(
                    conn,
                    stage=stage,
                    target_type="session_id",
                    target_id=debt_id,
                    status=status,
                    attempts=1,
                    created_at_ms=1_770_000_000_000,
                )
            record_daemon_stage_event(
                conn,
                attempt_id="a4",
                stage="full_parse",
                status="running",
                observed_at_ms=1_770_000_030_000,
                payload={"rss_current_mb": 44.0, "cgroup_memory_current_mb": 84.0},
                event_id="stage-a4",
            )
            conn.commit()

    @staticmethod
    def _seed_catchup_run(ops_db: Path) -> None:
        """Seed an embedding catch-up run into the archive ops tier.

        Archive file-sets store catch-up runs in ops.db and track the run outcome
        counts (scanned/embedded/error sessions, embedded messages, cost), so the daemon
        metrics read them from there rather than from the index db.
        """
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.ops_write import upsert_embedding_catchup_run
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        initialize_archive_database(ops_db, ArchiveTier.OPS)
        with sqlite3.connect(ops_db) as conn:
            upsert_embedding_catchup_run(
                conn,
                started_at_ms=1,
                finished_at_ms=2,
                status="interrupted",
                scanned_sessions=3,
                embedded_sessions=2,
                error_count=1,
                embedded_messages=2,
                estimated_cost_usd=0.003,
                error_message="max errors reached (1)",
            )
            conn.commit()

    def test_attempt_counts_round_trip(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        self._seed_ops(tmp_path / "ops.db")
        body = format_metrics(db)
        assert 'polylogue_live_ingest_attempts_total{status="completed"} 2' in body
        assert 'polylogue_live_ingest_attempts_total{status="failed"} 1' in body
        assert 'polylogue_live_ingest_attempts_total{status="running"} 1' in body
        assert "polylogue_live_ingest_attempts_in_flight 1" in body
        assert "polylogue_stale_cursor_writes_total 0" not in body
        assert 'polylogue_probe_unmeasured{probe="stale_cursor_writes_total"} 1' in body

    def test_duration_quantiles(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        self._seed_ops(tmp_path / "ops.db")
        body = format_metrics(db)
        # Two completed durations: 1.5 and 2.5.
        assert 'polylogue_live_ingest_attempt_duration_seconds{quantile="min"} 1.5' in body
        assert 'polylogue_live_ingest_attempt_duration_seconds{quantile="max"} 2.5' in body
        assert 'polylogue_live_ingest_attempt_duration_seconds{quantile="mean"} 2.0' in body

    def test_convergence_debt_grouped_by_stage(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        self._seed_ops(tmp_path / "ops.db")
        body = format_metrics(db)
        assert 'polylogue_convergence_debt_count{stage="convergence",status="failed"} 1' in body
        assert 'polylogue_convergence_debt_count{stage="convergence",status="deferred"} 1' in body
        assert 'polylogue_convergence_debt_count{stage="parse",status="failed"} 2' in body

    def test_convergence_debt_reads_archive_ops(self, tmp_path: Path) -> None:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.ops_write import add_convergence_debt
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        db = self._make_db(tmp_path)
        ops_db = db.with_name("ops.db")
        initialize_archive_database(ops_db, ArchiveTier.OPS)
        with sqlite3.connect(ops_db) as conn:
            add_convergence_debt(
                conn,
                stage="session_profile",
                target_type="session_id",
                target_id="conv-1",
                attempts=1,
                created_at_ms=1_770_000_000_000,
            )
            add_convergence_debt(
                conn,
                stage="session_profile",
                target_type="session_id",
                target_id="conv-2",
                attempts=1,
                created_at_ms=1_770_000_001_000,
            )
            add_convergence_debt(
                conn,
                stage="session_profile",
                target_type="session_id",
                target_id="conv-3",
                status="deferred",
                attempts=1,
                created_at_ms=1_770_000_002_000,
            )

        body = format_metrics(db)

        assert 'polylogue_convergence_debt_count{stage="session_profile",status="failed"} 2' in body
        assert 'polylogue_convergence_debt_count{stage="session_profile",status="deferred"} 1' in body

    def test_retired_index_telemetry_cannot_supply_current_ops_metrics(self, tmp_path: Path) -> None:
        """A populated retired table cannot turn unavailable or empty ops into positive current state.

        Anti-vacuity: restoring any index fallback publishes the sentinel attempt,
        duration, debt, or memory values on at least one of these scrapes.
        """
        db = self._make_db(tmp_path)
        with sqlite3.connect(db) as conn:
            conn.executescript(
                """
                CREATE TABLE live_ingest_attempt (
                    attempt_id TEXT PRIMARY KEY, status TEXT, started_at TEXT,
                    updated_at TEXT, convergence_time_s REAL,
                    rss_current_mb REAL, storage_route TEXT
                );
                CREATE TABLE live_convergence_debt (
                    debt_id TEXT PRIMARY KEY, stage TEXT, status TEXT
                );
                INSERT INTO live_ingest_attempt VALUES
                    ('retired', 'completed', '2026-05-01', '2026-05-01', 99.0, 999.0, 'archive_full');
                INSERT INTO live_convergence_debt VALUES ('retired', 'retired_stage', 'failed');
                """
            )

        missing_ops = format_metrics(db)
        assert 'polylogue_daemon_metrics_collection_available{group="ops_attempts"} 0' in missing_ops
        assert 'polylogue_daemon_metrics_collection_available{group="archive_index"} 1' in missing_ops
        assert 'polylogue_fts_trigger_present{trigger="messages_fts_ai"} 1' in missing_ops
        assert 'polylogue_live_ingest_attempts_total{status="completed"} 1' not in missing_ops
        assert 'polylogue_live_ingest_attempt_duration_seconds{quantile="min"} 99.0' not in missing_ops
        assert 'polylogue_convergence_debt_count{stage="retired_stage",status="failed"} 1' not in missing_ops
        assert 'polylogue_live_ingest_memory_mebibytes{kind="rss_current"} 999.0' not in missing_ops

        initialize_archive_database(tmp_path / "ops.db", ArchiveTier.OPS)
        empty_ops = format_metrics(db)
        assert 'polylogue_daemon_metrics_collection_available{group="ops_attempts"} 1' in empty_ops
        assert 'polylogue_live_ingest_attempts_total{status="completed"} 0' in empty_ops
        assert "polylogue_convergence_debt_count 0" in empty_ops
        assert 'polylogue_live_ingest_attempt_duration_seconds{quantile="min"}' not in empty_ops
        assert 'polylogue_live_ingest_memory_mebibytes{kind="rss_current"}' not in empty_ops

    def test_live_ingest_metrics_read_archive_ops_when_present(self, tmp_path: Path) -> None:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.ops_write import record_daemon_stage_event, record_ingest_attempt
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        db = self._make_db(tmp_path)
        ops_db = db.with_name("ops.db")
        initialize_archive_database(ops_db, ArchiveTier.OPS)
        with sqlite3.connect(ops_db) as conn:
            record_ingest_attempt(
                conn,
                attempt_id="v1-running",
                status="running",
                started_at_ms=1_770_000_000_000,
                heartbeat_at_ms=1_770_000_001_000,
                storage_route="archive_full",
            )
            record_ingest_attempt(
                conn,
                attempt_id="v1-completed",
                status="completed",
                started_at_ms=1_770_000_010_000,
                finished_at_ms=1_770_000_013_000,
                parsed_raw_count=9,
                materialized_count=6,
            )
            record_daemon_stage_event(
                conn,
                attempt_id="v1-running",
                stage="full_parse",
                status="running",
                observed_at_ms=1_770_000_001_000,
                payload={
                    "storage_route": "archive_full",
                    "rss_current_mb": 88.0,
                    "cgroup_memory_file_mb": 33.0,
                },
                event_id="stage-v1",
            )

        body = format_metrics(db)

        assert 'polylogue_live_ingest_attempts_total{status="completed"} 1' in body
        assert 'polylogue_live_ingest_attempts_total{status="failed"} 0' in body
        assert 'polylogue_live_ingest_attempts_total{status="running"} 1' in body
        assert 'polylogue_live_ingest_storage_route_total{route="archive_full"} 1' in body
        assert 'polylogue_live_ingest_storage_route_total{route="unknown"} 1' in body
        assert 'polylogue_live_ingest_attempt_duration_seconds{quantile="min"} 3.0' in body
        assert "polylogue_ingest_throughput_raw_rows_per_second 3.0" in body
        assert "polylogue_ingest_throughput_sessions_per_second 2.0" in body
        assert 'polylogue_live_ingest_memory_mebibytes{kind="rss_current"} 88.0' in body
        assert 'polylogue_live_ingest_memory_mebibytes{kind="cgroup_file"} 33.0' in body

    def test_throughput_metrics_read_ops_tier_from_archive_tiers(self, tmp_path: Path) -> None:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.ops_write import record_ingest_attempt
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        db = tmp_path / "index.db"
        ops_db = db.with_name("ops.db")
        initialize_archive_database(ops_db, ArchiveTier.OPS)
        with sqlite3.connect(ops_db) as conn:
            record_ingest_attempt(
                conn,
                attempt_id="v1-completed",
                status="completed",
                started_at_ms=1_770_000_010_000,
                finished_at_ms=1_770_000_015_000,
                parsed_raw_count=20,
                materialized_count=10,
            )

        body = format_metrics(db)

        assert "polylogue_ingest_throughput_raw_rows_per_second 4.0" in body
        assert "polylogue_ingest_throughput_sessions_per_second 2.0" in body

    def test_messages_per_second_never_carries_the_session_numerator(self, tmp_path: Path) -> None:
        """No message-throughput series at all while the ledger counts no messages.

        ``ingest_attempts`` records raw rows and materialized sessions only, so
        the honest rendering is an absent series -- not the session numerator
        under a messages name, and not the sample-less ``name 0`` line, which
        would assert a measured zero.

        Anti-vacuity: re-emitting the name, with either ``materialized_count``
        or an empty sample list, makes a
        ``polylogue_ingest_throughput_messages_per_second`` line appear and
        this assertion fail.
        """
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.ops_write import record_ingest_attempt
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        db = tmp_path / "index.db"
        ops_db = db.with_name("ops.db")
        initialize_archive_database(ops_db, ArchiveTier.OPS)
        with sqlite3.connect(ops_db) as conn:
            record_ingest_attempt(
                conn,
                attempt_id="v1-completed",
                status="completed",
                started_at_ms=1_770_000_010_000,
                finished_at_ms=1_770_000_020_000,
                parsed_raw_count=100,
                materialized_count=50,
            )

        body = format_metrics(db)

        assert "polylogue_ingest_throughput_messages_per_second" not in body
        assert "polylogue_ingest_throughput_raw_rows_per_second 10.0" in body
        assert "polylogue_ingest_throughput_sessions_per_second 5.0" in body

    def test_live_ingest_metrics_read_ops_tier_from_archive_tiers(self, tmp_path: Path) -> None:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.ops_write import add_convergence_debt, record_ingest_attempt
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        db = tmp_path / "index.db"
        ops_db = db.with_name("ops.db")
        initialize_archive_database(ops_db, ArchiveTier.OPS)
        with sqlite3.connect(ops_db) as conn:
            record_ingest_attempt(
                conn,
                attempt_id="v1-running",
                status="running",
                started_at_ms=1_770_000_000_000,
                heartbeat_at_ms=1_770_000_001_000,
            )
            add_convergence_debt(
                conn,
                stage="session_profile",
                target_type="session_id",
                target_id="conv-1",
                attempts=1,
                created_at_ms=1_770_000_000_000,
            )

        body = format_metrics(db)

        assert 'polylogue_live_ingest_attempts_total{status="running"} 1' in body
        assert "polylogue_live_ingest_attempts_in_flight 1" in body
        assert 'polylogue_convergence_debt_count{stage="session_profile",status="failed"} 1' in body

    def test_fts_trigger_presence_partial(self, tmp_path: Path) -> None:
        body = format_metrics(self._make_db(tmp_path))
        # The fixture has only the message FTS surface, so only its
        # active triggers are exported.
        assert 'polylogue_fts_trigger_present{trigger="messages_fts_ai"} 1' in body
        assert "polylogue_fts_triggers_all_present 1" in body

    def test_embedding_backlog_and_latest_catchup_state(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        self._seed_catchup_run(db.with_name("ops.db"))
        body = format_metrics(db)
        assert 'polylogue_embedding_sessions{state="total"} 3' in body
        assert 'polylogue_embedding_sessions{state="embedded"} 1' in body
        assert 'polylogue_embedding_sessions{state="pending"} 2' in body
        assert 'polylogue_embedding_sessions{state="failed"} 1' in body
        assert 'polylogue_embedding_messages{state="embedded"} 2' in body
        assert "polylogue_embedding_coverage_percent 33.33333333333333" in body
        assert 'polylogue_embedding_status_state{status="partial"} 1' in body
        assert 'polylogue_embedding_status_state{status="complete"} 0' in body
        assert "polylogue_embedding_retrieval_ready 1" in body
        # Archive catch-up runs track outcome session counts but not rebuild
        # mode, planned/skipped breakdowns, or planned message counts.
        assert 'polylogue_embedding_latest_catchup_run_info{rebuild="false",status="interrupted"} 1' in body
        # Archive-tier receipts store the number actually visited, not the
        # pre-run pending window.  Planned is therefore omitted instead of
        # duplicating processed and implying a false 100% run.
        assert 'polylogue_embedding_latest_catchup_sessions{state="planned"}' not in body
        assert 'polylogue_embedding_latest_catchup_sessions{state="processed"} 3' in body
        assert 'polylogue_embedding_latest_catchup_sessions{state="embedded"} 2' in body
        assert 'polylogue_embedding_latest_catchup_sessions{state="skipped"} 0' in body
        assert 'polylogue_embedding_latest_catchup_sessions{state="failed"} 1' in body
        assert 'polylogue_embedding_latest_catchup_messages{state="planned"}' not in body
        assert 'polylogue_embedding_latest_catchup_messages{state="embedded"} 2' in body
        assert "polylogue_embedding_latest_catchup_estimated_cost_usd 0.003" in body

    def test_embedding_backlog_reads_archive_sessions_from_archive_tiers(self, tmp_path: Path) -> None:
        db = tmp_path / "archive.db"
        ops_db = tmp_path / "ops.db"
        with sqlite3.connect(db) as conn:
            conn.executescript(
                """
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    origin TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    raw_id TEXT
                );
                CREATE TABLE messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    message_type TEXT NOT NULL DEFAULT 'message',
                    material_origin TEXT NOT NULL DEFAULT 'human_authored',
                    word_count INTEGER NOT NULL DEFAULT 8
                );
                CREATE TABLE raw_sessions (
                    raw_id TEXT PRIMARY KEY,
                    origin TEXT NOT NULL,
                    parsed_at_ms INTEGER,
                    validated_at_ms INTEGER,
                    parse_error TEXT,
                    validation_status TEXT
                );
                CREATE TABLE embedding_status (
                    session_id TEXT PRIMARY KEY,
                    message_count_embedded INTEGER NOT NULL,
                    needs_reindex INTEGER NOT NULL,
                    error_message TEXT
                );
                CREATE TABLE message_embedding_refs (message_id TEXT PRIMARY KEY, vector_derivation_hash BLOB);
                INSERT INTO sessions (session_id, origin, message_count) VALUES
                    ('s-embedded', 'codex-session', 2),
                    ('s-pending', 'codex-session', 3),
                    ('s-missing-status', 'claude-code-session', 1);
                INSERT INTO messages (message_id, session_id) VALUES
                    ('s-embedded:m1', 's-embedded'),
                    ('s-embedded:m2', 's-embedded'),
                    ('s-pending:m1', 's-pending'),
                    ('s-pending:m2', 's-pending'),
                    ('s-pending:m3', 's-pending'),
                    ('s-missing-status:m1', 's-missing-status');
                INSERT INTO raw_sessions VALUES
                    ('raw-1', 'codex-session', 1767225700000, 1767225700000, NULL, NULL),
                    ('raw-2', 'claude-code-session', NULL, 1767225700000, 'bad json', 'failed');
                INSERT INTO embedding_status VALUES
                    ('s-embedded', 2, 0, NULL),
                    ('s-pending', 1, 1, 'provider timeout');
                INSERT INTO message_embedding_refs VALUES ('m-1', zeroblob(32)), ('m-2', zeroblob(32));
                """
            )
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.ops_write import upsert_embedding_catchup_run
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        initialize_archive_database(ops_db, ArchiveTier.OPS)
        with sqlite3.connect(ops_db) as conn:
            upsert_embedding_catchup_run(
                conn,
                run_id="v1-run",
                status="completed",
                started_at_ms=1_767_225_700_000,
                finished_at_ms=1_767_225_701_000,
                scanned_sessions=4,
                embedded_sessions=2,
                skipped_sessions=1,
                error_count=0,
                embedded_messages=4,
                estimated_cost_usd=0.001,
            )

        body = format_metrics(db)

        assert 'polylogue_embedding_sessions{state="total"} 3' in body
        assert 'polylogue_embedding_sessions{state="embedded"} 1' in body
        assert 'polylogue_embedding_sessions{state="pending"} 2' in body
        assert 'polylogue_embedding_sessions{state="failed"} 1' in body
        assert 'polylogue_embedding_messages{state="embedded"} 2' in body
        assert "polylogue_embedding_coverage_percent 33.33333333333333" in body
        assert 'polylogue_embedding_status_state{status="partial"} 1' in body
        assert 'polylogue_embedding_latest_catchup_run_info{rebuild="false",status="completed"} 1' in body
        assert 'polylogue_embedding_latest_catchup_sessions{state="processed"} 4' in body
        assert 'polylogue_embedding_latest_catchup_sessions{state="embedded"} 2' in body
        assert 'polylogue_embedding_latest_catchup_sessions{state="skipped"} 1' in body
        assert 'polylogue_embedding_latest_catchup_messages{state="embedded"} 4' in body
        assert "polylogue_embedding_latest_catchup_estimated_cost_usd 0.001" in body
        assert 'polylogue_archive_sessions_total{source="codex-session"} 2' in body
        assert 'polylogue_archive_messages_total{source="codex-session"} 5' in body
        assert 'polylogue_raw_records_total{state="total"} 2' in body
        assert 'polylogue_raw_records_total{state="errors"} 1' in body
        assert 'polylogue_raw_records_by_source{source="claude-code-session"} 1' in body

    def test_embedding_backlog_reads_split_embeddings_tier(self, tmp_path: Path) -> None:
        index_db = tmp_path / "index.db"
        embeddings_db = tmp_path / "embeddings.db"
        with sqlite3.connect(index_db) as conn:
            conn.executescript(
                """
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    origin TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    raw_id TEXT
                );
                CREATE TABLE messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    message_type TEXT NOT NULL DEFAULT 'message',
                    material_origin TEXT NOT NULL DEFAULT 'human_authored',
                    word_count INTEGER NOT NULL DEFAULT 8,
                    content_hash TEXT
                );
                INSERT INTO sessions (session_id, origin, message_count) VALUES
                    ('s-embedded', 'codex-session', 2),
                    ('s-pending', 'codex-session', 1);
                INSERT INTO messages (message_id, session_id, content_hash) VALUES
                    ('s-embedded:m1', 's-embedded', 'h1'),
                    ('s-embedded:m2', 's-embedded', 'h2'),
                    ('s-pending:m1', 's-pending', 'h3');
                """
            )
        with sqlite3.connect(embeddings_db) as conn:
            conn.executescript(
                """
                CREATE TABLE embedding_status (
                    session_id TEXT PRIMARY KEY,
                    message_count_embedded INTEGER NOT NULL,
                    needs_reindex INTEGER NOT NULL,
                    error_message TEXT
                );
                CREATE TABLE message_embeddings_meta (
                    message_id TEXT PRIMARY KEY,
                    content_hash TEXT,
                    model TEXT,
                    dimension INTEGER,
                    embedded_at_ms INTEGER
                );
                CREATE TABLE message_embedding_refs (message_id TEXT PRIMARY KEY, vector_derivation_hash BLOB);
                INSERT INTO embedding_status VALUES
                    ('s-embedded', 2, 0, NULL),
                    ('s-pending', 0, 1, NULL);
                INSERT INTO message_embedding_refs VALUES
                    ('s-embedded:m1', zeroblob(32)),
                    ('s-embedded:m2', zeroblob(32));
                INSERT INTO message_embeddings_meta VALUES
                    ('s-embedded:m1', 'h1', 'voyage-4', 1024, 1),
                    ('s-embedded:m2', 'h2', 'voyage-4', 1024, 2);
                """
            )

        body = format_metrics(index_db)

        assert 'polylogue_embedding_sessions{state="total"} 2' in body
        assert 'polylogue_embedding_sessions{state="embedded"} 1' in body
        assert 'polylogue_embedding_sessions{state="pending"} 1' in body
        assert 'polylogue_embedding_messages{state="embedded"} 2' in body
        assert "polylogue_embedding_coverage_percent 50.0" in body
        assert 'polylogue_embedding_status_state{status="partial"} 1' in body
        assert "polylogue_embedding_retrieval_ready 1" in body

    def test_raw_record_metrics_read_archive_source_tier_from_index_db(self, tmp_path: Path) -> None:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        index_db = tmp_path / "index.db"
        source_db = tmp_path / "source.db"
        initialize_archive_database(index_db, ArchiveTier.INDEX)
        initialize_runtime_source_fixture(source_db)
        with sqlite3.connect(index_db) as conn:
            conn.executemany(
                """
                INSERT INTO sessions (native_id, origin, raw_id, content_hash)
                VALUES (?, ?, ?, ?)
                """,
                [
                    ("indexed", "codex-session", "raw-ok", b"c" * 32),
                    ("orphan", "claude-code-session", "raw-missing", b"d" * 32),
                ],
            )
        with sqlite3.connect(source_db) as conn:
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, source_path, source_index, blob_hash, blob_size,
                    acquired_at_ms, parsed_at_ms, validated_at_ms, validation_status, parse_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("raw-ok", "codex-session", "/tmp/ok.jsonl", 0, b"a" * 32, 10, 1, 2, 3, None, None),
            )
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, source_path, source_index, blob_hash, blob_size,
                    acquired_at_ms, validated_at_ms, validation_status, parse_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("raw-bad", "claude-code-session", "/tmp/bad.jsonl", 0, b"b" * 32, 10, 1, 2, "failed", "bad"),
            )

        body = format_metrics(index_db)

        assert 'polylogue_raw_records_total{state="total"} 2' in body
        assert 'polylogue_raw_records_total{state="parsed"} 1' in body
        assert 'polylogue_raw_records_total{state="validated"} 2' in body
        assert 'polylogue_raw_records_total{state="errors"} 1' in body
        assert 'polylogue_raw_records_by_source{source="codex-session"} 1' in body
        assert 'polylogue_raw_records_by_source{source="claude-code-session"} 1' in body
        assert 'polylogue_archive_source_index_links_total{source="codex-session",state="acquired_raw"} 1' in body
        assert 'polylogue_archive_source_index_links_total{source="codex-session",state="indexed_raw"} 1' in body
        assert 'polylogue_archive_source_index_links_total{source="codex-session",state="pending_index"} 0' in body
        assert (
            'polylogue_archive_source_index_links_total{source="claude-code-session",state="pending_index"} 1' in body
        )
        assert (
            'polylogue_archive_source_index_links_total{source="claude-code-session",state="orphan_index_link"} 1'
            in body
        )

    def test_archive_messages_total_by_origin(self, tmp_path: Path) -> None:
        """#1629: per-origin message counts read the ``sessions.message_count``
        rollup column instead of a full ``GROUP BY`` scan over ``messages``.

        Archive file-sets store ``message_count`` on each ``sessions`` row and labels
        the metric by the ``origin`` family token (``source=`` label).
        """
        db = tmp_path / "archive.db"
        with sqlite3.connect(db) as conn:
            conn.executescript(
                """
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    origin TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    raw_id TEXT
                );
                INSERT INTO sessions (session_id, origin, message_count) VALUES
                    ('c1', 'claude-code-session', 100),
                    ('c2', 'claude-code-session', 50),
                    ('c3', 'codex-session', 30);
                """
            )

        body = format_metrics(db)

        assert 'polylogue_archive_sessions_total{source="claude-code-session"} 2' in body
        assert 'polylogue_archive_sessions_total{source="codex-session"} 1' in body
        assert 'polylogue_archive_messages_total{source="claude-code-session"} 150' in body
        assert 'polylogue_archive_messages_total{source="codex-session"} 30' in body

    def test_embedding_metrics_tolerate_partial_tables(self, tmp_path: Path) -> None:
        db = tmp_path / "archive.db"
        with sqlite3.connect(db) as conn:
            conn.executescript("""
                CREATE TABLE embedding_status (
                    session_id TEXT PRIMARY KEY,
                    needs_reindex INTEGER NOT NULL,
                    error_message TEXT
                );
                INSERT INTO embedding_status VALUES ('conv-1', 1, NULL);
            """)

        body = format_metrics(db)

        assert 'polylogue_embedding_sessions{state="pending"} 1' in body
        assert "polylogue_embedding_coverage_percent 0.0" in body
        assert 'polylogue_embedding_status_state{status="none"} 1' in body
        assert "polylogue_embedding_retrieval_ready 0" in body

    def test_embedding_coverage_percent_is_nan_when_no_sessions_are_eligible(self, tmp_path: Path) -> None:
        """polylogue-oitx: sessions exist but none are eligible for embedding
        (no ``embedding_status``/``message_embeddings_meta`` table and no
        ``messages`` table to derive pending state from) -- a genuine
        measurement gap, e.g. the embedding schema branch was never queried.
        Before the fix this fabricated ``100.0`` (read as "fully embedded"
        by a dashboard/alert) instead of the real "not yet measurable" state.
        """
        db = tmp_path / "archive.db"
        with sqlite3.connect(db) as conn:
            conn.executescript("""
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    origin TEXT NOT NULL DEFAULT 'codex-session',
                    message_count INTEGER NOT NULL DEFAULT 0,
                    raw_id TEXT
                );
                INSERT INTO sessions (session_id, origin, message_count) VALUES ('s1', 'codex-session', 1);
            """)

        body = format_metrics(db)

        assert 'polylogue_embedding_sessions{state="total"} 1' in body
        assert 'polylogue_embedding_sessions{state="pending"} 0' in body
        assert "polylogue_embedding_coverage_percent NaN" in body

    def test_embedding_coverage_percent_reports_real_full_coverage(self, tmp_path: Path) -> None:
        """A genuinely fully-embedded archive must still report a real 100.0,
        not be swept into the unmeasured/NaN case by the zero-denominator fix.
        """
        db = tmp_path / "archive.db"
        with sqlite3.connect(db) as conn:
            conn.executescript("""
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    origin TEXT NOT NULL DEFAULT 'codex-session',
                    message_count INTEGER NOT NULL DEFAULT 0,
                    raw_id TEXT
                );
                CREATE TABLE embedding_status (
                    session_id TEXT PRIMARY KEY,
                    needs_reindex INTEGER NOT NULL,
                    error_message TEXT
                );
                INSERT INTO sessions (session_id, origin, message_count) VALUES ('s1', 'codex-session', 1);
                INSERT INTO embedding_status VALUES ('s1', 0, NULL);
            """)

        body = format_metrics(db)

        assert 'polylogue_embedding_sessions{state="total"} 1' in body
        assert 'polylogue_embedding_sessions{state="embedded"} 1' in body
        assert "polylogue_embedding_coverage_percent 100.0" in body


# ---------------------------------------------------------------------------
# HTTP handler integration — mirrors test_health_contract.py harness
# ---------------------------------------------------------------------------


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"
    archive_query_executor = ThreadPoolExecutor(max_workers=1)


class _MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler(method: str, path: str) -> DaemonAPIHandler:
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast("DaemonAPIHTTPServer", _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.headers = cast("Message[str, str]", _MockHeaders({"Content-Length": "0"}))
    handler.rfile = BytesIO(b"")
    handler.wfile = BytesIO()
    return handler


def _capture_text(handler: DaemonAPIHandler) -> MagicMock:
    send_text = MagicMock()
    handler._send_text = send_text  # type: ignore[method-assign]
    return send_text


class TestMetricsEndpoint:
    def test_archive_collector_failure_keeps_process_counters_on_http_route(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.operations import daemon_metrics as metrics_module

        def fail_storage(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("unreadable /synthetic/private-input")

        monkeypatch.setattr(metrics_module, "_emit_archive_storage_metrics", fail_storage)
        monkeypatch.setattr(
            metrics_module,
            "diagnostic_snapshot",
            lambda: {"queued": 3, "dropped": 7, "failures": 2, "delivered": 11, "undrained": 1},
        )
        handler = _make_handler("GET", "/metrics")
        send_text = _capture_text(handler)
        handler.do_GET()
        status, body = send_text.call_args.args
        assert status == HTTPStatus.OK
        assert "polylogue_daemon_uptime_seconds " in body
        assert 'polylogue_diagnostic_delivery_total{outcome="dropped"} 7' in body
        assert 'polylogue_daemon_metrics_collection_available{group="archive_storage"} 0' in body
        assert "/synthetic/private-input" not in body

    def test_metrics_route_responds_200_with_prometheus_content_type(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        handler = _make_handler("GET", "/metrics")
        send_text = _capture_text(handler)
        handler.do_GET()
        send_text.assert_called_once()
        status = send_text.call_args.args[0]
        content_type = send_text.call_args.kwargs["content_type"]
        assert status == HTTPStatus.OK
        assert content_type == PROMETHEUS_CONTENT_TYPE

    def test_metrics_endpoint_is_unauthenticated(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """Same posture as /healthz/* — scrapers don't carry credentials."""
        handler = _make_handler("GET", "/metrics")
        handler.server.auth_token = "secret-token"
        send_text = _capture_text(handler)
        handler.do_GET()
        send_text.assert_called_once()
        status = send_text.call_args.args[0]
        assert status == HTTPStatus.OK

    def test_metrics_body_contains_expected_series(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        handler = _make_handler("GET", "/metrics")
        send_text = _capture_text(handler)
        handler.do_GET()
        body = send_text.call_args.args[1]
        parsed = _parse_exposition(body)
        missing = EXPECTED_SERIES - set(parsed.keys())
        assert not missing, f"missing series in /metrics response: {sorted(missing)}"


@pytest.mark.parametrize("parsed_files", [0, 1])
def test_partial_ingest_is_counted_and_contributes_only_its_measured_units(
    tmp_path: Path,
    parsed_files: int,
) -> None:
    from polylogue.operations import daemon_metrics
    from polylogue.storage.sqlite.archive_tiers.ops_write import record_ingest_attempt

    db = TestFormatMetricsReadsArchiveState()._make_db(tmp_path)
    ops_db = tmp_path / "ops.db"
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        for attempt_id, status, started, finished, raw, sessions in (
            ("partial", "completed_with_failures", 1000, 3000, parsed_files, 3),
            ("failed", "failed", 4000, 5000, 99, 99),
            ("interrupted", "interrupted", 6000, 7000, 99, 99),
        ):
            record_ingest_attempt(
                conn,
                attempt_id=attempt_id,
                status=status,
                started_at_ms=started,
                finished_at_ms=finished,
                parsed_raw_count=raw,
                materialized_count=sessions,
            )
    body = daemon_metrics.render_metrics(db)
    assert 'polylogue_live_ingest_attempts_total{status="completed_with_failures"} 1' in body
    assert 'polylogue_live_ingest_attempts_total{status="interrupted"} 1' in body
    assert 'polylogue_live_ingest_attempts_total{status="failed"} 1' in body
    assert 'polylogue_live_ingest_attempt_duration_seconds{quantile="mean"} 2.0' in body
    assert f"polylogue_ingest_throughput_raw_rows_per_second {parsed_files / 2}" in body
    assert "polylogue_ingest_throughput_sessions_per_second 1.5" in body


def test_newest_zero_partial_batch_replaces_older_positive_throughput(tmp_path: Path) -> None:
    from polylogue.operations import daemon_metrics
    from polylogue.storage.sqlite.archive_tiers.ops_write import record_ingest_attempt

    db = TestFormatMetricsReadsArchiveState()._make_db(tmp_path)
    ops_db = tmp_path / "ops.db"
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        for attempt_id, status, started, finished, count in (
            ("older", "completed", 1000, 3000, 8),
            ("zero", "completed_with_failures", 4000, 6000, 0),
            ("failed", "failed", 7000, 8000, 99),
            ("interrupted", "interrupted", 9000, 10000, 99),
        ):
            record_ingest_attempt(
                conn,
                attempt_id=attempt_id,
                status=status,
                started_at_ms=started,
                finished_at_ms=finished,
                parsed_raw_count=count,
                materialized_count=count,
            )
    body = daemon_metrics.render_metrics(db)
    assert "polylogue_ingest_throughput_raw_rows_per_second 0.0" in body
    assert "polylogue_ingest_throughput_sessions_per_second 0.0" in body
