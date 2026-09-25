from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import OperationStatus, Origin
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.ops import OPS_BENIGN_DDL_CONVERGENCE_PLAN, OPS_DDL
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    ROUTE_OBSERVATION_ROW_CAP,
    ArchiveCursorLagSample,
    ArchiveDaemonLifecycle,
    ArchiveDaemonStageEvent,
    ArchiveEmbeddingCatchupRun,
    ArchiveRouteObservation,
    OpsCompactState,
    add_convergence_debt,
    latest_daemon_lifecycle,
    list_cursor_lag_samples,
    list_daemon_stage_events,
    list_embedding_catchup_runs,
    list_mcp_calls,
    list_route_observations,
    read_compact_state,
    read_cursor_lag_sample,
    read_daemon_stage_event,
    read_embedding_catchup_run,
    record_cursor_lag_sample,
    record_daemon_lifecycle_heartbeat,
    record_daemon_lifecycle_signal,
    record_daemon_lifecycle_start,
    record_daemon_lifecycle_stop,
    record_daemon_stage_event,
    record_ingest_attempt,
    record_mcp_call,
    record_route_observation,
    upsert_embedding_catchup_run,
    upsert_ingest_cursor,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    initialize_archive_tier(conn, ArchiveTier.OPS)
    return conn


def test_existing_ops_db_converges_old_status_checks_and_preserves_rows(tmp_path: Path) -> None:
    """Same-version OPS bootstrap repairs old checks before interrupted writes."""
    ops_db = tmp_path / "ops.db"
    conn = sqlite3.connect(ops_db)
    try:
        conn.executescript(
            """
            CREATE TABLE ingest_attempts (
                attempt_id TEXT PRIMARY KEY,
                source_path TEXT,
                origin TEXT,
                status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed')),
                phase TEXT,
                storage_route TEXT,
                started_at_ms INTEGER NOT NULL,
                heartbeat_at_ms INTEGER,
                finished_at_ms INTEGER,
                parsed_raw_count INTEGER NOT NULL DEFAULT 0,
                materialized_count INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                source_paths_json TEXT NOT NULL DEFAULT '[]',
                outcome_code TEXT NOT NULL DEFAULT 'legacy_unknown',
                retryable INTEGER,
                evidence_ref TEXT,
                diagnostic TEXT,
                remediation TEXT
            ) STRICT;
            CREATE TABLE embedding_catchup_runs (
                run_id TEXT PRIMARY KEY,
                started_at_ms INTEGER NOT NULL,
                finished_at_ms INTEGER,
                status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
                origin TEXT,
                scanned_sessions INTEGER NOT NULL DEFAULT 0,
                embedded_sessions INTEGER NOT NULL DEFAULT 0,
                skipped_sessions INTEGER NOT NULL DEFAULT 0,
                error_count INTEGER NOT NULL DEFAULT 0,
                embedded_messages INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd REAL,
                error_message TEXT
            ) STRICT;
            PRAGMA user_version = 1;
            """
        )

        conn.execute(
            "INSERT INTO ingest_attempts (attempt_id, status, started_at_ms) VALUES ('legacy-attempt', 'completed', 1)"
        )
        conn.execute(
            "INSERT INTO embedding_catchup_runs "
            "(run_id, started_at_ms, status, embedded_messages) VALUES ('legacy-run', 2, 'cancelled', 4)"
        )
        conn.commit()
    finally:
        conn.close()

    initialize_archive_database(ops_db, ArchiveTier.OPS)
    initialize_archive_database(ops_db, ArchiveTier.OPS)

    with sqlite3.connect(ops_db) as conn:
        embedding_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'embedding_catchup_runs'"
        ).fetchone()[0]
        assert "status IN ('running', 'completed', 'failed', 'interrupted', 'completed_with_failures')" in embedding_sql
        assert "cancelled" not in embedding_sql
        assert conn.execute("SELECT status, embedded_messages FROM embedding_catchup_runs").fetchone() == (
            "interrupted",
            4,
        )
        assert conn.execute("SELECT status FROM ingest_attempts WHERE attempt_id = 'legacy-attempt'").fetchone() == (
            "completed",
        )

        record_ingest_attempt(conn, attempt_id="new-attempt", status=OperationStatus.INTERRUPTED, started_at_ms=3)
        upsert_embedding_catchup_run(conn, run_id="new-run", status=OperationStatus.INTERRUPTED, started_at_ms=4)

        assert conn.execute("SELECT status FROM ingest_attempts WHERE attempt_id = 'new-attempt'").fetchone() == (
            "interrupted",
        )
        assert conn.execute("SELECT status FROM embedding_catchup_runs WHERE run_id = 'new-run'").fetchone() == (
            "interrupted",
        )


def test_fresh_ops_schema_declares_daemon_event_lifecycle_indexes(tmp_path: Path) -> None:
    ops_db = tmp_path / "ops.db"
    initialize_archive_database(ops_db, ArchiveTier.OPS)

    with sqlite3.connect(ops_db) as conn:
        indexes = {row[1] for row in conn.execute("PRAGMA index_list('daemon_events')")}

    assert {"idx_daemon_events_kind_id", "idx_daemon_events_lifecycle"} <= indexes


def test_existing_ops_db_applies_daemon_event_index_convergence_plan(tmp_path: Path) -> None:
    ops_db = tmp_path / "ops.db"
    with sqlite3.connect(ops_db) as conn:
        conn.executescript(
            """
            CREATE TABLE daemon_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                kind TEXT NOT NULL,
                operation_id TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}'
            ) STRICT;
            PRAGMA user_version = 1;
            """
        )
        initialize_archive_tier(conn, ArchiveTier.OPS)
        indexes = {row[1] for row in conn.execute("PRAGMA index_list('daemon_events')")}

    assert {"idx_daemon_events_kind_id", "idx_daemon_events_lifecycle"} <= indexes
    assert {entry.name for entry in OPS_BENIGN_DDL_CONVERGENCE_PLAN} >= {
        "create_idx_daemon_events_kind_id",
        "create_idx_daemon_events_lifecycle",
    }


#: Objects an earlier release declared in ``OPS_DDL`` and a named commit later
#: removed from it. ``CREATE TABLE IF NOT EXISTS`` never drops, and the
#: disposable ops tier has no migration chain, so before the convergence plan
#: gained ``DROP TABLE`` entries these survived in every already-bootstrapped
#: ops.db forever. This fixture reproduces that shape.
_RETIRED_OPS_TABLES: dict[str, str] = {
    "slo_samples": "idx_slo_samples_label_time",
    "query_runs": "idx_query_runs_started",
    "otlp_spans": "idx_ops_otlp_spans_trace",
    "otlp_telemetry": "idx_ops_otlp_telemetry_received",
}


def test_ops_convergence_drops_retired_tables(tmp_path: Path) -> None:
    """A reopened pre-retirement ops.db loses exactly the undeclared tables.

    Anti-vacuity: deleting the four ``drop_*`` entries from
    ``OPS_BENIGN_DDL_CONVERGENCE_PLAN`` leaves every retired table and index
    in place and fails the first assertion. The retained half is the opposite
    pin: ``polylogue_ops_schema_state`` is live bootstrap-internal state that
    canonical ``OPS_DDL`` also does not declare, so a structural "drop every
    table OPS_DDL omits" sweep passes the first assertion and fails this one.
    """
    ops_db = tmp_path / "ops.db"
    initialize_archive_database(ops_db, ArchiveTier.OPS)

    with sqlite3.connect(ops_db) as conn:
        for table, index in _RETIRED_OPS_TABLES.items():
            conn.executescript(
                f"""
                CREATE TABLE {table} (row_id TEXT PRIMARY KEY, observed_at_ms INTEGER NOT NULL) STRICT;
                CREATE INDEX {index} ON {table}(observed_at_ms);
                INSERT INTO {table}(row_id, observed_at_ms) VALUES ('kept-by-a-stale-schema', 1);
                """
            )
        seeded = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert set(_RETIRED_OPS_TABLES) <= seeded, "fixture did not reproduce a pre-retirement ops.db"

    with sqlite3.connect(ops_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.OPS)
        objects = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}

    assert not objects & set(_RETIRED_OPS_TABLES), "canonical DDL no longer declares these tables"
    assert not objects & set(_RETIRED_OPS_TABLES.values()), "a dropped table takes its indexes with it"
    assert "polylogue_ops_schema_state" in objects, "live bootstrap-internal state is not a retirement target"
    assert {"ingest_cursor", "ingest_attempts", "daemon_lifecycle", "daemon_events"} <= objects


def test_ops_retirement_entries_are_not_declared() -> None:
    """No plan entry may drop a table canonical ``OPS_DDL`` still declares.

    Anti-vacuity: adding ``DROP TABLE IF EXISTS context_injection_ledger`` --
    a table ``OPS_DDL`` still declares -- fails the final assertion here.
    ``devtools gate schema-manifest`` checks each entry's SQL shape but cannot
    know which names are still live, so shape validity alone never catches it.
    (Some declared names fail harder and earlier: retiring ``ingest_cursor``
    breaks ``_ensure_ops_runtime_columns`` while the canonical schema
    inventory is being built, so that variant never reaches this assertion.)
    """
    declared = set(re.findall(r"CREATE TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(\w+)\s*\(", OPS_DDL))
    assert "ingest_cursor" in declared, "the guard is vacuous if OPS_DDL parsing found nothing"

    dropped = {
        match.group(1)
        for match in (re.match(r"DROP TABLE IF EXISTS (\w+)$", entry.sql) for entry in OPS_BENIGN_DDL_CONVERGENCE_PLAN)
        if match is not None
    }
    assert dropped, "the guard is vacuous while the plan retires nothing"
    assert not dropped & declared


def test_ops_upsert_ingest_cursor_updates_single_row(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")

    upsert_ingest_cursor(
        conn,
        source_path="/tmp/source-a.jsonl",
        updated_at_ms=1_700_000_000,
        origin=Origin.CODEX_SESSION,
        stat_size=128,
        byte_offset=10,
    )
    upsert_ingest_cursor(
        conn,
        source_path="/tmp/source-a.jsonl",
        updated_at_ms=1_700_000_001,
        origin=Origin.CODEX_SESSION,
        stat_size=256,
        byte_offset=20,
        failure_count=2,
        next_retry_at="2026-05-24T00:01:00+00:00",
        excluded=True,
    )

    row = conn.execute(
        "SELECT stat_size, byte_offset, failure_count, next_retry_at, excluded FROM ingest_cursor WHERE source_path = ?",
        ("/tmp/source-a.jsonl",),
    ).fetchone()
    assert row is not None
    assert row[0] == 256
    assert row[1] == 20
    assert row[2:] == (2, "2026-05-24T00:01:00+00:00", 1)
    assert conn.execute("SELECT COUNT(*) FROM ingest_cursor").fetchone()[0] == 1


def test_ops_lifecycle_checks_reject_non_lifecycle_statuses(tmp_path: Path) -> None:
    """The fresh disposable bootstrap DDL rejects values outside the lifecycle subset."""
    conn = _connect(tmp_path / "ops.db")

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ingest_attempts (attempt_id, status, started_at_ms) VALUES (?, ?, ?)",
            ("pending-attempt", OperationStatus.PENDING.value, 1),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO embedding_catchup_runs (run_id, status, started_at_ms) VALUES (?, ?, ?)",
            ("cancelled-run", "cancelled", 1),
        )


def test_ops_writers_reject_admission_only_statuses(tmp_path: Path) -> None:
    """Writer validation keeps admission states from bypassing the ledger contract."""
    conn = _connect(tmp_path / "ops.db")

    with pytest.raises(ValueError, match="ingest attempt status"):
        record_ingest_attempt(conn, attempt_id="pending-attempt", status=OperationStatus.PENDING, started_at_ms=1)
    with pytest.raises(ValueError, match="embedding catchup status"):
        upsert_embedding_catchup_run(conn, run_id="pending-run", status=OperationStatus.PENDING, started_at_ms=1)


def test_record_ingest_attempt_records_one_row(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")
    attempt_id = record_ingest_attempt(
        conn,
        status=OperationStatus.RUNNING,
        source_path="/tmp/source-a.jsonl",
        origin=Origin.CHATGPT_EXPORT,
        phase="planning",
        started_at_ms=1_700_000_010,
        heartbeat_at_ms=1_700_000_011,
        parsed_raw_count=7,
        materialized_count=3,
        source_paths_json='["/tmp/source-a.jsonl"]',
        storage_route="archive_append",
    )

    row = conn.execute(
        """
        SELECT status, phase, parsed_raw_count, source_paths_json, storage_route
        FROM ingest_attempts
        WHERE attempt_id = ?
        """,
        (attempt_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == "running"
    assert row[1] == "planning"
    assert row[4] == "archive_append"
    assert row[2] == 7
    assert row[3] == '["/tmp/source-a.jsonl"]'
    assert conn.execute("SELECT COUNT(*) FROM ingest_attempts").fetchone()[0] == 1


def test_daemon_lifecycle_writes_preserve_signal_and_heartbeat(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")

    record_daemon_lifecycle_start(conn, run_id="run-1", started_at_ms=100, details={"pid": 12})
    record_daemon_lifecycle_heartbeat(conn, run_id="run-1", heartbeat_at_ms=200)
    record_daemon_lifecycle_signal(conn, run_id="run-1", signal_name="SIGTERM", observed_at_ms=300)
    record_daemon_lifecycle_stop(conn, run_id="run-1", stopped_at_ms=400, exit_kind="signal")

    row = latest_daemon_lifecycle(conn)

    assert row == ArchiveDaemonLifecycle(
        run_id="run-1",
        started_at_ms=100,
        stopped_at_ms=400,
        last_heartbeat_at_ms=400,
        signal="SIGTERM",
        exit_kind="signal",
        details={"pid": 12},
    )


def test_add_convergence_debt_adds_or_refreshes_one_row(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")

    add_convergence_debt(
        conn,
        debt_id="debt-1",
        stage="parse",
        target_type="source_path",
        target_id="/tmp/source-a.jsonl",
        priority=11,
        attempts=1,
        last_error="temporary parser failure",
        created_at_ms=1_700_000_020,
        updated_at_ms=1_700_000_021,
    )
    add_convergence_debt(
        conn,
        debt_id="debt-2",
        stage="parse",
        target_type="source_path",
        target_id="/tmp/source-a.jsonl",
        priority=20,
        attempts=1,
        last_error="still failing",
        created_at_ms=1_700_000_022,
        updated_at_ms=1_700_000_023,
    )

    row = conn.execute(
        "SELECT priority, attempts, last_error FROM convergence_debt "
        "WHERE stage = ? AND target_type = ? AND target_id = ?",
        ("parse", "source_path", "/tmp/source-a.jsonl"),
    ).fetchone()
    assert row is not None
    assert row[0] == 20
    assert row[1] == 2
    assert row[2] == "still failing"
    assert conn.execute("SELECT COUNT(*) FROM convergence_debt").fetchone()[0] == 1


def test_ops_vocabularies_round_trip_and_reject_at_typed_and_sql_boundaries(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")

    for index, status in enumerate(("running", "completed", "failed", "interrupted", "completed_with_failures")):
        record_ingest_attempt(
            conn,
            attempt_id=f"run-status-{index}",
            status=status,
            started_at_ms=index,
        )
    assert {
        row[0] for row in conn.execute("SELECT status FROM ingest_attempts WHERE attempt_id LIKE 'run-status-%'")
    } == {"running", "completed", "failed", "interrupted", "completed_with_failures"}
    for index, status in enumerate(("running", "completed", "failed", "interrupted", "completed_with_failures")):
        upsert_embedding_catchup_run(conn, run_id=f"catchup-status-{index}", status=status, started_at_ms=index)
    assert {
        row[0] for row in conn.execute("SELECT status FROM embedding_catchup_runs WHERE run_id LIKE 'catchup-status-%'")
    } == {"running", "completed", "failed", "interrupted", "completed_with_failures"}
    with pytest.raises(ValueError):
        record_ingest_attempt(conn, attempt_id="run-status-invalid", status="cancelled", started_at_ms=5)
    with pytest.raises(ValueError):
        upsert_embedding_catchup_run(conn, run_id="catchup-status-invalid", status="cancelled", started_at_ms=6)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ingest_attempts (attempt_id, status, started_at_ms) "
            "VALUES ('run-status-sql-invalid', 'cancelled', 6)"
        )

    for status in ("failed", "deferred"):
        add_convergence_debt(
            conn,
            stage="round-trip",
            target_type="session",
            target_id=status,
            status=status,
            created_at_ms=10,
        )
    assert conn.execute("SELECT DISTINCT status FROM convergence_debt ORDER BY status").fetchall() == [
        ("deferred",),
        ("failed",),
    ]

    with pytest.raises(ValueError, match="convergence debt status"):
        add_convergence_debt(
            conn,
            stage="invalid",
            target_type="session",
            target_id="bad",
            status="retrying",
            created_at_ms=11,
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO convergence_debt (debt_id, stage, target_type, target_id, status, created_at_ms, updated_at_ms) "
            "VALUES ('bad', 'invalid', 'session', 'bad', 'retrying', 1, 1)"
        )

    for index, severity in enumerate(("info", "warning", "error", "critical")):
        record_cursor_lag_sample(
            conn,
            sample_id=f"severity-{index}",
            family="synthetic",
            source_path=None,
            lag_ms=1,
            severity=severity,
            sampled_at_ms=index,
        )
    assert {row[0] for row in conn.execute("SELECT severity FROM cursor_lag_samples")} == {
        "info",
        "warning",
        "error",
        "critical",
    }
    with pytest.raises(ValueError, match="cursor lag severity"):
        record_cursor_lag_sample(
            conn,
            sample_id="invalid-severity",
            family="synthetic",
            source_path=None,
            lag_ms=1,
            severity="fatal",
            sampled_at_ms=5,
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO cursor_lag_samples "
            "(sample_id, family, lag_ms, stuck_file_count, p50_lag_ms, p95_lag_ms, severity, sampled_at_ms) "
            "VALUES ('invalid-severity-sql', 'synthetic', 1, 1, 1, 1, 'fatal', 6)"
        )

    for index, status in enumerate(("ok", "error", "degraded", "timed_out", "unavailable")):
        record_route_observation(
            conn,
            observation_id=f"route-status-{index}",
            trace_id=f"trace-status-{index}",
            surface="cli",
            route="cli.test",
            started_at_ms=1_790_000_000_000 + index,
            duration_ms=1,
            status=status,
            daemon_path="daemon" if index % 2 == 0 else "direct",
        )
    assert {
        row[0]
        for row in conn.execute("SELECT status FROM route_observations WHERE observation_id LIKE 'route-status-%'")
    } == {"ok", "error", "degraded", "timed_out", "unavailable"}
    with pytest.raises(ValueError, match="route observation status"):
        record_route_observation(
            conn,
            observation_id="route-status-invalid",
            trace_id="trace-status-invalid",
            surface="cli",
            route="cli.test",
            started_at_ms=1_790_000_000_020,
            duration_ms=1,
            status="cancelled",
        )
    with pytest.raises(ValueError, match="route daemon path"):
        record_route_observation(
            conn,
            observation_id="route-path-invalid",
            trace_id="trace-path-invalid",
            surface="cli",
            route="cli.test",
            started_at_ms=1_790_000_000_021,
            duration_ms=1,
            status="ok",
            daemon_path="unreachable",
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO route_observations "
            "(observation_id, trace_id, surface, route, daemon_path, started_at_ms, duration_ms, status, sampled) "
            "VALUES ('route-path-sql-invalid', 'trace', 'cli', 'cli.test', 'unreachable', 22, 1, 'ok', 1)"
        )


def test_record_cursor_lag_sample_writes_reads_and_filters(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")

    sample_id = record_cursor_lag_sample(
        conn,
        sample_id="lag-1",
        family="claude-code-session",
        source_path="/tmp/source-a.jsonl",
        lag_ms=120_000,
        stuck_file_count=2,
        p50_lag_ms=90_000,
        p95_lag_ms=117_000,
        severity="warning",
        sampled_at_ms=1_700_000_040,
    )
    record_cursor_lag_sample(
        conn,
        sample_id="lag-2",
        family="chatgpt-export",
        source_path="/tmp/source-b.jsonl",
        lag_ms=5_000,
        severity="info",
        sampled_at_ms=1_700_000_050,
    )
    record_cursor_lag_sample(
        conn,
        sample_id="lag-1",
        family="claude-code-session",
        source_path="/tmp/source-a.jsonl",
        lag_ms=240_000,
        stuck_file_count=3,
        p50_lag_ms=180_000,
        p95_lag_ms=237_000,
        severity="error",
        sampled_at_ms=1_700_000_060,
    )

    assert read_cursor_lag_sample(conn, sample_id) == ArchiveCursorLagSample(
        sample_id="lag-1",
        family="claude-code-session",
        source_path="/tmp/source-a.jsonl",
        lag_ms=240_000,
        stuck_file_count=3,
        p50_lag_ms=180_000,
        p95_lag_ms=237_000,
        severity="error",
        sampled_at_ms=1_700_000_060,
    )
    assert list_cursor_lag_samples(conn, family="claude-code-session", source_path="/tmp/source-a.jsonl") == (
        ArchiveCursorLagSample(
            sample_id="lag-1",
            family="claude-code-session",
            source_path="/tmp/source-a.jsonl",
            lag_ms=240_000,
            stuck_file_count=3,
            p50_lag_ms=180_000,
            p95_lag_ms=237_000,
            severity="error",
            sampled_at_ms=1_700_000_060,
        ),
    )
    assert conn.execute("SELECT COUNT(*) FROM cursor_lag_samples").fetchone()[0] == 2


def test_record_daemon_stage_event_writes_reads_and_filters(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")

    event_id = record_daemon_stage_event(
        conn,
        event_id="stage-1",
        attempt_id="attempt-1",
        stage="parse",
        status=OperationStatus.RUNNING,
        observed_at_ms=1_700_000_070,
        payload={"queued": 3},
    )
    record_daemon_stage_event(
        conn,
        event_id="stage-2",
        attempt_id="attempt-1",
        stage="converge",
        status="completed",
        observed_at_ms=1_700_000_080,
        payload={"materialized": 2},
    )
    record_daemon_stage_event(
        conn,
        event_id="stage-1",
        attempt_id="attempt-1",
        stage="parse",
        status="completed",
        observed_at_ms=1_700_000_090,
        payload={"parsed": 3},
    )

    assert read_daemon_stage_event(conn, event_id) == ArchiveDaemonStageEvent(
        event_id="stage-1",
        attempt_id="attempt-1",
        stage="parse",
        status="completed",
        observed_at_ms=1_700_000_090,
        payload={"parsed": 3},
    )
    assert list_daemon_stage_events(conn, attempt_id="attempt-1", stage="parse") == (
        ArchiveDaemonStageEvent(
            event_id="stage-1",
            attempt_id="attempt-1",
            stage="parse",
            status="completed",
            observed_at_ms=1_700_000_090,
            payload={"parsed": 3},
        ),
    )
    assert conn.execute("SELECT COUNT(*) FROM daemon_stage_events").fetchone()[0] == 2


def test_read_compact_state_reads_one_row_per_ops_helper(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")

    upsert_ingest_cursor(
        conn,
        source_path="/tmp/source-a.jsonl",
        updated_at_ms=1_700_000_100,
        origin=Origin.CLAUDE_CODE_SESSION,
        stat_size=64,
    )
    attempt_id = record_ingest_attempt(
        conn,
        attempt_id="attempt-1",
        status=OperationStatus.RUNNING,
        source_path="/tmp/source-a.jsonl",
        origin=Origin.CLAUDE_CODE_SESSION,
        started_at_ms=1_700_000_101,
        heartbeat_at_ms=1_700_000_102,
    )
    add_convergence_debt(
        conn,
        debt_id="debt-compact",
        stage="convergence",
        target_type="source_path",
        target_id="/tmp/source-a.jsonl",
        priority=7,
        attempts=1,
        created_at_ms=1_700_000_103,
    )

    state = read_compact_state(conn)
    assert state == OpsCompactState(
        cursor_count=1,
        ingest_attempt_total=1,
        ingest_attempt_running=1,
        ingest_attempt_completed=0,
        ingest_attempt_failed=0,
        convergence_debt_count=1,
        latest_attempt_id=attempt_id,
        latest_attempt_status="running",
        latest_cursor_path="/tmp/source-a.jsonl",
        latest_debt_stage="convergence",
        latest_debt_priority=7,
    )


def test_upsert_embedding_catchup_run_writes_and_reads_row(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")

    run_id = upsert_embedding_catchup_run(
        conn,
        run_id="run-1",
        status=OperationStatus.RUNNING,
        started_at_ms=1_700_000_500,
        finished_at_ms=None,
        origin=Origin.CLAUDE_CODE_SESSION,
        scanned_sessions=2,
        embedded_sessions=1,
        skipped_sessions=1,
        error_count=1,
        embedded_messages=5,
        estimated_cost_usd=0.123,
        error_message=None,
    )
    run = read_embedding_catchup_run(conn, run_id)
    assert run == ArchiveEmbeddingCatchupRun(
        run_id="run-1",
        started_at_ms=1_700_000_500,
        finished_at_ms=None,
        status=OperationStatus.RUNNING,
        origin=Origin.CLAUDE_CODE_SESSION.value,
        scanned_sessions=2,
        embedded_sessions=1,
        skipped_sessions=1,
        error_count=1,
        embedded_messages=5,
        estimated_cost_usd=0.123,
        error_message=None,
    )
    assert conn.execute("SELECT COUNT(*) FROM embedding_catchup_runs").fetchone()[0] == 1


def test_upsert_embedding_catchup_run_refreshes_status_and_list_filters(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")

    upsert_embedding_catchup_run(
        conn,
        run_id="run-2",
        status="running",
        started_at_ms=1_700_000_501,
    )
    upsert_embedding_catchup_run(
        conn,
        run_id="run-2",
        status=OperationStatus.COMPLETED,
        started_at_ms=1_700_000_600,
        finished_at_ms=1_700_000_700,
        scanned_sessions=2,
        embedded_sessions=2,
        skipped_sessions=0,
        error_count=0,
    )
    upsert_embedding_catchup_run(
        conn,
        run_id="run-3",
        status=OperationStatus.FAILED,
        started_at_ms=1_700_000_501,
        error_message="temporary issue",
    )

    completed_runs = list_embedding_catchup_runs(conn, status=OperationStatus.COMPLETED)
    assert len(completed_runs) == 1
    assert completed_runs[0] == ArchiveEmbeddingCatchupRun(
        run_id="run-2",
        started_at_ms=1_700_000_600,
        finished_at_ms=1_700_000_700,
        status="completed",
        origin=None,
        scanned_sessions=2,
        embedded_sessions=2,
        skipped_sessions=0,
        error_count=0,
        embedded_messages=0,
        estimated_cost_usd=None,
        error_message=None,
    )


def test_record_mcp_call_writes_reads_and_filters_by_session(tmp_path: Path) -> None:
    """polylogue-7s57: durable MCP call-log round trip, queryable by session id."""
    conn = _connect(tmp_path / "ops.db")

    call_id = record_mcp_call(
        conn,
        call_id="call-1",
        tool_name="get_resume_brief",
        session_id="codex-session:abc",
        started_at_ms=1_700_002_000,
        finished_at_ms=1_700_002_040,
        success=True,
    )
    record_mcp_call(
        conn,
        call_id="call-2",
        tool_name="compose_context_preamble",
        session_id=None,
        started_at_ms=1_700_002_100,
        finished_at_ms=1_700_002_130,
        success=False,
        error_detail="RuntimeError",
    )
    record_mcp_call(
        conn,
        call_id="call-3",
        tool_name="get_resume_brief",
        session_id="claude-code-session:def",
        started_at_ms=1_700_002_200,
        finished_at_ms=1_700_002_260,
        success=True,
    )
    record_mcp_call(
        conn,
        call_id="call-4",
        tool_name="compare_sessions",
        session_ids=("codex-session:abc", "claude-code-session:def"),
        started_at_ms=1_700_002_300,
        finished_at_ms=1_700_002_340,
        success=True,
    )

    assert call_id == "call-1"
    by_session = list_mcp_calls(conn, session_id="codex-session:abc")
    assert [entry.call_id for entry in by_session] == ["call-4", "call-1"]

    by_tool = list_mcp_calls(conn, tool_name="get_resume_brief")
    assert [entry.call_id for entry in by_tool] == ["call-3", "call-1"]

    assert conn.execute("SELECT COUNT(*) FROM mcp_call_log").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM mcp_call_session_refs").fetchone()[0] == 4
    assert {row[0] for row in conn.execute("SELECT DISTINCT relation FROM mcp_call_session_refs")} == {
        "primary",
        "member",
    }
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO mcp_call_session_refs (call_id, session_id, relation) "
            "VALUES ('call-1', 'invalid-session', 'related')"
        )


def test_record_route_observation_writes_reads_and_filters(tmp_path: Path) -> None:
    """polylogue-jtwu: route-latency evidence round trip, queryable by surface/route."""
    conn = _connect(tmp_path / "ops.db")

    observation_id = record_route_observation(
        conn,
        observation_id="obs-1",
        trace_id="trace-1",
        surface="cli",
        route="cli.status",
        verb="compact",
        daemon_path="direct",
        started_at_ms=1_700_003_000,
        duration_ms=384,
        status="ok",
        git_head="abc123def456",
        archive_epoch="epoch-1",
        attributes={"daemon_reachable": False},
    )
    record_route_observation(
        conn,
        observation_id="obs-2",
        trace_id="trace-2",
        surface="mcp",
        route="mcp.status.coordination",
        verb="detail",
        started_at_ms=1_700_003_500,
        duration_ms=5200,
        status="degraded",
        attributes={"archive_evidence_degraded": True},
    )
    record_route_observation(
        conn,
        observation_id="obs-3",
        trace_id="trace-3",
        surface="cli",
        route="cli.agents.status",
        started_at_ms=1_700_004_000,
        duration_ms=645,
        status="ok",
    )

    assert observation_id == "obs-1"

    by_surface = list_route_observations(conn, surface="cli")
    assert [row.observation_id for row in by_surface] == ["obs-3", "obs-1"]

    by_route = list_route_observations(conn, route="mcp.status.coordination")
    assert by_route == (
        ArchiveRouteObservation(
            observation_id="obs-2",
            trace_id="trace-2",
            surface="mcp",
            route="mcp.status.coordination",
            verb="detail",
            daemon_path=None,
            phase="total",
            started_at_ms=1_700_003_500,
            duration_ms=5200,
            status="degraded",
            git_head=None,
            archive_epoch=None,
            attributes={"archive_evidence_degraded": True},
            sampled=True,
        ),
    )

    since = list_route_observations(conn, since_ms=1_700_003_600)
    assert [row.observation_id for row in since] == ["obs-3"]

    read = list_route_observations(conn, surface="cli", route="cli.status")
    assert read
    assert read[0].daemon_path == "direct"
    assert read[0].git_head == "abc123def456"
    assert read[0].archive_epoch == "epoch-1"
    assert read[0].attributes == {"daemon_reachable": False}


def test_record_route_observation_prunes_by_retention_window(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")
    from polylogue.storage.sqlite.archive_tiers.ops_write import ROUTE_OBSERVATION_RETENTION_MS

    old_started_ms = 10_000_000_000  # far enough in the past to be pruned by the next write
    record_route_observation(
        conn,
        observation_id="old-1",
        trace_id="t-old",
        surface="cli",
        route="cli.status",
        started_at_ms=old_started_ms,
        duration_ms=100,
        status="ok",
    )
    assert conn.execute("SELECT COUNT(*) FROM route_observations").fetchone()[0] == 1

    record_route_observation(
        conn,
        observation_id="new-1",
        trace_id="t-new",
        surface="cli",
        route="cli.status",
        started_at_ms=old_started_ms + ROUTE_OBSERVATION_RETENTION_MS + 1,
        duration_ms=100,
        status="ok",
    )
    remaining = list_route_observations(conn)
    assert [row.observation_id for row in remaining] == ["new-1"]


def test_record_route_observation_caps_row_count(tmp_path: Path) -> None:
    """The cap holds once the table is over it.

    Seeded in bulk rather than through 20,005 individual calls. Each call commits
    its own transaction, measured at 12.9 ms of fsync apiece -- 258s to fill the
    table, which is why this test used to exceed its 120s timeout. That cost is a
    property of the write path, not of the cap logic under test here, and driving
    it 20,000 times measured the filesystem instead of the behaviour.
    """
    conn = _connect(tmp_path / "ops.db")
    base_ms = 1_700_000_000_000
    seeded = ROUTE_OBSERVATION_ROW_CAP + 4
    with conn:
        conn.executemany(
            "INSERT INTO route_observations (observation_id, trace_id, surface, route,"
            " started_at_ms, duration_ms, status, sampled) VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            [(f"obs-{i}", f"t-{i}", "cli", "cli.status", base_ms + i, 10, "ok") for i in range(seeded)],
        )

    # One real observation through the production path trips the cap.
    record_route_observation(
        conn,
        observation_id=f"obs-{seeded}",
        trace_id=f"t-{seeded}",
        surface="cli",
        route="cli.status",
        started_at_ms=base_ms + seeded,
        duration_ms=10,
        status="ok",
    )

    row_count = int(conn.execute("SELECT COUNT(*) FROM route_observations").fetchone()[0])
    assert row_count <= ROUTE_OBSERVATION_ROW_CAP
    # The oldest rows are the ones dropped -- the newest observation always survives.
    newest = list_route_observations(conn, limit=1)
    assert newest[0].observation_id == f"obs-{seeded}"
    assert conn.execute("SELECT 1 FROM route_observations WHERE observation_id = 'obs-0'").fetchone() is None


def test_reopening_a_current_ops_db_writes_nothing(tmp_path: Path) -> None:
    """A converged ops database is opened read-only by every later initializer.

    Anti-vacuity: restoring the unconditional DELETE/INSERT in
    ``_record_ops_schema_state`` commits a transaction on reopen and moves
    ``PRAGMA data_version`` as seen from the observer connection.
    """
    ops_db = tmp_path / "ops.db"
    initialize_archive_database(ops_db, ArchiveTier.OPS)

    observer = sqlite3.connect(ops_db)
    try:
        before = observer.execute("PRAGMA data_version").fetchone()[0]
        observer.execute("SELECT count(*) FROM polylogue_ops_schema_state").fetchone()
        initialize_archive_database(ops_db, ArchiveTier.OPS)
        after = observer.execute("PRAGMA data_version").fetchone()[0]
    finally:
        observer.close()

    assert after == before


def test_route_observation_writer_does_not_fsync_per_observation(tmp_path: Path) -> None:
    """polylogue-5lfcr: best-effort telemetry must not pay a durability fsync.

    ``ops.db`` is the disposable tier and ``record_route_observation``'s own
    docstring contrasts it with the durable, outbox-delivered
    ``record_mcp_call``. A synchronous commit per observation (12.88 ms
    measured) sits on the hot path of every route it measures.

    Anti-vacuity: drop the ``PRAGMA synchronous = OFF`` from
    ``open_observation_connection`` and the pragma read returns 2 (FULL).
    """
    from polylogue.operations.route_observation import open_observation_connection

    ops_db = tmp_path / "ops.db"
    ops_db.touch()
    conn = open_observation_connection(ops_db)
    try:
        assert int(conn.execute("PRAGMA synchronous").fetchone()[0]) == 0
    finally:
        conn.close()


def test_route_observation_row_is_visible_to_another_reader_after_the_call(tmp_path: Path) -> None:
    """Dropping the fsync must not drop the commit: the row is readable at once."""
    from polylogue.operations.route_observation import open_observation_connection

    ops_db = tmp_path / "ops.db"
    _connect(ops_db).close()

    writer = open_observation_connection(ops_db)
    try:
        record_route_observation(
            writer,
            trace_id="t-visible",
            surface="cli",
            route="find",
            started_at_ms=1_000,
            duration_ms=5,
            status="ok",
        )
    finally:
        writer.close()

    reader = sqlite3.connect(ops_db)
    try:
        assert reader.execute("SELECT COUNT(*) FROM route_observations WHERE trace_id = 't-visible'").fetchone()[0] == 1
    finally:
        reader.close()
