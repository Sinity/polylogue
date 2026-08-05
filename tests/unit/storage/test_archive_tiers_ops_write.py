from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import OperationStatus, Origin
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.ops import OPS_BENIGN_DDL_CONVERGENCE_PLAN
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    ROUTE_OBSERVATION_ROW_CAP,
    ArchiveCursorLagSample,
    ArchiveDaemonLifecycle,
    ArchiveDaemonStageEvent,
    ArchiveEmbeddingCatchupRun,
    ArchiveMcpCallLogEntry,
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
    read_mcp_call,
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
        assert "status IN ('running', 'completed', 'failed', 'interrupted')" in embedding_sql
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
    assert len(OPS_BENIGN_DDL_CONVERGENCE_PLAN) == 2


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

    with pytest.raises(ValueError, match="not a run lifecycle status"):
        record_ingest_attempt(conn, attempt_id="pending-attempt", status=OperationStatus.PENDING, started_at_ms=1)
    with pytest.raises(ValueError, match="not a run lifecycle status"):
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
    assert read_mcp_call(conn, "call-1") == ArchiveMcpCallLogEntry(
        call_id="call-1",
        tool_name="get_resume_brief",
        session_id="codex-session:abc",
        started_at_ms=1_700_002_000,
        finished_at_ms=1_700_002_040,
        duration_ms=40,
        success=True,
        error_detail=None,
    )
    assert read_mcp_call(conn, "call-2") == ArchiveMcpCallLogEntry(
        call_id="call-2",
        tool_name="compose_context_preamble",
        session_id=None,
        started_at_ms=1_700_002_100,
        finished_at_ms=1_700_002_130,
        duration_ms=30,
        success=False,
        error_detail="RuntimeError",
    )

    by_session = list_mcp_calls(conn, session_id="codex-session:abc")
    assert [entry.call_id for entry in by_session] == ["call-4", "call-1"]

    by_tool = list_mcp_calls(conn, tool_name="get_resume_brief")
    assert [entry.call_id for entry in by_tool] == ["call-3", "call-1"]

    assert conn.execute("SELECT COUNT(*) FROM mcp_call_log").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM mcp_call_session_refs").fetchone()[0] == 4


def test_read_mcp_call_missing_id_raises_key_error(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")
    with pytest.raises(KeyError):
        read_mcp_call(conn, "does-not-exist")


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
    conn = _connect(tmp_path / "ops.db")
    base_ms = 1_700_000_000_000
    for i in range(ROUTE_OBSERVATION_ROW_CAP + 5):
        record_route_observation(
            conn,
            observation_id=f"obs-{i}",
            trace_id=f"t-{i}",
            surface="cli",
            route="cli.status",
            started_at_ms=base_ms + i,
            duration_ms=10,
            status="ok",
        )
    row_count = int(conn.execute("SELECT COUNT(*) FROM route_observations").fetchone()[0])
    assert row_count <= ROUTE_OBSERVATION_ROW_CAP
    # The oldest rows are the ones dropped -- the newest observation always survives.
    newest = list_route_observations(conn, limit=1)
    assert newest[0].observation_id == f"obs-{ROUTE_OBSERVATION_ROW_CAP + 4}"
