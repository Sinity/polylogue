from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.core.enums import Origin
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    ArchiveCursorLagSample,
    ArchiveDaemonStageEvent,
    ArchiveEmbeddingCatchupRun,
    ArchiveOtlpSpan,
    OpsCompactState,
    add_convergence_debt,
    list_cursor_lag_samples,
    list_daemon_stage_events,
    list_embedding_catchup_runs,
    list_otlp_spans,
    read_compact_state,
    read_cursor_lag_sample,
    read_daemon_stage_event,
    read_embedding_catchup_run,
    read_otlp_span,
    record_cursor_lag_sample,
    record_daemon_stage_event,
    record_ingest_attempt,
    upsert_embedding_catchup_run,
    upsert_ingest_cursor,
    upsert_otlp_span,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    initialize_archive_tier(conn, ArchiveTier.OPS)
    return conn


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


def test_record_ingest_attempt_records_one_row(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")
    attempt_id = record_ingest_attempt(
        conn,
        status="running",
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
        status="running",
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
        status="running",
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
        status="running",
        started_at_ms=1_700_000_500,
        finished_at_ms=None,
        origin=Origin.CLAUDE_CODE_SESSION,
        scanned_sessions=2,
        embedded_messages=5,
        estimated_cost_usd=0.123,
        error_message=None,
    )
    run = read_embedding_catchup_run(conn, run_id)
    assert run == ArchiveEmbeddingCatchupRun(
        run_id="run-1",
        started_at_ms=1_700_000_500,
        finished_at_ms=None,
        status="running",
        origin=Origin.CLAUDE_CODE_SESSION.value,
        scanned_sessions=2,
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
        status="completed",
        started_at_ms=1_700_000_600,
        finished_at_ms=1_700_000_700,
    )
    upsert_embedding_catchup_run(
        conn,
        run_id="run-3",
        status="failed",
        started_at_ms=1_700_000_501,
        error_message="temporary issue",
    )

    completed_runs = list_embedding_catchup_runs(conn, status="completed")
    assert len(completed_runs) == 1
    assert completed_runs[0] == ArchiveEmbeddingCatchupRun(
        run_id="run-2",
        started_at_ms=1_700_000_600,
        finished_at_ms=1_700_000_700,
        status="completed",
        origin=None,
        scanned_sessions=0,
        embedded_messages=0,
        estimated_cost_usd=None,
        error_message=None,
    )


def test_upsert_otlp_span_writes_and_reads_row(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")

    upsert_otlp_span(
        conn,
        trace_id="trace-1",
        span_id="span-1",
        name="root",
        parent_span_id=None,
        origin=Origin.GEMINI_CLI_SESSION,
        kind="internal",
        started_at_ms=1_700_001_000,
        ended_at_ms=1_700_001_200,
        attributes_json='{"model":"mini"}',
        events_json='[{"name":"started"}]',
    )
    span = read_otlp_span(conn, "trace-1", "span-1")
    assert span == ArchiveOtlpSpan(
        trace_id="trace-1",
        span_id="span-1",
        parent_span_id=None,
        origin=Origin.GEMINI_CLI_SESSION.value,
        name="root",
        kind="internal",
        started_at_ms=1_700_001_000,
        ended_at_ms=1_700_001_200,
        attributes_json='{"model":"mini"}',
        events_json='[{"name":"started"}]',
    )


def test_upsert_otlp_span_refreshes_and_lists_by_trace(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "ops.db")

    upsert_otlp_span(
        conn,
        trace_id="trace-2",
        span_id="span-a",
        name="first",
        started_at_ms=1_700_001_100,
    )
    upsert_otlp_span(
        conn,
        trace_id="trace-2",
        span_id="span-b",
        name="second",
        started_at_ms=1_700_001_050,
    )
    upsert_otlp_span(
        conn,
        trace_id="trace-3",
        span_id="span-c",
        name="other",
        started_at_ms=1_700_001_200,
    )
    upsert_otlp_span(
        conn,
        trace_id="trace-2",
        span_id="span-a",
        name="first-refreshed",
        started_at_ms=1_700_001_150,
        ended_at_ms=1_700_001_400,
        attributes_json='{"updated":true}',
    )

    spans = list_otlp_spans(conn, trace_id="trace-2")
    assert len(spans) == 2
    assert spans[0].name == "first-refreshed"
    assert spans[0].span_id == "span-a"
    assert spans[0].started_at_ms == 1_700_001_150
    assert spans[1].span_id == "span-b"
    assert spans[1].started_at_ms == 1_700_001_050
