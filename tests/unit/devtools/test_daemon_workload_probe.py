from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from devtools.daemon_workload_probe import (
    REPORT_VERSION,
    compare,
    main,
    probe,
)
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    add_convergence_debt,
    record_cursor_lag_sample,
    record_daemon_stage_event,
    record_ingest_attempt,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _seed_minimal_archive(db: Path, source: Path) -> str:
    """Seed an archive with one raw + session + completed live attempt."""

    source.write_text('{"a": 1}\n')
    root = db.parent
    index_db = root / "index.db"
    source_db = root / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)

    source_conn = sqlite3.connect(source_db)
    try:
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
            ) VALUES ('raw-1', 'codex-session', 'provider-1', ?, ?, 0, 1770000000000)
            """,
            (str(source), b"r" * 32),
        )
        source_conn.commit()
    finally:
        source_conn.close()

    conn = sqlite3.connect(index_db)
    try:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, content_hash, created_at_ms, updated_at_ms
            ) VALUES ('provider-1', 'codex-session', 'raw-1', ?, 1770000000000, 1770000000000)
            """,
            (b"s" * 32,),
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, native_id, position, role, content_hash
            ) VALUES ('codex-session:provider-1', 'm1', 0, 'user', ?)
            """,
            (b"m" * 32,),
        )
        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, text
            ) VALUES ('codex-session:provider-1:m1', 'codex-session:provider-1', 0, 'text', 'hello')
            """
        )
        conn.commit()
    finally:
        conn.close()

    cursor = CursorStore(index_db)
    attempt_id = cursor.begin_ingest_attempt(paths=[source], input_bytes=10, queued_file_count=1)
    cursor.update_ingest_attempt(
        attempt_id,
        status="completed",
        phase="done",
        succeeded_file_count=1,
        source_payload_read_bytes=10,
        cursor_fingerprint_read_bytes=0,
    )
    return attempt_id


def _minimal_compare_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "report_version": REPORT_VERSION,
        "captured_at": "2026-06-01T00:00:00+00:00",
        "db_path": "/tmp/index.db",
        "attempt_counts": {},
        "storage_route_counts": {},
        "boundary_table_counts": {},
        "blob_lease_state": {},
        "gc_state": {},
        "fts_trigger_state": {},
        "convergence_debt": {},
        "convergence_stage_timings": {},
        "archive_tiers": {},
        "source_path_churn": [],
    }


def test_daemon_workload_probe_reports_attempts_and_plan_shape(tmp_path: Path) -> None:
    db = tmp_path / "archive.sqlite"
    source = tmp_path / "session.jsonl"
    _seed_minimal_archive(db, source)

    payload = probe(db)

    assert payload["ok"] is True
    assert payload["attempt_counts"]["total"] == 1
    assert payload["recent_attempts"][0]["read_amplification"] == 1.0
    source_plan = payload["query_plans"]["source_path_lookup"]
    assert source_plan["hazards"] == []
    assert any("idx_raw_sessions_source_path" in item for item in source_plan["plan"])


def test_daemon_workload_probe_reports_missing_database(tmp_path: Path) -> None:
    payload = probe(tmp_path / "missing.sqlite")

    assert payload["ok"] is False
    assert payload["error"] == "database does not exist"
    # Even on failure, the envelope carries the version and capture timestamp
    # so compare() can report a structured error instead of crashing.
    assert payload["report_version"] == REPORT_VERSION
    assert isinstance(payload["captured_at"], str) and payload["captured_at"]


def test_daemon_workload_probe_reports_ops_tier_from_db_anchor(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        record_ingest_attempt(
            conn,
            attempt_id="attempt-1",
            status="completed",
            phase="done",
            started_at_ms=1_770_000_000_000,
            heartbeat_at_ms=1_770_000_010_000,
            finished_at_ms=1_770_000_020_000,
            parsed_raw_count=3,
            materialized_count=2,
            source_paths_json='["/tmp/a.jsonl"]',
        )
        record_daemon_stage_event(
            conn,
            attempt_id="attempt-1",
            stage="done",
            status="completed",
            observed_at_ms=1_770_000_020_000,
            payload={
                "storage_route": "archive_full",
                "input_bytes": 100,
                "source_payload_read_bytes": 125,
                "cursor_fingerprint_read_bytes": 25,
                "parse_time_s": 1.5,
                "convergence_time_s": 2.5,
                "rss_current_mb": 42.0,
            },
        )
        add_convergence_debt(
            conn,
            stage="session_profile",
            target_type="session_id",
            target_id="s1",
            created_at_ms=1_770_000_030_000,
        )
        record_cursor_lag_sample(
            conn,
            sample_id="lag-1",
            family="claude-code-session",
            source_path="/tmp/a.jsonl",
            lag_ms=30_000,
            severity="warning",
            sampled_at_ms=1_770_000_040_000,
        )

    payload = probe(db)

    assert payload["ok"] is True
    assert payload["attempt_counts"]["total"] == 1
    assert payload["attempt_counts"]["completed"] == 1
    assert payload["storage_route_counts"]["archive_full"] == 1
    assert payload["storage_route_counts"]["unknown"] == 0
    assert payload["recent_attempts"][0]["read_amplification"] == 1.5
    assert payload["recent_attempts"][0]["storage_route"] == "archive_full"
    assert payload["recent_attempts"][0]["source_paths"] == ["/tmp/a.jsonl"]
    assert payload["convergence_stage_timings"]["sample_size"] == 1
    assert payload["daemon_resource_signal"] == {"available": True, "rss_current_mb": 42.0}
    assert payload["convergence_debt"]["failed_count"] == 1
    assert payload["cursor_lag_baselines"]["total_sample_count"] == 1
    assert payload["archive_tiers"]["present"] is True
    assert payload["archive_tiers"]["tiers"]["ops"]["exists"] is True
    assert payload["archive_tiers"]["tiers"]["ops"]["table_counts"]["ingest_attempts"] == 1
    assert payload["archive_tiers"]["missing_backup_required"] == ["source", "embeddings", "user"]
    layout = payload["archive_tiers"]["layout_readiness"]
    assert layout["archive_ready"] is False
    assert "missing_archive_tiers" in layout["blockers"]
    assert "missing_backup_required_tier:source" in layout["blockers"]
    assert "derived_readiness_unchecked:missing_index_tier" in layout["blockers"]


def test_daemon_workload_probe_reports_archive_tier_inventory(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    for tier in (
        ArchiveTier.SOURCE,
        ArchiveTier.INDEX,
        ArchiveTier.USER,
        ArchiveTier.OPS,
    ):
        initialize_archive_database(tmp_path / f"{tier.value}.db", tier)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms
            ) VALUES ('raw-1', 'codex-session', 'native-1', '/tmp/a.jsonl', ?, 10, 1)
            """,
            (b"x" * 32,),
        )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, content_hash
            ) VALUES ('native-1', 'codex-session', 'raw-1', ?)
            """,
            (b"y" * 32,),
        )
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, content_hash
            ) VALUES ('native-2', 'codex-session', 'raw-missing', ?)
            """,
            (b"z" * 32,),
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, native_id, position, role, content_hash
            ) VALUES ('codex-session:native-1', 'm1', 0, 'user', ?)
            """,
            (b"m" * 32,),
        )
        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, text
            ) VALUES ('codex-session:native-1:m1', 'codex-session:native-1', 0, 'text', 'hello')
            """
        )
        conn.execute(
            """
            INSERT INTO session_profiles (
                session_id, work_event_count, phase_count
            ) VALUES ('codex-session:native-1', 1, 0)
            """
        )
        conn.execute(
            """
            INSERT INTO session_work_events (
                session_id, position, work_event_type, summary, confidence,
                start_index, end_index
            ) VALUES ('codex-session:native-1', 0, 'implementation', 'built it', 0.8, 0, 0)
            """
        )
        conn.execute(
            """
            INSERT INTO insight_materialization (
                insight_type, session_id, materializer_version, materialized_at_ms
            ) VALUES ('session_profile', 'codex-session:native-1', 1, 1)
            """
        )
    with sqlite3.connect(tmp_path / "user.db") as conn:
        conn.execute(
            """
            INSERT INTO session_tags (session_id, tag, tag_source)
            VALUES ('codex-session:native-1', 'important', 'user')
            """
        )
        conn.execute(
            """
            INSERT INTO session_tags (session_id, tag, tag_source)
            VALUES ('missing-session', 'orphaned', 'user')
            """
        )
        conn.execute(
            """
            INSERT INTO assertions (
                assertion_id, target_ref, key, kind, value_json, created_at_ms, updated_at_ms
            ) VALUES (
                'correction-missing', 'insight:missing-session', 'tag_accept',
                'correction', '{"payload":{"tag":"orphaned"}}', 1, 1
            )
            """
        )

    payload = probe(db)
    tiers = payload["archive_tiers"]

    assert tiers["present"] is True
    assert tiers["complete"] is False
    assert tiers["present_count"] == 4
    assert tiers["observed_tier"] == "index"
    assert tiers["missing_backup_required"] == ["embeddings"]
    layout = tiers["layout_readiness"]
    assert layout["state"] == "not_archive_ready"
    assert layout["archive_ready"] is False
    assert "missing_archive_tiers" in layout["blockers"]
    assert "missing_backup_required_tier:embeddings" in layout["blockers"]
    assert "surface:raw_artifacts:missing_source_raw_sessions" in layout["blockers"]
    assert "surface:session_profiles:missing_profile_rows" in layout["blockers"]
    assert "surface:session_profiles:missing_session_profile_materialization" in layout["blockers"]
    assert "user_overlay_orphan_session_references" in layout["blockers"]
    assert layout["evidence"]["present_count"] == 4
    assert layout["evidence"]["blocked_surface_count"] >= 1
    assert tiers["user_overlay_orphans"]["checked"] is True
    assert tiers["user_overlay_orphans"]["total_orphan_session_references"] == 2
    assert tiers["user_overlay_orphans"]["orphan_session_reference_counts"]["session_tags"] == 1
    assert tiers["user_overlay_orphans"]["orphan_session_reference_counts"]["assertion_corrections"] == 1
    readiness = tiers["derived_readiness"]
    assert readiness["checked"] is True
    assert readiness["source_check_available"] is True
    assert readiness["counts"]["session_count"] == 2
    assert readiness["counts"]["raw_link_count"] == 2
    assert readiness["counts"]["missing_raw_session_count"] == 1
    assert readiness["counts"]["text_block_count"] == 1
    assert readiness["counts"]["messages_fts_count"] == 1
    assert readiness["counts"]["profile_row_count"] == 1
    assert readiness["counts"]["missing_profile_row_count"] == 1
    assert readiness["counts"]["work_event_row_count"] == 1
    assert readiness["counts"]["session_tag_count"] == 0
    assert readiness["counts"]["action_count"] == 0
    assert readiness["materialization_counts"]["session_profile"] == 1
    assert readiness["missing_materialization_counts"]["session_profile"] == 1
    assert readiness["missing_materialization_counts"]["work_events"] == 2
    assert readiness["ready"]["raw_links_ready"] is False
    assert readiness["ready"]["messages_fts_ready"] is True
    assert readiness["ready"]["profile_rows_ready"] is False
    surfaces = readiness["surface_readiness"]
    assert surfaces["archive_sessions"]["ready"] is True
    assert surfaces["archive_sessions"]["evidence"]["session_count"] == 2
    assert surfaces["raw_artifacts"]["ready"] is False
    assert surfaces["raw_artifacts"]["blockers"] == ["missing_source_raw_sessions"]
    assert surfaces["search"]["ready"] is True
    assert surfaces["session_profiles"]["ready"] is False
    assert "missing_profile_rows" in surfaces["session_profiles"]["blockers"]
    assert "missing_session_profile_materialization" in surfaces["session_profiles"]["blockers"]
    assert surfaces["timeline_work_events"]["ready"] is False
    assert surfaces["timeline_work_events"]["blockers"] == ["missing_work_events_materialization"]
    assert surfaces["tag_rollups"]["ready"] is True
    assert surfaces["tag_rollups"]["evidence"]["session_tag_count"] == 0
    assert surfaces["tool_usage"]["ready"] is True
    assert surfaces["tool_usage"]["evidence"]["action_count"] == 0
    assert tiers["schema_mismatches"] == []
    assert tiers["tiers"]["source"]["table_counts"]["raw_sessions"] == 1
    assert tiers["tiers"]["index"]["table_counts"]["sessions"] == 2
    assert tiers["tiers"]["user"]["table_counts"]["session_tags"] == 2
    assert tiers["tiers"]["embeddings"]["exists"] is False


def test_daemon_workload_probe_reports_layout_ready_for_clean_five_tier_archive(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    for tier in ArchiveTier:
        initialize_archive_database(tmp_path / f"{tier.value}.db", tier)

    payload = probe(db)

    layout = payload["archive_tiers"]["layout_readiness"]
    assert layout == {
        "state": "archive_ready",
        "archive_ready": True,
        "blockers": [],
        "evidence": {
            "present_count": 5,
            "expected_count": 5,
            "complete": True,
            "schema_mismatch_count": 0,
            "missing_backup_required_count": 0,
            "derived_readiness_checked": True,
            "derived_surface_count": 11,
            "blocked_surface_count": 0,
            "user_overlay_checked": True,
            "user_overlay_orphan_session_references": 0,
        },
    }


def test_daemon_workload_probe_reports_archive_source_path_churn(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    source_path = "/tmp/live-session.jsonl"
    for tier in (
        ArchiveTier.SOURCE,
        ArchiveTier.INDEX,
        ArchiveTier.OPS,
    ):
        initialize_archive_database(tmp_path / f"{tier.value}.db", tier)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms
            ) VALUES ('raw-full', 'codex-session', 'native-1', ?, 0, ?, 100, 1)
            """,
            (source_path, b"x" * 32),
        )
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms
            ) VALUES ('raw-append', 'codex-session', NULL, ?, -1, ?, 25, 2)
            """,
            (source_path, b"y" * 32),
        )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, content_hash
            ) VALUES ('native-1', 'codex-session', 'raw-full', ?)
            """,
            (b"z" * 32,),
        )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        record_ingest_attempt(
            conn,
            attempt_id="attempt-1",
            status="completed",
            phase="done",
            started_at_ms=1_770_000_000_000,
            heartbeat_at_ms=1_770_000_010_000,
            finished_at_ms=1_770_000_020_000,
            parsed_raw_count=2,
            materialized_count=1,
            source_paths_json=json.dumps([source_path]),
        )

    payload = probe(db)

    assert payload["ok"] is True
    assert payload["source_path_churn"] == [
        {
            "source_path": source_path,
            "storage_route": "archive_file_set",
            "raw_count": 2,
            "full_raw_count": 1,
            "append_raw_count": 1,
            "session_count": 1,
            "materialized_raw_count": 1,
            "orphan_raw_count": 1,
            "total_blob_bytes": 125,
            "latest_acquired_at": "1970-01-01T00:00:00.002000+00:00",
        }
    ]
    source_plan = payload["query_plans"]["source_path_lookup"]
    assert source_plan["storage_route"] == "archive_file_set"
    assert source_plan["hazards"] == []
    assert any("idx_raw_sessions_source_path" in item for item in source_plan["plan"])
    assert any("idx_sessions_raw_id" in item for item in source_plan["plan"])


def test_probe_payload_carries_stable_top_level_shape(tmp_path: Path) -> None:
    db = tmp_path / "archive.sqlite"
    _seed_minimal_archive(db, tmp_path / "session.jsonl")

    payload = probe(db)

    expected_keys = {
        "ok",
        "report_version",
        "captured_at",
        "db_path",
        "attempt_counts",
        "storage_route_counts",
        "recent_attempts",
        "convergence_stage_timings",
        "boundary_table_counts",
        "archive_tiers",
        "blob_lease_state",
        "gc_state",
        "fts_trigger_state",
        "daemon_resource_signal",
        "source_path_churn",
        "convergence_debt",
        "query_plans",
    }
    assert expected_keys.issubset(payload.keys())
    assert payload["report_version"] == REPORT_VERSION

    # Boundary counts cover the daemon-relevant tables.
    counts = payload["boundary_table_counts"]
    assert counts["raw_sessions"] == 1
    assert counts["sessions"] == 1
    assert counts["live_ingest_attempt"] == -1
    assert counts["ingest_attempts"] == 1

    # FTS trigger state reports the full expected set.
    fts = payload["fts_trigger_state"]
    assert fts["all_present"] is True
    assert set(fts["present"]) == set(fts["expected"])
    assert fts["missing"] == []

    # GC / lease tables are present after fresh schema init even if empty.
    assert payload["blob_lease_state"]["table_present"] is True
    assert payload["blob_lease_state"]["pending_lease_count"] == 0
    assert payload["gc_state"]["table_present"] is True
    assert payload["gc_state"]["high_water_generation"] == 0

    # The completed attempt produces a one-sample timing summary.
    timings = payload["convergence_stage_timings"]
    assert timings["sample_size"] == 1
    assert "mean" in timings["parse_time_s"]

    # Payload must be JSON-round-trippable (no datetime, set, Path, etc).
    assert json.loads(json.dumps(payload))["report_version"] == REPORT_VERSION


def test_probe_reads_ops_convergence_debt(tmp_path: Path) -> None:
    db = tmp_path / "archive.sqlite"
    _seed_minimal_archive(db, tmp_path / "session.jsonl")
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

    payload = probe(db)

    assert payload["convergence_debt"] == {
        "failed_count": 2,
        "by_stage": [{"stage": "session_profile", "failed_count": 2}],
    }


def test_probe_reads_ops_cursor_lag_baselines(tmp_path: Path) -> None:
    db = tmp_path / "archive.sqlite"
    _seed_minimal_archive(db, tmp_path / "session.jsonl")
    ops_db = db.with_name("ops.db")
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        record_cursor_lag_sample(
            conn,
            sample_id="lag-1",
            family="claude-code-session",
            source_path="/tmp/claude.jsonl",
            lag_ms=10_000,
            stuck_file_count=1,
            p50_lag_ms=10_000,
            p95_lag_ms=10_000,
            severity="warning",
            sampled_at_ms=1_770_000_000_000,
        )
        record_cursor_lag_sample(
            conn,
            sample_id="lag-2",
            family="claude-code-session",
            source_path="/tmp/claude.jsonl",
            lag_ms=20_000,
            stuck_file_count=2,
            p50_lag_ms=20_000,
            p95_lag_ms=20_000,
            severity="warning",
            sampled_at_ms=1_770_000_060_000,
        )

    payload = probe(db)

    assert payload["cursor_lag_baselines"] == {
        "table_present": True,
        "family_count": 1,
        "total_sample_count": 2,
        "families": [
            {
                "family": "claude-code-session",
                "sample_count": 2,
                "first_observed_at": "2026-02-02T02:40:00+00:00",
                "last_observed_at": "2026-02-02T02:41:00+00:00",
                "max_lag_s_seen": 20.0,
                "mean_lag_s": 15.0,
                "stuck_file_total": 3,
                "p50_lag_s": 15.0,
                "p95_lag_s": 19.5,
            }
        ],
    }


def test_probe_reports_missing_fts_triggers(tmp_path: Path) -> None:
    db = tmp_path / "archive.sqlite"
    _seed_minimal_archive(db, tmp_path / "session.jsonl")

    # Drop one expected FTS sync trigger and verify the probe flags it.
    conn = sqlite3.connect(db.with_name("index.db"))
    try:
        conn.execute("DROP TRIGGER IF EXISTS messages_fts_ai")
        conn.commit()
    finally:
        conn.close()

    payload = probe(db)
    fts = payload["fts_trigger_state"]
    assert fts["all_present"] is False
    assert "messages_fts_ai" in fts["missing"]


def test_compare_computes_structured_delta(tmp_path: Path) -> None:
    db = tmp_path / "archive.sqlite"
    source = tmp_path / "session.jsonl"
    _seed_minimal_archive(db, source)

    before_payload = probe(db)

    # Simulate a convergence cycle that added a second raw + session +
    # ran a second live attempt.
    second_source = tmp_path / "session-2.jsonl"
    second_source.write_text('{"b": 2}\n')
    source_conn = sqlite3.connect(db.with_name("source.db"))
    try:
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
            ) VALUES ('raw-2', 'codex-session', 'provider-2', ?, ?, 0, 1770086400000)
            """,
            (str(second_source), b"r" * 32),
        )
        source_conn.commit()
    finally:
        source_conn.close()

    conn = sqlite3.connect(db.with_name("index.db"))
    try:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, content_hash, raw_id, created_at_ms, updated_at_ms
            ) VALUES ('provider-2', 'codex-session', ?, 'raw-2', 1770086400000, 1770086400000)
            """,
            (b"t" * 32,),
        )
        conn.commit()
    finally:
        conn.close()
    cursor = CursorStore(db.with_name("index.db"))
    attempt_id = cursor.begin_ingest_attempt(paths=[second_source], input_bytes=20, queued_file_count=1)
    cursor.update_ingest_attempt(
        attempt_id,
        status="completed",
        phase="done",
        succeeded_file_count=1,
        source_payload_read_bytes=20,
        cursor_fingerprint_read_bytes=0,
    )

    after_payload = probe(db)

    diff = compare(before_payload, after_payload)
    assert diff["ok"] is True
    # Two new attempts (one new + the existing completed one — only the new
    # one increases the total, the completed/total counters both rise by 1).
    assert diff["attempt_counts"]["total"]["delta"] == 1
    assert diff["attempt_counts"]["completed"]["delta"] == 1
    assert diff["storage_route_counts"]["unknown"]["delta"] == 1
    # New session and raw row land in the boundary counts.
    assert diff["boundary_table_counts"]["sessions"]["delta"] == 1
    assert diff["boundary_table_counts"]["raw_sessions"]["delta"] == 1
    assert diff["boundary_table_counts"]["ingest_attempts"]["delta"] == 1
    # No FTS regression.
    assert diff["fts_trigger_state"]["regressed"] == []
    # Sample size for completed-attempt timings grew by one.
    assert diff["convergence_stage_timings"]["sample_size_after"] == 2


def test_compare_reports_archive_derived_readiness_deltas(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)

    before_payload = probe(db)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, content_hash
            ) VALUES ('native-1', 'codex-session', ?)
            """,
            (b"y" * 32,),
        )
    after_payload = probe(db)

    diff = compare(before_payload, after_payload)

    assert diff["ok"] is True
    readiness = diff["archive_tiers"]["derived_readiness"]
    layout = diff["archive_tiers"]["layout_readiness"]
    assert layout["archive_ready_before"] is False
    assert layout["archive_ready_after"] is False
    assert "surface:session_profiles:missing_profile_rows" in layout["introduced_blockers"]
    assert readiness["checked_before"] is True
    assert readiness["checked_after"] is True
    assert readiness["counts"]["session_count"]["delta"] == 1
    assert readiness["counts"]["missing_profile_row_count"]["delta"] == 1
    assert readiness["missing_materialization_counts"]["session_profile"]["delta"] == 1
    assert readiness["ready_before"]["profile_rows_ready"] is True
    assert readiness["ready_after"]["profile_rows_ready"] is False
    surfaces = readiness["surface_readiness"]
    assert surfaces["session_profiles"]["ready_before"] is True
    assert surfaces["session_profiles"]["ready_after"] is False
    assert surfaces["session_profiles"]["blockers_after"] == [
        "missing_profile_rows",
        "missing_session_profile_materialization",
    ]
    assert surfaces["session_profiles"]["evidence"]["missing_profile_row_count"]["delta"] == 1


def test_compare_reports_source_path_churn_deltas() -> None:
    before = _minimal_compare_payload()
    after = _minimal_compare_payload()
    before["source_path_churn"] = [
        {
            "source_path": "/tmp/live-session.jsonl",
            "storage_route": "archive_file_set",
            "raw_count": 2,
            "full_raw_count": 1,
            "append_raw_count": 1,
            "session_count": 1,
            "materialized_raw_count": 1,
            "orphan_raw_count": 1,
            "total_blob_bytes": 125,
            "latest_acquired_at": "2026-06-01T00:00:00+00:00",
        }
    ]
    after["source_path_churn"] = [
        {
            "source_path": "/tmp/live-session.jsonl",
            "storage_route": "archive_file_set",
            "raw_count": 3,
            "full_raw_count": 1,
            "append_raw_count": 2,
            "session_count": 3,
            "materialized_raw_count": 3,
            "orphan_raw_count": 0,
            "total_blob_bytes": 150,
            "latest_acquired_at": "2026-06-01T00:00:10+00:00",
        }
    ]

    diff = compare(before, after)

    assert diff["ok"] is True
    churn = diff["source_path_churn"]
    assert churn["path_count_before"] == 1
    assert churn["path_count_after"] == 1
    path = churn["paths"]["/tmp/live-session.jsonl"]
    assert path["storage_route_before"] == "archive_file_set"
    assert path["storage_route_after"] == "archive_file_set"
    assert path["counts"]["raw_count"]["delta"] == 1
    assert path["counts"]["append_raw_count"]["delta"] == 1
    assert path["counts"]["materialized_raw_count"]["delta"] == 2
    assert path["counts"]["orphan_raw_count"]["delta"] == -1


def test_compare_rejects_version_mismatch() -> None:
    before = {"ok": True, "report_version": REPORT_VERSION}
    after = {"ok": True, "report_version": REPORT_VERSION + 99}

    diff = compare(before, after)
    assert diff["ok"] is False
    assert "report_version mismatch" in diff["error"]


def test_compare_rejects_failed_input() -> None:
    before = {"ok": False, "report_version": REPORT_VERSION, "error": "db missing"}
    after = {"ok": True, "report_version": REPORT_VERSION}

    diff = compare(before, after)
    assert diff["ok"] is False
    assert "compare requires two successful" in diff["error"]


def test_cli_compare_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "archive.sqlite"
    _seed_minimal_archive(db, tmp_path / "session.jsonl")

    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(json.dumps(probe(db)))
    after_path.write_text(json.dumps(probe(db)))

    exit_code = main(["--compare", str(before_path), str(after_path), "--json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    # The probe imports may surface schema-bootstrap log lines on stdout in
    # some environments; the compare output itself is the trailing JSON
    # document.  Slice from the first '{' to the end.
    out = captured.out
    diff = json.loads(out[out.index("{") :])
    assert diff["ok"] is True
    # No deltas when comparing a snapshot against itself.
    assert diff["attempt_counts"]["total"]["delta"] == 0
    assert diff["boundary_table_counts"]["sessions"]["delta"] == 0
