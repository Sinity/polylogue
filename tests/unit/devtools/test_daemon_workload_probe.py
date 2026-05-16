from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from devtools.daemon_workload_probe import (
    REPORT_VERSION,
    compare,
    main,
    probe,
)
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.schema import _ensure_schema


def _seed_minimal_archive(db: Path, source: Path) -> str:
    """Seed an archive with one raw + conversation + completed live attempt."""

    source.write_text('{"a": 1}\n')
    conn = sqlite3.connect(db)
    try:
        _ensure_schema(conn)
        conn.execute(
            """
            INSERT INTO raw_conversations (
                raw_id, provider_name, source_path, blob_size, acquired_at
            ) VALUES ('raw-1', 'codex', ?, 0, '2026-01-01T00:00:00Z')
            """,
            (str(source),),
        )
        conn.execute(
            """
            INSERT INTO conversations (
                conversation_id, provider_name, provider_conversation_id,
                source_name, content_hash, version, raw_id
            ) VALUES ('conv-1', 'codex', 'provider-1', 'codex', 'hash-1', 1, 'raw-1')
            """
        )
        conn.commit()
    finally:
        conn.close()

    cursor = CursorStore(db)
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
    assert any("idx_raw_conv_source_path_raw_id" in item for item in source_plan["plan"])


def test_daemon_workload_probe_reports_missing_database(tmp_path: Path) -> None:
    payload = probe(tmp_path / "missing.sqlite")

    assert payload["ok"] is False
    assert payload["error"] == "database does not exist"
    # Even on failure, the envelope carries the version and capture timestamp
    # so compare() can report a structured error instead of crashing.
    assert payload["report_version"] == REPORT_VERSION
    assert isinstance(payload["captured_at"], str) and payload["captured_at"]


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
        "recent_attempts",
        "convergence_stage_timings",
        "boundary_table_counts",
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
    assert counts["raw_conversations"] == 1
    assert counts["conversations"] == 1
    assert counts["live_ingest_attempt"] == 1

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


def test_probe_reports_missing_fts_triggers(tmp_path: Path) -> None:
    db = tmp_path / "archive.sqlite"
    _seed_minimal_archive(db, tmp_path / "session.jsonl")

    # Drop one expected FTS sync trigger and verify the probe flags it.
    conn = sqlite3.connect(db)
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

    # Simulate a convergence cycle that added a second raw + conversation +
    # ran a second live attempt.
    second_source = tmp_path / "session-2.jsonl"
    second_source.write_text('{"b": 2}\n')
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            """
            INSERT INTO raw_conversations (
                raw_id, provider_name, source_path, blob_size, acquired_at
            ) VALUES ('raw-2', 'codex', ?, 0, '2026-01-02T00:00:00Z')
            """,
            (str(second_source),),
        )
        conn.execute(
            """
            INSERT INTO conversations (
                conversation_id, provider_name, provider_conversation_id,
                source_name, content_hash, version, raw_id
            ) VALUES ('conv-2', 'codex', 'provider-2', 'codex', 'hash-2', 1, 'raw-2')
            """
        )
        conn.commit()
    finally:
        conn.close()
    cursor = CursorStore(db)
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
    # New conversation and raw row land in the boundary counts.
    assert diff["boundary_table_counts"]["conversations"]["delta"] == 1
    assert diff["boundary_table_counts"]["raw_conversations"]["delta"] == 1
    assert diff["boundary_table_counts"]["live_ingest_attempt"]["delta"] == 1
    # No FTS regression.
    assert diff["fts_trigger_state"]["regressed"] == []
    # Sample size for completed-attempt timings grew by one.
    assert diff["convergence_stage_timings"]["sample_size_after"] == 2


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
    assert diff["boundary_table_counts"]["conversations"]["delta"] == 0
