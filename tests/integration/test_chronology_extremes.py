"""Integration tests for timestamp edge cases during ingestion (E4-E5).

Tests that extreme timestamps, mixed formats, and missing values are handled correctly.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace
from tests.infra.large_batches import generate_timestamp_patterns, write_jsonl_file

pytestmark = [pytest.mark.integration, pytest.mark.chaos]


# =============================================================================
# Timestamp Edge Case Tests
# =============================================================================


def test_epoch_near_zero_timestamps_ingested(tmp_path):
    """Test that timestamps near Unix epoch (1970) are ingested correctly."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    # Generate epoch-near records
    patterns = generate_timestamp_patterns()
    epoch_records = patterns["epoch_near_zero"]

    # Write to JSONL
    jsonl_path = inbox / "epoch_near.jsonl"
    lines = [json.dumps(record) for record in epoch_records]
    write_jsonl_file(jsonl_path, lines)

    # Run ingestion
    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    # Verify records were ingested
    db_path = workspace["paths"]["db_path"]
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        assert count == 5, f"Expected 5 epoch records, got {count}"


def test_y2038_adjacent_timestamps_ingested(tmp_path):
    """Test that timestamps near Unix Y2K38 boundary (2038) are ingested."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    # Generate Y2038-adjacent records
    patterns = generate_timestamp_patterns()
    y2038_records = patterns["y2038_adjacent"]

    jsonl_path = inbox / "y2038.jsonl"
    lines = [json.dumps(record) for record in y2038_records]
    write_jsonl_file(jsonl_path, lines)

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        assert count == 5, f"Expected 5 Y2038 records, got {count}"


def test_far_future_timestamps_ingested(tmp_path):
    """Test that far-future timestamps (year 2065+) are accepted."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    # Generate far-future records
    patterns = generate_timestamp_patterns()
    future_records = patterns["far_future"]

    jsonl_path = inbox / "far_future.jsonl"
    lines = [json.dumps(record) for record in future_records]
    write_jsonl_file(jsonl_path, lines)

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        assert count == 5, f"Expected 5 far-future records, got {count}"


# =============================================================================
# Mixed Timestamp Format Tests (E4)
# =============================================================================


def test_mixed_timestamp_formats_coexist(tmp_path):
    """Test that ISO, epoch float, and epoch string formats coexist in same file.

    Records may have:
    - ISO 8601 strings: "2024-01-15T10:30:00+00:00"
    - Epoch floats: 1705312200.0
    - Epoch strings: "1705312260"
    """
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    # Generate mixed format records
    patterns = generate_timestamp_patterns()
    mixed_records = patterns["mixed_formats"]

    jsonl_path = inbox / "mixed_formats.jsonl"
    lines = [json.dumps(record) for record in mixed_records]
    write_jsonl_file(jsonl_path, lines)

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        # Should ingest all 3 despite different timestamp formats
        assert count == 3, f"Expected 3 records with mixed formats, got {count}"


# =============================================================================
# Missing Timestamp Tests (E5)
# =============================================================================


def test_missing_timestamps_handled(tmp_path):
    """Test that records without timestamps coexist with timestamped ones.

    Records may be missing the timestamp field entirely, and this should
    not block ingestion of records that have timestamps.
    """
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    # Generate mix of present and absent timestamps
    patterns = generate_timestamp_patterns()
    missing_ts_records = patterns["missing_timestamps"]

    jsonl_path = inbox / "missing_ts.jsonl"
    lines = [json.dumps(record) for record in missing_ts_records]
    write_jsonl_file(jsonl_path, lines)

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        # Should ingest all 3 (the 2 with timestamps and the 1 without)
        assert count == 3, f"Expected 3 records (mix of with/without TS), got {count}"


# =============================================================================
# Chronological Ordering Tests
# =============================================================================


def test_chronological_ordering_preserved(tmp_path):
    """Test that output ordering respects input timestamp order.

    After ingestion, records should be retrievable in a consistent order
    (typically chronological by timestamp, or by insertion order if no timestamp).
    """
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    # Create records with known timestamps (1 hour apart)
    import time
    base_time = time.time()

    records = []
    for i in range(10):
        records.append({
            "type": "message",
            "role": "user" if i % 2 == 0 else "assistant",
            "id": f"msg_{i:02d}",
            "timestamp": (base_time + i * 3600),  # Epoch floats, 1 hour apart
            "content": [{
                "type": "input_text" if i % 2 == 0 else "output_text",
                "text": f"Message {i}",
            }],
        })

    jsonl_path = inbox / "chronological.jsonl"
    lines = [json.dumps(record) for record in records]
    write_jsonl_file(jsonl_path, lines)

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    with sqlite3.connect(db_path) as conn:
        # Query messages in order
        cursor = conn.execute(
            "SELECT id FROM messages ORDER BY timestamp ASC",
        )
        ids = [row[0] for row in cursor.fetchall()]

    # Verify we got all 10 records
    assert len(ids) == 10, f"Expected 10 records, got {len(ids)}"

    # Verify ordering (should match input order since timestamps are sequential)
    expected_ids = [f"msg_{i:02d}" for i in range(10)]
    assert ids == expected_ids, f"Order mismatch: got {ids}, expected {expected_ids}"


def test_tomorrow_timestamps_ingested(tmp_path):
    """Test that future timestamps (tomorrow) are accepted without error."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    # Generate tomorrow's records
    patterns = generate_timestamp_patterns()
    tomorrow_records = patterns["tomorrow"]

    jsonl_path = inbox / "tomorrow.jsonl"
    lines = [json.dumps(record) for record in tomorrow_records]
    write_jsonl_file(jsonl_path, lines)

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        assert count == 5, f"Expected 5 tomorrow records, got {count}"


# =============================================================================
# Comprehensive Timestamp Robustness Test
# =============================================================================


def test_all_timestamp_patterns_in_single_inbox(tmp_path):
    """Integration test: all timestamp patterns in one inbox."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    patterns = generate_timestamp_patterns()

    # Write all patterns to separate JSONL files
    for pattern_name, records in patterns.items():
        jsonl_path = inbox / f"{pattern_name}.jsonl"
        lines = [json.dumps(record) for record in records]
        write_jsonl_file(jsonl_path, lines)

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]

    # Expected: epoch_near_zero (5) + y2038 (5) + far_future (5) + tomorrow (5) + mixed_formats (3) + missing_timestamps (3)
    # Total: 26 records
    expected = 5 + 5 + 5 + 5 + 3 + 3
    assert count == expected, f"Expected {expected} records across all patterns, got {count}"
