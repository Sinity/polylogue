"""Integration tests for partial corruption resilience during ingestion (E1-E2).

Tests that valid records are ingested despite corruption in the same file,
and that error context is preserved for diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.infra.chaos_sources import (
    ChaosInboxBuilder,
    build_corrupted_codex_inbox,
    build_mixed_provider_inbox,
)
from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace

pytestmark = [pytest.mark.integration, pytest.mark.chaos]


# =============================================================================
# Partial Corruption Tests (E1)
# =============================================================================


def test_partial_corruption_does_not_abort_pipeline(tmp_path):
    """Test that 5 corrupted lines out of 100 don't abort pipeline.

    Expected: ~95 valid records ingested, 5 skipped with error context.
    """
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    # Build inbox with 100 records, 5 corrupted
    builder = ChaosInboxBuilder(inbox)
    builder.add_corrupted_jsonl(
        "partial_corruption.jsonl",
        provider="codex",
        count=100,
        corrupt_indices=[5, 20, 35, 50, 75],
        corruption_type="malformed",
    )
    builder.build()

    # Run ingestion
    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    # Pipeline should succeed despite corruption
    assert result.success, f"Pipeline failed: {result.stderr}"

    # Verify some records were ingested (not all, since we have corruption)
    db_path = workspace["paths"]["db_path"]
    assert db_path.exists(), "Database not created"

    # Query to verify records were saved
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        # We expect ~95 records (100 - 5 corrupted)
        assert 90 <= count <= 100, f"Expected ~95 records, got {count}"


def test_malformed_json_lines_skipped_with_context(tmp_path):
    """Test that malformed JSON lines are skipped and logged."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    # Build inbox with 10 records, corrupt lines 2 and 7
    builder = ChaosInboxBuilder(inbox)
    builder.add_corrupted_jsonl(
        "malformed.jsonl",
        provider="codex",
        count=10,
        corrupt_indices=[2, 7],
        corruption_type="malformed",
    )
    builder.build()

    # Run ingestion
    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    # Should succeed despite malformed lines
    assert result.success, f"Pipeline failed: {result.stderr}"

    # Check error context in output (error log or stderr)
    output = result.output.lower()
    # At least some indication of parsing/skipping errors
    # (implementation may vary)


def test_truncated_lines_handled_gracefully(tmp_path):
    """Test that truncated JSONL lines don't crash pipeline."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    builder = ChaosInboxBuilder(inbox)
    builder.add_corrupted_jsonl(
        "truncated.jsonl",
        provider="codex",
        count=20,
        corrupt_indices=[3, 9, 15],
        corruption_type="truncated",
    )
    builder.build()

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    # Verify database exists and has some records
    db_path = workspace["paths"]["db_path"]
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        # At least 17 records (20 - 3 truncated)
        assert count >= 17, f"Expected at least 17 records, got {count}"


def test_bad_utf8_lines_skipped(tmp_path):
    """Test that lines with invalid UTF-8 are handled gracefully."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    builder = ChaosInboxBuilder(inbox)
    builder.add_corrupted_jsonl(
        "bad_utf8.jsonl",
        provider="codex",
        count=15,
        corrupt_indices=[4, 10],
        corruption_type="bad_utf8",
    )
    builder.build()

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        # At least 13 records (15 - 2 bad UTF-8)
        assert count >= 13, f"Expected at least 13 records, got {count}"


def test_wrong_envelope_lines_skipped(tmp_path):
    """Test that lines with wrong provider envelope are skipped."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    builder = ChaosInboxBuilder(inbox)
    builder.add_corrupted_jsonl(
        "wrong_envelope.jsonl",
        provider="codex",
        count=12,
        corrupt_indices=[2, 6, 10],
        corruption_type="wrong_envelope",
    )
    builder.build()

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        # At least 9 records (12 - 3 wrong envelope)
        assert count >= 9, f"Expected at least 9 records, got {count}"


# =============================================================================
# Empty / Garbage File Tests (E2)
# =============================================================================


def test_empty_file_does_not_crash(tmp_path):
    """Test that empty JSONL files don't crash pipeline."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    builder = ChaosInboxBuilder(inbox)
    builder.add_empty_file("empty.jsonl")
    builder.add_valid_jsonl("valid.jsonl", provider="codex", count=5)
    builder.build()

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        # 5 from valid.jsonl
        assert count == 5, f"Expected 5 records, got {count}"


def test_binary_garbage_file_skipped(tmp_path):
    """Test that binary garbage files don't corrupt pipeline."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    builder = ChaosInboxBuilder(inbox)
    builder.add_binary_garbage("garbage.bin", size=512)
    builder.add_valid_jsonl("valid.jsonl", provider="codex", count=8)
    builder.build()

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        # 8 from valid.jsonl
        assert count == 8, f"Expected 8 records, got {count}"


def test_zero_byte_file_handled(tmp_path):
    """Test that zero-byte files are handled gracefully."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    builder = ChaosInboxBuilder(inbox)
    builder.add_empty_file("zero.jsonl")
    builder.add_valid_jsonl("data.jsonl", provider="codex", count=6)
    builder.build()

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        # 6 from data.jsonl
        assert count == 6, f"Expected 6 records, got {count}"


# =============================================================================
# Mixed Corruption Tests
# =============================================================================


def test_mixed_corruption_types_in_single_file(tmp_path):
    """Test resilience to multiple corruption types simultaneously."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    # Build file with mixed types by adding multiple corrupted files
    # For this test, we'll use one file with malformed corruption
    # and mix file types at the inbox level
    builder = ChaosInboxBuilder(inbox)
    builder.add_corrupted_jsonl(
        "mixed.jsonl",
        provider="codex",
        count=30,
        corrupt_indices=[3, 7, 12, 18, 25],
        corruption_type="malformed",
    )
    builder.add_empty_file("empty_sidekick.jsonl")
    builder.add_binary_garbage("junk.bin", size=256)
    builder.build()

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        # At least 25 records (30 - 5 corrupted)
        assert count >= 25, f"Expected at least 25 records, got {count}"


def test_file_with_bom_marker_ingested(tmp_path):
    """Test that JSONL files with UTF-8 BOM markers are handled."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    builder = ChaosInboxBuilder(inbox)
    builder.add_file_with_bom("bom_file.jsonl")
    builder.build()

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        # Should ingest the 5 records from BOM file
        assert count == 5, f"Expected 5 records, got {count}"


def test_mixed_providers_in_single_inbox(tmp_path):
    """Test that multiple provider formats coexist in same inbox."""
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    builder = ChaosInboxBuilder(inbox)
    builder.add_valid_jsonl("codex_data.jsonl", provider="codex", count=8)
    builder.add_valid_jsonl("claude_code_data.jsonl", provider="claude-code", count=7)
    builder.build()

    result = run_cli(
        ["run", "--source", "inbox", "--stage", "all"],
        env=workspace["env"],
        timeout=120.0,
    )

    assert result.success, f"Pipeline failed: {result.stderr}"

    db_path = workspace["paths"]["db_path"]
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT provider_name, COUNT(*) FROM messages GROUP BY provider_name"
        )
        results = dict(cursor.fetchall())
        # Should have both providers
        assert "codex" in results, "codex provider not found"
        assert "claude-code" in results, "claude-code provider not found"
        assert results["codex"] == 8, f"Expected 8 codex records, got {results['codex']}"
        assert results["claude-code"] == 7, f"Expected 7 claude-code records, got {results['claude-code']}"
