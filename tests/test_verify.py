"""Tests for the verify CLI command."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.storage.db import default_db_path, open_connection
from tests.factories import DbFactory


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


def _extract_json(output: str) -> dict:
    """Extract JSON from CLI output, skipping non-JSON lines."""
    lines = output.strip().split("\n")
    # Find first line that starts with { and join all subsequent lines
    json_start = next((i for i, line in enumerate(lines) if line.strip().startswith("{")), None)
    if json_start is None:
        raise ValueError(f"No JSON found in output: {output}")
    json_str = "\n".join(lines[json_start:])
    return json.loads(json_str)


class TestVerifyCommand:
    """Tests for polylogue verify command."""

    def test_verify_clean_database(self, workspace_env, cli_runner):
        """Verify command succeeds on clean database with valid data."""
        db_path = default_db_path()
        factory = DbFactory(db_path)

        # Create valid conversation with messages
        factory.create_conversation(
            id="conv1",
            provider="chatgpt",
            title="Test Conversation",
            messages=[
                {"id": "m1", "role": "user", "text": "hello"},
                {"id": "m2", "role": "assistant", "text": "world"},
            ],
        )

        result = cli_runner.invoke(cli, ["verify"])
        assert result.exit_code == 0
        assert "ok" in result.output.lower() or "✓" in result.output

    def test_verify_json_output(self, workspace_env, cli_runner):
        """Verify --json flag produces valid JSON."""
        db_path = default_db_path()
        factory = DbFactory(db_path)

        factory.create_conversation(
            id="conv1",
            provider="claude",
            messages=[{"id": "m1", "role": "user", "text": "test"}],
        )

        result = cli_runner.invoke(cli, ["--plain", "verify", "--json"])
        assert result.exit_code == 0

        # Parse JSON output
        data = _extract_json(result.output)
        assert "checks" in data
        assert "summary" in data
        assert isinstance(data["checks"], list)
        assert isinstance(data["summary"], dict)

        # Verify summary has expected keys
        assert "ok" in data["summary"]
        assert "warning" in data["summary"]
        assert "error" in data["summary"]

    def test_verify_detects_orphan_messages(self, workspace_env, cli_runner):
        """Verify detects messages without conversations."""
        db_path = default_db_path()

        # Disable foreign key constraints temporarily to insert orphan message
        with open_connection(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                """
                INSERT INTO messages (
                    message_id, conversation_id, role, text,
                    timestamp, content_hash, version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "orphan-msg",
                    "non-existent-conv",
                    "user",
                    "orphaned text",
                    "2024-01-01T00:00:00Z",
                    "abc123",
                    1,
                ),
            )
            conn.execute("PRAGMA foreign_keys = ON")

        result = cli_runner.invoke(cli, ["--plain", "verify", "--json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the orphaned_messages check
        orphan_check = next(
            (c for c in data["checks"] if c["name"] == "orphaned_messages"),
            None,
        )
        assert orphan_check is not None
        assert orphan_check["status"] == "error"
        assert orphan_check["count"] == 1

        # Summary should show at least one error
        assert data["summary"]["error"] >= 1

    def test_verify_verbose_output(self, workspace_env, cli_runner):
        """Verify -v flag increases detail with provider breakdown."""
        db_path = default_db_path()
        factory = DbFactory(db_path)

        # Create conversations from multiple providers
        factory.create_conversation(
            id="conv1",
            provider="chatgpt",
            messages=[{"id": "m1", "role": "user", "text": "hello"}],
        )
        factory.create_conversation(
            id="conv2",
            provider="claude",
            messages=[{"id": "m2", "role": "user", "text": "world"}],
        )

        # Run verify without verbose
        result_normal = cli_runner.invoke(cli, ["verify"])
        assert result_normal.exit_code == 0

        # Run verify with verbose
        result_verbose = cli_runner.invoke(cli, ["verify", "-v"])
        assert result_verbose.exit_code == 0

        # Verbose output should contain provider names for breakdowns
        # (provider_distribution check always has breakdown)
        assert "chatgpt" in result_verbose.output or "claude" in result_verbose.output

    def test_verify_detects_empty_conversations(self, workspace_env, cli_runner):
        """Verify detects conversations with no messages (warning status)."""
        db_path = default_db_path()

        # Create a conversation with no messages
        with open_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO conversations (
                    conversation_id, provider_name, provider_conversation_id,
                    title, created_at, updated_at, content_hash, version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "empty-conv",
                    "test",
                    "ext-empty",
                    "Empty Conversation",
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                    "def456",
                    1,
                ),
            )

        result = cli_runner.invoke(cli, ["--plain", "verify", "--json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the empty_conversations check
        empty_check = next(
            (c for c in data["checks"] if c["name"] == "empty_conversations"),
            None,
        )
        assert empty_check is not None
        assert empty_check["status"] == "warning"
        assert empty_check["count"] == 1

    def test_verify_no_duplicate_conversation_ids(self, workspace_env, cli_runner):
        """Verify duplicate_conversations check passes when there are no duplicates."""
        db_path = default_db_path()
        factory = DbFactory(db_path)

        # Create unique conversations (duplicates prevented by UNIQUE constraint)
        factory.create_conversation(
            id="conv1",
            provider="test",
            messages=[{"id": "m1", "role": "user", "text": "test1"}],
        )
        factory.create_conversation(
            id="conv2",
            provider="test",
            messages=[{"id": "m2", "role": "user", "text": "test2"}],
        )

        result = cli_runner.invoke(cli, ["--plain", "verify", "--json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the duplicate_conversations check
        dup_check = next(
            (c for c in data["checks"] if c["name"] == "duplicate_conversations"),
            None,
        )
        assert dup_check is not None
        assert dup_check["status"] == "ok"
        assert dup_check["count"] == 0  # No duplicates found

    def test_verify_detects_fts_sync_issues(self, workspace_env, cli_runner):
        """Verify detects FTS sync issues when FTS table is missing."""
        db_path = default_db_path()
        factory = DbFactory(db_path)

        # Create some messages
        factory.create_conversation(
            id="conv1",
            provider="test",
            messages=[{"id": "m1", "role": "user", "text": "test"}],
        )

        # Manually drop FTS table to simulate desync
        with open_connection(db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS messages_fts")

        result = cli_runner.invoke(cli, ["--plain", "verify", "--json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the fts_sync check
        fts_check = next(
            (c for c in data["checks"] if c["name"] == "fts_sync"),
            None,
        )
        assert fts_check is not None
        # Should be warning due to missing FTS table
        assert fts_check["status"] == "warning"

    def test_verify_plain_output_format(self, workspace_env, cli_runner):
        """Verify --plain flag produces plain text output without colors."""
        db_path = default_db_path()
        factory = DbFactory(db_path)

        factory.create_conversation(
            id="conv1",
            provider="test",
            messages=[{"id": "m1", "role": "user", "text": "test"}],
        )

        result = cli_runner.invoke(cli, ["--plain", "verify"])
        assert result.exit_code == 0

        # Plain output should use OK/WARN/ERR instead of symbols
        assert "OK" in result.output or "ok" in result.output.lower()
        # Should not contain ANSI color codes
        assert "[green]" not in result.output
        assert "[red]" not in result.output

    def test_verify_summary_counts(self, workspace_env, cli_runner):
        """Verify summary shows correct counts of ok/warning/error checks."""
        db_path = default_db_path()
        factory = DbFactory(db_path)

        # Create valid data
        factory.create_conversation(
            id="conv1",
            provider="test",
            messages=[{"id": "m1", "role": "user", "text": "test"}],
        )

        result = cli_runner.invoke(cli, ["verify"])
        assert result.exit_code == 0

        # Check summary line format
        assert "Summary:" in result.output
        assert "ok" in result.output
        assert "warnings" in result.output or "warning" in result.output
        assert "errors" in result.output or "error" in result.output

    def test_verify_empty_database(self, workspace_env, cli_runner):
        """Verify command succeeds on empty database."""
        # Don't create any data - just verify empty DB

        result = cli_runner.invoke(cli, ["verify"])
        assert result.exit_code == 0

        # Should report 0 conversations, 0 messages
        assert "0 conversations" in result.output or "0 message" in result.output

    def test_verify_sqlite_integrity_check(self, workspace_env, cli_runner):
        """Verify includes SQLite integrity check in results."""
        db_path = default_db_path()
        factory = DbFactory(db_path)

        factory.create_conversation(
            id="conv1",
            provider="test",
            messages=[{"id": "m1", "role": "user", "text": "test"}],
        )

        result = cli_runner.invoke(cli, ["--plain", "verify", "--json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the sqlite_integrity check
        integrity_check = next(
            (c for c in data["checks"] if c["name"] == "sqlite_integrity"),
            None,
        )
        assert integrity_check is not None
        assert integrity_check["status"] == "ok"
        assert integrity_check["detail"] == "ok"
