"""Tests for polylogue check command.

Coverage targets:
- check_command: CLI health check command
- --repair: Repair mode that runs actual fixes
- --vacuum: VACUUM database after repair
- --json: JSON output format
- --verbose: Show breakdown by provider
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.check import check_command
from polylogue.health import (
    HealthCheck,
    HealthReport,
    RepairResult,
    VerifyStatus,
    repair_dangling_fts,
    repair_empty_conversations,
    repair_orphaned_attachments,
    repair_orphaned_messages,
    run_all_repairs,
)


@pytest.fixture
def runner():
    """CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_env():
    """Create mock AppEnv for tests."""
    mock_ui = MagicMock()
    mock_ui.plain = True
    mock_ui.console = MagicMock()
    mock_ui.summary = MagicMock()

    env = MagicMock()
    env.ui = mock_ui
    return env


@pytest.fixture
def sample_health_report():
    """Create a sample health report with issues."""
    return HealthReport(
        checks=[
            HealthCheck("config", VerifyStatus.OK, detail="Zero-config"),
            HealthCheck("database", VerifyStatus.OK, detail="DB reachable"),
            HealthCheck("orphaned_messages", VerifyStatus.ERROR, count=5, detail="5 orphaned messages"),
            HealthCheck("empty_conversations", VerifyStatus.WARNING, count=2, detail="2 empty conversations"),
            HealthCheck("fts_sync", VerifyStatus.OK, detail="FTS in sync"),
        ],
        summary={"ok": 3, "warning": 1, "error": 1},
    )


@pytest.fixture
def healthy_report():
    """Create a completely healthy report."""
    return HealthReport(
        checks=[
            HealthCheck("config", VerifyStatus.OK, detail="Zero-config"),
            HealthCheck("database", VerifyStatus.OK, detail="DB reachable"),
            HealthCheck("orphaned_messages", VerifyStatus.OK, detail="No orphaned messages"),
            HealthCheck("empty_conversations", VerifyStatus.OK, detail="No empty conversations"),
            HealthCheck("fts_sync", VerifyStatus.OK, detail="FTS in sync"),
        ],
        summary={"ok": 5, "warning": 0, "error": 0},
    )


class TestCheckCommand:
    """Tests for the check command."""

    def test_check_displays_health_status(self, runner, mock_env, healthy_report):
        """Check command displays health status."""
        with patch("polylogue.cli.commands.check.load_effective_config"), \
             patch("polylogue.cli.commands.check.get_health", return_value=healthy_report):
            result = runner.invoke(check_command, obj=mock_env)

            assert result.exit_code == 0
            # Verify summary was called
            assert mock_env.ui.summary.called

    def test_check_json_output(self, runner, mock_env, healthy_report):
        """Check --json outputs JSON format."""
        with patch("polylogue.cli.commands.check.load_effective_config"), \
             patch("polylogue.cli.commands.check.get_health", return_value=healthy_report):
            result = runner.invoke(check_command, ["--json"], obj=mock_env)

            assert result.exit_code == 0
            # JSON output should be printed
            calls = mock_env.ui.console.print.call_args_list
            assert any("ok" in str(c).lower() for c in calls)

    def test_check_vacuum_requires_repair(self, runner, mock_env):
        """--vacuum requires --repair flag."""
        result = runner.invoke(check_command, ["--vacuum"], obj=mock_env)

        # Should fail with message about requiring --repair
        assert result.exit_code != 0

    def test_check_repair_on_healthy_db(self, runner, mock_env, healthy_report):
        """Repair mode on healthy database shows no issues."""
        with patch("polylogue.cli.commands.check.load_effective_config"), \
             patch("polylogue.cli.commands.check.get_health", return_value=healthy_report):
            result = runner.invoke(check_command, ["--repair"], obj=mock_env)

            assert result.exit_code == 0
            calls = mock_env.ui.console.print.call_args_list
            output = " ".join(str(c) for c in calls)
            assert "no issues to repair" in output.lower()

    def test_check_repair_runs_fixes(self, runner, mock_env, sample_health_report):
        """Repair mode runs repair functions when issues exist."""
        repair_results = [
            RepairResult("orphaned_messages", 5, True, "Deleted 5 orphaned messages"),
            RepairResult("empty_conversations", 2, True, "Deleted 2 empty conversations"),
            RepairResult("dangling_fts", 0, True, "FTS in sync"),
            RepairResult("orphaned_attachments", 0, True, "No orphaned attachments"),
        ]

        with patch("polylogue.cli.commands.check.load_effective_config"), \
             patch("polylogue.cli.commands.check.get_health", return_value=sample_health_report), \
             patch("polylogue.cli.commands.check.run_all_repairs", return_value=repair_results):
            result = runner.invoke(check_command, ["--repair"], obj=mock_env)

            assert result.exit_code == 0
            calls = mock_env.ui.console.print.call_args_list
            output = " ".join(str(c) for c in calls)
            assert "repair" in output.lower()
            assert "7" in output  # 5 + 2 total repaired

    def test_check_repair_with_vacuum(self, runner, mock_env, sample_health_report):
        """Repair with --vacuum runs VACUUM after repairs."""
        repair_results = [
            RepairResult("orphaned_messages", 5, True, "Deleted 5 orphaned messages"),
        ]

        with patch("polylogue.cli.commands.check.load_effective_config"), \
             patch("polylogue.cli.commands.check.get_health", return_value=sample_health_report), \
             patch("polylogue.cli.commands.check.run_all_repairs", return_value=repair_results), \
             patch("polylogue.storage.backends.sqlite.open_connection") as mock_conn, \
             patch("polylogue.storage.backends.sqlite.default_db_path", return_value=Path("/tmp/test.db")):
            # Create mock connection that properly handles context manager
            mock_connection = MagicMock()
            mock_conn.return_value.__enter__ = MagicMock(return_value=mock_connection)
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            result = runner.invoke(check_command, ["--repair", "--vacuum"], obj=mock_env)

            assert result.exit_code == 0
            calls = mock_env.ui.console.print.call_args_list
            output = " ".join(str(c) for c in calls)
            assert "vacuum" in output.lower()


class TestRepairFunctions:
    """Tests for individual repair functions."""

    def test_repair_orphaned_messages(self, workspace_env):
        """repair_orphaned_messages deletes orphaned messages."""
        from polylogue.config import Config
        from polylogue.storage.backends.sqlite import open_connection
        from tests.helpers import db_setup

        db_path = db_setup(workspace_env)

        # Insert orphaned message (no corresponding conversation)
        # Disable foreign keys temporarily to create orphaned data
        with open_connection(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                "INSERT INTO messages (message_id, conversation_id, role, text, content_hash, version) VALUES (?, ?, ?, ?, ?, ?)",
                ("orphan-msg-1", "non-existent-conv", "user", "orphaned", "hash123", 1),
            )
            conn.commit()
            conn.execute("PRAGMA foreign_keys = ON")

            # Verify it exists
            count = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", ("non-existent-conv",)).fetchone()[0]
            assert count == 1

        config = MagicMock(spec=Config)
        result = repair_orphaned_messages(config)

        assert result.success
        assert result.repaired_count == 1
        assert "orphaned" in result.detail.lower()

    def test_repair_empty_conversations(self, workspace_env):
        """repair_empty_conversations deletes empty conversations."""
        from polylogue.config import Config
        from polylogue.storage.backends.sqlite import open_connection
        from tests.helpers import db_setup

        db_path = db_setup(workspace_env)

        # Insert empty conversation (no messages)
        with open_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO conversations
                   (conversation_id, provider_name, provider_conversation_id, title, created_at, updated_at, content_hash, version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("empty-conv-1", "test", "ext-1", "Empty Conv", "2024-01-01", "2024-01-01", "hash123", 1),
            )
            conn.commit()

            # Verify it exists
            count = conn.execute("SELECT COUNT(*) FROM conversations WHERE conversation_id = ?", ("empty-conv-1",)).fetchone()[0]
            assert count == 1

        config = MagicMock(spec=Config)
        result = repair_empty_conversations(config)

        assert result.success
        assert result.repaired_count == 1
        assert "empty" in result.detail.lower()

    def test_repair_dangling_fts_no_table(self, workspace_env):
        """repair_dangling_fts handles missing FTS table gracefully."""
        from polylogue.config import Config
        from polylogue.storage.backends.sqlite import open_connection
        from tests.helpers import db_setup

        db_path = db_setup(workspace_env)

        # Drop FTS table if it exists
        with open_connection(db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS messages_fts")
            conn.commit()

        config = MagicMock(spec=Config)
        result = repair_dangling_fts(config)

        assert result.success
        assert result.repaired_count == 0
        assert "does not exist" in result.detail

    def test_repair_orphaned_attachments(self, workspace_env):
        """repair_orphaned_attachments cleans up orphaned attachments."""
        from polylogue.config import Config
        from polylogue.storage.backends.sqlite import open_connection
        from tests.helpers import db_setup

        db_path = db_setup(workspace_env)

        # Insert orphaned attachment ref (non-existent message)
        # Disable foreign keys temporarily to create orphaned data
        with open_connection(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            # First add an attachment
            conn.execute(
                "INSERT INTO attachments (attachment_id, mime_type, size_bytes, ref_count) VALUES (?, ?, ?, ?)",
                ("orphan-att-1", "image/png", 1024, 0),
            )
            # Add orphaned ref
            conn.execute(
                "INSERT INTO attachment_refs (ref_id, attachment_id, conversation_id, message_id) VALUES (?, ?, ?, ?)",
                ("ref-1", "orphan-att-1", "non-existent-conv", "non-existent-msg"),
            )
            conn.commit()
            conn.execute("PRAGMA foreign_keys = ON")

        config = MagicMock(spec=Config)
        result = repair_orphaned_attachments(config)

        assert result.success
        assert result.repaired_count >= 1

    def test_run_all_repairs(self, workspace_env):
        """run_all_repairs runs all repair functions."""
        from polylogue.config import Config

        config = MagicMock(spec=Config)

        with patch("polylogue.health.repair_orphaned_messages") as mock_orphan, \
             patch("polylogue.health.repair_empty_conversations") as mock_empty, \
             patch("polylogue.health.repair_dangling_fts") as mock_fts, \
             patch("polylogue.health.repair_orphaned_attachments") as mock_att:
            mock_orphan.return_value = RepairResult("orphaned_messages", 0, True, "OK")
            mock_empty.return_value = RepairResult("empty_conversations", 0, True, "OK")
            mock_fts.return_value = RepairResult("dangling_fts", 0, True, "OK")
            mock_att.return_value = RepairResult("orphaned_attachments", 0, True, "OK")

            results = run_all_repairs(config)

            assert len(results) == 4
            assert all(r.success for r in results)


class TestVerboseMode:
    """Tests for verbose output mode."""

    def test_verbose_shows_breakdown(self, runner, mock_env):
        """--verbose shows breakdown by provider."""
        report = HealthReport(
            checks=[
                HealthCheck(
                    "orphaned_messages",
                    VerifyStatus.WARNING,
                    count=10,
                    detail="10 orphaned messages",
                    breakdown={"chatgpt": 6, "claude": 4},
                ),
            ],
            summary={"ok": 0, "warning": 1, "error": 0},
        )

        with patch("polylogue.cli.commands.check.load_effective_config"), \
             patch("polylogue.cli.commands.check.get_health", return_value=report):
            result = runner.invoke(check_command, ["--verbose"], obj=mock_env)

            assert result.exit_code == 0
            # In verbose mode, summary should be called with breakdown info
            assert mock_env.ui.summary.called
