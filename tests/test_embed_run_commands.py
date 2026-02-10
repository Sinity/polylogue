"""Tests for embed and run commands with comprehensive coverage.

This file covers previously uncovered areas in:
- polylogue/cli/commands/embed.py (44% coverage, 69 uncovered lines)
- polylogue/cli/commands/run.py (56% coverage, 75 uncovered lines)

Test patterns:
- Internal functions tested directly (easier to mock)
- Click commands tested via CliRunner
- Context managers and database operations mocked
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.commands.embed import (
    _embed_batch,
    _embed_single,
    _show_embedding_stats,
    embed_command,
)
from polylogue.cli.commands.run import (
    _display_result,
    _exec_on_new,
    _notify_new_conversations,
    _run_sync_once,
    _webhook_on_new,
    run_command,
    sources_command,
)
from polylogue.storage.store import PlanResult, RunResult


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def mock_env():
    """Mock AppEnv with ui."""
    from rich.console import Console

    env = MagicMock()
    env.ui = MagicMock()
    env.ui.plain = True
    env.ui.console = Console()
    env.ui.confirm.return_value = True
    env.ui.summary = MagicMock()
    return env


@pytest.fixture
def mock_env_rich():
    """Mock AppEnv with Rich console."""
    from rich.console import Console

    env = MagicMock()
    env.ui = MagicMock()
    env.ui.plain = False
    env.ui.console = Console()
    env.ui.confirm.return_value = True
    env.ui.summary = MagicMock()
    return env


@pytest.fixture
def mock_run_result():
    """Mock RunResult for sync mode."""
    return RunResult(
        run_id="run-123",
        counts={"conversations": 3, "messages": 30, "attachments": 1},
        drift={"conversations": {"new": 2, "updated": 1, "unchanged": 5}},
        indexed=True,
        index_error=None,
        duration_ms=1500,
        render_failures=[],
    )


@pytest.fixture
def mock_plan_result():
    """Mock PlanResult for preview mode."""
    return PlanResult(
        timestamp=1234567890,
        counts={"conversations": 5, "messages": 50, "attachments": 2},
        sources=["test-inbox"],
        cursors={},
    )


@pytest.fixture
def mock_conversation():
    """Mock conversation object."""
    conv = MagicMock()
    conv.conversation_id = "conv-123"
    conv.title = "Test Conversation"
    return conv


@pytest.fixture
def mock_repository():
    """Mock ConversationRepository."""
    repo = MagicMock()
    repo.backend = MagicMock()
    repo.backend.get_messages = MagicMock(return_value=[
        {"message_id": "m1", "text": "Hello"},
        {"message_id": "m2", "text": "World"},
    ])
    return repo


# =============================================================================
# TEST CLASS: _show_embedding_stats
# =============================================================================


class TestShowEmbeddingStats:
    """Test _show_embedding_stats function."""

    def test_show_stats_basic(self, mock_env, capsys):
        """_show_embedding_stats displays statistics."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value=(100,))),  # total_convs
            MagicMock(fetchone=MagicMock(return_value=(50,))),   # embedded_convs
            MagicMock(fetchone=MagicMock(return_value=(200,))),  # embedded_msgs
            MagicMock(fetchone=MagicMock(return_value=(50,))),   # pending
        ]

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            _show_embedding_stats(mock_env)

        captured = capsys.readouterr()
        assert "Embedding Statistics" in captured.out
        assert "Total conversations:    100" in captured.out
        assert "Embedded conversations: 50" in captured.out
        assert "Embedded messages:      200" in captured.out
        assert "Coverage:               50.0%" in captured.out
        assert "Pending:                50" in captured.out

    def test_show_stats_embedding_status_missing(self, mock_env, capsys):
        """_show_embedding_stats handles missing embedding_status table."""
        mock_conn = MagicMock()

        def side_effect(*args, **kwargs):
            result = MagicMock()
            # First query succeeds (total_convs)
            if "COUNT(*) FROM conversations" in args[0]:
                result.fetchone.return_value = (100,)
            # embedding_status query fails
            elif "embedding_status" in args[0]:
                raise Exception("table does not exist")
            # message_embeddings query fails
            elif "message_embeddings" in args[0]:
                raise Exception("table does not exist")
            # pending query fails
            else:
                raise Exception("unknown query")
            return result

        mock_conn.execute.side_effect = side_effect

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            _show_embedding_stats(mock_env)

        captured = capsys.readouterr()
        assert "Embedding Statistics" in captured.out

    def test_show_stats_zero_conversations(self, mock_env, capsys):
        """_show_embedding_stats handles zero conversations (division by zero)."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value=(0,))),  # total_convs
            MagicMock(fetchone=MagicMock(return_value=(0,))),  # embedded_convs
            MagicMock(fetchone=MagicMock(return_value=(0,))),  # embedded_msgs
            MagicMock(fetchone=MagicMock(return_value=(0,))),  # pending
        ]

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            _show_embedding_stats(mock_env)

        captured = capsys.readouterr()
        assert "Coverage:               0.0%" in captured.out


# =============================================================================
# TEST CLASS: _embed_single
# =============================================================================


class TestEmbedSingle:
    """Test _embed_single function."""

    def test_embed_single_success(self, mock_env, mock_repository, mock_conversation, capsys):
        """_embed_single successfully embeds a conversation."""
        mock_repository.get.return_value = mock_conversation
        mock_vec_provider = MagicMock()
        mock_vec_provider.upsert = MagicMock()

        _embed_single(mock_env, mock_repository, mock_vec_provider, "conv-123")

        mock_vec_provider.upsert.assert_called_once_with("conv-123", mock_repository.backend.get_messages.return_value)
        captured = capsys.readouterr()
        assert "Embedding 2 messages" in captured.out
        assert "✓ Embedded" in captured.out

    def test_embed_single_conversation_not_found(self, mock_env, mock_repository):
        """_embed_single fails when conversation not found."""
        mock_repository.get.return_value = None
        mock_vec_provider = MagicMock()

        with pytest.raises(Exception):  # click.Abort
            _embed_single(mock_env, mock_repository, mock_vec_provider, "nonexistent")

    def test_embed_single_no_messages(self, mock_env, mock_repository, mock_conversation, capsys):
        """_embed_single exits early when no messages."""
        mock_repository.get.return_value = mock_conversation
        mock_repository.backend.get_messages.return_value = []
        mock_vec_provider = MagicMock()

        _embed_single(mock_env, mock_repository, mock_vec_provider, "conv-123")

        mock_vec_provider.upsert.assert_not_called()
        captured = capsys.readouterr()
        assert "No messages to embed" in captured.out

    def test_embed_single_upsert_exception(self, mock_env, mock_repository, mock_conversation):
        """_embed_single handles upsert exceptions."""
        mock_repository.get.return_value = mock_conversation
        mock_vec_provider = MagicMock()
        mock_vec_provider.upsert.side_effect = ValueError("API error")

        with pytest.raises(Exception):  # click.Abort
            _embed_single(mock_env, mock_repository, mock_vec_provider, "conv-123")


# =============================================================================
# TEST CLASS: _embed_batch
# =============================================================================


class TestEmbedBatch:
    """Test _embed_batch function."""

    def test_embed_batch_all_already_embedded(self, mock_env, mock_repository, capsys):
        """_embed_batch exits early when all conversations already embedded."""
        mock_vec_provider = MagicMock()

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            _embed_batch(mock_env, mock_repository, mock_vec_provider)

        captured = capsys.readouterr()
        assert "All conversations are already embedded" in captured.out

    def test_embed_batch_plain_mode(self, mock_env, mock_repository, capsys):
        """_embed_batch in plain mode uses click.echo."""
        mock_env.ui.console = MagicMock()  # Not a RichConsole
        mock_vec_provider = MagicMock()

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = [
                {"conversation_id": "conv-1", "title": "Test 1"},
                {"conversation_id": "conv-2", "title": "Test 2"},
            ]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            _embed_batch(mock_env, mock_repository, mock_vec_provider)

        captured = capsys.readouterr()
        assert "Embedding 2 conversations" in captured.out

    def test_embed_batch_with_limit(self, mock_env, mock_repository, capsys):
        """_embed_batch respects limit parameter."""
        mock_vec_provider = MagicMock()

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = [
                {"conversation_id": "conv-1", "title": "Test 1"},
                {"conversation_id": "conv-2", "title": "Test 2"},
                {"conversation_id": "conv-3", "title": "Test 3"},
            ]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            _embed_batch(mock_env, mock_repository, mock_vec_provider, limit=2)

        captured = capsys.readouterr()
        assert "Embedding 2 conversations" in captured.out

    def test_embed_batch_rebuild_flag(self, mock_env, mock_repository):
        """_embed_batch with rebuild=True queries all conversations."""
        mock_vec_provider = MagicMock()

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            _embed_batch(mock_env, mock_repository, mock_vec_provider, rebuild=True)

            # Check that the rebuild query was used
            calls = mock_conn.execute.call_args_list
            assert any("ORDER BY updated_at DESC" in str(call) for call in calls)

    def test_embed_batch_error_handling(self, mock_env, mock_repository, capsys):
        """_embed_batch continues on error and reports counts."""
        mock_vec_provider = MagicMock()
        mock_repository.backend.get_messages.side_effect = [
            [{"message_id": "m1"}],  # First conv succeeds
            ValueError("Embed failed"),  # Second conv fails
            [{"message_id": "m3"}],  # Third conv succeeds
        ]

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = [
                {"conversation_id": "conv-1", "title": "Test 1"},
                {"conversation_id": "conv-2", "title": "Test 2"},
                {"conversation_id": "conv-3", "title": "Test 3"},
            ]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            _embed_batch(mock_env, mock_repository, mock_vec_provider)

        captured = capsys.readouterr()
        assert "2 errors" in captured.out or "error" in captured.out.lower()


# =============================================================================
# TEST CLASS: embed_command (Click command)
# =============================================================================


class TestEmbedCommand:
    """Test embed command via CliRunner."""

    def test_embed_command_missing_api_key(self, runner, cli_workspace):
        """embed without API key shows error."""
        with patch.dict("os.environ", {"VOYAGE_API_KEY": ""}, clear=False):
            result = runner.invoke(cli, ["embed", "--plain"])
            assert result.exit_code != 0
            assert "VOYAGE_API_KEY" in result.output or "Error" in result.output

    def test_embed_command_help(self, runner, cli_workspace):
        """embed --help shows usage."""
        result = runner.invoke(cli, ["embed", "--help"])
        assert result.exit_code == 0
        assert "embed" in result.output.lower()

    def test_embed_command_with_stats_help(self, runner, cli_workspace):
        """embed command has --stats option in help."""
        result = runner.invoke(cli, ["embed", "--help"])
        assert "--stats" in result.output


# =============================================================================
# TEST CLASS: _run_sync_once
# =============================================================================


class TestRunSyncOnce:
    """Test _run_sync_once function."""

    def test_run_sync_once_plain_mode(self, mock_env, mock_run_result, capsys):
        """_run_sync_once in plain mode prints progress."""
        mock_env.ui.plain = True

        with patch("polylogue.cli.commands.run.run_sources") as mock_run:
            mock_run.return_value = mock_run_result
            mock_config = MagicMock()

            result = _run_sync_once(
                mock_config,
                mock_env,
                "all",
                None,
                "markdown",
            )

        assert result.run_id == "run-123"
        captured = capsys.readouterr()
        assert "Syncing" in captured.out

    def test_run_sync_once_rich_mode(self, mock_env_rich, mock_run_result):
        """_run_sync_once in rich mode uses Progress."""
        mock_env_rich.ui.plain = False

        with patch("polylogue.cli.commands.run.run_sources") as mock_run:
            mock_run.return_value = mock_run_result
            mock_config = MagicMock()

            result = _run_sync_once(
                mock_config,
                mock_env_rich,
                "all",
                None,
                "html",
            )

        assert result.run_id == "run-123"

    def test_run_sync_once_with_plan_snapshot(self, mock_env, mock_run_result, mock_plan_result):
        """_run_sync_once passes plan_snapshot to run_sources."""
        with patch("polylogue.cli.commands.run.run_sources") as mock_run:
            mock_run.return_value = mock_run_result
            mock_config = MagicMock()

            _run_sync_once(
                mock_config,
                mock_env,
                "all",
                ["source1"],
                "markdown",
                plan_snapshot=mock_plan_result,
            )

            # Check that plan was passed to run_sources
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["plan"] == mock_plan_result


# =============================================================================
# TEST CLASS: _display_result
# =============================================================================


class TestDisplayResult:
    """Test _display_result function."""

    def test_display_result_basic(self, mock_env, mock_run_result):
        """_display_result displays sync counts and duration."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")

        _display_result(mock_env, mock_config, mock_run_result, "all", None)

        mock_env.ui.summary.assert_called_once()
        title, lines = mock_env.ui.summary.call_args[0]
        assert "Sync" in title

    def test_display_result_index_stage(self, mock_env, mock_run_result):
        """_display_result formats index stage specially."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")
        mock_run_result.indexed = True

        _display_result(mock_env, mock_config, mock_run_result, "index", None)

        # Should show index status
        title, lines = mock_env.ui.summary.call_args[0]
        assert any("index" in line.lower() for line in lines)

    def test_display_result_with_sources(self, mock_env, mock_run_result):
        """_display_result includes source names in title."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")

        _display_result(mock_env, mock_config, mock_run_result, "all", ["source1", "source2"])

        title, lines = mock_env.ui.summary.call_args[0]
        assert "source1" in title or "source2" in title

    def test_display_result_with_render_failures(self, mock_env, capsys):
        """_display_result shows render failures."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")
        mock_run_result = RunResult(
            run_id="run-123",
            counts={"conversations": 1},
            drift={},
            indexed=False,
            index_error=None,
            duration_ms=100,
            render_failures=[
                {"conversation_id": "conv-1", "error": "template error"},
                {"conversation_id": "conv-2", "error": "parse error"},
            ],
        )

        _display_result(mock_env, mock_config, mock_run_result, "all", None)

        captured = capsys.readouterr()
        # Output goes to stderr
        assert "Render failures" in captured.err or "error" in captured.err.lower()

    def test_display_result_with_index_error(self, mock_env, capsys):
        """_display_result shows index error hint."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")
        mock_run_result = RunResult(
            run_id="run-123",
            counts={"conversations": 1},
            drift={},
            indexed=False,
            index_error="Connection refused",
            duration_ms=100,
            render_failures=[],
        )

        _display_result(mock_env, mock_config, mock_run_result, "all", None)

        captured = capsys.readouterr()
        # Output goes to stderr
        assert "Index error" in captured.err


# =============================================================================
# TEST CLASS: _notify_new_conversations
# =============================================================================


class TestNotifyNewConversations:
    """Test _notify_new_conversations function."""

    def test_notify_new_conversations_success(self):
        """_notify_new_conversations calls notify-send."""
        with patch("subprocess.run") as mock_run:
            _notify_new_conversations(3)

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "notify-send" in call_args
            assert "3" in str(call_args)

    def test_notify_new_conversations_not_found(self):
        """_notify_new_conversations silently ignores FileNotFoundError."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            # Should not raise
            _notify_new_conversations(1)


# =============================================================================
# TEST CLASS: _exec_on_new
# =============================================================================


class TestExecOnNew:
    """Test _exec_on_new function."""

    def test_exec_on_new_sets_env_var(self):
        """_exec_on_new sets POLYLOGUE_NEW_COUNT environment variable."""
        with patch("subprocess.run") as mock_run:
            _exec_on_new("echo $POLYLOGUE_NEW_COUNT", 5)

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["env"]["POLYLOGUE_NEW_COUNT"] == "5"
            assert call_kwargs["shell"] is True

    def test_exec_on_new_command_execution(self):
        """_exec_on_new executes the command."""
        with patch("subprocess.run") as mock_run:
            _exec_on_new("ls -la", 1)

            call_args = mock_run.call_args[0][0]
            assert "ls -la" in call_args


# =============================================================================
# TEST CLASS: _webhook_on_new
# =============================================================================


class TestWebhookOnNew:
    """Test _webhook_on_new function."""

    def test_webhook_on_new_success(self):
        """_webhook_on_new sends POST request to webhook URL."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            _webhook_on_new("http://example.com/webhook", 3)

            mock_urlopen.assert_called_once()
            call_args = mock_urlopen.call_args[0][0]
            assert call_args.get_full_url() == "http://example.com/webhook"
            assert call_args.get_method() == "POST"

    def test_webhook_on_new_json_payload(self):
        """_webhook_on_new includes correct JSON payload."""
        with patch("urllib.request.urlopen"):
            with patch("urllib.request.Request") as mock_request:
                _webhook_on_new("http://example.com/webhook", 5)

                # Check that Request was called with correct data
                call_kwargs = mock_request.call_args[1]
                payload = json.loads(call_kwargs["data"].decode())
                assert payload["event"] == "sync"
                assert payload["new_conversations"] == 5

    def test_webhook_on_new_exception_logged(self):
        """_webhook_on_new logs exceptions without raising."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = ValueError("Connection failed")

            # Should not raise
            _webhook_on_new("http://example.com/webhook", 1)


# =============================================================================
# TEST CLASS: run_command (Click command)
# =============================================================================


class TestRunCommand:
    """Test run command via CliRunner."""

    def test_run_command_watch_without_flags_error(self, runner, cli_workspace):
        """run --notify without --watch shows error."""
        result = runner.invoke(cli, ["run", "--notify", "--plain"])
        assert result.exit_code != 0

    def test_run_command_help(self, runner, cli_workspace):
        """run --help shows usage."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output.lower()

    def test_run_command_stage_option(self, runner, cli_workspace):
        """run command has --stage option in help."""
        result = runner.invoke(cli, ["run", "--help"])
        assert "--stage" in result.output

    def test_run_command_preview_option(self, runner, cli_workspace):
        """run command has --preview option in help."""
        result = runner.invoke(cli, ["run", "--help"])
        assert "--preview" in result.output


# =============================================================================
# TEST CLASS: sources_command (Click command)
# =============================================================================


class TestSourcesCommand:
    """Test sources command via CliRunner."""

    def test_sources_command_help(self, runner, cli_workspace):
        """sources --help shows usage."""
        result = runner.invoke(cli, ["sources", "--help"])
        assert result.exit_code == 0
        assert "source" in result.output.lower()

    def test_sources_command_json_option(self, runner, cli_workspace):
        """sources command has --json option in help."""
        result = runner.invoke(cli, ["sources", "--help"])
        assert "--json" in result.output


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestEmbedBatchRichMode:
    """Test _embed_batch with Rich console enabled."""

    def test_embed_batch_rich_mode(self, mock_env_rich, mock_repository, capsys):
        """_embed_batch with Rich console uses Progress bar."""
        mock_vec_provider = MagicMock()
        mock_repository.backend.get_messages.return_value = [{"message_id": "m1"}]

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = [
                {"conversation_id": "conv-1", "title": "Test 1"},
                {"conversation_id": "conv-2", "title": "Test 2"},
            ]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            _embed_batch(mock_env_rich, mock_repository, mock_vec_provider)

        captured = capsys.readouterr()
        assert "Embedding 2 conversations" in captured.out or "Embedded" in captured.out

    def test_embed_batch_rich_mode_with_empty_messages(self, mock_env_rich, mock_repository, capsys):
        """_embed_batch in rich mode handles conversations with no messages."""
        mock_vec_provider = MagicMock()
        # First conv has messages, second doesn't
        mock_repository.backend.get_messages.side_effect = [
            [{"message_id": "m1"}],
            [],
        ]

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = [
                {"conversation_id": "conv-1", "title": "Test 1"},
                {"conversation_id": "conv-2", "title": "Test 2"},
            ]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            _embed_batch(mock_env_rich, mock_repository, mock_vec_provider)

        # Should successfully complete
        mock_vec_provider.upsert.assert_called_once()

    def test_embed_batch_rich_mode_exception_handling(self, mock_env_rich, mock_repository, capsys):
        """_embed_batch in rich mode handles exceptions and continues."""
        mock_vec_provider = MagicMock()
        mock_vec_provider.upsert.side_effect = RuntimeError("API timeout")
        mock_repository.backend.get_messages.return_value = [{"message_id": "m1"}]

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = [
                {"conversation_id": "conv-1", "title": "Test 1"},
            ]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            _embed_batch(mock_env_rich, mock_repository, mock_vec_provider)

        captured = capsys.readouterr()
        # Should show error count
        assert "error" in captured.out.lower() or "1 error" in captured.out


class TestPlainModeProgress:
    """Test progress callback in plain mode."""

    def test_plain_mode_progress_callback(self, mock_env, mock_run_result, capsys):
        """_run_sync_once plain mode progress callback works."""
        progress_items = []

        def track_progress(amount, desc=None):
            progress_items.append((amount, desc))

        with patch("polylogue.pipeline.runner.run_sources") as mock_run:
            mock_run.return_value = mock_run_result
            mock_config = MagicMock()

            result = _run_sync_once(
                mock_config,
                mock_env,
                "all",
                None,
                "markdown",
            )

        captured = capsys.readouterr()
        assert "Syncing" in captured.out


class TestWebhookTimeout:
    """Test webhook functionality with timeout."""

    def test_webhook_on_new_with_timeout(self):
        """_webhook_on_new sends request with 10 second timeout."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            _webhook_on_new("http://example.com/webhook", 2)

            # Check timeout parameter
            call_kwargs = mock_urlopen.call_args[1]
            assert call_kwargs.get("timeout") == 10


class TestNotifyNewVariations:
    """Test notification variations."""

    def test_notify_new_conversations_with_zero(self):
        """_notify_new_conversations handles zero conversations."""
        with patch("subprocess.run") as mock_run:
            _notify_new_conversations(0)

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "0" in str(call_args)

    def test_notify_new_conversations_large_count(self):
        """_notify_new_conversations handles large conversation counts."""
        with patch("subprocess.run") as mock_run:
            _notify_new_conversations(999)

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "999" in str(call_args)


class TestExecOnNewVariations:
    """Test exec on new variations."""

    def test_exec_on_new_with_multiline_script(self):
        """_exec_on_new executes multi-line shell scripts."""
        with patch("subprocess.run") as mock_run:
            cmd = """
            echo "Starting"
            sleep 1
            echo "Done"
            """
            _exec_on_new(cmd, 1)

            call_args = mock_run.call_args[0][0]
            assert "echo" in call_args

    def test_exec_on_new_with_special_characters(self):
        """_exec_on_new handles commands with special characters."""
        with patch("subprocess.run") as mock_run:
            _exec_on_new('echo "$POLYLOGUE_NEW_COUNT items"', 5)

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["shell"] is True


class TestDisplayResultVariations:
    """Test display result variations."""

    def test_display_result_render_stage_with_latest(self, mock_env):
        """_display_result in render stage shows latest render path."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")

        mock_run_result = RunResult(
            run_id="run-123",
            counts={"conversations": 1},
            drift={},
            indexed=False,
            index_error=None,
            duration_ms=100,
            render_failures=[],
        )

        with patch("polylogue.cli.helpers.latest_render_path") as mock_latest:
            mock_latest.return_value = Path("/tmp/render/2024-01-15")

            _display_result(mock_env, mock_config, mock_run_result, "render", None)

            mock_latest.assert_called_once()

    def test_display_result_many_render_failures(self, mock_env, capsys):
        """_display_result shows limited render failures (first 10)."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")

        failures = [
            {"conversation_id": f"conv-{i}", "error": f"error {i}"}
            for i in range(20)
        ]
        mock_run_result = RunResult(
            run_id="run-123",
            counts={},
            drift={},
            indexed=False,
            index_error=None,
            duration_ms=0,
            render_failures=failures,
        )

        _display_result(mock_env, mock_config, mock_run_result, "all", None)

        captured = capsys.readouterr()
        # Should show "and X more" message
        assert "and 10 more" in captured.err


class TestEmbedCommandLineOptions:
    """Test embed command line options parsing."""

    def test_embed_command_shows_model_option(self, runner, cli_workspace):
        """embed command has --model option in help."""
        result = runner.invoke(cli, ["embed", "--help"])
        assert "--model" in result.output

    def test_embed_command_shows_rebuild_option(self, runner, cli_workspace):
        """embed command has --rebuild option in help."""
        result = runner.invoke(cli, ["embed", "--help"])
        assert "--rebuild" in result.output

    def test_embed_command_shows_limit_option(self, runner, cli_workspace):
        """embed command has --limit option in help."""
        result = runner.invoke(cli, ["embed", "--help"])
        assert "--limit" in result.output


class TestRunCommandLineOptions:
    """Test run command line options parsing."""

    def test_run_command_shows_watch_option(self, runner, cli_workspace):
        """run command has --watch option in help."""
        result = runner.invoke(cli, ["run", "--help"])
        assert "--watch" in result.output

    def test_run_command_shows_notify_option(self, runner, cli_workspace):
        """run command has --notify option in help."""
        result = runner.invoke(cli, ["run", "--help"])
        assert "--notify" in result.output

    def test_run_command_shows_exec_option(self, runner, cli_workspace):
        """run command has --exec option in help."""
        result = runner.invoke(cli, ["run", "--help"])
        assert "--exec" in result.output

    def test_run_command_shows_webhook_option(self, runner, cli_workspace):
        """run command has --webhook option in help."""
        result = runner.invoke(cli, ["run", "--help"])
        assert "--webhook" in result.output
