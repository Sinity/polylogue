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
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.commands.embed import (
    _embed_batch,
    _embed_single,
    _show_embedding_stats,
)
from polylogue.cli.commands.run import (
    _display_result,
    _exec_on_new,
    _notify_new_conversations,
    _run_sync_once,
    _webhook_on_new,
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

    @pytest.mark.parametrize("query_results,expected_coverage,expected_pending", [
        # (total_convs, embedded_convs, embedded_msgs, pending), coverage, pending_text
        ([(100,), (50,), (200,), (50,)], "50.0%", "50"),
        ([(0,), (0,), (0,), (0,)], "0.0%", "0"),
        ([(200,), (100,), (500,), (100,)], "50.0%", "100"),
    ])
    def test_show_stats_variants(self, mock_env, capsys, query_results, expected_coverage, expected_pending):
        """_show_embedding_stats displays statistics for various query results."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value=qr)) for qr in query_results
        ]

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            _show_embedding_stats(mock_env)

        captured = capsys.readouterr()
        assert "Embedding Statistics" in captured.out
        assert f"Coverage:               {expected_coverage}" in captured.out
        assert f"Pending:                {expected_pending}" in captured.out

    def test_show_stats_embedding_status_missing(self, mock_env, capsys):
        """_show_embedding_stats handles missing embedding_status table."""
        mock_conn = MagicMock()

        def side_effect(*args, **kwargs):
            result = MagicMock()
            # First query succeeds (total_convs)
            if "COUNT(*) FROM conversations" in args[0]:
                result.fetchone.return_value = (100,)
            # embedding_status query fails
            elif "embedding_status" in args[0] or "message_embeddings" in args[0]:
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

        with pytest.raises(click.Abort):
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

        with pytest.raises(click.Abort):
            _embed_single(mock_env, mock_repository, mock_vec_provider, "conv-123")


# =============================================================================
# TEST CLASS: _embed_batch
# =============================================================================


class TestEmbedBatch:
    """Test _embed_batch function."""

    @pytest.mark.parametrize("num_convs,limit,rebuild,expected_output", [
        # num_convs, limit, rebuild, expected_output_substring
        (0, None, False, "All conversations are already embedded"),
        (2, None, False, "Embedding 2 conversations"),
        (3, 2, False, "Embedding 2 conversations"),
        (3, None, True, "Embedding 3 conversations"),
        (3, None, False, "Embedding 3 conversations"),
    ])
    def test_embed_batch_variants(self, mock_env, mock_repository, capsys, num_convs, limit, rebuild, expected_output):
        """_embed_batch handles various configurations of conversations, limits, and rebuild flag."""
        mock_env.ui.console = MagicMock()  # Plain mode
        mock_vec_provider = MagicMock()

        convs = [{"conversation_id": f"conv-{i}", "title": f"Test {i}"} for i in range(1, num_convs + 1)]

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = convs
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            kwargs = {"limit": limit, "rebuild": rebuild} if limit or rebuild else {"rebuild": rebuild}
            _embed_batch(mock_env, mock_repository, mock_vec_provider, **kwargs)

        captured = capsys.readouterr()
        assert expected_output in captured.out

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

    @pytest.mark.parametrize("option", ["--stats", "--model", "--rebuild", "--limit"])
    def test_embed_command_options(self, runner, cli_workspace, option):
        """embed command has expected options in help."""
        result = runner.invoke(cli, ["embed", "--help"])
        assert option in result.output


# =============================================================================
# TEST CLASS: _run_sync_once
# =============================================================================


class TestRunSyncOnce:
    """Test _run_sync_once function."""

    @pytest.mark.parametrize("ui_env,render_format,has_plan_snapshot", [
        # ui_fixture, render_format, include_plan_snapshot
        ("mock_env", "markdown", False),
        ("mock_env_rich", "html", False),
        ("mock_env", "markdown", True),
    ])
    def test_run_sync_once_variants(self, request, mock_run_result, mock_plan_result, capsys, ui_env, render_format, has_plan_snapshot):
        """_run_sync_once handles different UI modes, formats, and plan configurations."""
        env = request.getfixturevalue(ui_env)

        with patch("polylogue.cli.commands.run.run_sources") as mock_run:
            mock_run.return_value = mock_run_result
            mock_config = MagicMock()

            kwargs = {
                "cfg": mock_config,
                "env": env,
                "stage": "all",
                "selected_sources": None,
                "render_format": render_format,
            }
            if has_plan_snapshot:
                kwargs["plan_snapshot"] = mock_plan_result
                # Verify plan was passed
                _run_sync_once(**kwargs)
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["plan"] == mock_plan_result
            else:
                result = _run_sync_once(**kwargs)
                assert result.run_id == "run-123"
                if env.ui.plain:
                    captured = capsys.readouterr()
                    assert "Syncing" in captured.out


# =============================================================================
# TEST CLASS: _display_result
# =============================================================================


class TestDisplayResult:
    """Test _display_result function."""

    @pytest.mark.parametrize("stage,sources,has_failures,failure_count", [
        # stage, sources, has_render_failures, num_failures
        ("all", None, False, 0),
        ("index", None, False, 0),
        ("render", ["source1", "source2"], False, 0),
        ("all", None, True, 2),
        ("all", None, True, 20),
    ])
    def test_display_result_variants(self, mock_env, capsys, stage, sources, has_failures, failure_count):
        """_display_result handles various stages, sources, and failure scenarios."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")

        if has_failures:
            failures = [
                {"conversation_id": f"conv-{i}", "error": f"error {i}"}
                for i in range(failure_count)
            ]
            mock_run_result = RunResult(
                run_id="run-123",
                counts={"conversations": failure_count},
                drift={},
                indexed=stage == "index",
                index_error=None if stage != "all" else None,
                duration_ms=100,
                render_failures=failures,
            )
        else:
            mock_run_result = RunResult(
                run_id="run-123",
                counts={"conversations": 1 if not sources else len(sources)},
                drift={"conversations": {"new": 2, "updated": 1, "unchanged": 5}} if stage == "all" else {},
                indexed=stage == "index",
                index_error=None,
                duration_ms=100,
                render_failures=[],
            )

        with patch("polylogue.cli.helpers.latest_render_path") as mock_latest:
            mock_latest.return_value = Path("/tmp/render/2024-01-15") if stage == "render" else None

            _display_result(mock_env, mock_config, mock_run_result, stage, sources)

        mock_env.ui.summary.assert_called_once()
        title, lines = mock_env.ui.summary.call_args[0]
        assert "Sync" in title or stage in title or any(s in title for s in (sources or []))

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
        assert "Index error" in captured.err


# =============================================================================
# TEST CLASS: _notify_new_conversations
# =============================================================================


class TestNotifyNewConversations:
    """Test _notify_new_conversations function."""

    @pytest.mark.parametrize("num_conversations", [1, 3, 999])
    def test_notify_new_conversations_variants(self, num_conversations):
        """_notify_new_conversations calls notify-send with various conversation counts."""
        with patch("subprocess.run") as mock_run:
            _notify_new_conversations(num_conversations)

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "notify-send" in call_args
            assert str(num_conversations) in str(call_args)

    def test_notify_zero_is_noop(self):
        """_notify_new_conversations does nothing for 0 new conversations."""
        with patch("subprocess.run") as mock_run:
            _notify_new_conversations(0)
            mock_run.assert_not_called()

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

    @pytest.mark.parametrize("command,num_conversations", [
        ("echo $POLYLOGUE_NEW_COUNT", 5),
        ("ls -la", 1),
        ("my-script --count 5", 1),
    ])
    def test_exec_on_new_variants(self, command, num_conversations):
        """_exec_on_new executes commands with correct environment and settings.

        Commands run with shell=False for security.  Shell metacharacters
        like ;, &&, |, backticks, and $() are rejected at construction time.
        """
        with patch("subprocess.run") as mock_run:
            _exec_on_new(command, num_conversations)

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            # shell=False is the default when passing a list — verify no shell kwarg
            assert call_kwargs[1].get("shell") is not True
            assert call_kwargs[1]["env"]["POLYLOGUE_NEW_COUNT"] == str(num_conversations)

    @pytest.mark.parametrize("dangerous_command", [
        "echo hello; rm -rf /",
        "echo hello && cat /etc/passwd",
        "echo `whoami`",
        "echo $(id)",
    ])
    def test_exec_on_new_rejects_dangerous_commands(self, dangerous_command):
        """_exec_on_new rejects commands with shell metacharacters."""
        with pytest.raises(ValueError, match="unsafe"):
            _exec_on_new(dangerous_command, 1)


# =============================================================================
# TEST CLASS: _webhook_on_new
# =============================================================================


class TestWebhookOnNew:
    """Test _webhook_on_new function."""

    @pytest.mark.parametrize("url,num_conversations", [
        ("http://example.com/webhook", 3),
        ("http://example.com/webhook", 5),
        ("https://api.example.com/sync", 1),
    ])
    def test_webhook_on_new_variants(self, url, num_conversations):
        """_webhook_on_new sends POST requests with correct payloads and timeouts."""
        # Mock SSRF validation (DNS resolution unavailable in sandboxed tests)
        fake_addrinfo = [(2, 1, 6, "", ("93.184.216.34", 443))]
        with patch("polylogue.pipeline.events.socket.getaddrinfo", return_value=fake_addrinfo):
            with patch("urllib.request.urlopen") as mock_urlopen:
                with patch("urllib.request.Request") as mock_request:
                    _webhook_on_new(url, num_conversations)

                    mock_urlopen.assert_called_once()
                    call_kwargs = mock_urlopen.call_args[1]
                    assert call_kwargs.get("timeout") == 10

                    # Check request details
                    mock_request.assert_called_once()
                    call_args = mock_request.call_args[0]
                    assert url in str(call_args)

                # Check payload
                call_kwargs = mock_request.call_args[1]
                payload = json.loads(call_kwargs["data"].decode())
                assert payload["event"] == "sync"
                assert payload["new_conversations"] == num_conversations

    def test_webhook_on_new_exception_logged(self):
        """_webhook_on_new logs exceptions without raising."""
        fake_addrinfo = [(2, 1, 6, "", ("93.184.216.34", 80))]
        with patch("polylogue.pipeline.events.socket.getaddrinfo", return_value=fake_addrinfo):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = ConnectionError("Connection failed")

                # Should not raise
                _webhook_on_new("http://example.com/webhook", 1)


# =============================================================================
# TEST CLASS: run_command (Click command)
# =============================================================================


class TestEmbedRunCommand:
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

    @pytest.mark.parametrize("option", ["--stage", "--preview", "--watch", "--notify", "--exec", "--webhook"])
    def test_run_command_options(self, runner, cli_workspace, option):
        """run command has expected options in help."""
        result = runner.invoke(cli, ["run", "--help"])
        assert option in result.output


# =============================================================================
# TEST CLASS: sources_command (Click command)
# =============================================================================


class TestEmbedSourcesCommand:
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

    @pytest.mark.parametrize("num_convs,messages_side_effect,exception_type", [
        # (num_conversations, get_messages_side_effect, expected_exception_or_none)
        (2, [{"message_id": "m1"}] * 2, None),
        (2, [[{"message_id": "m1"}], []], None),
        (1, [{"message_id": "m1"}], RuntimeError),
    ])
    def test_embed_batch_rich_mode_variants(self, mock_env_rich, mock_repository, capsys, num_convs, messages_side_effect, exception_type):
        """_embed_batch in rich mode handles various message and exception scenarios."""
        mock_vec_provider = MagicMock()

        if exception_type:
            mock_vec_provider.upsert.side_effect = exception_type("API timeout")
        else:
            if isinstance(messages_side_effect, list) and len(messages_side_effect) > 0 and not isinstance(messages_side_effect[0], dict):
                # It's a list of lists/results
                mock_repository.backend.get_messages.side_effect = messages_side_effect
            else:
                mock_repository.backend.get_messages.return_value = messages_side_effect

        convs = [{"conversation_id": f"conv-{i}", "title": f"Test {i}"} for i in range(1, num_convs + 1)]

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = convs
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            _embed_batch(mock_env_rich, mock_repository, mock_vec_provider)

        if not exception_type:
            captured = capsys.readouterr()
            assert "Embedding" in captured.out or "Embedded" in captured.out
        else:
            captured = capsys.readouterr()
            assert "error" in captured.out.lower()


class TestPlainModeProgress:
    """Test progress callback in plain mode."""

    def test_plain_mode_progress_callback(self, mock_env, mock_run_result, capsys):
        """_run_sync_once plain mode progress callback works."""
        progress_items = []

        def track_progress(amount, desc=None):
            progress_items.append((amount, desc))

        with patch("polylogue.cli.commands.run.run_sources") as mock_run:
            mock_run.return_value = mock_run_result
            mock_config = MagicMock()

            _run_sync_once(
                mock_config,
                mock_env,
                "all",
                None,
                "markdown",
            )

        captured = capsys.readouterr()
        assert "Syncing" in captured.out
