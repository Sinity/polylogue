"""Focused tests for embedding commands and helpers."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.commands.embed import (
    _embed_batch,
    _embed_single,
    _show_embedding_stats,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_env():
    from rich.console import Console

    env = MagicMock()
    env.ui = MagicMock()
    env.ui.plain = True
    env.ui.console = Console()
    env.ui.confirm.return_value = True
    env.ui.summary = MagicMock()
    env.config = MagicMock(db_path=None)
    env.repository = MagicMock()
    return env


@pytest.fixture
def mock_env_rich():
    from rich.console import Console

    env = MagicMock()
    env.ui = MagicMock()
    env.ui.plain = False
    env.ui.console = Console()
    env.ui.confirm.return_value = True
    env.ui.summary = MagicMock()
    env.config = MagicMock(db_path=None)
    env.repository = MagicMock()
    return env


@pytest.fixture
def mock_conversation():
    conv = MagicMock()
    conv.id = "conv-123"
    conv.title = "Test Conversation"
    return conv


_MOCK_MESSAGES = [
    {"message_id": "m1", "text": "Hello"},
    {"message_id": "m2", "text": "World"},
]


@pytest.fixture
def mock_repository():
    repo = MagicMock()
    repo.backend = MagicMock()
    repo.backend.get_messages = AsyncMock(return_value=_MOCK_MESSAGES)
    return repo


@pytest.fixture
def mock_repository_async(mock_conversation):
    repo = MagicMock()
    repo.view = AsyncMock(return_value=mock_conversation)
    repo.get_messages = AsyncMock(return_value=_MOCK_MESSAGES)
    return repo


class TestShowEmbeddingStats:
    @pytest.mark.parametrize(
        ("query_results", "expected_coverage", "expected_pending"),
        [
            ([(100,), (50,), (200,), (50,)], "50.0%", "50"),
            ([(0,), (0,), (0,), (0,)], "0.0%", "0"),
            ([(200,), (100,), (500,), (100,)], "50.0%", "100"),
        ],
    )
    def test_show_stats_variants(self, mock_env, capsys, query_results, expected_coverage, expected_pending):
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [MagicMock(fetchone=MagicMock(return_value=qr)) for qr in query_results]

        with patch("polylogue.storage.backends.connection.open_connection") as mock_open:
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            _show_embedding_stats(mock_env)

        captured = capsys.readouterr()
        assert "Embedding Statistics" in captured.out
        assert f"Coverage:               {expected_coverage}" in captured.out
        assert f"Pending:                {expected_pending}" in captured.out

    def test_show_stats_embedding_status_missing(self, mock_env, capsys):
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value=(100,))),
            sqlite3.OperationalError("table does not exist"),
            sqlite3.OperationalError("table does not exist"),
            sqlite3.OperationalError("table does not exist"),
        ]

        with patch("polylogue.storage.backends.connection.open_connection") as mock_open:
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            _show_embedding_stats(mock_env)

        captured = capsys.readouterr()
        assert "Embedding Statistics" in captured.out


class TestEmbedSingle:
    def test_embed_single_success(self, mock_env, mock_repository_async, capsys):
        mock_vec_provider = MagicMock()
        _embed_single(mock_env, mock_repository_async, mock_vec_provider, "conv-123")
        mock_vec_provider.upsert.assert_called_once_with(
            "conv-123", mock_repository_async.get_messages.return_value
        )
        captured = capsys.readouterr()
        assert "Embedding 2 messages" in captured.out
        assert "✓ Embedded" in captured.out

    def test_embed_single_conversation_not_found(self, mock_env, mock_repository_async):
        mock_repository_async.view = AsyncMock(return_value=None)
        with pytest.raises(click.Abort):
            _embed_single(mock_env, mock_repository_async, MagicMock(), "nonexistent")

    def test_embed_single_no_messages(self, mock_env, mock_repository_async, capsys):
        mock_repository_async.get_messages = AsyncMock(return_value=[])
        mock_vec_provider = MagicMock()
        _embed_single(mock_env, mock_repository_async, mock_vec_provider, "conv-123")
        mock_vec_provider.upsert.assert_not_called()
        assert "No messages to embed" in capsys.readouterr().out

    def test_embed_single_upsert_exception(self, mock_env, mock_repository_async):
        mock_vec_provider = MagicMock()
        mock_vec_provider.upsert.side_effect = ValueError("API error")
        with pytest.raises(click.Abort):
            _embed_single(mock_env, mock_repository_async, mock_vec_provider, "conv-123")


class TestEmbedBatch:
    @pytest.mark.parametrize(
        ("num_convs", "limit", "rebuild", "expected_output"),
        [
            (0, None, False, "All conversations are already embedded"),
            (2, None, False, "Embedding 2 conversations"),
            (3, 2, False, "Embedding 2 conversations"),
            (3, None, True, "Embedding 3 conversations"),
        ],
    )
    def test_embed_batch_variants(self, mock_env, mock_repository, capsys, num_convs, limit, rebuild, expected_output):
        mock_env.ui.console = MagicMock()
        mock_vec_provider = MagicMock()
        convs = [{"conversation_id": f"conv-{i}", "title": f"Test {i}"} for i in range(1, num_convs + 1)]

        with patch("polylogue.storage.backends.connection.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchmany.side_effect = [convs, []]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            kwargs = {"limit": limit, "rebuild": rebuild} if limit is not None else {"rebuild": rebuild}
            _embed_batch(mock_env, mock_repository, mock_vec_provider, **kwargs)

        assert expected_output in capsys.readouterr().out

    def test_embed_batch_rebuild_flag(self, mock_env, mock_repository):
        mock_vec_provider = MagicMock()
        with patch("polylogue.storage.backends.connection.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchmany.side_effect = [[], []]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            _embed_batch(mock_env, mock_repository, mock_vec_provider, rebuild=True)
        assert any("ORDER BY updated_at DESC" in str(call) for call in mock_conn.execute.call_args_list)

    def test_embed_batch_error_handling(self, mock_env, mock_repository, capsys):
        mock_vec_provider = MagicMock()
        mock_repository.backend.get_messages = AsyncMock(
            side_effect=[[{"message_id": "m1"}], ValueError("Embed failed"), [{"message_id": "m3"}]]
        )
        convs = [
            {"conversation_id": "conv-1", "title": "Test 1"},
            {"conversation_id": "conv-2", "title": "Test 2"},
            {"conversation_id": "conv-3", "title": "Test 3"},
        ]
        with patch("polylogue.storage.backends.connection.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchmany.side_effect = [convs, []]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            _embed_batch(mock_env, mock_repository, mock_vec_provider)
        assert "error" in capsys.readouterr().out.lower()


class TestEmbedCommand:
    def test_embed_command_missing_api_key(self, runner, cli_workspace):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": ""}, clear=False):
            result = runner.invoke(cli, ["--plain", "embed"])
        assert result.exit_code != 0
        assert "VOYAGE_API_KEY" in result.output or "Error" in result.output

    def test_embed_command_help(self, runner, cli_workspace):
        result = runner.invoke(cli, ["embed", "--help"])
        assert result.exit_code == 0
        assert "embed" in result.output.lower()

    @pytest.mark.parametrize("option", ["--stats", "--model", "--rebuild", "--limit", "--conversation"])
    def test_embed_command_help_lists_options(self, runner, cli_workspace, option):
        result = runner.invoke(cli, ["embed", "--help"])
        assert option in result.output

    def test_embed_command_stats_short_circuit(self, runner, cli_workspace):
        with patch("polylogue.cli.commands.embed._show_embedding_stats") as mock_stats:
            result = runner.invoke(cli, ["--plain", "embed", "--stats"])
        assert result.exit_code == 0
        mock_stats.assert_called_once()

    def test_embed_command_single_conversation_dispatches(self, runner, cli_workspace):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "key"}, clear=False), patch(
            "polylogue.storage.search_providers.create_vector_provider"
        ) as mock_create, patch("polylogue.cli.commands.embed._embed_single") as mock_embed_single:
            mock_create.return_value = MagicMock(model="voyage-4")
            result = runner.invoke(cli, ["--plain", "embed", "--conversation", "conv-123"])
        assert result.exit_code == 0
        mock_embed_single.assert_called_once()

    def test_embed_command_batch_dispatches_limit_and_rebuild(self, runner, cli_workspace):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "key"}, clear=False), patch(
            "polylogue.storage.search_providers.create_vector_provider"
        ) as mock_create, patch("polylogue.cli.commands.embed._embed_batch") as mock_embed_batch:
            mock_create.return_value = MagicMock(model="voyage-4")
            result = runner.invoke(cli, ["--plain", "embed", "--rebuild", "--limit", "5"])
        assert result.exit_code == 0
        mock_embed_batch.assert_called_once()
        assert mock_embed_batch.call_args.kwargs == {"rebuild": True, "limit": 5}

    def test_embed_command_sets_non_default_model(self, runner, cli_workspace):
        vec_provider = MagicMock(model="voyage-4")
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "key"}, clear=False), patch(
            "polylogue.storage.search_providers.create_vector_provider",
            return_value=vec_provider,
        ), patch("polylogue.cli.commands.embed._embed_batch"):
            result = runner.invoke(cli, ["--plain", "embed", "--model", "voyage-4-lite"])
        assert result.exit_code == 0
        assert vec_provider.model == "voyage-4-lite"


class TestEmbedBatchRichMode:
    @pytest.mark.parametrize(
        ("num_convs", "messages_side_effect", "exception_type"),
        [
            (2, [{"message_id": "m1"}] * 2, None),
            (2, [[{"message_id": "m1"}], []], None),
            (1, [{"message_id": "m1"}], RuntimeError),
        ],
    )
    def test_embed_batch_rich_mode_variants(self, mock_env_rich, mock_repository, capsys, num_convs, messages_side_effect, exception_type):
        mock_vec_provider = MagicMock()
        if exception_type:
            mock_vec_provider.upsert.side_effect = exception_type("API timeout")
        elif isinstance(messages_side_effect, list) and messages_side_effect and not isinstance(messages_side_effect[0], dict):
            mock_repository.backend.get_messages = AsyncMock(side_effect=messages_side_effect)
        else:
            mock_repository.backend.get_messages = AsyncMock(return_value=messages_side_effect)

        convs = [{"conversation_id": f"conv-{i}", "title": f"Test {i}"} for i in range(1, num_convs + 1)]
        with patch("polylogue.storage.backends.connection.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchmany.side_effect = [convs, []]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            _embed_batch(mock_env_rich, mock_repository, mock_vec_provider)

        captured = capsys.readouterr()
        if exception_type is None:
            assert "Embedding" in captured.out or "Embedded" in captured.out
        else:
            assert "error" in captured.out.lower()
