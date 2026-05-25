"""Focused tests for embedding commands and helpers."""

from __future__ import annotations

import json
from typing import TypeAlias
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest

from polylogue.cli.shared.embed_runtime import embed_batch, embed_single
from polylogue.cli.shared.embed_stats import show_embedding_stats

MessageRow: TypeAlias = dict[str, str]


@pytest.fixture
def mock_env() -> MagicMock:
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
def mock_env_rich() -> MagicMock:
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
def mock_conversation() -> MagicMock:
    conv = MagicMock()
    conv.id = "conv-123"
    conv.title = "Test Conversation"
    return conv


_MOCK_MESSAGES: list[MessageRow] = [
    {"message_id": "m1", "text": "Hello"},
    {"message_id": "m2", "text": "World"},
]


def _embedding_status_payload(
    *,
    total_conversations: int = 0,
    embedded_conversations: int = 0,
    embedded_messages: int = 0,
    pending_conversations: int = 0,
    retrieval_bands: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    status = "empty" if total_conversations == 0 else "none" if embedded_conversations == 0 else "partial"
    if total_conversations > 0 and pending_conversations == 0 and embedded_conversations > 0:
        status = "complete"
    return {
        "config_enabled": False,
        "has_voyage_api_key": False,
        "daemon_stage_enabled": False,
        "status": status,
        "total_conversations": total_conversations,
        "embedded_conversations": embedded_conversations,
        "embedded_messages": embedded_messages,
        "pending_conversations": pending_conversations,
        "pending_messages": None,
        "pending_messages_exact": False,
        "embedding_coverage_percent": round(
            embedded_conversations / total_conversations * 100,
            1,
        )
        if total_conversations
        else 0.0,
        "retrieval_ready": embedded_messages > 0,
        "freshness_status": status,
        "stale_messages": 0,
        "messages_missing_provenance": 0,
        "oldest_embedded_at": None,
        "newest_embedded_at": None,
        "embedding_models": {},
        "embedding_dimensions": {},
        "retrieval_bands": retrieval_bands or {},
        "failure_count": 0,
        "total_estimated_cost_usd": 0.0,
        "latest_catchup_run": None,
    }


@pytest.fixture
def mock_repository() -> MagicMock:
    repo = MagicMock()
    repo.backend = MagicMock()
    repo.backend.queries = MagicMock()
    repo.backend.queries.get_messages = AsyncMock(return_value=_MOCK_MESSAGES)
    return repo


@pytest.fixture
def mock_repository_async(mock_conversation: MagicMock) -> MagicMock:
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
    def test_show_stats_variants(
        self,
        mock_env: MagicMock,
        capsys: pytest.CaptureFixture[str],
        query_results: list[tuple[int]],
        expected_coverage: str,
        expected_pending: str,
    ) -> None:
        with patch(
            "polylogue.cli.shared.embed_stats.embedding_status_payload",
            return_value=_embedding_status_payload(
                total_conversations=int(query_results[0][0]),
                embedded_conversations=int(query_results[1][0]),
                embedded_messages=int(query_results[2][0]),
                pending_conversations=int(query_results[3][0]),
            ),
        ):
            show_embedding_stats(mock_env)

        captured = capsys.readouterr()
        assert "Embedding Statistics" in captured.out
        assert f"Coverage:              {expected_coverage}" in captured.out
        assert f"Pending:               {expected_pending}" in captured.out
        assert "Retrieval ready:" in captured.out

    def test_show_stats_embedding_status_missing(self, mock_env: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        with patch(
            "polylogue.cli.shared.embed_stats.embedding_status_payload",
            return_value=_embedding_status_payload(total_conversations=100, pending_conversations=100),
        ):
            show_embedding_stats(mock_env)

        captured = capsys.readouterr()
        assert "Embedding Statistics" in captured.out

    def test_show_stats_json_output(self, mock_env: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        with patch(
            "polylogue.cli.shared.embed_stats.embedding_status_payload",
            return_value=_embedding_status_payload(
                total_conversations=100,
                embedded_conversations=40,
                embedded_messages=200,
                pending_conversations=60,
                retrieval_bands={
                    "transcript_embeddings": {"ready": False, "status": "partial"},
                    "evidence_retrieval": {"ready": True, "status": "ready"},
                },
            ),
        ):
            show_embedding_stats(mock_env, json_output=True)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "partial"
        assert payload["embedded_conversations"] == 40
        assert payload["pending_conversations"] == 60
        assert payload["retrieval_ready"] is True
        assert payload["retrieval_bands"]["evidence_retrieval"]["ready"] is True


class TestEmbedSingle:
    def testembed_single_success(
        self, mock_env: MagicMock, mock_repository_async: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_vec_provider = MagicMock()
        embed_single(mock_env, mock_repository_async, mock_vec_provider, "conv-123")
        mock_vec_provider.upsert.assert_called_once_with("conv-123", mock_repository_async.get_messages.return_value)
        captured = capsys.readouterr()
        assert "Embedding 2 messages" in captured.out
        assert "✓ Embedded" in captured.out

    def testembed_single_conversation_not_found(self, mock_env: MagicMock, mock_repository_async: MagicMock) -> None:
        mock_repository_async.view = AsyncMock(return_value=None)
        with pytest.raises(click.Abort):
            embed_single(mock_env, mock_repository_async, MagicMock(), "nonexistent")

    def testembed_single_no_messages(
        self, mock_env: MagicMock, mock_repository_async: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_repository_async.get_messages = AsyncMock(return_value=[])
        mock_vec_provider = MagicMock()
        embed_single(mock_env, mock_repository_async, mock_vec_provider, "conv-123")
        mock_vec_provider.upsert.assert_not_called()
        assert "No messages to embed" in capsys.readouterr().out

    def testembed_single_upsert_exception(self, mock_env: MagicMock, mock_repository_async: MagicMock) -> None:
        mock_vec_provider = MagicMock()
        mock_vec_provider.upsert.side_effect = ValueError("API error")
        with pytest.raises(click.Abort):
            embed_single(mock_env, mock_repository_async, mock_vec_provider, "conv-123")


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
    def testembed_batch_variants(
        self,
        mock_env: MagicMock,
        mock_repository: MagicMock,
        capsys: pytest.CaptureFixture[str],
        num_convs: int,
        limit: int | None,
        rebuild: bool,
        expected_output: str,
    ) -> None:
        mock_env.ui.console = MagicMock()
        mock_vec_provider = MagicMock()
        convs = [{"conversation_id": f"conv-{i}", "title": f"Test {i}"} for i in range(1, num_convs + 1)]

        with patch("polylogue.storage.sqlite.connection.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchmany.side_effect = [convs, []]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            if limit is None:
                embed_batch(mock_env, mock_repository, mock_vec_provider, rebuild=rebuild)
            else:
                embed_batch(mock_env, mock_repository, mock_vec_provider, max_conversations=limit, rebuild=rebuild)

        assert expected_output in capsys.readouterr().out

    def testembed_batch_rebuild_flag(self, mock_env: MagicMock, mock_repository: MagicMock) -> None:
        mock_vec_provider = MagicMock()
        with patch("polylogue.storage.sqlite.connection.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchmany.side_effect = [[], []]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            embed_batch(mock_env, mock_repository, mock_vec_provider, rebuild=True)
        assert any("ORDER BY COALESCE(c.updated_at, '')" in str(call) for call in mock_conn.execute.call_args_list)

    def testembed_batch_error_handling(
        self, mock_env: MagicMock, mock_repository: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_vec_provider = MagicMock()
        mock_repository.backend.queries.get_messages = AsyncMock(
            side_effect=[[{"message_id": "m1"}], ValueError("Embed failed"), [{"message_id": "m3"}]]
        )
        convs = [
            {"conversation_id": "conv-1", "title": "Test 1"},
            {"conversation_id": "conv-2", "title": "Test 2"},
            {"conversation_id": "conv-3", "title": "Test 3"},
        ]
        with patch("polylogue.storage.sqlite.connection.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchmany.side_effect = [convs, []]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            embed_batch(mock_env, mock_repository, mock_vec_provider)
        assert "error" in capsys.readouterr().out.lower()


class TestEmbedBatchRichMode:
    @pytest.mark.parametrize(
        ("num_convs", "messages_side_effect", "exception_type"),
        [
            (2, [{"message_id": "m1"}] * 2, None),
            (2, [[{"message_id": "m1"}], []], None),
            (1, [{"message_id": "m1"}], RuntimeError),
        ],
    )
    def testembed_batch_rich_mode_variants(
        self,
        mock_env_rich: MagicMock,
        mock_repository: MagicMock,
        capsys: pytest.CaptureFixture[str],
        num_convs: int,
        messages_side_effect: object,
        exception_type: type[Exception] | None,
    ) -> None:
        mock_vec_provider = MagicMock()
        if exception_type:
            mock_vec_provider.upsert.side_effect = exception_type("API timeout")
        elif (
            isinstance(messages_side_effect, list)
            and messages_side_effect
            and not isinstance(messages_side_effect[0], dict)
        ):
            mock_repository.backend.queries.get_messages = AsyncMock(side_effect=messages_side_effect)
        else:
            mock_repository.backend.queries.get_messages = AsyncMock(return_value=messages_side_effect)

        convs = [{"conversation_id": f"conv-{i}", "title": f"Test {i}"} for i in range(1, num_convs + 1)]
        with patch("polylogue.storage.sqlite.connection.open_connection") as mock_open:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchmany.side_effect = [convs, []]
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            embed_batch(mock_env_rich, mock_repository, mock_vec_provider)

        captured = capsys.readouterr()
        if exception_type is None:
            assert "Embedding" in captured.out or "Embedded" in captured.out
        else:
            assert "error" in captured.out.lower()
