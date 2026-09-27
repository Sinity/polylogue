"""Focused tests for embedding commands and helpers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli.shared.embed_stats import show_embedding_stats


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


def _embedding_status_payload(
    *,
    total_sessions: int = 0,
    embedded_sessions: int = 0,
    embedded_messages: int = 0,
    pending_sessions: int = 0,
    retrieval_bands: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    status = "empty" if total_sessions == 0 else "none" if embedded_sessions == 0 else "partial"
    if total_sessions > 0 and pending_sessions == 0 and embedded_sessions > 0:
        status = "complete"
    return {
        "config_enabled": False,
        "has_voyage_api_key": False,
        "daemon_stage_enabled": False,
        "configured_model": "voyage-4",
        "configured_dimension": 1024,
        "monthly_cost_cap_usd": 5.0,
        "status": status,
        "total_sessions": total_sessions,
        "embedded_sessions": embedded_sessions,
        "blocked_sessions": 0,
        "embedded_messages": embedded_messages,
        "pending_sessions": pending_sessions,
        "pending_messages": None,
        "pending_messages_exact": False,
        "embedding_coverage_percent": round(
            embedded_sessions / total_sessions * 100,
            1,
        )
        if total_sessions
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
        "failure_details": [],
        "total_estimated_cost_usd": 0.0,
        "latest_catchup_run": None,
        "latest_material_catchup_run": None,
        "next_action": {
            "code": "archive_empty" if total_sessions == 0 else "set_voyage_key",
            "command": None if total_sessions == 0 else "polylogue ops embed enable --voyage-api-key ...",
            "reason": "Archive contains no sessions to embed."
            if total_sessions == 0
            else "Semantic retrieval needs a Voyage API key before embedding can run.",
        },
    }


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
                total_sessions=int(query_results[0][0]),
                embedded_sessions=int(query_results[1][0]),
                embedded_messages=int(query_results[2][0]),
                pending_sessions=int(query_results[3][0]),
            ),
        ):
            show_embedding_stats(mock_env)

        captured = capsys.readouterr()
        assert "Embedding Statistics" in captured.out
        assert f"Session coverage:     {expected_coverage}" in captured.out
        assert f"Pending:              {expected_pending}" in captured.out
        assert "Retrieval ready:" in captured.out

    def test_show_stats_embedding_status_missing(self, mock_env: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        with patch(
            "polylogue.cli.shared.embed_stats.embedding_status_payload",
            return_value=_embedding_status_payload(total_sessions=100, pending_sessions=100),
        ):
            show_embedding_stats(mock_env)

        captured = capsys.readouterr()
        assert "Embedding Statistics" in captured.out

    def test_show_stats_json_output(self, mock_env: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        with patch(
            "polylogue.cli.shared.embed_stats.embedding_status_payload",
            return_value=_embedding_status_payload(
                total_sessions=100,
                embedded_sessions=40,
                embedded_messages=200,
                pending_sessions=60,
                retrieval_bands={
                    "transcript_embeddings": {"ready": False, "status": "partial"},
                    "evidence_retrieval": {"ready": True, "status": "ready"},
                },
            ),
        ):
            show_embedding_stats(mock_env, json_output=True)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "partial"
        assert payload["embedded_sessions"] == 40
        assert payload["pending_sessions"] == 60
        assert payload["retrieval_ready"] is True
        assert payload["retrieval_bands"]["evidence_retrieval"]["ready"] is True
