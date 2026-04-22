"""CLI tests for neighboring-conversation discovery."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from click.testing import CliRunner

from polylogue.cli.commands.neighbors import neighbors_command
from polylogue.lib.models import ConversationSummary
from polylogue.lib.neighbor_candidates import ConversationNeighborCandidate, NeighborReason
from polylogue.types import ConversationId, Provider


def _candidate() -> ConversationNeighborCandidate:
    return ConversationNeighborCandidate(
        summary=ConversationSummary(
            id=ConversationId("candidate"),
            provider=Provider.CODEX,
            title="Archive Lock Retries",
            updated_at=datetime(2026, 4, 22, 14, 0, tzinfo=timezone.utc),
            message_count=2,
        ),
        rank=1,
        score=3.25,
        reasons=(
            NeighborReason(
                kind="same_title",
                detail="same normalized title: Archive Lock Retries",
                weight=3.0,
            ),
        ),
        source_conversation_id="target",
    )


def test_neighbors_command_emits_json_payload(cli_runner: CliRunner) -> None:
    env = MagicMock()
    env.operations = MagicMock()
    env.operations.neighbor_candidates = AsyncMock(return_value=[_candidate()])

    result = cli_runner.invoke(
        neighbors_command,
        ["--id", "target", "--json"],
        obj=env,
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    neighbor = payload["result"]["neighbors"][0]
    assert neighbor["conversation"]["id"] == "candidate"
    assert neighbor["reasons"][0]["kind"] == "same_title"
    env.operations.neighbor_candidates.assert_called_once_with(
        conversation_id="target",
        query=None,
        provider=None,
        limit=10,
        window_hours=24,
    )


def test_neighbors_command_requires_id_or_query(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(neighbors_command, [], obj=MagicMock())

    assert result.exit_code != 0
    assert "provide --id or --query" in result.output
