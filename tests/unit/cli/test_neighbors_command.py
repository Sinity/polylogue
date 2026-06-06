"""CLI tests for neighboring-session discovery."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from click.testing import CliRunner

from polylogue.archive.models import SessionSummary
from polylogue.archive.session.neighbor_candidates import NeighborReason, SessionNeighborCandidate
from polylogue.cli.commands.neighbors import neighbors_command
from polylogue.core.enums import Origin
from polylogue.types import SessionId


def _candidate() -> SessionNeighborCandidate:
    return SessionNeighborCandidate(
        summary=SessionSummary(
            id=SessionId("candidate"),
            origin=Origin.CODEX_SESSION,
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
        source_session_id="target",
    )


def test_neighbors_command_emits_json_payload(cli_runner: CliRunner) -> None:
    env = MagicMock()
    env.polylogue = MagicMock()
    env.polylogue.neighbor_candidates = AsyncMock(return_value=[_candidate()])

    result = cli_runner.invoke(
        neighbors_command,
        ["--id", "target", "--origin", "codex-session", "--format", "json"],
        obj=env,
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    neighbor = payload["result"]["neighbors"][0]
    assert neighbor["session"]["id"] == "candidate"
    assert neighbor["reasons"][0]["kind"] == "same_title"
    env.polylogue.neighbor_candidates.assert_called_once_with(
        session_id="target",
        query=None,
        provider="codex",
        limit=10,
        window_hours=24,
    )


def test_neighbors_command_requires_id_or_query(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(neighbors_command, [], obj=MagicMock())

    assert result.exit_code != 0
    assert "provide --id" in result.output
    assert "--query" in result.output
