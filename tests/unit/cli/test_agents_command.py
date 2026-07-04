"""CLI tests for the agent coordination projections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.agents import agents_command
from polylogue.coordination.payloads import (
    AgentCoordinationPayload,
    CoordinationLimitsPayload,
    CoordinationProvenancePayload,
    CoordinationRepoPayload,
    CoordinationSelfPayload,
    CoordinationView,
    CoordinationWorkItemPayload,
)


def _payload(view: str = "status") -> AgentCoordinationPayload:
    provenance = CoordinationProvenancePayload(source="test", confidence=1.0, freshness="fixture")
    return AgentCoordinationPayload(
        view=cast(CoordinationView, view),
        generated_at="2026-07-04T18:00:00+00:00",
        repo=CoordinationRepoPayload(
            cwd="/repo",
            root="/repo",
            branch="feature/test",
            head="abc123",
            provenance=provenance,
        ),
        self=CoordinationSelfPayload(
            agent_kind="codex", pid=123, cwd="/repo", branch="feature/test", provenance=provenance
        ),
        work_item=CoordinationWorkItemPayload(
            source="beads", ref="polylogue-s7ae.1", confidence=0.95, provenance=provenance
        ),
        limits=CoordinationLimitsPayload(peer_limit=10, resource_limit=10, changed_path_limit=40, command_chars=220),
        provenance=(provenance,),
    )


def test_agents_status_json_uses_shared_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Path | None, int]] = []

    def fake_build(*, view: str, cwd: Path | None, limit: int) -> AgentCoordinationPayload:
        calls.append((view, cwd, limit))
        return _payload(view)

    monkeypatch.setattr("polylogue.cli.commands.agents.build_coordination_envelope", fake_build)

    result = CliRunner().invoke(
        agents_command,
        ["status", "--cwd", "/repo", "--limit", "3", "--json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    body = json.loads(result.output)
    assert body["view"] == "status"
    assert body["work_item"]["ref"] == "polylogue-s7ae.1"
    assert calls == [("status", Path("/repo"), 3)]


def test_agents_work_item_text_is_compact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "polylogue.cli.commands.agents.build_coordination_envelope", lambda **kwargs: _payload(kwargs["view"])
    )

    result = CliRunner().invoke(
        agents_command,
        ["work-item", "--format", "text"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Agent coordination (work-item)" in result.output
    assert "polylogue-s7ae.1" in result.output


def test_agents_status_markdown_and_tree_use_shared_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "polylogue.cli.commands.agents.build_coordination_envelope", lambda **kwargs: _payload(kwargs["view"])
    )

    markdown = CliRunner().invoke(agents_command, ["status", "--format", "markdown"], catch_exceptions=False)
    tree = CliRunner().invoke(agents_command, ["status", "--format", "tree"], catch_exceptions=False)

    assert markdown.exit_code == 0
    assert "# Agent Coordination Mission Control" in markdown.output
    assert "polylogue-s7ae.1" in markdown.output
    assert tree.exit_code == 0
    assert "coordination" in tree.output
    assert "work polylogue-s7ae.1" in tree.output
