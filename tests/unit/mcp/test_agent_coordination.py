"""MCP coordination tool and prompt contracts."""

from __future__ import annotations

import json
from typing import cast

import pytest

from polylogue.coordination.payloads import (
    AgentCoordinationPayload,
    CoordinationLimitsPayload,
    CoordinationProvenancePayload,
    CoordinationRepoPayload,
    CoordinationSelfPayload,
    CoordinationView,
    CoordinationWorkItemPayload,
)
from tests.infra.mcp import MCPServerUnderTest, invoke_surface


def _payload(view: str = "status") -> AgentCoordinationPayload:
    provenance = CoordinationProvenancePayload(source="test", confidence=1.0, freshness="fixture")
    return AgentCoordinationPayload(
        view=cast(CoordinationView, view),
        generated_at="2026-07-04T18:00:00+00:00",
        repo=CoordinationRepoPayload(cwd="/repo", root="/repo", branch="feature/test", provenance=provenance),
        self=CoordinationSelfPayload(
            agent_kind="codex", pid=123, cwd="/repo", branch="feature/test", provenance=provenance
        ),
        work_item=CoordinationWorkItemPayload(
            source="beads", ref="polylogue-s7ae.1", confidence=0.95, provenance=provenance
        ),
        limits=CoordinationLimitsPayload(peer_limit=10, resource_limit=10, changed_path_limit=40, command_chars=220),
        provenance=(provenance,),
    )


def test_agent_coordination_tool_returns_shared_payload(
    mcp_server: MCPServerUnderTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "polylogue.mcp.server_tools.build_coordination_envelope", lambda **kwargs: _payload(kwargs["view"])
    )

    raw = invoke_surface(mcp_server._tool_manager._tools["agent_coordination"].fn, view="work-item", limit=5)
    body = json.loads(raw)

    assert body["view"] == "work-item"
    assert body["work_item"]["source"] == "beads"
    assert body["work_item"]["ref"] == "polylogue-s7ae.1"


def test_agent_coordination_prompt_embeds_bounded_envelope(
    mcp_server: MCPServerUnderTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("polylogue.coordination.build_coordination_envelope", lambda **kwargs: _payload(kwargs["view"]))

    raw = invoke_surface(mcp_server._prompt_manager._prompts["agent_coordination_brief"].fn, view="conflicts", limit=5)

    assert "bounded coordination envelope" in raw
    assert '"view": "conflicts"' in raw
    assert "Treat overlaps as awareness" in raw
