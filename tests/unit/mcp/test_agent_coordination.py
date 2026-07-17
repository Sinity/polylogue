"""MCP coordination tool and prompt contracts."""

from __future__ import annotations

import json
from typing import cast

import pytest

from polylogue.coordination.payloads import (
    AgentCoordinationPayload,
    CoordinationActivityEpisodePayload,
    CoordinationContextFlowRefPayload,
    CoordinationLimitsPayload,
    CoordinationProofRefPayload,
    CoordinationProvenancePayload,
    CoordinationRepoPayload,
    CoordinationSelfPayload,
    CoordinationSessionTreeNodePayload,
    CoordinationSessionTreePayload,
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
            identity_status="resolved",
            agent_kind="codex",
            logical_id="codex:thread",
            owner_pid=123,
            invocation_pid=456,
            cwd="/repo",
            branch="feature/test",
            session_ref="thread",
            provenance=provenance,
        ),
        work_item=CoordinationWorkItemPayload(
            source="beads", ref="polylogue-s7ae.1", confidence=0.95, provenance=provenance
        ),
        session_trees=(
            CoordinationSessionTreePayload(
                target_session_id="codex-session:thread",
                root_session_id="codex-session:root",
                nodes=(
                    CoordinationSessionTreeNodePayload(
                        session_id="codex-session:thread",
                        source_name="codex-session",
                        depth=1,
                        is_target=True,
                    ),
                ),
                provenance=provenance,
            ),
        ),
        activity_episodes=(
            CoordinationActivityEpisodePayload(
                ref="run:thread",
                session_id="codex-session:thread",
                run_ref="run:thread",
                kind="run",
                status="completed",
                provenance=provenance,
            ),
        ),
        proof_refs=(
            CoordinationProofRefPayload(
                ref="event:tool",
                session_id="codex-session:thread",
                kind="tool_finished",
                status="passed",
                provenance=provenance,
            ),
        ),
        context_flow_refs=(
            CoordinationContextFlowRefPayload(
                ref="context:thread:start",
                session_id="codex-session:thread",
                run_ref="run:thread",
                boundary="session_start",
                provenance=provenance,
            ),
        ),
        limits=CoordinationLimitsPayload(peer_limit=10, resource_limit=10, changed_path_limit=40, command_chars=220),
        provenance=(provenance,),
    )


def test_agent_coordination_tool_returns_shared_payload(
    mcp_server: MCPServerUnderTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_build(**kwargs: object) -> AgentCoordinationPayload:
        calls.append(bool(kwargs["detail"]))
        return _payload(str(kwargs["view"]))

    monkeypatch.setattr("polylogue.mcp.server_tools.build_coordination_envelope", fake_build)

    raw = invoke_surface(
        mcp_server._tool_manager._tools["agent_coordination"].fn,
        view="work-item",
        limit=5,
        detail=True,
    )
    body = json.loads(raw)

    assert body["view"] == "work-item"
    assert body["self"]["identity_status"] == "resolved"
    assert body["self"]["logical_id"] == "codex:thread"
    assert body["self"]["owner_pid"] == 123
    assert body["self"]["invocation_pid"] == 456
    assert body["work_item"]["source"] == "beads"
    assert body["work_item"]["ref"] == "polylogue-s7ae.1"
    assert body["session_trees"][0]["target_session_id"] == "codex-session:thread"
    assert body["activity_episodes"][0]["kind"] == "run"
    assert body["proof_refs"][0]["status"] == "passed"
    assert body["context_flow_refs"][0]["boundary"] == "session_start"
    assert calls == [True]


def test_agent_coordination_compact_cache_is_bypassable(
    mcp_server: MCPServerUnderTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_build(**kwargs: object) -> AgentCoordinationPayload:
        calls.append(bool(kwargs["detail"]))
        return _payload(str(kwargs["view"]))

    monkeypatch.setattr("polylogue.mcp.server_tools.build_coordination_envelope", fake_build)
    tool = mcp_server._tool_manager._tools["agent_coordination"].fn

    invoke_surface(tool, view="status", limit=5)
    invoke_surface(tool, view="status", limit=5)
    invoke_surface(tool, view="status", limit=5, fresh=True)
    invoke_surface(tool, view="status", limit=5, detail=True)

    # The second compact call hits the real server-local cache; fresh and
    # detail requests retain their live/evidence semantics.
    assert calls == [False, False, True]


def test_agent_coordination_prompt_embeds_bounded_envelope(
    mcp_server: MCPServerUnderTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_build(**kwargs: object) -> AgentCoordinationPayload:
        calls.append(bool(kwargs["detail"]))
        return _payload(str(kwargs["view"]))

    monkeypatch.setattr("polylogue.coordination.build_coordination_envelope", fake_build)

    raw = invoke_surface(
        mcp_server._prompt_manager._prompts["agent_coordination_brief"].fn,
        view="conflicts",
        limit=5,
        detail=True,
    )

    assert "bounded coordination envelope" in raw
    assert '"view": "conflicts"' in raw
    assert "Treat overlaps as awareness" in raw
    assert calls == [True]
