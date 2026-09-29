"""status(scope="coordination") must dispatch to coordination logic, not archive stats.

Regression test for polylogue-qink: the six-tool cutover surface declared a
"coordination" scope on ``status`` but every scope value except "operation"
fell through to the same generic ``archive.stats()`` projection, so
``status(scope="coordination")`` silently returned archive stats and never
touched :func:`build_coordination_envelope`.
"""

from __future__ import annotations

import json
from pathlib import Path
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
        generated_at="2026-07-18T18:00:00+00:00",
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
            source="beads", ref="polylogue-qink", confidence=0.95, provenance=provenance
        ),
        limits=CoordinationLimitsPayload(peer_limit=10, resource_limit=10, changed_path_limit=40, command_chars=220),
        provenance=(provenance,),
    )


def test_status_coordination_scope_returns_coordination_envelope_not_archive_stats(
    mcp_server: MCPServerUnderTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_build(**kwargs: object) -> AgentCoordinationPayload:
        calls.append(bool(kwargs["detail"]))
        return _payload(str(kwargs["view"]))

    # Both the fresh/detail path (server_cutover.py's own import) and the
    # cached path (CoordinationEnvelopeCache's internal collector, resolved
    # against coordination.envelope's own globals) need patching.
    monkeypatch.setattr("polylogue.coordination.build_coordination_envelope", fake_build)
    monkeypatch.setattr("polylogue.coordination.envelope.build_coordination_envelope", fake_build)

    raw = invoke_surface(mcp_server._tool_manager._tools["status"].fn, scope="coordination")
    body = json.loads(raw)

    assert body["scope"] == "coordination"
    assert "archive" not in body
    assert body["coordination"]["self"]["logical_id"] == "codex:thread"
    assert body["coordination"]["work_item"]["ref"] == "polylogue-qink"
    assert calls == [False]


def test_status_coordination_scope_detail_bypasses_cache(
    mcp_server: MCPServerUnderTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_build(**kwargs: object) -> AgentCoordinationPayload:
        calls.append(bool(kwargs["detail"]))
        return _payload(str(kwargs["view"]))

    monkeypatch.setattr("polylogue.coordination.build_coordination_envelope", fake_build)
    monkeypatch.setattr("polylogue.coordination.envelope.build_coordination_envelope", fake_build)

    tool = mcp_server._tool_manager._tools["status"].fn
    invoke_surface(tool, scope="coordination")
    invoke_surface(tool, scope="coordination")
    invoke_surface(tool, scope="coordination", include=("detail",))

    # The second compact call hits the warm cache (no new build call); the
    # detail request always goes live.
    assert calls == [False, True]


def test_status_coordination_writes_no_tier_from_the_mcp_process(
    mcp_server: MCPServerUnderTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MCP process is not the ops tier's owner, so it records no route receipt.

    ``status(scope="coordination")`` used to time itself through
    ``observe_route`` and write the receipt into ``ops.db`` with a plain
    ``sqlite3.connect`` from the MCP process (polylogue-k5iaf). The whole tool
    call is already delivered to the daemon through the MCP call-log outbox.

    Anti-vacuity: restore the ``observe_route`` wrapper and ``ops.db`` is
    recorded here.
    """
    from polylogue.maintenance.offline_guard import refuse_writable_tier_opens

    monkeypatch.setattr("polylogue.coordination.build_coordination_envelope", lambda **kwargs: _payload())
    monkeypatch.setattr("polylogue.coordination.envelope.build_coordination_envelope", lambda **kwargs: _payload())
    opened: list[Path] = []
    with refuse_writable_tier_opens(opened.append):
        raw = invoke_surface(mcp_server._tool_manager._tools["status"].fn, scope="coordination", include=("detail",))

    assert json.loads(raw)["scope"] == "coordination"
    assert opened == []
