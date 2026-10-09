"""Session-list routes obey the declared envelope and preserve owner verdicts."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.archive.query.search_contract import ArchiveSearchResult, SearchExecution
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.mcp.archive_support import archive_session_list_payload
from polylogue.mcp.declarations.registry import declaration_for_tool
from polylogue.mcp.server_cutover import _query_sessions
from polylogue.mcp.server_resources import register_resources
from polylogue.mcp.server_support import ServerCallbacks, _json_payload
from polylogue.operations.session_contracts import Coverage, SessionPage
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["advanced", "typed", "sessions-resource", "origin-resource"])
@pytest.mark.parametrize("count", [0, 1])
async def test_session_list_routes_obey_query_envelope(monkeypatch: pytest.MonkeyPatch, route: str, count: int) -> None:
    """Removing outcome/unit from either adapter breaks this shared route law."""
    from mcp.server.mcpserver import MCPServer

    rows = [
        ArchiveSessionSummary(
            session_id="codex-session:neutral",
            native_id="neutral",
            origin="codex-session",
            title="Neutral",
            created_at=None,
            updated_at=None,
            message_count=0,
            word_count=0,
            tags=(),
        )
    ][:count]
    archive = MagicMock()
    archive.list_summaries.return_value = rows
    archive.count_sessions.return_value = count
    transaction = SimpleNamespace(run=AsyncMock(side_effect=lambda read: read(archive)))
    monkeypatch.setattr("polylogue.archive.query.transaction.QueryTransaction", lambda *a, **k: transaction)
    owner = SessionPage[Any](
        items=[],
        total=count,
        limit=10,
        offset=0,
        coverage=Coverage(authority="indexed-archive", complete=True),
        outcome="ok" if count else "empty",
    )
    # The typed route owns its item models. This probe binds its verdict and
    # metadata without asking a storage reader to execute again.
    owner.items = list(archive_session_list_payload(archive, SessionQuerySpec(limit=10)).items)
    monkeypatch.setattr("polylogue.operations.session_reads.execute_session_operation", AsyncMock(return_value=owner))
    hooks = MagicMock(spec=list(ServerCallbacks.__annotations__))
    hooks.clamp_limit.side_effect = lambda value: value if isinstance(value, int) and value > 0 else 10
    hooks.get_config.return_value = SimpleNamespace(archive_root=Path("/neutral"), db_path=Path("/neutral/index.db"))
    hooks.json_payload.side_effect = _json_payload
    hooks.response_context.side_effect = lambda *args: nullcontext()
    if route.endswith("resource"):
        server = MCPServer("neutral")
        register_resources(server, hooks)
        if route == "sessions-resource":
            raw = await server._resource_manager._resources["polylogue://sessions"].read()
        else:
            raw = await server._resource_manager._templates["polylogue://origin/{name}/recent"].fn(name="codex-session")
    else:
        raw = await _query_sessions(
            hooks,
            expression=None,
            limit=10,
            offset=0,
            origin=None,
            tag=None,
            repo=None,
            since=None,
            until=None,
            sort="random" if route == "advanced" else None,
            min_messages=None,
            max_messages=None,
            min_words=None,
        )
    body = json.loads(raw)
    contract = declaration_for_tool("query").contract_kind
    assert isinstance(contract, tuple)
    kind, required = contract
    assert kind == "envelope"
    assert required <= body.keys()
    assert body["unit"] == "sessions"
    assert body["total"] == count
    assert len(body["items"]) == count
    assert body["outcome"]["state"] == ("ok" if count else "empty")


@pytest.mark.asyncio
async def test_typed_session_list_preserves_zero_row_named_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typed owner's degraded empty page cannot become authoritative empty."""
    owner = SessionPage[Any](
        items=[],
        total=0,
        limit=10,
        offset=0,
        outcome="degraded",
        coverage=Coverage(authority="indexed-archive", complete=False, gaps=["missing_digest", "missing_profile"]),
    )
    monkeypatch.setattr("polylogue.operations.session_reads.execute_session_operation", AsyncMock(return_value=owner))
    hooks = MagicMock(spec=list(ServerCallbacks.__annotations__))
    hooks.clamp_limit.side_effect = lambda value: value
    hooks.json_payload.side_effect = _json_payload
    raw = await _query_sessions(
        hooks,
        expression=None,
        limit=10,
        offset=0,
        origin=None,
        tag=None,
        repo=None,
        since=None,
        until=None,
        sort=None,
        min_messages=None,
        max_messages=None,
        min_words=None,
    )
    assert json.loads(raw)["outcome"] == {
        "state": "degraded",
        "reason": "missing_digest",
        "detail": {"gaps": ["missing_digest", "missing_profile"]},
    }


def test_retained_ranked_list_preserves_zero_row_lane_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every live payload constructor supplies the execution's terminal verdict."""
    result = ArchiveSearchResult(
        hits=[],
        retrieval_lane="semantic",
        execution=SearchExecution(requested_lanes=("vector",), executed_lanes=(), unavailable_lanes=("vector",)),
    )
    monkeypatch.setattr("polylogue.archive.query.archive_execution.archive_search_hits", lambda *a, **k: result)
    body = archive_session_list_payload(
        MagicMock(archive_root=Path("/neutral")), SessionQuerySpec(similar_session_id="codex-session:neutral", limit=10)
    ).model_dump(mode="json")
    assert body["total"] is None
    assert body["items"] == []
    assert body["outcome"] == {
        "state": "degraded",
        "reason": "lane_unavailable:vector",
        "detail": {"gaps": ["lane_unavailable:vector"]},
    }
