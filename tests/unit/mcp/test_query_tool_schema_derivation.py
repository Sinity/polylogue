"""Contract: query tool inputSchema is derived from the request dataclass.

Audit #1282 (R3) refactored the query tools so their parameters are
auto-generated from :class:`MCPSessionQueryRequest`. This test guards
against silent drift between the dataclass and the published MCP tool schema:
if a field is added to or removed from the dataclass and a tool definition is
not regenerated alongside, the published wire schema will fail to match.
"""

from __future__ import annotations

import asyncio
from dataclasses import fields
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from polylogue.archive.query.fields import mcp_query_field_names
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.mcp.archive_support import archive_query_filters, archive_search_payload, archive_session_list_payload
from polylogue.mcp.query_contracts import MCPSessionQueryRequest
from polylogue.mcp.server_tools import register_query_tools
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSearchHit, ArchiveSessionSummary

SchemaMap = dict[str, dict[str, Any]]


def _clamp_limit(value: int | object) -> int:
    return value if isinstance(value, int) and value > 0 else 10


def _registered_schemas() -> SchemaMap:
    hooks = MagicMock()
    hooks.clamp_limit = _clamp_limit

    server = FastMCP("schema-derivation-test")
    register_query_tools(server, hooks)
    tools = asyncio.run(server.list_tools())
    return {tool.name: tool.inputSchema for tool in tools if tool.name in {"search", "list_sessions", "facets"}}


def _dataclass_field_names() -> set[str]:
    return {field.name for field in fields(MCPSessionQueryRequest)}


@pytest.fixture(scope="module")
def schemas() -> SchemaMap:
    return _registered_schemas()


def test_search_schema_matches_dataclass_fields(schemas: SchemaMap) -> None:
    schema = schemas["search"]
    properties = set(schema.get("properties", {}).keys())
    assert properties == _dataclass_field_names(), (
        "search inputSchema drifted from MCPSessionQueryRequest. "
        f"missing={sorted(_dataclass_field_names() - properties)}, "
        f"extra={sorted(properties - _dataclass_field_names())}"
    )


def test_list_sessions_schema_matches_supported_request_fields(
    schemas: SchemaMap,
) -> None:
    schema = schemas["list_sessions"]
    properties = set(schema.get("properties", {}).keys())
    expected = _dataclass_field_names() - {"query", "include_affordances"}
    assert properties == expected, (
        "list_sessions inputSchema drifted from MCPSessionQueryRequest. "
        f"missing={sorted(expected - properties)}, extra={sorted(properties - expected)}"
    )


def test_facets_schema_matches_dataclass_fields(schemas: SchemaMap) -> None:
    schema = schemas["facets"]
    properties = set(schema.get("properties", {}).keys())
    assert properties == _dataclass_field_names(), (
        "facets inputSchema drifted from MCPSessionQueryRequest. "
        f"missing={sorted(_dataclass_field_names() - properties)}, "
        f"extra={sorted(properties - _dataclass_field_names())}"
    )


def test_mcp_query_request_matches_canonical_query_field_registry() -> None:
    """MCP request fields stay aligned with query fields marked MCP-capable."""

    assert mcp_query_field_names() <= _dataclass_field_names()


def test_search_marks_query_required(schemas: SchemaMap) -> None:
    required = schemas["search"].get("required") or []
    assert "query" in required, "search must keep ``query`` required for backwards compatibility"


def test_list_sessions_has_no_required_parameters(
    schemas: SchemaMap,
) -> None:
    required = schemas["list_sessions"].get("required") or []
    assert required == [], "list_sessions historically accepts zero required parameters"


def test_facets_query_is_optional(schemas: SchemaMap) -> None:
    required = schemas["facets"].get("required") or []
    assert "query" in schemas["facets"].get("properties", {})
    assert required == [], "facets supports query-scoped counts but must remain callable with no filters"


def test_limit_and_offset_preserve_pydantic_constraints(schemas: SchemaMap) -> None:
    """The derived schema must preserve the ``ge=1``/``ge=0`` annotations from the TypeAliases."""
    for name in ("search", "list_sessions", "facets"):
        properties = schemas[name].get("properties", {})
        limit = properties["limit"]
        offset = properties["offset"]
        assert isinstance(limit, dict) and limit.get("minimum") == 1, f"{name}: limit lost ge=1 constraint"
        assert isinstance(offset, dict) and offset.get("minimum") == 0, f"{name}: offset lost ge=0 constraint"


def test_archive_query_filters_forward_max_words() -> None:
    spec = MCPSessionQueryRequest(max_words=12).build_spec(_clamp_limit)

    assert archive_query_filters(spec)["max_words"] == 12


def test_archive_list_sessions_routes_near_session_to_query_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_archive_search_hits(
        plan: SessionQueryPlan,
        **kwargs: object,
    ) -> tuple[list[tuple[ArchiveSessionSearchHit, ArchiveSessionSummary]], str]:
        observed["similar_session_id"] = plan.similar_session_id
        observed["archive_root"] = kwargs["archive_root"]
        return [], "semantic"

    monkeypatch.setattr("polylogue.archive.query.archive_execution.archive_search_hits", fake_archive_search_hits)
    archive = MagicMock()
    archive.archive_root = Path("/archive")
    spec = MCPSessionQueryRequest(similar_session_id="seed-session", limit=5).build_spec(_clamp_limit)

    payload = archive_session_list_payload(archive, spec, archive_root=Path("/archive"))

    assert observed == {"similar_session_id": "seed-session", "archive_root": Path("/archive")}
    assert payload.items == ()
    assert payload.total is None


def test_archive_list_sessions_stops_after_requested_distinct_page() -> None:
    summaries = {
        "codex-session:first": ArchiveSessionSummary(
            session_id="codex-session:first",
            native_id="first",
            origin="codex-session",
            title="First",
            created_at=None,
            updated_at=None,
            message_count=1,
            word_count=10,
            tags=(),
        ),
        "codex-session:second": ArchiveSessionSummary(
            session_id="codex-session:second",
            native_id="second",
            origin="codex-session",
            title="Second",
            created_at=None,
            updated_at=None,
            message_count=1,
            word_count=10,
            tags=(),
        ),
    }

    def hit(session_id: str) -> ArchiveSessionSearchHit:
        return ArchiveSessionSearchHit(
            rank=1,
            session_id=session_id,
            block_id=f"{session_id}:block",
            message_id=f"{session_id}:message",
            origin="codex-session",
            title=summaries[session_id].title,
            snippet="needle",
        )

    archive = MagicMock()
    archive.count_search_sessions.return_value = 3
    archive.search_summaries.side_effect = [[hit("codex-session:first"), *[hit("codex-session:second")] * 249]]
    archive.read_summary.side_effect = summaries.__getitem__
    spec = MCPSessionQueryRequest(contains="needle", limit=2).build_spec(_clamp_limit)

    payload = archive_session_list_payload(archive, spec)

    assert [item.id for item in payload.items] == ["codex-session:first", "codex-session:second"]
    assert payload.total == 3
    assert payload.next_offset == 2
    assert [item.match_count for item in payload.items] == [1, 249]
    assert all(item.match_count_is_exact is False for item in payload.items)
    assert archive.search_summaries.call_count == 1


def test_archive_search_routes_near_session_to_query_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = ArchiveSessionSummary(
        session_id="codex-session:near-result",
        native_id="near-result",
        origin="codex-session",
        title="Nearby",
        created_at=None,
        updated_at=None,
        message_count=1,
        word_count=10,
        tags=(),
    )
    hit = ArchiveSessionSearchHit(
        rank=1,
        session_id=summary.session_id,
        block_id="block-1",
        message_id="message-1",
        origin=summary.origin,
        title=summary.title,
        snippet="nearby match",
    )

    def fake_archive_search_hits(
        plan: SessionQueryPlan,
        **kwargs: object,
    ) -> tuple[list[tuple[ArchiveSessionSearchHit, ArchiveSessionSummary]], str]:
        assert plan.similar_session_id == "seed-session"
        assert kwargs["archive_root"] == Path("/archive")
        return [(hit, summary)], "semantic"

    monkeypatch.setattr("polylogue.archive.query.archive_execution.archive_search_hits", fake_archive_search_hits)
    archive = MagicMock()
    archive.archive_root = Path("/archive")
    archive.read_summary.return_value = summary
    spec = MCPSessionQueryRequest(similar_session_id="seed-session", limit=5).build_spec(_clamp_limit)

    payload = archive_search_payload(
        archive,
        spec,
        query="",
        limit=5,
        offset=0,
        retrieval_lane="semantic",
        sort=None,
        archive_root=Path("/archive"),
    )

    assert payload.retrieval_lane == "semantic"
    assert [hit.session.id for hit in payload.hits] == ["codex-session:near-result"]
