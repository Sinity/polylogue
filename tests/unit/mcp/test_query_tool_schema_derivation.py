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
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from polylogue.archive.query.fields import mcp_query_field_names
from polylogue.mcp.query_contracts import MCPSessionQueryRequest
from polylogue.mcp.server_tools import register_query_tools

SchemaMap = dict[str, dict[str, Any]]


def _registered_schemas() -> SchemaMap:
    hooks = MagicMock()
    hooks.clamp_limit = lambda value: int(value) if value else 10

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


def test_list_sessions_schema_matches_dataclass_fields_excluding_query(
    schemas: SchemaMap,
) -> None:
    schema = schemas["list_sessions"]
    properties = set(schema.get("properties", {}).keys())
    expected = _dataclass_field_names() - {"query"}
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
