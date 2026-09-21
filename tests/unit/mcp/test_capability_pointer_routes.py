"""Every route the query capability resource advertises must be callable.

polylogue-2qx.8: ``polylogue://capabilities/query`` told a caller to reach the
positive and negative query corpora through a ``query_completions`` tool that
MCP never registered, and to reach a unit's field names through
``explain(subject="capability", unit=...)`` -- an argument ``explain`` does not
accept. Both read as supported capability and neither could be called.

Advertising an unregistered route is the same defect as a declared-but-unrouted
origin, so the contract here is generic rather than per-pointer: every
``{"tool": ..., "arguments": {...}}` pointer anywhere in the catalog must name
a declared tool, pass only arguments that tool accepts, and return a real
payload when invoked.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from polylogue.archive.query.metadata import query_unit_descriptors
from polylogue.mcp.declarations.registry import declared_tool_names
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async
from tests.infra.storage_records import SessionBuilder

Pointer = dict[str, Any]


def _seeded_archive(tmp_path: Path) -> Path:
    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root):
        builder = SessionBuilder(archive_root / "index.db", "capability-pointer").provider("codex-session")
        builder.add_message(role="user", text="capability pointer seed").save()
    return archive_root


def _pointers(node: object, path: str = "") -> Iterator[tuple[str, Pointer]]:
    """Yield every ``{"tool": ...}`` advertisement anywhere in a payload."""

    if isinstance(node, dict):
        if isinstance(node.get("tool"), str):
            yield path, cast(Pointer, node)
        for key, value in node.items():
            yield from _pointers(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _pointers(value, f"{path}[{index}]")


async def _catalog(mcp_server: MCPServerUnderTest, archive_root: Path) -> dict[str, Any]:
    from polylogue import Polylogue

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        resource = mcp_server._resource_manager._resources["polylogue://capabilities/query"]
        return cast(dict[str, Any], json.loads(await invoke_surface_async(resource.fn)))


def _unique_pointers(catalog: dict[str, Any]) -> list[tuple[str, Pointer]]:
    seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
    unique: list[tuple[str, Pointer]] = []
    for path, pointer in _pointers(catalog):
        key = (pointer["tool"], tuple(sorted(cast(dict[str, Any], pointer.get("arguments", {})).items())))
        if key in seen:
            continue
        seen.add(key)
        unique.append((path, pointer))
    return unique


@pytest.mark.asyncio
async def test_every_advertised_pointer_names_a_declared_tool_and_accepted_arguments(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    """Anti-vacuity: point ``examples_via`` back at ``query_completions``, or give
    ``fields_via`` back its ``unit`` argument on ``subject="capability"``, and
    this names the unreachable route."""
    catalog = await _catalog(mcp_server, _seeded_archive(tmp_path))
    pointers = _unique_pointers(catalog)
    assert pointers, "the query capability catalog advertises no routes at all"

    declared = declared_tool_names()
    tools = mcp_server._tool_manager._tools
    unreachable: list[str] = []
    for path, pointer in pointers:
        tool = pointer["tool"]
        if tool not in declared:
            unreachable.append(f"{path}: tool {tool!r} is not a declared MCP tool")
            continue
        parameters = inspect.signature(tools[tool].fn).parameters
        rejected = sorted(name for name in cast(dict[str, Any], pointer.get("arguments", {})) if name not in parameters)
        if rejected:
            unreachable.append(f"{path}: {tool}() does not accept {rejected}")
    assert unreachable == []

    advertised = {path for path, _ in pointers}
    assert ".corpus.examples_via" in advertised
    assert ".corpus.errors_via" in advertised
    assert any(path.endswith(".fields_via") for path in advertised)


@pytest.mark.asyncio
async def test_every_advertised_pointer_returns_a_payload_when_called(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    """Accepting the arguments is not enough; the route has to answer."""
    archive_root = _seeded_archive(tmp_path)
    catalog = await _catalog(mcp_server, archive_root)

    from polylogue import Polylogue

    failures: list[str] = []
    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        for path, pointer in _unique_pointers(catalog):
            tool = mcp_server._tool_manager._tools[pointer["tool"]]
            body = json.loads(await invoke_surface_async(tool.fn, **cast(dict[str, Any], pointer.get("arguments", {}))))
            if body.get("error") is not None or body.get("is_error"):
                failures.append(f"{path}: {pointer['tool']} returned {body.get('code')}: {body.get('message')}")
    assert failures == []


@pytest.mark.asyncio
async def test_completions_subject_returns_the_shared_python_api_payload(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    """MCP is an adapter over the shared completion route, not a second one.

    Anti-vacuity: delete the ``subject == "completions"`` branch in
    ``server_cutover.explain`` and this fails -- the response becomes the
    generic result/recovery body with no ``candidates``.
    """
    from polylogue import Polylogue

    archive_root = _seeded_archive(tmp_path)
    explain = mcp_server._tool_manager._tools["explain"].fn
    archive = Polylogue(archive_root=archive_root)

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=archive),
    ):
        # Narrow requests so the comparison is byte-for-byte rather than a
        # budget-bounded page; the budget envelope is exercised separately.
        for kind, unit, incomplete in (
            ("error", None, "since"),
            ("example", None, "delegations"),
            ("terminal-field", "message", "tool"),
            ("projection-unit", None, ""),
        ):
            body = json.loads(
                await invoke_surface_async(explain, subject="completions", kind=kind, unit=unit, search=incomplete)
            )
            shared = await archive.query_completions(kind, incomplete=incomplete, unit=unit)
            assert body.get("budget_exceeded") is not True, f"{kind} no longer fits the response budget"
            assert body == {"subject": "completions", **cast(dict[str, Any], shared)}
            assert body["candidates"], f"{kind} completions came back empty"


@pytest.mark.asyncio
async def test_field_pointer_returns_exactly_the_unit_fields_the_catalog_counts(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    """``fields_via`` is the detail route for the field names the index omits.

    Naming a callable tool is not enough -- the route the catalog points at
    has to return the thing the catalog says it omitted. Anti-vacuity: point
    ``fields_via`` back at ``subject="capability"``; ``explain`` still answers,
    but with the capability declaration page instead of this unit's fields.
    """
    from polylogue import Polylogue

    archive_root = _seeded_archive(tmp_path)
    catalog = await _catalog(mcp_server, archive_root)
    by_unit = {str(unit["unit"]): unit for unit in cast(list[dict[str, Any]], catalog["units"])}
    descriptors = {str(descriptor.unit): descriptor for descriptor in query_unit_descriptors(terminal_supported=True)}

    checked = 0
    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        for unit_name, unit in by_unit.items():
            pointer = cast(Pointer | None, unit.get("fields_via"))
            if pointer is None:  # this response carried the names inline
                assert set(cast(list[str], unit["fields"])) == {field.name for field in descriptors[unit_name].fields}
                continue
            tool = mcp_server._tool_manager._tools[pointer["tool"]]
            body = json.loads(await invoke_surface_async(tool.fn, **cast(dict[str, Any], pointer.get("arguments", {}))))
            page = cast(dict[str, Any], body["page"]) if body.get("budget_exceeded") else body
            assert "candidates" in page, (
                f"{unit_name} fields_via answered with {sorted(page)} -- the advertised detail "
                "route does not return this unit's field names"
            )
            candidates = cast(list[dict[str, Any]], page["candidates"])
            offered = {str(item["value"]) for item in candidates}
            expected = {field.name for field in descriptors[unit_name].fields}
            # A budget-bounded first page is a prefix, never a different set.
            assert offered <= expected, f"{unit_name} fields_via offered names that are not its fields"
            if not body.get("budget_exceeded"):
                assert offered == expected
                assert unit["field_count"] == len(expected)
            checked += 1

    assert checked or by_unit, "the catalog published no units at all"


@pytest.mark.asyncio
async def test_completions_refuse_an_undeclared_kind_without_a_traceback(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    """An unknown vocabulary is a typed refusal, not a 500-shaped surprise."""
    from polylogue import Polylogue

    archive_root = _seeded_archive(tmp_path)
    explain = mcp_server._tool_manager._tools["explain"].fn
    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        body = json.loads(await invoke_surface_async(explain, subject="completions", kind="not-a-kind"))
        missing_unit = json.loads(await invoke_surface_async(explain, subject="completions", kind="terminal-field"))

    assert body["code"] == "invalid_argument"
    assert "not-a-kind" in body["message"]
    assert missing_unit["code"] == "invalid_argument"
    assert "unit" in missing_unit["message"]
