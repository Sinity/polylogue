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
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async
from tests.infra.storage_records import SessionBuilder

Pointer = dict[str, Any]


def _seeded_archive(tmp_path: Path) -> Path:
    archive_root = tmp_path / "archive"
    from polylogue.operations.fts_derivation import stamp_fts_readiness_binding

    def seed() -> None:
        with ArchiveStore(archive_root) as archive:
            builder = SessionBuilder(archive_root / "index.db", "capability-pointer").provider("codex-session")
            builder.add_message(role="user", text="capability pointer seed").save()
            assert stamp_fts_readiness_binding(archive._conn)
            archive._conn.commit()

    run_off_event_loop(seed)
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
            expected = cast(dict[str, Any], shared)
            candidates = expected["candidates"]
            assert body == {
                "subject": "completions",
                **expected,
                "candidates": candidates[:25],
                "total": len(candidates),
                "limit": 25,
                "offset": 0,
                "next_offset": 25 if len(candidates) > 25 else None,
            }
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
            assert page["total"] == unit["field_count"] == len(expected)
            if not body.get("budget_exceeded") and page["next_offset"] is None:
                assert offered == expected
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


@pytest.mark.asyncio
async def test_completions_page_against_shared_owner(mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
    """Without candidate paging, the registered MCP route repeats the whole owner payload."""
    from polylogue import Polylogue

    root = _seeded_archive(tmp_path)
    archive = Polylogue(archive_root=root)
    explain = mcp_server._tool_manager._tools["explain"].fn
    shared = await archive.query_completions("field")
    candidates = cast(list[Any], shared["candidates"])
    assert len(candidates) > 4
    with patch("polylogue.mcp.server._get_polylogue", return_value=archive):
        first = json.loads(await invoke_surface_async(explain, subject="completions", kind="field", limit=2))
        second = json.loads(
            await invoke_surface_async(
                explain, subject="completions", kind="field", limit=2, offset=first["next_offset"]
            )
        )
    assert first["candidates"] + second["candidates"] == candidates[:4]
    assert second["offset"] == 2
    assert second["total"] == len(candidates)


@pytest.mark.asyncio
async def test_capability_discovery_includes_messages(mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
    """The older list-only vocabulary omitted the runtime messages view."""
    from polylogue import Polylogue
    from polylogue.archive.session_projections import mcp_read_view_names

    root = _seeded_archive(tmp_path)
    with patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=root)):
        body = json.loads(
            await invoke_surface_async(mcp_server._tool_manager._tools["explain"].fn, subject="capability", limit=1)
        )
    page = body.get("page") or body
    assert set(page["read_views"]) == set(mcp_read_view_names())
    assert "messages" in page["read_views"]


@pytest.mark.asyncio
@pytest.mark.parametrize("archive_state", ["missing", "empty", "populated"])
async def test_capability_pages_keep_declarations_without_full_statistics(
    mcp_server: MCPServerUnderTest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive_state: str
) -> None:
    """Missing evidence stays unknown; measured counts never invoke stats or hydration."""
    from polylogue import Polylogue

    root = tmp_path / "archive"
    if archive_state == "populated":
        root = _seeded_archive(tmp_path)
    elif archive_state == "empty":
        from polylogue.operations.fts_derivation import stamp_fts_readiness_binding

        def stamp_empty() -> None:
            with ArchiveStore(root) as archive:
                assert stamp_fts_readiness_binding(archive._conn)
                archive._conn.commit()

        run_off_event_loop(stamp_empty)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("capability pages must not aggregate statistics or hydrate sessions")

    monkeypatch.setattr(ArchiveStore, "stats", forbidden)
    monkeypatch.setattr(ArchiveStore, "read_session", forbidden)
    poly = Polylogue(archive_root=root)
    explain = mcp_server._tool_manager._tools["explain"].fn
    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        first = json.loads(await invoke_surface_async(explain, subject="capability", limit=1))
        second = json.loads(
            await invoke_surface_async(explain, subject="capability", limit=1, offset=first["next_offset"])
        )
    assert first["items"] and second["items"]
    assert first["items"][0]["declaration_id"] != second["items"][0]["declaration_id"]
    assert first["read_view_profile_ids"] == second["read_view_profile_ids"]
    assert "chronicle" in first["read_view_profile_ids"]
    for page in (first, second):
        if archive_state == "missing":
            assert page["outcome"]["state"] == "degraded"
            assert page["outcome"]["reason"] == "archive_counts_unavailable"
            assert page["snapshot"]["freshness"] == "unknown"
            assert all(item["observed_count"] is None and item["status"] == "unknown" for item in page["items"])
        else:
            assert page["outcome"]["state"] == "ok"
            assert page["snapshot"]["freshness"] == "request-current"
    if archive_state != "missing":
        assert await poly.storage_counts() == {
            "total_sessions": 1 if archive_state == "populated" else 0,
            "total_messages": 1 if archive_state == "populated" else 0,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("evidence_state", ["healthy", "debt", "unbound", "missing", "skew"])
async def test_capability_evidence_uses_public_origins_specific_counts_and_query_readiness(
    mcp_server: MCPServerUnderTest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence_state: str
) -> None:
    import sqlite3

    from polylogue.sources.origin_specs import public_origin_tokens
    from polylogue.storage.fts.derivation import FtsDerivationAdapter
    from polylogue.storage.sqlite.archive_tiers.ops_write import add_convergence_debt
    from tests.infra.mcp import installed_runtime_services

    root = tmp_path / "archive" if evidence_state == "missing" else _seeded_archive(tmp_path)
    if evidence_state == "unbound":
        with sqlite3.connect(root / "index.db") as conn:
            conn.execute("DELETE FROM messages_fts_readiness_binding")
    elif evidence_state == "skew":
        with sqlite3.connect(root / "index.db") as conn:
            conn.execute("UPDATE schema_identity SET identity='synthetic-skew' WHERE tier='index'")
    elif evidence_state == "debt":
        with sqlite3.connect(root / "ops.db") as conn:
            add_convergence_debt(
                conn, stage="materialize", target_type="session", target_id="fixture", created_at_ms=1000
            )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("capability evidence must not scan statistics, blocks or sessions")

    monkeypatch.setattr(ArchiveStore, "stats", forbidden)
    monkeypatch.setattr(ArchiveStore, "read_session", forbidden)
    monkeypatch.setattr(FtsDerivationAdapter, "inspect_partition", forbidden)
    explain = mcp_server._tool_manager._tools["explain"].fn
    with installed_runtime_services(root):
        pages = [
            json.loads(await invoke_surface_async(explain, subject="capability", search=search))
            for search in ("query.unit.message", "query.unit.block", "tags")
        ]
    for page in pages:
        assert page["items"]
        assert {item["origin"] for item in page["evidence"]["origins"]} == set(public_origin_tokens())
        assert all("origins" not in item["evidence"] for item in page["items"])
        assert all(item["evidence"]["freshness"] == page["snapshot"]["freshness"] for item in page["items"])
    message = next(item for item in pages[0]["items"] if item["kind"] == "unit" and item["name"] == "message")
    block = next(item for item in pages[1]["items"] if item["kind"] == "unit" and item["name"] == "block")
    tags = next(item for item in pages[2]["items"] if item["kind"] == "field" and item["name"] == "tags")
    assert block["observed_count"] is None and tags["observed_count"] is None
    if evidence_state in {"missing", "skew"}:
        assert message["observed_count"] is None and message["status"] == "unknown"
        assert all(page["outcome"]["reason"] == "archive_counts_unavailable" for page in pages)
        assert all(page["snapshot"]["freshness"] == "unknown" for page in pages)
    else:
        assert message["observed_count"] == 1
        expected_state = {"healthy": "ready", "debt": "stale", "unbound": "unknown"}[evidence_state]
        assert all(page["evidence"]["readiness"]["state"] == expected_state for page in pages)
        assert all(item["evidence"]["readiness_state"] == expected_state for page in pages for item in page["items"])
        if evidence_state == "healthy":
            assert message["status"] == "supported_and_observed"
            assert block["status"] == tags["status"] == "unknown"
            assert all(page["outcome"]["state"] == "ok" for page in pages)
        elif evidence_state == "debt":
            assert all(page["outcome"]["state"] == "degraded" for page in pages)
            assert all(page["snapshot"]["freshness"] == "stale_or_degraded" for page in pages)
            assert all(item["status"] == "stale_or_degraded" for page in pages for item in page["items"])
        else:
            assert message["status"] == "unknown"
            assert all(page["outcome"]["reason"] == "query_binding_unavailable" for page in pages)
            assert all(page["snapshot"]["freshness"] == "unknown" for page in pages)
