"""Native cross-surface query laws (archive).

The archive layer stack (SQLiteBackend → SessionRepository → facade) is
gone; the archive is the only store. These tests pin the
universal query invariants of the archive read surfaces and assert that the two
archive read surfaces — the async ``Polylogue`` facade and the MCP
``list_sessions`` / ``search`` tools — agree on the same archive.

Laws asserted (per surface and across surfaces):
  * count == len(list)
  * origin filters partition the archive (disjoint, subset of all)
  * filtered results are a subset of the full listing
  * search hits are a subset of the listing and isolate the unique token
  * limit returns at most N and a subset of all ids
  * the facade id set equals the MCP id set for every projection

The CLI native ``list``/``count`` surface is intentionally excluded — see the
prod-gap note in the archive-gap report: the archive CLI list path emits an
empty envelope even when the archive has matching sessions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from tests.infra.storage_records import SessionBuilder, db_setup

# ---------------------------------------------------------------------------
# Native surface adapters
# ---------------------------------------------------------------------------


class _FacadeSurface:
    name = "facade"

    def __init__(self, *, archive_root: Path, db_path: Path) -> None:
        self._archive = Polylogue(archive_root=archive_root, db_path=db_path)

    async def list_ids(self, *, origin: str | None = None, limit: int | None = None) -> tuple[str, ...]:
        convs = await self._archive.list_sessions(origin=origin, limit=limit)
        return tuple(sorted(str(c.id) for c in convs))

    async def count(self, *, origin: str | None = None) -> int:
        return len(await self._archive.list_sessions(origin=origin, limit=None))

    async def search_ids(self, query: str, *, limit: int = 50) -> tuple[str, ...]:
        result = await self._archive.search(query, limit=limit)
        return tuple(sorted(str(hit.session_id) for hit in result.hits))

    async def close(self) -> None:
        await self._archive.close()


class _MCPSurface:
    name = "mcp"

    def __init__(self, *, db_path: Path) -> None:
        from polylogue.mcp.server import build_server
        from polylogue.mcp.server_support import _set_runtime_services
        from polylogue.services import build_runtime_services

        self._services = build_runtime_services(db_path=db_path)
        _set_runtime_services(self._services)
        self._server = build_server(role="admin")

    def _tool(self, name: str):  # type: ignore[no-untyped-def]
        return self._server._tool_manager._tools[name].fn

    async def list_ids(self, *, origin: str | None = None, limit: int | None = None) -> tuple[str, ...]:
        payload = await self._tool("list_sessions")(
            limit=1000 if limit is None else limit,
            origin=origin,
        )
        items = json.loads(payload).get("items", [])
        return tuple(sorted(str(item["id"]) for item in items))

    async def count(self, *, origin: str | None = None) -> int:
        return len(await self.list_ids(origin=origin, limit=1000))

    async def search_ids(self, query: str, *, limit: int = 50) -> tuple[str, ...]:
        payload = await self._tool("search")(query=query, limit=limit)
        parsed = json.loads(payload)
        hits = parsed.get("hits", parsed.get("items", []))
        ids: list[str] = []
        for hit in hits:
            if "session_id" in hit:
                ids.append(str(hit["session_id"]))
            elif "id" in hit:
                ids.append(str(hit["id"]))
            elif isinstance(hit.get("session"), dict) and "id" in hit["session"]:
                ids.append(str(hit["session"]["id"]))
        return tuple(sorted(ids))

    async def close(self) -> None:
        from polylogue.mcp.server_support import _set_runtime_services

        await self._services.close()
        _set_runtime_services(None)


def _seed(workspace_env: dict[str, Path], *, chatgpt: int = 2, claude: int = 3) -> dict[str, list[str]]:
    """Seed the archive; return archive session ids grouped by provider."""
    db_path = db_setup(workspace_env)
    ids: dict[str, list[str]] = {"chatgpt": [], "claude-ai": []}
    for i in range(chatgpt):
        b = (
            SessionBuilder(db_path, f"chatgpt-conv-{i}")
            .provider("chatgpt")
            .title(f"ChatGPT conv {i}")
            .add_message(role="user", text=f"chatgpt question {i}")
            .add_message(role="assistant", text=f"chatgpt answer {i}")
        )
        b.save()
        ids["chatgpt"].append(b.native_session_id())
    for i in range(claude):
        b = (
            SessionBuilder(db_path, f"claude-conv-{i}")
            .provider("claude-ai")
            .title(f"Claude conv {i}")
            .add_message(role="user", text=f"claude question {i}")
            .add_message(role="assistant", text=f"claude answer {i}")
        )
        b.save()
        ids["claude-ai"].append(b.native_session_id())
    return ids


# ---------------------------------------------------------------------------
# Count laws
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.asyncio
async def test_count_equals_list_length_across_surfaces(workspace_env: dict[str, Path]) -> None:
    _seed(workspace_env, chatgpt=2, claude=3)
    db_path = db_setup(workspace_env)
    facade = _FacadeSurface(archive_root=workspace_env["archive_root"], db_path=db_path)
    mcp = _MCPSurface(db_path=db_path)
    try:
        for surface in (facade, mcp):
            ids = await surface.list_ids(limit=None)
            assert await surface.count() == len(ids) == 5, surface.name
    finally:
        await facade.close()
        await mcp.close()


@pytest.mark.contract
@pytest.mark.asyncio
async def test_count_empty_archive(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    facade = _FacadeSurface(archive_root=workspace_env["archive_root"], db_path=db_path)
    mcp = _MCPSurface(db_path=db_path)
    try:
        for surface in (facade, mcp):
            assert await surface.count() == 0, surface.name
            assert await surface.list_ids(limit=None) == (), surface.name
    finally:
        await facade.close()
        await mcp.close()


# ---------------------------------------------------------------------------
# Id-set agreement across archive surfaces
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.asyncio
async def test_full_listing_id_sets_agree(workspace_env: dict[str, Path]) -> None:
    _seed(workspace_env, chatgpt=2, claude=2)
    db_path = db_setup(workspace_env)
    facade = _FacadeSurface(archive_root=workspace_env["archive_root"], db_path=db_path)
    mcp = _MCPSurface(db_path=db_path)
    try:
        assert await facade.list_ids(limit=None) == await mcp.list_ids(limit=None)
    finally:
        await facade.close()
        await mcp.close()


@pytest.mark.contract
@pytest.mark.asyncio
async def test_origin_filtered_id_sets_agree(workspace_env: dict[str, Path]) -> None:
    expected = _seed(workspace_env, chatgpt=3, claude=2)
    db_path = db_setup(workspace_env)
    facade = _FacadeSurface(archive_root=workspace_env["archive_root"], db_path=db_path)
    mcp = _MCPSurface(db_path=db_path)
    try:
        for provider, origin, count in (("chatgpt", "chatgpt-export", 3), ("claude-ai", "claude-ai-export", 2)):
            facade_ids = await facade.list_ids(origin=origin, limit=None)
            mcp_ids = await mcp.list_ids(origin=origin, limit=None)
            assert facade_ids == mcp_ids
            assert len(facade_ids) == count
            assert set(facade_ids) == set(expected[provider])
    finally:
        await facade.close()
        await mcp.close()


# ---------------------------------------------------------------------------
# Filter partition laws
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.asyncio
async def test_origin_filters_partition_the_archive(workspace_env: dict[str, Path]) -> None:
    _seed(workspace_env, chatgpt=2, claude=4)
    db_path = db_setup(workspace_env)
    facade = _FacadeSurface(archive_root=workspace_env["archive_root"], db_path=db_path)
    mcp = _MCPSurface(db_path=db_path)
    try:
        for surface in (facade, mcp):
            all_ids = set(await surface.list_ids(limit=None))
            chatgpt_ids = set(await surface.list_ids(origin="chatgpt-export", limit=None))
            claude_ids = set(await surface.list_ids(origin="claude-ai-export", limit=None))
            # subset
            assert chatgpt_ids <= all_ids, surface.name
            assert claude_ids <= all_ids, surface.name
            # disjoint
            assert chatgpt_ids.isdisjoint(claude_ids), surface.name
            # exhaustive
            assert chatgpt_ids | claude_ids == all_ids, surface.name
            assert len(chatgpt_ids) == 2 and len(claude_ids) == 4, surface.name
    finally:
        await facade.close()
        await mcp.close()


# ---------------------------------------------------------------------------
# Search laws
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.asyncio
async def test_search_isolates_unique_token_and_is_subset(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    unique = (
        SessionBuilder(db_path, "unique-conv")
        .provider("chatgpt")
        .title("Unique Result")
        .add_message(role="user", text="xyzzy special token")
        .add_message(role="assistant", text="the xyzzy value is important")
    )
    unique.save()
    other = (
        SessionBuilder(db_path, "other-conv")
        .provider("claude-ai")
        .title("Other Session")
        .add_message(role="user", text="unrelated message about weather")
        .add_message(role="assistant", text="the weather is fine")
    )
    other.save()
    unique_id = unique.native_session_id()
    other_id = other.native_session_id()

    facade = _FacadeSurface(archive_root=workspace_env["archive_root"], db_path=db_path)
    mcp = _MCPSurface(db_path=db_path)
    try:
        for surface in (facade, mcp):
            all_ids = set(await surface.list_ids(limit=None))
            hits = set(await surface.search_ids("xyzzy"))
            assert hits <= all_ids, surface.name
            assert unique_id in hits, surface.name
            assert other_id not in hits, surface.name
    finally:
        await facade.close()
        await mcp.close()


@pytest.mark.contract
@pytest.mark.asyncio
async def test_search_unknown_token_returns_empty(workspace_env: dict[str, Path]) -> None:
    _seed(workspace_env, chatgpt=2, claude=2)
    db_path = db_setup(workspace_env)
    facade = _FacadeSurface(archive_root=workspace_env["archive_root"], db_path=db_path)
    mcp = _MCPSurface(db_path=db_path)
    try:
        for surface in (facade, mcp):
            assert await surface.search_ids("zzznomatch_xyz_unique") == (), surface.name
    finally:
        await facade.close()
        await mcp.close()


# ---------------------------------------------------------------------------
# Limit laws
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.asyncio
async def test_limit_returns_subset_of_all(workspace_env: dict[str, Path]) -> None:
    _seed(workspace_env, chatgpt=4, claude=4)
    db_path = db_setup(workspace_env)
    facade = _FacadeSurface(archive_root=workspace_env["archive_root"], db_path=db_path)
    mcp = _MCPSurface(db_path=db_path)
    try:
        for surface in (facade, mcp):
            all_ids = set(await surface.list_ids(limit=None))
            assert len(all_ids) == 8, surface.name
            limited = await surface.list_ids(limit=3)
            assert len(limited) == 3, surface.name
            assert set(limited) <= all_ids, surface.name
    finally:
        await facade.close()
        await mcp.close()
