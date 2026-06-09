"""Retrieval readiness laws over the archive.

These tests prove that origin filters, FTS search, counts, and aggregate
archive facts are consistent across the archive read surfaces (the ``ArchiveStore``
substrate, the async ``Polylogue`` facade, and the MCP ``list_sessions`` /
``search`` tools), and that the archive FTS index covers exactly the indexable
content blocks.

The FTS readiness contract (``check_fts_readiness``) is a pure guard and is
asserted directly.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.errors import DatabaseError
from polylogue.storage.fts.fts_lifecycle import check_fts_readiness
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder, _record_to_parsed_session, db_setup


@dataclass(frozen=True)
class _Scenario:
    name: str
    provider: str
    title: str
    messages: tuple[tuple[str, str], ...]


def _scenarios() -> tuple[_Scenario, ...]:
    return (
        _Scenario(
            name="chatgpt-retrieval-1",
            provider="chatgpt",
            title="ChatGPT session about testing",
            messages=(
                ("user", "How do I write property tests?"),
                ("assistant", "Property tests verify invariants using random inputs"),
            ),
        ),
        _Scenario(
            name="claude-retrieval-1",
            provider="claude-code",
            title="Claude Code session on refactoring",
            messages=(
                ("user", "Refactor the storage module"),
                ("assistant", "I will restructure the query layer"),
                ("user", "Also fix the property tests"),
            ),
        ),
        _Scenario(
            name="claude-retrieval-2",
            provider="claude-code",
            title="Claude debugging memory leak",
            messages=(
                ("user", "Memory keeps growing during ingest"),
                ("assistant", "The blob store path has a leak"),
            ),
        ),
        _Scenario(
            name="codex-retrieval-1",
            provider="codex",
            title="Codex adding authentication",
            messages=(("user", "Add OAuth2 authentication"),),
        ),
    )


def _seed(workspace_env: dict[str, Path]) -> tuple[Path, dict[str, str]]:
    db_path = db_setup(workspace_env)
    ids: dict[str, str] = {}
    with ArchiveStore(workspace_env["archive_root"]) as archive:
        for scenario in _scenarios():
            builder = SessionBuilder(db_path, scenario.name).provider(scenario.provider).title(scenario.title)
            for role, text in scenario.messages:
                builder.add_message(role=role, text=text)
            archive.write_parsed(_record_to_parsed_session(builder.conv, builder.messages, builder.attachments))
            ids[scenario.name] = builder.native_session_id()
    return db_path, ids


# ---------------------------------------------------------------------------
# Archive read surfaces
# ---------------------------------------------------------------------------


class _SubstrateSurface:
    name = "substrate"

    def __init__(self, *, archive_root: Path) -> None:
        self._archive_root = archive_root

    async def provider_ids(self, origin: str) -> tuple[str, ...]:
        with ArchiveStore.open_existing(self._archive_root) as archive:
            return tuple(sorted(s.session_id for s in archive.list_summaries(origin=origin, limit=100)))

    async def search_ids(self, query: str) -> tuple[str, ...]:
        with ArchiveStore.open_existing(self._archive_root) as archive:
            return tuple(sorted({hit.session_id for hit in archive.search_summaries(query, limit=100)}))

    async def archive_facts(self) -> tuple[int, int, dict[str, int], tuple[str, ...]]:
        with ArchiveStore.open_existing(self._archive_root) as archive:
            summaries = archive.list_summaries(limit=1000)
        ids = tuple(sorted(s.session_id for s in summaries))
        origin_counts: dict[str, int] = {}
        total_messages = 0
        for s in summaries:
            total_messages += s.message_count
            origin_counts[s.origin] = origin_counts.get(s.origin, 0) + 1
        return len(summaries), total_messages, origin_counts, ids

    async def close(self) -> None:
        return None


class _FacadeSurface:
    name = "facade"

    def __init__(self, *, archive_root: Path, db_path: Path) -> None:
        self._archive = Polylogue(archive_root=archive_root, db_path=db_path)

    async def provider_ids(self, origin: str) -> tuple[str, ...]:
        convs = await self._archive.list_sessions(origin=origin, limit=100)
        return tuple(sorted(str(c.id) for c in convs))

    async def search_ids(self, query: str) -> tuple[str, ...]:
        result = await self._archive.search(query, limit=100)
        return tuple(sorted({str(hit.session_id) for hit in result.hits}))

    async def archive_facts(self) -> tuple[int, int, dict[str, int], tuple[str, ...]]:
        stats = await self._archive.stats()
        convs = await self._archive.list_sessions(limit=1000)
        ids = tuple(sorted(str(c.id) for c in convs))
        return stats.session_count, stats.message_count, dict(stats.origins), ids

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

    async def provider_ids(self, origin: str) -> tuple[str, ...]:
        payload = await self._server._tool_manager._tools["list_sessions"].fn(limit=100, origin=origin)
        items = json.loads(payload).get("items", [])
        return tuple(sorted({str(i["id"]) for i in items}))

    async def search_ids(self, query: str) -> tuple[str, ...]:
        payload = await self._server._tool_manager._tools["search"].fn(query=query, limit=100)
        parsed = json.loads(payload)
        hits = parsed.get("hits", parsed.get("items", []))
        ids: list[str] = []
        for hit in hits:
            if "session_id" in hit:
                ids.append(str(hit["session_id"]))
            elif isinstance(hit.get("session"), dict) and "id" in hit["session"]:
                ids.append(str(hit["session"]["id"]))
        return tuple(sorted(set(ids)))

    async def archive_facts(self) -> tuple[int, int, dict[str, int], tuple[str, ...]]:
        payload = await self._server._tool_manager._tools["list_sessions"].fn(limit=1000)
        items = json.loads(payload).get("items", [])
        ids = tuple(sorted(str(i["id"]) for i in items))
        origin_counts: dict[str, int] = {}
        total_messages = 0
        for item in items:
            total_messages += int(item.get("message_count", 0))
            origin = str(item["origin"])
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
        return len(items), total_messages, origin_counts, ids

    async def close(self) -> None:
        from polylogue.mcp.server_support import _set_runtime_services

        await self._services.close()
        _set_runtime_services(None)


@dataclass
class _SurfaceSet:
    substrate: _SubstrateSurface
    facade: _FacadeSurface
    mcp: _MCPSurface
    ids: dict[str, str]

    @property
    def all(self) -> tuple[object, ...]:
        return (self.substrate, self.facade, self.mcp)

    async def close(self) -> None:
        await self.facade.close()
        await self.mcp.close()
        await self.substrate.close()


@pytest.fixture()
def retrieval_archive(workspace_env: dict[str, Path]) -> tuple[Path, dict[str, str]]:
    return _seed(workspace_env)


@pytest.fixture()
async def surfaces(
    workspace_env: dict[str, Path],
    retrieval_archive: tuple[Path, dict[str, str]],
) -> AsyncIterator[_SurfaceSet]:
    db_path, ids = retrieval_archive
    s = _SurfaceSet(
        substrate=_SubstrateSurface(archive_root=workspace_env["archive_root"]),
        facade=_FacadeSurface(archive_root=workspace_env["archive_root"], db_path=db_path),
        mcp=_MCPSurface(db_path=db_path),
        ids=ids,
    )
    try:
        yield s
    finally:
        await s.close()


# ---------------------------------------------------------------------------
# FTS readiness contract (pure)
# ---------------------------------------------------------------------------


def test_fts_readiness_rejects_negative_gap_and_missing_triggers() -> None:
    with pytest.raises(DatabaseError):
        check_fts_readiness(
            {"exists": True, "ready": False, "indexed_rows": 110, "total_rows": 100, "triggers_present": True}
        )
    with pytest.raises(DatabaseError):
        check_fts_readiness(
            {"exists": True, "ready": False, "indexed_rows": 99, "total_rows": 100, "triggers_present": False}
        )
    with pytest.raises(DatabaseError):
        check_fts_readiness(
            {"exists": True, "ready": False, "indexed_rows": 99, "total_rows": 100, "triggers_present": True}
        )


# ---------------------------------------------------------------------------
# Cross-surface retrieval agreement
# ---------------------------------------------------------------------------


class TestRetrievalSurfaceAgreement:
    @pytest.mark.asyncio
    async def test_archive_facts_agree_across_surfaces(self, surfaces: _SurfaceSet) -> None:
        facts = [await surface.archive_facts() for surface in surfaces.all]  # type: ignore[attr-defined]
        reference = facts[0]
        for surface, fact in zip(surfaces.all, facts, strict=True):
            assert fact == reference, f"{surface.name} disagrees: {fact} != {reference}"  # type: ignore[attr-defined]
        total_sessions, total_messages, origin_counts, _ids = reference
        assert total_sessions == 4
        assert total_messages == 8
        assert origin_counts == {"chatgpt-export": 1, "claude-code-session": 2, "codex-session": 1}

    @pytest.mark.asyncio
    async def test_origin_filters_partition_the_archive(self, surfaces: _SurfaceSet) -> None:
        _conv, _msg, _prov, all_ids = await surfaces.facade.archive_facts()
        all_set = set(all_ids)
        partitions: dict[str, tuple[str, ...]] = {}
        for origin, names in (
            ("chatgpt-export", ("chatgpt-retrieval-1",)),
            ("claude-code-session", ("claude-retrieval-1", "claude-retrieval-2")),
            ("codex-session", ("codex-retrieval-1",)),
        ):
            expected = tuple(sorted(surfaces.ids[name] for name in names))
            for surface in surfaces.all:
                ids = await surface.provider_ids(origin)  # type: ignore[attr-defined]
                assert ids == expected, f"{surface.name} returned {ids} for origin {origin}"  # type: ignore[attr-defined]
            partitions[origin] = expected
        # disjoint + exhaustive
        union: set[str] = set()
        for ids in partitions.values():
            assert union.isdisjoint(ids)
            union |= set(ids)
        assert union == all_set

    @pytest.mark.asyncio
    async def test_fts_search_is_consistent_across_surfaces(self, surfaces: _SurfaceSet) -> None:
        cases = (
            ("property", ("chatgpt-retrieval-1", "claude-retrieval-1")),
            ("memory", ("claude-retrieval-2",)),
            ("authentication", ("codex-retrieval-1",)),
        )
        for query, names in cases:
            expected = tuple(sorted(surfaces.ids[name] for name in names))
            for surface in surfaces.all:
                ids = await surface.search_ids(query)  # type: ignore[attr-defined]
                assert ids == expected, f"{surface.name} returned {ids} for query {query!r}"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Archive FTS index invariant
# ---------------------------------------------------------------------------


class TestRetrievalIndexInvariants:
    def test_fts_index_covers_exactly_indexable_blocks(
        self,
        retrieval_archive: tuple[Path, dict[str, str]],
    ) -> None:
        db_path, _ = retrieval_archive
        with sqlite3.connect(db_path) as conn:
            indexable_blocks = {
                str(row[0])
                for row in conn.execute("SELECT block_id FROM blocks WHERE NULLIF(text, '') IS NOT NULL").fetchall()
            }
            indexed_blocks = {
                str(row[0])
                for row in conn.execute(
                    "SELECT b.block_id FROM messages_fts f JOIN blocks b ON b.rowid = f.rowid"
                ).fetchall()
            }
        assert indexed_blocks == indexable_blocks

    @pytest.mark.asyncio
    async def test_message_counts_match_across_surfaces(
        self,
        retrieval_archive: tuple[Path, dict[str, str]],
        workspace_env: dict[str, Path],
    ) -> None:
        db_path, ids = retrieval_archive
        # Substrate per-session message counts.
        with ArchiveStore.open_existing(workspace_env["archive_root"]) as archive:
            substrate_counts = {s.session_id: s.message_count for s in archive.list_summaries(limit=1000)}

        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            facade_counts: dict[str, int] = {}
            for session_id in ids.values():
                session = await poly.get_session(session_id)
                assert session is not None
                facade_counts[session_id] = len(session.messages)

        assert facade_counts == substrate_counts
