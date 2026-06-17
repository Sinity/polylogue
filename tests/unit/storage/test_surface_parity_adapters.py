"""Cross-surface query parity over the archive.

The archive substrate (SQLite backend, repository) and the CLI list path are no
longer usable read surfaces for this matrix (the CLI native list path is a known
prod gap; the repository/operations layers are legacy and reject the native DB).
This file closes the parity loop on the two archive read surfaces that fully
express the filter matrix:

  * ``ArchiveStore`` — the archive storage substrate (``list_summaries``)
  * ``MCPSurface`` — the MCP ``list_sessions`` tool (the published adapter)

plus the async ``Polylogue`` facade for the projections it exposes
(provider filter, full listing). Each parity case is expressed once as a
filter and projected through every surface that can express it; the assertion is
that all surfaces return the same id tuple.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder, _record_to_parsed_session, db_setup

# ---------------------------------------------------------------------------
# Scenario corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scenario:
    name: str
    provider: str
    title: str
    messages: tuple[tuple[str, str], ...]  # (role, text)


def _scenarios() -> tuple[_Scenario, ...]:
    return (
        _Scenario(
            name="parity-chatgpt-1",
            provider="chatgpt",
            title="ChatGPT aardvark session",
            messages=(("user", "aardvark question"), ("assistant", "aardvark answer")),
        ),
        _Scenario(
            name="parity-claude-1",
            provider="claude-code",
            title="Claude buffalo session",
            messages=(
                ("user", "buffalo question"),
                ("assistant", "buffalo answer"),
                ("user", "buffalo followup"),
            ),
        ),
        _Scenario(
            name="parity-claude-2",
            provider="claude-code",
            title="Claude long detailed session",
            messages=(
                ("user", "long detailed question one"),
                ("assistant", "long detailed answer one"),
                ("user", "long detailed question two"),
                ("assistant", "long detailed answer two"),
                ("user", "long detailed question three"),
            ),
        ),
        _Scenario(
            name="parity-codex-1",
            provider="codex",
            title="Codex single-message session",
            messages=(("user", "codex isolated request"),),
        ),
    )


def _origin_for_provider(provider: str | None) -> str | None:
    if provider is None:
        return None
    return origin_from_provider(Provider.from_string(provider)).value


def _seed(workspace_env: dict[str, Path]) -> tuple[Path, dict[str, str]]:
    """Seed the archive; return (index_db_path, {scenario_name: native_id})."""
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


@dataclass(frozen=True)
class _Case:
    name: str
    expected: tuple[str, ...]
    provider: str | None = None
    search: str | None = None
    min_messages: int | None = None
    max_messages: int | None = None
    min_words: int | None = None
    limit: int | None = None
    # Surfaces that can express this case (others are skipped for the case).
    surfaces: tuple[str, ...] = ("substrate", "mcp", "facade")


# ---------------------------------------------------------------------------
# Native surfaces
# ---------------------------------------------------------------------------


def _hit_id(hit: dict[str, object]) -> str:
    """Extract the session id from an MCP search hit envelope."""
    if "session_id" in hit:
        return str(hit["session_id"])
    if "id" in hit:
        return str(hit["id"])
    session = hit.get("session")
    if isinstance(session, dict) and "id" in session:
        return str(session["id"])
    raise KeyError(f"no session id in search hit: {sorted(hit)}")


class _SubstrateSurface:
    name = "substrate"

    def __init__(self, *, archive_root: Path) -> None:
        self._archive_root = archive_root

    async def ids(self, case: _Case) -> tuple[str, ...]:
        with ArchiveStore.open_existing(self._archive_root) as archive:
            if case.search is not None:
                hits = archive.search_summaries(case.search, limit=case.limit or 100)
                return tuple(sorted({hit.session_id for hit in hits}))
            origin = None
            if case.provider is not None:
                from polylogue.core.enums import Provider
                from polylogue.core.sources import origin_from_provider

                origin = origin_from_provider(Provider.from_string(case.provider)).value
            summaries = archive.list_summaries(
                origin=origin,
                min_messages=case.min_messages,
                max_messages=case.max_messages,
                min_words=case.min_words,
                limit=case.limit or 100,
            )
            return tuple(sorted(summary.session_id for summary in summaries))

    async def close(self) -> None:
        return None


class _MCPSurface:
    name = "mcp"

    def __init__(self, *, db_path: Path) -> None:
        from polylogue.mcp.server import build_server
        from polylogue.mcp.server_support import _set_runtime_services
        from polylogue.services import build_runtime_services

        self._services = build_runtime_services(db_path=db_path)
        _set_runtime_services(self._services)
        self._server = build_server(role="admin")

    async def ids(self, case: _Case) -> tuple[str, ...]:
        if case.search is not None:
            payload = await self._server._tool_manager._tools["search"].fn(query=case.search, limit=case.limit or 100)
            parsed = json.loads(payload)
            hits = parsed.get("hits", parsed.get("items", []))
            return tuple(sorted({_hit_id(hit) for hit in hits}))
        payload = await self._server._tool_manager._tools["list_sessions"].fn(
            limit=case.limit if case.limit is not None else 100,
            origin=_origin_for_provider(case.provider),
            min_messages=case.min_messages,
            max_messages=case.max_messages,
            min_words=case.min_words,
        )
        items = json.loads(payload).get("items", [])
        return tuple(sorted({str(item["id"]) for item in items}))

    async def close(self) -> None:
        from polylogue.mcp.server_support import _set_runtime_services

        await self._services.close()
        _set_runtime_services(None)


class _FacadeSurface:
    name = "facade"

    def __init__(self, *, archive_root: Path, db_path: Path) -> None:
        self._archive = Polylogue(archive_root=archive_root, db_path=db_path)

    async def ids(self, case: _Case) -> tuple[str, ...]:
        if case.search is not None:
            result = await self._archive.search(case.search, limit=case.limit or 100)
            return tuple(sorted({str(hit.session_id) for hit in result.hits}))
        convs = await self._archive.list_sessions(
            origin=_origin_for_provider(case.provider), limit=case.limit if case.limit is not None else 100
        )
        return tuple(sorted({str(c.id) for c in convs}))

    async def close(self) -> None:
        await self._archive.close()


@dataclass
class _SurfaceSet:
    substrate: _SubstrateSurface
    mcp: _MCPSurface
    facade: _FacadeSurface
    ids: dict[str, str] = field(default_factory=dict)

    def for_case(self, case: _Case) -> list[object]:
        mapping = {"substrate": self.substrate, "mcp": self.mcp, "facade": self.facade}
        return [mapping[name] for name in case.surfaces]

    async def close(self) -> None:
        await self.facade.close()
        await self.mcp.close()
        await self.substrate.close()


@pytest.fixture()
async def surfaces(workspace_env: dict[str, Path]) -> AsyncIterator[_SurfaceSet]:
    db_path, ids = _seed(workspace_env)
    s = _SurfaceSet(
        substrate=_SubstrateSurface(archive_root=workspace_env["archive_root"]),
        mcp=_MCPSurface(db_path=db_path),
        facade=_FacadeSurface(archive_root=workspace_env["archive_root"], db_path=db_path),
        ids=ids,
    )
    try:
        yield s
    finally:
        await s.close()


async def _assert_parity(surface_set: _SurfaceSet, case: _Case) -> None:
    expected = tuple(sorted(surface_set.ids[name] for name in case.expected))
    surfaces = surface_set.for_case(case)
    assert len(surfaces) >= 2, f"case {case.name!r} must be projected through >= 2 surfaces"
    for surface in surfaces:
        ids = await surface.ids(case)  # type: ignore[attr-defined]
        assert ids == expected, f"{surface.name} returned {ids} for {case.name!r}; expected {expected}"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Provider parity (substrate + mcp + facade)
# ---------------------------------------------------------------------------


class TestProviderParity:
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_provider_chatgpt(self, surfaces: _SurfaceSet) -> None:
        await _assert_parity(surfaces, _Case("provider-chatgpt", ("parity-chatgpt-1",), provider="chatgpt"))

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_provider_claude(self, surfaces: _SurfaceSet) -> None:
        await _assert_parity(
            surfaces,
            _Case("provider-claude", ("parity-claude-1", "parity-claude-2"), provider="claude-code"),
        )

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_provider_codex(self, surfaces: _SurfaceSet) -> None:
        await _assert_parity(surfaces, _Case("provider-codex", ("parity-codex-1",), provider="codex"))


# ---------------------------------------------------------------------------
# Text-search (FTS) parity (substrate + mcp + facade)
# ---------------------------------------------------------------------------


class TestSearchParity:
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_search_unique_token(self, surfaces: _SurfaceSet) -> None:
        await _assert_parity(surfaces, _Case("search-aardvark", ("parity-chatgpt-1",), search="aardvark"))

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_search_shared_token(self, surfaces: _SurfaceSet) -> None:
        await _assert_parity(surfaces, _Case("search-buffalo", ("parity-claude-1",), search="buffalo"))


# ---------------------------------------------------------------------------
# Stats-join filter parity (substrate + mcp — the highest drift risk)
# ---------------------------------------------------------------------------


class TestStatsJoinParity:
    """min/max-messages and min-words are stats-join pushdowns.

    The archive facade ``list_sessions`` does not expose these filters, so
    parity here is asserted between the archive substrate (``ArchiveStore``) and the
    MCP adapter, which both implement the pushdown.
    """

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_min_messages(self, surfaces: _SurfaceSet) -> None:
        # counts: chatgpt-1=2, claude-1=3, claude-2=5, codex-1=1. min=3 → claude-1, claude-2.
        await _assert_parity(
            surfaces,
            _Case(
                "min-messages-3",
                ("parity-claude-1", "parity-claude-2"),
                min_messages=3,
                surfaces=("substrate", "mcp"),
            ),
        )

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_max_messages(self, surfaces: _SurfaceSet) -> None:
        # max=2 → chatgpt-1 (2), codex-1 (1).
        await _assert_parity(
            surfaces,
            _Case(
                "max-messages-2",
                ("parity-chatgpt-1", "parity-codex-1"),
                max_messages=2,
                surfaces=("substrate", "mcp"),
            ),
        )

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_min_words(self, surfaces: _SurfaceSet) -> None:
        # claude-2 carries 5 messages of ~4 words each → ~20 words; below otherwise.
        await _assert_parity(
            surfaces,
            _Case("min-words-15", ("parity-claude-2",), min_words=15, surfaces=("substrate", "mcp")),
        )


# ---------------------------------------------------------------------------
# Limit parity (substrate + mcp + facade agree on a single chosen id)
# ---------------------------------------------------------------------------


class TestLimitParity:
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_provider_with_limit_one(self, surfaces: _SurfaceSet) -> None:
        """provider=claude-code + limit=1 returns one id; surfaces must agree on it."""
        case = _Case("provider-claude-limit-1", (), provider="claude-code", limit=1)
        seen: dict[str, tuple[str, ...]] = {}
        for surface in surfaces.for_case(case):
            ids = await surface.ids(case)  # type: ignore[attr-defined]
            assert len(ids) == 1, f"{surface.name} returned {len(ids)} ids for limit=1: {ids}"  # type: ignore[attr-defined]
            seen[surface.name] = ids  # type: ignore[attr-defined]
        ref = next(iter(seen.values()))
        for name, ids in seen.items():
            assert ids == ref, f"limit-1 disagreement: {name}={ids} vs reference={ref}"


# ---------------------------------------------------------------------------
# Combined-filter parity (substrate + mcp)
# ---------------------------------------------------------------------------


class TestCombinedFilterParity:
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_provider_plus_min_messages(self, surfaces: _SurfaceSet) -> None:
        # claude-code sessions with >= 4 messages → claude-2 only.
        await _assert_parity(
            surfaces,
            _Case(
                "claude-min-messages-4",
                ("parity-claude-2",),
                provider="claude-code",
                min_messages=4,
                surfaces=("substrate", "mcp"),
            ),
        )
