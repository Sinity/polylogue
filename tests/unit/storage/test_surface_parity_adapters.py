"""Cross-surface query parity for CLI and MCP adapters (#1060).

These tests close the surface-parity loop on the public read adapters.  The
substrate parity tests in ``test_query_parity.py`` already pin agreement
between SQLite, the repository, and the async facade.  The adapter tests
here add the Click CLI (``polylogue list --format json``) and the MCP
server tool (``list_conversations``) so any drift between the substrate
filter chain and the published surfaces is caught locally.

Surface stack exercised here (top to bottom):

  CLISurface  --polylogue list --format json--+
                                              |
  MCPSurface  --list_conversations------------+
                                              |---> ConversationFilter
  FacadeSurface --Polylogue.operations--------+    --> SQLiteBackend
                                              |
  RepositorySurface --ConversationRepository--+

Every parity case is expressed once as an :class:`ArchiveQueryCase` and
projected through every surface; the assertion is that all surfaces return
the same id tuple.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest

from tests.infra.archive_scenarios import (
    ArchiveScenario,
    ScenarioMessage,
    seed_workspace_scenarios,
)
from tests.infra.query_cases import ArchiveQueryCase
from tests.infra.surfaces import (
    ArchiveSurfaceSet,
    build_adapter_surface_set,
)

# ---------------------------------------------------------------------------
# Scenario corpus
# ---------------------------------------------------------------------------


def _parity_scenarios() -> tuple[ArchiveScenario, ...]:
    """Provider-diverse scenarios shaped to exercise the parity matrix.

    Each scenario is uniquely identifiable along multiple axes:
      * provider — three providers, asymmetric counts (1/2/1)
      * message text — stable FTS tokens (``aardvark``, ``buffalo``)
      * message volume — counts span 1/2/3/5 so min/max filters bite
    """
    return (
        ArchiveScenario(
            name="parity-chatgpt-1",
            provider="chatgpt",
            title="ChatGPT aardvark conversation",
            messages=(
                ScenarioMessage(role="user", text="aardvark question", message_id="parity-chatgpt-1-u1"),
                ScenarioMessage(role="assistant", text="aardvark answer", message_id="parity-chatgpt-1-a1"),
            ),
        ),
        ArchiveScenario(
            name="parity-claude-1",
            provider="claude-code",
            title="Claude buffalo session",
            messages=(
                ScenarioMessage(role="user", text="buffalo question", message_id="parity-claude-1-u1"),
                ScenarioMessage(role="assistant", text="buffalo answer", message_id="parity-claude-1-a1"),
                ScenarioMessage(role="user", text="buffalo followup", message_id="parity-claude-1-u2"),
            ),
        ),
        ArchiveScenario(
            name="parity-claude-2",
            provider="claude-code",
            title="Claude long detailed conversation",
            messages=(
                ScenarioMessage(role="user", text="long detailed question one", message_id="parity-claude-2-u1"),
                ScenarioMessage(role="assistant", text="long detailed answer one", message_id="parity-claude-2-a1"),
                ScenarioMessage(role="user", text="long detailed question two", message_id="parity-claude-2-u2"),
                ScenarioMessage(role="assistant", text="long detailed answer two", message_id="parity-claude-2-a2"),
                ScenarioMessage(role="user", text="long detailed question three", message_id="parity-claude-2-u3"),
            ),
        ),
        ArchiveScenario(
            name="parity-codex-1",
            provider="codex",
            title="Codex single-message conversation",
            messages=(ScenarioMessage(role="user", text="codex isolated request", message_id="parity-codex-1-u1"),),
        ),
    )


@pytest.fixture()
def parity_archive(workspace_env: Mapping[str, Path]) -> tuple[Path, tuple[ArchiveScenario, ...]]:
    scenarios = _parity_scenarios()
    db_path, _ = seed_workspace_scenarios(workspace_env, scenarios)
    return db_path, scenarios


@pytest.fixture()
async def adapter_surfaces(
    workspace_env: Mapping[str, Path],
    parity_archive: tuple[Path, tuple[ArchiveScenario, ...]],
) -> AsyncIterator[ArchiveSurfaceSet]:
    db_path, _ = parity_archive
    surfaces = build_adapter_surface_set(
        db_path=db_path,
        archive_root=workspace_env["archive_root"],
    )
    try:
        yield surfaces
    finally:
        await surfaces.close()


# ---------------------------------------------------------------------------
# Parity assertions
# ---------------------------------------------------------------------------


async def _assert_parity(
    surfaces: ArchiveSurfaceSet,
    case: ArchiveQueryCase,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Run ``case`` through every surface and assert id-tuple agreement."""
    expected = tuple(sorted(case.expected_ids))
    projections: dict[str, tuple[str, ...]] = {}
    for surface in surfaces.surfaces:
        ids = await surface.query_ids(case)
        projections[surface.name] = ids
        assert ids == expected, f"{surface.name} returned {ids} for {case.name!r}; expected {expected}"
        count = await surface.query_count(case)
        assert count == len(expected), f"{surface.name} count={count} for {case.name!r}; expected {len(expected)}"
    return expected, projections


# ---------------------------------------------------------------------------
# Provider parity
# ---------------------------------------------------------------------------


class TestProviderParity:
    @pytest.mark.contract
    @pytest.mark.asyncio()
    async def test_provider_chatgpt_parity(
        self,
        adapter_surfaces: ArchiveSurfaceSet,
    ) -> None:
        case = ArchiveQueryCase(
            name="provider-chatgpt",
            provider="chatgpt",
            expected_ids=("parity-chatgpt-1",),
        )
        await _assert_parity(adapter_surfaces, case)

    @pytest.mark.contract
    @pytest.mark.asyncio()
    async def test_provider_claude_parity(
        self,
        adapter_surfaces: ArchiveSurfaceSet,
    ) -> None:
        case = ArchiveQueryCase(
            name="provider-claude",
            provider="claude-code",
            expected_ids=("parity-claude-1", "parity-claude-2"),
        )
        await _assert_parity(adapter_surfaces, case)

    @pytest.mark.contract
    @pytest.mark.asyncio()
    async def test_provider_codex_parity(
        self,
        adapter_surfaces: ArchiveSurfaceSet,
    ) -> None:
        case = ArchiveQueryCase(
            name="provider-codex",
            provider="codex",
            expected_ids=("parity-codex-1",),
        )
        await _assert_parity(adapter_surfaces, case)


# ---------------------------------------------------------------------------
# Text-search (FTS) parity
# ---------------------------------------------------------------------------


class TestSearchParity:
    @pytest.mark.contract
    @pytest.mark.asyncio()
    async def test_search_unique_token_parity(
        self,
        adapter_surfaces: ArchiveSurfaceSet,
    ) -> None:
        case = ArchiveQueryCase(
            name="search-aardvark",
            search_text="aardvark",
            expected_ids=("parity-chatgpt-1",),
        )
        await _assert_parity(adapter_surfaces, case)

    @pytest.mark.contract
    @pytest.mark.asyncio()
    async def test_search_shared_token_parity(
        self,
        adapter_surfaces: ArchiveSurfaceSet,
    ) -> None:
        case = ArchiveQueryCase(
            name="search-buffalo",
            search_text="buffalo",
            expected_ids=("parity-claude-1",),
        )
        await _assert_parity(adapter_surfaces, case)


# ---------------------------------------------------------------------------
# Stats-join filter parity — the highest drift risk
# ---------------------------------------------------------------------------


class TestStatsJoinParity:
    """min_messages/max_messages/min_words are SQL pushdowns via stats join.

    The pushdown is configured per surface in ``_needs_stats_join`` and the
    spec; if any surface fails to forward a flag, this test catches it.
    """

    @pytest.mark.contract
    @pytest.mark.asyncio()
    async def test_min_messages_filter_parity(
        self,
        adapter_surfaces: ArchiveSurfaceSet,
    ) -> None:
        # Counts: chatgpt-1=2, claude-1=3, claude-2=5, codex-1=1.
        # min_messages=3 → claude-1, claude-2.
        case = ArchiveQueryCase(
            name="min-messages-3",
            min_messages=3,
            expected_ids=("parity-claude-1", "parity-claude-2"),
        )
        await _assert_parity(adapter_surfaces, case)

    @pytest.mark.contract
    @pytest.mark.asyncio()
    async def test_max_messages_filter_parity(
        self,
        adapter_surfaces: ArchiveSurfaceSet,
    ) -> None:
        # max_messages=2 → chatgpt-1 (2 msgs), codex-1 (1 msg).
        case = ArchiveQueryCase(
            name="max-messages-2",
            max_messages=2,
            expected_ids=("parity-chatgpt-1", "parity-codex-1"),
        )
        await _assert_parity(adapter_surfaces, case)

    @pytest.mark.contract
    @pytest.mark.asyncio()
    async def test_min_words_filter_parity(
        self,
        adapter_surfaces: ArchiveSurfaceSet,
    ) -> None:
        # claude-2 carries 5 messages of ~4 words each → ~20 words; everyone
        # else is below. min_words=15 → claude-2.
        case = ArchiveQueryCase(
            name="min-words-15",
            min_words=15,
            expected_ids=("parity-claude-2",),
        )
        await _assert_parity(adapter_surfaces, case)


# ---------------------------------------------------------------------------
# Limit/offset parity
# ---------------------------------------------------------------------------


class TestLimitOffsetParity:
    @pytest.mark.contract
    @pytest.mark.asyncio()
    async def test_provider_with_limit_subset_parity(
        self,
        adapter_surfaces: ArchiveSurfaceSet,
    ) -> None:
        """provider=claude-code + limit=1 returns one id; the id must agree across surfaces.

        We don't pin which id is selected (sort order is surface-dependent
        at this level), only that every surface returns the same single id.
        """
        case = ArchiveQueryCase(
            name="provider-claude-limit-1",
            provider="claude-code",
            limit=1,
            # expected_ids is filled at assert time below — we only require
            # the size is 1 and surfaces agree on the chosen id.
            expected_ids=(),
        )
        # Use a relaxed parity assertion: collect ids per surface, then
        # assert all surfaces returned the same single id.
        seen: dict[str, tuple[str, ...]] = {}
        for surface in adapter_surfaces.surfaces:
            ids = await surface.query_ids(case)
            assert len(ids) == 1, f"{surface.name} returned {len(ids)} ids for limit=1: {ids}"
            seen[surface.name] = ids
        ref = next(iter(seen.values()))
        for name, ids in seen.items():
            assert ids == ref, f"limit-1 disagreement: {name}={ids} vs reference={ref}"


# ---------------------------------------------------------------------------
# Combined-filter parity — provider + stats-join
# ---------------------------------------------------------------------------


class TestCombinedFilterParity:
    @pytest.mark.contract
    @pytest.mark.asyncio()
    async def test_provider_plus_min_messages_parity(
        self,
        adapter_surfaces: ArchiveSurfaceSet,
    ) -> None:
        # claude-code conversations with >= 4 messages → claude-2 only.
        case = ArchiveQueryCase(
            name="claude-min-messages-4",
            provider="claude-code",
            min_messages=4,
            expected_ids=("parity-claude-2",),
        )
        await _assert_parity(adapter_surfaces, case)
