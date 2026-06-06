"""Cross-surface agreement: prove repository, hydration, and records agree.

When the same session is viewed through different surfaces, the
semantic facts must be identical. A failure here means two surfaces
disagree about what's in the archive.
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
from tests.infra.oracles import (
    assert_archive_surfaces_agree,
    assert_provider_partition_exhaustive,
    assert_session_surfaces_agree,
)
from tests.infra.query_cases import ArchiveQueryCase
from tests.infra.semantic_facts import SessionFacts
from tests.infra.surfaces import ArchiveSurfaceSet, build_archive_surface_set


@pytest.fixture()
def multi_provider_archive(workspace_env: Mapping[str, Path]) -> tuple[Path, tuple[ArchiveScenario, ...]]:
    """Populate a DB with sessions across providers."""
    scenarios = (
        ArchiveScenario(
            name="chatgpt-xsurf-1",
            provider="chatgpt",
            title="GPT chat",
            messages=(
                ScenarioMessage(role="user", text="Hello GPT", message_id="gpt-u1"),
                ScenarioMessage(role="assistant", text="Hello user", message_id="gpt-a1"),
            ),
        ),
        ArchiveScenario(
            name="claude-xsurf-1",
            provider="claude-code",
            title="Claude session",
            messages=(
                ScenarioMessage(role="user", text="Refactor this", message_id="claude-u1"),
                ScenarioMessage(role="assistant", text="Done", message_id="claude-a1"),
                ScenarioMessage(role="user", text="Thanks", message_id="claude-u2"),
            ),
        ),
        ArchiveScenario(
            name="codex-xsurf-1",
            provider="codex",
            title="Codex work",
            messages=(ScenarioMessage(role="user", text="Generate code", message_id="codex-u1"),),
        ),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, scenarios)
    return db_path, scenarios


@pytest.fixture()
async def multi_provider_surfaces(
    workspace_env: Mapping[str, Path],
    multi_provider_archive: tuple[Path, tuple[ArchiveScenario, ...]],
) -> AsyncIterator[ArchiveSurfaceSet]:
    db_path, scenarios = multi_provider_archive
    surfaces = build_archive_surface_set(
        db_path=db_path,
        archive_root=workspace_env["archive_root"],
        scenarios=scenarios,
    )
    try:
        yield surfaces
    finally:
        await surfaces.close()


# ---------------------------------------------------------------------------
# Record-level vs hydration-level facts agree
# ---------------------------------------------------------------------------


class TestRecordVsHydrationAgreement:
    @pytest.mark.asyncio()
    async def test_facts_agree_for_each_session_across_surfaces(
        self,
        multi_provider_archive: tuple[Path, tuple[ArchiveScenario, ...]],
        multi_provider_surfaces: ArchiveSurfaceSet,
    ) -> None:
        _, scenarios = multi_provider_archive
        for scenario in scenarios:
            facts = [await surface.session_facts(scenario) for surface in multi_provider_surfaces.surfaces]
            assert_session_surfaces_agree(*facts)


# ---------------------------------------------------------------------------
# Archive-level facts: direct SQL vs count queries agree
# ---------------------------------------------------------------------------


class TestArchiveFactsConsistency:
    @pytest.mark.asyncio()
    async def test_archive_facts_internally_consistent_across_surfaces(
        self,
        multi_provider_surfaces: ArchiveSurfaceSet,
    ) -> None:
        facts = [await surface.archive_facts() for surface in multi_provider_surfaces.surfaces]
        assert_archive_surfaces_agree(*facts)
        db_facts = facts[0]
        assert db_facts.total_sessions == 3
        assert sum(db_facts.provider_counts.values()) == db_facts.total_sessions
        assert db_facts.total_messages > 0
        assert db_facts.provider_counts.get("chatgpt") == 1
        assert db_facts.provider_counts.get("claude-code") == 1
        assert db_facts.provider_counts.get("codex") == 1

    @pytest.mark.asyncio()
    async def test_provider_partition_exhaustive(
        self,
        multi_provider_surfaces: ArchiveSurfaceSet,
    ) -> None:
        """Every session belongs to exactly one provider in the facts."""
        facts = await multi_provider_surfaces.surfaces[0].archive_facts()
        all_ids = set(facts.session_ids)
        ids_by_provider: dict[str, tuple[str, ...]] = {}
        for provider in facts.provider_counts:
            query_case = ArchiveQueryCase(name=f"provider-{provider}", provider=provider, expected_ids=())
            provider_ids = [await surface.query_ids(query_case) for surface in multi_provider_surfaces.surfaces]
            first = provider_ids[0]
            assert all(ids == first for ids in provider_ids), provider_ids
            ids_by_provider[provider] = first

        assert_provider_partition_exhaustive(all_session_ids=all_ids, ids_by_provider=ids_by_provider)


# ---------------------------------------------------------------------------
# Synthetic roundtrip: generated payload facts match hydrated facts
# ---------------------------------------------------------------------------


class TestSyntheticRoundtripFactAgreement:
    @pytest.mark.parametrize("source_name", ["chatgpt", "claude-code", "claude-ai", "codex", "gemini"])
    def test_parsed_vs_hydrated_facts_agree(
        self,
        source_name: str,
        workspace_env: Mapping[str, Path],
    ) -> None:
        from polylogue.schemas.synthetic.core import SyntheticCorpus
        from polylogue.storage.sqlite.connection import open_connection
        from tests.infra.pipeline_roundtrip import parse_and_transform_payload, save_transform_and_hydrate
        from tests.infra.storage_records import db_setup

        corpus = SyntheticCorpus.for_provider(source_name)
        raw_bytes = corpus.generate(count=1, seed=99)[0]

        roundtrip = parse_and_transform_payload(
            source_name,
            raw_bytes,
            workspace_env["archive_root"],
            f"xsurf-{source_name}",
        )

        db_path = db_setup(workspace_env)
        with open_connection(db_path) as conn:
            hydrated = save_transform_and_hydrate(roundtrip.transform, conn)
            hydrated_facts = SessionFacts.from_domain_session(hydrated)

            assert hydrated_facts.message_count == len(roundtrip.parsed.messages)
            assert hydrated_facts.provider == str(roundtrip.parsed.source_name)
            assert hydrated_facts.title == roundtrip.parsed.title
