"""Retrieval readiness laws over shared archive scenarios.

These tests prove provider filters, FTS search, counts, and aggregate archive
facts agree across direct SQL, hydrated records, repository, and facade
surfaces.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest

from polylogue.errors import DatabaseError
from polylogue.storage.fts.fts_lifecycle import check_fts_readiness
from tests.infra.archive_scenarios import ArchiveScenario, ScenarioMessage, seed_workspace_scenarios
from tests.infra.oracles import assert_archive_surfaces_agree, assert_provider_partition_exhaustive
from tests.infra.query_cases import ArchiveQueryCase
from tests.infra.surfaces import ArchiveSurfaceSet, build_archive_surface_set


@pytest.fixture()
def retrieval_archive(workspace_env: Mapping[str, Path]) -> tuple[Path, tuple[ArchiveScenario, ...]]:
    """Seed provider-diverse conversations with stable search terms."""
    scenarios = (
        ArchiveScenario(
            name="chatgpt-retrieval-1",
            provider="chatgpt",
            title="ChatGPT conversation about testing",
            messages=(
                ScenarioMessage(role="user", text="How do I write property tests?", message_id="chatgpt-u1"),
                ScenarioMessage(
                    role="assistant",
                    text="Property tests verify invariants using random inputs",
                    message_id="chatgpt-a1",
                ),
            ),
        ),
        ArchiveScenario(
            name="claude-retrieval-1",
            provider="claude-code",
            title="Claude Code session on refactoring",
            messages=(
                ScenarioMessage(role="user", text="Refactor the storage module", message_id="claude1-u1"),
                ScenarioMessage(role="assistant", text="I will restructure the query layer", message_id="claude1-a1"),
                ScenarioMessage(role="user", text="Also fix the property tests", message_id="claude1-u2"),
            ),
        ),
        ArchiveScenario(
            name="claude-retrieval-2",
            provider="claude-code",
            title="Claude debugging memory leak",
            messages=(
                ScenarioMessage(role="user", text="Memory keeps growing during ingest", message_id="claude2-u1"),
                ScenarioMessage(role="assistant", text="The blob store path has a leak", message_id="claude2-a1"),
            ),
        ),
        ArchiveScenario(
            name="codex-retrieval-1",
            provider="codex",
            title="Codex adding authentication",
            messages=(ScenarioMessage(role="user", text="Add OAuth2 authentication", message_id="codex-u1"),),
        ),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, scenarios)
    return db_path, scenarios


@pytest.fixture()
async def retrieval_surfaces(
    workspace_env: Mapping[str, Path],
    retrieval_archive: tuple[Path, tuple[ArchiveScenario, ...]],
) -> AsyncIterator[ArchiveSurfaceSet]:
    db_path, scenarios = retrieval_archive
    surfaces = build_archive_surface_set(
        db_path=db_path,
        archive_root=workspace_env["archive_root"],
        scenarios=scenarios,
    )
    try:
        yield surfaces
    finally:
        await surfaces.close()


def _provider_cases() -> tuple[ArchiveQueryCase, ...]:
    return (
        ArchiveQueryCase(
            name="provider-chatgpt",
            provider="chatgpt",
            expected_ids=("chatgpt-retrieval-1",),
        ),
        ArchiveQueryCase(
            name="provider-claude",
            provider="claude-code",
            expected_ids=("claude-retrieval-1", "claude-retrieval-2"),
        ),
        ArchiveQueryCase(
            name="provider-codex",
            provider="codex",
            expected_ids=("codex-retrieval-1",),
        ),
    )


def _search_cases() -> tuple[ArchiveQueryCase, ...]:
    return (
        ArchiveQueryCase(
            name="search-property",
            search_text="property",
            expected_ids=("chatgpt-retrieval-1", "claude-retrieval-1"),
        ),
        ArchiveQueryCase(
            name="search-memory",
            search_text="memory",
            expected_ids=("claude-retrieval-2",),
        ),
        ArchiveQueryCase(
            name="search-authentication",
            search_text="authentication",
            expected_ids=("codex-retrieval-1",),
        ),
    )


def test_fts_readiness_rejects_negative_gap_and_missing_triggers() -> None:
    with pytest.raises(DatabaseError):
        check_fts_readiness(
            {
                "exists": True,
                "ready": False,
                "indexed_rows": 110,
                "total_rows": 100,
                "triggers_present": True,
            }
        )

    with pytest.raises(DatabaseError):
        check_fts_readiness(
            {
                "exists": True,
                "ready": False,
                "indexed_rows": 99,
                "total_rows": 100,
                "triggers_present": False,
            }
        )


class TestRetrievalSurfaceAgreement:
    @pytest.mark.asyncio()
    async def test_archive_facts_agree_across_surfaces(self, retrieval_surfaces: ArchiveSurfaceSet) -> None:
        facts = [await surface.archive_facts() for surface in retrieval_surfaces.surfaces]

        assert_archive_surfaces_agree(*facts)
        assert facts[0].total_conversations == 4
        assert facts[0].total_messages == 8
        assert facts[0].provider_counts == {"chatgpt": 1, "claude-code": 2, "codex": 1}

    @pytest.mark.asyncio()
    async def test_provider_filters_partition_the_archive(self, retrieval_surfaces: ArchiveSurfaceSet) -> None:
        all_ids = set((await retrieval_surfaces.surfaces[0].archive_facts()).conversation_ids)
        ids_by_provider: dict[str, tuple[str, ...]] = {}

        for case in _provider_cases():
            assert case.provider is not None
            projected_ids = [await surface.query_ids(case) for surface in retrieval_surfaces.surfaces]
            expected_ids = tuple(sorted(case.expected_ids))
            for surface, ids in zip(retrieval_surfaces.surfaces, projected_ids, strict=True):
                assert ids == expected_ids, f"{surface.name} returned {ids} for {case.name}"
                assert await surface.query_count(case) == len(expected_ids)
            ids_by_provider[case.provider] = expected_ids

        assert_provider_partition_exhaustive(all_conversation_ids=all_ids, ids_by_provider=ids_by_provider)

    @pytest.mark.asyncio()
    async def test_fts_search_is_consistent_across_surfaces(self, retrieval_surfaces: ArchiveSurfaceSet) -> None:
        for case in _search_cases():
            expected_ids = tuple(sorted(case.expected_ids))
            for surface in retrieval_surfaces.surfaces:
                ids = await surface.query_ids(case)
                assert ids == expected_ids, f"{surface.name} returned {ids} for {case.name}"
                assert await surface.query_count(case) == len(expected_ids)


class TestRetrievalIndexInvariants:
    def test_fts_index_contains_exactly_text_messages(
        self,
        retrieval_archive: tuple[Path, tuple[ArchiveScenario, ...]],
    ) -> None:
        from polylogue.storage.sqlite.connection import open_connection

        db_path, _ = retrieval_archive
        with open_connection(db_path) as conn:
            expected_text_messages = {
                str(row["message_id"])
                for row in conn.execute("SELECT message_id FROM messages WHERE text IS NOT NULL").fetchall()
            }
            indexed_messages = {
                str(row["message_id"]) for row in conn.execute("SELECT message_id FROM messages_fts").fetchall()
            }

        assert indexed_messages == expected_text_messages

    @pytest.mark.asyncio()
    async def test_repository_message_counts_match_storage_facts(
        self,
        retrieval_archive: tuple[Path, tuple[ArchiveScenario, ...]],
    ) -> None:
        from polylogue.storage.sqlite.connection import open_connection
        from tests.infra.archive_scenarios import repository_for_scenario_db

        db_path, scenarios = retrieval_archive
        expected_counts: dict[str, int] = {}
        with open_connection(db_path) as conn:
            for scenario in scenarios:
                facts = scenario.facts_from_connection(conn)
                expected_counts[facts.conversation_id] = facts.message_count

        repository = repository_for_scenario_db(db_path)
        try:
            observed_counts = await repository.get_message_counts_batch(list(expected_counts))
        finally:
            await repository.close()

        assert observed_counts == expected_counts
