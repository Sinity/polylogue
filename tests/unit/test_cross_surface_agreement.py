"""Cross-surface agreement: prove repository, hydration, and records agree.

When the same conversation is viewed through different surfaces, the
semantic facts must be identical. A failure here means two surfaces
disagree about what's in the archive.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.infra.archive_scenarios import (
    ArchiveScenario,
    ScenarioMessage,
    repository_for_scenario_db,
    seed_workspace_scenarios,
)
from tests.infra.oracles import (
    assert_archive_surfaces_agree,
    assert_conversation_surfaces_agree,
    assert_provider_partition_exhaustive,
)
from tests.infra.semantic_facts import ArchiveFacts, ConversationFacts


@pytest.fixture()
def multi_provider_archive(workspace_env: Mapping[str, Path]) -> tuple[Path, list[ArchiveScenario]]:
    """Populate a DB with conversations across providers."""
    scenarios = [
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
    ]
    db_path, _ = seed_workspace_scenarios(workspace_env, scenarios)
    return db_path, scenarios


# ---------------------------------------------------------------------------
# Record-level vs hydration-level facts agree
# ---------------------------------------------------------------------------


class TestRecordVsHydrationAgreement:
    def test_facts_agree_for_each_conversation(
        self, multi_provider_archive: tuple[Path, list[ArchiveScenario]]
    ) -> None:
        from polylogue.storage.backends.connection import open_connection

        db_path, scenarios = multi_provider_archive
        with open_connection(db_path) as conn:
            for scenario in scenarios:
                assert_conversation_surfaces_agree(
                    scenario.facts_from_connection(conn),
                    scenario.hydrated_facts_from_connection(conn),
                )

    @pytest.mark.asyncio()
    async def test_repository_facts_agree_with_storage(
        self,
        multi_provider_archive: tuple[Path, list[ArchiveScenario]],
    ) -> None:
        from polylogue.storage.backends.connection import open_connection

        db_path, scenarios = multi_provider_archive
        repository = repository_for_scenario_db(db_path)
        try:
            with open_connection(db_path) as conn:
                for scenario in scenarios:
                    assert_conversation_surfaces_agree(
                        scenario.facts_from_connection(conn),
                        await scenario.facts_from_repository(repository),
                    )
        finally:
            await repository.close()


# ---------------------------------------------------------------------------
# Archive-level facts: direct SQL vs count queries agree
# ---------------------------------------------------------------------------


class TestArchiveFactsConsistency:
    @pytest.mark.asyncio()
    async def test_archive_facts_internally_consistent(
        self,
        multi_provider_archive: tuple[Path, list[ArchiveScenario]],
    ) -> None:
        from polylogue.storage.backends.connection import open_connection

        db_path, _ = multi_provider_archive
        repository = repository_for_scenario_db(db_path)
        try:
            with open_connection(db_path) as conn:
                db_facts = ArchiveFacts.from_db_connection(conn)
            repository_facts = ArchiveFacts.from_conversations(await repository.list(limit=None))
        finally:
            await repository.close()

        assert_archive_surfaces_agree(db_facts, repository_facts)
        assert db_facts.total_conversations == 3
        assert sum(db_facts.provider_counts.values()) == db_facts.total_conversations
        assert db_facts.total_messages > 0
        assert db_facts.provider_counts.get("chatgpt") == 1
        assert db_facts.provider_counts.get("claude-code") == 1
        assert db_facts.provider_counts.get("codex") == 1

    def test_provider_partition_exhaustive(self, multi_provider_archive: tuple[Path, list[ArchiveScenario]]) -> None:
        """Every conversation belongs to exactly one provider in the facts."""
        from polylogue.storage.backends.connection import open_connection

        db_path, _ = multi_provider_archive
        with open_connection(db_path) as conn:
            facts = ArchiveFacts.from_db_connection(conn)

            all_ids = {
                r["conversation_id"] for r in conn.execute("SELECT conversation_id FROM conversations").fetchall()
            }
            ids_by_provider: dict[str, set[str]] = {}
            for provider in facts.provider_counts:
                ids_by_provider[provider] = {
                    r["conversation_id"]
                    for r in conn.execute(
                        "SELECT conversation_id FROM conversations WHERE provider_name = ?", (provider,)
                    ).fetchall()
                }

            assert_provider_partition_exhaustive(all_conversation_ids=all_ids, ids_by_provider=ids_by_provider)


# ---------------------------------------------------------------------------
# Synthetic roundtrip: generated payload facts match hydrated facts
# ---------------------------------------------------------------------------


class TestSyntheticRoundtripFactAgreement:
    @pytest.mark.parametrize("provider_name", ["chatgpt", "claude-code", "claude-ai", "codex", "gemini"])
    def test_parsed_vs_hydrated_facts_agree(
        self,
        provider_name: str,
        workspace_env: Mapping[str, Path],
    ) -> None:
        from polylogue.schemas.synthetic.core import SyntheticCorpus
        from polylogue.storage.backends.connection import open_connection
        from tests.infra.pipeline_roundtrip import parse_and_transform_payload, save_transform_and_hydrate
        from tests.infra.storage_records import db_setup

        corpus = SyntheticCorpus.for_provider(provider_name)
        raw_bytes = corpus.generate(count=1, seed=99)[0]

        roundtrip = parse_and_transform_payload(
            provider_name,
            raw_bytes,
            workspace_env["archive_root"],
            f"xsurf-{provider_name}",
        )

        db_path = db_setup(workspace_env)
        with open_connection(db_path) as conn:
            hydrated = save_transform_and_hydrate(roundtrip.transform, conn)
            hydrated_facts = ConversationFacts.from_domain_conversation(hydrated)

            assert hydrated_facts.message_count == len(roundtrip.parsed.messages)
            assert hydrated_facts.provider == str(roundtrip.parsed.provider_name)
            assert hydrated_facts.title == roundtrip.parsed.title
