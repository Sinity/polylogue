"""Stateful archive lifecycle harnesses for verification tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.archive_scenarios import ArchiveScenario, repository_for_scenario_db
from tests.infra.oracles import assert_archive_surfaces_agree, assert_conversation_surfaces_agree
from tests.infra.semantic_facts import ArchiveFacts
from tests.infra.surfaces import ArchiveSurfaceSet, build_archive_surface_set


@dataclass(slots=True)
class RepositoryLifecycleHarness:
    """Model repository state transitions and assert archive invariants."""

    db_path: Path
    repository: ConversationRepository
    surfaces: ArchiveSurfaceSet

    @classmethod
    def create(
        cls,
        *,
        db_path: Path,
        archive_root: Path,
        scenarios: tuple[ArchiveScenario, ...],
    ) -> RepositoryLifecycleHarness:
        return cls(
            db_path=db_path,
            repository=repository_for_scenario_db(db_path),
            surfaces=build_archive_surface_set(
                db_path=db_path,
                archive_root=archive_root,
                scenarios=scenarios,
            ),
        )

    async def close(self) -> None:
        await self.repository.close()
        await self.surfaces.close()

    async def assert_conversation_visible(self, scenario: ArchiveScenario) -> None:
        facts = [await surface.conversation_facts(scenario) for surface in self.surfaces.surfaces]
        assert_conversation_surfaces_agree(*facts)

    async def assert_archive_agrees(self, *, total_conversations: int) -> ArchiveFacts:
        facts = [await surface.archive_facts() for surface in self.surfaces.surfaces]
        assert_archive_surfaces_agree(*facts)
        assert facts[0].total_conversations == total_conversations
        return facts[0]

    async def assert_metadata_contains(
        self,
        conversation_id: str,
        expected: Mapping[str, JSONValue],
    ) -> JSONDocument:
        metadata = await self.repository.get_metadata(conversation_id)
        for key, value in expected.items():
            assert metadata.get(key) == value
        return metadata

    async def assert_tags(self, expected: dict[str, int], *, provider: str | None = None) -> None:
        assert await self.repository.list_tags(provider=provider) == expected

    async def add_tag_and_assert_visible(self, scenario: ArchiveScenario, tag: str) -> None:
        await self.repository.add_tag(scenario.resolved_conversation_id, tag)
        await self.assert_conversation_visible(scenario)

    async def remove_tag_and_assert_visible(self, scenario: ArchiveScenario, tag: str) -> None:
        await self.repository.remove_tag(scenario.resolved_conversation_id, tag)
        await self.assert_conversation_visible(scenario)

    async def update_metadata_and_assert_visible(
        self,
        scenario: ArchiveScenario,
        key: str,
        value: JSONValue,
    ) -> None:
        await self.repository.update_metadata(scenario.resolved_conversation_id, key, value)
        await self.assert_conversation_visible(scenario)

    async def delete_metadata_and_assert_visible(self, scenario: ArchiveScenario, key: str) -> None:
        await self.repository.delete_metadata(scenario.resolved_conversation_id, key)
        await self.assert_conversation_visible(scenario)

    async def delete_conversation_and_assert_absent(self, scenario: ArchiveScenario) -> None:
        conversation_id = scenario.resolved_conversation_id
        assert await self.repository.delete_conversation(conversation_id) is True
        assert await self.repository.delete_conversation(conversation_id) is False
        assert await self.repository.get(conversation_id) is None
        self.assert_no_dangling_runtime_rows(conversation_id)

    def assert_no_dangling_runtime_rows(self, conversation_id: str) -> None:
        table_names = ("messages", "content_blocks", "attachment_refs", "action_events")
        with open_connection(self.db_path) as conn:
            assert (
                conn.execute(
                    "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                is None
            )
            for table_name in table_names:
                count = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {table_name} WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()[0]
                )
                assert count == 0, f"{table_name} retained rows for deleted conversation {conversation_id!r}"


__all__ = ["RepositoryLifecycleHarness"]
