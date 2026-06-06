"""Stateful archive lifecycle harnesses for verification tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from polylogue.api import Polylogue
from polylogue.core.json import JSONValue
from tests.infra.archive_scenarios import ArchiveScenario, archive_for_scenario_db, open_index_db
from tests.infra.oracles import assert_archive_surfaces_agree, assert_session_surfaces_agree
from tests.infra.semantic_facts import ArchiveFacts
from tests.infra.surfaces import ArchiveSurfaceSet, build_archive_surface_set


@dataclass(slots=True)
class RepositoryLifecycleHarness:
    """Model archive state transitions and assert cross-surface invariants.

    Mutations route through the ``Polylogue`` facade over ``index.db``; the
    old ``SessionRepository`` substrate is not part of this harness.
    """

    db_path: Path
    repository: Polylogue
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
            repository=archive_for_scenario_db(db_path),
            surfaces=build_archive_surface_set(
                db_path=db_path,
                archive_root=archive_root,
                scenarios=scenarios,
            ),
        )

    async def close(self) -> None:
        await self.repository.close()
        await self.surfaces.close()

    async def assert_session_visible(self, scenario: ArchiveScenario) -> None:
        facts = [await surface.session_facts(scenario) for surface in self.surfaces.surfaces]
        assert_session_surfaces_agree(*facts)

    async def assert_archive_agrees(self, *, total_sessions: int) -> ArchiveFacts:
        facts = [await surface.archive_facts() for surface in self.surfaces.surfaces]
        assert_archive_surfaces_agree(*facts)
        assert facts[0].total_sessions == total_sessions
        return facts[0]

    async def assert_metadata_contains(
        self,
        session_id: str,
        expected: Mapping[str, JSONValue],
    ) -> dict[str, str]:
        metadata = await self.repository.get_metadata(session_id)
        for key, value in expected.items():
            assert metadata.get(key) == value
        return metadata

    async def assert_tags(self, expected: dict[str, int], *, origin: str | None = None) -> None:
        actual = await self.repository.list_tags(origin=origin)
        assert actual == expected, f"expected={expected!r} actual={actual!r}"

    async def add_tag_and_assert_visible(self, scenario: ArchiveScenario, tag: str) -> None:
        await self.repository.add_tag(scenario.native_session_id, tag)
        await self.assert_session_visible(scenario)

    async def remove_tag_and_assert_visible(self, scenario: ArchiveScenario, tag: str) -> None:
        await self.repository.remove_tag(scenario.native_session_id, tag)
        await self.assert_session_visible(scenario)

    async def update_metadata_and_assert_visible(
        self,
        scenario: ArchiveScenario,
        key: str,
        value: JSONValue,
    ) -> None:
        await self.repository.update_metadata(scenario.native_session_id, key, str(value))
        await self.assert_session_visible(scenario)

    async def delete_metadata_and_assert_visible(self, scenario: ArchiveScenario, key: str) -> None:
        await self.repository.delete_metadata(scenario.native_session_id, key)
        await self.assert_session_visible(scenario)

    async def delete_session_and_assert_absent(self, scenario: ArchiveScenario) -> None:
        session_id = scenario.native_session_id
        assert await self.repository.delete_session(session_id) is True
        assert await self.repository.delete_session(session_id) is False
        assert await self.repository.get_session(session_id) is None
        self.assert_no_dangling_runtime_rows(session_id)

    def assert_no_dangling_runtime_rows(self, session_id: str) -> None:
        """Assert the archive session row and its child rows are all gone."""
        table_names = ("messages", "blocks", "attachment_refs")
        with open_index_db(self.db_path) as conn:
            assert (
                conn.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                is None
            )
            for table_name in table_names:
                count = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {table_name} WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()[0]
                )
                assert count == 0, f"{table_name} retained rows for deleted session {session_id!r}"


__all__ = ["RepositoryLifecycleHarness"]
