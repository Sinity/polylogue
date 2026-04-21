"""Archive surface adapters for scenario-driven verification tests."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from polylogue.facade import Polylogue
from polylogue.storage.backends.connection import open_connection
from tests.infra.archive_scenarios import ArchiveScenario, repository_for_scenario_db
from tests.infra.query_cases import ArchiveQueryCase
from tests.infra.semantic_facts import ArchiveFacts, ConversationFacts


def _sorted_unique(values: list[str] | tuple[str, ...] | set[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _provider_ids_from_connection(conn: sqlite3.Connection, provider: str) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT conversation_id FROM conversations WHERE provider_name = ? ORDER BY conversation_id",
        (provider,),
    ).fetchall()
    return tuple(str(row["conversation_id"]) for row in rows)


def _search_ids_from_connection(conn: sqlite3.Connection, search_text: str) -> tuple[str, ...]:
    rows = conn.execute(
        """
        SELECT DISTINCT conversation_id
        FROM messages_fts
        WHERE messages_fts MATCH ?
        ORDER BY conversation_id
        """,
        (search_text,),
    ).fetchall()
    return tuple(str(row["conversation_id"]) for row in rows)


class ArchiveSurfaceAdapter(Protocol):
    """Common semantic projection surface over a seeded archive."""

    @property
    def name(self) -> str: ...

    async def conversation_facts(self, scenario: ArchiveScenario) -> ConversationFacts: ...

    async def archive_facts(self) -> ArchiveFacts: ...

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]: ...

    async def query_count(self, query_case: ArchiveQueryCase) -> int: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SQLiteRecordSurface:
    """Direct SQL/storage-record projection."""

    db_path: Path
    name: str = "sqlite-records"

    async def conversation_facts(self, scenario: ArchiveScenario) -> ConversationFacts:
        with open_connection(self.db_path) as conn:
            return scenario.facts_from_connection(conn)

    async def archive_facts(self) -> ArchiveFacts:
        with open_connection(self.db_path) as conn:
            return ArchiveFacts.from_db_connection(conn)

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        with open_connection(self.db_path) as conn:
            if query_case.provider is not None:
                return _provider_ids_from_connection(conn, query_case.provider)
            if query_case.search_text is not None:
                return _search_ids_from_connection(conn, query_case.search_text)
        raise ValueError(f"Query case {query_case.name!r} does not define a supported projection")

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class SQLiteHydratedSurface:
    """Direct SQL projection after storage-record hydration."""

    db_path: Path
    scenarios: tuple[ArchiveScenario, ...]
    name: str = "sqlite-hydrated"

    async def conversation_facts(self, scenario: ArchiveScenario) -> ConversationFacts:
        with open_connection(self.db_path) as conn:
            return scenario.hydrated_facts_from_connection(conn)

    async def archive_facts(self) -> ArchiveFacts:
        with open_connection(self.db_path) as conn:
            conversations = [
                scenario.hydrated_facts_from_connection(conn)
                for scenario in sorted(self.scenarios, key=lambda item: item.resolved_conversation_id)
            ]
        return ArchiveFacts(
            total_conversations=len(conversations),
            provider_counts=_provider_counts(conversations),
            total_messages=sum(facts.message_count for facts in conversations),
            conversation_ids=tuple(facts.conversation_id for facts in conversations),
        )

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        with open_connection(self.db_path) as conn:
            if query_case.provider is not None:
                return _provider_ids_from_connection(conn, query_case.provider)
            if query_case.search_text is not None:
                return _search_ids_from_connection(conn, query_case.search_text)
        raise ValueError(f"Query case {query_case.name!r} does not define a supported projection")

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        return None


class RepositorySurface:
    """Async repository projection."""

    name = "repository"

    def __init__(self, db_path: Path) -> None:
        self._repository = repository_for_scenario_db(db_path)

    async def conversation_facts(self, scenario: ArchiveScenario) -> ConversationFacts:
        return await scenario.facts_from_repository(self._repository)

    async def archive_facts(self) -> ArchiveFacts:
        return ArchiveFacts.from_conversations(await self._repository.list(limit=None))

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        if query_case.provider is not None:
            conversations = await self._repository.list(provider=query_case.provider, limit=None)
        elif query_case.search_text is not None:
            conversations = await self._repository.search(query_case.search_text, limit=100)
        else:
            raise ValueError(f"Query case {query_case.name!r} does not define a supported projection")
        return _sorted_unique([str(conversation.id) for conversation in conversations])

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        if query_case.provider is not None:
            return await self._repository.count(provider=query_case.provider)
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        await self._repository.close()


class FacadeSurface:
    """Public async facade projection."""

    name = "facade"

    def __init__(self, *, archive_root: Path, db_path: Path) -> None:
        self._archive = Polylogue(archive_root=archive_root, db_path=db_path)

    async def conversation_facts(self, scenario: ArchiveScenario) -> ConversationFacts:
        conversation = await self._archive.get_conversation(scenario.resolved_conversation_id)
        if conversation is None:
            raise AssertionError(f"Scenario {scenario.resolved_conversation_id!r} missing through facade")
        return ConversationFacts.from_domain_conversation(conversation)

    async def archive_facts(self) -> ArchiveFacts:
        return ArchiveFacts.from_conversations(await self._archive.list_conversations(limit=None))

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        if query_case.provider is not None:
            conversations = await self._archive.list_conversations(provider=query_case.provider, limit=None)
            return _sorted_unique([str(conversation.id) for conversation in conversations])
        if query_case.search_text is not None:
            result = await self._archive.search(query_case.search_text, limit=100)
            return _sorted_unique([hit.conversation_id for hit in result.hits])
        raise ValueError(f"Query case {query_case.name!r} does not define a supported projection")

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        await self._archive.close()


@dataclass(slots=True)
class ArchiveSurfaceSet:
    """Closable group of archive surface adapters."""

    surfaces: tuple[ArchiveSurfaceAdapter, ...]

    async def close(self) -> None:
        for surface in self.surfaces:
            await surface.close()


def build_archive_surface_set(
    *,
    db_path: Path,
    archive_root: Path,
    scenarios: tuple[ArchiveScenario, ...],
) -> ArchiveSurfaceSet:
    """Build the standard cross-surface set for a scenario archive."""
    surfaces: tuple[ArchiveSurfaceAdapter, ...] = (
        SQLiteRecordSurface(db_path),
        SQLiteHydratedSurface(db_path, scenarios),
        RepositorySurface(db_path),
        FacadeSurface(archive_root=archive_root, db_path=db_path),
    )
    return ArchiveSurfaceSet(
        surfaces=surfaces,
    )


def _provider_counts(conversations: list[ConversationFacts]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for conversation in conversations:
        counts[conversation.provider] = counts.get(conversation.provider, 0) + 1
    return counts


__all__ = [
    "ArchiveSurfaceAdapter",
    "ArchiveSurfaceSet",
    "FacadeSurface",
    "RepositorySurface",
    "SQLiteHydratedSurface",
    "SQLiteRecordSurface",
    "build_archive_surface_set",
]
