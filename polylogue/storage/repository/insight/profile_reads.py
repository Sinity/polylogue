"""Hydrated session-profile durable insight reads for the repository.

The insight *read* layer consumed by CLI/MCP/API surfaces is the SQL layer in
``storage/sqlite/archive_tiers/archive.py`` (wired through
``analysis/registry.py``). What survives here are the repository-level
profile reads that still have a caller: ``get_session_profiles_batch`` (used by
``api/archive.py`` to hydrate analyzed-session batches) and
``get_session_profile_record`` (the repository-route probe in
``tests/infra/surfaces.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.session.session_profile import SessionProfile
from polylogue.storage.derived.insight_read_support import hydrate_mapping
from polylogue.storage.derived.session.profiles import hydrate_session_profile
from polylogue.storage.runtime import SessionProfileRecord

if TYPE_CHECKING:
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class RepositoryInsightProfileReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def get_session_profile_record(self, session_id: str) -> SessionProfileRecord | None:
        return await self.queries.get_session_profile(session_id)

    async def get_session_profile_records_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, SessionProfileRecord]:
        return await self.queries.get_session_profiles_batch(session_ids)

    async def get_session_profiles_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, SessionProfile]:
        records = await self.get_session_profile_records_batch(session_ids)
        return hydrate_mapping(records, hydrate_session_profile)


__all__ = ["RepositoryInsightProfileReadMixin"]
