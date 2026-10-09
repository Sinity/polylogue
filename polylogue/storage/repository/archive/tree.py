"""Canonical session-topology reads for the repository.

Parent, child, root and rooted-tree answers are projected from
``session_links`` through the one graph engine
(``derive_session_topology_async``). The ``sessions.parent_session_id`` /
``sessions.root_session_id`` columns are write-side accelerators: they may
speed a lookup, but they never synthesize an edge here and never stand in for
an edge's provenance (composability, status, inheritance, method, evidence).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.session.domain_models import Session
from polylogue.storage.runtime import SessionRecord

if TYPE_CHECKING:
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class RepositoryArchiveTreeMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

        async def get(self, session_id: str) -> Session | None: ...

        async def _hydrate_sessions(
            self,
            session_records: list[SessionRecord],
            *,
            ordered_ids: list[str] | None = None,
            queries: SQLiteQueryStore,
        ) -> list[Session]: ...

    async def get_session_tree(self, session_id: str) -> list[Session]:
        async with self.queries.read_snapshot() as queries:
            topology = await queries.get_session_topology(session_id)
            if topology is None:
                return []
            records: list[SessionRecord] = []
            for node in topology.nodes:
                record = await queries.get_session(str(node.session_id))
                if record is not None:
                    records.append(record)
            return await self._hydrate_sessions(
                records,
                ordered_ids=[record.session_id for record in records],
                queries=queries,
            )


__all__ = ["RepositoryArchiveTreeMixin"]
