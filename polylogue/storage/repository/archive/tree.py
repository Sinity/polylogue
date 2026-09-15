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

from polylogue.analysis.topology import SessionTopology
from polylogue.archive.session.domain_models import Session
from polylogue.storage.derived.topology.derivation import derive_session_topology_async
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
        ) -> list[Session]: ...

    async def _topology(self, session_id: str) -> SessionTopology | None:
        return await derive_session_topology_async(self.queries, session_id)

    async def get_parent(self, session_id: str) -> Session | None:
        topology = await self._topology(session_id)
        if topology is None:
            return None
        for edge in topology.edges:
            if edge.composable and str(edge.child_id) == session_id and edge.parent_id is not None:
                return await self.get(str(edge.parent_id))
        return None

    async def get_children(self, session_id: str) -> list[Session]:
        topology = await self._topology(session_id)
        if topology is None:
            return []
        child_ids = sorted(
            {
                str(edge.child_id)
                for edge in topology.edges
                if edge.composable and edge.parent_id is not None and str(edge.parent_id) == session_id
            }
        )
        children: list[Session] = []
        for child_id in child_ids:
            child = await self.get(child_id)
            if child is not None:
                children.append(child)
        return children

    async def get_root(self, session_id: str) -> Session:
        topology = await self._topology(session_id)
        if topology is None:
            raise ValueError(f"Session {session_id} not found")
        root = await self.get(str(topology.root_id))
        if root is None:
            raise ValueError(f"Session {session_id} not found")
        return root

    async def get_session_tree(self, session_id: str) -> list[Session]:
        topology = await self._topology(session_id)
        if topology is None:
            return []
        records: list[SessionRecord] = []
        for node in topology.nodes:
            record = await self.queries.get_session(str(node.session_id))
            if record is not None:
                records.append(record)
        return await self._hydrate_sessions(
            records,
            ordered_ids=[record.session_id for record in records],
        )


__all__ = ["RepositoryArchiveTreeMixin"]
