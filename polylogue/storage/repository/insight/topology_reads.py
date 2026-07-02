"""Session-topology read mixin for the repository facade.

Exposes the lineage graph derived in
:mod:`polylogue.storage.insights.topology` through a single async entry
point. Surface adapters (MCP ``get_session_tree``, future reader panes,
context image/bundle projections) compose this rather than re-walking parent
pointers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.insights.topology import SessionTopology
from polylogue.storage.insights.topology import derive_session_topology_async

if TYPE_CHECKING:
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class RepositoryInsightTopologyReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def get_session_topology(self, session_id: str) -> SessionTopology | None:
        """Return the resolved lineage graph rooted at ``session_id``.

        Returns ``None`` when the session is not present in the
        archive. Cycles and unresolved native parent edges are surfaced
        through the returned :class:`SessionTopology`.
        """

        return await derive_session_topology_async(self.queries, session_id)


__all__ = ["RepositoryInsightTopologyReadMixin"]
