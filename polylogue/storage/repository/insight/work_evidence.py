"""Repository route for provider-neutral work topology and claims."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.insights.work_evidence import WorkEvidenceGraph, WorkEvidenceTraversal
from polylogue.storage.query_models import WorkEvidenceTraversalQuery
from polylogue.storage.sqlite.queries import work_evidence as work_evidence_q

if TYPE_CHECKING:
    from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol


class RepositoryWorkEvidenceMixin:
    if TYPE_CHECKING:
        _backend: RepositoryBackendProtocol

    async def replace_work_evidence_graph(self, graph: WorkEvidenceGraph) -> None:
        """Store one graph snapshot for generic work-evidence traversal."""

        async with self._backend.connection() as conn:
            await work_evidence_q.replace_work_evidence_graph(conn, graph, self._backend.transaction_depth)

    async def traverse_work_evidence(
        self,
        *,
        graph_id: str,
        focal_ref: str,
        direction: str = "both",
        edge_kinds: tuple[str, ...] = (),
        limit: int | None = 100,
    ) -> WorkEvidenceTraversal | None:
        return await self._backend.queries.get_work_evidence_traversal(
            WorkEvidenceTraversalQuery(
                graph_id=graph_id,
                focal_ref=focal_ref,
                direction=direction,  # type: ignore[arg-type]
                edge_kinds=edge_kinds,
                limit=limit,
            )
        )


__all__ = ["RepositoryWorkEvidenceMixin"]
