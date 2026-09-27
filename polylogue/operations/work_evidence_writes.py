"""The one write route for a stored work-evidence graph.

Work-evidence graphs live in ``index.db``. The two CLI commands that build
them -- incident-evidence materialization and work-effect reconciliation --
used to persist them from the CLI process through a ``SessionRepository``
opened there, which made each ``--yes`` a second archive writer beside the
resident daemon (polylogue-5vps8 / polylogue-re6s3). Building a graph is a
read; persisting one is not. The CLI now computes the graph and submits it
to the declared ``mutation.work_evidence.graph.replace`` operation, and the
daemon handler below is the only code that replaces a stored graph.

A replace carries an optional ``expected_base_digest``. Reconciliation is a
read-modify-write: it loads a graph, derives a new one from it, and replaces
it. Without the check, a graph replaced between that read and this write
would be silently overwritten with a result derived from content that no
longer exists. With it, the write is refused with a typed conflict and the
operator re-runs against current content. Materialization builds a graph
from sessions rather than from a prior graph, so it replaces unconditionally.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from http import HTTPStatus

from polylogue.analysis.work_evidence import WorkEvidenceGraph
from polylogue.core.errors import PolylogueError
from polylogue.storage.repository import SessionRepository

#: Digest of "no graph is stored under this id".
ABSENT_WORK_EVIDENCE_GRAPH_DIGEST = "absent"


class WorkEvidenceGraphConflictError(PolylogueError, ValueError):
    """The stored graph changed after the caller derived its replacement.

    A ``ValueError`` with a declared ``code``, so the daemon rejects the
    operation with that code rather than failing it as an internal error.
    """

    code = "work_evidence_graph_conflict"
    http_status_code = HTTPStatus.CONFLICT

    def __init__(self, graph_id: str, *, expected: str, current: str) -> None:
        super().__init__(
            f"work-evidence graph {graph_id!r} changed after it was read "
            f"(expected {expected}, found {current}); re-run against the current graph"
        )
        self.graph_id = graph_id
        self.expected = expected
        self.current = current


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def work_evidence_graph_digest(graph: WorkEvidenceGraph | None) -> str:
    """Return an order-independent content digest of one graph, or the absent digest."""
    if graph is None:
        return ABSENT_WORK_EVIDENCE_GRAPH_DIGEST
    # Storage does not keep node/edge order, so the digest must not see it:
    # a graph and its own read-back are the same content.
    document = graph.model_dump(mode="json")
    for member in ("nodes", "edges"):
        document[member] = sorted(_canonical(item) for item in document[member])
    canonical = _canonical(document)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class WorkEvidenceGraphReplacement:
    """What one replace changed."""

    graph_id: str
    previous_digest: str
    digest: str
    node_count: int
    edge_count: int

    @property
    def changed(self) -> bool:
        return self.previous_digest != self.digest

    def to_dict(self) -> dict[str, object]:
        return {
            "graph_id": self.graph_id,
            "previous_digest": self.previous_digest,
            "digest": self.digest,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "changed": self.changed,
        }


async def replace_work_evidence_graph_checked(
    repository: SessionRepository,
    graph: WorkEvidenceGraph,
    *,
    expected_base_digest: str | None,
) -> WorkEvidenceGraphReplacement:
    """Replace one stored graph, refusing when its base moved.

    ``expected_base_digest`` of ``None`` replaces unconditionally. The read
    and the replace run inside the daemon's single writer, so no other archive
    write can land between them.
    """
    current = await repository.get_work_evidence_graph(graph.graph_id)
    previous_digest = work_evidence_graph_digest(current)
    if expected_base_digest is not None and expected_base_digest != previous_digest:
        raise WorkEvidenceGraphConflictError(graph.graph_id, expected=expected_base_digest, current=previous_digest)
    digest = work_evidence_graph_digest(graph)
    if digest != previous_digest:
        await repository.replace_work_evidence_graph(graph)
    return WorkEvidenceGraphReplacement(
        graph_id=graph.graph_id,
        previous_digest=previous_digest,
        digest=digest,
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
    )


__all__ = [
    "ABSENT_WORK_EVIDENCE_GRAPH_DIGEST",
    "WorkEvidenceGraphConflictError",
    "WorkEvidenceGraphReplacement",
    "replace_work_evidence_graph_checked",
    "work_evidence_graph_digest",
]
