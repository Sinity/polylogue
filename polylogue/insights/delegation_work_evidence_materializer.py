"""Materialize the canonical delegation view into the generic work graph."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from polylogue.archive.query.predicate import QueryBoolPredicate
from polylogue.core.refs import ObjectRef
from polylogue.insights.delegation_work_evidence import materialize_delegation_work_evidence_graph
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

DELEGATION_WORK_EVIDENCE_GRAPH_ID = "delegation:archive"


def delegation_work_evidence_snapshot(archive_root: Path) -> ObjectRef:
    """Return a content-derived snapshot for the current delegation view."""

    index_db = Path(archive_root) / "index.db"
    with sqlite3.connect(index_db) as conn:
        rows = conn.execute("SELECT * FROM delegations ORDER BY parent_session_id, child_session_id").fetchall()
    payload = json.dumps(rows, separators=(",", ":"), default=str).encode()
    return ObjectRef(kind="context-snapshot", object_id=f"delegations:{hashlib.sha256(payload).hexdigest()[:24]}")


def materialize_delegation_work_evidence_archive(archive_root: Path) -> int:
    """Replace the archive delegation projection and return its row count."""

    archive_root = Path(archive_root)
    snapshot = delegation_work_evidence_snapshot(archive_root)
    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        rows = archive.query_delegations(QueryBoolPredicate("and", ()), limit=100_001)
    if len(rows) > 100_000:
        raise ValueError("delegation work-evidence materialization exceeded its bounded population")
    graph = materialize_delegation_work_evidence_graph(
        graph_id=DELEGATION_WORK_EVIDENCE_GRAPH_ID,
        corpus_snapshot_ref=snapshot,
        rows=rows,
    )
    _replace_graph(archive_root / "index.db", graph)
    return len(rows)


def delegation_work_evidence_materialization_needed(archive_root: Path) -> bool:
    """Return whether the stored delegation graph represents current evidence."""

    index_db = Path(archive_root) / "index.db"
    snapshot = delegation_work_evidence_snapshot(archive_root).format()
    with sqlite3.connect(index_db) as conn:
        row = conn.execute(
            "SELECT corpus_snapshot_ref FROM work_evidence_graphs WHERE graph_id = ?",
            (DELEGATION_WORK_EVIDENCE_GRAPH_ID,),
        ).fetchone()
    return row is None or str(row[0]) != snapshot


def _replace_graph(index_db: Path, graph: object) -> None:
    # Keep this synchronous: convergence stages own a synchronous SQLite lease.
    from polylogue.insights.work_evidence import WorkEvidenceGraph

    if not isinstance(graph, WorkEvidenceGraph):
        raise TypeError("expected WorkEvidenceGraph")
    with sqlite3.connect(index_db) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM work_evidence_graphs WHERE graph_id = ?", (graph.graph_id,))
        conn.execute(
            "INSERT INTO work_evidence_graphs(graph_id, corpus_snapshot_ref) VALUES (?, ?)",
            (graph.graph_id, graph.corpus_snapshot_ref.format()),
        )
        conn.executemany(
            """
            INSERT INTO work_evidence_nodes(
                graph_id, node_ref, node_kind, label, evidence_refs_json, corpus_snapshot_ref,
                authority, confidence, occurred_at_ms, actor_ref, execution_context_id,
                execution_context_known_json, execution_context_unknown_json, role,
                execution_context_addressed, association_state, claim_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    graph.graph_id,
                    node.ref.format(),
                    node.kind,
                    node.label,
                    json.dumps([ref.format() for ref in node.evidence_refs]),
                    node.corpus_snapshot_ref.format(),
                    node.authority,
                    node.confidence,
                    node.occurred_at_ms,
                    node.actor_ref.format() if node.actor_ref else None,
                    node.execution_context_ref.context_id if node.execution_context_ref else None,
                    json.dumps(list(node.execution_context_ref.known_fields)) if node.execution_context_ref else "[]",
                    json.dumps(list(node.execution_context_ref.unknown_fields)) if node.execution_context_ref else "[]",
                    node.role,
                    int(node.execution_context_ref.content_addressed) if node.execution_context_ref else None,
                    node.association_state,
                    node.claim_text,
                )
                for node in graph.nodes
            ],
        )
        conn.executemany(
            """
            INSERT INTO work_evidence_edges(
                graph_id, edge_ref, edge_kind, source_ref, target_ref, evidence_refs_json,
                corpus_snapshot_ref, authority, confidence, occurred_at_ms, association_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    graph.graph_id,
                    edge.ref.format(),
                    edge.kind,
                    edge.source_ref.format(),
                    edge.target_ref.format(),
                    json.dumps([ref.format() for ref in edge.evidence_refs]),
                    edge.corpus_snapshot_ref.format(),
                    edge.authority,
                    edge.confidence,
                    edge.occurred_at_ms,
                    edge.association_state,
                )
                for edge in graph.edges
            ],
        )


__all__ = [
    "DELEGATION_WORK_EVIDENCE_GRAPH_ID",
    "delegation_work_evidence_materialization_needed",
    "delegation_work_evidence_snapshot",
    "materialize_delegation_work_evidence_archive",
]
