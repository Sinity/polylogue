"""Materialize the canonical delegation facts into the generic work graph."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from polylogue.analysis.delegation_work_evidence import materialize_delegation_work_evidence_graph
from polylogue.archive.query.predicate import QueryBoolPredicate
from polylogue.core.refs import ObjectRef
from polylogue.core.stage_admission import admit_stage_write
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.sqlite.managed_connection import sqlite_connection

DELEGATION_WORK_EVIDENCE_GRAPH_ID = "delegation:archive"


#: Row ceiling for the delegation snapshot. Matches the bound
#: :func:`materialize_delegation_work_evidence_archive` already enforces on
#: ``query_delegations``; enforcing it in the snapshot means the *freshness
#: probe* is bounded too, not only the materialize path.
MAX_DELEGATION_SNAPSHOT_ROWS = 100_000

#: Byte ceiling for the digested payload. A hostile export can keep the row
#: count small while making ``instruction_payload``/``artifact_text``
#: arbitrarily large, so a row count alone is not a bound.
MAX_DELEGATION_SNAPSHOT_BYTES = 64 * 1024 * 1024

_SNAPSHOT_ROW_SEPARATOR = b"\x1e"


def delegation_work_evidence_snapshot(archive_root: Path) -> ObjectRef:
    """Return a content-derived snapshot for the current delegation facts.

    Digested incrementally: the cursor is iterated row by row and each row is
    folded into one SHA-256, so peak memory is one row rather than the whole
    view plus a full JSON copy of it. Both ceilings raise the same typed
    refusal :func:`materialize_delegation_work_evidence_archive` uses, so an
    attacker-sized delegation population is refused on the freshness probe
    instead of growing the daemon's RSS on every convergence pass.
    """

    index_db = Path(archive_root) / "index.db"
    digest = hashlib.sha256()
    row_count = 0
    byte_count = 0
    with sqlite_connection(index_db) as conn:
        # ``SELECT *`` deliberately: the digest must stay as sensitive as the
        # whole relation, and a hand-kept column list would silently stop
        # tracking a column added later -- freshness would go blind exactly
        # where a new field carries new evidence. The table's own declaration
        # fixes the column order, so the digest is stable across runs, and a
        # genuine schema change correctly forces one re-materialization.
        # polylogue-a7xr.22: reads ``delegation_facts`` (the rename-only
        # ``delegations`` view is gone). ``delegation_id`` is the primary key
        # and is content-derived (``COALESCE(instruction_tool_use_block_id,
        # parent || ':' || child)``), so ordering by it is both a total order
        # -- the previous parent/child ordering left ties free to permute
        # between runs -- and deterministic across rebuilds.
        cursor = conn.execute("SELECT * FROM delegation_facts ORDER BY delegation_id")
        for row in cursor:
            row_count += 1
            if row_count > MAX_DELEGATION_SNAPSHOT_ROWS:
                raise ValueError("delegation work-evidence materialization exceeded its bounded population")
            payload = json.dumps(list(row), separators=(",", ":"), default=str).encode()
            byte_count += len(payload)
            if byte_count > MAX_DELEGATION_SNAPSHOT_BYTES:
                raise ValueError("delegation work-evidence materialization exceeded its bounded population")
            digest.update(payload)
            digest.update(_SNAPSHOT_ROW_SEPARATOR)
    return ObjectRef(kind="context-snapshot", object_id=f"delegations:{digest.hexdigest()[:24]}")


def materialize_delegation_work_evidence_archive(archive_root: Path) -> int:
    """Replace the archive delegation projection and return its row count."""

    archive_root = Path(archive_root)
    snapshot = delegation_work_evidence_snapshot(archive_root)
    # Archive reads are pinned and lifecycle-controlled.  Publication remains
    # the synchronous transaction below and is intentionally independent of
    # this read boundary.
    with open_operation_read(archive_root) as pinned:
        archive = pinned.archive
        rows = archive.query_delegations(QueryBoolPredicate("and", ()), limit=100_001)
    if len(rows) > 100_000:
        raise ValueError("delegation work-evidence materialization exceeded its bounded population")
    graph = materialize_delegation_work_evidence_graph(
        graph_id=DELEGATION_WORK_EVIDENCE_GRAPH_ID,
        corpus_snapshot_ref=snapshot,
        rows=rows,
    )
    # The projection above is archive-wide compute; only this replacement is
    # a write, so it is the only part that enters the daemon writer.
    admit_stage_write(
        "convergence.stage.delegation_work_evidence.publish", lambda: _replace_graph(archive_root / "index.db", graph)
    )
    return len(rows)


def delegation_work_evidence_materialization_needed(archive_root: Path) -> bool:
    """Return whether the stored delegation graph represents current evidence."""

    index_db = Path(archive_root) / "index.db"
    snapshot = delegation_work_evidence_snapshot(archive_root).format()
    with sqlite_connection(index_db) as conn:
        row = conn.execute(
            "SELECT corpus_snapshot_ref FROM work_evidence_graphs WHERE graph_id = ?",
            (DELEGATION_WORK_EVIDENCE_GRAPH_ID,),
        ).fetchone()
    return row is None or str(row[0]) != snapshot


def _replace_graph(index_db: Path, graph: object) -> None:
    # Keep this synchronous: convergence stages own a synchronous SQLite lease.
    from polylogue.analysis.work_evidence import WorkEvidenceGraph

    if not isinstance(graph, WorkEvidenceGraph):
        raise TypeError("expected WorkEvidenceGraph")
    with sqlite_connection(index_db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
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
    "MAX_DELEGATION_SNAPSHOT_BYTES",
    "MAX_DELEGATION_SNAPSHOT_ROWS",
    "delegation_work_evidence_materialization_needed",
    "delegation_work_evidence_snapshot",
    "materialize_delegation_work_evidence_archive",
]
