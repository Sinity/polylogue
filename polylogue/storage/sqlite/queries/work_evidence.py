"""SQLite persistence and traversal for provider-neutral work evidence."""

from __future__ import annotations

import json
from typing import cast

import aiosqlite

from polylogue.core.refs import ActorRef, ExecutionContextRef, ObjectRef
from polylogue.insights.work_evidence import (
    WorkEvidenceAssociationState,
    WorkEvidenceAuthority,
    WorkEvidenceEdge,
    WorkEvidenceEdgeKind,
    WorkEvidenceGraph,
    WorkEvidenceNode,
    WorkEvidenceNodeKind,
    WorkEvidenceTraversal,
    parse_work_evidence_source_ref,
)
from polylogue.storage.query_models import WorkEvidenceTraversalQuery

__all__ = ["get_work_evidence_traversal", "replace_work_evidence_graph"]


def _json_array(value: object) -> tuple[str, ...]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("stored work-evidence array must be a JSON string array")
    return tuple(parsed)


def _node_from_row(row: aiosqlite.Row) -> WorkEvidenceNode:
    context_id = row["execution_context_id"]
    context = (
        ExecutionContextRef(
            context_id=str(context_id),
            known_fields=_json_array(row["execution_context_known_json"]),
            unknown_fields=_json_array(row["execution_context_unknown_json"]),
            content_addressed=bool(row["execution_context_addressed"]),
        )
        if context_id is not None
        else None
    )
    return WorkEvidenceNode(
        ref=ObjectRef.parse(str(row["node_ref"])),
        kind=cast(WorkEvidenceNodeKind, str(row["node_kind"])),
        label=str(row["label"]),
        evidence_refs=tuple(parse_work_evidence_source_ref(value) for value in _json_array(row["evidence_refs_json"])),
        corpus_snapshot_ref=ObjectRef.parse(str(row["corpus_snapshot_ref"])),
        authority=cast(WorkEvidenceAuthority, str(row["authority"])),
        confidence=float(row["confidence"]),
        occurred_at_ms=int(row["occurred_at_ms"]) if row["occurred_at_ms"] is not None else None,
        actor_ref=ActorRef.parse(str(row["actor_ref"])) if row["actor_ref"] is not None else None,
        execution_context_ref=context,
        association_state=cast(WorkEvidenceAssociationState, str(row["association_state"])),
        claim_text=str(row["claim_text"]) if row["claim_text"] is not None else None,
    )


def _edge_from_row(row: aiosqlite.Row) -> WorkEvidenceEdge:
    return WorkEvidenceEdge(
        ref=ObjectRef.parse(str(row["edge_ref"])),
        kind=cast(WorkEvidenceEdgeKind, str(row["edge_kind"])),
        source_ref=ObjectRef.parse(str(row["source_ref"])),
        target_ref=ObjectRef.parse(str(row["target_ref"])),
        evidence_refs=tuple(parse_work_evidence_source_ref(value) for value in _json_array(row["evidence_refs_json"])),
        corpus_snapshot_ref=ObjectRef.parse(str(row["corpus_snapshot_ref"])),
        authority=cast(WorkEvidenceAuthority, str(row["authority"])),
        confidence=float(row["confidence"]),
        occurred_at_ms=int(row["occurred_at_ms"]) if row["occurred_at_ms"] is not None else None,
        association_state=cast(WorkEvidenceAssociationState, str(row["association_state"])),
    )


async def replace_work_evidence_graph(
    conn: aiosqlite.Connection,
    graph: WorkEvidenceGraph,
    transaction_depth: int,
) -> None:
    """Atomically replace one rebuildable graph snapshot."""

    await conn.execute("DELETE FROM work_evidence_graphs WHERE graph_id = ?", (graph.graph_id,))
    await conn.execute(
        "INSERT INTO work_evidence_graphs(graph_id, corpus_snapshot_ref) VALUES (?, ?)",
        (graph.graph_id, graph.corpus_snapshot_ref.format()),
    )
    await conn.executemany(
        """
        INSERT INTO work_evidence_nodes(
            graph_id, node_ref, node_kind, label, evidence_refs_json, corpus_snapshot_ref,
            authority, confidence, occurred_at_ms, actor_ref, execution_context_id,
            execution_context_known_json, execution_context_unknown_json,
            execution_context_addressed, association_state, claim_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                int(node.execution_context_ref.content_addressed) if node.execution_context_ref else None,
                node.association_state,
                node.claim_text,
            )
            for node in graph.nodes
        ],
    )
    await conn.executemany(
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
    if transaction_depth == 0:
        await conn.commit()


async def get_work_evidence_traversal(
    conn: aiosqlite.Connection,
    query: WorkEvidenceTraversalQuery,
) -> WorkEvidenceTraversal | None:
    """Return an incoming, outgoing, or bidirectional one-hop neighborhood."""

    focal = ObjectRef.parse(query.focal_ref).format()
    graph_cursor = await conn.execute(
        "SELECT corpus_snapshot_ref FROM work_evidence_graphs WHERE graph_id = ?",
        (query.graph_id,),
    )
    if await graph_cursor.fetchone() is None:
        return None
    node_cursor = await conn.execute(
        "SELECT * FROM work_evidence_nodes WHERE graph_id = ? AND node_ref = ?",
        (query.graph_id, focal),
    )
    focal_row = await node_cursor.fetchone()
    if focal_row is None:
        return None

    predicates: list[str] = []
    params: list[object] = [query.graph_id]
    if query.direction == "incoming":
        predicates.append("target_ref = ?")
        params.append(focal)
    elif query.direction == "outgoing":
        predicates.append("source_ref = ?")
        params.append(focal)
    else:
        predicates.append("(source_ref = ? OR target_ref = ?)")
        params.extend([focal, focal])
    if query.edge_kinds:
        placeholders = ", ".join("?" for _ in query.edge_kinds)
        predicates.append(f"edge_kind IN ({placeholders})")
        params.extend(query.edge_kinds)
    sql = "SELECT * FROM work_evidence_edges WHERE graph_id = ? AND " + " AND ".join(predicates)
    sql += " ORDER BY occurred_at_ms, edge_ref"
    if query.limit is not None:
        sql += " LIMIT ?"
        params.append(query.limit)
    edge_cursor = await conn.execute(sql, tuple(params))
    edges = tuple(_edge_from_row(row) for row in await edge_cursor.fetchall())
    node_refs = {focal}
    for edge in edges:
        node_refs.add(edge.source_ref.format())
        node_refs.add(edge.target_ref.format())
    placeholders = ", ".join("?" for _ in node_refs)
    neighbors_cursor = await conn.execute(
        f"SELECT * FROM work_evidence_nodes WHERE graph_id = ? AND node_ref IN ({placeholders}) ORDER BY node_ref",
        (query.graph_id, *sorted(node_refs)),
    )
    nodes = tuple(_node_from_row(row) for row in await neighbors_cursor.fetchall())
    return WorkEvidenceTraversal(
        graph_id=query.graph_id,
        focal_ref=ObjectRef.parse(focal),
        nodes=nodes,
        edges=edges,
    )
