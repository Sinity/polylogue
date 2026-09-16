"""Runtime-reported thread state as provider-neutral work evidence.

An agent runtime that keeps its own orchestration record of a thread -- a
curated title and parent/child spawn relationships -- reaches ``index.db``
through the shared work-evidence graph, not through a provider-named table.
One graph holds one evidence scope (one install root):

* a thread becomes an ``execution-context`` node whose id is
  ``agent-thread:<scope>::<thread_id>``;
* its curated title becomes a ``claim`` node joined by a ``claimed`` edge,
  so a title is evidence a runtime asserted, never a fact about the archive;
* a spawn relationship becomes an ``invoked`` edge from the parent context to
  the child context, carrying the runtime's own lifecycle label in
  ``source_state_label``;
* the retained export the scope was computed from is the graph's
  ``corpus_snapshot_ref`` (its blob hash), ``source_evidence_ref`` (its raw
  artifact) and receipt ordering.

Rows here are recomputed from the durable export on every save and reindex.
A snapshot is complete only for its own scope, so objects missing from the
newest export are marked ``superseded`` rather than deleted: the retained raw
stays recoverable evidence and another install root is untouched.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from polylogue.logging import DEBUG, emit

#: Graph-id namespace for runtime-reported thread state.
GRAPH_PREFIX = "agent-thread-state:"

#: Execution-context id namespace for one runtime thread.
CONTEXT_PREFIX = "agent-thread:"

_SCOPE_SEPARATOR = "::"


def thread_state_graph_id(source_scope: str) -> str:
    """Return the graph id holding one evidence scope's thread state."""
    return f"{GRAPH_PREFIX}{source_scope}"


def thread_context_id(source_scope: str, thread_id: str) -> str:
    """Return the execution-context id for one runtime thread."""
    return f"{CONTEXT_PREFIX}{source_scope}{_SCOPE_SEPARATOR}{thread_id}"


def thread_context_ref(source_scope: str, thread_id: str) -> str:
    """Return the node ref for one runtime thread."""
    return f"execution-context:{thread_context_id(source_scope, thread_id)}"


def thread_id_from_context_ref(node_ref: str) -> str:
    """Return the runtime thread id carried by an execution-context node ref."""
    _, _, tail = node_ref.partition(":")
    if _SCOPE_SEPARATOR not in tail:
        return ""
    return tail.rsplit(_SCOPE_SEPARATOR, 1)[1]


def _title_claim_ref(source_scope: str, thread_id: str) -> str:
    return f"work-claim:{CONTEXT_PREFIX}title:{source_scope}{_SCOPE_SEPARATOR}{thread_id}"


def _spawn_edge_ref(source_scope: str, parent_thread_id: str, child_thread_id: str) -> str:
    return (
        f"work-edge:{CONTEXT_PREFIX}spawn:{source_scope}"
        f"{_SCOPE_SEPARATOR}{parent_thread_id}{_SCOPE_SEPARATOR}{child_thread_id}"
    )


def _title_edge_ref(source_scope: str, thread_id: str) -> str:
    return f"work-edge:{CONTEXT_PREFIX}title:{source_scope}{_SCOPE_SEPARATOR}{thread_id}"


@dataclass(frozen=True, slots=True)
class ThreadStateProvenance:
    """Which retained export one scope's graph was computed from."""

    raw_id: str
    blob_hash: str
    observed_at_ms: int
    observation_order: int


@dataclass(frozen=True, slots=True)
class ThreadRecord:
    """One runtime-reported thread, reduced to its graph-bearing facts."""

    thread_id: str
    title: str | None
    occurred_at_ms: int | None


@dataclass(frozen=True, slots=True)
class SpawnRecord:
    """One runtime-reported parent/child spawn relationship."""

    parent_thread_id: str
    child_thread_id: str
    status: str


def read_provenance(conn: sqlite3.Connection, *, source_scope: str | None = None) -> ThreadStateProvenance | None:
    """Return one scope's provenance, or the newest across scopes."""
    try:
        if source_scope is None:
            row = conn.execute(
                "SELECT source_evidence_ref, corpus_snapshot_ref, observed_at_ms, observation_order "
                "FROM work_evidence_graphs WHERE graph_id LIKE ? AND source_evidence_ref IS NOT NULL "
                "ORDER BY observed_at_ms DESC, observation_order DESC, source_evidence_ref DESC, graph_id DESC "
                "LIMIT 1",
                (f"{GRAPH_PREFIX}%",),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT source_evidence_ref, corpus_snapshot_ref, observed_at_ms, observation_order "
                "FROM work_evidence_graphs WHERE graph_id = ? AND source_evidence_ref IS NOT NULL",
                (thread_state_graph_id(source_scope),),
            ).fetchone()
    except sqlite3.Error as exc:
        emit(
            "storage.agent_thread_state.provenance_unreadable",
            level=DEBUG,
            outcome="unmeasured",
            reason="the index tier is unreadable, so no scope provenance is known",
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return None
    if row is None:
        return None
    raw_id = str(row[0]).removeprefix("artifact:")
    blob_hash = str(row[1]).removeprefix("artifact:")
    return ThreadStateProvenance(raw_id, blob_hash, int(row[2]), int(row[3]))


def write_thread_state_graph(
    conn: sqlite3.Connection,
    *,
    source_scope: str,
    threads: Sequence[ThreadRecord],
    spawn_edges: Sequence[SpawnRecord],
    raw_id: str,
    blob_hash: str,
    observed_at_ms: int,
    observation_order: int = 0,
) -> bool:
    """Reconcile one scope's graph from the content of one retained export.

    Returns whether the graph was written. An export older than the one
    already projected is skipped and reported as ``False``: replay applies
    raws in no particular order, and a live database that went A -> B -> A
    reuses A's content-derived raw id, so the durable receipt order is what
    says which observation is current.
    """
    current = read_provenance(conn, source_scope=source_scope)
    # Receipt timestamps and rowids are normally unique, but callers can
    # legitimately replay synthetic receipts with equal ordering fields.
    # Include the content identity as a final tie-break so equal-key replay is
    # deterministic rather than dependent on which raw arrived first.
    incoming_key = (observed_at_ms, observation_order, raw_id, blob_hash)
    if (
        current is not None
        and (current.observed_at_ms, current.observation_order, current.raw_id, current.blob_hash) > incoming_key
    ):
        return False

    graph_id = thread_state_graph_id(source_scope)
    snapshot_ref = f"artifact:{blob_hash}"
    evidence_json = json.dumps([f"artifact:{raw_id}"])
    conn.execute(
        """
        INSERT INTO work_evidence_graphs(
            graph_id, corpus_snapshot_ref, source_evidence_ref, observed_at_ms, observation_order
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(graph_id) DO UPDATE SET
            corpus_snapshot_ref = excluded.corpus_snapshot_ref,
            source_evidence_ref = excluded.source_evidence_ref,
            observed_at_ms = excluded.observed_at_ms,
            observation_order = excluded.observation_order
        """,
        (graph_id, snapshot_ref, f"artifact:{raw_id}", observed_at_ms, observation_order),
    )
    conn.execute(
        "UPDATE work_evidence_nodes SET association_state = 'superseded' WHERE graph_id = ?",
        (graph_id,),
    )
    conn.execute(
        "UPDATE work_evidence_edges SET association_state = 'superseded' WHERE graph_id = ?",
        (graph_id,),
    )

    context_rows: dict[str, tuple[str, int | None]] = {}
    for thread in threads:
        if not thread.thread_id:
            continue
        label = (thread.title or "").strip() or thread.thread_id
        context_rows[thread.thread_id] = (label, thread.occurred_at_ms)
    for edge in spawn_edges:
        # An edge endpoint the newest export's thread list is silent about is
        # still a context this runtime named; the graph's foreign keys require
        # both endpoints to exist.
        for endpoint in (edge.parent_thread_id, edge.child_thread_id):
            if endpoint and endpoint not in context_rows:
                context_rows[endpoint] = (endpoint, None)

    node_sql = """
        INSERT INTO work_evidence_nodes(
            graph_id, node_ref, node_kind, label, evidence_refs_json, corpus_snapshot_ref,
            authority, confidence, occurred_at_ms, actor_ref, execution_context_id,
            execution_context_known_json, execution_context_unknown_json, role,
            execution_context_addressed, association_state, claim_text
        ) VALUES (?, ?, ?, ?, ?, ?, 'provider', 1.0, ?, NULL, ?, '[]', '[]', 'unknown', 0, 'resolved', ?)
        ON CONFLICT(graph_id, node_ref) DO UPDATE SET
            node_kind = excluded.node_kind,
            label = excluded.label,
            evidence_refs_json = excluded.evidence_refs_json,
            corpus_snapshot_ref = excluded.corpus_snapshot_ref,
            authority = excluded.authority,
            confidence = excluded.confidence,
            occurred_at_ms = excluded.occurred_at_ms,
            execution_context_id = excluded.execution_context_id,
            execution_context_addressed = excluded.execution_context_addressed,
            association_state = excluded.association_state,
            claim_text = excluded.claim_text
    """
    conn.executemany(
        node_sql,
        [
            (
                graph_id,
                thread_context_ref(source_scope, thread_id),
                "execution-context",
                label,
                evidence_json,
                snapshot_ref,
                occurred_at_ms,
                thread_context_id(source_scope, thread_id),
                None,
            )
            for thread_id, (label, occurred_at_ms) in sorted(context_rows.items())
        ],
    )
    titled = [
        (thread.thread_id, (thread.title or "").strip(), thread.occurred_at_ms)
        for thread in threads
        if thread.thread_id and (thread.title or "").strip()
    ]
    conn.executemany(
        node_sql,
        [
            (
                graph_id,
                _title_claim_ref(source_scope, thread_id),
                "claim",
                title,
                evidence_json,
                snapshot_ref,
                occurred_at_ms,
                None,
                title,
            )
            for thread_id, title, occurred_at_ms in titled
        ],
    )

    edge_sql = """
        INSERT INTO work_evidence_edges(
            graph_id, edge_ref, edge_kind, source_ref, target_ref, evidence_refs_json,
            corpus_snapshot_ref, authority, confidence, occurred_at_ms, association_state,
            source_state_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'provider', 1.0, ?, 'resolved', ?)
        ON CONFLICT(graph_id, edge_ref) DO UPDATE SET
            edge_kind = excluded.edge_kind,
            source_ref = excluded.source_ref,
            target_ref = excluded.target_ref,
            evidence_refs_json = excluded.evidence_refs_json,
            corpus_snapshot_ref = excluded.corpus_snapshot_ref,
            authority = excluded.authority,
            confidence = excluded.confidence,
            occurred_at_ms = excluded.occurred_at_ms,
            association_state = excluded.association_state,
            source_state_label = excluded.source_state_label
    """
    conn.executemany(
        edge_sql,
        [
            (
                graph_id,
                _title_edge_ref(source_scope, thread_id),
                "claimed",
                thread_context_ref(source_scope, thread_id),
                _title_claim_ref(source_scope, thread_id),
                evidence_json,
                snapshot_ref,
                occurred_at_ms,
                None,
            )
            for thread_id, _title, occurred_at_ms in titled
        ],
    )
    conn.executemany(
        edge_sql,
        [
            (
                graph_id,
                _spawn_edge_ref(source_scope, edge.parent_thread_id, edge.child_thread_id),
                "invoked",
                thread_context_ref(source_scope, edge.parent_thread_id),
                thread_context_ref(source_scope, edge.child_thread_id),
                evidence_json,
                snapshot_ref,
                observed_at_ms,
                edge.status or "unknown",
            )
            for edge in spawn_edges
            if edge.parent_thread_id and edge.child_thread_id
        ],
    )
    return True


#: Recency within one scope: a snapshot revision marks what it no longer names
#: superseded, so a still-current object outranks retained absent evidence
#: before any cross-scope receipt ordering applies.
_RECENCY = (
    "ORDER BY CASE WHEN {alias}.association_state = 'superseded' THEN 1 ELSE 0 END, "
    "g.observed_at_ms DESC, g.observation_order DESC, g.graph_id DESC"
)


def _scope_predicate(source_scope: str | None) -> tuple[str, list[str]]:
    if source_scope is None:
        return "g.graph_id LIKE ?", [f"{GRAPH_PREFIX}%"]
    return "g.graph_id = ?", [thread_state_graph_id(source_scope)]


def read_thread_titles(
    conn: sqlite3.Connection,
    *,
    thread_ids: Sequence[str] | None = None,
    source_scope: str | None = None,
) -> dict[str, str]:
    """Return ``{thread_id: title}`` from the claimed titles in the graph.

    Any failure (missing table, locked file) degrades to an empty mapping,
    matching every other sidecar source in the title ladder.
    """
    predicate, parameters = _scope_predicate(source_scope)
    base = f"""
        SELECT e.source_ref, n.claim_text
        FROM work_evidence_edges AS e
        JOIN work_evidence_graphs AS g ON g.graph_id = e.graph_id
        JOIN work_evidence_nodes AS n ON n.graph_id = e.graph_id AND n.node_ref = e.target_ref
        WHERE {predicate} AND e.edge_kind = 'claimed' AND n.node_kind = 'claim'
          AND n.claim_text IS NOT NULL
    """
    ordering = _RECENCY.format(alias="e")
    titles: dict[str, str] = {}
    try:
        rows: list[tuple[object, ...]] = []
        if thread_ids is None:
            rows = conn.execute(f"{base} {ordering}", parameters).fetchall()
        else:
            wanted = list(dict.fromkeys(thread_ids))
            if not wanted:
                return {}
            for start in range(0, len(wanted), 500):
                chunk = wanted[start : start + 500]
                if source_scope is None:
                    # Scope is unknown, so match on the thread-id tail of the
                    # context ref rather than on a ref we cannot spell.
                    clause = " OR ".join("e.source_ref LIKE ?" for _ in chunk)
                    chunk_parameters = [f"%{_SCOPE_SEPARATOR}{item}" for item in chunk]
                else:
                    clause = " OR ".join("e.source_ref = ?" for _ in chunk)
                    chunk_parameters = [thread_context_ref(source_scope, item) for item in chunk]
                rows.extend(
                    conn.execute(
                        f"{base} AND ({clause}) {ordering}",
                        [*parameters, *chunk_parameters],
                    ).fetchall()
                )
    except sqlite3.Error as exc:
        emit(
            "storage.agent_thread_state.titles_unreadable",
            level=DEBUG,
            outcome="unmeasured",
            reason="the index tier is unreadable, so the title lane degrades to empty",
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return {}
    for row in rows:
        thread_id = thread_id_from_context_ref(str(row[0]))
        title = row[1]
        if thread_id and thread_id not in titles and isinstance(title, str) and title.strip():
            titles[thread_id] = title.strip()
    return titles


def read_spawn_edges(conn: sqlite3.Connection, *, source_scope: str | None = None) -> dict[tuple[str, str], str]:
    """Return ``{(parent_thread_id, child_thread_id): status}`` from the graph."""
    predicate, parameters = _scope_predicate(source_scope)
    try:
        rows = conn.execute(
            f"""
            SELECT e.source_ref, e.target_ref, e.source_state_label
            FROM work_evidence_edges AS e
            JOIN work_evidence_graphs AS g ON g.graph_id = e.graph_id
            WHERE {predicate} AND e.edge_kind = 'invoked'
            {_RECENCY.format(alias="e")}
            """,
            parameters,
        ).fetchall()
    except sqlite3.Error as exc:
        emit(
            "storage.agent_thread_state.spawn_edges_unreadable",
            level=DEBUG,
            outcome="unmeasured",
            reason="the index tier is unreadable, so no spawn evidence is reported",
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return {}
    edges: dict[tuple[str, str], str] = {}
    for row in rows:
        parent = thread_id_from_context_ref(str(row[0]))
        child = thread_id_from_context_ref(str(row[1]))
        if parent and child:
            edges.setdefault((parent, child), str(row[2] or "unknown"))
    return edges


def read_spawn_edge_children(conn: sqlite3.Connection) -> set[str]:
    """Return every child thread id the graph carries a spawn edge for."""
    return {child for _parent, child in read_spawn_edges(conn)}


def read_parent_thread_id(
    conn: sqlite3.Connection, child_thread_id: str, *, source_scope: str | None = None
) -> str | None:
    """Return the projected parent of ``child_thread_id``, or ``None`` when silent.

    ``None`` means the graph is silent about this child, which is not the same
    as it naming a different parent; only the latter is a conflict.
    """
    if not child_thread_id:
        return None
    predicate, parameters = _scope_predicate(source_scope)
    if source_scope is None:
        child_predicate = "e.target_ref LIKE ?"
        child_parameter = f"%{_SCOPE_SEPARATOR}{child_thread_id}"
    else:
        child_predicate = "e.target_ref = ?"
        child_parameter = thread_context_ref(source_scope, child_thread_id)
    try:
        row = conn.execute(
            f"""
            SELECT e.source_ref
            FROM work_evidence_edges AS e
            JOIN work_evidence_graphs AS g ON g.graph_id = e.graph_id
            WHERE {predicate} AND e.edge_kind = 'invoked' AND {child_predicate}
            {_RECENCY.format(alias="e")}, e.source_ref
            LIMIT 1
            """,
            [*parameters, child_parameter],
        ).fetchone()
    except sqlite3.Error as exc:
        emit(
            "storage.agent_thread_state.spawn_parent_unreadable",
            level=DEBUG,
            outcome="unmeasured",
            reason="the index tier is unreadable, so the graph is treated as silent about this child",
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return None
    if row is None or row[0] is None:
        return None
    parent = thread_id_from_context_ref(str(row[0])).strip()
    return parent or None


__all__ = [
    "CONTEXT_PREFIX",
    "GRAPH_PREFIX",
    "SpawnRecord",
    "ThreadRecord",
    "ThreadStateProvenance",
    "read_parent_thread_id",
    "read_provenance",
    "read_spawn_edge_children",
    "read_spawn_edges",
    "read_thread_titles",
    "thread_context_id",
    "thread_context_ref",
    "thread_id_from_context_ref",
    "thread_state_graph_id",
    "write_thread_state_graph",
]
