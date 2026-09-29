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
from collections.abc import Callable, Collection, Iterable, Sequence
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
                "ORDER BY observation_order DESC, source_evidence_ref DESC, graph_id DESC "
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


#: The durable receipt order of a retained export, by raw id; ``None`` when
#: the source tier holds no such export.
ExportOrder = Callable[[str], int | None]

_NODE_SQL = """
    INSERT INTO work_evidence_nodes(
        graph_id, node_ref, node_kind, label, evidence_refs_json, corpus_snapshot_ref,
        authority, confidence, occurred_at_ms, actor_ref, execution_context_id,
        execution_context_known_json, execution_context_unknown_json, role,
        execution_context_addressed, association_state, claim_text
    ) VALUES (?, ?, ?, ?, ?, ?, 'provider', 1.0, ?, NULL, ?, '[]', '[]', 'unknown', 0, ?, ?)
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

_EDGE_SQL = """
    INSERT INTO work_evidence_edges(
        graph_id, edge_ref, edge_kind, source_ref, target_ref, evidence_refs_json,
        corpus_snapshot_ref, authority, confidence, occurred_at_ms, association_state,
        source_state_label
    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'provider', 1.0, ?, ?, ?)
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

#: One graph row an export names: its ref, then the SQL parameters that
#: follow ``graph_id`` up to ``association_state``, then the ones after it.
_GraphRow = tuple[str, tuple[object, ...], tuple[object, ...]]


def _export_rows(
    source_scope: str,
    threads: Sequence[ThreadRecord],
    spawn_edges: Sequence[SpawnRecord],
    *,
    raw_id: str,
    blob_hash: str,
    observed_at_ms: int,
) -> tuple[list[_GraphRow], list[_GraphRow]]:
    """The node and edge rows one export names, in foreign-key order."""
    snapshot_ref = f"artifact:{blob_hash}"
    evidence_json = json.dumps([f"artifact:{raw_id}"])
    context_rows: dict[str, tuple[str, int | None]] = {}
    for thread in threads:
        if not thread.thread_id:
            continue
        label = (thread.title or "").strip() or thread.thread_id
        context_rows[thread.thread_id] = (label, thread.occurred_at_ms)
    for edge in spawn_edges:
        # An edge endpoint the export's thread list is silent about is still
        # a context this runtime named; the graph's foreign keys require both
        # endpoints to exist.
        for endpoint in (edge.parent_thread_id, edge.child_thread_id):
            if endpoint and endpoint not in context_rows:
                context_rows[endpoint] = (endpoint, None)
    titled = [
        (thread.thread_id, (thread.title or "").strip(), thread.occurred_at_ms)
        for thread in threads
        if thread.thread_id and (thread.title or "").strip()
    ]
    nodes: list[_GraphRow] = []
    for thread_id, (label, occurred_at_ms) in sorted(context_rows.items()):
        ref = thread_context_ref(source_scope, thread_id)
        nodes.append(
            (
                ref,
                (
                    ref,
                    "execution-context",
                    label,
                    evidence_json,
                    snapshot_ref,
                    occurred_at_ms,
                    thread_context_id(source_scope, thread_id),
                ),
                (None,),
            )
        )
    for thread_id, title, occurred_at_ms in titled:
        ref = _title_claim_ref(source_scope, thread_id)
        nodes.append((ref, (ref, "claim", title, evidence_json, snapshot_ref, occurred_at_ms, None), (title,)))
    edges: list[_GraphRow] = []
    for thread_id, _title, occurred_at_ms in titled:
        ref = _title_edge_ref(source_scope, thread_id)
        edges.append(
            (
                ref,
                (
                    ref,
                    "claimed",
                    thread_context_ref(source_scope, thread_id),
                    _title_claim_ref(source_scope, thread_id),
                    evidence_json,
                    snapshot_ref,
                    occurred_at_ms,
                ),
                (None,),
            )
        )
    for edge in spawn_edges:
        if not (edge.parent_thread_id and edge.child_thread_id):
            continue
        ref = _spawn_edge_ref(source_scope, edge.parent_thread_id, edge.child_thread_id)
        edges.append(
            (
                ref,
                (
                    ref,
                    "invoked",
                    thread_context_ref(source_scope, edge.parent_thread_id),
                    thread_context_ref(source_scope, edge.child_thread_id),
                    evidence_json,
                    snapshot_ref,
                    observed_at_ms,
                ),
                (edge.status or "unknown",),
            )
        )
    return nodes, edges


def _bound_rows(graph_id: str, rows: Sequence[_GraphRow], state: str) -> list[tuple[object, ...]]:
    return [(graph_id, *head, state, *tail) for _ref, head, tail in rows]


def _write_nodes(conn: sqlite3.Connection, graph_id: str, rows: Sequence[_GraphRow], state: str) -> None:
    conn.executemany(_NODE_SQL, _bound_rows(graph_id, rows, state))


def _write_edges(conn: sqlite3.Connection, graph_id: str, rows: Sequence[_GraphRow], state: str) -> None:
    conn.executemany(_EDGE_SQL, _bound_rows(graph_id, rows, state))


def _retain_older_export_rows(
    conn: sqlite3.Connection,
    graph_id: str,
    nodes: Sequence[_GraphRow],
    edges: Sequence[_GraphRow],
    *,
    incoming_key: tuple[int, str, str],
    export_order: ExportOrder,
) -> bool:
    """Fold an export older than the current one into the retained rows.

    The current export's rows stay as they are. Every other row the older
    export names is superseded evidence: it is added when absent and
    rewritten when the export that last wrote it is older still, so the
    retained rows are those of the newest export naming each object whatever
    order the exports arrive in.
    """
    orders: dict[str, int] = {}

    def writer_key(evidence_refs_json: str, snapshot_ref: str) -> tuple[int, str, str]:
        refs = json.loads(evidence_refs_json)
        writer = str(refs[0]).removeprefix("artifact:") if refs else ""
        if writer not in orders:
            # An export the source tier no longer holds ranks below any it holds.
            ranked = export_order(writer)
            orders[writer] = -1 if ranked is None else ranked
        return (orders[writer], writer, snapshot_ref.removeprefix("artifact:"))

    def retainable(table: str, ref_column: str, rows: Sequence[_GraphRow]) -> list[_GraphRow]:
        existing = {
            str(ref): (str(state), str(evidence), str(snapshot))
            for ref, state, evidence, snapshot in conn.execute(
                f"SELECT {ref_column}, association_state, evidence_refs_json, corpus_snapshot_ref "
                f"FROM {table} WHERE graph_id = ?",
                (graph_id,),
            )
        }
        kept: list[_GraphRow] = []
        for row in rows:
            held = existing.get(row[0])
            if held is None or (held[0] == "superseded" and writer_key(held[1], held[2]) < incoming_key):
                kept.append(row)
        return kept

    retained_nodes = retainable("work_evidence_nodes", "node_ref", nodes)
    retained_edges = retainable("work_evidence_edges", "edge_ref", edges)
    _write_nodes(conn, graph_id, retained_nodes, "superseded")
    _write_edges(conn, graph_id, retained_edges, "superseded")
    return bool(retained_nodes or retained_edges)


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
    export_order: ExportOrder,
) -> bool:
    """Reconcile one scope's graph from the content of one retained export.

    Returns whether any row was written. The newest export by durable receipt
    order is current: replay applies raws in no particular order, and a live
    database that went A -> B -> A reuses A's content-derived raw id. An
    older export still contributes the objects it names as superseded rows,
    so the retained rows do not depend on arrival order; ``export_order``
    ranks the export that wrote each such row.
    """
    current = read_provenance(conn, source_scope=source_scope)
    # The durable receipt order decides; the wall-clock stamp is reported
    # only, because a clock rollback between observations must not reorder
    # them. Receipt orders are normally unique, but callers can legitimately
    # replay synthetic receipts with equal orders, so the content identity is
    # a final tie-break that keeps equal-key replay deterministic.
    incoming_key = (observation_order, raw_id, blob_hash)
    graph_id = thread_state_graph_id(source_scope)
    nodes, edges = _export_rows(
        source_scope, threads, spawn_edges, raw_id=raw_id, blob_hash=blob_hash, observed_at_ms=observed_at_ms
    )
    if current is not None and (current.observation_order, current.raw_id, current.blob_hash) > incoming_key:
        return _retain_older_export_rows(
            conn, graph_id, nodes, edges, incoming_key=incoming_key, export_order=export_order
        )

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
        (graph_id, f"artifact:{blob_hash}", f"artifact:{raw_id}", observed_at_ms, observation_order),
    )
    conn.execute(
        "UPDATE work_evidence_nodes SET association_state = 'superseded' WHERE graph_id = ?",
        (graph_id,),
    )
    conn.execute(
        "UPDATE work_evidence_edges SET association_state = 'superseded' WHERE graph_id = ?",
        (graph_id,),
    )
    _write_nodes(conn, graph_id, nodes, "resolved")
    _write_edges(conn, graph_id, edges, "resolved")
    return True


#: Recency within one scope: a snapshot revision marks what it no longer names
#: superseded, so a still-current object outranks retained absent evidence
#: before any cross-scope receipt ordering applies.
_RECENCY = (
    "ORDER BY CASE WHEN {alias}.association_state = 'superseded' THEN 1 ELSE 0 END, "
    "g.observation_order DESC, g.graph_id DESC"
)


def _scope_predicate(source_scope: str | None) -> tuple[str, list[str]]:
    if source_scope is None:
        return "g.graph_id LIKE ?", [f"{GRAPH_PREFIX}%"]
    return "g.graph_id = ?", [thread_state_graph_id(source_scope)]


_TITLE_TABLES = ("work_evidence_edges", "work_evidence_graphs", "work_evidence_nodes")


def read_thread_titles(
    conn: sqlite3.Connection,
    *,
    thread_ids: Sequence[str] | None = None,
    source_scope: str | None = None,
) -> dict[str, str]:
    """Return ``{thread_id: title}`` from the claimed titles in the graph.

    Only a genuinely absent evidence table means "no titles". Any other read
    failure (an interrupted or expired frame, a locked or unreadable file)
    propagates: reporting it as absence would let ingest persist sessions
    without their retained titles instead of retrying.
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
    present = conn.execute(
        "SELECT COUNT(DISTINCT name) FROM sqlite_master WHERE type = 'table' AND name IN (?, ?, ?)",
        _TITLE_TABLES,
    ).fetchone()[0]
    if present != len(_TITLE_TABLES):
        # Only a genuinely absent evidence table means "no titles"; every read
        # failure below propagates.
        emit(
            "storage.agent_thread_state.titles_absent",
            level=DEBUG,
            outcome="empty",
            reason="the index tier has no work-evidence graph tables, so there are no retained titles",
        )
        return {}
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


def _spawn_parent_rows(
    conn: sqlite3.Connection, source_scope: str | None, child_predicate: str = "", child_parameters: Sequence[str] = ()
) -> list[tuple[str, str, str]]:
    """``(graph_id, parent ref, child ref)`` rows, each graph's current parent first."""
    predicate, parameters = _scope_predicate(source_scope)
    rows = conn.execute(
        f"""
        SELECT e.graph_id, e.source_ref, e.target_ref
        FROM work_evidence_edges AS e
        JOIN work_evidence_graphs AS g ON g.graph_id = e.graph_id
        WHERE {predicate} AND e.edge_kind = 'invoked' {child_predicate}
        {_RECENCY.format(alias="e")}, e.source_ref
        """,
        [*parameters, *child_parameters],
    ).fetchall()
    return [(str(graph_id), str(source_ref), str(target_ref)) for graph_id, source_ref, target_ref in rows]


def _agreed_parents(rows: Iterable[tuple[str, str, str]], wanted: Collection[str] | None) -> dict[str, str]:
    """Each child's parent when every scope that names the child agrees on it.

    A thread id is a scope's own name: one id under two install roots can be
    spawned by different parents, and receipt recency across roots says
    nothing about which root a caller's rollout came from. Disagreeing
    scopes therefore leave the child without a projected parent.
    """
    per_scope: dict[str, dict[str, str]] = {}
    for graph_id, source_ref, target_ref in rows:
        child = thread_id_from_context_ref(target_ref)
        if (wanted is not None and child not in wanted) or graph_id in per_scope.get(child, {}):
            continue
        parent = thread_id_from_context_ref(source_ref).strip()
        if parent:
            per_scope.setdefault(child, {})[graph_id] = parent
    return {
        child: next(iter(set(parents.values())))
        for child, parents in per_scope.items()
        if len(set(parents.values())) == 1
    }


def read_spawn_parents(
    conn: sqlite3.Connection, child_thread_ids: Iterable[str], *, source_scope: str | None = None
) -> dict[str, str]:
    """Return ``{child_thread_id: parent_thread_id}`` for the children the graph is not silent about.

    Each child gets the parent :func:`read_parent_thread_id` would report for
    the same scope. Comparing this per child before and after a snapshot
    revision is what says whose projected parent moved; the set of edges ever
    seen cannot, because superseded edges are retained and a parent that
    returns (A, then B, then A) adds no new edge. A read failure propagates:
    the caller is mid-write and must not re-derive from a graph it could not
    read.
    """
    wanted = {child for child in child_thread_ids if child}
    if not wanted:
        return {}
    return _agreed_parents(_spawn_parent_rows(conn, source_scope), wanted)


def read_spawn_edge_children(conn: sqlite3.Connection) -> set[str]:
    """Return every child thread id the graph carries a spawn edge for."""
    return {child for _parent, child in read_spawn_edges(conn)}


def read_parent_thread_id(
    conn: sqlite3.Connection, child_thread_id: str, *, source_scope: str | None = None
) -> str | None:
    """Return the projected parent of ``child_thread_id``, or ``None`` when silent.

    ``None`` means the graph is silent about this child, which is not the same
    as it naming a different parent; only the latter is a conflict. Without a
    scope, the child's parent is known only where every scope naming it
    agrees (see :func:`_agreed_parents`).
    """
    if not child_thread_id:
        return None
    if source_scope is None:
        child_predicate = "AND e.target_ref LIKE ?"
        child_parameter = f"%{_SCOPE_SEPARATOR}{child_thread_id}"
    else:
        child_predicate = "AND e.target_ref = ?"
        child_parameter = thread_context_ref(source_scope, child_thread_id)
    try:
        rows = _spawn_parent_rows(conn, source_scope, child_predicate, (child_parameter,))
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
    return _agreed_parents(rows, {child_thread_id}).get(child_thread_id)


__all__ = [
    "CONTEXT_PREFIX",
    "GRAPH_PREFIX",
    "ExportOrder",
    "SpawnRecord",
    "ThreadRecord",
    "ThreadStateProvenance",
    "read_parent_thread_id",
    "read_provenance",
    "read_spawn_edge_children",
    "read_spawn_edges",
    "read_spawn_parents",
    "read_thread_titles",
    "thread_context_id",
    "thread_context_ref",
    "thread_id_from_context_ref",
    "thread_state_graph_id",
    "write_thread_state_graph",
]
