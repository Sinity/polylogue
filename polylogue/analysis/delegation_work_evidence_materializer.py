"""Materialize the canonical delegation facts into the generic work graph."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

from polylogue.analysis.delegation_work_evidence import (
    ASSOCIATION_STATE_RANK,
    materialize_delegation_work_evidence_graph,
)
from polylogue.analysis.work_evidence import WorkEvidenceEdge, WorkEvidenceNode
from polylogue.archive.query.predicate import QueryBoolPredicate
from polylogue.core.refs import ObjectRef
from polylogue.core.stage_admission import admit_stage_write
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.sqlite.archive_tiers.archive_query_reads import DelegationPageKey
from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection

DELEGATION_WORK_EVIDENCE_GRAPH_ID = "delegation:archive"


#: Rows read per page while materializing. A pacing bound only: every page
#: is read, so the materialized graph always covers the whole population.
DELEGATION_READ_PAGE_ROWS = 1_000

#: Instruction and artifact text bytes read per page. Row count alone does
#: not bound memory: a page of multi-megabyte tool results would. A page
#: always holds at least one row, so every row is still read.
DELEGATION_READ_PAGE_TEXT_BYTES = 16 * 1024 * 1024

_NODE_COLUMNS = (
    "node_ref",
    "node_kind",
    "label",
    "evidence_refs_json",
    "corpus_snapshot_ref",
    "authority",
    "confidence",
    "occurred_at_ms",
    "actor_ref",
    "execution_context_id",
    "execution_context_known_json",
    "execution_context_unknown_json",
    "role",
    "execution_context_addressed",
    "association_state",
    "claim_text",
)
_EDGE_COLUMNS = (
    "edge_ref",
    "edge_kind",
    "source_ref",
    "target_ref",
    "evidence_refs_json",
    "corpus_snapshot_ref",
    "authority",
    "confidence",
    "occurred_at_ms",
    "association_state",
)

_SNAPSHOT_ROW_SEPARATOR = b"\x1e"


def _snapshot_connection(conn: sqlite3.Connection) -> ObjectRef:
    """Digest delegation facts using the caller's already-pinned index view."""

    digest = hashlib.sha256()
    # ``SELECT *`` deliberately: the digest must stay as sensitive as the
    # whole relation, and a hand-kept column list would silently stop tracking
    # a column added later. ``delegation_id`` is the content-derived primary
    # key, so ordering by it is a total order, deterministic across rebuilds.
    cursor = conn.execute("SELECT * FROM delegation_facts ORDER BY delegation_id")
    for row in cursor:
        digest.update(json.dumps(list(row), separators=(",", ":"), default=str).encode())
        digest.update(_SNAPSHOT_ROW_SEPARATOR)
    return ObjectRef(kind="context-snapshot", object_id=f"delegations:{digest.hexdigest()[:24]}")


def delegation_work_evidence_snapshot(archive_root: Path) -> ObjectRef:
    """Return a content-derived snapshot for the current delegation facts.

    Digested incrementally: the cursor is iterated row by row and each row is
    folded into one SHA-256, so peak memory is one row rather than the whole
    view plus a full JSON copy of it. The digest covers every row; no
    population size is refused.
    """

    with open_operation_read(Path(archive_root)) as pinned:
        return _snapshot_connection(pinned.archive._conn)


def materialize_delegation_work_evidence_archive(archive_root: Path) -> int:
    """Replace the archive delegation projection and return its row count."""

    archive_root = Path(archive_root)
    count = 0
    # Resident memory is one page: each page's partial graph is folded into
    # a private on-disk scratch graph, which publication then streams.
    with tempfile.TemporaryDirectory(prefix="polylogue-delegation-graph-") as scratch_dir:
        scratch_path = Path(scratch_dir) / "graph.db"
        with closing(sqlite3.connect(scratch_path)) as scratch:
            _create_scratch_graph(scratch)
            # Archive reads are pinned and lifecycle-controlled. Publication
            # remains the synchronous transaction below and is intentionally
            # independent of this read boundary.
            with open_operation_read(archive_root) as pinned:
                archive = pinned.archive
                # The snapshot and every page read the same pinned view.
                snapshot = _snapshot_connection(archive._conn)
                after: DelegationPageKey | None = None
                while True:
                    # Keyset, not OFFSET: an offset page rescans every
                    # earlier row, so the whole read was quadratic.
                    page = archive.query_delegations(
                        QueryBoolPredicate("and", ()),
                        limit=DELEGATION_READ_PAGE_ROWS,
                        after=after,
                        max_text_bytes=DELEGATION_READ_PAGE_TEXT_BYTES,
                    )
                    if not page:
                        break
                    _fold_page(
                        scratch,
                        materialize_delegation_work_evidence_graph(
                            graph_id=DELEGATION_WORK_EVIDENCE_GRAPH_ID,
                            corpus_snapshot_ref=snapshot,
                            rows=page,
                        ),
                    )
                    count += len(page)
                    after = DelegationPageKey.after_row(page[-1])
        # The projection above is archive-wide compute; only this replacement
        # is a write, so it is the only part that enters the daemon writer.
        admit_stage_write(
            "convergence.stage.delegation_work_evidence.publish",
            lambda: _replace_graph(archive_root, snapshot, scratch_path),
        )
    return count


def _create_scratch_graph(scratch: sqlite3.Connection) -> None:
    scratch.execute(f"CREATE TABLE nodes ({', '.join(_NODE_COLUMNS)}, PRIMARY KEY(node_ref))")
    scratch.execute(f"CREATE TABLE edges ({', '.join(_EDGE_COLUMNS)}, PRIMARY KEY(edge_ref))")
    # An attempt node's evidence refs are unioned across every page that
    # touches it (a high-fan-in child session). Deduplicating through a
    # PRIMARY KEY upsert here, instead of round-tripping the whole
    # accumulated set through Python on every page, keeps each page's cost
    # to its own rows: no page reads or re-serializes another page's
    # contribution. ``nodes.evidence_refs_json`` is left as each page's own
    # value for an attempt node; publication aggregates the durable value.
    scratch.execute(
        "CREATE TABLE node_evidence_refs (node_ref TEXT, evidence_ref TEXT, PRIMARY KEY(node_ref, evidence_ref))"
    )


def _fold_page(scratch: sqlite3.Connection, graph: object) -> None:
    """Fold one page's graph into the scratch graph, as one pass would.

    A later page replaces a call, claim or edge with the same identity, and
    merges an attempt node exactly as the single-pass projection does: the
    stronger association state is kept, and evidence refs accumulate in
    ``node_evidence_refs`` rather than being unioned in Python (see that
    table's comment).
    """
    from polylogue.analysis.work_evidence import WorkEvidenceGraph

    if not isinstance(graph, WorkEvidenceGraph):
        raise TypeError("expected WorkEvidenceGraph")
    node_marks = ", ".join("?" for _ in _NODE_COLUMNS)
    attempt_refs: list[tuple[str, str]] = []
    for node in graph.nodes:
        row = _node_row(node)
        if node.kind == "attempt":
            node_ref = str(row[0])
            attempt_refs.extend((node_ref, ref.format()) for ref in node.evidence_refs)
            existing = scratch.execute("SELECT association_state FROM nodes WHERE node_ref = ?", (row[0],)).fetchone()
            if existing is not None:
                state = max(existing[0], node.association_state, key=lambda value: ASSOCIATION_STATE_RANK[value])
                merged = dict(zip(_NODE_COLUMNS, row, strict=True))
                merged.update(
                    association_state=state,
                    confidence=1.0 if state == "resolved" else 0.5,
                )
                row = tuple(merged[column] for column in _NODE_COLUMNS)
        scratch.execute(f"INSERT OR REPLACE INTO nodes VALUES ({node_marks})", row)
    if attempt_refs:
        scratch.executemany("INSERT OR IGNORE INTO node_evidence_refs VALUES (?, ?)", attempt_refs)
    edge_marks = ", ".join("?" for _ in _EDGE_COLUMNS)
    scratch.executemany(
        f"INSERT OR REPLACE INTO edges VALUES ({edge_marks})", (_edge_row(edge) for edge in graph.edges)
    )
    scratch.commit()


def _node_row(node: WorkEvidenceNode) -> tuple[object, ...]:
    context = node.execution_context_ref
    return (
        node.ref.format(),
        node.kind,
        node.label,
        json.dumps([ref.format() for ref in node.evidence_refs]),
        node.corpus_snapshot_ref.format(),
        node.authority,
        node.confidence,
        node.occurred_at_ms,
        node.actor_ref.format() if node.actor_ref else None,
        context.context_id if context else None,
        json.dumps(list(context.known_fields)) if context else "[]",
        json.dumps(list(context.unknown_fields)) if context else "[]",
        node.role,
        int(context.content_addressed) if context else None,
        node.association_state,
        node.claim_text,
    )


def _edge_row(edge: WorkEvidenceEdge) -> tuple[object, ...]:
    return (
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


def delegation_work_evidence_materialization_needed(archive_root: Path) -> bool:
    """Return whether the stored delegation graph represents current evidence."""

    with open_operation_read(Path(archive_root)) as pinned:
        snapshot = _snapshot_connection(pinned.archive._conn).format()
        row = pinned.archive._conn.execute(
            "SELECT corpus_snapshot_ref FROM work_evidence_graphs WHERE graph_id = ?",
            (DELEGATION_WORK_EVIDENCE_GRAPH_ID,),
        ).fetchone()
    return row is None or str(row[0]) != snapshot


#: An attempt node's durable evidence_refs_json is the deduplicated, sorted
#: union accumulated across every page in node_evidence_refs (see
#: _fold_page); every other kind keeps its own row value. The aggregate runs
#: server-side per row as the cursor streams, so publication never
#: materializes the whole node set or a node's whole evidence-ref set in
#: Python.
_NODE_SELECT_COLUMNS = ", ".join(
    (
        "CASE WHEN node_kind = 'attempt' THEN "
        "COALESCE((SELECT json_group_array(evidence_ref) FROM "
        "(SELECT evidence_ref FROM node_evidence_refs WHERE node_evidence_refs.node_ref = nodes.node_ref "
        "ORDER BY evidence_ref)), '[]') "
        "ELSE evidence_refs_json END AS evidence_refs_json"
        if column == "evidence_refs_json"
        else column
    )
    for column in _NODE_COLUMNS
)


def _published_node_rows(scratch: sqlite3.Connection) -> sqlite3.Cursor:
    """The scratch graph's nodes with each attempt's final aggregated refs."""
    return scratch.execute(f"SELECT {_NODE_SELECT_COLUMNS} FROM nodes ORDER BY node_ref")


def _replace_graph(archive_root: Path, snapshot: ObjectRef, scratch_path: Path) -> None:
    # Keep this synchronous: convergence stages own a synchronous SQLite lease.
    graph_id = DELEGATION_WORK_EVIDENCE_GRAPH_ID
    index_db = resolve_active_index_path(archive_root)
    with closing(sqlite3.connect(scratch_path)) as scratch:
        conn = open_isolated_write_connection(
            index_db,
            purpose="convergence.stage.delegation_work_evidence.publish",
            archive_root=archive_root,
        )
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM work_evidence_graphs WHERE graph_id = ?", (graph_id,))
            conn.execute(
                "INSERT INTO work_evidence_graphs(graph_id, corpus_snapshot_ref) VALUES (?, ?)",
                (graph_id, snapshot.format()),
            )
            conn.executemany(
                f"INSERT INTO work_evidence_nodes(graph_id, {', '.join(_NODE_COLUMNS)}) "
                f"VALUES (?, {', '.join('?' for _ in _NODE_COLUMNS)})",
                _prefixed(graph_id, _published_node_rows(scratch)),
            )
            conn.executemany(
                f"INSERT INTO work_evidence_edges(graph_id, {', '.join(_EDGE_COLUMNS)}) "
                f"VALUES (?, {', '.join('?' for _ in _EDGE_COLUMNS)})",
                _prefixed(graph_id, scratch.execute(f"SELECT {', '.join(_EDGE_COLUMNS)} FROM edges ORDER BY edge_ref")),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


def _prefixed(graph_id: str, rows: sqlite3.Cursor) -> Iterator[tuple[object, ...]]:
    for row in rows:
        yield (graph_id, *row)


__all__ = [
    "DELEGATION_WORK_EVIDENCE_GRAPH_ID",
    "DELEGATION_READ_PAGE_ROWS",
    "DELEGATION_READ_PAGE_TEXT_BYTES",
    "delegation_work_evidence_materialization_needed",
    "delegation_work_evidence_snapshot",
    "materialize_delegation_work_evidence_archive",
]
