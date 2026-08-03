"""Correlate acquired Codex ``thread_spawn_edges`` against inferred topology.

bd polylogue-foee (AC#2). ``sources/parsers/codex.py`` infers a
``BranchType.SUBAGENT`` ``session_links`` edge structurally, from in-session
evidence on the CHILD's own transcript (``source.subagent.thread_spawn`` /
``forked_from_id``). ``polylogue-0jf4`` separately acquires Codex's own
orchestration-level record of the same relationship --
``thread_spawn_edges`` from ``state_5.sqlite`` -- as durable
``codex_thread_spawn_edge`` ``raw_hook_events`` (see
``sources/live/batch.py::_write_codex_thread_state_evidence``). Until this
module, nothing ever read the acquired edges back to compare them against
what the transcript-based inference already produced.

This is a read-only reconciliation, mirroring the pattern
``context.hermes_lifecycle_reconciliation`` established: a bridge over two
durable/rebuildable tiers (``source.db`` hook-event spool,
``index.db`` ingested ``session_links``) that makes the comparison visible
without mutating either side. Codex's own ``thread_spawn_edges`` record can
carry edges the transcript never proves (e.g. a child that crashed or is
still running, per ``sources/parsers/codex_state.py``'s module docstring),
so this reports both directions: inferred edges now backed by authoritative
evidence, and authoritative edges the transcript-based inference never
produced.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from polylogue.core.enums import LinkType, Origin
from polylogue.storage.sqlite.archive_tiers.source_write import list_hook_events


@dataclass(frozen=True, slots=True)
class CodexSpawnEdgeReconciliation:
    """Archive-wide comparison of acquired vs. transcript-inferred Codex spawn edges.

    Edges are identified by ``(parent_thread_id, child_thread_id)`` pairs --
    the raw Codex thread id space both the hook-event payload and
    ``sessions.native_id`` share for ``Origin.CODEX_SESSION``.
    """

    total_authoritative_edges: int
    total_inferred_subagent_links: int
    backed_by_authoritative_count: int
    inferred_only_count: int
    authoritative_only_count: int
    inferred_only_edges: tuple[tuple[str, str], ...]
    authoritative_only_edges: tuple[tuple[str, str], ...]


def _authoritative_spawn_edges(source_conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """Return ``{(parent_thread_id, child_thread_id): status}`` from acquired
    ``codex_thread_spawn_edge`` hook events."""
    edges: dict[tuple[str, str], str] = {}
    for event in list_hook_events(source_conn, origin=Origin.CODEX_SESSION):
        if event.event_type != "codex_thread_spawn_edge":
            continue
        payload = event.payload
        parent = payload.get("parent_thread_id")
        child = payload.get("child_thread_id")
        status = payload.get("status")
        if isinstance(parent, str) and parent and isinstance(child, str) and child:
            edges[(parent, child)] = status if isinstance(status, str) and status else "unknown"
    return edges


def _inferred_subagent_edges(index_conn: sqlite3.Connection) -> set[tuple[str, str]]:
    """Return ``{(parent_thread_id, child_thread_id)}`` for every codex-session
    ``SUBAGENT`` ``session_links`` row -- the edges ``parsers/codex.py``
    infers structurally from the child's own transcript evidence, never from
    ``thread_spawn_edges``."""
    rows = index_conn.execute(
        """
        SELECT sl.dst_native_id AS parent_native_id, s.native_id AS child_native_id
        FROM session_links sl
        JOIN sessions s ON s.session_id = sl.src_session_id
        WHERE s.origin = ? AND sl.dst_origin = ? AND sl.link_type = ?
        """,
        (Origin.CODEX_SESSION.value, Origin.CODEX_SESSION.value, LinkType.SUBAGENT.value),
    ).fetchall()
    # Row-factory agnostic (positional indices): callers may pass a plain
    # tuple-factory connection, not necessarily one with sqlite3.Row set.
    return {(str(row[0]), str(row[1])) for row in rows}


def reconcile_codex_spawn_edges(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
) -> CodexSpawnEdgeReconciliation:
    """Reconcile acquired Codex spawn-edge evidence against inferred topology.

    ``source_conn`` reads the durable hook-event spool (``source.db``);
    ``index_conn`` reads the ingested ``session_links`` topology
    (``index.db``). Neither side is mutated -- see module docstring for why
    this stays read-only for now.
    """
    authoritative = _authoritative_spawn_edges(source_conn)
    inferred = _inferred_subagent_edges(index_conn)
    authoritative_keys = set(authoritative.keys())
    backed = authoritative_keys & inferred
    inferred_only = inferred - authoritative_keys
    authoritative_only = authoritative_keys - inferred
    return CodexSpawnEdgeReconciliation(
        total_authoritative_edges=len(authoritative),
        total_inferred_subagent_links=len(inferred),
        backed_by_authoritative_count=len(backed),
        inferred_only_count=len(inferred_only),
        authoritative_only_count=len(authoritative_only),
        inferred_only_edges=tuple(sorted(inferred_only)),
        authoritative_only_edges=tuple(sorted(authoritative_only)),
    )


__all__ = ["CodexSpawnEdgeReconciliation", "reconcile_codex_spawn_edges"]
