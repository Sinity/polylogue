"""Correlate acquired Codex ``thread_spawn_edges`` against inferred topology.

``sources/parsers/codex.py`` infers a ``BranchType.SUBAGENT``
``session_links`` edge structurally, from in-session evidence on the CHILD's
own transcript (``source.subagent.thread_spawn`` / ``forked_from_id``).
Acquisition separately retains Codex's own orchestration-level record of the
same relationship -- ``thread_spawn_edges`` from ``state_5.sqlite`` -- and
projects it into ``index.db``'s ``codex_thread_spawn_edges``
(``sources/codex_state_projection.py``).

This is a read-only reconciliation, mirroring the pattern
``context.hermes_lifecycle_reconciliation`` established: a bridge over two
read models (the projected spawn edges, the ingested ``session_links``) that
makes the comparison visible without mutating either side. Codex's own record
can carry edges the transcript never proves (a child that crashed or is still
running), so this reports both directions: inferred edges now backed by
authoritative evidence, and authoritative edges the transcript-based inference
never produced.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from polylogue.core.enums import LinkType, Origin
from polylogue.sources.codex_state_projection import read_spawn_edges


@dataclass(frozen=True, slots=True)
class CodexSpawnEdgeReconciliation:
    """Archive-wide comparison of acquired vs. transcript-inferred Codex spawn edges.

    Edges are identified by ``(parent_thread_id, child_thread_id)`` pairs --
    the raw Codex thread id space both the projection and ``sessions.native_id``
    share for ``Origin.CODEX_SESSION``.
    """

    total_authoritative_edges: int
    total_inferred_subagent_links: int
    backed_by_authoritative_count: int
    inferred_only_count: int
    authoritative_only_count: int
    inferred_only_edges: tuple[tuple[str, str], ...]
    authoritative_only_edges: tuple[tuple[str, str], ...]


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


def reconcile_codex_spawn_edges(index_conn: sqlite3.Connection) -> CodexSpawnEdgeReconciliation:
    """Reconcile acquired Codex spawn-edge evidence against inferred topology.

    ``index_conn`` reads both the projected ``codex_thread_spawn_edges`` and
    the ingested ``session_links`` topology. Neither side is mutated -- see
    the module docstring for why this stays read-only for now.
    """
    authoritative = read_spawn_edges(index_conn)
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
