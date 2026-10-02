"""Codex thread state as derived index-tier data.

Codex's own orchestration record of a thread -- the curated ``threads.title``
and the ``thread_spawn_edges`` parent/child relationships -- is retained
durably as one canonical logical export per state-database revision. It
reaches reads through ``index.db``, recomputed from the current export on
every save and on every reindex.

It is never durable per-row material: a row minted from a database row is a
projection no prover can reproduce against the live database, so it would sit
as unresolved residue for as long as the archive exists.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polylogue.storage.sqlite.agent_thread_state as agent_thread_state
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider
from polylogue.logging import get_logger
from polylogue.sources.parsers import codex_state
from polylogue.storage.sqlite.agent_thread_state import SpawnRecord, ThreadRecord

logger = get_logger(__name__)

#: The declared member kind whose export carries thread titles and spawn edges.
THREAD_STATE_KIND = "thread_state"


def thread_state_member_filenames() -> tuple[str, ...]:
    """Return the declared basenames whose export carries thread state."""
    from polylogue.sources.origin_specs import database_capability_for_provider

    capability = database_capability_for_provider(Provider.CODEX)
    if capability is None:
        return ()
    return tuple(
        member.filename
        for member in capability.members
        if member.kind == THREAD_STATE_KIND and member.disposition != "out-of-scope"
    )


def codex_state_source_scope(source_path: str) -> str:
    """Return the Codex-install scope shared by state and rollout evidence.

    ``state_5.sqlite`` sits directly in the install directory while rollouts
    live below ``sessions/``.  The retained source path is sufficient to join
    both forms even after the original files have disappeared.
    """
    # Source paths are retained as diagnostics and may be spelled relative to
    # the watcher, while rollout paths are normally absolute.  Scope identity
    # must not depend on the resolving process: ``resolve()`` would join a
    # relative path against the calling process's CWD and follow every
    # existing symlink prefix, so the same durable ``raw_sessions.source_path``
    # would yield different scopes in the daemon and in the CLI and split the
    # state-to-rollout join.  Normalize lexically instead, which needs neither
    # the source to still exist nor a particular CWD-independent filesystem.
    path = Path(os.path.normpath(Path(source_path).expanduser()))
    if path.name in thread_state_member_filenames():
        return str(path.parent)
    for parent in path.parents:
        if parent.name == "sessions":
            return str(parent.parent)
    return str(path.parent)


#: The newest ``raw_payload`` receipt of raw ``r``: the durable order of one
#: observation of a retained export.
_RECEIPT = """
    SELECT b.{column}
    FROM blob_refs AS b
    WHERE b.ref_id = r.raw_id AND b.ref_type = 'raw_payload'
    ORDER BY b.rowid DESC
    LIMIT 1
"""


def retained_export_order(source_conn: sqlite3.Connection | None) -> agent_thread_state.ExportOrder:
    """Rank retained exports by raw id in durable receipt order.

    Without a source tier no export is ranked, so an older export arriving
    after a newer one only adds the rows the graph does not yet hold.
    """

    def order(raw_id: str) -> int | None:
        if source_conn is None:
            return None
        row = source_conn.execute(
            f"""
            SELECT COALESCE(({_RECEIPT.format(column="rowid")}), 0)
            FROM raw_sessions AS r
            WHERE r.raw_id = ?
            """,
            (raw_id,),
        ).fetchone()
        return None if row is None else int(row[0])

    return order


@dataclass(frozen=True, slots=True)
class _ThreadGraphRecords:
    records: Iterable[codex_state.CodexThreadRecord]

    def __iter__(self) -> Iterator[ThreadRecord]:
        for thread in self.records:
            yield ThreadRecord(
                thread_id=thread.thread_id,
                title=thread.title or None,
                occurred_at_ms=thread.updated_at_ms or thread.created_at_ms or None,
            )


@dataclass(frozen=True, slots=True)
class _SpawnGraphRecords:
    records: Iterable[codex_state.CodexSpawnEdge]

    def __iter__(self) -> Iterator[SpawnRecord]:
        for edge in self.records:
            yield SpawnRecord(
                parent_thread_id=edge.parent_thread_id,
                child_thread_id=edge.child_thread_id,
                status=edge.status or "unknown",
            )


def write_thread_state_projection(
    index_conn: sqlite3.Connection,
    snapshot: codex_state.CodexStateSnapshot,
    *,
    raw_id: str,
    blob_hash: str,
    observed_at_ms: int,
    observation_order: int = 0,
    source_scope: str = "",
    source_conn: sqlite3.Connection | None,
) -> bool:
    """Publish paged sealed state and re-decide changed children in bounded pages."""
    from polylogue.storage.sqlite.archive_tiers.write import rederive_codex_spawn_parent_links

    # This comparison belongs to the existing graph writer's TEMP schema.
    # It holds no source handle and never escapes this publication window.
    table = f"codex_projection_children_{uuid.uuid4().hex}"
    index_conn.execute(f"CREATE TEMP TABLE {table} (child TEXT PRIMARY KEY, prior_scope TEXT, prior_any TEXT)")
    primary: BaseException | None = None
    try:

        def remember(child: str) -> None:
            check_compute_cancelled()
            if not child or index_conn.execute(f"SELECT 1 FROM {table} WHERE child = ?", (child,)).fetchone():
                return
            prior_scope = agent_thread_state.read_parent_thread_id(index_conn, child, source_scope=source_scope)
            prior_any = agent_thread_state.read_parent_thread_id(index_conn, child)
            index_conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?)", (child, prior_scope, prior_any))

        graph_id = agent_thread_state.thread_state_graph_id(source_scope)
        after = ""
        while True:
            check_compute_cancelled()
            rows = index_conn.execute(
                "SELECT DISTINCT target_ref FROM work_evidence_edges WHERE graph_id = ? "
                "AND edge_kind = 'invoked' AND target_ref > ? ORDER BY target_ref LIMIT 256",
                (graph_id, after),
            ).fetchall()
            if not rows:
                break
            after = str(rows[-1][0])
            for (target,) in rows:
                remember(agent_thread_state.thread_id_from_context_ref(str(target)))
        for edge in snapshot.spawn_edges:
            remember(edge.child_thread_id)

        written = agent_thread_state.write_thread_state_graph(
            index_conn,
            source_scope=source_scope,
            threads=_ThreadGraphRecords(snapshot.threads),
            spawn_edges=_SpawnGraphRecords(snapshot.spawn_edges),
            raw_id=raw_id,
            blob_hash=blob_hash,
            observed_at_ms=observed_at_ms,
            observation_order=observation_order,
            export_order=retained_export_order(source_conn),
        )
        if written:
            after = ""
            while True:
                check_compute_cancelled()
                rows = index_conn.execute(
                    f"SELECT child, prior_scope, prior_any FROM {table} WHERE child > ? ORDER BY child LIMIT 256",
                    (after,),
                ).fetchall()
                if not rows:
                    break
                after = str(rows[-1][0])
                moved = [
                    str(child)
                    for child, prior_scope, prior_any in rows
                    if prior_scope
                    != agent_thread_state.read_parent_thread_id(index_conn, str(child), source_scope=source_scope)
                    or prior_any != agent_thread_state.read_parent_thread_id(index_conn, str(child))
                ]
                rederive_codex_spawn_parent_links(index_conn, moved, source_conn=source_conn)
        return written
    except BaseException as failure:
        primary = failure
        raise
    finally:
        try:
            index_conn.execute(f"DROP TABLE {table}")
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note(f"thread-state comparison cleanup failed: {cleanup!r}")


def apply_prepared_state_snapshot(
    archive: Any,
    raw_id: str,
    *,
    snapshot: codex_state.CodexStateSnapshot,
    blob_hash: str,
    observed_at_ms: int,
    source_path: str,
) -> bool:
    """Recompute the projection from one retained export, when the index is open.

    Returns whether the projection was written. Acquire-only ingestion holds
    no index handle at all, and that is not a failure: the export is durable,
    so the next pass with a derived tier recomputes from it.

    The newest durable ``raw_payload`` receipt orders this observation.
    ``observed_at_ms`` stands in only for a raw with no receipt row yet.
    """
    index_conn = archive.index_connection
    if index_conn is None:
        return False
    try:
        receipt_at_ms, receipt_order = archive.raw_revision_observation_order(raw_id)
    except KeyError:
        receipt_at_ms, receipt_order = observed_at_ms, 0
    return write_thread_state_projection(
        index_conn,
        snapshot,
        raw_id=raw_id,
        blob_hash=blob_hash,
        observed_at_ms=receipt_at_ms,
        observation_order=receipt_order,
        source_scope=codex_state_source_scope(source_path),
        source_conn=archive.source_connection,
    )


def read_thread_titles(
    index_conn: sqlite3.Connection,
    *,
    thread_ids: Sequence[str] | None = None,
    source_path: str | None = None,
) -> dict[str, str]:
    """Return ``{thread_id: title}`` from the projected thread-state graph."""
    return agent_thread_state.read_thread_titles(
        index_conn,
        thread_ids=thread_ids,
        source_scope=codex_state_source_scope(source_path) if source_path else None,
    )


def read_spawn_edges(index_conn: sqlite3.Connection, *, source_path: str | None = None) -> dict[tuple[str, str], str]:
    """Return ``{(parent_thread_id, child_thread_id): status}`` from the graph."""
    return agent_thread_state.read_spawn_edges(
        index_conn,
        source_scope=codex_state_source_scope(source_path) if source_path else None,
    )


def read_parent_thread_id(
    index_conn: sqlite3.Connection,
    child_thread_id: str,
    *,
    source_path: str | None = None,
    before_input: Callable[[str, tuple[str, ...], str, tuple[object, ...]], None] | None = None,
) -> str | None:
    """Return the projected parent of ``child_thread_id``, or ``None`` when silent.

    ``None`` means the graph is silent about this child, which is not the same
    as it naming a different parent; only the latter is a conflict.
    """
    return agent_thread_state.read_parent_thread_id(
        index_conn,
        child_thread_id,
        source_scope=codex_state_source_scope(source_path) if source_path else None,
        before_input=before_input,
    )


__all__ = [
    "THREAD_STATE_KIND",
    "apply_prepared_state_snapshot",
    "codex_state_source_scope",
    "read_parent_thread_id",
    "read_spawn_edges",
    "read_thread_titles",
    "retained_export_order",
    "thread_state_member_filenames",
    "write_thread_state_projection",
]
