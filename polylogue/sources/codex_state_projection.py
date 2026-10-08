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
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polylogue.storage.sqlite.agent_thread_state as agent_thread_state
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider
from polylogue.logging import get_logger
from polylogue.sources.parsers import codex_state
from polylogue.storage.io_phase_metrics import connection_cursor
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


def retained_export_order(source_read: SessionSourceRead | None) -> agent_thread_state.ExportOrder:
    """Rank the same selected retained exports; absent Source ranks no export."""

    def order(raw_id: str) -> int | None:
        return None if source_read is None else source_read.raw_export_order(raw_id)

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
    source_read: SessionSourceRead | None,
) -> bool:
    """Publish paged sealed state and re-decide changed children in bounded pages."""
    from polylogue.storage.sqlite.archive_tiers.write import rederive_codex_spawn_parent_links

    # This comparison belongs to the existing graph writer's TEMP schema.
    # It holds no source handle and never escapes this publication window.
    table = f"codex_projection_children_{uuid.uuid4().hex}"
    with connection_cursor(
        index_conn, f"CREATE TEMP TABLE {table} (child TEXT PRIMARY KEY, prior_scope TEXT, prior_any TEXT)"
    ):
        pass
    primary: BaseException | None = None
    try:

        def remember(child: str) -> None:
            check_compute_cancelled()
            if not child:
                return
            with connection_cursor(index_conn, f"SELECT 1 FROM {table} WHERE child = ?", (child,)) as rows:
                if rows.fetchone() is not None:
                    return
            prior_scope = agent_thread_state.read_parent_thread_id(index_conn, child, source_scope=source_scope)
            prior_any = agent_thread_state.read_parent_thread_id(index_conn, child)
            with connection_cursor(
                index_conn, f"INSERT INTO {table} VALUES (?, ?, ?)", (child, prior_scope, prior_any)
            ):
                pass

        graph_id = agent_thread_state.thread_state_graph_id(source_scope)
        after = ""
        while True:
            check_compute_cancelled()
            with connection_cursor(
                index_conn,
                "SELECT DISTINCT target_ref FROM work_evidence_edges WHERE graph_id = ? "
                "AND edge_kind = 'invoked' AND target_ref > ? ORDER BY target_ref LIMIT 256",
                (graph_id, after),
            ) as cursor:
                rows = cursor.fetchall()
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
            export_order=retained_export_order(source_read),
        )
        if written:
            after = ""
            while True:
                check_compute_cancelled()
                with connection_cursor(
                    index_conn,
                    f"SELECT child, prior_scope, prior_any FROM {table} WHERE child > ? ORDER BY child LIMIT 256",
                    (after,),
                ) as cursor:
                    rows = cursor.fetchall()
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
                rederive_codex_spawn_parent_links(index_conn, moved, source_read=source_read)
        return written
    except BaseException as failure:
        primary = failure
        raise
    finally:
        try:
            with connection_cursor(index_conn, f"DROP TABLE {table}"):
                pass
        except BaseException as cleanup:
            if primary is None:
                raise
            raise BaseExceptionGroup(
                "Thread-state projection and comparison cleanup failed", [primary, cleanup]
            ) from primary


def read_thread_titles(
    index_conn: sqlite3.Connection,
    *,
    thread_ids: Sequence[str] | None = None,
    source_path: str | None = None,
    before_input: Callable[[str, tuple[str, ...], str, tuple[object, ...]], None] | None = None,
) -> dict[str, str]:
    """Return ``{thread_id: title}`` from the projected thread-state graph."""
    return agent_thread_state.read_thread_titles(
        index_conn,
        thread_ids=thread_ids,
        source_scope=codex_state_source_scope(source_path) if source_path else None,
        before_input=before_input,
    )


def iter_thread_title_candidates(
    index_conn: sqlite3.Connection,
    *,
    thread_ids: Iterable[str] | None = None,
    source_path: str | None = None,
    before_input: Callable[[str, tuple[str, ...], str, tuple[object, ...]], None] | None = None,
) -> Generator[tuple[str, str], None, None]:
    """Borrow the canonical original title selection without collecting it."""
    yield from agent_thread_state.iter_thread_title_candidates(
        index_conn,
        thread_ids=thread_ids,
        source_scope=codex_state_source_scope(source_path) if source_path else None,
        before_input=before_input,
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
    row_read: agent_thread_state.ThreadParentRowRead | None = None,
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
        row_read=row_read,
    )


__all__ = [
    "THREAD_STATE_KIND",
    "codex_state_source_scope",
    "read_parent_thread_id",
    "read_spawn_edges",
    "read_thread_titles",
    "retained_export_order",
    "thread_state_member_filenames",
    "write_thread_state_projection",
]


@dataclass(slots=True)
class PreparedThreadStateProjection:
    """One graph and reachable link operand carrier on the original witness."""

    graph: agent_thread_state.PreparedThreadStateGraph
    links: PreparedCodexLinkInputs
    cohort: PreparedThreadStateCohort | None = None

    def close(self) -> None:
        self.graph.close()

    def apply(self, connection: sqlite3.Connection) -> bool:
        if self.cohort is None:
            raise RuntimeError("thread projection has no completed original cohort")
        self.cohort.apply(connection)
        return self.graph.written


def prepare_thread_state_projection(
    seal: PreparedIndexMutation,
    snapshot: codex_state.CodexStateSnapshot,
    *,
    directory: Path,
    raw_id: str,
    blob_hash: str,
    observed_at_ms: int,
    observation_order: int,
    source_scope: str,
    source_read: SessionSourceRead,
) -> PreparedThreadStateProjection:
    """Prepare the canonical graph decisions and every possible link input."""
    from polylogue.storage.sqlite.archive_tiers.write import prepare_codex_link_inputs

    original = seal.observer("index")
    graph = agent_thread_state.prepare_thread_state_graph(
        seal,
        directory=directory,
        source_scope=source_scope,
        threads=_ThreadGraphRecords(snapshot.threads),
        spawn_edges=_SpawnGraphRecords(snapshot.spawn_edges),
        raw_id=raw_id,
        blob_hash=blob_hash,
        observed_at_ms=observed_at_ms,
        observation_order=observation_order,
        export_order=retained_export_order(source_read),
    )
    scratch = graph.owner.require_connection()
    try:
        for sql in (
            "CREATE TABLE codex_projection_children(child TEXT PRIMARY KEY,prior_scope TEXT,prior_any TEXT)",
            "CREATE TABLE codex_projection_original_edges(physical_rowid INTEGER PRIMARY KEY,source_ref TEXT)",
            "CREATE TABLE codex_projection_produced_edges(physical_rowid INTEGER PRIMARY KEY,source_ref TEXT)",
            "CREATE TABLE codex_projection_parents(native_id TEXT PRIMARY KEY)",
        ):
            with connection_cursor(scratch, sql):
                pass

        def remember(child: str) -> None:
            check_compute_cancelled()
            if not child:
                return
            with connection_cursor(
                scratch, "SELECT 1 FROM codex_projection_children WHERE child=?", (child,)
            ) as cursor:
                if cursor.fetchone() is not None:
                    return
            prior_scope = agent_thread_state.read_parent_thread_id(
                original,
                child,
                source_scope=source_scope,
                before_input=seal.before_index_input,
            )
            prior_any = agent_thread_state.read_parent_thread_id(original, child, before_input=seal.before_index_input)
            with connection_cursor(
                scratch, "INSERT INTO codex_projection_children VALUES(?,?,?)", (child, prior_scope, prior_any)
            ):
                pass

        graph_id = agent_thread_state.thread_state_graph_id(source_scope)
        after = 0
        while True:
            with connection_cursor(
                original,
                "SELECT rowid FROM work_evidence_edges WHERE graph_id=? AND edge_kind='invoked' "
                "AND rowid>? ORDER BY rowid LIMIT 256",
                (graph_id, after),
            ) as cursor:
                page = [int(row[0]) for row in cursor]
            if not page:
                break
            after = page[-1]
            for rowid in page:
                seal.before_index_input(
                    "work_evidence_edges",
                    ("graph_id", "edge_kind", "target_ref"),
                    "SELECT rowid FROM work_evidence_edges WHERE rowid=?",
                    (rowid,),
                )
                with connection_cursor(
                    original, "SELECT target_ref FROM work_evidence_edges WHERE rowid=?", (rowid,)
                ) as cursor:
                    row = cursor.fetchone()
                if row is None:
                    raise RuntimeError("original thread edge disappeared inside its pinned preparation")
                remember(agent_thread_state.thread_id_from_context_ref(str(row[0])))
        for edge in snapshot.spawn_edges:
            remember(edge.child_thread_id)
            with connection_cursor(
                scratch, "INSERT OR IGNORE INTO codex_projection_parents VALUES(?)", (edge.parent_thread_id.strip(),)
            ):
                pass

        after_child = ""
        while True:
            with connection_cursor(
                scratch,
                "SELECT child FROM codex_projection_children WHERE child>? ORDER BY child LIMIT 256",
                (after_child,),
            ) as cursor:
                children = [str(row[0]) for row in cursor]
            if not children:
                break
            after_child = children[-1]
            for child in children:
                with closing(agent_thread_state.parent_thread_candidate_coordinates(original, child)) as coordinates:
                    for edge_rowid, graph_rowid in coordinates:
                        seal.before_index_input(
                            "work_evidence_edges",
                            ("graph_id", "edge_kind", "target_ref", "association_state", "source_ref"),
                            "SELECT rowid FROM work_evidence_edges WHERE rowid=?",
                            (edge_rowid,),
                        )
                        seal.before_index_input(
                            "work_evidence_graphs",
                            ("graph_id", "observation_order"),
                            "SELECT rowid FROM work_evidence_graphs WHERE rowid=?",
                            (graph_rowid,),
                        )
                        with connection_cursor(
                            original, "SELECT source_ref FROM work_evidence_edges WHERE rowid=?", (edge_rowid,)
                        ) as cursor:
                            row = cursor.fetchone()
                        if row is None:
                            raise RuntimeError("original graph candidate disappeared inside its pinned preparation")
                        with connection_cursor(
                            scratch,
                            "INSERT OR IGNORE INTO codex_projection_original_edges VALUES(?,?)",
                            (edge_rowid, row[0]),
                        ):
                            pass
                        parent = agent_thread_state.thread_id_from_context_ref(str(row[0])).strip()
                        if parent:
                            with connection_cursor(
                                scratch, "INSERT OR IGNORE INTO codex_projection_parents VALUES(?)", (parent,)
                            ):
                                pass

        def identifiers(table: str, column: str) -> Generator[str, None, None]:
            with connection_cursor(scratch, f"SELECT {column} FROM {table} ORDER BY {column}") as cursor:
                for (value,) in cursor:
                    yield str(value)

        with (
            closing(identifiers("codex_projection_children", "child")) as selected_children,
            closing(identifiers("codex_projection_parents", "native_id")) as selected_parents,
        ):
            links = prepare_codex_link_inputs(
                seal,
                owner=graph.owner,
                child_native_ids=selected_children,
                parent_native_ids=selected_parents,
                source_read=source_read,
            )
        scratch.commit()
        return PreparedThreadStateProjection(graph, links)
    except BaseException as primary:
        try:
            graph.close()
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "thread projection preparation and physical cleanup failed", [primary, cleanup]
            ) from None
        raise


@dataclass(slots=True)
class PreparedThreadStateCohort:
    """One final graph/link postimage on the original graph scratch custody."""

    projections: tuple[PreparedThreadStateProjection, ...]
    applied: bool = False

    @property
    def first(self) -> PreparedThreadStateProjection:
        return self.projections[0]

    def source_ref(self, physical_rowid: int) -> str:
        scratch = self.first.graph.owner.require_connection()
        with connection_cursor(
            scratch,
            "SELECT source_ref FROM codex_projection_produced_edges WHERE physical_rowid=?",
            (physical_rowid,),
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            with connection_cursor(
                scratch,
                "SELECT source_ref FROM codex_projection_original_edges WHERE physical_rowid=?",
                (physical_rowid,),
            ) as cursor:
                row = cursor.fetchone()
        if row is None:
            raise RuntimeError("cohort parent selector reached an unprepared graph coordinate")
        return str(row[0])

    def apply(self, connection: sqlite3.Connection) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import TABLE_SPECS
        from polylogue.storage.sqlite.archive_tiers.write import (
            PreparedCodexSpawnParentLinks,
            rederive_codex_spawn_parent_links,
        )
        from polylogue.storage.sqlite.reference_seal import ReferenceSealError, current_index_mutation_scope

        if self.applied:
            return
        first = self.first
        scope = current_index_mutation_scope()
        if scope is None or scope.seal is not first.graph.reference_seal:
            raise ReferenceSealError("prepared graph cohort requires its original admitted Index scope")
        scope.require_new_work(connection)
        scratch = first.graph.owner.require_connection()
        # Every selected scope's complete postimage precedes any shared parent
        # decision. Superseded rows remain part of that postimage.
        for table, keys in (
            ("work_evidence_graphs", ("graph_id",)),
            ("work_evidence_nodes", ("graph_id", "node_ref")),
            ("work_evidence_edges", ("graph_id", "edge_ref")),
        ):
            columns = tuple(column.name for column in TABLE_SPECS[table].writable_columns)
            names = ",".join(columns)
            updates = ",".join(f"{column}=excluded.{column}" for column in columns if column not in keys)
            sql = (
                f"INSERT INTO {table}({names}) VALUES({','.join('?' for _ in columns)}) "
                f"ON CONFLICT({','.join(keys)}) DO UPDATE SET {updates}"
            )
            selected_names = ",".join(f"held.{column}" for column in columns)
            order = ",".join(f"held.{key}" for key in keys)
            with connection_cursor(
                scratch,
                f"SELECT {selected_names} FROM {table} held JOIN codex_projection_cohort_scopes selected "
                f"ON held.graph_id=selected.graph_id WHERE selected.written=1 ORDER BY {order}",
            ) as rows:
                agent_thread_state._write_graph_rows(connection, sql, rows)

        with connection_cursor(
            scratch,
            "SELECT held.graph_id,held.edge_ref,held.source_ref FROM work_evidence_edges held "
            "JOIN codex_projection_cohort_scopes selected ON held.graph_id=selected.graph_id "
            "WHERE selected.written=1 AND held.edge_kind='invoked' ORDER BY held.graph_id,held.edge_ref",
        ) as rows:
            for graph_id, edge_ref, source_ref in rows:
                check_compute_cancelled()
                with connection_cursor(
                    connection,
                    "SELECT rowid FROM work_evidence_edges WHERE graph_id=? AND edge_ref=?",
                    (graph_id, edge_ref),
                ) as cursor:
                    selected = cursor.fetchone()
                if selected is None:
                    raise RuntimeError("published cohort edge has no physical postimage")
                with connection_cursor(
                    scratch,
                    "INSERT OR REPLACE INTO codex_projection_produced_edges VALUES(?,?)",
                    (selected[0], source_ref),
                ):
                    pass

        links = PreparedCodexSpawnParentLinks(first.links, connection)
        after = ""
        while True:
            with connection_cursor(
                scratch,
                "SELECT DISTINCT child FROM codex_projection_cohort_children WHERE child>? ORDER BY child LIMIT 256",
                (after,),
            ) as cursor:
                page = tuple(str(row[0]) for row in cursor)
            if not page:
                break
            after = page[-1]
            moved: list[str] = []
            for child in page:
                check_compute_cancelled()
                with connection_cursor(
                    scratch,
                    "SELECT source_scope,prior_scope,prior_any FROM codex_projection_cohort_children "
                    "WHERE child=? ORDER BY source_scope",
                    (child,),
                ) as rows:
                    for source_scope, prior_scope, prior_any in rows:
                        if prior_scope != agent_thread_state.read_parent_thread_id(
                            connection, child, source_scope=source_scope, row_read=self
                        ) or prior_any != agent_thread_state.read_parent_thread_id(connection, child, row_read=self):
                            moved.append(child)
                            break
            rederive_codex_spawn_parent_links(connection, moved, source_read=None, prepared=links, graph_read=self)
        self.applied = True


def prepare_thread_state_cohort(
    projections: tuple[PreparedThreadStateProjection, ...], *, source_read: SessionSourceRead
) -> PreparedThreadStateCohort | None:
    """Fold prepared snapshots with the existing complete-snapshot reducer."""
    from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import TABLE_SPECS

    if not projections:
        return None
    first = projections[0]
    seal = first.graph.reference_seal
    scratch = first.graph.owner.require_connection()
    original = seal.observer("index")
    for projection in projections:
        if projection.graph.reference_seal is not seal:
            raise RuntimeError("thread cohort cannot combine different original witnesses")
    for table in ("work_evidence_graphs", "work_evidence_nodes", "work_evidence_edges"):
        with connection_cursor(scratch, f"CREATE TABLE {table}({TABLE_SPECS[table].ddl_body}) STRICT"):
            pass
    with connection_cursor(
        scratch,
        "CREATE TABLE codex_projection_cohort_scopes(source_scope TEXT PRIMARY KEY,graph_id TEXT UNIQUE,"
        "written INTEGER NOT NULL DEFAULT 0)",
    ):
        pass
    with connection_cursor(
        scratch,
        "CREATE TABLE codex_projection_cohort_children(child TEXT,source_scope TEXT,prior_scope TEXT,"
        "prior_any TEXT,PRIMARY KEY(child,source_scope))",
    ):
        pass

    for projection in projections:
        graph = projection.graph
        with connection_cursor(
            scratch, "SELECT 1 FROM codex_projection_cohort_scopes WHERE source_scope=?", (graph.source_scope,)
        ) as cursor:
            seeded = cursor.fetchone() is not None
        if not seeded:
            graph_id = agent_thread_state.thread_state_graph_id(graph.source_scope)
            for table in ("work_evidence_graphs", "work_evidence_nodes", "work_evidence_edges"):
                columns = tuple(column.name for column in TABLE_SPECS[table].writable_columns)
                names = ",".join(columns)
                after = 0
                while True:
                    with connection_cursor(
                        original,
                        f"SELECT rowid FROM {table} WHERE graph_id=? AND rowid>? ORDER BY rowid LIMIT 256",
                        (graph_id, after),
                    ) as cursor:
                        page = tuple(int(row[0]) for row in cursor)
                    if not page:
                        break
                    after = page[-1]
                    for rowid in page:
                        check_compute_cancelled()
                        seal.before_index_input(table, columns, f"SELECT rowid FROM {table} WHERE rowid=?", (rowid,))
                        with connection_cursor(
                            original, f"SELECT {names} FROM {table} WHERE rowid=?", (rowid,)
                        ) as cursor:
                            row = cursor.fetchone()
                        if row is None:
                            raise RuntimeError("original scope graph disappeared inside its pinned read")
                        with connection_cursor(
                            scratch,
                            f"INSERT INTO {table}({names}) VALUES({','.join('?' for _ in columns)})",
                            tuple(row),
                        ):
                            pass
            with connection_cursor(
                scratch,
                "INSERT INTO codex_projection_cohort_scopes(source_scope,graph_id) VALUES(?,?)",
                (graph.source_scope, graph_id),
            ):
                pass

        donor = graph.owner.require_connection()
        with connection_cursor(donor, "SELECT child,prior_scope,prior_any FROM codex_projection_children") as rows:
            for child, prior_scope, prior_any in rows:
                with connection_cursor(
                    scratch,
                    "INSERT OR IGNORE INTO codex_projection_cohort_children VALUES(?,?,?,?)",
                    (child, graph.source_scope, prior_scope, prior_any),
                ):
                    pass
        if projection is first:
            continue
        for table in (
            "codex_projection_original_edges",
            "codex_link_input_session_queue",
            "codex_link_input_targets",
            "codex_link_input_sessions",
            "codex_link_input_claims",
            "codex_link_input_links",
        ):
            with connection_cursor(scratch, f"PRAGMA table_info({table})") as cursor:
                primary_keys = tuple((int(row[5]), int(row[0]), str(row[1])) for row in cursor if row[5])
            primary_keys = tuple(sorted(primary_keys))
            if not primary_keys:
                raise RuntimeError("prepared cohort input table has no declared identity")
            with connection_cursor(donor, f"SELECT * FROM {table}") as rows:
                for row in rows:
                    check_compute_cancelled()
                    with connection_cursor(
                        scratch, f"INSERT OR IGNORE INTO {table} VALUES({','.join('?' for _ in row)})", tuple(row)
                    ):
                        pass
                    predicate = " AND ".join(f"{name} IS ?" for _order, _index, name in primary_keys)
                    parameters = tuple(row[index] for _order, index, _name in primary_keys)
                    with connection_cursor(scratch, f"SELECT * FROM {table} WHERE {predicate}", parameters) as cursor:
                        held = cursor.fetchone()
                    if held is None or tuple(held) != tuple(row):
                        raise RuntimeError("cohort inputs disagree on one original retained coordinate")

    # Ordering is over retained header operands, not session/document bodies.
    for projection in sorted(
        projections, key=lambda item: (item.graph.observation_order, item.graph.raw_id, item.graph.blob_hash)
    ):
        graph = projection.graph
        current = agent_thread_state.read_provenance(scratch, source_scope=graph.source_scope)
        incoming = (graph.observation_order, graph.raw_id, graph.blob_hash)
        graph_id = agent_thread_state.thread_state_graph_id(graph.source_scope)
        with closing(graph.rows("node")) as nodes, closing(graph.rows("edge")) as edges:
            if current is not None and (current.observation_order, current.raw_id, current.blob_hash) > incoming:
                graph.written = agent_thread_state._retain_older_export_rows(
                    scratch,
                    graph_id,
                    nodes,
                    edges,
                    incoming_key=incoming,
                    export_order=retained_export_order(source_read),
                )
            else:
                agent_thread_state._apply_current_thread_state_graph(
                    scratch,
                    graph_id,
                    nodes,
                    edges,
                    raw_id=graph.raw_id,
                    blob_hash=graph.blob_hash,
                    observed_at_ms=graph.observed_at_ms,
                    observation_order=graph.observation_order,
                )
                graph.written = True
        if graph.written:
            with connection_cursor(
                scratch,
                "UPDATE codex_projection_cohort_scopes SET written=1 WHERE source_scope=?",
                (graph.source_scope,),
            ):
                pass
    scratch.commit()
    cohort = PreparedThreadStateCohort(projections)
    for projection in projections:
        projection.cohort = cohort
    return cohort


if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.write import PreparedCodexLinkInputs, SessionSourceRead
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
