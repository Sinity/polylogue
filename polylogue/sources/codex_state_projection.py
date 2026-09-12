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

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.core.enums import Origin, Provider
from polylogue.logging import get_logger
from polylogue.sources.parsers import codex_state

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


@dataclass(frozen=True, slots=True)
class RetainedStateExport:
    """The newest retained state export for one Codex install."""

    raw_id: str
    blob_hash: str
    source_scope: str
    observed_at_ms: int
    observation_order: int


def codex_state_source_scope(source_path: str) -> str:
    """Return the Codex-install scope shared by state and rollout evidence.

    ``state_5.sqlite`` sits directly in the install directory while rollouts
    live below ``sessions/``.  The retained source path is sufficient to join
    both forms even after the original files have disappeared.
    """
    # Source paths are retained as diagnostics and may be spelled relative to
    # the watcher, while rollout paths are normally absolute.  Scope identity
    # must not depend on that presentation detail: normalize the path
    # lexically (without requiring the source to still exist) before joining
    # state exports to rollout evidence.
    path = Path(source_path).expanduser().resolve(strict=False)
    if path.name in thread_state_member_filenames():
        return str(path.parent)
    for parent in path.parents:
        if parent.name == "sessions":
            return str(parent.parent)
    return str(path.parent)


def latest_retained_state_exports(source_conn: sqlite3.Connection) -> tuple[RetainedStateExport, ...]:
    """Return the newest retained state export per Codex-install scope.

    Ordered exactly as ``raw_revision_observation_order`` orders one raw --
    newest ``raw_payload`` receipt first -- so a reconciliation pass and a
    per-export apply never disagree about which observation is current. A live
    database that went A -> B -> A reuses A's content-derived raw id, so the
    receipt log, not ``raw_sessions.acquired_at_ms``, is the authority.

    Scans the durable tier, so callers reconcile once per pass rather than
    once per session.
    """
    filenames = thread_state_member_filenames()
    if not filenames:
        return ()
    clauses = " OR ".join("source_path = ? OR source_path LIKE ?" for _ in filenames)
    parameters: list[str] = [Origin.CODEX_SESSION.value]
    for filename in filenames:
        parameters.extend((filename, f"%/{filename}"))
    receipt = """
        SELECT b.{column}
        FROM blob_refs AS b
        WHERE b.ref_id = r.raw_id AND b.ref_type = 'raw_payload'
        ORDER BY b.acquired_at_ms DESC, b.rowid DESC
        LIMIT 1
    """
    try:
        rows = source_conn.execute(
            f"""
            SELECT
                r.raw_id,
                lower(hex(r.blob_hash)),
                r.source_path,
                COALESCE(({receipt.format(column="acquired_at_ms")}), r.acquired_at_ms),
                COALESCE(({receipt.format(column="rowid")}), r.rowid)
            FROM raw_sessions AS r
            WHERE r.origin = ? AND r.parse_error IS NULL AND ({clauses})
            ORDER BY 4 DESC, 5 DESC, r.raw_id DESC
            """,
            parameters,
        ).fetchall()
    except sqlite3.Error as exc:
        logger.debug("Failed to read the newest retained Codex state export: %s", exc)
        return ()
    newest: dict[str, RetainedStateExport] = {}
    for raw_id, blob_hash, source_path, observed_at_ms, observation_order in rows:
        source_scope = codex_state_source_scope(str(source_path))
        if source_scope in newest:
            continue
        newest[source_scope] = RetainedStateExport(
            str(raw_id), str(blob_hash), source_scope, int(observed_at_ms), int(observation_order)
        )
    return tuple(newest[scope] for scope in sorted(newest))


def latest_retained_state_export(source_conn: sqlite3.Connection) -> RetainedStateExport | None:
    """Return the newest export archive-wide for legacy diagnostic callers.

    Projection reconciliation must use :func:`latest_retained_state_exports`.
    This compatibility helper deliberately has no projection semantics.
    """
    exports = latest_retained_state_exports(source_conn)
    if not exports:
        return None
    return max(exports, key=lambda item: (item.observed_at_ms, item.observation_order, item.raw_id, item.source_scope))


@dataclass(frozen=True, slots=True)
class ProjectionProvenance:
    """Which retained export the current projection was computed from."""

    raw_id: str
    blob_hash: str
    observed_at_ms: int
    observation_order: int


def projection_provenance(
    index_conn: sqlite3.Connection, *, source_scope: str | None = None
) -> ProjectionProvenance | None:
    """Return the current projection provenance, optionally for one scope."""
    try:
        if source_scope is None:
            row = index_conn.execute(
                "SELECT raw_id, blob_hash, observed_at_ms, observation_order "
                "FROM codex_thread_state_provenance "
                "ORDER BY observed_at_ms DESC, observation_order DESC, raw_id DESC, source_scope DESC LIMIT 1"
            ).fetchone()
        else:
            row = index_conn.execute(
                "SELECT raw_id, blob_hash, observed_at_ms, observation_order "
                "FROM codex_thread_state_provenance WHERE source_scope = ?",
                (source_scope,),
            ).fetchone()
    except sqlite3.Error as exc:
        logger.debug("Failed to read the Codex thread-state projection provenance: %s", exc)
        return None
    if row is None:
        return None
    return ProjectionProvenance(str(row[0]), str(row[1]), int(row[2]), int(row[3]))


def write_thread_state_projection(
    index_conn: sqlite3.Connection,
    snapshot: codex_state.CodexStateSnapshot,
    *,
    raw_id: str,
    blob_hash: str,
    observed_at_ms: int,
    observation_order: int = 0,
    source_scope: str = "",
) -> bool:
    """Reconcile one scope from the content of one retained export.

    A snapshot is complete only for its own scope.  It therefore marks older
    objects in that scope absent, rather than deleting them: the retained raw
    remains recoverable evidence and another Codex root is untouched.

    An export older than the one already projected is skipped and reported as
    ``False``: replay applies raws in no particular order, and a live database
    that went A -> B -> A reuses A's content-derived raw id, so the durable
    receipt order is what says which observation is current.
    """
    current = projection_provenance(index_conn, source_scope=source_scope)
    # Receipt timestamps and rowids are normally unique, but callers can
    # legitimately replay synthetic receipts with equal ordering fields.
    # Include the content identity as a final tie-break so equal-key replay is
    # deterministic rather than dependent on which raw arrived first.
    incoming_key = (observed_at_ms, observation_order, raw_id, blob_hash)
    if (
        current is not None
        and (
            current.observed_at_ms,
            current.observation_order,
            current.raw_id,
            current.blob_hash,
        )
        > incoming_key
    ):
        return False
    index_conn.execute(
        "UPDATE codex_thread_state SET source_present = 0 WHERE source_scope = ?",
        (source_scope,),
    )
    index_conn.execute(
        "UPDATE codex_thread_spawn_edges SET source_present = 0 WHERE source_scope = ?",
        (source_scope,),
    )
    index_conn.executemany(
        """
        INSERT INTO codex_thread_state (
            source_scope, thread_id, title, cwd, created_at_ms, updated_at_ms, source,
            model, agent_nickname, agent_role, archived, observed_at_ms, observation_order, source_present
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_scope, thread_id) DO UPDATE SET
            title = excluded.title,
            cwd = excluded.cwd,
            created_at_ms = excluded.created_at_ms,
            updated_at_ms = excluded.updated_at_ms,
            source = excluded.source,
            model = excluded.model,
            agent_nickname = excluded.agent_nickname,
            agent_role = excluded.agent_role,
            archived = excluded.archived,
            observed_at_ms = excluded.observed_at_ms,
            observation_order = excluded.observation_order,
            source_present = excluded.source_present
        """,
        [
            (
                source_scope,
                thread.thread_id,
                thread.title or None,
                thread.cwd or None,
                thread.created_at_ms,
                thread.updated_at_ms,
                thread.source or None,
                thread.model,
                thread.agent_nickname,
                thread.agent_role,
                1 if thread.archived else 0,
                observed_at_ms,
                observation_order,
                1,
            )
            for thread in snapshot.threads
        ],
    )
    index_conn.executemany(
        """
        INSERT INTO codex_thread_spawn_edges (
            source_scope, parent_thread_id, child_thread_id, status, observed_at_ms, observation_order, source_present
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_scope, parent_thread_id, child_thread_id) DO UPDATE SET
            status = excluded.status,
            observed_at_ms = excluded.observed_at_ms,
            observation_order = excluded.observation_order,
            source_present = excluded.source_present
        """,
        [
            (
                source_scope,
                edge.parent_thread_id,
                edge.child_thread_id,
                edge.status or "unknown",
                observed_at_ms,
                observation_order,
                1,
            )
            for edge in snapshot.spawn_edges
        ],
    )
    index_conn.execute(
        """
        INSERT INTO codex_thread_state_provenance (
            source_scope, raw_id, blob_hash, observed_at_ms, observation_order
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source_scope) DO UPDATE SET
            raw_id = excluded.raw_id,
            blob_hash = excluded.blob_hash,
            observed_at_ms = excluded.observed_at_ms,
            observation_order = excluded.observation_order
        """,
        (source_scope, raw_id, blob_hash, observed_at_ms, observation_order),
    )
    return True


def apply_retained_state_export(
    archive: Any,
    raw_id: str,
    *,
    export_path: Path,
    blob_hash: str,
    observed_at_ms: int,
    source_path: str,
) -> bool:
    """Recompute the projection from one retained export, when the index is open.

    Returns whether the projection was written. Acquire-only ingestion holds
    no index handle at all, and that is not a failure: the export is durable,
    so the next pass with a derived tier recomputes from it.

    The durable ``raw_payload`` receipt orders this observation, the same term
    :func:`latest_retained_state_export` reads, so the two routes never
    disagree about which export is current. ``observed_at_ms`` stands in only
    for a raw with no receipt row yet.
    """
    index_conn = archive.index_connection
    if index_conn is None:
        return False
    snapshot = codex_state.parse_codex_state_db(export_path, immutable=True)
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
    )


def ensure_thread_state_projection(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    *,
    blob_path_for_hash: Callable[[str], Path | None],
) -> bool:
    """Reconcile the projection against the newest retained state export.

    Replay applies raws in no particular order, so a session can be written
    before the state export it describes. This derives the projection from
    durable evidence instead of from replay order; it is a no-op once the
    projection already names the newest export.
    """
    latest_exports = latest_retained_state_exports(source_conn)
    if not latest_exports:
        return False
    changed = False
    for latest in latest_exports:
        current = projection_provenance(index_conn, source_scope=latest.source_scope)
        if (
            current is not None
            and current.raw_id == latest.raw_id
            and current.blob_hash == latest.blob_hash
            and (current.observed_at_ms, current.observation_order) == (latest.observed_at_ms, latest.observation_order)
        ):
            continue
        export_path = blob_path_for_hash(latest.blob_hash)
        if export_path is None or not Path(export_path).is_file():
            continue
        try:
            snapshot = codex_state.parse_codex_state_db(Path(export_path), immutable=True)
        except sqlite3.Error as exc:
            logger.warning("codex state: retained export %s is not readable as thread state: %s", latest.raw_id, exc)
            continue
        changed = (
            write_thread_state_projection(
                index_conn,
                snapshot,
                raw_id=latest.raw_id,
                blob_hash=latest.blob_hash,
                observed_at_ms=latest.observed_at_ms,
                observation_order=latest.observation_order,
                source_scope=latest.source_scope,
            )
            or changed
        )
    return changed


def read_thread_titles(
    index_conn: sqlite3.Connection,
    *,
    thread_ids: Sequence[str] | None = None,
    source_path: str | None = None,
) -> dict[str, str]:
    """Return ``{thread_id: title}`` from the projected Codex thread state.

    Any failure (missing table, locked file) degrades to an empty mapping,
    matching every other sidecar source in the title ladder.
    """
    titles: dict[str, str] = {}
    try:
        source_scope = codex_state_source_scope(source_path) if source_path else None
        if thread_ids is None:
            if source_scope is None:
                rows = index_conn.execute(
                    "SELECT thread_id, title FROM codex_thread_state WHERE title IS NOT NULL "
                    "ORDER BY observed_at_ms DESC, observation_order DESC, source_scope DESC"
                ).fetchall()
            else:
                rows = index_conn.execute(
                    "SELECT thread_id, title FROM codex_thread_state WHERE title IS NOT NULL AND source_scope = ?",
                    (source_scope,),
                ).fetchall()
        else:
            wanted = list(dict.fromkeys(thread_ids))
            if not wanted:
                return {}
            rows = []
            for start in range(0, len(wanted), 500):
                chunk = wanted[start : start + 500]
                placeholders = ", ".join("?" for _ in chunk)
                if source_scope is None:
                    rows.extend(
                        index_conn.execute(
                            "SELECT thread_id, title FROM codex_thread_state "
                            f"WHERE title IS NOT NULL AND thread_id IN ({placeholders}) "
                            "ORDER BY observed_at_ms DESC, observation_order DESC, source_scope DESC",
                            chunk,
                        ).fetchall()
                    )
                else:
                    rows.extend(
                        index_conn.execute(
                            "SELECT thread_id, title FROM codex_thread_state "
                            f"WHERE title IS NOT NULL AND source_scope = ? AND thread_id IN ({placeholders})",
                            [source_scope, *chunk],
                        ).fetchall()
                    )
    except sqlite3.Error as exc:
        logger.debug("Failed to read projected Codex thread titles: %s", exc)
        return {}
    for row in rows:
        thread_id, title = str(row[0]), row[1]
        if thread_id and thread_id not in titles and isinstance(title, str) and title.strip():
            titles[thread_id] = title.strip()
    return titles


def read_spawn_edges(index_conn: sqlite3.Connection, *, source_path: str | None = None) -> dict[tuple[str, str], str]:
    """Return ``{(parent_thread_id, child_thread_id): status}`` from the projection."""
    try:
        source_scope = codex_state_source_scope(source_path) if source_path else None
        if source_scope is None:
            rows = index_conn.execute(
                "SELECT parent_thread_id, child_thread_id, status FROM codex_thread_spawn_edges "
                "ORDER BY observed_at_ms DESC, observation_order DESC, source_scope DESC"
            ).fetchall()
        else:
            rows = index_conn.execute(
                "SELECT parent_thread_id, child_thread_id, status FROM codex_thread_spawn_edges WHERE source_scope = ?",
                (source_scope,),
            ).fetchall()
    except sqlite3.Error as exc:
        logger.debug("Failed to read projected Codex spawn edges: %s", exc)
        return {}
    edges: dict[tuple[str, str], str] = {}
    for row in rows:
        edges.setdefault((str(row[0]), str(row[1])), str(row[2]) or "unknown")
    return edges


def read_parent_thread_id(
    index_conn: sqlite3.Connection, child_thread_id: str, *, source_path: str | None = None
) -> str | None:
    """Return the projected parent of ``child_thread_id``, or ``None`` when silent.

    ``None`` means the projection is silent about this child, which is not the
    same as it naming a different parent; only the latter is a conflict.
    """
    if not child_thread_id:
        return None
    try:
        source_scope = codex_state_source_scope(source_path) if source_path else None
        if source_scope is None:
            row = index_conn.execute(
                """
                SELECT parent_thread_id
                FROM codex_thread_spawn_edges
                WHERE child_thread_id = ?
                ORDER BY observed_at_ms DESC, observation_order DESC, source_scope DESC, parent_thread_id
                LIMIT 1
                """,
                (child_thread_id,),
            ).fetchone()
        else:
            row = index_conn.execute(
                """
                SELECT parent_thread_id
                FROM codex_thread_spawn_edges
                WHERE source_scope = ? AND child_thread_id = ?
                ORDER BY observed_at_ms DESC, observation_order DESC, parent_thread_id
                LIMIT 1
                """,
                (source_scope, child_thread_id),
            ).fetchone()
    except sqlite3.Error as exc:
        logger.debug("Failed to read the projected Codex spawn-edge parent: %s", exc)
        return None
    if row is None or row[0] is None:
        return None
    parent = str(row[0]).strip()
    return parent or None


__all__ = [
    "THREAD_STATE_KIND",
    "ProjectionProvenance",
    "RetainedStateExport",
    "apply_retained_state_export",
    "codex_state_source_scope",
    "ensure_thread_state_projection",
    "latest_retained_state_export",
    "latest_retained_state_exports",
    "projection_provenance",
    "read_parent_thread_id",
    "read_spawn_edges",
    "read_thread_titles",
    "thread_state_member_filenames",
    "write_thread_state_projection",
]
