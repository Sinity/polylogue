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


def latest_retained_state_export(source_conn: sqlite3.Connection) -> tuple[str, str, int] | None:
    """Return ``(raw_id, blob_hash, acquired_at_ms)`` of the newest state export.

    Scans the durable tier, so callers reconcile once per pass rather than
    once per session.
    """
    filenames = thread_state_member_filenames()
    if not filenames:
        return None
    clauses = " OR ".join("source_path = ? OR source_path LIKE ?" for _ in filenames)
    parameters: list[str] = [Origin.CODEX_SESSION.value]
    for filename in filenames:
        parameters.extend((filename, f"%/{filename}"))
    row = source_conn.execute(
        f"""
        SELECT raw_id, lower(hex(blob_hash)), acquired_at_ms
        FROM raw_sessions
        WHERE origin = ? AND parse_error IS NULL AND ({clauses})
        ORDER BY acquired_at_ms DESC, raw_id DESC
        LIMIT 1
        """,
        parameters,
    ).fetchone()
    if row is None:
        return None
    return str(row[0]), str(row[1]), int(row[2])


@dataclass(frozen=True, slots=True)
class ProjectionProvenance:
    """Which retained export the current projection was computed from."""

    raw_id: str
    blob_hash: str
    observed_at_ms: int
    observation_order: int


def projection_provenance(index_conn: sqlite3.Connection) -> ProjectionProvenance | None:
    """Return which retained export the current projection was computed from."""
    try:
        row = index_conn.execute(
            "SELECT raw_id, blob_hash, observed_at_ms, observation_order "
            "FROM codex_thread_state_provenance WHERE singleton = 0"
        ).fetchone()
    except sqlite3.Error:
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
) -> bool:
    """Replace the projection with the content of one retained export.

    A whole-table replace is the honest shape: the export states the complete
    thread set as of its revision, so a thread Codex deleted must leave the
    projection too.

    An export older than the one already projected is skipped and reported as
    ``False``: replay applies raws in no particular order, and a live database
    that went A -> B -> A reuses A's content-derived raw id, so the durable
    receipt order is what says which observation is current.
    """
    current = projection_provenance(index_conn)
    if current is not None and (current.observed_at_ms, current.observation_order) > (
        observed_at_ms,
        observation_order,
    ):
        return False
    index_conn.execute("DELETE FROM codex_thread_state")
    index_conn.execute("DELETE FROM codex_thread_spawn_edges")
    index_conn.executemany(
        """
        INSERT INTO codex_thread_state (
            thread_id, title, cwd, created_at_ms, updated_at_ms, source,
            model, agent_nickname, agent_role, archived, observed_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            title = excluded.title,
            cwd = excluded.cwd,
            created_at_ms = excluded.created_at_ms,
            updated_at_ms = excluded.updated_at_ms,
            source = excluded.source,
            model = excluded.model,
            agent_nickname = excluded.agent_nickname,
            agent_role = excluded.agent_role,
            archived = excluded.archived,
            observed_at_ms = excluded.observed_at_ms
        """,
        [
            (
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
            )
            for thread in snapshot.threads
        ],
    )
    index_conn.executemany(
        """
        INSERT INTO codex_thread_spawn_edges (
            parent_thread_id, child_thread_id, status, observed_at_ms
        ) VALUES (?, ?, ?, ?)
        ON CONFLICT(parent_thread_id, child_thread_id) DO UPDATE SET
            status = excluded.status,
            observed_at_ms = excluded.observed_at_ms
        """,
        [
            (edge.parent_thread_id, edge.child_thread_id, edge.status or "unknown", observed_at_ms)
            for edge in snapshot.spawn_edges
        ],
    )
    index_conn.execute(
        """
        INSERT INTO codex_thread_state_provenance (
            singleton, raw_id, blob_hash, observed_at_ms, observation_order
        ) VALUES (0, ?, ?, ?, ?)
        ON CONFLICT(singleton) DO UPDATE SET
            raw_id = excluded.raw_id,
            blob_hash = excluded.blob_hash,
            observed_at_ms = excluded.observed_at_ms,
            observation_order = excluded.observation_order
        """,
        (raw_id, blob_hash, observed_at_ms, observation_order),
    )
    return True


def apply_retained_state_export(
    archive: Any,
    raw_id: str,
    *,
    export_path: Path,
    blob_hash: str,
    observed_at_ms: int,
) -> bool:
    """Recompute the projection from one retained export, when the index is open.

    Returns whether the projection was written. Acquire-only ingestion holds
    no index handle at all, and that is not a failure: the export is durable,
    so the next pass with a derived tier recomputes from it.
    """
    index_conn = archive.index_connection
    if index_conn is None:
        return False
    snapshot = codex_state.parse_codex_state_db(export_path, immutable=True)
    receipt_at_ms, receipt_order = archive.raw_revision_observation_order(raw_id)
    return write_thread_state_projection(
        index_conn,
        snapshot,
        raw_id=raw_id,
        blob_hash=blob_hash,
        observed_at_ms=max(observed_at_ms, receipt_at_ms),
        observation_order=receipt_order,
    )


def ensure_thread_state_projection(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    *,
    blob_path_for_hash: Any,
) -> bool:
    """Reconcile the projection against the newest retained state export.

    Replay applies raws in no particular order, so a session can be written
    before the state export it describes. This derives the projection from
    durable evidence instead of from replay order; it is a no-op once the
    projection already names the newest export.
    """
    latest = latest_retained_state_export(source_conn)
    if latest is None:
        return False
    raw_id, blob_hash, observed_at_ms = latest
    current = projection_provenance(index_conn)
    if current is not None and current.raw_id == raw_id and current.blob_hash == blob_hash:
        return False
    export_path = blob_path_for_hash(blob_hash)
    if export_path is None or not Path(export_path).is_file():
        return False
    try:
        snapshot = codex_state.parse_codex_state_db(Path(export_path), immutable=True)
    except sqlite3.Error as exc:
        logger.warning("codex state: retained export %s is not readable as thread state: %s", raw_id, exc)
        return False
    return write_thread_state_projection(
        index_conn,
        snapshot,
        raw_id=raw_id,
        blob_hash=blob_hash,
        observed_at_ms=observed_at_ms,
    )


def read_thread_titles(
    index_conn: sqlite3.Connection,
    *,
    thread_ids: Any = None,
) -> dict[str, str]:
    """Return ``{thread_id: title}`` from the projected Codex thread state.

    Any failure (missing table, locked file) degrades to an empty mapping,
    matching every other sidecar source in the title ladder.
    """
    titles: dict[str, str] = {}
    try:
        if thread_ids is None:
            rows = index_conn.execute(
                "SELECT thread_id, title FROM codex_thread_state WHERE title IS NOT NULL"
            ).fetchall()
        else:
            wanted = list(dict.fromkeys(thread_ids))
            if not wanted:
                return {}
            rows = []
            for start in range(0, len(wanted), 500):
                chunk = wanted[start : start + 500]
                placeholders = ", ".join("?" for _ in chunk)
                rows.extend(
                    index_conn.execute(
                        "SELECT thread_id, title FROM codex_thread_state "
                        f"WHERE title IS NOT NULL AND thread_id IN ({placeholders})",
                        chunk,
                    ).fetchall()
                )
    except sqlite3.Error as exc:
        logger.debug("Failed to read projected Codex thread titles: %s", exc)
        return {}
    for row in rows:
        thread_id, title = str(row[0]), row[1]
        if thread_id and isinstance(title, str) and title.strip():
            titles[thread_id] = title.strip()
    return titles


def read_spawn_edges(index_conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """Return ``{(parent_thread_id, child_thread_id): status}`` from the projection."""
    try:
        rows = index_conn.execute(
            "SELECT parent_thread_id, child_thread_id, status FROM codex_thread_spawn_edges"
        ).fetchall()
    except sqlite3.Error as exc:
        logger.debug("Failed to read projected Codex spawn edges: %s", exc)
        return {}
    return {(str(row[0]), str(row[1])): str(row[2]) or "unknown" for row in rows}


def read_parent_thread_id(index_conn: sqlite3.Connection, child_thread_id: str) -> str | None:
    """Return the projected parent of ``child_thread_id``, or ``None`` when silent.

    ``None`` means the projection is silent about this child, which is not the
    same as it naming a different parent; only the latter is a conflict.
    """
    if not child_thread_id:
        return None
    try:
        row = index_conn.execute(
            """
            SELECT parent_thread_id
            FROM codex_thread_spawn_edges
            WHERE child_thread_id = ?
            ORDER BY observed_at_ms DESC, parent_thread_id
            LIMIT 1
            """,
            (child_thread_id,),
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None or row[0] is None:
        return None
    parent = str(row[0]).strip()
    return parent or None


__all__ = [
    "THREAD_STATE_KIND",
    "ProjectionProvenance",
    "apply_retained_state_export",
    "ensure_thread_state_projection",
    "latest_retained_state_export",
    "projection_provenance",
    "read_parent_thread_id",
    "read_spawn_edges",
    "read_thread_titles",
    "thread_state_member_filenames",
    "write_thread_state_projection",
]
