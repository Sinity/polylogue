"""Seed runtime-reported thread state through the production graph writer.

Tests used to insert into provider-named index tables directly. The
work-evidence graph is the only home for that evidence now, and its node/edge
shape is production semantics, so fixtures go through the real writer
(:func:`polylogue.storage.sqlite.agent_thread_state.write_thread_state_graph`)
rather than re-spelling its SQL.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from polylogue.storage.sqlite.agent_thread_state import (
    SpawnRecord,
    ThreadRecord,
    write_thread_state_graph,
)

__all__ = ["seed_spawn_edges", "seed_thread_state", "seed_thread_titles"]


def seed_thread_state(
    conn: sqlite3.Connection,
    *,
    threads: Sequence[tuple[str, str]] = (),
    spawn_edges: Sequence[tuple[str, str, str]] = (),
    source_scope: str = "",
    raw_id: str = "test-raw",
    blob_hash: str = "test-blob",
    observed_at_ms: int = 1_000,
    observation_order: int = 0,
) -> None:
    """Project one scope's thread state: ``(thread_id, title)`` and spawn triples.

    Seeds carry no source tier, so no retained export is ranked.
    """
    write_thread_state_graph(
        conn,
        source_scope=source_scope,
        threads=[ThreadRecord(thread_id, title, observed_at_ms) for thread_id, title in threads],
        spawn_edges=[SpawnRecord(parent, child, status) for parent, child, status in spawn_edges],
        raw_id=raw_id,
        blob_hash=blob_hash,
        observed_at_ms=observed_at_ms,
        observation_order=observation_order,
        export_order=lambda _raw_id: None,
    )
    conn.commit()


def seed_spawn_edges(
    conn: sqlite3.Connection,
    edges: Sequence[tuple[str, str, str]],
    **kwargs: object,
) -> None:
    """Project spawn edges alone."""
    seed_thread_state(conn, spawn_edges=edges, **kwargs)  # type: ignore[arg-type]


def seed_thread_titles(
    conn: sqlite3.Connection,
    titles: Sequence[tuple[str, str]],
    **kwargs: object,
) -> None:
    """Project curated thread titles alone."""
    seed_thread_state(conn, threads=titles, **kwargs)  # type: ignore[arg-type]


def codex_state_export(threads: Sequence[tuple[str, str]]) -> bytes:
    """Build a declared logical state export from neutral thread/title pairs."""
    from polylogue.sources.sqlite_export import logical_export_bytes
    from polylogue.sources.sqlite_snapshot import member_export_scope

    with TemporaryDirectory(prefix="synthetic-state-") as directory:
        source = Path(directory) / "state_5.sqlite"
        with closing(sqlite3.connect(source)) as conn:
            conn.executescript(
                "CREATE TABLE threads (id TEXT, title TEXT, cwd TEXT, created_at_ms INTEGER, "
                "updated_at_ms INTEGER, source TEXT, model TEXT, agent_nickname TEXT, agent_role TEXT, archived INTEGER);"
                "CREATE TABLE thread_spawn_edges (parent_thread_id TEXT, child_thread_id TEXT, status TEXT);"
            )
            conn.executemany(
                "INSERT INTO threads VALUES (?, ?, '/repo', 1000, 2000, 'cli', 'synthetic', NULL, NULL, 0)", threads
            )
            conn.commit()
        return logical_export_bytes(source, scope=member_export_scope(source))
