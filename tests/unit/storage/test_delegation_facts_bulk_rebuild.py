"""``rebuild_all_delegation_facts_sync`` (polylogue-v6i3): the bulk-build
readiness repopulate step for ``delegation_facts``.

``delegation_facts_source`` (the view backing ``delegation_facts_insert_sql``)
is scoped by the ``delegation_refresh_scope`` allow-list table rather than an
inline per-parent predicate. ``rebuild_all_delegation_facts_sync`` populates
that scope with every session id instead of one parent at a time, turning the
existing per-cohort insert into an archive-wide one with no new SQL shape to
maintain. These tests prove: the bulk rebuild produces byte-identical content
to the existing per-session ``refresh_delegation_facts_for_session`` path for
the same underlying dispatch/link evidence, across resolved/ambiguous/edge-
only/unresolved mapping states; and the scope-population step is load-bearing
(without it, ``delegation_facts_source``'s EXISTS-gated view produces zero
rows for every parent, not a coincidentally-correct empty set)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.delegation_facts import (
    rebuild_all_delegation_facts_sync,
    refresh_delegation_facts_for_session,
)

_HASH = b"x" * 32


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _insert_session(conn: sqlite3.Connection, *, native_id: str, origin: str = "claude-code-session") -> str:
    conn.execute(
        """
        INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms)
        VALUES (?, ?, ?, ?, 1767225600000, 1767225601000)
        """,
        (native_id, origin, f"session {native_id}", _HASH),
    )
    return str(
        conn.execute(
            "SELECT session_id FROM sessions WHERE native_id = ? AND origin = ?", (native_id, origin)
        ).fetchone()["session_id"]
    )


def _insert_message(conn: sqlite3.Connection, *, session_id: str, native_id: str, position: int) -> str:
    conn.execute(
        """
        INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash, occurred_at_ms)
        VALUES (?, ?, ?, 'assistant', 'message', ?, ?)
        """,
        (session_id, native_id, position, _HASH, 1767225600000 + position),
    )
    return str(
        conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? AND native_id = ?", (session_id, native_id)
        ).fetchone()["message_id"]
    )


def _insert_dispatch_action(
    conn: sqlite3.Connection,
    *,
    message_id: str,
    session_id: str,
    position: int,
    tool_id: str,
    result_text: str | None = "done",
) -> None:
    conn.execute(
        """
        INSERT INTO blocks (
            message_id, session_id, position, block_type, tool_name, tool_id, semantic_type, tool_input
        ) VALUES (?, ?, ?, 'tool_use', 'Task', ?, 'subagent', '{}')
        """,
        (message_id, session_id, position, tool_id),
    )
    if result_text is not None:
        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, text, tool_id,
                tool_result_is_error, tool_result_exit_code
            ) VALUES (?, ?, ?, 'tool_result', ?, ?, 0, 0)
            """,
            (message_id, session_id, position + 1, result_text, tool_id),
        )


def _insert_session_link(
    conn: sqlite3.Connection,
    *,
    child_session_id: str,
    dst_origin: str,
    dst_native_id: str,
    parent_session_id: str | None,
    link_type: str = "subagent",
) -> None:
    conn.execute(
        """
        INSERT INTO session_links (
            src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id, observed_at_ms
        ) VALUES (?, ?, ?, ?, ?, 1767225600000)
        """,
        (child_session_id, dst_origin, dst_native_id, link_type, parent_session_id),
    )


def _build_delegation_corpus(conn: sqlite3.Connection) -> list[str]:
    """Three parent cohorts spanning resolved, ambiguous, and edge-only
    mapping states -- so the comparison below is not just the trivial single-
    row case."""
    parent_ids = []

    # Resolved: one dispatch, one resolved child.
    parent_a = _insert_session(conn, native_id="parent-a")
    child_a = _insert_session(conn, native_id="child-a")
    msg_a = _insert_message(conn, session_id=parent_a, native_id="dispatch", position=0)
    _insert_dispatch_action(conn, message_id=msg_a, session_id=parent_a, position=0, tool_id="task-a")
    _insert_session_link(
        conn,
        child_session_id=child_a,
        dst_origin="claude-code-session",
        dst_native_id="parent-a",
        parent_session_id=parent_a,
    )
    parent_ids.append(parent_a)

    # Ambiguous: two dispatches, one resolved child (rank-pairing can't disambiguate).
    parent_b = _insert_session(conn, native_id="parent-b")
    child_b = _insert_session(conn, native_id="child-b")
    msg_b = _insert_message(conn, session_id=parent_b, native_id="dispatch", position=0)
    _insert_dispatch_action(conn, message_id=msg_b, session_id=parent_b, position=0, tool_id="task-b1")
    _insert_dispatch_action(conn, message_id=msg_b, session_id=parent_b, position=2, tool_id="task-b2")
    _insert_session_link(
        conn,
        child_session_id=child_b,
        dst_origin="claude-code-session",
        dst_native_id="parent-b",
        parent_session_id=parent_b,
    )
    parent_ids.append(parent_b)

    # Edge-only: a resolved child with no parent Task action at all (Codex-style).
    parent_c = _insert_session(conn, native_id="parent-c", origin="codex-session")
    child_c = _insert_session(conn, native_id="child-c", origin="codex-session")
    _insert_session_link(
        conn, child_session_id=child_c, dst_origin="codex-session", dst_native_id="parent-c", parent_session_id=parent_c
    )
    parent_ids.append(parent_c)

    return parent_ids


def _delegation_facts_rows(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    rows = conn.execute(
        """
        SELECT delegation_id, parent_session_id, child_session_id, mapping_state, link_confidence,
               link_method, inheritance, branch_point_message_id, instruction_message_id,
               instruction_tool_use_block_id, instruction_payload, dispatch_turn_model, requested_model,
               artifact_block_id, artifact_text, result_is_error, result_exit_code, result_status,
               parent_origin
        FROM delegation_facts
        ORDER BY delegation_id
        """
    ).fetchall()
    return sorted(tuple(row) for row in rows)


def test_rebuild_all_delegation_facts_matches_per_session_refresh(tmp_path: Path) -> None:
    conn_per_session = _connect(tmp_path / "per_session.db")
    parent_ids = _build_delegation_corpus(conn_per_session)
    for parent_id in parent_ids:
        refresh_delegation_facts_for_session(conn_per_session, parent_id)
    conn_per_session.commit()
    per_session_rows = _delegation_facts_rows(conn_per_session)
    conn_per_session.close()

    conn_bulk = _connect(tmp_path / "bulk.db")
    _build_delegation_corpus(conn_bulk)
    # Before the bulk rebuild, delegation_facts is untouched by the raw
    # INSERTs above (no write_parsed_session_to_archive transaction ran, so
    # the 'session-write' guard was never set -- the ordinary
    # blocks_action_pairs_ai/session_links_delegation_facts_ai triggers fire
    # on each raw insert and already populate it via the trigger path). Clear
    # it explicitly so this test proves the BULK rebuild path, not residual
    # trigger-driven content.
    conn_bulk.execute("DELETE FROM delegation_facts")
    conn_bulk.commit()
    rebuild_all_delegation_facts_sync(conn_bulk)
    conn_bulk.commit()
    bulk_rows = _delegation_facts_rows(conn_bulk)
    # The scope table must never leak past the bulk rebuild call.
    assert conn_bulk.execute("SELECT COUNT(*) FROM delegation_refresh_scope").fetchone()[0] == 0
    conn_bulk.close()

    assert bulk_rows == per_session_rows
    assert bulk_rows, "corpus produced no delegation_facts rows -- comparison would be vacuous"
    mapping_states = {row[3] for row in bulk_rows}
    assert mapping_states == {"resolved", "ambiguous", "edge_only"}, mapping_states


def test_rebuild_all_delegation_facts_scope_population_is_load_bearing(tmp_path: Path) -> None:
    """Anti-vacuity: without populating delegation_refresh_scope, the insert
    produces zero rows for every parent (the view's EXISTS clauses never
    match), proving the scope-population step is not incidental."""
    conn = _connect(tmp_path / "index.db")
    _build_delegation_corpus(conn)
    conn.execute("DELETE FROM delegation_facts")
    conn.commit()

    from polylogue.storage.sqlite.delegation_facts import delegation_facts_insert_sql

    # Real corpus, real dispatch/link evidence -- but delegation_refresh_scope
    # deliberately left empty (skipping rebuild_all_delegation_facts_sync's
    # scope-population step).
    conn.execute(delegation_facts_insert_sql("?"))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM delegation_facts").fetchone()[0] == 0
    conn.close()
