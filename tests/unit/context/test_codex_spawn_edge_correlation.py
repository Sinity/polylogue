"""Reconcile projected Codex thread_spawn_edges against inferred topology.

Exercises ``context.codex_spawn_edge_correlation`` directly against a real
``index.db`` connection -- the facade-level integration test lives alongside
the other reconciliation facade tests in
``tests/unit/api/test_facade_contracts.py``.
"""

from __future__ import annotations

import sqlite3

from polylogue.context.codex_spawn_edge_correlation import reconcile_codex_spawn_edges
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL, INDEX_SCHEMA_VERSION

_HASH = b"x" * 32


def _index_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    conn.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
    return conn


def _write_spawn_edge(
    index_conn: sqlite3.Connection, *, parent_thread_id: str, child_thread_id: str, status: str = "closed"
) -> None:
    index_conn.execute(
        "INSERT INTO codex_thread_spawn_edges (parent_thread_id, child_thread_id, status, observed_at_ms) "
        "VALUES (?, ?, ?, ?)",
        (parent_thread_id, child_thread_id, status, 1_000),
    )
    index_conn.commit()


def _seed_subagent_link(index_conn: sqlite3.Connection, *, parent_thread_id: str, child_thread_id: str) -> None:
    index_conn.execute(
        "INSERT INTO sessions (native_id, origin, title, content_hash, message_count) VALUES (?, ?, ?, ?, ?)",
        (child_thread_id, "codex-session", "test", _HASH, 1),
    )
    index_conn.execute(
        "INSERT INTO session_links (src_session_id, dst_origin, dst_native_id, link_type, observed_at_ms) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"codex-session:{child_thread_id}", "codex-session", parent_thread_id, "subagent", 1_000),
    )
    index_conn.commit()


def test_inferred_edge_backed_by_matching_authoritative_evidence() -> None:
    index_conn = _index_conn()
    _write_spawn_edge(index_conn, parent_thread_id="parent-1", child_thread_id="child-1")
    _seed_subagent_link(index_conn, parent_thread_id="parent-1", child_thread_id="child-1")

    report = reconcile_codex_spawn_edges(index_conn)

    assert report.total_authoritative_edges == 1
    assert report.total_inferred_subagent_links == 1
    assert report.backed_by_authoritative_count == 1
    assert report.inferred_only_count == 0
    assert report.authoritative_only_count == 0


def test_inferred_only_and_authoritative_only_edges_are_both_visible() -> None:
    index_conn = _index_conn()
    # Authoritative evidence for an edge the transcript never proved (e.g. a
    # crashed child -- see module docstring).
    _write_spawn_edge(index_conn, parent_thread_id="parent-a", child_thread_id="child-crashed")
    # Transcript-inferred edge with no authoritative counterpart.
    _seed_subagent_link(index_conn, parent_thread_id="parent-b", child_thread_id="child-inferred-only")

    report = reconcile_codex_spawn_edges(index_conn)

    assert report.total_authoritative_edges == 1
    assert report.total_inferred_subagent_links == 1
    assert report.backed_by_authoritative_count == 0
    assert report.inferred_only_count == 1
    assert report.authoritative_only_count == 1
    assert report.inferred_only_edges == (("parent-b", "child-inferred-only"),)
    assert report.authoritative_only_edges == (("parent-a", "child-crashed"),)


def test_no_evidence_either_side_reconciles_as_empty() -> None:
    index_conn = _index_conn()

    report = reconcile_codex_spawn_edges(index_conn)

    assert report.total_authoritative_edges == 0
    assert report.total_inferred_subagent_links == 0
    assert report.backed_by_authoritative_count == 0
