"""bd polylogue-foee (AC#2): reconcile acquired Codex thread_spawn_edges
against transcript-inferred session_links topology.

Exercises ``context.codex_spawn_edge_correlation`` directly against real
``source.db``/``index.db`` connections -- the facade-level integration test
lives alongside the other reconciliation facade tests in
``tests/unit/api/test_facade_contracts.py``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.context.codex_spawn_edge_correlation import reconcile_codex_spawn_edges
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL, INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent, write_source_hook_event
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_HASH = b"x" * 32


def _index_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    conn.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
    return conn


def _source_conn(archive_root: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _init_source_db(archive_root: Path) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(archive_root / "source.db") as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        conn.commit()


def _write_spawn_edge(
    archive_root: Path, *, parent_thread_id: str, child_thread_id: str, status: str = "closed"
) -> None:
    payload: dict[str, object] = {
        "parent_thread_id": parent_thread_id,
        "child_thread_id": child_thread_id,
        "status": status,
    }
    encoded = json.dumps(payload).encode("utf-8")
    with sqlite3.connect(archive_root / "source.db") as conn:
        write_source_hook_event(
            conn,
            origin="codex-session",
            source_path="synthetic:state-db",
            payload=encoded,
            acquired_at_ms=1_000,
            raw_id=f"raw-{parent_thread_id}-{child_thread_id}",
            hook_event=ArchiveHookEvent(
                hook_event_id=f"codex-thread-spawn-edge:{parent_thread_id}:{child_thread_id}",
                origin="codex-session",
                source_path="synthetic:state-db",
                event_type="codex_thread_spawn_edge",
                payload=payload,
                observed_at_ms=1_000,
                native_id=f"{parent_thread_id}:{child_thread_id}:codex_thread_spawn_edge",
                session_native_id=parent_thread_id,
            ),
        )


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


def test_inferred_edge_backed_by_matching_authoritative_evidence(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_source_db(archive_root)
    _write_spawn_edge(archive_root, parent_thread_id="parent-1", child_thread_id="child-1")

    index_conn = _index_conn()
    _seed_subagent_link(index_conn, parent_thread_id="parent-1", child_thread_id="child-1")

    source_conn = _source_conn(archive_root)
    report = reconcile_codex_spawn_edges(source_conn, index_conn)

    assert report.total_authoritative_edges == 1
    assert report.total_inferred_subagent_links == 1
    assert report.backed_by_authoritative_count == 1
    assert report.inferred_only_count == 0
    assert report.authoritative_only_count == 0


def test_inferred_only_and_authoritative_only_edges_are_both_visible(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_source_db(archive_root)
    # Authoritative evidence for an edge the transcript never proved (e.g. a
    # crashed child -- see module docstring).
    _write_spawn_edge(archive_root, parent_thread_id="parent-a", child_thread_id="child-crashed")

    index_conn = _index_conn()
    # Transcript-inferred edge with no authoritative counterpart.
    _seed_subagent_link(index_conn, parent_thread_id="parent-b", child_thread_id="child-inferred-only")

    source_conn = _source_conn(archive_root)
    report = reconcile_codex_spawn_edges(source_conn, index_conn)

    assert report.total_authoritative_edges == 1
    assert report.total_inferred_subagent_links == 1
    assert report.backed_by_authoritative_count == 0
    assert report.inferred_only_count == 1
    assert report.authoritative_only_count == 1
    assert report.inferred_only_edges == (("parent-b", "child-inferred-only"),)
    assert report.authoritative_only_edges == (("parent-a", "child-crashed"),)


def test_no_evidence_either_side_reconciles_as_empty(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_source_db(archive_root)
    index_conn = _index_conn()

    source_conn = _source_conn(archive_root)
    report = reconcile_codex_spawn_edges(source_conn, index_conn)

    assert report.total_authoritative_edges == 0
    assert report.total_inferred_subagent_links == 0
    assert report.backed_by_authoritative_count == 0
