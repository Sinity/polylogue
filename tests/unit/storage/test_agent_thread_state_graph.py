"""The work-evidence graph carries everything the Codex-named tables carried.

``codex_thread_state`` (curated titles), ``codex_thread_spawn_edges``
(parent/child spawns and the runtime's own lifecycle label) and
``codex_thread_state_provenance`` (which export a scope was computed from,
with its durable receipt order) are gone; one graph per evidence scope holds
all three facts. Each test below names the mutation that makes it red.
"""

from __future__ import annotations

import sqlite3

import pytest

from polylogue.storage.sqlite.agent_thread_state import (
    SpawnRecord,
    ThreadRecord,
    read_parent_thread_id,
    read_provenance,
    read_spawn_edge_children,
    read_spawn_edges,
    read_thread_titles,
    write_thread_state_graph,
)
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL


@pytest.fixture
def index_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(INDEX_DDL)
    return conn


def _write(conn: sqlite3.Connection, **kwargs: object) -> bool:
    defaults: dict[str, object] = {
        "source_scope": "/install",
        "threads": [ThreadRecord("parent-thread", "Curated title", 2_000)],
        "spawn_edges": [SpawnRecord("parent-thread", "child-thread", "closed")],
        "raw_id": "raw-1",
        "blob_hash": "blob-1",
        "observed_at_ms": 1_000,
        "observation_order": 1,
    }
    defaults.update(kwargs)
    return write_thread_state_graph(conn, **defaults)  # type: ignore[arg-type]


def test_graph_carries_titles_spawns_and_provenance(index_conn: sqlite3.Connection) -> None:
    """Anti-vacuity: drop any of the three writes and one assertion below fails."""
    assert _write(index_conn) is True

    assert read_thread_titles(index_conn) == {"parent-thread": "Curated title"}
    assert read_thread_titles(index_conn, thread_ids=["parent-thread"]) == {"parent-thread": "Curated title"}
    assert read_thread_titles(index_conn, thread_ids=["parent-thread"], source_scope="/install") == {
        "parent-thread": "Curated title"
    }
    assert read_spawn_edges(index_conn) == {("parent-thread", "child-thread"): "closed"}
    assert read_spawn_edge_children(index_conn) == {"child-thread"}
    assert read_parent_thread_id(index_conn, "child-thread") == "parent-thread"
    assert read_parent_thread_id(index_conn, "child-thread", source_scope="/install") == "parent-thread"

    provenance = read_provenance(index_conn)
    assert provenance is not None
    assert (provenance.raw_id, provenance.blob_hash) == ("raw-1", "blob-1")
    assert (provenance.observed_at_ms, provenance.observation_order) == (1_000, 1)


def test_graph_shape_is_neutral_nodes_and_edges(index_conn: sqlite3.Connection) -> None:
    """Anti-vacuity: write the thread as a claim (or the title as a label only)
    and these kind assertions fail."""
    _write(index_conn)

    kinds = dict(index_conn.execute("SELECT node_ref, node_kind FROM work_evidence_nodes").fetchall())
    assert set(kinds.values()) == {"execution-context", "claim"}
    edge_kinds = {row[0] for row in index_conn.execute("SELECT edge_kind FROM work_evidence_edges")}
    assert edge_kinds == {"invoked", "claimed"}
    authorities = {row[0] for row in index_conn.execute("SELECT authority FROM work_evidence_nodes")}
    assert authorities == {"provider"}
    evidence = {row[0] for row in index_conn.execute("SELECT evidence_refs_json FROM work_evidence_nodes")}
    assert evidence == {'["artifact:raw-1"]'}


def test_deleting_the_invoked_edges_breaks_parent_resolution(index_conn: sqlite3.Connection) -> None:
    """Anti-vacuity for every parent-resolution caller: the spawn fact lives in
    the ``invoked`` edges and nowhere else."""
    _write(index_conn)
    index_conn.execute("DELETE FROM work_evidence_edges WHERE edge_kind = 'invoked'")

    assert read_parent_thread_id(index_conn, "child-thread") is None
    assert read_spawn_edges(index_conn) == {}
    # The title claim is a separate edge kind and must survive.
    assert read_thread_titles(index_conn) == {"parent-thread": "Curated title"}


def test_absent_objects_stay_readable_and_are_marked_superseded(index_conn: sqlite3.Connection) -> None:
    """A newer snapshot silent about an object marks it superseded instead of
    deleting it. Anti-vacuity: delete instead of mark and the read is empty."""
    _write(index_conn)
    assert (
        _write(
            index_conn,
            threads=[ThreadRecord("new-thread", "New title", 3_000)],
            spawn_edges=[],
            raw_id="raw-2",
            blob_hash="blob-2",
            observed_at_ms=2_000,
            observation_order=2,
        )
        is True
    )

    assert read_thread_titles(index_conn) == {"new-thread": "New title", "parent-thread": "Curated title"}
    assert read_spawn_edges(index_conn) == {("parent-thread", "child-thread"): "closed"}
    states = dict(
        index_conn.execute(
            "SELECT node_ref, association_state FROM work_evidence_nodes WHERE node_kind = 'execution-context'"
        ).fetchall()
    )
    assert states["execution-context:agent-thread:/install::new-thread"] == "resolved"
    assert states["execution-context:agent-thread:/install::parent-thread"] == "superseded"


def test_an_older_export_does_not_overwrite_a_newer_graph(index_conn: sqlite3.Connection) -> None:
    """Replay applies raws in no particular order. Anti-vacuity: drop the
    receipt-order comparison and the older title wins."""
    _write(index_conn, observed_at_ms=2_000, observation_order=2)

    assert (
        _write(
            index_conn,
            threads=[ThreadRecord("parent-thread", "Older title", 1_000)],
            raw_id="raw-0",
            blob_hash="blob-0",
            observed_at_ms=1_000,
            observation_order=1,
        )
        is False
    )
    assert read_thread_titles(index_conn) == {"parent-thread": "Curated title"}


def test_one_scopes_snapshot_does_not_supersede_another_scope(index_conn: sqlite3.Connection) -> None:
    """Anti-vacuity: key the graph on the tier instead of the scope and the
    second write marks the first install's evidence superseded."""
    _write(index_conn, source_scope="/install-a")
    _write(
        index_conn,
        source_scope="/install-b",
        threads=[ThreadRecord("b-thread", "B title", 2_000)],
        spawn_edges=[],
        raw_id="raw-b",
        blob_hash="blob-b",
        observed_at_ms=2_000,
        observation_order=2,
    )

    assert read_thread_titles(index_conn) == {"b-thread": "B title", "parent-thread": "Curated title"}
    assert read_thread_titles(index_conn, source_scope="/install-a") == {"parent-thread": "Curated title"}
    assert (
        index_conn.execute(
            "SELECT COUNT(*) FROM work_evidence_nodes WHERE association_state = 'superseded'"
        ).fetchone()[0]
        == 0
    )


def test_index_tier_carries_no_provider_named_relation() -> None:
    """The DDL rule the fold is worth keeping. Anti-vacuity: append a
    provider-named table to the index DDL and this must fail."""
    from devtools.verify_schema_manifest import _PROVIDER_TOKENS, _provider_named_index_objects, _strip_sql_noise

    assert _provider_named_index_objects() == []
    assert "codex" in _PROVIDER_TOKENS
    # A vocabulary value naming a provider is not a provider-named object.
    assert "claude-code-session" not in _strip_sql_noise("origin IN ('claude-code-session')")
