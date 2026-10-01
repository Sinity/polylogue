"""The work-evidence graph carries everything the Codex-named tables carried.

``codex_thread_state`` (curated titles), ``codex_thread_spawn_edges``
(parent/child spawns and the runtime's own lifecycle label) and
``codex_thread_state_provenance`` (which export a scope was computed from,
with its durable receipt order) are gone; one graph per evidence scope holds
all three facts. Each test below names the mutation that makes it red.
"""

from __future__ import annotations

import hashlib
import itertools
import sqlite3
from collections.abc import Sequence
from pathlib import Path

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
        "export_order": lambda _raw_id: None,
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


def test_one_thread_id_in_two_scopes_keeps_each_scopes_title(index_conn: sqlite3.Connection) -> None:
    """One thread ID under two install roots is one session with two scoped titles.

    A session reads the scope of the rollout that produced it, so each root's
    rollout resolves its own root's title and neither write overwrites the
    other.

    Anti-vacuity: drop ``source_path`` from the Codex title read, or key the
    thread on its ID alone, and both rollouts resolve the newer root's title.
    """
    from polylogue.sources import codex_state_projection

    for order, root in enumerate(("/roots/a/.codex", "/roots/b/.codex"), start=1):
        _write(
            index_conn,
            source_scope=root,
            threads=[ThreadRecord("shared-thread", f"title from {root}", 2_000)],
            spawn_edges=[],
            raw_id=f"raw-{order}",
            blob_hash=f"blob-{order}",
            observed_at_ms=order * 1_000,
            observation_order=order,
        )

    for root in ("/roots/a/.codex", "/roots/b/.codex"):
        rollout = f"{root}/sessions/2026/01/01/rollout-shared-thread.jsonl"
        assert codex_state_projection.read_thread_titles(
            index_conn, thread_ids=["shared-thread"], source_path=rollout
        ) == {"shared-thread": f"title from {root}"}
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


def test_title_read_reports_absent_tables_as_no_titles() -> None:
    """A tier without the evidence graph has no retained titles."""
    assert read_thread_titles(sqlite3.connect(":memory:"), thread_ids=["parent-thread"]) == {}


def test_title_read_propagates_an_interrupted_read(index_conn: sqlite3.Connection) -> None:
    """An interrupted read is a failure, not an absence of titles.

    Anti-vacuity: restore the blanket ``except sqlite3.Error: return {}`` in
    ``read_thread_titles`` and this returns ``{}``, which ingest would then
    persist as "this thread has no retained title".
    """
    _write(index_conn)
    index_conn.set_progress_handler(lambda: 1, 1)
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        read_thread_titles(index_conn, thread_ids=["parent-thread"])


def test_a_later_receipt_wins_even_when_its_clock_rolled_back(index_conn: sqlite3.Connection) -> None:
    """A -> B -> A after a wall-clock rollback: A's second receipt is newer by
    receipt order although its stamp is older than B's. Anti-vacuity: rank by
    ``observed_at_ms`` first and B's title stays current."""
    _write(index_conn, raw_id="raw-b", blob_hash="blob-b", observed_at_ms=300, observation_order=2)

    assert (
        _write(
            index_conn,
            threads=[ThreadRecord("parent-thread", "Returned title", 1_000)],
            raw_id="raw-a",
            blob_hash="blob-a",
            observed_at_ms=250,
            observation_order=3,
        )
        is True
    )
    assert read_thread_titles(index_conn) == {"parent-thread": "Returned title"}
    provenance = read_provenance(index_conn)
    assert provenance is not None
    assert (provenance.raw_id, provenance.observation_order) == ("raw-a", 3)


#: Three retained exports of one scope, oldest first: the first two name
#: ``kept-thread`` with conflicting revisions, the newest omits it.
_EXPORTS: dict[str, dict[str, object]] = {
    "raw-1": {
        "threads": [ThreadRecord("kept-thread", "First title", 1_000)],
        "spawn_edges": [SpawnRecord("parent-thread", "kept-thread", "running")],
        "blob_hash": "blob-1",
        "observed_at_ms": 1_000,
        "observation_order": 1,
    },
    "raw-2": {
        "threads": [ThreadRecord("kept-thread", "Revised title", 2_000)],
        "spawn_edges": [SpawnRecord("parent-thread", "kept-thread", "closed")],
        "blob_hash": "blob-2",
        "observed_at_ms": 2_000,
        "observation_order": 2,
    },
    "raw-3": {
        "threads": [ThreadRecord("other-thread", "Other title", 3_000)],
        "spawn_edges": [],
        "blob_hash": "blob-3",
        "observed_at_ms": 3_000,
        "observation_order": 3,
    },
}


def _export_order(raw_id: str) -> int | None:
    export = _EXPORTS.get(raw_id)
    return None if export is None else int(export["observation_order"])  # type: ignore[call-overload]


def _graph_rows(conn: sqlite3.Connection) -> tuple[list[tuple[object, ...]], ...]:
    return (
        conn.execute("SELECT * FROM work_evidence_graphs ORDER BY graph_id").fetchall(),
        conn.execute("SELECT * FROM work_evidence_nodes ORDER BY node_ref").fetchall(),
        conn.execute("SELECT * FROM work_evidence_edges ORDER BY edge_ref").fetchall(),
    )


def _apply_exports(order: Sequence[str]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(INDEX_DDL)
    for raw_id in order:
        _write(conn, raw_id=raw_id, export_order=_export_order, **_EXPORTS[raw_id])
    return conn


def test_an_export_older_than_the_current_one_still_retains_what_it_names() -> None:
    """Retained rows do not depend on whether replay visits the older export first.

    Live order (S1, then S2 omitting the thread) leaves S1's title readable
    as superseded evidence; a replay visiting S2 first must end the same.

    Anti-vacuity: skip an export older than the current graph outright and the
    replay order has no title for ``kept-thread``.
    """
    live = _apply_exports(["raw-1", "raw-3"])
    replay = _apply_exports(["raw-3", "raw-1"])

    for conn in (live, replay):
        assert read_thread_titles(conn, thread_ids=["kept-thread"]) == {"kept-thread": "First title"}
        assert read_parent_thread_id(conn, "kept-thread") == "parent-thread"
    assert _graph_rows(replay) == _graph_rows(live)


@pytest.mark.parametrize("order", list(itertools.permutations(_EXPORTS)))
def test_retained_rows_hold_the_newest_revision_that_names_them(order: tuple[str, ...]) -> None:
    """Conflicting revisions of a superseded object resolve by receipt order.

    Every arrival order of three exports yields the rows of the in-order
    (live) application: ``kept-thread`` keeps the newer ``Revised title`` and
    ``closed`` label from S2, superseded by S3, whether S1 arrives before or
    after S2.

    Anti-vacuity: let an older export add only absent rows, or overwrite
    retained rows unconditionally, and an order that delivers S1 after (or
    before) S2 keeps ``First title`` or ``running``.
    """
    live = _apply_exports(sorted(_EXPORTS))
    replay = _apply_exports(order)

    assert read_thread_titles(replay, thread_ids=["kept-thread"]) == {"kept-thread": "Revised title"}
    assert read_spawn_edges(replay) == {("parent-thread", "kept-thread"): "closed"}
    assert _graph_rows(replay) == _graph_rows(live)


def test_projection_ranks_retained_rows_by_their_exports_durable_receipts(tmp_path: Path) -> None:
    """The Codex projection ranks each retained row by its export's receipt.

    S3 lands first, then S2, then S1: S1 is older than both, so the row S2
    retained for ``kept-thread`` must stand.

    Anti-vacuity: rank retained rows without the source tier's receipts and
    S1, arriving last, overwrites S2's title with ``First title``.
    """
    from polylogue.sources import codex_state_projection
    from polylogue.sources.parsers import codex_state
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    index = sqlite3.connect(tmp_path / "index.db")
    index.execute("PRAGMA foreign_keys = ON")
    index.executescript(INDEX_DDL)
    for raw_id, export in _EXPORTS.items():
        digest = hashlib.sha256(raw_id.encode()).digest()
        source.execute(
            "INSERT INTO raw_sessions (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms) "
            "VALUES (?, 'codex-session', '/install/state_5.sqlite', 0, ?, 1, 1)",
            (raw_id, digest),
        )
        source.execute(
            "INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms) "
            "VALUES (?, ?, 'raw_payload', '/install/state_5.sqlite', 1, ?)",
            (digest, raw_id, export["observed_at_ms"]),
        )
    source.commit()

    for raw_id in ("raw-3", "raw-2", "raw-1"):
        export = _EXPORTS[raw_id]
        order = codex_state_projection.retained_export_order(source)(raw_id)
        assert order is not None
        snapshot = codex_state.CodexStateSnapshot(
            threads=tuple(
                codex_state.CodexThreadRecord(
                    thread_id=thread.thread_id,
                    title=thread.title or "",
                    cwd="/work",
                    created_at_ms=thread.occurred_at_ms or 0,
                    updated_at_ms=thread.occurred_at_ms or 0,
                    source="cli",
                    model=None,
                    agent_nickname=None,
                    agent_role=None,
                    archived=False,
                )
                for thread in export["threads"]  # type: ignore[attr-defined]
            ),
            spawn_edges=tuple(
                codex_state.CodexSpawnEdge(edge.parent_thread_id, edge.child_thread_id, edge.status)
                for edge in export["spawn_edges"]  # type: ignore[attr-defined]
            ),
        )
        codex_state_projection.write_thread_state_projection(
            index,
            snapshot,
            raw_id=raw_id,
            blob_hash=str(export["blob_hash"]),
            observed_at_ms=int(export["observed_at_ms"]),  # type: ignore[call-overload]
            observation_order=order,
            source_scope="/install",
            source_conn=source,
        )

    assert read_thread_titles(index, thread_ids=["kept-thread"]) == {"kept-thread": "Revised title"}
    assert read_thread_titles(index, thread_ids=["other-thread"]) == {"other-thread": "Other title"}
    provenance = read_provenance(index, source_scope="/install")
    assert provenance is not None and provenance.raw_id == "raw-3"


@pytest.mark.parametrize("reader", [read_parent_thread_id, read_provenance, read_spawn_edges])
def test_graph_authority_reads_propagate_interruption(index_conn: sqlite3.Connection, reader: object) -> None:
    """A failed authority read cannot authorize a replacement graph as absent."""
    _write(index_conn)
    index_conn.set_progress_handler(lambda: 1, 1)
    try:
        with pytest.raises(sqlite3.OperationalError):
            if reader is read_parent_thread_id:
                read_parent_thread_id(index_conn, "child-thread")
            elif reader is read_provenance:
                read_provenance(index_conn)
            else:
                read_spawn_edges(index_conn)
    finally:
        index_conn.set_progress_handler(None, 0)
    assert read_parent_thread_id(index_conn, "child-thread") == "parent-thread"
    assert read_thread_titles(index_conn) == {"parent-thread": "Curated title"}
