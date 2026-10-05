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
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

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
from polylogue.storage.sqlite.archive_tiers.write import ConnectionSessionSourceRead


@pytest.fixture(params=["plain", "measured"])
def index_conn(request: pytest.FixtureRequest) -> Iterator[sqlite3.Connection]:
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

    conn = connect_measured(":memory:") if request.param == "measured" else sqlite3.connect(":memory:")
    owner = NativeSQLCustodyOwner(conn) if request.param == "measured" else None
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(INDEX_DDL)
        yield owner.require_connection() if owner is not None else conn
    finally:
        if owner is not None:
            owner.close()
        else:
            conn.close()


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
    arguments: dict[str, Any] = defaults
    return write_thread_state_graph(conn, **arguments)


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
        order = codex_state_projection.retained_export_order(ConnectionSessionSourceRead(source))(raw_id)
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
            source_read=ConnectionSessionSourceRead(source),
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


def test_thread_parent_accounts_each_exact_scope_winner_before_hydration(
    index_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import closing, contextmanager

    from polylogue.storage.sqlite import agent_thread_state

    first = "first-" + "x" * 20000
    second = "second-" + "y" * 20000
    _write(index_conn, source_scope="/one", spawn_edges=[SpawnRecord(first, "child-thread", "closed")])
    _write(index_conn, source_scope="/two", spawn_edges=[SpawnRecord(second, "child-thread", "closed")])
    index_conn.commit()
    assert read_parent_thread_id(index_conn, "child-thread") is None
    accounted: dict[int, int] = {}
    from polylogue.storage.io_phase_metrics import connection_cursor as actual_cursor

    def before_input(table: str, columns: tuple[str, ...], sql: str, parameters: tuple[object, ...]) -> None:
        assert table == "work_evidence_edges" and columns == ("source_ref",)
        with closing(index_conn.execute(sql, parameters)) as metadata:
            rowid = metadata.fetchone()[0]
        with closing(
            index_conn.execute(
                "SELECT length(CAST(source_ref AS BLOB)) FROM work_evidence_edges WHERE rowid=?", (rowid,)
            )
        ) as metadata:
            accounted[rowid] = metadata.fetchone()[0]

    @contextmanager
    def guarded_cursor(
        connection: sqlite3.Connection, sql: str, parameters: tuple[object, ...] = ()
    ) -> Iterator[sqlite3.Cursor]:
        if sql.startswith("SELECT source_ref FROM work_evidence_edges WHERE rowid="):
            physical_rowid = parameters[0]
            assert isinstance(physical_rowid, int)
            assert accounted[physical_rowid] > 20000
        with actual_cursor(connection, sql, parameters) as cursor:
            yield cursor

    monkeypatch.setattr(agent_thread_state, "connection_cursor", guarded_cursor)
    assert read_parent_thread_id(index_conn, "child-thread", before_input=before_input) is None
    assert len(accounted) == 2
    assert not index_conn.in_transaction


def test_title_candidate_stream_does_not_prefetch_the_complete_id_cohort(index_conn: sqlite3.Connection) -> None:
    from contextlib import closing

    from polylogue.storage.sqlite.agent_thread_state import iter_thread_title_candidates

    _write(index_conn)
    consumed = 0

    def identities() -> Iterator[str]:
        nonlocal consumed
        for _ in range(1201):
            consumed += 1
            yield "parent-thread"

    expected = read_thread_titles(index_conn, thread_ids=["parent-thread"])
    with closing(iter_thread_title_candidates(index_conn, thread_ids=identities())) as rows:
        assert next(rows) == ("parent-thread", expected["parent-thread"])
        assert 0 < consumed < 1201
        observed = consumed
    assert consumed == observed
    assert index_conn.execute("SELECT COUNT(*) FROM work_evidence_nodes").fetchone()[0] > 0


def test_title_candidate_stream_retains_scope_and_first_winner(index_conn: sqlite3.Connection) -> None:
    from contextlib import closing

    from polylogue.storage.sqlite.agent_thread_state import iter_thread_title_candidates

    _write(index_conn)
    _write(
        index_conn,
        source_scope="/other-install",
        raw_id="raw-2",
        blob_hash="blob-2",
        threads=[ThreadRecord("other-thread", "Other title", 2000)],
    )
    expected = read_thread_titles(index_conn, thread_ids=["parent-thread", "other-thread"], source_scope="/install")
    selected: dict[str, str] = {}
    with closing(
        iter_thread_title_candidates(
            index_conn, thread_ids=iter(["parent-thread", "other-thread", "parent-thread"]), source_scope="/install"
        )
    ) as rows:
        for identity, title in rows:
            selected.setdefault(identity, title)
    assert selected == expected == {"parent-thread": "Curated title"}


def test_title_candidate_stream_early_close_retires_its_original_read_frame(index_conn: sqlite3.Connection) -> None:
    from contextlib import closing

    from polylogue.storage.sqlite.agent_thread_state import iter_thread_title_candidates

    _write(index_conn)
    index_conn.commit()
    assert not index_conn.in_transaction
    with closing(iter_thread_title_candidates(index_conn, thread_ids=iter(["parent-thread"]))) as rows:
        assert next(rows) == ("parent-thread", "Curated title")
        assert index_conn.in_transaction
    assert not index_conn.in_transaction
    assert read_thread_titles(index_conn, thread_ids=["parent-thread"]) == {"parent-thread": "Curated title"}


def test_retained_title_tape_covers_full_original_id_scope_and_shared_digest(index_conn: sqlite3.Connection) -> None:
    from contextlib import closing

    from polylogue.sources.retained_title_index import RetainedTitleIndex
    from polylogue.sources.revision_backfill import _enrichment_evidence_digest
    from polylogue.storage.sqlite.agent_thread_state import iter_thread_title_candidates

    threads = [ThreadRecord(f"thread-{index:04}", f"Title {index}", 2000) for index in range(1201)]
    _write(index_conn, threads=threads)
    identities = [thread.thread_id for thread in threads]
    expected = read_thread_titles(index_conn, thread_ids=identities)
    with closing(iter_thread_title_candidates(index_conn, thread_ids=iter(identities))) as selected:
        titles = RetainedTitleIndex(selected)
    path = titles._path
    try:
        assert len(titles) == 1201
        assert titles[identities[0]] == expected[identities[0]]
        assert titles[identities[-1]] == expected[identities[-1]]
        assert dict(titles) == expected
        actual_digest = _enrichment_evidence_digest({"retained_state_titles": titles, "neutral": [0, None]})
        assert actual_digest == _enrichment_evidence_digest({"retained_state_titles": expected, "neutral": [0, None]})
        changed = dict(expected)
        changed[identities[-1]] += " changed"
        assert actual_digest != _enrichment_evidence_digest({"retained_state_titles": changed, "neutral": [0, None]})
    finally:
        titles.close()
    assert not path.exists()


def test_retained_title_tape_preserves_first_selected_value_and_exact_strings() -> None:
    from polylogue.sources.retained_title_index import RetainedTitleIndex

    titles = RetainedTitleIndex(iter([("a", ""), ("a", "later"), ("b", "e\u0301"), ("c", "é")]))
    try:
        assert dict(titles) == {"a": "", "b": "e\u0301", "c": "é"}
        assert len(titles) == 3
    finally:
        titles.close()


def test_retained_title_tape_cancellation_settles_original_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile

    from polylogue.core.prepared_file import VerificationCancelledError
    from polylogue.sources import retained_title_index

    actual_directory = tempfile.TemporaryDirectory
    directories: list[Path] = []
    failure = VerificationCancelledError("synthetic cancellation at retained title row")
    visited = 0

    def directory(
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
        ignore_cleanup_errors: bool = False,
        *,
        delete: bool = True,
    ) -> tempfile.TemporaryDirectory[str]:
        owned = actual_directory(suffix, prefix, dir, ignore_cleanup_errors, delete=delete)
        directories.append(Path(owned.name))
        return owned

    def cancelled() -> None:
        nonlocal visited
        visited += 1
        if visited == 3:
            raise failure

    monkeypatch.setattr(tempfile, "TemporaryDirectory", directory)
    monkeypatch.setattr(retained_title_index, "check_compute_cancelled", cancelled)
    with pytest.raises(VerificationCancelledError) as observed:
        retained_title_index.RetainedTitleIndex((str(index), "neutral") for index in range(1201))
    assert observed.value is failure
    assert visited == 3
    assert len(directories) == 1
    assert not directories[0].exists()
