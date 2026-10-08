"""Archive-backed delegation work-evidence materialization."""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from polylogue.analysis import delegation_work_evidence_materializer as materializer
from polylogue.analysis.delegation_work_evidence_materializer import (
    DELEGATION_WORK_EVIDENCE_GRAPH_ID,
    delegation_work_evidence_materialization_needed,
    delegation_work_evidence_snapshot,
    materialize_delegation_work_evidence_archive,
)
from polylogue.core.stage_admission import stage_write_admission
from polylogue.daemon.convergence_stages import make_delegation_work_evidence_stage
from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection
from polylogue.storage.sqlite.write_lease import UnleasedWriteError, arm_write_lease_enforcement, write_lease
from tests.infra.delegation_packets import seed_delegations


def test_materializer_replaces_archive_projection_and_tracks_delegation_freshness(tmp_path: Path) -> None:
    seed_delegations(tmp_path)

    assert delegation_work_evidence_materialization_needed(tmp_path) is True
    assert materialize_delegation_work_evidence_archive(tmp_path) == 1
    assert delegation_work_evidence_materialization_needed(tmp_path) is False

    with sqlite3.connect(tmp_path / "index.db") as conn:
        graph = conn.execute(
            "SELECT corpus_snapshot_ref FROM work_evidence_graphs WHERE graph_id = ?",
            (DELEGATION_WORK_EVIDENCE_GRAPH_ID,),
        ).fetchone()
        node_kinds = {
            row[0]
            for row in conn.execute(
                "SELECT node_kind FROM work_evidence_nodes WHERE graph_id = ?",
                (DELEGATION_WORK_EVIDENCE_GRAPH_ID,),
            )
        }
        edge_kinds = {
            row[0]
            for row in conn.execute(
                "SELECT edge_kind FROM work_evidence_edges WHERE graph_id = ?",
                (DELEGATION_WORK_EVIDENCE_GRAPH_ID,),
            )
        }

    assert graph is not None
    assert node_kinds == {"call", "attempt"}
    assert edge_kinds == {"invoked"}

    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE blocks SET tool_input = ? WHERE tool_id = 'task-1'", ('{"prompt":"test"}',))

    assert delegation_work_evidence_materialization_needed(tmp_path) is True
    assert materialize_delegation_work_evidence_archive(tmp_path) == 1
    assert delegation_work_evidence_materialization_needed(tmp_path) is False

    # A replacement must remove rows that disappeared from canonical evidence;
    # otherwise stale nodes remain traversable after source evidence changes.
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM session_links")
        conn.execute("DELETE FROM blocks WHERE tool_id = 'task-1'")
        conn.commit()

    assert materialize_delegation_work_evidence_archive(tmp_path) == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM work_evidence_nodes WHERE graph_id = ?",
                (DELEGATION_WORK_EVIDENCE_GRAPH_ID,),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM work_evidence_edges WHERE graph_id = ?",
                (DELEGATION_WORK_EVIDENCE_GRAPH_ID,),
            ).fetchone()[0]
            == 0
        )


def test_materializer_pins_digest_and_rows_to_the_same_index_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The published graph label and queried rows share one operation snapshot.

    Anti-vacuity: separate freshness/read connections produce distinct SQLite
    connection identities, so this fails if either read leaves the pinned
    operation connection.
    """
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    seed_delegations(tmp_path)
    connections: list[sqlite3.Connection] = []
    snapshot = materializer._snapshot_connection
    query = ArchiveStore.query_delegations

    def record_snapshot(conn: sqlite3.Connection) -> object:
        connections.append(conn)
        return snapshot(conn)

    def record_query(archive: ArchiveStore, *args: object, **kwargs: object) -> object:
        connections.append(archive._conn)
        return query(archive, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(materializer, "_snapshot_connection", record_snapshot)
    monkeypatch.setattr(ArchiveStore, "query_delegations", record_query)

    assert materialize_delegation_work_evidence_archive(tmp_path) == 1
    # The snapshot, every page, and the terminating empty page.
    assert len(connections) >= 2
    assert all(conn is connections[0] for conn in connections)


def test_delegation_stage_reads_without_daemon_writer_lease(tmp_path: Path) -> None:
    """The freshness probe and materialization read run outside writer admission."""
    seed_delegations(tmp_path)
    stage = make_delegation_work_evidence_stage(tmp_path / "index.db")

    def admit(actor: str, work: Callable[[], object]) -> object:
        with write_lease(actor, archive_root=tmp_path):
            return work()

    with arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError):
            open_isolated_write_connection(tmp_path / "index.db", purpose="test.unadmitted", archive_root=tmp_path)
        with stage_write_admission(admit):
            assert stage.check(tmp_path / "source.jsonl") is True
            assert stage.execute(tmp_path / "source.jsonl") is True
            assert stage.check(tmp_path / "source.jsonl") is False


@pytest.mark.parametrize("conventional_path", ["missing", "stale-shadow"])
def test_stage_uses_active_index_generation_after_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conventional_path: str
) -> None:
    """The active pointer wins when the conventional index is missing or stale."""
    seed_delegations(tmp_path)
    # Moving or copying only the main file of a WAL database drops the frames
    # still in its -wal; settle them into index.db first, as a promotion does.
    conn = sqlite3.connect(tmp_path / "index.db")
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    generation = tmp_path / ".index-generations" / "promoted"
    generation.mkdir(parents=True)
    if conventional_path == "missing":
        (tmp_path / "index.db").rename(generation / "index.db")
    else:
        shutil.copy2(tmp_path / "index.db", generation / "index.db")
    (tmp_path / ".index-active-pointer").write_text(str(generation / "index.db"), encoding="utf-8")

    real_needed = materializer.delegation_work_evidence_materialization_needed
    real_materialize = materializer.materialize_delegation_work_evidence_archive

    def needed(root: Path) -> bool:
        assert root == tmp_path
        return real_needed(root)

    def materialize(root: Path) -> int:
        assert root == tmp_path
        return real_materialize(root)

    monkeypatch.setattr(materializer, "delegation_work_evidence_materialization_needed", needed)
    monkeypatch.setattr(materializer, "materialize_delegation_work_evidence_archive", materialize)

    stage = make_delegation_work_evidence_stage(tmp_path / "index.db")
    subject = tmp_path / "source.jsonl"
    assert stage.check(subject) is True
    assert stage.execute(subject) is True
    assert delegation_work_evidence_materialization_needed(tmp_path) is False

    if conventional_path == "stale-shadow":
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM work_evidence_graphs").fetchone()[0] == 0
    with sqlite3.connect(generation / "index.db") as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM work_evidence_graphs WHERE graph_id = ?",
                (DELEGATION_WORK_EVIDENCE_GRAPH_ID,),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM work_evidence_nodes WHERE graph_id = ?",
                (DELEGATION_WORK_EVIDENCE_GRAPH_ID,),
            ).fetchone()[0]
            == 2
        )

    with sqlite3.connect(generation / "index.db") as conn:
        conn.execute("DELETE FROM session_links")
        conn.execute("DELETE FROM blocks WHERE tool_id = 'task-1'")
    assert stage.check(subject) is True
    assert stage.execute(subject) is True
    with sqlite3.connect(generation / "index.db") as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM work_evidence_nodes WHERE graph_id = ?",
                (DELEGATION_WORK_EVIDENCE_GRAPH_ID,),
            ).fetchone()[0]
            == 0
        )
    assert delegation_work_evidence_materialization_needed(tmp_path) is False


def test_convergence_stage_reports_probe_and_materialization_failures_as_pending_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed probe assumes work; a failed materialization reports degraded.

    PRs #5072/#5073 retired prose logging: ``polylogue.logging.emit`` writes
    structured events to its own sinks and never touches stdlib logging, so
    ``caplog.text`` is empty by construction. Per AGENTS.md this asserts the
    stable event token and its declared fields, not a sentence.

    Anti-vacuity: make ``check``'s ``except`` return False (or drop the
    ``emit``) and the probe half goes red on the return value or the missing
    ``daemon.stage.check_failed`` record; swallow the materialization
    exception into a success and the ``degraded``/``materialization_failed``
    terminal event disappears.
    """
    from polylogue.logging import capture

    seed_delegations(tmp_path)
    stage = make_delegation_work_evidence_stage(tmp_path / "index.db")

    def fail_probe(_archive_root: Path) -> bool:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(
        "polylogue.analysis.delegation_work_evidence_materializer.delegation_work_evidence_materialization_needed",
        fail_probe,
    )
    with capture() as records:
        assert stage.check(tmp_path / "source.jsonl") is True
    probe_failures = [record for record in records if record["event"] == "daemon.stage.check_failed"]
    assert [
        (record["stage"], record["outcome"], record["reason"], record["error_type"]) for record in probe_failures
    ] == [("delegation_work_evidence", "degraded", "probe_failed_assuming_work", "OperationalError")]

    def fail_materialization(_archive_root: Path) -> int:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(
        "polylogue.analysis.delegation_work_evidence_materializer.materialize_delegation_work_evidence_archive",
        fail_materialization,
    )
    with capture() as records:
        assert stage.execute(tmp_path / "source.jsonl") is False
    terminal = [record for record in records if record["event"] == "daemon.stage.execute.degraded"]
    assert [(record["stage"], record["outcome"], record["reason"], record["error_type"]) for record in terminal] == [
        ("delegation_work_evidence", "degraded", "materialization_failed", "OperationalError")
    ]


def test_delegation_population_larger_than_a_read_page_materializes_completely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every delegation is materialized; no population size is refused.

    Anti-vacuity (polylogue-zt8is): a single bounded read, or a count ceiling
    like the old 100,000-row refusal, stops at the first page and returns 2
    (or raises) instead of 3 for a three-delegation archive read one row a page.
    """
    seed_delegations(tmp_path, count=3)
    monkeypatch.setattr(materializer, "DELEGATION_READ_PAGE_ROWS", 1)

    assert delegation_work_evidence_materialization_needed(tmp_path) is True
    assert materialize_delegation_work_evidence_archive(tmp_path) == 3
    assert delegation_work_evidence_materialization_needed(tmp_path) is False


def test_delegation_snapshot_digest_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    """The incremental digest must not cause spurious re-materialization.

    Anti-vacuity: a digest that varies between two reads of unchanged rows
    (nondeterministic column order, unordered iteration, or a per-call salt)
    makes the equality assertion red; a digest that ignores materialized
    content makes the inequality assertion red.
    """
    seed_delegations(tmp_path)
    first = delegation_work_evidence_snapshot(tmp_path)
    second = delegation_work_evidence_snapshot(tmp_path)
    assert first.format() == second.format()

    assert materialize_delegation_work_evidence_archive(tmp_path) >= 1
    assert delegation_work_evidence_materialization_needed(tmp_path) is False

    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM session_links")
        conn.execute("DELETE FROM blocks WHERE tool_id = 'task-1'")
        conn.commit()
    assert delegation_work_evidence_snapshot(tmp_path).format() != first.format()
    assert delegation_work_evidence_materialization_needed(tmp_path) is True


def test_folding_pages_matches_the_one_pass_projection(tmp_path: Path) -> None:
    """The on-disk fold reproduces the single-pass graph, row for row.

    Anti-vacuity: a fold that replaced an attempt node already seen on an
    earlier page, instead of merging it, keeps one evidence ref and the later
    ``resolved`` state, where the one-pass projection unions both refs and
    keeps the stronger ``contradicted`` state.
    """
    from contextlib import closing

    from polylogue.analysis.delegation_work_evidence import materialize_delegation_work_evidence_graph
    from polylogue.core.refs import ObjectRef
    from tests.unit.insights.test_delegation_work_evidence import _row

    rows = [
        _row(parent_session_id="codex-session:a", instruction_tool_use_block_id="t1", mapping_state="quarantined"),
        _row(parent_session_id="codex-session:b", instruction_tool_use_block_id="t2", artifact_text="done"),
        _row(parent_session_id="codex-session:c", child_session_id=None, instruction_tool_use_block_id="t3"),
    ]
    snapshot = ObjectRef(kind="context-snapshot", object_id="delegations:fold")
    graph_id = materializer.DELEGATION_WORK_EVIDENCE_GRAPH_ID
    whole = materialize_delegation_work_evidence_graph(graph_id=graph_id, corpus_snapshot_ref=snapshot, rows=rows)
    with closing(sqlite3.connect(tmp_path / "scratch.db")) as scratch:
        materializer._create_scratch_graph(scratch)
        for row in rows:
            page = materialize_delegation_work_evidence_graph(
                graph_id=graph_id, corpus_snapshot_ref=snapshot, rows=[row]
            )
            materializer._fold_page(scratch, page)
        nodes = materializer._published_node_rows(scratch).fetchall()
        edges = scratch.execute("SELECT * FROM edges ORDER BY edge_ref").fetchall()

    # An attempt node's evidence_refs_json is produced by SQLite's
    # json_group_array at publish time (materializer._published_node_rows),
    # not Python's json.dumps: same sorted, deduplicated content, different
    # (compact) whitespace. Compare that one column parsed; every other
    # column, including every non-attempt node's own evidence_refs_json,
    # is still Python-serialized and compared verbatim.
    refs_index = materializer._NODE_COLUMNS.index("evidence_refs_json")

    def _normalized(row: tuple[object, ...]) -> tuple[object, ...]:
        return row[:refs_index] + (json.loads(str(row[refs_index])),) + row[refs_index + 1 :]

    assert [_normalized(row) for row in nodes] == [_normalized(materializer._node_row(node)) for node in whole.nodes]
    assert edges == [materializer._edge_row(edge) for edge in whole.edges]


def test_folding_a_high_fan_in_attempt_never_reads_its_own_accumulation(tmp_path: Path) -> None:
    """Each page's fold cost is its own rows, not the whole accumulated set.

    Anti-vacuity (Codex): fold an attempt node's evidence refs by reading and
    re-serializing the full accumulated ``evidence_refs_json`` column on
    every page (the union-in-Python approach) and this test still passes for
    correctness, but every one of the 200 folds below does a full read of
    the growing set -- what actually changed is that ``_fold_page`` no
    longer touches ``evidence_refs_json`` for an existing attempt node at
    all: assert that directly by checking the column is never read back
    larger than one page's own contribution during folding, i.e. the
    ``node_evidence_refs`` table alone holds the full accumulation.
    """
    from contextlib import closing

    from polylogue.analysis.delegation_work_evidence import materialize_delegation_work_evidence_graph
    from polylogue.core.refs import ObjectRef
    from tests.unit.insights.test_delegation_work_evidence import _row

    snapshot = ObjectRef(kind="context-snapshot", object_id="delegations:fan-in")
    graph_id = materializer.DELEGATION_WORK_EVIDENCE_GRAPH_ID
    with closing(sqlite3.connect(tmp_path / "scratch.db")) as scratch:
        materializer._create_scratch_graph(scratch)
        for index in range(200):
            row = _row(parent_session_id=f"codex-session:parent-{index}")
            page = materialize_delegation_work_evidence_graph(
                graph_id=graph_id, corpus_snapshot_ref=snapshot, rows=[row]
            )
            materializer._fold_page(scratch, page)
            # After every fold, the row's own evidence_refs_json column
            # never grows past its own page's contribution -- the union
            # lives entirely in node_evidence_refs, not in this column.
            stored = scratch.execute("SELECT evidence_refs_json FROM nodes WHERE node_kind = 'attempt'").fetchone()
            assert len(json.loads(stored[0])) <= 1

        published = {row[0]: row[3] for row in materializer._published_node_rows(scratch) if row[1] == "attempt"}
        (refs_json,) = published.values()
        assert len(json.loads(refs_json)) == 200


@pytest.mark.parametrize("direction", ["asc", "desc"])
def test_keyset_delegation_pages_equal_the_one_pass_read(tmp_path: Path, direction: Literal["asc", "desc"]) -> None:
    """Resuming after each page's last row reproduces the whole ordered read.

    Anti-vacuity (Codex): a key that is not total over the ORDER BY, or a
    comparison against the wrong direction, skips or repeats rows, so the
    paged rows differ from the single read.
    """
    from polylogue.archive.query.predicate import QueryBoolPredicate
    from polylogue.operations.operation_context import open_operation_read
    from polylogue.storage.sqlite.archive_tiers.archive_query_reads import DelegationPageKey

    seed_delegations(tmp_path, count=4)
    every = QueryBoolPredicate("and", ())
    with open_operation_read(tmp_path) as pinned:
        archive = pinned.archive
        whole = archive.query_delegations(every, limit=100, sort_direction=direction)
        paged = []
        after = None
        while page := archive.query_delegations(every, limit=1, sort_direction=direction, after=after):
            paged.extend(page)
            after = DelegationPageKey.after_row(page[-1])
    assert len(whole) == 4
    assert paged == whole


def test_a_keyset_key_coalesces_only_absent_ids_as_the_sql_order_does() -> None:
    """An empty-string block id is a key value, exactly as ``COALESCE`` sees it.

    Anti-vacuity: keying with ``or`` resumes after the child session id for
    such a row, a position the SQL order never had, so the next page skips or
    repeats rows.
    """
    from types import SimpleNamespace

    from polylogue.storage.sqlite.archive_tiers.archive_query_reads import DelegationPageKey

    empty = DelegationPageKey.after_row(
        cast(Any, SimpleNamespace(parent_session_id="p", instruction_tool_use_block_id="", child_session_id="c"))
    )
    assert (empty.order_key, empty.edge_only) == ("", False)
    edge = DelegationPageKey.after_row(
        cast(Any, SimpleNamespace(parent_session_id="p", instruction_tool_use_block_id=None, child_session_id="c"))
    )
    assert (edge.order_key, edge.edge_only) == ("c", True)


def test_materializer_reads_delegations_by_keyset_not_offset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each materializer page resumes from a key; none rescans by offset.

    Anti-vacuity (Codex): an OFFSET page makes SQLite visit and discard every
    earlier row, so the full read is quadratic in the delegation count.
    """
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    seed_delegations(tmp_path, count=3)
    monkeypatch.setattr(materializer, "DELEGATION_READ_PAGE_ROWS", 1)
    calls: list[tuple[int, object]] = []
    real = ArchiveStore.query_delegations

    def spy(self: ArchiveStore, *args: object, **kwargs: object) -> object:
        calls.append((int(kwargs.get("offset", 0)), kwargs.get("after")))  # type: ignore[call-overload]
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ArchiveStore, "query_delegations", spy)
    assert materialize_delegation_work_evidence_archive(tmp_path) == 3
    assert [offset for offset, _after in calls] == [0] * len(calls)
    assert calls[0][1] is None and all(after is not None for _offset, after in calls[1:])


def test_delegation_pages_are_bounded_by_text_bytes_and_still_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page ends at its text-byte budget, and the population is complete.

    Anti-vacuity (Codex): a row-count page alone holds up to a thousand
    multi-megabyte tool results in memory. Under a one-byte budget each page
    must carry exactly one row, and every row must still be materialized.
    """
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    seed_delegations(tmp_path, count=3)
    monkeypatch.setattr(materializer, "DELEGATION_READ_PAGE_TEXT_BYTES", 1)
    page_sizes: list[int] = []
    real = ArchiveStore.query_delegations

    def spy(self: ArchiveStore, *args: object, **kwargs: object) -> object:
        page = real(self, *args, **kwargs)  # type: ignore[arg-type]
        page_sizes.append(len(page))
        return page

    monkeypatch.setattr(ArchiveStore, "query_delegations", spy)
    assert materialize_delegation_work_evidence_archive(tmp_path) == 3
    assert page_sizes == [1, 1, 1, 0]


def test_a_keyset_delegation_page_seeks_by_index_instead_of_sorting(tmp_path: Path) -> None:
    """The keyset page's ORDER BY is served by an index, with no sort step.

    Anti-vacuity (Codex): with no index over the query order, SQLite used a
    temporary B-tree for the order, so every page re-sorted the rest of a
    parent's cohort and the read stayed quadratic.
    """
    from polylogue.archive.query.predicate import QueryBoolPredicate
    from polylogue.operations.operation_context import open_operation_read
    from polylogue.storage.sqlite.archive_tiers.archive_query_reads import DelegationPageKey

    seed_delegations(tmp_path, count=3)
    statements: list[str] = []
    with open_operation_read(tmp_path) as pinned:
        archive = pinned.archive
        (first,) = archive.query_delegations(QueryBoolPredicate("and", ()), limit=1)
        archive._conn.set_trace_callback(statements.append)
        try:
            archive.query_delegations(QueryBoolPredicate("and", ()), limit=1, after=DelegationPageKey.after_row(first))
        finally:
            archive._conn.set_trace_callback(None)
        (statement,) = [text for text in statements if "FROM delegation_facts" in text]
        plan = [str(row[-1]) for row in archive._conn.execute(f"EXPLAIN QUERY PLAN {statement}")]
    assert not any("TEMP B-TREE" in detail for detail in plan), plan
