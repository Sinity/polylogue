"""Archive-backed delegation work-evidence materialization."""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path

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
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.write_guard import install_archive_write_guard
from polylogue.storage.sqlite.write_lease import UnleasedWriteError, arm_write_lease_enforcement, write_lease


def _seed_delegation(archive_root: Path) -> None:
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(archive_root)
    initialize_archive_database(archive_root / "index.db", ArchiveTier.INDEX)
    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms)
            VALUES ('parent', 'claude-code-session', 'Parent', ?, 1, 2)
            """,
            (b"p" * 32,),
        )
        parent_id = conn.execute(
            "SELECT session_id FROM sessions WHERE origin = 'claude-code-session' AND native_id = 'parent'"
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, title, content_hash, created_at_ms, updated_at_ms, branch_type, parent_session_id
            ) VALUES ('child', 'claude-code-session', 'Child', ?, 1, 2, 'subagent', ?)
            """,
            (b"c" * 32, parent_id),
        )
        child_id = conn.execute(
            "SELECT session_id FROM sessions WHERE origin = 'claude-code-session' AND native_id = 'child'"
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash, occurred_at_ms)
            VALUES (?, 'dispatch', 0, 'assistant', 'message', ?, 1)
            """,
            (parent_id, b"m" * 32),
        )
        message_id = conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? AND native_id = 'dispatch'", (parent_id,)
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, tool_name, tool_id, semantic_type, tool_input
            ) VALUES (?, ?, 0, 'tool_use', 'Task', 'task-1', 'subagent', '{"prompt":"review"}')
            """,
            (message_id, parent_id),
        )
        # block_id is generated as message_id || ':' || position; a literal
        # tool_id ("task-1") is a different value and would leave the join
        # in delegation_facts_source (index.py) unresolved.
        block_id = conn.execute(
            "SELECT block_id FROM blocks WHERE message_id = ? AND position = 0", (message_id,)
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO session_links (
                src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id,
                parent_tool_use_block_id, observed_at_ms
            ) VALUES (?, 'claude-code-session', 'parent', 'subagent', ?, ?, 1)
            """,
            (child_id, parent_id, block_id),
        )


def test_materializer_replaces_archive_projection_and_tracks_delegation_freshness(tmp_path: Path) -> None:
    _seed_delegation(tmp_path)

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

    _seed_delegation(tmp_path)
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
    assert len(connections) == 2
    assert connections[0] is connections[1]


def test_delegation_stage_reads_without_daemon_writer_lease(tmp_path: Path) -> None:
    """The freshness probe and materialization read run outside writer admission."""
    _seed_delegation(tmp_path)
    stage = make_delegation_work_evidence_stage(tmp_path / "index.db")

    def admit(actor: str, work: Callable[[], object]) -> object:
        with write_lease(actor, archive_root=tmp_path):
            return work()

    with install_archive_write_guard(), arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError):
            sqlite3.connect(tmp_path / "index.db")
        with stage_write_admission(admit):
            assert stage.check(tmp_path / "source.jsonl") is True
            assert stage.execute(tmp_path / "source.jsonl") is True
            assert stage.check(tmp_path / "source.jsonl") is False


@pytest.mark.parametrize("conventional_path", ["missing", "stale-shadow"])
def test_stage_uses_active_index_generation_after_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conventional_path: str
) -> None:
    """The active pointer wins when the conventional index is missing or stale."""
    _seed_delegation(tmp_path)
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
    ``caplog.text`` is empty by construction. Per CLAUDE.md this asserts the
    stable event token and its declared fields, not a sentence.

    Anti-vacuity: make ``check``'s ``except`` return False (or drop the
    ``emit``) and the probe half goes red on the return value or the missing
    ``daemon.stage.check_failed`` record; swallow the materialization
    exception into a success and the ``degraded``/``materialization_failed``
    terminal event disappears.
    """
    from polylogue.logging import capture

    _seed_delegation(tmp_path)
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


def test_delegation_snapshot_refuses_row_ceiling_on_the_freshness_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The freshness probe is bounded, not just the materialize path.

    Anti-vacuity: restoring the unbounded
    ``SELECT * FROM delegation_facts ... .fetchall()`` snapshot makes
    ``delegation_work_evidence_materialization_needed`` return a bool for an
    over-ceiling population instead of raising, and this test goes red.
    """
    _seed_delegation(tmp_path)
    monkeypatch.setattr(materializer, "MAX_DELEGATION_SNAPSHOT_ROWS", 0)

    with pytest.raises(ValueError, match="bounded population"):
        delegation_work_evidence_materialization_needed(tmp_path)

    # The materialize path inherits the same bound, because it digests first.
    with pytest.raises(ValueError, match="bounded population"):
        materialize_delegation_work_evidence_archive(tmp_path)


def test_delegation_snapshot_refuses_byte_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Few rows carrying attacker-sized text are refused too.

    Anti-vacuity: a row-count-only bound (or the original unbounded snapshot)
    accepts a small row set with multi-MB payloads, and this test goes red.
    """
    _seed_delegation(tmp_path)
    monkeypatch.setattr(materializer, "MAX_DELEGATION_SNAPSHOT_BYTES", 1)

    with pytest.raises(ValueError, match="bounded population"):
        delegation_work_evidence_materialization_needed(tmp_path)


def test_delegation_snapshot_digest_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    """The incremental digest must not cause spurious re-materialization.

    Anti-vacuity: a digest that varies between two reads of unchanged rows
    (nondeterministic column order, unordered iteration, or a per-call salt)
    makes the equality assertion red; a digest that ignores materialized
    content makes the inequality assertion red.
    """
    _seed_delegation(tmp_path)
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
