"""Archive-backed delegation work-evidence materialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.analysis.delegation_work_evidence_materializer import (
    DELEGATION_WORK_EVIDENCE_GRAPH_ID,
    delegation_work_evidence_materialization_needed,
    materialize_delegation_work_evidence_archive,
)
from polylogue.daemon.convergence_stages import make_delegation_work_evidence_stage
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _seed_delegation(archive_root: Path) -> None:
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
            INSERT INTO session_links (
                src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id, observed_at_ms
            ) VALUES (?, 'claude-code-session', 'parent', 'subagent', ?, 1)
            """,
            (child_id, parent_id),
        )
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

    # A replacement must remove rows that disappeared from the canonical view;
    # otherwise stale nodes remain traversable after source evidence changes.
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM delegation_facts")

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


def test_convergence_stage_reports_probe_and_materialization_failures_as_pending_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _seed_delegation(tmp_path)
    stage = make_delegation_work_evidence_stage(tmp_path / "index.db")

    def fail_probe(_archive_root: Path) -> bool:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(
        "polylogue.analysis.delegation_work_evidence_materializer.delegation_work_evidence_materialization_needed",
        fail_probe,
    )
    with caplog.at_level("WARNING"):
        assert stage.check(tmp_path / "source.jsonl") is True
    assert "delegation work-evidence freshness probe failed" in caplog.text

    def fail_materialization(_archive_root: Path) -> int:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(
        "polylogue.analysis.delegation_work_evidence_materializer.materialize_delegation_work_evidence_archive",
        fail_materialization,
    )
    with caplog.at_level("WARNING"):
        assert stage.execute(tmp_path / "source.jsonl") is False
    assert "delegation work-evidence materialization failed" in caplog.text
