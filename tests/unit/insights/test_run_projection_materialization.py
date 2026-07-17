"""Run-projection relation reads: source-derived only, no materialized cache.

polylogue-dab stopped materializing session_runs/session_observed_events/
session_context_snapshots into cache tables (they no longer exist in the
schema at all). These tests exercise the CTE-based source-derived read
path in run_projection_relations.py / session_insight_run_projection_reads.py
directly against a real archive, replacing the old materialize-then-read
parity tests that exercised the now-deleted write path
(session_insight_run_projection_writes.py, storage.py's
replace_session_runs_sync family). See polylogue-itvd.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId
from polylogue.insights.transforms import compile_session_digest
from polylogue.storage.query_models import RunProjectionListQuery
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.queries.session_insight_run_projection_reads import (
    list_context_snapshots,
    list_observed_events,
    list_runs,
)


def _session() -> Session:
    return Session(
        id=SessionId("codex-session:demo"),
        origin=Origin.CODEX_SESSION,
        title="Ship the backlog",
        git_branch="feature/demo",
        working_directories=("/realm/project/polylogue",),
        messages=MessageCollection(
            messages=[
                Message(
                    id="m1",
                    role=Role.USER,
                    text="Goal: burn down the backlog\nNext: merge PR #1911",
                ),
                Message(
                    id="m2",
                    role=Role.ASSISTANT,
                    text="Review posted on PR #1911\nRead review on PR #1911",
                    blocks=[
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "tool_input": {"command": "devtools verify --quick"},
                        },
                        {
                            "type": "tool_result",
                            "tool_id": "tool-1",
                            "text": "ruff check ... ok\n20 passed",
                        },
                    ],
                ),
            ]
        ),
    )


async def test_run_projection_reads_source_rows_for_claude_code_session(tmp_path: Path) -> None:
    from tests.infra.storage_records import SessionBuilder

    db_path = tmp_path / "index.db"
    (
        SessionBuilder(db_path, "source-claude-code")
        .provider("claude-code")
        .git_branch("feature/source-runs")
        .title("source-derived run projection")
        .add_message(
            "m-tool",
            role="assistant",
            text="Inspected the run projection relation.",
            blocks=[
                {"type": "tool_use", "id": "tool-1", "name": "Bash", "tool_input": {"command": "pytest -k runs"}},
                {"type": "tool_result", "tool_id": "tool-1", "text": "passed", "tool_result_exit_code": 0},
            ],
        )
        .save()
    )
    session_id = "claude-code-session:ext-source-claude-code"
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        runs = await list_runs(conn, RunProjectionListQuery(session_id=session_id, role="main", limit=None))
        events = await list_observed_events(
            conn,
            RunProjectionListQuery(session_id=session_id, kind="tool_finished", limit=None),
        )
        snapshots = await list_context_snapshots(
            conn,
            RunProjectionListQuery(session_id=session_id, boundary="session_start", limit=None),
        )

    assert [record.run.run_ref.format() for record in runs] == [f"run:{session_id}"]
    assert runs[0].run.git_branch == "feature/source-runs"
    assert [record.event.kind for record in events] == ["tool_finished"]
    assert events[0].event.tool_name == "Bash"
    assert events[0].event.status == "ok"
    assert [record.snapshot.snapshot_ref.format() for record in snapshots] == [
        f"context-snapshot:{session_id}:session_start"
    ]


async def test_run_projection_reads_source_rows_for_codex_session(tmp_path: Path) -> None:
    from tests.infra.storage_records import SessionBuilder

    db_path = tmp_path / "index.db"
    (
        SessionBuilder(db_path, "source-codex")
        .provider("codex")
        .git_branch("feature/no-cache")
        .title("codex source relation")
        .add_message(
            "m-tool",
            role="assistant",
            text="Ran a tool without run cache tables.",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "tool-absent",
                    "name": "mcp__serena__find_symbol",
                    "tool_input": {"name_path": "ArchiveStore/query_runs"},
                },
                {"type": "tool_result", "tool_id": "tool-absent", "text": "found"},
            ],
        )
        .save()
    )
    session_id = "codex-session:ext-source-codex"
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        runs = await list_runs(conn, RunProjectionListQuery(session_id=session_id, role="main", limit=None))
        events = await list_observed_events(
            conn,
            RunProjectionListQuery(session_id=session_id, kind="tool_finished", query="serena", limit=None),
        )
        snapshots = await list_context_snapshots(
            conn,
            RunProjectionListQuery(session_id=session_id, boundary="session_start", limit=None),
        )

    assert [record.run.run_ref.format() for record in runs] == [f"run:{session_id}"]
    assert runs[0].run.harness == "codex"
    assert [record.event.tool_name for record in events] == ["mcp__serena__find_symbol"]
    assert [record.snapshot.run_ref.format() for record in snapshots] == [f"run:{session_id}"]


def test_subagent_and_child_main_runs_do_not_collide(tmp_path: Path) -> None:
    """A subagent session's run row is distinct from its own child main run.

    Source-derived subagent detection (run_projection_relations.py) keys
    role/boundary off ``sessions.branch_type = 'subagent'`` directly, one
    run row per subagent session -- unlike the old materialized writer,
    which synthesized N virtual subagent run rows per Task-tool report
    under the parent's session_id. The collision this test used to guard
    against (duplicate report ids producing colliding synthesized run_refs)
    has no equivalent in the new model: each subagent session gets exactly
    one run row, keyed by its own session_id, so it structurally cannot
    collide with another session's main run_ref.
    """
    conn = sqlite3.connect(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    conn.executescript(INDEX_DDL)
    conn.execute(
        "INSERT INTO sessions(native_id, origin, content_hash) VALUES(?, ?, ?)",
        ("parent", "codex-session", b"\x00" * 32),
    )
    conn.execute(
        "INSERT INTO sessions(native_id, origin, content_hash, branch_type, parent_session_id) VALUES(?, ?, ?, ?, ?)",
        ("child", "codex-session", b"\x01" * 32, "subagent", "codex-session:parent"),
    )
    conn.commit()

    from polylogue.storage.sqlite.run_projection_relations import run_relation_sql

    rows = conn.execute(f"{run_relation_sql()} SELECT run_ref, session_id, role FROM runs ORDER BY run_ref").fetchall()
    assert [(row["run_ref"], row["session_id"], row["role"]) for row in rows] == [
        ("run:codex-session:child", "codex-session:child", "subagent"),
        ("run:codex-session:parent", "codex-session:parent", "main"),
    ]
    conn.close()


def test_continuation_session_projection_boundary_is_resume() -> None:
    """polylogue-aoe5: a genuine continuation/resume session gets boundary='resume'."""
    session = _session().model_copy(
        update={
            "id": SessionId("codex-session:demo-continuation"),
            "parent_id": SessionId("codex-session:demo-root"),
            "branch_type": BranchType.CONTINUATION,
        }
    )
    assert session.is_continuation

    projection = compile_session_digest(session, session_links=()).run_projection
    main_snapshot = projection.context_snapshots[0]
    main_run = projection.runs[0]

    assert main_snapshot.boundary == "resume"
    assert main_snapshot.snapshot_ref.object_id == "codex-session:demo-continuation:resume"
    assert main_run.context_snapshot_ref == main_snapshot.snapshot_ref

    # A fresh (non-continuation) session is unaffected.
    fresh_projection = compile_session_digest(_session(), session_links=()).run_projection
    assert fresh_projection.context_snapshots[0].boundary == "session_start"


async def test_continuation_session_resume_boundary_read_through_source(tmp_path: Path) -> None:
    """The source-derived read path reflects branch_type='continuation' directly.

    The cheap ``sessions``-derived relation (``run_projection_relations.py``)
    must report boundary='resume' for a continuation session without any
    separate materialization step.
    """
    from tests.infra.storage_records import SessionBuilder

    db_path = tmp_path / "index.db"
    (
        SessionBuilder(db_path, "resume-boundary-root")
        .provider("claude-code")
        .title("Resume Boundary Root")
        .add_message("root-u1", role="user", text="Start the work.")
        .save()
    )
    (
        SessionBuilder(db_path, "resume-boundary-child")
        .provider("claude-code")
        .title("Resume Boundary Continuation")
        .parent_session("ext-resume-boundary-root")
        .branch_type("continuation")
        .add_message("child-u1", role="user", text="Continue the work after a crash.")
        .save()
    )
    session_id = "claude-code-session:ext-resume-boundary-child"

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        source_snapshots = await list_context_snapshots(conn, RunProjectionListQuery(session_id=session_id, limit=None))
        assert [record.snapshot.boundary for record in source_snapshots] == ["resume"]
        assert source_snapshots[0].snapshot.snapshot_ref.object_id == f"{session_id}:resume"

        resume_filtered = await list_context_snapshots(
            conn, RunProjectionListQuery(session_id=session_id, boundary="resume", limit=None)
        )
        assert [record.snapshot.snapshot_ref for record in resume_filtered] == [
            source_snapshots[0].snapshot.snapshot_ref
        ]

        main_runs = await list_runs(conn, RunProjectionListQuery(session_id=session_id, role="main", limit=None))
        assert main_runs[0].run.context_snapshot_ref == source_snapshots[0].snapshot.snapshot_ref
