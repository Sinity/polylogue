"""Parity test for the materialized run-projection read-models.

Rows written by the run-projection materializer must hydrate back to exactly the
output of ``compile_session_digest(session, session_links=()).run_projection`` —
the same compute the runtime query path performs. This guards the #2384 keystone:
the derived tables are a faithful materialization, not a lossy re-derivation.
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
from polylogue.insights.transforms import compile_session_digest
from polylogue.storage.insights.session.run_projection_rows import (
    build_session_context_snapshot_records,
    build_session_observed_event_records,
    build_session_run_records,
)
from polylogue.storage.query_models import RunProjectionListQuery
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.queries.session_insight_run_projection_reads import (
    list_context_snapshots,
    list_observed_events,
    list_runs,
)
from polylogue.storage.sqlite.queries.session_insight_run_projection_writes import (
    replace_session_context_snapshots,
    replace_session_observed_events,
    replace_session_runs,
)
from polylogue.types import SessionId

_MATERIALIZED_AT = "2026-06-27T00:00:00+00:00"


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


async def _seed_archive(conn: aiosqlite.Connection) -> str:
    conn.row_factory = sqlite3.Row
    await conn.executescript(INDEX_DDL)
    await conn.execute(
        "INSERT INTO sessions(native_id, origin, content_hash) VALUES(?, ?, ?)",
        ("demo", "codex-session", b"\x00" * 32),
    )
    await conn.commit()
    return "codex-session:demo"


async def test_run_projection_materialization_parity(tmp_path: Path) -> None:
    session = _session()
    projection = compile_session_digest(session, session_links=()).run_projection

    run_records = build_session_run_records(projection, materialized_at=_MATERIALIZED_AT)
    event_records = build_session_observed_event_records(projection, materialized_at=_MATERIALIZED_AT)
    snapshot_records = build_session_context_snapshot_records(projection, materialized_at=_MATERIALIZED_AT)

    # The projection exercises every table: a main run, the session_started +
    # tool_finished + review observed events, and the session_start snapshot.
    assert len(run_records) == len(projection.runs) >= 1
    assert len(event_records) == len(projection.events) >= 1
    assert len(snapshot_records) == len(projection.context_snapshots) >= 1

    db_path = tmp_path / "index.db"
    async with aiosqlite.connect(db_path) as conn:
        session_id = await _seed_archive(conn)
        await replace_session_runs(conn, session_id, run_records, 0)
        await replace_session_observed_events(conn, session_id, event_records, 0)
        await replace_session_context_snapshots(conn, session_id, snapshot_records, 0)

        got_runs = await list_runs(conn, RunProjectionListQuery(session_id=session_id, limit=None))
        got_events = await list_observed_events(conn, RunProjectionListQuery(session_id=session_id, limit=None))
        got_snapshots = await list_context_snapshots(conn, RunProjectionListQuery(session_id=session_id, limit=None))

        # Query reads are source-derived first: cheap canonical rows are served
        # from sessions/blocks, while richer non-duplicate materialized rows
        # remain available as enrichment.
        assert [record.run.run_ref for record in got_runs] == [projection.runs[0].run_ref]
        assert {record.event.kind for record in got_events} >= {"session_started", "tool_finished"}
        assert [record.snapshot.snapshot_ref for record in got_snapshots] == [
            projection.context_snapshots[0].snapshot_ref
        ]

        # Typed columns back SQL filters.
        main_runs = await list_runs(conn, RunProjectionListQuery(session_id=session_id, role="main", limit=None))
        assert [record.run.run_ref for record in main_runs] == [
            run.run_ref for run in projection.runs if run.role == "main"
        ]

        started = await list_observed_events(conn, RunProjectionListQuery(kind="session_started", limit=None))
        assert [record.event.kind for record in started] == ["session_started"]

        session_start_snaps = await list_context_snapshots(
            conn, RunProjectionListQuery(boundary="session_start", limit=None)
        )
        assert all(record.snapshot.boundary == "session_start" for record in session_start_snaps)
        assert session_start_snaps

        # Re-materialization is idempotent (DELETE-then-INSERT, no duplicate rows).
        await replace_session_runs(conn, session_id, run_records, 0)
        rerun = await list_runs(conn, RunProjectionListQuery(session_id=session_id, limit=None))
        assert [record.run.run_ref for record in rerun] == [projection.runs[0].run_ref]


async def test_run_projection_reads_source_rows_when_cache_tables_empty(tmp_path: Path) -> None:
    from tests.infra.storage_records import SessionBuilder

    db_path = tmp_path / "index.db"
    (
        SessionBuilder(db_path, "source-empty-cache")
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
    session_id = "claude-code-session:ext-source-empty-cache"
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        await conn.execute("DELETE FROM session_runs")
        await conn.execute("DELETE FROM session_observed_events")
        await conn.execute("DELETE FROM session_context_snapshots")
        await conn.commit()

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


async def test_run_projection_reads_source_rows_when_cache_tables_absent(tmp_path: Path) -> None:
    from tests.infra.storage_records import SessionBuilder

    db_path = tmp_path / "index.db"
    (
        SessionBuilder(db_path, "source-absent-cache")
        .provider("codex")
        .git_branch("feature/no-cache")
        .title("absent cache source relation")
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
    session_id = "codex-session:ext-source-absent-cache"
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        await conn.execute("DROP TABLE session_runs")
        await conn.execute("DROP TABLE session_observed_events")
        await conn.execute("DROP TABLE session_context_snapshots")
        await conn.commit()

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
    """Duplicate parent-side subagent ids and a child main run must coexist."""
    from polylogue.insights.run_projection import build_run_projection
    from polylogue.insights.transforms import SubagentReport, TransformRawRef
    from polylogue.storage.insights.session.storage import replace_session_runs_bulk_sync

    parent_ref = TransformRawRef(session_id="codex-session:parent")
    report_0_ref = TransformRawRef(session_id="codex-session:parent", message_id="m1", block_index=0, ref_kind="block")
    report_1_ref = TransformRawRef(session_id="codex-session:parent", message_id="m2", block_index=0, ref_kind="block")
    child_ref = TransformRawRef(session_id="codex-session:child")
    parent = build_run_projection(
        session_id="codex-session:parent",
        source_origin="codex-session",
        title="parent",
        git_branch=None,
        working_directories=(),
        session_raw_refs=(parent_ref,),
        tool_summaries=(),
        subagent_reports=(
            SubagentReport(
                subagent_type="Explore",
                tool_id="shared-tool",
                task_id="task-a",
                child_session_id="codex-session:child",
                prompt="first",
                final_report_preview="first done",
                raw_refs=(report_0_ref,),
            ),
            SubagentReport(
                subagent_type="Explore",
                tool_id="shared-tool",
                task_id="task-b",
                child_session_id="codex-session:child",
                prompt="second",
                final_report_preview="second done",
                raw_refs=(report_1_ref,),
            ),
        ),
        session_digest_events=(),
    )
    parent_again = build_run_projection(
        session_id="codex-session:parent",
        source_origin="codex-session",
        title="parent",
        git_branch=None,
        working_directories=(),
        session_raw_refs=(parent_ref,),
        tool_summaries=(),
        subagent_reports=(
            SubagentReport(
                subagent_type="Explore",
                tool_id="shared-tool",
                task_id="task-a",
                child_session_id="codex-session:child",
                prompt="first",
                final_report_preview="first done",
                raw_refs=(report_0_ref,),
            ),
            SubagentReport(
                subagent_type="Explore",
                tool_id="shared-tool",
                task_id="task-b",
                child_session_id="codex-session:child",
                prompt="second",
                final_report_preview="second done",
                raw_refs=(report_1_ref,),
            ),
        ),
        session_digest_events=(),
    )
    child = build_run_projection(
        session_id="codex-session:child",
        source_origin="codex-session",
        title="child",
        git_branch=None,
        working_directories=(),
        session_raw_refs=(child_ref,),
        tool_summaries=(),
        subagent_reports=(),
        session_digest_events=(),
    )
    assert [run.run_ref for run in parent.runs] == [run.run_ref for run in parent_again.runs]

    conn = sqlite3.connect(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    conn.executescript(INDEX_DDL)
    for native in ("parent", "child"):
        conn.execute(
            "INSERT INTO sessions(native_id, origin, content_hash) VALUES(?, ?, ?)",
            (native, "codex-session", b"\x00" * 32),
        )
    conn.commit()

    replace_session_runs_bulk_sync(
        conn,
        {
            "codex-session:parent": build_session_run_records(parent, materialized_at=_MATERIALIZED_AT),
            "codex-session:child": build_session_run_records(child, materialized_at=_MATERIALIZED_AT),
        },
    )

    rows = conn.execute("SELECT run_ref, session_id, role FROM session_runs ORDER BY run_ref").fetchall()
    assert [(row["run_ref"], row["session_id"], row["role"]) for row in rows] == [
        ("run:codex-session:child", "codex-session:child", "main"),
        ("run:codex-session:parent", "codex-session:parent", "main"),
        ("run:codex-session:parent:subagent:0:shared-tool", "codex-session:parent", "subagent"),
        ("run:codex-session:parent:subagent:1:shared-tool", "codex-session:parent", "subagent"),
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


async def test_continuation_session_resume_boundary_read_through_source_and_materialized(tmp_path: Path) -> None:
    """The SQL source fallback and the materializer agree on boundary='resume'.

    The cheap ``sessions``-derived read path (``run_projection_relations.py``)
    must reflect ``branch_type='continuation'`` without requiring the session
    to be re-materialized, and materializing afterward must not produce a
    duplicate context-snapshot row for the same run.
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

        # session_context_snapshots is populated by daemon convergence, not by
        # SessionBuilder.save(); the source/cheap path must already carry the
        # resume boundary before any materialization happens.
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

        # Now materialize explicitly and confirm the read path still returns
        # exactly one resume-boundary snapshot for this session (no duplicate
        # row from the union of source + materialized rows).
        session = Session(
            id=SessionId(session_id),
            origin=Origin.CLAUDE_CODE_SESSION,
            title="Resume Boundary Continuation",
            parent_id=SessionId("claude-code-session:ext-resume-boundary-root"),
            branch_type=BranchType.CONTINUATION,
            messages=MessageCollection(
                messages=[Message(id="child-u1", role=Role.USER, text="Continue the work after a crash.")]
            ),
        )
        projection = compile_session_digest(session, session_links=()).run_projection
        snapshot_records = build_session_context_snapshot_records(projection, materialized_at=_MATERIALIZED_AT)
        await replace_session_context_snapshots(conn, session_id, snapshot_records, 0)

        after_materialize = await list_context_snapshots(
            conn, RunProjectionListQuery(session_id=session_id, limit=None)
        )
        assert [record.snapshot.boundary for record in after_materialize] == ["resume"]
