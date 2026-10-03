"""Source-derived run projections through the canonical archive query route.

These controls preserve tool pairing, typed hydration, continuation boundaries
and session-scoped window costs without a parallel async read facade.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

pytestmark = pytest.mark.storage_scale

from polylogue.analysis.transforms import compile_session_digest
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.query.expression import parse_unit_source_expression
from polylogue.archive.query.predicate import QueryPredicate
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL


def _predicate(expression: str) -> QueryPredicate:
    source = parse_unit_source_expression(expression)
    assert source is not None
    return source.predicate


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


def test_run_projection_reads_source_rows_for_claude_code_session(tmp_path: Path) -> None:
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
    with ArchiveStore(db_path.parent, read_only=True) as archive:
        runs = archive.query_runs(_predicate(f"runs where session.id:{session_id} AND role:main"))
        events = archive.query_observed_events(
            _predicate(f"observed-events where session.id:{session_id} AND kind:tool_finished")
        )
        snapshots = archive.query_context_snapshots(
            _predicate(f"context-snapshots where session.id:{session_id} AND boundary:session_start")
        )

    assert [record.run.run_ref.format() for record in runs] == [f"run:{session_id}"]
    assert runs[0].run.git_branch == "feature/source-runs"
    assert [record.event.kind for record in events] == ["tool_finished"]
    assert events[0].event.tool_name == "Bash"
    assert events[0].event.status == "ok"
    assert [record.snapshot.snapshot_ref.format() for record in snapshots] == [
        f"context-snapshot:{session_id}:session_start"
    ]


def test_run_projection_reads_source_rows_for_codex_session(tmp_path: Path) -> None:
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
                {"type": "tool_result", "tool_id": "tool-absent", "text": "found", "tool_result_is_error": 0},
            ],
        )
        .save()
    )
    session_id = "codex-session:ext-source-codex"
    with ArchiveStore(db_path.parent, read_only=True) as archive:
        runs = archive.query_runs(_predicate(f"runs where session.id:{session_id} AND role:main"))
        events = archive.query_observed_events(
            _predicate(
                f"observed-events where session.id:{session_id} AND kind:tool_finished AND tool:mcp__serena__find_symbol"
            )
        )
        snapshots = archive.query_context_snapshots(
            _predicate(f"context-snapshots where session.id:{session_id} AND boundary:session_start")
        )

    assert [record.run.run_ref.format() for record in runs] == [f"run:{session_id}"]
    assert runs[0].run.harness == "codex"
    assert [record.event.tool_name for record in events] == ["mcp__serena__find_symbol"]
    assert [record.snapshot.run_ref.format() for record in snapshots] == [f"run:{session_id}"]


def test_a_reused_tool_id_pairs_by_rank_and_does_not_fan_out(tmp_path: Path) -> None:
    """polylogue-3sic0: one tool_id used twice yields two events, not four.

    ``tool_finished_base`` joined ``blocks`` to ``blocks`` on
    ``(session_id, tool_id)`` alone. A provider that re-emits one tool_id --
    a retry, a loop -- has N uses and M results under it, and the equality
    join returned every N*M combination: extra tool_finished events, each
    pairing a use with a result it never produced. ``action_pairs`` (the
    canonical pairing behind the ``actions`` view) already ranks both sides
    by transcript order and pairs same-rank rows; this relation now reads it.

    Anti-vacuity, verified by reverting: restore the
    ``JOIN blocks r ON r.session_id = u.session_id AND r.tool_id = u.tool_id
    AND r.block_type = 'tool_result'`` join and this fails with four events
    instead of two, and with the first (successful) retry reported as
    ``failed`` because it cross-pairs onto the second result.
    """
    from tests.infra.storage_records import SessionBuilder

    db_path = tmp_path / "index.db"
    (
        SessionBuilder(db_path, "reused-tool-id")
        .provider("codex")
        .title("one tool_id, two attempts")
        .add_message(
            "m-first",
            role="assistant",
            text="First attempt.",
            blocks=[
                {"type": "tool_use", "id": "tool-retry", "name": "Bash", "tool_input": {"command": "first-attempt"}},
                {"type": "tool_result", "tool_id": "tool-retry", "text": "ok", "tool_result_exit_code": 0},
            ],
        )
        .add_message(
            "m-second",
            role="assistant",
            text="Second attempt under the same id.",
            blocks=[
                {"type": "tool_use", "id": "tool-retry", "name": "Bash", "tool_input": {"command": "second-attempt"}},
                {"type": "tool_result", "tool_id": "tool-retry", "text": "boom", "tool_result_exit_code": 1},
            ],
        )
        .save()
    )
    session_id = "codex-session:ext-reused-tool-id"
    with ArchiveStore(db_path.parent, read_only=True) as archive:
        events = archive.query_observed_events(
            _predicate(f"observed-events where session.id:{session_id} AND kind:tool_finished")
        )

    # Two uses, two results, two events -- and each use keeps its OWN result,
    # so the successful first attempt is not reported through the second
    # attempt's failure.
    assert [(record.event.command, record.event.status) for record in events] == [
        ("first-attempt", "ok"),
        ("second-attempt", "failed"),
    ]


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


def test_continuation_session_resume_boundary_read_through_source(tmp_path: Path) -> None:
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

    with ArchiveStore(db_path.parent, read_only=True) as archive:
        source_snapshots = archive.query_context_snapshots(
            _predicate(f"context-snapshots where session.id:{session_id}")
        )
        assert [record.snapshot.boundary for record in source_snapshots] == ["resume"]
        assert source_snapshots[0].snapshot.snapshot_ref.object_id == f"{session_id}:resume"

        resume_filtered = archive.query_context_snapshots(
            _predicate(f"context-snapshots where session.id:{session_id} AND boundary:resume")
        )
        assert [record.snapshot.snapshot_ref for record in resume_filtered] == [
            source_snapshots[0].snapshot.snapshot_ref
        ]

        main_runs = archive.query_runs(_predicate(f"runs where session.id:{session_id} AND role:main"))
        assert main_runs[0].run.context_snapshot_ref == source_snapshots[0].snapshot.snapshot_ref


def test_run_projection_relations_expose_typed_columns_not_a_payload_bundle(tmp_path: Path) -> None:
    """The three relations carry typed columns, with no payload_json round trip.

    polylogue-dab.1: `tool_finished_base` already computes tool_name, tool_id,
    command, handler_kind and status as typed columns. The relation used to
    bundle those same five values into a `json_object(...) AS payload_json`,
    and every consumer -- the ObservedEvent hydrator, the query-unit aggregate
    and filter lowering, and the coordination proof payload -- then
    json_extract-ed them back out one field at a time. The run and
    context-snapshot relations carried a `payload_json` column that no reader
    ever touched at all.

    Anti-vacuity: restoring `json_object(...) AS payload_json` to any of the
    three relations makes the column assertions fail, and reverting
    `observed_event_from_row` to `json.loads(row["payload_json"])` makes the
    hydration assertions fail with a missing-column error rather than a wrong
    value.
    """
    import sqlite3 as _sqlite3

    from polylogue.storage.sqlite.run_projection_relations import (
        context_snapshot_relation_sql,
        observed_event_relation_sql,
        run_relation_sql,
    )
    from tests.infra.storage_records import SessionBuilder

    db_path = tmp_path / "index.db"
    (
        SessionBuilder(db_path, "typed-columns")
        .provider("claude-code")
        .title("typed run projection columns")
        .add_message(
            "m-tool",
            role="assistant",
            text="Ran a command.",
            blocks=[
                {"type": "tool_use", "id": "tool-1", "name": "Bash", "tool_input": {"command": "pytest -k runs"}},
                {"type": "tool_result", "tool_id": "tool-1", "text": "passed", "tool_result_exit_code": 0},
            ],
        )
        .save()
    )
    session_id = "claude-code-session:ext-typed-columns"

    conn = _sqlite3.connect(db_path)
    conn.row_factory = _sqlite3.Row
    try:
        for relation_sql, name in (
            (run_relation_sql(), "runs"),
            (observed_event_relation_sql(source_where="1"), "observed_events"),
            (context_snapshot_relation_sql(), "context_snapshots"),
        ):
            cursor = conn.execute(f"{relation_sql} SELECT * FROM {name} LIMIT 0")
            columns = [description[0] for description in cursor.description]
            assert "payload_json" not in columns, f"{name} still carries a payload_json bundle: {columns}"

        cursor = conn.execute(
            f"{observed_event_relation_sql(source_where='1')}"
            " SELECT tool_name, tool_id, command, handler_kind, status"
            " FROM observed_events WHERE kind = 'tool_finished'"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0]["tool_name"] == "Bash"
    assert rows[0]["status"] == "ok"

    with ArchiveStore(db_path.parent, read_only=True) as archive:
        events = archive.query_observed_events(
            _predicate(f"observed-events where session.id:{session_id} AND kind:tool_finished")
        )

    assert len(events) == 1
    event = events[0].event
    assert event.tool_name == "Bash"
    assert event.tool_id == "tool-1"
    assert event.status == "ok"
    assert event.handler_kind is not None


@pytest.mark.parametrize(
    ("status", "parent_prefix_is_evidence"),
    [(None, True), ("repaired", True), ("quarantined", False), ("authority-contradicted", False)],
)
def test_compaction_snapshot_cites_only_a_composable_parent_prefix(
    tmp_path: Path, status: str | None, parent_prefix_is_evidence: bool
) -> None:
    """A compaction snapshot's evidence matches what the child's transcript composes.

    Anti-vacuity: drop the ``topology_status_composes_sql`` predicate and a
    quarantined or authority-contradicted link with a populated branch point
    still contributes the parent's prefix messages as evidence.
    """
    from polylogue.storage.sqlite.run_projection_relations import context_snapshot_relation_sql
    from tests.infra.storage_records import SessionBuilder

    db_path = tmp_path / "index.db"
    parent = SessionBuilder(db_path, "compaction-parent").provider("claude-code")
    for index in range(3):
        parent = parent.add_message(f"p{index}", role="user", text=f"parent message {index}")
    parent.save()
    child = (
        SessionBuilder(db_path, "compaction-child")
        .provider("claude-code")
        .add_message("c0", role="user", text="child tail")
    )
    child.save()
    parent_id, child_id = parent.native_session_id(), child.native_session_id()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        parent_messages = [
            row[0]
            for row in conn.execute(
                "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position", (parent_id,)
            )
        ]
        assert len(parent_messages) == 3
        conn.execute(
            "INSERT INTO session_links (src_session_id, dst_origin, dst_native_id, link_type, "
            "resolved_dst_session_id, branch_point_message_id, inheritance, status, observed_at_ms) "
            "VALUES (?, 'claude-code-session', ?, 'fork', ?, ?, 'prefix-sharing', ?, 0)",
            (child_id, parent_id.split(":", 1)[1], parent_id, parent_messages[1], status),
        )
        conn.execute(
            "INSERT INTO session_events (session_id, position, event_type, boundary_start_position, "
            "boundary_end_position) VALUES (?, 0, 'compaction', 0, 5)",
            (child_id,),
        )
        conn.commit()
        (evidence_json,) = conn.execute(
            f"{context_snapshot_relation_sql()} SELECT evidence_refs_json FROM context_snapshots "
            "WHERE boundary = 'compaction' AND session_id = ?",
            (child_id,),
        ).fetchone()
    finally:
        conn.close()

    evidence = set(json.loads(evidence_json))
    inherited = {f"{parent_id}::{message_id}" for message_id in parent_messages[:2]}
    assert (evidence & inherited) == (inherited if parent_prefix_is_evidence else set())
    assert f"{parent_id}::{parent_messages[2]}" not in evidence


def _observed_event_read_steps(db_path: Path, session_id: str) -> tuple[int, list[str]]:
    """SQLite VM steps (in units of 100) spent listing one session's events."""
    steps = 0

    def count() -> int:
        nonlocal steps
        steps += 1
        return 0

    with ArchiveStore(db_path.parent, read_only=True) as archive:
        archive._conn.set_progress_handler(count, 100)
        try:
            events = archive.query_observed_events(
                _predicate(f"observed-events where session.id:{session_id} AND kind:tool_finished")
            )
        finally:
            archive._conn.set_progress_handler(None, 0)
    return steps, [str(record.event.tool_id) for record in events]


def test_single_session_observed_events_rank_only_that_session(tmp_path: Path) -> None:
    """A one-session read must not rank every archived tool block.

    ``ranked_tool_uses``/``ranked_tool_results`` project only ``block_id``,
    so the outer ``session_id`` filter cannot be pushed into their windows.
    The cost of reading one small session must therefore stay flat when an
    unrelated session with many tool calls is added.

    Anti-vacuity: drop ``session_scoped`` from ``ArchiveStore.query_observed_events`` (or
    the per-window ``session_id = ?`` predicates) and the second read ranks
    the bulk session's 400 pairs too, multiplying its VM step count.
    """
    from tests.infra.storage_records import SessionBuilder

    db_path = tmp_path / "index.db"
    (
        SessionBuilder(db_path, "small")
        .provider("codex")
        .add_message(
            "m-tool",
            role="assistant",
            text="One tool call.",
            blocks=[
                {"type": "tool_use", "id": "tool-small", "name": "Bash", "tool_input": {"command": "true"}},
                {"type": "tool_result", "tool_id": "tool-small", "text": "ok", "tool_result_exit_code": 0},
            ],
        )
        .save()
    )
    session_id = "codex-session:ext-small"
    baseline_steps, baseline_events = _observed_event_read_steps(db_path, session_id)

    bulk = SessionBuilder(db_path, "bulk").provider("codex")
    for index in range(400):
        bulk = bulk.add_message(
            f"m-bulk-{index}",
            role="assistant",
            text="Bulk tool call.",
            blocks=[
                {"type": "tool_use", "id": f"tool-{index}", "name": "Bash", "tool_input": {"command": "true"}},
                {"type": "tool_result", "tool_id": f"tool-{index}", "text": "ok", "tool_result_exit_code": 0},
            ],
        )
    bulk.save()
    scoped_steps, scoped_events = _observed_event_read_steps(db_path, session_id)

    assert baseline_events == scoped_events == ["tool-small"]
    assert scoped_steps <= baseline_steps * 2 + 10, (baseline_steps, scoped_steps)


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("session.id:codex-session:ext-bound-left", ["left"]),
        ("(session.id:codex-session:ext-bound-left OR session.id:codex-session:ext-bound-right)", ["left", "right"]),
        ("(session.id:codex-session:ext-bound-left OR tool:Bash)", ["left", "right"]),
        ("session.id:codex-session:ext-bound-left AND session.id:codex-session:ext-bound-right", []),
    ],
)
def test_observed_event_session_bound_preserves_boolean_scope(tmp_path: Path, scope: str, expected: list[str]) -> None:
    """A safe physical bound preserves OR branches and contradictory AND scopes."""
    from tests.infra.storage_records import SessionBuilder

    for name in ("left", "right"):
        (
            SessionBuilder(tmp_path / "index.db", f"bound-{name}")
            .provider("codex")
            .add_message(
                "tool",
                role="assistant",
                text="Attempt.",
                blocks=[
                    {"type": "tool_use", "id": "reused", "name": "Bash", "tool_input": {"command": name}},
                    {"type": "tool_result", "tool_id": "reused", "text": "ok", "tool_result_exit_code": 0},
                ],
            )
            .save()
        )
    with ArchiveStore(tmp_path, read_only=True) as archive:
        predicate = _predicate(f"observed-events where ({scope}) AND kind:tool_finished")
        events = archive.query_observed_events(predicate)
        counts = archive.query_unit_counts("observed-event", predicate)
    assert [row.event.command for row in events] == expected
    assert sum(row.count for row in counts) == len(expected)


@pytest.mark.parametrize(
    ("suffix", "expected_kinds"),
    [
        ("", ["session_started", "tool_finished"]),
        (" AND kind:session_started", ["session_started"]),
        (" AND (kind:session_started OR kind:tool_finished)", ["session_started", "tool_finished"]),
        (" AND kind:session_started AND kind:tool_finished", []),
    ],
)
def test_observed_event_source_pushdown_preserves_unrestricted_and_impossible_scopes(
    tmp_path: Path, suffix: str, expected_kinds: list[str]
) -> None:
    from tests.infra.storage_records import SessionBuilder

    (
        SessionBuilder(tmp_path / "index.db", "unrestricted-events")
        .provider("codex")
        .add_message(
            "tool",
            role="assistant",
            text="Attempt.",
            blocks=[
                {"type": "tool_use", "id": "tool", "name": "Bash", "tool_input": {"command": "true"}},
                {"type": "tool_result", "tool_id": "tool", "text": "ok", "tool_result_exit_code": 0},
            ],
        )
        .save()
    )
    with ArchiveStore(tmp_path, read_only=True) as archive:
        predicate = _predicate(f"observed-events where session.id:codex-session:ext-unrestricted-events{suffix}")
        events = archive.query_observed_events(predicate)
        counts = archive.query_unit_counts("observed-event", predicate)
    assert [row.event.kind for row in events] == expected_kinds
    assert sum(row.count for row in counts) == len(expected_kinds)
