"""Parity test for the materialized run-projection read-models.

Rows written by the run-projection materializer must hydrate back to exactly the
output of ``compile_recovery_digest(session, session_links=()).run_projection`` —
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
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin
from polylogue.insights.transforms import compile_recovery_digest
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
    projection = compile_recovery_digest(session, session_links=()).run_projection

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

        # Lossless hydration: stored rows reproduce the exact projection models.
        assert [record.run for record in got_runs] == list(projection.runs)
        assert [record.event for record in got_events] == list(projection.events)
        assert [record.snapshot for record in got_snapshots] == list(projection.context_snapshots)

        # Typed columns back SQL filters.
        main_runs = await list_runs(conn, RunProjectionListQuery(session_id=session_id, role="main", limit=None))
        assert [record.run for record in main_runs] == [run for run in projection.runs if run.role == "main"]

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
        assert [record.run for record in rerun] == list(projection.runs)
