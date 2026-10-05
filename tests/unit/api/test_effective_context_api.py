"""Effective-context API contract (polylogue-4ts.5).

``Polylogue.get_effective_context`` must report what the model actually saw at
a position: a compaction boundary replaces its recorded message range with the
materialized summary, so the effective context is strictly narrower than the
composed transcript the same session returns for forks.

Anti-vacuity: dropping the boundary columns from ``session_events``, or letting
the read fall back to the full transcript, makes the first assertion return all
five messages and the test red.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from polylogue.api import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.index_writer import write_fixture_index_session

if TYPE_CHECKING:
    import aiosqlite

    from polylogue.storage.runtime import MessageRecord

_SESSION_ID = "codex-session:compaction-effective-context"


def _message(native_id: str, role: Role, text: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=native_id,
        role=role,
        text=text,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _seed_on_writer(db_path: Path, *, extra_events: tuple[ParsedSessionEvent, ...] = ()) -> None:
    conn = connect_measured(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="compaction-effective-context",
        title="compaction effective context",
        messages=[
            _message("m0", Role.USER, "first ask"),
            _message("m1", Role.ASSISTANT, "first answer"),
            _message("m2", Role.SYSTEM, "compaction summary"),
            _message("m3", Role.USER, "post-compaction ask"),
            _message("m4", Role.ASSISTANT, "post-compaction answer"),
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="compaction",
                source_message_provider_id="m2",
                boundary_start_position=0,
                boundary_end_position=1,
                boundary_message_position=2,
                payload={"type": "compaction"},
            ),
            *extra_events,
        ],
    )
    write_fixture_index_session(conn, session)
    conn.commit()
    conn.close()


def _seed(db_path: Path, *, extra_events: tuple[ParsedSessionEvent, ...] = ()) -> None:
    """Run the synchronous seed off any running event loop."""
    return run_off_event_loop(lambda: _seed_on_writer(db_path, extra_events=extra_events))


@pytest.mark.asyncio
async def test_effective_context_replaces_the_boundary_range_with_its_summary(
    workspace_env: dict[str, Path],
) -> None:
    db_path = workspace_env["archive_root"] / "index.db"
    _seed(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        effective = await polylogue.get_effective_context(_SESSION_ID)
        session = await polylogue.get_session(_SESSION_ID)
    finally:
        await polylogue.close()

    assert effective is not None
    assert [message["text"] for message in effective] == [
        "compaction summary",
        "post-compaction ask",
        "post-compaction answer",
    ]
    # The full composed transcript — what a fork replays — keeps the replaced range.
    assert session is not None
    assert len(session.messages) == 5


@pytest.mark.asyncio
async def test_effective_context_before_the_boundary_is_the_plain_prefix(
    workspace_env: dict[str, Path],
) -> None:
    db_path = workspace_env["archive_root"] / "index.db"
    _seed(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        effective = await polylogue.get_effective_context(_SESSION_ID, at_position=1)
        missing = await polylogue.get_effective_context("codex-session:absent")
    finally:
        await polylogue.close()

    assert effective is not None
    assert [message["text"] for message in effective] == ["first ask", "first answer"]
    assert missing is None


@pytest.mark.asyncio
async def test_effective_context_ignores_a_partial_stored_boundary(
    workspace_env: dict[str, Path],
) -> None:
    """A half-populated range is not precise context evidence."""
    db_path = workspace_env["archive_root"] / "index.db"
    _seed(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE session_events SET boundary_start_position = NULL WHERE event_type = 'compaction'")
    conn.commit()
    conn.close()

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        effective = await polylogue.get_effective_context(_SESSION_ID, at_position=3)
    finally:
        await polylogue.close()

    assert effective is not None
    assert [message["text"] for message in effective] == [
        "first ask",
        "first answer",
        "compaction summary",
        "post-compaction ask",
    ]


def _seed_with_fork_on_writer(db_path: Path) -> None:
    """Parent carrying a compaction boundary plus a fork that replays its prefix."""
    conn = connect_measured(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="compaction-effective-context",
        title="compaction effective context",
        messages=[
            _message("m0", Role.USER, "first ask"),
            _message("m1", Role.ASSISTANT, "first answer"),
            _message("m2", Role.SYSTEM, "compaction summary"),
            _message("m3", Role.USER, "post-compaction ask"),
            _message("m4", Role.ASSISTANT, "post-compaction answer"),
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="compaction",
                source_message_provider_id="m2",
                boundary_start_position=0,
                boundary_end_position=1,
                boundary_message_position=2,
                payload={"type": "compaction"},
            )
        ],
    )
    write_fixture_index_session(conn, parent)
    fork = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="compaction-effective-context-fork",
        title="fork of the compacted session",
        parent_session_provider_id="compaction-effective-context",
        branch_type=BranchType.FORK,
        messages=[
            _message("m0", Role.USER, "first ask"),
            _message("m1", Role.ASSISTANT, "first answer"),
            _message("m2", Role.SYSTEM, "compaction summary"),
            _message("m3", Role.USER, "post-compaction ask"),
            _message("f4", Role.ASSISTANT, "fork diverges here"),
        ],
    )
    write_fixture_index_session(conn, fork)
    conn.commit()
    conn.close()


def _seed_with_fork(db_path: Path) -> None:
    """Run the synchronous seed off any running event loop."""
    return run_off_event_loop(lambda: _seed_with_fork_on_writer(db_path))


@pytest.mark.asyncio
async def test_effective_context_and_composed_fork_prefix_differ_at_the_same_position(
    workspace_env: dict[str, Path],
) -> None:
    """What the model saw is narrower than what a fork replays, at one position.

    Anti-vacuity: if ``get_effective_context`` fell back to the composed
    prefix, both sides of this comparison would carry "first ask"/"first
    answer" and the inequality assertion would be red.
    """
    db_path = workspace_env["archive_root"] / "index.db"
    _seed_with_fork(db_path)

    at_position = 3
    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        effective = await polylogue.get_effective_context(_SESSION_ID, at_position=at_position)
        fork = await polylogue.get_session("codex-session:compaction-effective-context-fork")
    finally:
        await polylogue.close()

    assert effective is not None
    assert [message["text"] for message in effective] == [
        "compaction summary",
        "post-compaction ask",
    ]

    # The fork physically replays the parent's whole prefix, replaced range and
    # all: composition is what a resumed run inherits, not what the model saw.
    # ``position`` is per-session -- the fork's stored tail restarts at 0 -- so
    # the prefix is taken in composed transcript order.
    assert fork is not None
    composed_prefix = [m.text for m in fork.messages][: at_position + 1]
    assert composed_prefix == [
        "first ask",
        "first answer",
        "compaction summary",
        "post-compaction ask",
    ]
    assert [message["text"] for message in effective] != composed_prefix


@pytest.mark.asyncio
@pytest.mark.parametrize(("start", "end"), [(-1, 1), (2, 1)])
async def test_effective_context_rejects_invalid_ranges(workspace_env: dict[str, Path], start: int, end: int) -> None:
    """Non-null but invalid bounds previously authorized dropping the prefix."""
    db_path = workspace_env["archive_root"] / "index.db"
    _seed(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE session_events SET boundary_start_position = ?, boundary_end_position = ?",
            (start, end),
        )
    poly = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        result = await poly.get_effective_context(_SESSION_ID, at_position=3)
        assert result is not None
        assert [row["text"] for row in result] == [
            "first ask",
            "first answer",
            "compaction summary",
            "post-compaction ask",
        ]
    finally:
        await poly.close()


@pytest.mark.asyncio
async def test_newest_incomplete_compaction_does_not_reuse_an_older_summary(
    workspace_env: dict[str, Path],
) -> None:
    """The complete-row SQL filter formerly picked the old three-message view."""
    db_path = workspace_env["archive_root"] / "index.db"
    _seed(
        db_path,
        extra_events=(
            ParsedSessionEvent(
                event_type="compaction",
                boundary_start_position=0,
                boundary_end_position=3,
                payload={"type": "compaction", "summary": "unavailable"},
            ),
        ),
    )
    poly = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        result = await poly.get_effective_context(_SESSION_ID)
        assert result is not None
        assert len(result) == 5
        assert result[0]["text"] == "first ask"
    finally:
        await poly.close()


@pytest.mark.asyncio
async def test_effective_context_includes_a_forks_inherited_prefix(
    workspace_env: dict[str, Path],
) -> None:
    """The old own-row read returns just the divergent tail of this real fork."""
    db_path = workspace_env["archive_root"] / "index.db"
    _seed_with_fork(db_path)
    poly = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        result = await poly.get_effective_context("codex-session:compaction-effective-context-fork", at_position=3)
        assert result is not None
        assert [row["text"] for row in result] == [
            "first ask",
            "first answer",
            "compaction summary",
            "post-compaction ask",
        ]
    finally:
        await poly.close()


@pytest.mark.asyncio
async def test_effective_context_reads_boundary_in_the_message_snapshot(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real WAL commit between message and boundary reads must not tear the view."""
    from polylogue.storage.sqlite.queries import message_query_reads

    db_path = workspace_env["archive_root"] / "index.db"
    _seed(db_path)
    with sqlite3.connect(db_path) as setup:
        setup.execute("PRAGMA journal_mode = WAL")
    read_own = message_query_reads._own_messages
    committed = False

    async def interleaved(conn: aiosqlite.Connection, session_id: str) -> list[MessageRecord]:
        nonlocal committed
        records = await read_own(conn, session_id)
        if not committed:
            committed = True
            with sqlite3.connect(db_path) as writer:
                writer.execute("UPDATE session_events SET boundary_message_id = NULL")
        return records

    monkeypatch.setattr(message_query_reads, "_own_messages", interleaved)
    poly = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        result = await poly.get_effective_context(_SESSION_ID)
        assert committed
        assert result is not None
        assert [row["text"] for row in result] == [
            "compaction summary",
            "post-compaction ask",
            "post-compaction answer",
        ]
    finally:
        await poly.close()


@pytest.mark.asyncio
async def test_effective_context_hydrates_structured_tool_blocks(workspace_env: dict[str, Path]) -> None:
    """The previous direct query return serialized an empty block list."""
    db_path = workspace_env["archive_root"] / "index.db"

    def seed() -> None:
        conn = connect_measured(db_path)
        conn.row_factory = sqlite3.Row
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        write_fixture_index_session(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="effective-tool-blocks",
                messages=[
                    ParsedMessage(
                        provider_message_id="tool-message",
                        role=Role.ASSISTANT,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Read",
                                tool_id="call-read",
                                tool_input={"file_path": "src/main.rs"},
                            )
                        ],
                    )
                ],
            ),
        )
        conn.commit()
        conn.close()

    run_off_event_loop(seed)
    poly = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        result = await poly.get_effective_context("codex-session:effective-tool-blocks")
        assert result is not None and len(result) == 1
        blocks = result[0]["blocks"]
        assert isinstance(blocks, list) and len(blocks) == 1
        assert blocks[0]["tool_name"] == "Read"
        assert blocks[0]["tool_input"] == {"file_path": "src/main.rs"}
    finally:
        await poly.close()


def test_context_snapshot_start_precedes_the_first_compaction(tmp_path: Path) -> None:
    """A position-zero event formerly sorted ahead of the synthetic start."""
    from polylogue.storage.sqlite.run_projection_relations import context_snapshot_relation_sql

    db_path = tmp_path / "index.db"
    _seed(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            context_snapshot_relation_sql()
            + " SELECT boundary, position FROM context_snapshots WHERE session_id = ? ORDER BY position, snapshot_ref",
            (_SESSION_ID,),
        ).fetchall()
    assert rows == [("session_start", 0), ("compaction", 1)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("session_id", "at_position"),
    [
        (_SESSION_ID, None),
        (_SESSION_ID, 1),
        (_SESSION_ID, 3),
        ("codex-session:compaction-effective-context-fork", None),
        ("codex-session:compaction-effective-context-fork", 3),
    ],
)
async def test_pinned_operation_and_api_share_one_effective_context_decision(
    workspace_env: dict[str, Path], session_id: str, at_position: int | None
) -> None:
    """Anti-vacuity: the pinned operation route read only a session's own rows,
    so for the fork it answered with the divergent tail alone while the API
    answered with the composed prefix."""
    from polylogue.operations.daemon_reads import execute_read_operation
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = workspace_env["archive_root"]
    db_path = root / "index.db"
    _seed_with_fork(db_path)
    poly = Polylogue(archive_root=root, db_path=db_path)
    try:
        api = await poly.get_effective_context(session_id, at_position=at_position)
    finally:
        await poly.close()
    assert api is not None
    with ArchiveStore.open_existing(root) as archive:
        result = execute_read_operation(
            "read.effective_context",
            {"session_id": session_id, "at_position": at_position},
            archive=archive,
            serving_identity="test",
        )
    payload = result["payload"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert [message["text"] for message in messages] == [message["text"] for message in api]
    if session_id.endswith("-fork"):
        assert [message["text"] for message in api][:2] == ["first ask", "first answer"]
