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

import pytest

from polylogue.api import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

_SESSION_ID = "codex-session:compaction-effective-context"


def _message(native_id: str, role: Role, text: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=native_id,
        role=role,
        text=text,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _seed(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
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
            )
        ],
    )
    write_parsed_session_to_archive(conn, session)
    conn.commit()
    conn.close()


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


def _seed_with_fork(db_path: Path) -> None:
    """Parent carrying a compaction boundary plus a fork that replays its prefix."""
    conn = sqlite3.connect(db_path)
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
    write_parsed_session_to_archive(conn, parent)
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
    write_parsed_session_to_archive(conn, fork)
    conn.commit()
    conn.close()


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
