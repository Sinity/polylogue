"""Lineage normalization (#2467): a prefix-sharing child (fork / resume /
spawned subagent / auto-compaction copy) copies the parent's leading context.
The archive must store only the child's divergent tail plus a lineage edge with
a branch point, and reads must compose the parent prefix back in.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import aiosqlite

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    read_archive_session_envelope,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.queries.message_query_reads import get_messages


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _msg(pid: str, role: Role, text: str, position: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=pid,
        role=role,
        text=text,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


async def _read_texts(path: Path, session_id: str) -> list[str]:
    conn = await aiosqlite.connect(path)
    try:
        conn.row_factory = aiosqlite.Row
        records = await get_messages(conn, session_id)
        return [r.text for r in records]
    finally:
        await conn.close()


def test_prefix_sharing_child_stores_only_tail_and_composes(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    conn = _connect(db)

    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0),
            _msg("p1", Role.ASSISTANT, "hi there", 1),
            _msg("p2", Role.USER, "parent continues alone", 2),
        ],
    )
    parent_id = write_parsed_session_to_archive(conn, parent)

    # Child forked after parent[1]: it replays p0/p1 (identical content, fresh
    # provider ids, as a real fork does) then diverges into its own work.
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        title="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[
            _msg("c0", Role.USER, "hello", 0),
            _msg("c1", Role.ASSISTANT, "hi there", 1),
            _msg("cx", Role.USER, "child diverges here", 2),
            _msg("cy", Role.ASSISTANT, "child reply", 3),
        ],
    )
    child_id = write_parsed_session_to_archive(conn, child)

    # Only the divergent tail is physically stored under the child, at its
    # original positions (2, 3) — the inherited prefix is not duplicated.
    stored = conn.execute(
        "SELECT position FROM messages WHERE session_id = ? ORDER BY position",
        (child_id,),
    ).fetchall()
    assert [row[0] for row in stored] == [2, 3]

    # Aggregate count reflects the child's own messages only (no double count).
    message_count = conn.execute(
        "SELECT message_count FROM sessions WHERE session_id = ?",
        (child_id,),
    ).fetchone()[0]
    assert message_count == 2

    # The lineage edge records the branch point and the inheritance kind.
    link = conn.execute(
        """
        SELECT inheritance, branch_point_message_id, resolved_dst_session_id
        FROM session_links WHERE src_session_id = ?
        """,
        (child_id,),
    ).fetchone()
    assert link["inheritance"] == "prefix-sharing"
    assert link["branch_point_message_id"] is not None
    assert link["resolved_dst_session_id"] == parent_id

    # The sync envelope read (MCP get_session / CLI read) also composes.
    envelope = read_archive_session_envelope(conn, child_id)
    assert ["".join(block.text or "" for block in message.blocks) for message in envelope.messages] == [
        "hello",
        "hi there",
        "child diverges here",
        "child reply",
    ]

    conn.close()

    # Reading the child via the async query path composes the same transcript.
    composed = asyncio.run(_read_texts(db, child_id))
    assert composed == ["hello", "hi there", "child diverges here", "child reply"]


def test_spawned_fresh_child_keeps_all_messages(tmp_path: Path) -> None:
    """A child that shares no leading prefix with its parent (a fresh Task
    subagent) is stored whole; the edge is 'spawned-fresh' with no branch
    point, and reads do not prepend the parent."""
    db = tmp_path / "index.db"
    conn = _connect(db)

    parent = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="root",
        title="root",
        messages=[
            _msg("r0", Role.USER, "do the whole task", 0),
            _msg("r1", Role.ASSISTANT, "working", 1),
        ],
    )
    write_parsed_session_to_archive(conn, parent)

    child = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="root:agent-abc",
        title="subagent",
        parent_session_provider_id="root",
        branch_type=BranchType.SUBAGENT,
        messages=[
            _msg("s0", Role.USER, "fresh subagent prompt", 0),
            _msg("s1", Role.ASSISTANT, "fresh subagent answer", 1),
        ],
    )
    child_id = write_parsed_session_to_archive(conn, child)

    stored = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
        (child_id,),
    ).fetchone()[0]
    assert stored == 2

    link = conn.execute(
        "SELECT inheritance, branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    assert link["inheritance"] == "spawned-fresh"
    assert link["branch_point_message_id"] is None

    conn.close()
    composed = asyncio.run(_read_texts(db, child_id))
    assert composed == ["fresh subagent prompt", "fresh subagent answer"]
