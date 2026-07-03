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
    ParsedSessionEvent,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    _provider_usage_cumulative_baseline,
    read_archive_session_envelope,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.queries.message_query_reads import (
    get_messages,
    get_messages_batch,
    get_messages_paginated,
    iter_messages,
)


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


async def _read_texts(path: Path, session_id: str) -> list[str | None]:
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


def test_prefix_sharing_child_provider_usage_rollup_counts_only_tail(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    conn = _connect(db)

    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0),
            _msg("p1", Role.ASSISTANT, "hi there", 1),
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="p1",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 20,
                        "output_tokens": 10,
                        "total_tokens": 110,
                    },
                },
            )
        ],
    )
    write_parsed_session_to_archive(conn, parent)

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
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="c1",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 20,
                        "output_tokens": 10,
                        "total_tokens": 110,
                    },
                },
            ),
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="cy",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "total_token_usage": {
                        "input_tokens": 160,
                        "cached_input_tokens": 30,
                        "output_tokens": 25,
                        "total_tokens": 185,
                    },
                },
            ),
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="cy",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "total_token_usage": {
                        "total_tokens": 272_000,
                    },
                },
            ),
        ],
    )
    child_id = write_parsed_session_to_archive(conn, child)

    usage = conn.execute(
        """
        SELECT input_tokens, output_tokens, cache_read_tokens, cost_provenance
        FROM session_model_usage
        WHERE session_id = ? AND model_name = 'gpt-5-codex'
        """,
        (child_id,),
    ).fetchone()
    # Child total after subtracting the parent branch-point baseline:
    # input 60, cached 10, output 15. Disjoint billing lanes therefore store
    # fresh input 50, cache read 10, output 15.
    assert dict(usage) == {
        "input_tokens": 50,
        "output_tokens": 15,
        "cache_read_tokens": 10,
        "cost_provenance": "origin_reported",
    }
    events = conn.execute(
        """
        SELECT source_message_id, total_input_tokens, total_cached_input_tokens, total_output_tokens, total_tokens
        FROM session_provider_usage_events
        WHERE session_id = ?
        ORDER BY position
        """,
        (child_id,),
    ).fetchall()
    assert [dict(row) for row in events] == [
        {
            "source_message_id": f"{child_id}:cy",
            "total_input_tokens": 60,
            "total_cached_input_tokens": 10,
            "total_output_tokens": 15,
            "total_tokens": 75,
        },
        {
            "source_message_id": f"{child_id}:cy",
            "total_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 271_890,
        },
    ]


def test_provider_usage_baseline_follows_ancestor_branch_point(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    conn = _connect(db)

    ancestor = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="ancestor",
        title="ancestor",
        messages=[
            _msg("a0", Role.USER, "root prompt", 0),
            _msg("a1", Role.ASSISTANT, "root answer", 1),
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="a1",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "total_token_usage": {
                        "input_tokens": 200,
                        "cached_input_tokens": 40,
                        "output_tokens": 20,
                        "total_tokens": 220,
                    },
                },
            )
        ],
    )
    ancestor_id = write_parsed_session_to_archive(conn, ancestor)
    parent_id = "codex-session:parent"
    branch_point = f"{ancestor_id}:a1"

    baseline = _provider_usage_cumulative_baseline(conn, parent_id, branch_point)

    assert baseline == {
        "total_input_tokens": 200,
        "total_output_tokens": 20,
        "total_cached_input_tokens": 40,
        "total_cache_write_tokens": 0,
        "total_reasoning_output_tokens": 0,
        "total_tokens": 220,
    }


def test_child_before_parent_is_reextracted_on_resolution(tmp_path: Path) -> None:
    """A prefix-sharing child ingested before its parent is stored whole, then
    normalized (inherited prefix deleted) once the parent arrives."""
    db = tmp_path / "index.db"
    conn = _connect(db)

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
    # Parent absent → stored whole, edge not yet extracted.
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (child_id,)).fetchone()[0] == 4
    assert (
        conn.execute("SELECT inheritance FROM session_links WHERE src_session_id = ?", (child_id,)).fetchone()[0]
        is None
    )

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
    write_parsed_session_to_archive(conn, parent)

    # Resolution re-extracted the child: only its tail remains, edge recorded.
    stored = conn.execute(
        "SELECT position FROM messages WHERE session_id = ? ORDER BY position", (child_id,)
    ).fetchall()
    assert [row[0] for row in stored] == [2, 3]
    link = conn.execute(
        "SELECT inheritance, branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    assert link["inheritance"] == "prefix-sharing"
    assert link["branch_point_message_id"] is not None

    conn.close()
    composed = asyncio.run(_read_texts(db, child_id))
    assert composed == ["hello", "hi there", "child diverges here", "child reply"]


def test_child_before_parent_reextracts_cleanly_when_foreign_keys_suspended(tmp_path: Path) -> None:
    """Bulk ingest suspends FKs while FTS triggers are dropped; re-extract must
    still remove rows that would normally be deleted by message cascades."""
    db = tmp_path / "index.db"
    conn = _connect(db)

    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        title="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.SUBAGENT,
        messages=[
            _msg("c0", Role.USER, "hello", 0),
            _msg("c1", Role.ASSISTANT, "hi there", 1),
            _msg("cx", Role.USER, "child diverges here", 2),
            _msg("cy", Role.ASSISTANT, "child reply", 3),
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="capture_gap",
                source_message_provider_id="c1",
                payload={"summary": "prefix event"},
            ),
        ],
    )
    child_id = write_parsed_session_to_archive(conn, child)
    assert conn.execute("SELECT COUNT(*) FROM blocks WHERE session_id = ?", (child_id,)).fetchone()[0] == 4

    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("BEGIN IMMEDIATE")
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
    write_parsed_session_to_archive(conn, parent, manage_transaction=False)

    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    dangling_blocks = conn.execute(
        """
        SELECT COUNT(*)
        FROM blocks b
        WHERE b.session_id = ?
          AND NOT EXISTS (
                SELECT 1 FROM messages m WHERE m.message_id = b.message_id
          )
        """,
        (child_id,),
    ).fetchone()[0]
    assert dangling_blocks == 0
    stored_positions = conn.execute(
        "SELECT position FROM messages WHERE session_id = ? ORDER BY position",
        (child_id,),
    ).fetchall()
    assert [row[0] for row in stored_positions] == [2, 3]
    assert (
        conn.execute(
            "SELECT source_message_id FROM session_events WHERE session_id = ?",
            (child_id,),
        ).fetchone()[0]
        is None
    )
    conn.rollback()
    conn.close()


def test_child_before_parent_reextracts_provider_usage_tail(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    conn = _connect(db)

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
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="c1",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 20,
                        "output_tokens": 10,
                        "total_tokens": 110,
                    },
                },
            ),
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="cy",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "total_token_usage": {
                        "input_tokens": 160,
                        "cached_input_tokens": 30,
                        "output_tokens": 25,
                        "total_tokens": 185,
                    },
                },
            ),
        ],
    )
    child_id = write_parsed_session_to_archive(conn, child)
    assert (
        conn.execute("SELECT input_tokens FROM session_model_usage WHERE session_id = ?", (child_id,)).fetchone()[0]
        == 130
    )

    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0),
            _msg("p1", Role.ASSISTANT, "hi there", 1),
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="p1",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 20,
                        "output_tokens": 10,
                        "total_tokens": 110,
                    },
                },
            )
        ],
    )
    write_parsed_session_to_archive(conn, parent)

    usage = conn.execute(
        """
        SELECT input_tokens, output_tokens, cache_read_tokens, cost_provenance
        FROM session_model_usage
        WHERE session_id = ? AND model_name = 'gpt-5-codex'
        """,
        (child_id,),
    ).fetchone()
    assert dict(usage) == {
        "input_tokens": 50,
        "output_tokens": 15,
        "cache_read_tokens": 10,
        "cost_provenance": "origin_reported",
    }


def test_parent_reingest_keeps_child_composing(tmp_path: Path) -> None:
    """Regression for the FK-cascade bug (#2467 audit H1): re-ingesting a parent
    via full replace must NOT null the child's branch point. branch_point_message_id
    is deliberately not a FK, so the deterministic message id survives the parent's
    DELETE+re-INSERT and the child keeps composing the full transcript."""
    db = tmp_path / "index.db"
    conn = _connect(db)

    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0),
            _msg("p1", Role.ASSISTANT, "hi there", 1),
        ],
    )
    write_parsed_session_to_archive(conn, parent)

    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        title="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[
            _msg("c0", Role.USER, "hello", 0),
            _msg("c1", Role.ASSISTANT, "hi there", 1),
            _msg("cx", Role.USER, "child diverges", 2),
        ],
    )
    child_id = write_parsed_session_to_archive(conn, child)
    assert [
        "".join(b.text or "" for b in m.blocks) for m in read_archive_session_envelope(conn, child_id).messages
    ] == ["hello", "hi there", "child diverges"]

    # Parent grows and is re-ingested (full replace) — the common production case.
    parent_grown = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0),
            _msg("p1", Role.ASSISTANT, "hi there", 1),
            _msg("p2", Role.USER, "parent keeps going", 2),
        ],
    )
    write_parsed_session_to_archive(conn, parent_grown)

    # The child's branch point survived; it still composes the full transcript.
    link = conn.execute(
        "SELECT inheritance, branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    assert link["inheritance"] == "prefix-sharing"
    assert link["branch_point_message_id"] is not None
    assert [
        "".join(b.text or "" for b in m.blocks) for m in read_archive_session_envelope(conn, child_id).messages
    ] == ["hello", "hi there", "child diverges"]


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


def _build_parent_and_fork(db: Path) -> tuple[str, str]:
    """Persist a parent and a prefix-sharing fork; return (parent_id, child_id).

    The fork replays the parent's first two messages then diverges, so its full
    logical transcript is 4 messages while only 2 (its tail) are physically
    stored under the child.
    """
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
    conn.close()
    return parent_id, child_id


def test_fork_composes_on_paginated_batch_and_iter(tmp_path: Path) -> None:
    """All read surfaces compose a fork's full logical transcript, not the
    tail-only physical rows (#2470)."""
    db = tmp_path / "index.db"
    _parent_id, child_id = _build_parent_and_fork(db)
    full = ["hello", "hi there", "child diverges here", "child reply"]

    async def _exercise() -> None:
        conn = await aiosqlite.connect(db)
        try:
            conn.row_factory = aiosqlite.Row

            # Paginated: total is the composed length; pages slice the composed list.
            page1, total = await get_messages_paginated(conn, child_id, limit=2, offset=0)
            assert total == 4
            assert [r.text for r in page1] == full[:2]
            page2, total2 = await get_messages_paginated(conn, child_id, limit=2, offset=2)
            assert total2 == 4
            assert [r.text for r in page2] == full[2:]

            # Batch: the child's entry carries the composed transcript.
            result, all_messages = await get_messages_batch(conn, [child_id])
            assert [r.text for r in result[child_id]] == full
            # all_messages must include every composed record for block hydration.
            assert {r.message_id for r in result[child_id]} <= {r.message_id for r in all_messages}

            # Streaming: iter_messages yields the composed transcript in order.
            streamed = [r.text async for r in iter_messages(conn, child_id)]
            assert streamed == full

            # limit is honored over the composed stream.
            limited = [r.text async for r in iter_messages(conn, child_id, limit=3)]
            assert limited == full[:3]
        finally:
            await conn.close()

    asyncio.run(_exercise())


def test_shared_signature_cache_composes_correctly(tmp_path: Path) -> None:
    """A batch-scoped signature cache shared across writes must not corrupt
    lineage composition (#2475). Two forks of one parent and a parent re-ingest
    all share a single cache dict; every child must still compose its full
    logical transcript and the parent re-ingest invalidation must hold.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    cache: dict[str, list[tuple[str, str]]] = {}

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
    write_parsed_session_to_archive(conn, parent, signature_cache=cache)

    def _fork(name: str, tail_user: str, tail_assistant: str) -> str:
        child = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=name,
            title=name,
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=[
                _msg(f"{name}0", Role.USER, "hello", 0),
                _msg(f"{name}1", Role.ASSISTANT, "hi there", 1),
                _msg(f"{name}x", Role.USER, tail_user, 2),
                _msg(f"{name}y", Role.ASSISTANT, tail_assistant, 3),
            ],
        )
        # Two SEPARATE write calls that SHARE one signature_cache dict.
        return write_parsed_session_to_archive(conn, child, signature_cache=cache)

    fork_a_id = _fork("forka", "fork A diverges", "fork A reply")
    fork_b_id = _fork("forkb", "fork B diverges", "fork B reply")

    def _composed(session_id: str) -> list[str]:
        return [
            "".join(block.text or "" for block in message.blocks)
            for message in read_archive_session_envelope(conn, session_id).messages
        ]

    assert _composed(fork_a_id) == ["hello", "hi there", "fork A diverges", "fork A reply"]
    assert _composed(fork_b_id) == ["hello", "hi there", "fork B diverges", "fork B reply"]

    # Re-ingest the parent with a grown tail through the SAME shared cache. The
    # per-write invalidation must drop the parent's stale own-signatures so both
    # forks still compose the (unchanged) shared prefix + their own tails.
    parent_grown = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0),
            _msg("p1", Role.ASSISTANT, "hi there", 1),
            _msg("p2", Role.USER, "parent continues alone", 2),
            _msg("p3", Role.ASSISTANT, "parent grows more", 3),
        ],
    )
    write_parsed_session_to_archive(conn, parent_grown, signature_cache=cache)

    assert _composed(fork_a_id) == ["hello", "hi there", "fork A diverges", "fork A reply"]
    assert _composed(fork_b_id) == ["hello", "hi there", "fork B diverges", "fork B reply"]

    conn.close()
