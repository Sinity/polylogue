"""Lineage normalization (#2467): a prefix-sharing child (fork / resume /
spawned subagent / auto-compaction copy) copies the parent's leading context.
The archive must store only the child's divergent tail plus a lineage edge with
a branch point, and reads must compose the parent prefix back in.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.sources.parsers.hermes_state import parse_state_db
from polylogue.storage.runtime import LineageCompleteness
from polylogue.storage.sqlite.archive_tiers import write as _write_module
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    _MAX_LINEAGE_DEPTH,
    _provider_usage_cumulative_baseline,
    read_archive_session_envelope,
    repair_stale_prefix_branch_points,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.queries import message_query_reads as _message_query_reads_module
from polylogue.storage.sqlite.queries.message_query_reads import (
    get_messages,
    get_messages_batch,
    get_messages_paginated,
    get_messages_with_lineage_completeness,
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

    # The sync envelope read (MCP get_session_summary / CLI read) also composes.
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


def test_stale_immediate_parent_branch_point_repairs_to_composed_ancestor(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    conn = _connect(db)

    ancestor = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="ancestor",
        title="ancestor",
        messages=[
            _msg("a0", Role.USER, "hello", 0),
            _msg("a1", Role.ASSISTANT, "hi there", 1),
        ],
    )
    ancestor_id = write_parsed_session_to_archive(conn, ancestor)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        parent_session_provider_id="ancestor",
        branch_type=BranchType.FORK,
        messages=[
            _msg("p0", Role.USER, "hello", 0),
            _msg("p1", Role.ASSISTANT, "hi there", 1),
        ],
    )
    parent_id = write_parsed_session_to_archive(conn, parent)
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (parent_id,)).fetchone()[0] == 0

    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        title="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[
            _msg("c0", Role.USER, "hello", 0),
            _msg("c1", Role.ASSISTANT, "hi there", 1),
            _msg("c2", Role.USER, "child tail", 2),
        ],
    )
    child_id = write_parsed_session_to_archive(conn, child)
    stale_branch_point = f"{parent_id}:a1"
    conn.execute(
        """
        UPDATE session_links
        SET branch_point_message_id = ?
        WHERE src_session_id = ?
        """,
        (stale_branch_point, child_id),
    )
    conn.commit()
    assert [message.blocks[0].text for message in read_archive_session_envelope(conn, child_id).messages] == [
        "child tail"
    ]

    repaired = repair_stale_prefix_branch_points(conn)
    conn.commit()

    assert repaired == 1
    branch_point = conn.execute(
        "SELECT branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()[0]
    assert branch_point == f"{ancestor_id}:a1"
    assert [message.blocks[0].text for message in read_archive_session_envelope(conn, child_id).messages] == [
        "hello",
        "hi there",
        "child tail",
    ]
    conn.close()


def test_stale_non_materialized_msg_branch_point_repairs_to_predecessor(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    conn = _connect(db)

    ancestor = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="ancestor",
        title="ancestor",
        messages=[
            _msg("msg-10", Role.USER, "inherited prompt", 0),
        ],
    )
    ancestor_id = write_parsed_session_to_archive(conn, ancestor)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        parent_session_provider_id="ancestor",
        branch_type=BranchType.FORK,
        messages=[
            _msg("msg-10", Role.USER, "inherited prompt", 0),
            _msg("msg-20", Role.USER, "parent tail", 1),
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
            _msg("msg-10", Role.USER, "inherited prompt", 0),
            _msg("msg-21", Role.USER, "child tail", 1),
        ],
    )
    child_id = write_parsed_session_to_archive(conn, child)

    conn.execute(
        """
        UPDATE session_links
        SET branch_point_message_id = ?
        WHERE src_session_id = ?
        """,
        (f"{parent_id}:msg-12", child_id),
    )
    conn.commit()
    assert [message.blocks[0].text for message in read_archive_session_envelope(conn, child_id).messages] == [
        "child tail"
    ]

    repaired = repair_stale_prefix_branch_points(conn)
    conn.commit()

    assert repaired == 1
    branch_point = conn.execute(
        "SELECT branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()[0]
    assert branch_point == f"{ancestor_id}:msg-10"
    assert [message.blocks[0].text for message in read_archive_session_envelope(conn, child_id).messages] == [
        "inherited prompt",
        "child tail",
    ]
    conn.close()


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
    parent_id = write_parsed_session_to_archive(conn, parent, manage_transaction=False)

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
    event_ref = conn.execute(
        """
        SELECT source_message_id, source_message_provider_id
        FROM session_events WHERE session_id = ?
        """,
        (child_id,),
    ).fetchone()
    assert dict(event_ref) == {
        "source_message_id": f"{parent_id}:p1",
        "source_message_provider_id": "c1",
    }

    # A source-tier rebuild reparses the child while the parent already exists.
    # The provider reference and canonical parent resolution must be identical.
    write_parsed_session_to_archive(conn, child, force_replace=True, manage_transaction=False)
    rebuilt_event_ref = conn.execute(
        """
        SELECT source_message_id, source_message_provider_id
        FROM session_events WHERE session_id = ?
        """,
        (child_id,),
    ).fetchone()
    assert dict(rebuilt_event_ref) == dict(event_ref)
    conn.rollback()
    conn.close()


def test_child_before_parent_reextracts_empty_tail_by_session(tmp_path: Path) -> None:
    """A child that is entirely inherited should remove dependents by session.

    This covers the rebuild hot path where a large child replay is later found to
    have no divergent tail. The cleanup must remove message-owned projections
    even while foreign keys are suspended.
    """
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
        ],
    )
    child_id = write_parsed_session_to_archive(conn, child)
    row = conn.execute(
        """
        SELECT m.message_id, b.block_id
        FROM messages m
        JOIN blocks b ON b.message_id = m.message_id
        WHERE m.session_id = ?
        ORDER BY m.position, b.position
        LIMIT 1
        """,
        (child_id,),
    ).fetchone()
    conn.execute(
        """
        INSERT INTO web_content_constructs (
            session_id, message_id, block_id, position, provider, construct_type, provider_key
        ) VALUES (?, ?, ?, 0, 'codex', 'content_reference', 'test')
        """,
        (child_id, row["message_id"], row["block_id"]),
    )
    conn.commit()

    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("BEGIN IMMEDIATE")
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0),
            _msg("p1", Role.ASSISTANT, "hi there", 1),
        ],
    )
    write_parsed_session_to_archive(conn, parent, manage_transaction=False)

    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (child_id,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM blocks WHERE session_id = ?", (child_id,)).fetchone()[0] == 0
    assert (
        conn.execute("SELECT COUNT(*) FROM web_content_constructs WHERE session_id = ?", (child_id,)).fetchone()[0] == 0
    )
    link = conn.execute(
        "SELECT inheritance, branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    assert link["inheritance"] == "prefix-sharing"
    assert link["branch_point_message_id"] is not None
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
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="cy",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "actual_cost_usd": 0.125,
                    "cost_status": "actual",
                    "cost_source": "hermes_state_db",
                    "billing_provider": "openrouter",
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
    provenance_rows = conn.execute(
        """
        SELECT payload_json
        FROM session_provider_usage_events
        WHERE session_id = ?
          AND json_extract(payload_json, '$.actual_cost_usd') IS NOT NULL
        """,
        (child_id,),
    ).fetchall()
    assert len(provenance_rows) == 1
    assert json.loads(provenance_rows[0]["payload_json"])["actual_cost_usd"] == 0.125


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


@pytest.mark.parametrize("end_reason", ["compression", "compaction"])
def test_hermes_compression_tail_composes_and_delegate_stays_fresh(
    tmp_path: Path,
    end_reason: str,
) -> None:
    state_db = tmp_path / "state.db"
    with sqlite3.connect(state_db) as source:
        source.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version VALUES (16);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                model_config TEXT,
                parent_session_id TEXT,
                started_at REAL,
                ended_at REAL,
                end_reason TEXT,
                title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL,
                tool_calls TEXT,
                observed INTEGER,
                active INTEGER,
                compacted INTEGER
            );
            INSERT INTO sessions VALUES
                ('parent', 'cli', '{}', NULL, 1.0, 3.0, 'compression', 'Parent'),
                ('continuation', 'cli', '{}', 'parent', 4.0, NULL, NULL, 'Continuation'),
                ('delegate', 'tool', '{}', 'parent', 5.0, NULL, NULL, 'Delegate');
            INSERT INTO messages VALUES
                (1, 'parent', 'user', 'before', 1.0, NULL, 0, 1, 0),
                (2, 'parent', 'assistant', 'summary', 2.0, NULL, 0, 1, 0),
                (3, 'continuation', 'user', 'after', 4.0, NULL, 0, 1, 0),
                (4, 'continuation', 'assistant', 'continued', 4.5, NULL, 0, 1, 0),
                (5, 'delegate', 'assistant', 'fresh work', 5.0, NULL, 0, 1, 0);
            """
        )
        source.execute(
            "UPDATE sessions SET model_config = ? WHERE id = 'delegate'",
            (json.dumps({"_delegate_from": "parent"}),),
        )
        source.execute(
            "UPDATE sessions SET end_reason = ? WHERE id = 'parent'",
            (end_reason,),
        )

    parsed = parse_state_db(state_db)
    by_raw_id = {session.provider_session_id.split("@", 1)[0]: session for session in parsed}
    parent = by_raw_id["parent"]
    continuation = by_raw_id["continuation"]
    delegate = by_raw_id["delegate"]
    assert continuation.branch_type is BranchType.CONTINUATION
    assert any(
        event.event_type == "compaction" and event.payload.get("end_reason") == end_reason
        for event in parent.session_events
    )
    assert [message.text for message in continuation.messages] == ["before", "summary", "after", "continued"]
    assert delegate.branch_type is BranchType.SUBAGENT
    assert [message.text for message in delegate.messages] == ["fresh work"]

    db = tmp_path / "index.db"
    conn = _connect(db)
    parent_id = write_parsed_session_to_archive(conn, parent)
    continuation_id = write_parsed_session_to_archive(conn, continuation)
    delegate_id = write_parsed_session_to_archive(conn, delegate)

    physical = conn.execute(
        """
        SELECT b.text
        FROM messages AS m
        JOIN blocks AS b ON b.message_id = m.message_id
        WHERE m.session_id = ? AND b.block_type = 'text'
        ORDER BY m.position, b.position
        """,
        (continuation_id,),
    ).fetchall()
    assert [row[0] for row in physical] == ["after", "continued"]
    parent_tip = conn.execute(
        "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position DESC LIMIT 1",
        (parent_id,),
    ).fetchone()[0]
    continuation_link = conn.execute(
        """
        SELECT link_type, inheritance, branch_point_message_id
        FROM session_links WHERE src_session_id = ?
        """,
        (continuation_id,),
    ).fetchone()
    assert tuple(continuation_link) == ("continuation", "prefix-sharing", parent_tip)
    delegate_link = conn.execute(
        "SELECT link_type, inheritance, branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (delegate_id,),
    ).fetchone()
    assert tuple(delegate_link) == ("subagent", "spawned-fresh", None)
    assert [
        "".join(block.text or "" for block in message.blocks)
        for message in read_archive_session_envelope(conn, continuation_id).messages
    ] == ["before", "summary", "after", "continued"]

    write_parsed_session_to_archive(conn, parent)
    write_parsed_session_to_archive(conn, continuation)
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (continuation_id,),
        ).fetchone()[0]
        == 2
    )
    assert (
        conn.execute(
            "SELECT branch_point_message_id FROM session_links WHERE src_session_id = ?",
            (continuation_id,),
        ).fetchone()[0]
        == parent_tip
    )
    conn.close()
    assert asyncio.run(_read_texts(db, continuation_id)) == ["before", "summary", "after", "continued"]

    late_db = tmp_path / "late-index.db"
    late_conn = _connect(late_db)
    late_continuation_id = write_parsed_session_to_archive(late_conn, continuation)
    late_parent_id = write_parsed_session_to_archive(late_conn, parent)
    late_link = late_conn.execute(
        "SELECT inheritance, branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (late_continuation_id,),
    ).fetchone()
    late_parent_tip = late_conn.execute(
        "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position DESC LIMIT 1",
        (late_parent_id,),
    ).fetchone()[0]
    assert tuple(late_link) == ("prefix-sharing", late_parent_tip)
    assert (
        late_conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (late_continuation_id,),
        ).fetchone()[0]
        == 2
    )
    late_conn.close()
    assert asyncio.run(_read_texts(late_db, late_continuation_id)) == [
        "before",
        "summary",
        "after",
        "continued",
    ]


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


def _setup_interleaving_fixture(db: Path) -> tuple[sqlite3.Connection, str, str]:
    """A parent+prefix-sharing-child pair on a WAL-mode file DB, so a second
    connection can commit a concurrent write without blocking the reader
    (4ts.4 regression harness)."""
    conn = _connect(db)
    conn.execute("PRAGMA journal_mode=WAL")

    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0),
            _msg("p1", Role.ASSISTANT, "hi there", 1),
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
            _msg("cx", Role.USER, "child diverges", 2),
        ],
    )
    child_id = write_parsed_session_to_archive(conn, child)
    return conn, parent_id, child_id


def _concurrently_mutate_parent_block_text(db: Path, parent_id: str) -> None:
    """Simulate a concurrent writer editing the parent's shared-prefix content
    mid-composition, via a second connection to the same WAL-mode file."""
    writer = sqlite3.connect(db)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute(
        """
        UPDATE blocks SET text = 'hi there (concurrently edited)'
        WHERE message_id = (
            SELECT message_id FROM messages
            WHERE session_id = ? AND position = 1
        )
        """,
        (parent_id,),
    )
    writer.commit()
    writer.close()


def test_sync_composition_holds_one_snapshot_across_concurrent_parent_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4ts.4: read_archive_session_envelope must not tear when a concurrent
    writer edits the parent's shared prefix mid-composition. A hook fires the
    concurrent edit right when the child's own edge lookup runs (after the
    child's own read, before the recursive parent read) -- if the composition
    were not held in one transaction, the parent read below would observe the
    mid-flight edit and the composed transcript would mix stale and fresh
    parent content with the (unaffected) child's own tail."""
    db = tmp_path / "index.db"
    conn, parent_id, child_id = _setup_interleaving_fixture(db)
    conn.close()

    # Re-open on a fresh connection so the read below starts a clean snapshot,
    # matching how a live reader (CLI/MCP/API) connects independently of the
    # writer/daemon connection.
    reader = _connect(db)
    reader.execute("PRAGMA journal_mode=WAL")

    real_edge_lookup = _write_module._prefix_sharing_edge_sync
    fired = {"count": 0}

    def _hook(conn_inner: sqlite3.Connection, session_id: str) -> tuple[str, str] | None:
        if session_id == child_id and fired["count"] == 0:
            fired["count"] += 1
            assert conn_inner.in_transaction, "composition must already hold a transaction before this hook fires"
            _concurrently_mutate_parent_block_text(db, parent_id)
        return real_edge_lookup(conn_inner, session_id)

    monkeypatch.setattr(_write_module, "_prefix_sharing_edge_sync", _hook)

    envelope = read_archive_session_envelope(reader, child_id)
    texts = ["".join(block.text or "" for block in message.blocks) for message in envelope.messages]

    assert fired["count"] == 1, "the interleaving hook never fired -- test is not exercising the race"
    # Old-consistent: the reader's held snapshot predates the concurrent edit,
    # so it must see the ORIGINAL parent text, not a torn mix.
    assert texts == ["hello", "hi there", "child diverges"]

    reader.close()

    # The concurrent edit itself did land (proving it wasn't silently a no-op) --
    # a fresh read afterwards sees the new text.
    post = _connect(db)
    post_envelope = read_archive_session_envelope(post, child_id)
    post_texts = ["".join(block.text or "" for block in message.blocks) for message in post_envelope.messages]
    assert post_texts == ["hello", "hi there (concurrently edited)", "child diverges"]
    post.close()


def test_async_composition_holds_one_snapshot_across_concurrent_parent_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async twin of the sync 4ts.4 regression above: get_messages must not
    tear when a concurrent writer edits the parent's shared prefix mid-walk."""
    db = tmp_path / "index.db"
    conn, parent_id, child_id = _setup_interleaving_fixture(db)
    conn.close()

    real_edge_lookup = _message_query_reads_module._prefix_sharing_edge
    fired = {"count": 0}

    async def _hook(conn_inner: aiosqlite.Connection, session_id: str) -> tuple[str, str] | None:
        if session_id == child_id and fired["count"] == 0:
            fired["count"] += 1
            assert conn_inner.in_transaction, "composition must already hold a transaction before this hook fires"
            _concurrently_mutate_parent_block_text(db, parent_id)
        return await real_edge_lookup(conn_inner, session_id)

    monkeypatch.setattr(_message_query_reads_module, "_prefix_sharing_edge", _hook)

    async def _run() -> list[str | None]:
        reader = await aiosqlite.connect(db)
        try:
            reader.row_factory = aiosqlite.Row
            await reader.execute("PRAGMA journal_mode=WAL")
            records = await get_messages(reader, child_id)
            return [r.text for r in records]
        finally:
            await reader.close()

    texts = asyncio.run(_run())

    assert fired["count"] == 1, "the interleaving hook never fired -- test is not exercising the race"
    assert texts == ["hello", "hi there", "child diverges"]


def test_sync_and_async_report_incomplete_on_dangling_branch_point(tmp_path: Path) -> None:
    """4ts.6: a dangling branch point (parent message hard-deleted) must
    report lineage_complete=False, not silently serve the child's own tail
    as if it were the whole transcript."""
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

    # Hard-delete the parent's messages, leaving a dangling branch point --
    # session_links.branch_point_message_id is deliberately not a FK (see
    # module docstring), so this doesn't cascade-null the link.
    conn.execute("DELETE FROM messages WHERE session_id = (SELECT session_id FROM sessions WHERE native_id = 'parent')")
    conn.commit()

    envelope = read_archive_session_envelope(conn, child_id)
    assert envelope.lineage_complete is False
    assert envelope.lineage_truncation_reason == "dangling_branch_point"
    # The child's own tail is still returned, just flagged incomplete.
    assert ["".join(b.text or "" for b in m.blocks) for m in envelope.messages] == ["child diverges"]

    conn.close()

    async def _run() -> tuple[list[str | None], LineageCompleteness]:
        reader = await aiosqlite.connect(db)
        try:
            reader.row_factory = aiosqlite.Row
            records, completeness = await get_messages_with_lineage_completeness(reader, child_id)
            return [r.text for r in records], completeness
        finally:
            await reader.close()

    texts, completeness = asyncio.run(_run())
    assert texts == ["child diverges"]
    assert completeness.complete is False
    assert completeness.truncation_reason == "dangling_branch_point"


def test_sync_report_incomplete_at_depth_limit(tmp_path: Path) -> None:
    """4ts.6: a lineage chain deeper than _MAX_LINEAGE_DEPTH must report
    lineage_complete=False with reason depth_limit -- ancestors beyond the
    cutoff are silently dropped otherwise."""
    db = tmp_path / "index.db"
    conn = _connect(db)

    # Build a chain of _MAX_LINEAGE_DEPTH + 1 sessions, each forking from the
    # previous with a one-message divergent tail. The leaf is beyond the cutoff.
    provider_session_id = "root"
    write_parsed_session_to_archive(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=provider_session_id,
            title="root",
            messages=[_msg("root-0", Role.USER, "root message", 0)],
        ),
    )
    leaf_id = None
    for level in range(_MAX_LINEAGE_DEPTH + 1):
        child_provider_id = f"level-{level}"
        leaf_id = write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=child_provider_id,
                title=child_provider_id,
                parent_session_provider_id=provider_session_id,
                branch_type=BranchType.FORK,
                messages=[
                    _msg("root-0", Role.USER, "root message", 0),
                    _msg(f"tail-{level}", Role.ASSISTANT, f"level {level} tail", 1),
                ],
            ),
        )
        provider_session_id = child_provider_id
    assert leaf_id is not None

    envelope = read_archive_session_envelope(conn, leaf_id)
    assert envelope.lineage_complete is False
    assert envelope.lineage_truncation_reason == "depth_limit"

    conn.close()


def test_async_reports_incomplete_at_its_own_depth_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """4ts.6, async twin: get_messages_with_lineage_completeness is ITERATIVE
    (not recursive), so it has its own, much larger _MAX_LINEAGE_DEPTH (1024)
    than the sync recursive path's 64 -- a chain that trips the sync limit
    does NOT trip the async one. Patch the async limit down so a small,
    fast chain exercises its own depth-limit detection directly."""
    monkeypatch.setattr(_message_query_reads_module, "_MAX_LINEAGE_DEPTH", 3)

    db = tmp_path / "index.db"
    conn = _connect(db)
    provider_session_id = "root"
    write_parsed_session_to_archive(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=provider_session_id,
            title="root",
            messages=[_msg("root-0", Role.USER, "root message", 0)],
        ),
    )
    leaf_id = None
    for level in range(4):  # one more hop than the patched limit of 3
        child_provider_id = f"level-{level}"
        leaf_id = write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=child_provider_id,
                title=child_provider_id,
                parent_session_provider_id=provider_session_id,
                branch_type=BranchType.FORK,
                messages=[
                    _msg("root-0", Role.USER, "root message", 0),
                    _msg(f"tail-{level}", Role.ASSISTANT, f"level {level} tail", 1),
                ],
            ),
        )
        provider_session_id = child_provider_id
    assert leaf_id is not None
    conn.close()

    async def _run() -> LineageCompleteness:
        reader = await aiosqlite.connect(db)
        try:
            reader.row_factory = aiosqlite.Row
            _records, completeness = await get_messages_with_lineage_completeness(reader, leaf_id)
            return completeness
        finally:
            await reader.close()

    completeness = asyncio.run(_run())
    assert completeness.complete is False
    assert completeness.truncation_reason == "depth_limit"


def test_shallow_chain_reports_complete(tmp_path: Path) -> None:
    """Sanity check: a normal, shallow fork reports lineage_complete=True --
    the completeness signal must not be trivially always-false."""
    db = tmp_path / "index.db"
    conn = _connect(db)

    write_parsed_session_to_archive(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="parent",
            title="parent",
            messages=[_msg("p0", Role.USER, "hello", 0)],
        ),
    )
    child_id = write_parsed_session_to_archive(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=[
                _msg("p0", Role.USER, "hello", 0),
                _msg("cx", Role.USER, "child diverges", 1),
            ],
        ),
    )

    envelope = read_archive_session_envelope(conn, child_id)
    assert envelope.lineage_complete is True
    assert envelope.lineage_truncation_reason is None
    conn.close()
