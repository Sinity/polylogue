"""Lineage normalization (#2467): a prefix-sharing child (fork / resume /
spawned subagent / auto-compaction copy) copies the parent's leading context.
The archive must store only the child's divergent tail plus a lineage edge with
a branch point, and reads must compose the parent prefix back in.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import aiosqlite
import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider, ToolOutcome
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.sources.parsers.hermes_state import parse_state_db
from polylogue.storage.derived.session.derivation import archive_session_partition_statuses
from polylogue.storage.derived.session.input_binding import session_input_bindings
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION, LineageCompleteness
from polylogue.storage.sqlite.archive_tiers import write as _write_module
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    IDENTITY_INVALIDATION_DEBT_STAGE,
    _repair_stale_prefix_branch_points_db,
    _upsert_session_link,
    count_dangling_prefix_branch_points,
    read_archive_session_envelope,
)
from polylogue.storage.sqlite.queries import message_query_reads as _message_query_reads_module
from polylogue.storage.sqlite.queries.message_query_reads import (
    get_messages,
    get_messages_batch,
    get_messages_paginated,
    get_messages_with_lineage_completeness,
    iter_messages,
)
from tests.infra.identity import archive_message_id
from tests.infra.index_writer import (
    _fixture_writer_admission,
    close_fixture_index_connection,
    fixture_index_mutation_scope,
    prepared_fixture_index_batch,
    write_fixture_index_session,
)
from tests.infra.session_profiles import write_session_profile


def _connect(path: Path) -> sqlite3.Connection:
    conn = connect_measured(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _msg(
    pid: str,
    role: Role,
    text: str,
    position: int,
    *,
    variant_index: int = 0,
    timestamp: str | None = None,
) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=pid,
        role=role,
        text=text,
        position=position,
        variant_index=variant_index,
        is_active_path=True,
        is_active_leaf=False,
        timestamp=timestamp,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def test_session_link_writer_rejects_unknown_inheritance_before_storage() -> None:
    with pytest.raises(ValueError, match="lineage inheritance"):
        _upsert_session_link(
            cast(sqlite3.Connection, None),  # Validation must happen before the connection is touched.
            src_session_id="child",
            dst_origin="codex-session",
            dst_native_id="parent",
            link_type="subagent",
            branch_point_message_id=None,
            inheritance="shared-prefix",
            status=None,
            parent_tool_use_block_id=None,
            method="parser-parent",
            confidence=1.0,
            evidence_json="{}",
            observed_at_ms=1,
        )


def _seed_fresh_session_products(conn: sqlite3.Connection, session_id: str, *, message_count: int) -> None:
    """Materialize the derived partition a converger would have written for
    ``session_id`` as it stands, stamping the value-complete input binding so
    inspection reports it current.

    Publication also consumes the session's captured profile demand (#5525);
    a session with pending demand is stale however fresh its profile row is,
    so the seed deletes the demand row as
    ``publish_prepared_session_insight_partition``
    does."""
    binding = session_input_bindings(conn, (session_id,))[session_id]
    source_updated_at, source_sort_key, origin = conn.execute(
        """
        SELECT datetime(updated_at_ms / 1000, 'unixepoch'), CAST(sort_key_ms AS REAL) / 1000.0, origin
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    write_session_profile(
        conn,
        session_id,
        materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
        materialized_at="",
        source_updated_at=source_updated_at,
        source_sort_key=source_sort_key,
        input_content_hash=binding,
        input_row_count=message_count,
        source_name=origin,
        message_count=message_count,
    )
    conn.execute(
        "INSERT INTO session_latency_profiles (session_id, materializer_version, materialized_at, source_name)"
        " VALUES (?, ?, '', '')",
        (session_id, SESSION_INSIGHT_MATERIALIZER_VERSION),
    )
    conn.execute("DELETE FROM session_profile_demand WHERE session_id = ?", (session_id,))


def _nonvalid_partitions(conn: sqlite3.Connection) -> list[str]:
    statuses = archive_session_partition_statuses(conn, materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION)
    return sorted(session_id for session_id, status in statuses.items() if status != "valid")


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
    parent_id = write_fixture_index_session(conn, parent)

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
    child_id = write_fixture_index_session(conn, child)

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
    # Production dependency: read_archive_session_envelope records exact
    # composition provenance instead of making renderers infer a prefix from
    # message position. Removing source_session_id or the bounded edge facts
    # makes these assertions fail.
    assert envelope.lineage_inheritance == "prefix-sharing"
    assert envelope.lineage_branch_point_message_id == link["branch_point_message_id"]
    assert [message.source_session_id for message in envelope.messages] == [
        parent_id,
        parent_id,
        child_id,
        child_id,
    ]

    close_fixture_index_connection(conn)

    # Reading the child via the async query path composes the same transcript.
    composed = asyncio.run(_read_texts(db, child_id))
    assert composed == ["hello", "hi there", "child diverges here", "child reply"]


def test_prefix_sharing_child_provider_usage_keeps_its_own_reported_totals(tmp_path: Path) -> None:
    """A prefix-sharing child drops usage bound to the replayed prefix (the
    parent already owns that observation) and keeps its OWN cumulative totals
    verbatim. polylogue-uoq3x: the child's counter is session-scoped, so the
    parent's branch-point cumulative is not a baseline to subtract.

    A lane the provider did not report stays NULL (#5530 distinguishes absent
    usage from explicit zero): the ``total_tokens``-only event stores NULL
    input/cached/output totals, and it must not displace the latest lane
    totals in the rollup."""
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
    write_fixture_index_session(conn, parent)

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
    child_id = write_fixture_index_session(conn, child)

    # The final event reports only an aggregate total, so its billing lanes are
    # incomplete even though the earlier exact lane counters remain available.
    usage = conn.execute(
        """
        SELECT input_tokens, output_tokens, cache_read_tokens, provider_lanes_complete,
               CASE WHEN provider_cost_usd IS NOT NULL THEN 'origin_reported'
                    WHEN catalog_cost_usd IS NOT NULL THEN 'priced' END AS cost_provenance
        FROM session_model_usage
        WHERE session_id = ? AND model_name = 'gpt-5-codex'
        """,
        (child_id,),
    ).fetchone()
    # The child's own latest cumulative is input 160, cached 30, output 25.
    # Disjoint billing lanes therefore store fresh input 130, cache read 30,
    # output 25 -- the parent's 100/20/10 is NOT subtracted.
    assert dict(usage) == {
        "input_tokens": 130,
        "output_tokens": 25,
        "cache_read_tokens": 30,
        "provider_lanes_complete": 0,
        "cost_provenance": None,
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
            "source_message_id": archive_message_id(child_id, "cy"),
            "total_input_tokens": 160,
            "total_cached_input_tokens": 30,
            "total_output_tokens": 25,
            "total_tokens": 185,
        },
        {
            "source_message_id": archive_message_id(child_id, "cy"),
            "total_input_tokens": None,
            "total_cached_input_tokens": None,
            "total_output_tokens": None,
            "total_tokens": 272_000,
        },
    ]


def test_replaying_chain_stores_each_link_reported_cumulative(tmp_path: Path) -> None:
    """polylogue-uoq3x: every link of a replaying chain stores its OWN reported
    cumulative totals.

    Anti-vacuity: the chain must be at least four links. The writer used to
    rebase each child against the PARENT'S STORED row, which for a mid-chain
    parent had itself already been rebased, so the error compounded. Link 2's
    subtraction is correct on its own (its parent is the un-rebased root); only
    link 3 onward exposes the compounding. Reinstating the subtraction turns
    the reported ``10, 13, 16, 19, 22, 25`` into the stored sawtooth
    ``10, 3, 13, 6, 16, 9`` and makes this test red.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)

    reported = (10, 13, 16, 19, 22, 25)
    turns: list[tuple[str, Role, str]] = []
    messages: list[ParsedMessage] = []
    session_ids: list[str] = []
    for link, total in enumerate(reported):
        turns += [(f"u{link}", Role.USER, f"prompt {link}"), (f"a{link}", Role.ASSISTANT, f"answer {link}")]
        messages = [_msg(pid, role, text, index) for index, (pid, role, text) in enumerate(turns)]
        session_ids.append(
            write_fixture_index_session(
                conn,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=f"s{link}",
                    title=f"s{link}",
                    parent_session_provider_id=(f"s{link - 1}" if link else None),
                    messages=messages,
                    session_events=[
                        ParsedSessionEvent(
                            event_type="token_count",
                            source_message_provider_id=f"a{link}",
                            payload={
                                "type": "token_count",
                                "model": "gpt-5-codex",
                                "total_token_usage": {
                                    "input_tokens": total,
                                    "output_tokens": total,
                                    "total_tokens": 2 * total,
                                },
                            },
                        )
                    ],
                ),
            )
        )

    # Every link past the root really is a replaying child: without the edge
    # there is nothing to rebase against and the assertion below is vacuous.
    assert [
        str(row[0])
        for row in conn.execute(
            "SELECT src_session_id FROM session_links WHERE inheritance = 'prefix-sharing' ORDER BY src_session_id"
        ).fetchall()
    ] == session_ids[1:]

    stored = [
        conn.execute(
            """
            SELECT total_input_tokens, total_output_tokens, total_tokens
            FROM session_provider_usage_events
            WHERE session_id = ? AND provider_event_type = 'token_count'
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        for session_id in session_ids
    ]
    assert [[tuple(row) for row in rows] for rows in stored] == [[(total, total, 2 * total)] for total in reported]
    rollups = [
        conn.execute(
            "SELECT input_tokens, output_tokens FROM session_model_usage WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        for session_id in session_ids
    ]
    assert [tuple(row) for row in rollups] == [(total, total) for total in reported]


def test_replaying_chain_with_equal_counters_keeps_every_usage_row(tmp_path: Path) -> None:
    """polylogue-uoq3x: identical reported counters must not annihilate rows.

    Anti-vacuity: with the baseline subtraction reinstated each odd link is
    clamped to zero and the companion "drop all-zero rows" delete removes it,
    so three of six links lose their usage evidence outright.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)

    turns: list[tuple[str, Role, str]] = []
    messages: list[ParsedMessage] = []
    session_ids: list[str] = []
    for link in range(6):
        turns += [(f"u{link}", Role.USER, f"prompt {link}"), (f"a{link}", Role.ASSISTANT, f"answer {link}")]
        messages = [_msg(pid, role, text, index) for index, (pid, role, text) in enumerate(turns)]
        session_ids.append(
            write_fixture_index_session(
                conn,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=f"s{link}",
                    title=f"s{link}",
                    parent_session_provider_id=(f"s{link - 1}" if link else None),
                    messages=messages,
                    session_events=[
                        ParsedSessionEvent(
                            event_type="token_count",
                            source_message_provider_id=f"a{link}",
                            payload={
                                "type": "token_count",
                                "model": "gpt-5-codex",
                                "total_token_usage": {
                                    "input_tokens": 10,
                                    "output_tokens": 10,
                                    "total_tokens": 20,
                                },
                            },
                        )
                    ],
                ),
            )
        )

    counts = [
        conn.execute(
            "SELECT COUNT(*) FROM session_provider_usage_events WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        for session_id in session_ids
    ]
    assert counts == [1] * 6


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
    child_id = write_fixture_index_session(conn, child)
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
    write_fixture_index_session(conn, parent)

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

    close_fixture_index_connection(conn)
    composed = asyncio.run(_read_texts(db, child_id))
    assert composed == ["hello", "hi there", "child diverges here", "child reply"]


def test_prefix_sharing_tail_survives_timestamps_that_precede_its_branch_point(tmp_path: Path) -> None:
    """Inheritance is positional content ancestry, never a timestamp bound.

    Nineteen prefix-sharing children in the live index begin with a tail
    message whose ``occurred_at_ms`` predates the parent branch point, and
    their tails are internally non-monotonic: Claude auto-compaction replays
    the original timestamps, and Hermes observer branches carry capture skew.
    The edge, the stored tail, and the composed transcript must all follow
    content position and ignore the wall clock.

    Anti-vacuity: two mutations turn this red. Ordering the read by
    ``occurred_at_ms`` swaps each session's two trailing messages, because both
    the child's tail and the spawned-fresh control are deliberately recorded out
    of clock order. Adding any timestamp lower bound against the branch point
    drops the tail entirely and leaves a two-message transcript. The
    spawned-fresh control keeps the negative case distinct: no shared prefix, so
    nothing may be suppressed.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)

    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0, timestamp="2026-01-01T00:00:00+00:00"),
            _msg("p1", Role.ASSISTANT, "hi there", 1, timestamp="2026-01-01T00:05:00+00:00"),
            _msg("p2", Role.USER, "parent continues alone", 2, timestamp="2026-01-01T00:09:00+00:00"),
        ],
    )
    parent_id = write_fixture_index_session(conn, parent)

    # The child replays the parent's two leading messages, then diverges with a
    # tail whose clock runs BEHIND the branch point (00:05) and backwards within
    # itself (00:04 then 00:02).
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        title="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[
            _msg("c0", Role.USER, "hello", 0, timestamp="2026-01-01T00:00:00+00:00"),
            _msg("c1", Role.ASSISTANT, "hi there", 1, timestamp="2026-01-01T00:05:00+00:00"),
            _msg("cx", Role.USER, "child diverges here", 2, timestamp="2026-01-01T00:04:00+00:00"),
            _msg("cy", Role.ASSISTANT, "child reply", 3, timestamp="2026-01-01T00:02:00+00:00"),
        ],
    )
    child_id = write_fixture_index_session(conn, child)

    link = conn.execute(
        "SELECT inheritance, branch_point_message_id, resolved_dst_session_id FROM session_links"
        " WHERE src_session_id = ?",
        (child_id,),
    ).fetchall()
    assert len(link) == 1
    assert link[0]["inheritance"] == "prefix-sharing"
    assert link[0]["resolved_dst_session_id"] == parent_id
    assert link[0]["branch_point_message_id"] == archive_message_id(parent_id, "p1")

    # Only the divergent tail is stored, in its own position order.
    stored = conn.execute(
        "SELECT position, occurred_at_ms FROM messages WHERE session_id = ? ORDER BY position", (child_id,)
    ).fetchall()
    assert [row["position"] for row in stored] == [2, 3]
    assert stored[0]["occurred_at_ms"] > stored[1]["occurred_at_ms"], "tail must stay clock-inverted"
    assert stored[0]["occurred_at_ms"] < 1_767_225_900_000, "tail must stay behind the 00:05 branch point"

    # No inherited block is duplicated onto the child.
    child_blocks = conn.execute(
        "SELECT text FROM blocks WHERE session_id = ? ORDER BY message_id, position", (child_id,)
    ).fetchall()
    assert [row["text"] for row in child_blocks] == ["child diverges here", "child reply"]

    # A spawned-fresh sibling shares no prefix, so it keeps every message and
    # records the other inheritance mode. Its clock runs backwards too: a session
    # with no lineage edge is read in the same content-position order as a
    # composed one.
    fresh = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="fresh",
        title="fresh",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[
            _msg("f0", Role.USER, "unrelated opening", 0, timestamp="2026-01-01T00:04:00+00:00"),
            _msg("f1", Role.ASSISTANT, "unrelated reply", 1, timestamp="2026-01-01T00:02:00+00:00"),
        ],
    )
    fresh_id = write_fixture_index_session(conn, fresh)
    fresh_link = conn.execute("SELECT inheritance FROM session_links WHERE src_session_id = ?", (fresh_id,)).fetchone()
    assert fresh_link["inheritance"] == "spawned-fresh"
    assert [
        row[0]
        for row in conn.execute(
            "SELECT position FROM messages WHERE session_id = ? ORDER BY position", (fresh_id,)
        ).fetchall()
    ] == [0, 1]

    close_fixture_index_connection(conn)

    assert asyncio.run(_read_texts(db, child_id)) == [
        "hello",
        "hi there",
        "child diverges here",
        "child reply",
    ]
    assert asyncio.run(_read_texts(db, fresh_id)) == ["unrelated opening", "unrelated reply"]


# The parent's three messages then the child's two-message tail, at content
# positions 0,1,2 and 2,3. Every shape carries the same content, so composition
# must return the same transcript from all of them.
_TIMESTAMP_SHAPES: dict[str, tuple[str | None, ...]] = {
    "reversed": (
        "2026-01-01T00:09:00+00:00",
        "2026-01-01T00:05:00+00:00",
        "2026-01-01T00:01:00+00:00",
        "2026-01-01T00:04:00+00:00",
        "2026-01-01T00:02:00+00:00",
    ),
    "equal": (("2026-01-01T00:03:00+00:00",) * 5),
    "missing": (None, None, None, None, None),
    # Only the tail's clock is absent, so a timestamp-ordered read would sink it
    # to the end or float it to the front depending on the NULL convention.
    "tail_missing": (
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:05:00+00:00",
        "2026-01-01T00:09:00+00:00",
        None,
        None,
    ),
    "misleading": (
        "2026-01-01T00:07:00+00:00",
        "2026-01-01T00:02:00+00:00",
        "2026-01-01T00:08:00+00:00",
        "2026-01-01T00:00:30+00:00",
        "2026-01-01T00:00:10+00:00",
    ),
}

_COMPOSED_TEXTS = ["hello", "hi there", "child diverges here", "child reply"]


def _write_prefix_sharing_pair(conn: sqlite3.Connection, stamps: tuple[str | None, ...]) -> tuple[str, str]:
    """Write a parent and a prefix-sharing child carrying ``stamps``."""
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0, timestamp=stamps[0]),
            _msg("p1", Role.ASSISTANT, "hi there", 1, timestamp=stamps[1]),
            _msg("p2", Role.USER, "parent continues alone", 2, timestamp=stamps[2]),
        ],
    )
    parent_id = write_fixture_index_session(conn, parent)
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        title="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[
            _msg("c0", Role.USER, "hello", 0, timestamp=stamps[0]),
            _msg("c1", Role.ASSISTANT, "hi there", 1, timestamp=stamps[1]),
            _msg("cz", Role.USER, "child diverges here", 2, timestamp=stamps[3]),
            _msg("ca", Role.ASSISTANT, "child reply", 3, timestamp=stamps[4]),
        ],
    )
    child_id = write_fixture_index_session(conn, child)
    return parent_id, child_id


@pytest.mark.parametrize("shape", sorted(_TIMESTAMP_SHAPES))
def test_prefix_inheritance_is_identical_under_every_timestamp_shape(tmp_path: Path, shape: str) -> None:
    """Equal, missing, reversed and misleading clocks produce one lineage.

    The five shapes carry identical content at identical positions and differ
    only in ``occurred_at_ms``. Extraction, the stored tail, the branch point and
    the composed transcript must not vary across them -- which is what makes the
    branch point positional content ancestry rather than a timestamp bound.

    Anti-vacuity: every shape turns red if the read keys on ``occurred_at_ms``.
    ``reversed`` and ``misleading`` invert on the clock itself; ``equal``,
    ``missing`` and ``tail_missing`` invert on the ``message_id`` tiebreaker a
    clock-keyed read needs, because the tail's provider ids (``cz`` then ``ca``)
    sort backwards against their content positions. Adding any timestamp lower
    bound against the branch point empties the tail on ``reversed`` and
    ``misleading``.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    parent_id, child_id = _write_prefix_sharing_pair(conn, _TIMESTAMP_SHAPES[shape])

    links = conn.execute(
        "SELECT inheritance, branch_point_message_id, resolved_dst_session_id FROM session_links"
        " WHERE src_session_id = ?",
        (child_id,),
    ).fetchall()
    assert len(links) == 1
    assert links[0]["inheritance"] == "prefix-sharing"
    assert links[0]["resolved_dst_session_id"] == parent_id
    assert links[0]["branch_point_message_id"] == archive_message_id(parent_id, "p1")

    assert [
        row[0]
        for row in conn.execute(
            "SELECT position FROM messages WHERE session_id = ? ORDER BY position", (child_id,)
        ).fetchall()
    ] == [2, 3]
    assert [
        row[0]
        for row in conn.execute(
            "SELECT b.text FROM blocks b JOIN messages m ON m.message_id = b.message_id"
            " WHERE b.session_id = ? ORDER BY m.position, b.position",
            (child_id,),
        ).fetchall()
    ] == ["child diverges here", "child reply"]

    close_fixture_index_connection(conn)
    assert asyncio.run(_read_texts(db, child_id)) == _COMPOSED_TEXTS


def test_transcript_reads_are_invariant_under_timestamp_mutation(tmp_path: Path) -> None:
    """Rewriting every stored clock cannot move a single message.

    A consumer whose answer changes when only ``occurred_at_ms`` changes is
    reading the wall clock as content order. Inverting the column against
    content position is the strongest form of that probe: it is exactly the
    sequence a timestamp-keyed read would return. Both sessions are checked --
    the child exercises lineage composition, the parent has no edge and so
    exercises the ``iter_messages`` keyset cursor, which is the route that used
    to key on the clock.

    Anti-vacuity: every route here keys on content position, and restoring
    ``occurred_at_ms`` to any one of them reverses that route alone -- on the
    pre-mutation read, whose fixture clock already disagrees with position, and
    on the post-mutation comparison, which no clock-keyed read can hold.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    parent_id, child_id = _write_prefix_sharing_pair(conn, _TIMESTAMP_SHAPES["reversed"])
    close_fixture_index_connection(conn)

    parent_texts = ["hello", "hi there", "parent continues alone"]
    before_child = asyncio.run(_read_all_routes(db, child_id))
    before_parent = asyncio.run(_read_all_routes(db, parent_id))
    assert before_child == dict.fromkeys(before_child, _COMPOSED_TEXTS)
    assert before_parent == dict.fromkeys(before_parent, parent_texts)

    conn = sqlite3.connect(db)
    # Invert the clock against content position across BOTH sessions, so the
    # parent prefix and the child tail each read backwards on a timestamp key.
    conn.execute("UPDATE messages SET occurred_at_ms = 1000000 - position * 1000")
    conn.commit()
    close_fixture_index_connection(conn)

    assert asyncio.run(_read_all_routes(db, child_id)) == before_child
    assert asyncio.run(_read_all_routes(db, parent_id)) == before_parent


async def _read_all_routes(path: Path, session_id: str) -> dict[str, list[str | None]]:
    """Read one session's transcript through every route that states its order."""
    conn = await aiosqlite.connect(path)
    try:
        conn.row_factory = aiosqlite.Row
        paginated, total, _completeness = await get_messages_paginated(conn, session_id, limit=100, offset=0)
        batched, _all_messages = await get_messages_batch(conn, [session_id])
        routes = {
            "get_messages": [record.text for record in await get_messages(conn, session_id)],
            "get_messages_paginated": [record.text for record in paginated],
            "get_messages_batch": [record.text for record in batched[session_id]],
            "iter_messages": [record.text async for record in iter_messages(conn, session_id, chunk_size=2)],
        }
        assert total == len(routes["get_messages_paginated"])
        return routes
    finally:
        await conn.close()


def test_late_parent_resolution_invalidates_child_derived_products(tmp_path: Path) -> None:
    """Re-extraction drops the child's derived session products.

    Their staleness predicate compares the session's sort key, updated-at and
    content hash, and re-extraction moves none of the three, so a profile
    materialized over the whole child would report fresh forever. Anti-vacuity:
    the seeded profile is fresh by construction (asserted before the parent
    arrives), so dropping the invalidating deletes from
    ``_reextract_prefix_tail_db`` leaves the stale rows in place and fails the
    retained-row assertion. The child's re-captured profile demand (#5525)
    also marks it stale, so the partition-status assertion alone does not
    prove the deletes.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)

    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        title="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[
            _msg("c0", Role.USER, "hello", 0, timestamp="2026-01-01T00:00:00+00:00"),
            _msg("c1", Role.ASSISTANT, "hi there", 1, timestamp="2026-01-01T00:01:00+00:00"),
            _msg("cx", Role.USER, "child diverges here", 2, timestamp="2026-01-01T00:02:00+00:00"),
            _msg("cy", Role.ASSISTANT, "child reply", 3, timestamp="2026-01-01T00:03:00+00:00"),
        ],
    )
    child_id = write_fixture_index_session(conn, child)
    _seed_fresh_session_products(conn, child_id, message_count=4)
    assert _nonvalid_partitions(conn) == []
    conn.commit()

    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0, timestamp="2026-01-01T00:00:00+00:00"),
            _msg("p1", Role.ASSISTANT, "hi there", 1, timestamp="2026-01-01T00:01:00+00:00"),
            _msg("p2", Role.USER, "parent continues alone", 2, timestamp="2026-01-01T00:05:00+00:00"),
        ],
    )
    write_fixture_index_session(conn, parent)

    stored = conn.execute(
        "SELECT position FROM messages WHERE session_id = ? ORDER BY position", (child_id,)
    ).fetchall()
    assert [row[0] for row in stored] == [2, 3]
    for relation in ("session_profiles", "session_latency_profiles"):
        retained = conn.execute(f"SELECT COUNT(*) FROM {relation} WHERE session_id = ?", (child_id,)).fetchone()[0]
        assert retained == 0, f"{relation} retained the pre-extraction projection"
    assert child_id in _nonvalid_partitions(conn)

    close_fixture_index_connection(conn)


@pytest.mark.parametrize("child_first", [False, True], ids=["parent-first", "child-first"])
def test_variant_prefix_lineage_converges_across_order_and_parent_replacement(
    tmp_path: Path, child_first: bool
) -> None:
    """The production write/read routes agree on every sibling variant.

    This is the deterministic reduction of the state-machine order failure:
    parent-first and child-first ingestion must leave the same child tail,
    resolved link, and composed transcript.  Replacing the parent with changed
    content at the branch point then invalidates the branch-point witness: the
    child keeps the transcript it inherited, materialized once as its own rows
    (both sibling variants), and the edge becomes spawned-fresh, while the
    parent reads its new values.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        updated_at="2027-01-01T00:00:01Z",
        messages=[
            _msg("p0", Role.USER, "root", 0),
            _msg("p1", Role.ASSISTANT, "primary v1", 1),
            _msg("p1-alt", Role.ASSISTANT, "sibling v1", 1, variant_index=1),
        ],
    )
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        updated_at="2027-01-01T00:00:02Z",
        messages=[
            _msg("c0", Role.USER, "root", 0),
            _msg("c1", Role.ASSISTANT, "primary v1", 1),
            _msg("c1-alt", Role.ASSISTANT, "sibling v1", 1, variant_index=1),
            _msg("c2", Role.USER, "child tail", 2),
        ],
    )

    if child_first:
        child_id = write_fixture_index_session(conn, child)
        parent_id = write_fixture_index_session(conn, parent)
    else:
        parent_id = write_fixture_index_session(conn, parent)
        child_id = write_fixture_index_session(conn, child)

    physical = conn.execute(
        "SELECT native_id, position, variant_index FROM messages WHERE session_id = ? ORDER BY position, variant_index",
        (child_id,),
    ).fetchall()
    assert [tuple(row) for row in physical] == [("c2", 2, 0)]
    link = conn.execute(
        "SELECT resolved_dst_session_id, branch_point_message_id, inheritance, status "
        "FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    assert tuple(link) == (
        parent_id,
        archive_message_id(parent_id, "p1-alt"),
        "prefix-sharing",
        None,
    )
    assert asyncio.run(_read_texts(db, child_id)) == ["root", "primary v1", "sibling v1", "child tail"]

    replacement = parent.model_copy(
        update={
            "updated_at": "2027-01-01T00:00:03Z",
            "messages": [
                _msg("p0", Role.USER, "root", 0),
                _msg("p1", Role.ASSISTANT, "primary v2", 1),
                _msg("p1-alt", Role.ASSISTANT, "sibling v2", 1, variant_index=1),
            ],
        }
    )
    write_fixture_index_session(conn, replacement)

    # A rewritten branch point no longer re-stamps the child's witness
    # (aa23fea308): the child keeps its pre-write transcript as its own rows.
    assert asyncio.run(_read_texts(db, child_id)) == ["root", "primary v1", "sibling v1", "child tail"]
    assert asyncio.run(_read_texts(db, parent_id)) == ["root", "primary v2", "sibling v2"]
    assert [
        tuple(row)
        for row in conn.execute(
            "SELECT native_id, position, variant_index FROM messages WHERE session_id = ? ORDER BY position, variant_index",
            (child_id,),
        ).fetchall()
    ] == [("p0", 0, 0), ("p1", 1, 0), ("p1-alt", 1, 1), ("c2", 2, 0)]
    assert tuple(
        conn.execute(
            "SELECT resolved_dst_session_id, branch_point_message_id, inheritance, status "
            "FROM session_links WHERE src_session_id = ?",
            (child_id,),
        ).fetchone()
    ) == (parent_id, None, "spawned-fresh", None)
    close_fixture_index_connection(conn)


def test_missing_variant_branch_point_keeps_the_child_whole(tmp_path: Path) -> None:
    """A full parent replacement that drops the child's variant branch point
    neither substitutes a shorter sibling prefix nor strands the child: the
    child keeps the prefix it replayed, as its own rows (polylogue-gy2yu)."""
    db = tmp_path / "index.db"
    conn = _connect(db)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        updated_at="2027-01-01T00:00:01Z",
        messages=[
            _msg("p0", Role.USER, "root", 0),
            _msg("p1", Role.ASSISTANT, "primary", 1),
            _msg("p1-alt", Role.ASSISTANT, "branch cut", 1, variant_index=1),
        ],
    )
    parent_id = write_fixture_index_session(conn, parent)
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        updated_at="2027-01-01T00:00:02Z",
        messages=[
            _msg("c0", Role.USER, "root", 0),
            _msg("c1", Role.ASSISTANT, "primary", 1),
            _msg("c1-alt", Role.ASSISTANT, "branch cut", 1, variant_index=1),
            _msg("c2", Role.USER, "child tail", 2),
        ],
    )
    child_id = write_fixture_index_session(conn, child)
    link = conn.execute(
        "SELECT branch_point_message_id, inheritance, status FROM session_links WHERE src_session_id = ?", (child_id,)
    ).fetchone()
    assert tuple(link) == (archive_message_id(parent_id, "p1-alt"), "prefix-sharing", None)
    before = [message.blocks[0].text for message in read_archive_session_envelope(conn, child_id).messages]
    assert before[-1] == "child tail" and len(before) > 1

    write_fixture_index_session(
        conn,
        parent.model_copy(
            update={
                "updated_at": "2027-01-01T00:00:03Z",
                "messages": [
                    _msg("p0", Role.USER, "root", 0),
                    _msg("p1", Role.ASSISTANT, "primary", 1),
                ],
            }
        ),
    )

    link = conn.execute(
        "SELECT branch_point_message_id, inheritance, status FROM session_links WHERE src_session_id = ?", (child_id,)
    ).fetchone()
    assert tuple(link) == (None, "spawned-fresh", None)
    envelope = read_archive_session_envelope(conn, child_id)
    assert [message.blocks[0].text for message in envelope.messages] == before
    assert envelope.lineage_complete is True
    # Sibling variants stay one turn: the copies keep their source coordinates.
    coordinates = conn.execute(
        "SELECT position, variant_index FROM messages WHERE session_id = ? ORDER BY position, variant_index",
        (child_id,),
    )
    assert [tuple(row) for row in coordinates] == [(0, 0), (1, 0), (1, 1), (2, 0)]
    close_fixture_index_connection(conn)


def test_reingest_after_dangling_ancestor_does_not_fabricate_a_prefix(tmp_path: Path) -> None:
    """Writer-side alignment must use the same dangling-cut bound as readers."""
    db = tmp_path / "index.db"
    conn = _connect(db)
    root = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="root",
        messages=[
            _msg("r0", Role.USER, "root prompt", 0),
            _msg("r1", Role.ASSISTANT, "root reply", 1),
        ],
    )
    root_id = write_fixture_index_session(conn, root)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        parent_session_provider_id="root",
        branch_type=BranchType.FORK,
        messages=[
            _msg("p0", Role.USER, "root prompt", 0),
            _msg("p1", Role.ASSISTANT, "root reply", 1),
            _msg("p2", Role.USER, "parent tail", 2),
        ],
    )
    parent_id = write_fixture_index_session(conn, parent)
    conn.execute("DELETE FROM messages WHERE message_id = ?", (archive_message_id(root_id, "r1"),))
    conn.commit()

    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[
            _msg("c0", Role.USER, "root prompt", 0),
            _msg("c1", Role.ASSISTANT, "root reply", 1),
            _msg("c2", Role.USER, "parent tail", 2),
            _msg("c3", Role.ASSISTANT, "child tail", 3),
        ],
    )
    child_id = write_fixture_index_session(conn, child)

    # The parent's branch cut is absent.  It contributes only its owned tail
    # to signature alignment, so the child's full replay is preserved rather
    # than deduplicating a surviving but semantically incomplete root prefix.
    link = conn.execute(
        "SELECT resolved_dst_session_id, branch_point_message_id, inheritance "
        "FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    assert tuple(link) == (parent_id, None, "spawned-fresh")
    physical = conn.execute(
        "SELECT native_id FROM messages WHERE session_id = ? ORDER BY position, variant_index",
        (child_id,),
    ).fetchall()
    assert [row[0] for row in physical] == ["c0", "c1", "c2", "c3"]
    close_fixture_index_connection(conn)


def test_nested_dangling_ancestor_keeps_only_reachable_tails(tmp_path: Path) -> None:
    """A dangling cut propagates through descendants without substituting a prefix.

    The child has a valid branch point in its immediate parent, but that parent
    itself inherits through a now-missing root branch point.  The real composed
    read route must preserve the reachable parent and child tails, mark the
    complete lineage as incomplete, and leave both physical tails and links
    intact for a later authoritative re-ingest to repair.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    root = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="root",
        messages=[
            _msg("r0", Role.USER, "root prompt", 0),
            _msg("r1", Role.ASSISTANT, "root reply", 1),
        ],
    )
    root_id = write_fixture_index_session(conn, root)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        parent_session_provider_id="root",
        branch_type=BranchType.FORK,
        messages=[
            _msg("p0", Role.USER, "root prompt", 0),
            _msg("p1", Role.ASSISTANT, "root reply", 1),
            _msg("p2", Role.USER, "parent tail", 2),
        ],
    )
    parent_id = write_fixture_index_session(conn, parent)
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[
            _msg("c0", Role.USER, "root prompt", 0),
            _msg("c1", Role.ASSISTANT, "root reply", 1),
            _msg("c2", Role.USER, "parent tail", 2),
            _msg("c3", Role.ASSISTANT, "child tail", 3),
        ],
    )
    child_id = write_fixture_index_session(conn, child)

    before_physical = conn.execute(
        "SELECT session_id, native_id, position, variant_index FROM messages "
        "WHERE session_id IN (?, ?) ORDER BY session_id, position, variant_index",
        (parent_id, child_id),
    ).fetchall()
    before_links = conn.execute(
        "SELECT src_session_id, resolved_dst_session_id, branch_point_message_id, inheritance, status "
        "FROM session_links WHERE src_session_id IN (?, ?) ORDER BY src_session_id",
        (parent_id, child_id),
    ).fetchall()

    # This is the failure shape: the parent edge still resolves to the root
    # session, but its branch-point message disappeared.  Do not rewrite a
    # resolved edge as 'unresolved': the typed degradation belongs on the
    # composed read result while its known relation remains queryable.
    conn.execute("DELETE FROM messages WHERE message_id = ?", (archive_message_id(root_id, "r1"),))
    conn.commit()

    envelope = read_archive_session_envelope(conn, child_id)
    assert [message.blocks[0].text for message in envelope.messages] == ["parent tail", "child tail"]
    assert envelope.lineage_complete is False
    assert envelope.lineage_truncation_reason == "dangling_branch_point"
    assert (
        conn.execute(
            "SELECT session_id, native_id, position, variant_index FROM messages "
            "WHERE session_id IN (?, ?) ORDER BY session_id, position, variant_index",
            (parent_id, child_id),
        ).fetchall()
        == before_physical
    )
    assert (
        conn.execute(
            "SELECT src_session_id, resolved_dst_session_id, branch_point_message_id, inheritance, status "
            "FROM session_links WHERE src_session_id IN (?, ?) ORDER BY src_session_id",
            (parent_id, child_id),
        ).fetchall()
        == before_links
    )
    close_fixture_index_connection(conn)


def test_full_replace_graph_failure_rolls_back_then_retries_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full writer transaction cannot expose replacement/link mixed state.

    Inject after the real graph-resolution phase, where replacement rows, link
    resolution, stale-cut repair, and projections have all had a chance to
    mutate.  The failed replacement must leave the earlier physical rows,
    relation row, and composed read exactly intact; the same input can then be
    retried repeatedly with one converged state.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        updated_at="2027-01-01T00:00:01Z",
        messages=[
            _msg("p0", Role.USER, "root", 0),
            _msg("p1", Role.ASSISTANT, "parent v1", 1),
        ],
    )
    parent_id = write_fixture_index_session(conn, parent)
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[
            _msg("c0", Role.USER, "root", 0),
            _msg("c1", Role.ASSISTANT, "parent v1", 1),
            _msg("c2", Role.USER, "child tail", 2),
        ],
    )
    child_id = write_fixture_index_session(conn, child)

    def _state() -> tuple[list[tuple[object, ...]], tuple[object, ...], list[str | None], bool, str | None]:
        physical = [
            tuple(row)
            for row in conn.execute(
                "SELECT session_id, native_id, position, variant_index FROM messages "
                "WHERE session_id IN (?, ?) ORDER BY session_id, position, variant_index",
                (parent_id, child_id),
            ).fetchall()
        ]
        link = tuple(
            conn.execute(
                "SELECT resolved_dst_session_id, branch_point_message_id, inheritance, status "
                "FROM session_links WHERE src_session_id = ?",
                (child_id,),
            ).fetchone()
        )
        envelope = read_archive_session_envelope(conn, child_id)
        return (
            physical,
            link,
            [message.blocks[0].text for message in envelope.messages],
            envelope.lineage_complete,
            envelope.lineage_truncation_reason,
        )

    before = _state()
    replacement = parent.model_copy(
        update={
            "updated_at": "2027-01-01T00:00:02Z",
            "messages": [
                _msg("p0", Role.USER, "root", 0),
                _msg("p1", Role.ASSISTANT, "parent v2", 1),
            ],
        }
    )
    real_resolve = _write_module._resolve_session_graph
    fired = {"count": 0}

    def _fail_after_graph_resolution(
        conn_inner: sqlite3.Connection,
        session_id: str,
        native_id: str,
        origin: str,
        *,
        cache: dict[str, list[tuple[str, str]]] | None = None,
        add_timing: Callable[[str, float], None] | None = None,
        bulk_fts: bool = False,
        bulk_build: bool = False,
        invalidated_session_ids: set[str] | None = None,
        source_read: _write_module.SessionSourceRead | None = None,
    ) -> None:
        real_resolve(
            conn_inner,
            session_id,
            native_id,
            origin,
            cache=cache,
            add_timing=add_timing,
            bulk_fts=bulk_fts,
            bulk_build=bulk_build,
            invalidated_session_ids=invalidated_session_ids,
            source_read=source_read,
        )
        fired["count"] += 1
        raise RuntimeError("fault after graph resolution")

    monkeypatch.setattr(_write_module, "_resolve_session_graph", _fail_after_graph_resolution)
    with pytest.raises(RuntimeError, match="fault after graph resolution"):
        write_fixture_index_session(conn, replacement)
    assert fired["count"] == 1, "fault injection did not reach graph resolution"
    assert _state() == before

    monkeypatch.setattr(_write_module, "_resolve_session_graph", real_resolve)
    write_fixture_index_session(conn, replacement)
    after_retry = _state()
    # The replacement changes the branch-point content, so the child keeps its
    # pre-write prefix as its own rows (aa23fea308) and stays complete.
    assert after_retry[2:] == (["root", "parent v1", "child tail"], True, None)
    assert after_retry[1][2] == "spawned-fresh"
    write_fixture_index_session(conn, replacement)
    assert _state() == after_retry
    close_fixture_index_connection(conn)


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
    ancestor_id = write_fixture_index_session(conn, ancestor)
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
    parent_id = write_fixture_index_session(conn, parent)
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
    child_id = write_fixture_index_session(conn, child)
    stale_branch_point = archive_message_id(parent_id, "a1")
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

    repaired = _repair_stale_prefix_branch_points_db(conn, {child_id})
    conn.commit()

    assert repaired == 1
    branch_point = conn.execute(
        "SELECT branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()[0]
    assert branch_point == archive_message_id(ancestor_id, "a1")
    assert [message.blocks[0].text for message in read_archive_session_envelope(conn, child_id).messages] == [
        "hello",
        "hi there",
        "child tail",
    ]
    close_fixture_index_connection(conn)


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
    ancestor_id = write_fixture_index_session(conn, ancestor)
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
    parent_id = write_fixture_index_session(conn, parent)
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
    child_id = write_fixture_index_session(conn, child)

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

    repaired = _repair_stale_prefix_branch_points_db(conn, {child_id})
    conn.commit()

    assert repaired == 1
    branch_point = conn.execute(
        "SELECT branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()[0]
    assert branch_point == archive_message_id(ancestor_id, "msg-10")
    assert [message.blocks[0].text for message in read_archive_session_envelope(conn, child_id).messages] == [
        "inherited prompt",
        "child tail",
    ]
    close_fixture_index_connection(conn)


def test_child_before_parent_reextracts_cleanly_when_foreign_keys_suspended(tmp_path: Path) -> None:
    """Bulk ingest suspends FKs while FTS triggers are dropped; re-extract must
    still remove rows that would normally be deleted by message cascades.

    The parent references the same attachment on its copy of the shared
    message, so the prefix is inherited whole and the child's own reference
    goes with the deleted row."""
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
        attachments=[
            ParsedAttachment(
                provider_attachment_id="prefix-attachment",
                message_provider_id="c1",
                name="prefix.txt",
                path="prefix.txt",
            )
        ],
    )
    child_id = write_fixture_index_session(conn, child)
    assert conn.execute("SELECT COUNT(*) FROM blocks WHERE session_id = ?", (child_id,)).fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM attachment_native_ids").fetchone()[0] == 1

    conn.execute("PRAGMA foreign_keys = OFF")
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0),
            _msg("p1", Role.ASSISTANT, "hi there", 1),
            _msg("p2", Role.USER, "parent continues alone", 2),
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="prefix-attachment",
                message_provider_id="p1",
                name="prefix.txt",
                path="prefix.txt",
            )
        ],
    )
    parent_id = write_fixture_index_session(conn, parent)

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
    # The child's reference went with its deleted prefix row, native ids
    # included; the shared attachment row survives on the parent's reference
    # with a count that says so.
    assert conn.execute("SELECT COUNT(*) FROM attachment_refs WHERE session_id = ?", (child_id,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM attachment_native_ids").fetchone()[0] == 1
    assert [tuple(row) for row in conn.execute("SELECT ref_count FROM attachments")] == [(1,)]
    event_ref = conn.execute(
        """
        SELECT source_message_id, source_message_provider_id
        FROM session_events WHERE session_id = ?
        """,
        (child_id,),
    ).fetchone()
    assert dict(event_ref) == {
        "source_message_id": archive_message_id(parent_id, "p1"),
        "source_message_provider_id": "c1",
    }

    # A source-tier rebuild reparses the child while the parent already exists.
    # The provider reference and canonical parent resolution must be identical.
    write_fixture_index_session(conn, child, force_replace=True)
    rebuilt_event_ref = conn.execute(
        """
        SELECT source_message_id, source_message_provider_id
        FROM session_events WHERE session_id = ?
        """,
        (child_id,),
    ).fetchone()
    assert dict(rebuilt_event_ref) == dict(event_ref)
    conn.rollback()
    close_fixture_index_connection(conn)


def test_child_before_parent_reextracts_empty_tail_by_session(tmp_path: Path) -> None:
    """A child that is entirely inherited should remove dependents by session.

    This covers the rebuild hot path where a large child replay is later found to
    have no divergent tail. The cleanup must remove message-owned projections
    even while foreign keys are suspended. The parent references the child's
    attachment on its own copy, so nothing the child owns keeps a row.
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
        attachments=[
            ParsedAttachment(
                provider_attachment_id="empty-tail-attachment",
                message_provider_id="c1",
                name="empty-tail.txt",
                path="empty-tail.txt",
            )
        ],
    )
    child_id = write_fixture_index_session(conn, child)
    assert conn.execute("SELECT COUNT(*) FROM attachment_native_ids").fetchone()[0] == 1
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
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0),
            _msg("p1", Role.ASSISTANT, "hi there", 1),
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="empty-tail-attachment",
                message_provider_id="p1",
                name="empty-tail.txt",
                path="empty-tail.txt",
            )
        ],
    )
    write_fixture_index_session(conn, parent)

    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (child_id,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM blocks WHERE session_id = ?", (child_id,)).fetchone()[0] == 0
    # Same cleanup as the partial-tail path: the child's reference and its
    # native ids are gone, and the shared row counts only the parent's.
    assert conn.execute("SELECT COUNT(*) FROM attachment_refs WHERE session_id = ?", (child_id,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM attachment_native_ids").fetchone()[0] == 1
    assert [tuple(row) for row in conn.execute("SELECT ref_count FROM attachments")] == [(1,)]
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
    close_fixture_index_connection(conn)


def test_child_before_parent_reextracts_provider_usage_tail(tmp_path: Path) -> None:
    """A child written before its parent keeps its own divergent-tail usage.

    When the parent arrives, re-extraction drops the usage bound to the shared
    prefix (``c1``) and keeps the tail's rows and rollup unchanged. Since #5530
    an explicit zero cumulative is a measurement that supersedes an earlier
    positive one (``test_explicit_zero_cumulative_supersedes_prior_positive``),
    so the measured-zero tick comes before the latest cumulative here; the
    rollup is the child's own latest reported cumulative, 160/30/25.
    """
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
            # A measured-zero tick: every lane explicitly 0 (polylogue-1pzmq).
            # It precedes the child's latest cumulative because an explicit
            # zero cumulative is a measurement that supersedes an earlier
            # positive one (#5530).
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="cy",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "request_id": "req-zero",
                    "last_token_usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cached_input_tokens": 0,
                        "cache_write_tokens": 0,
                        "reasoning_output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "total_token_usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cached_input_tokens": 0,
                        "cache_write_tokens": 0,
                        "reasoning_output_tokens": 0,
                        "total_tokens": 0,
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
    child_id = write_fixture_index_session(conn, child)
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
    write_fixture_index_session(conn, parent)

    usage = conn.execute(
        """
        SELECT input_tokens, output_tokens, cache_read_tokens,
               CASE WHEN provider_cost_usd IS NOT NULL THEN 'origin_reported'
                    WHEN catalog_cost_usd IS NOT NULL THEN 'priced' END AS cost_provenance
        FROM session_model_usage
        WHERE session_id = ? AND model_name = 'gpt-5-codex'
        """,
        (child_id,),
    ).fetchone()
    assert dict(usage) == {
        "input_tokens": 130,
        "output_tokens": 25,
        "cache_read_tokens": 30,
        "cost_provenance": "priced",
    }
    # polylogue-664l: session_provider_usage_events dropped its 8 Hermes
    # billing-provenance columns (index v61, zero production readers). The
    # billing-only event above has no column to hold its facts, so
    # `_provider_usage_event_has_evidence` writes no row for it. The "c1" row
    # is deleted by the prefix-tail reextraction (its source message is in the
    # shared parent prefix); only the divergent-tail "cy" rows survive, and
    # polylogue-uoq3x keeps their reported cumulatives verbatim rather than
    # rebasing them on the parent.
    remaining = conn.execute(
        """
        SELECT total_input_tokens, total_tokens, request_id
        FROM session_provider_usage_events
        WHERE session_id = ?
        ORDER BY position
        """,
        (child_id,),
    ).fetchall()
    # The tick reports a measured zero on every lane and keeps its provider
    # correlation id (polylogue-1pzmq). polylogue-uoq3x
    # removed the companion "delete every all-zero row" sweep that ran here: it
    # existed only to clear rows the baseline subtraction had clamped to zero,
    # and it destroyed this row -- and its request id -- along with them.
    assert [dict(row) for row in remaining] == [
        {"total_input_tokens": 0, "total_tokens": 0, "request_id": "req-zero"},
        {"total_input_tokens": 160, "total_tokens": 185, "request_id": None},
    ]


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
    write_fixture_index_session(conn, parent)

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
    child_id = write_fixture_index_session(conn, child)
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
    write_fixture_index_session(conn, parent_grown)

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
    write_fixture_index_session(conn, parent)

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
    child_id = write_fixture_index_session(conn, child)

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

    close_fixture_index_connection(conn)
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
    # polylogue-7y53q: the continuation carries only its own divergent tail and
    # DECLARES where it diverged, at the positions it occupies in the composed
    # transcript. Everything below still proves the archive stores that tail,
    # records the prefix-sharing edge at the parent's tip, and recomposes the
    # whole transcript on read -- the parse just stopped replaying it first.
    assert [message.text for message in continuation.messages] == ["after", "continued"]
    assert [message.position for message in continuation.messages] == [2, 3]
    assert continuation.branch_point_provider_message_id == parent.messages[-1].provider_message_id
    assert delegate.branch_type is BranchType.SUBAGENT
    assert [message.text for message in delegate.messages] == ["fresh work"]

    db = tmp_path / "index.db"
    conn = _connect(db)
    parent_id = write_fixture_index_session(conn, parent)
    continuation_id = write_fixture_index_session(conn, continuation)
    delegate_id = write_fixture_index_session(conn, delegate)

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

    write_fixture_index_session(conn, parent)
    write_fixture_index_session(conn, continuation)
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
    close_fixture_index_connection(conn)
    assert asyncio.run(_read_texts(db, continuation_id)) == ["before", "summary", "after", "continued"]

    late_root = tmp_path / "late-archive"
    late_root.mkdir()
    late_db = late_root / "index.db"
    late_conn = _connect(late_db)
    late_continuation_id = write_fixture_index_session(late_conn, continuation)
    late_parent_id = write_fixture_index_session(late_conn, parent)
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
    close_fixture_index_connection(late_conn)
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
    parent_id = write_fixture_index_session(conn, parent)
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
    child_id = write_fixture_index_session(conn, child)
    close_fixture_index_connection(conn)
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
            page1, total, completeness1 = await get_messages_paginated(conn, child_id, limit=2, offset=0)
            assert total == 4
            assert [r.text for r in page1] == full[:2]
            assert completeness1.complete is True
            page2, total2, completeness2 = await get_messages_paginated(conn, child_id, limit=2, offset=2)
            assert total2 == 4
            assert [r.text for r in page2] == full[2:]
            assert completeness2.complete is True

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


def test_fork_pages_and_completeness_skip_full_hydration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "index.db"
    _parent_id, child_id = _build_parent_and_fork(db)

    async def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a bounded lineage read hydrated an entire segment")

    monkeypatch.setattr(_message_query_reads_module, "_own_messages", forbidden)

    async def exercise() -> None:
        reader = await aiosqlite.connect(db)
        reader.row_factory = aiosqlite.Row
        statements: list[str] = []
        await reader.set_trace_callback(statements.append)
        try:
            for offset, expected in enumerate(("hello", "hi there", "child diverges here", "child reply")):
                page, total, completeness = await get_messages_paginated(reader, child_id, limit=1, offset=offset)
                assert [record.text for record in page] == [expected]
                assert total == 4
                assert completeness.complete
            page, total, _ = await get_messages_paginated(
                reader, child_id, message_role=(Role.USER,), limit=1, offset=1
            )
            assert [record.text for record in page] == ["child diverges here"]
            assert total == 2
            assert (await _message_query_reads_module.get_lineage_completeness(reader, child_id)).complete
            stream = iter_messages(reader, child_id, chunk_size=1)
            assert (await anext(stream)).text == "hello"
            assert reader.in_transaction
            await stream.aclose()
            assert not reader.in_transaction
            assert [row.text async for row in iter_messages(reader, child_id, chunk_size=1)] == [
                "hello",
                "hi there",
                "child diverges here",
                "child reply",
            ]
            assert [
                row.text
                async for row in iter_messages(reader, child_id, chunk_size=1, message_roles=(Role.USER,), limit=1)
            ] == ["hello"]
            row_reads = [sql for sql in statements if "SELECT" in sql and "FROM messages m JOIN sessions s" in sql]
            assert row_reads
            assert all(" LIMIT " in sql for sql in row_reads)
        finally:
            await reader.close()

    asyncio.run(exercise())


def test_cycle_is_typed_in_sync_and_async_reads(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    _parent_id, child_id = _build_parent_and_fork(db)
    conn = _connect(db)
    conn.execute(
        "UPDATE session_links SET resolved_dst_session_id = ? WHERE src_session_id = ? AND inheritance = 'prefix-sharing'",
        (child_id, child_id),
    )
    conn.commit()
    envelope = read_archive_session_envelope(conn, child_id)
    assert envelope.lineage_complete is False
    assert envelope.lineage_truncation_reason == "cycle"
    close_fixture_index_connection(conn)

    async def exercise() -> None:
        reader = await aiosqlite.connect(db)
        reader.row_factory = aiosqlite.Row
        try:
            full, full_completeness = await get_messages_with_lineage_completeness(reader, child_id)
            page, total, page_completeness = await get_messages_paginated(reader, child_id, limit=1)
            probe = await _message_query_reads_module.get_lineage_completeness(reader, child_id)
            assert [record.message_id for record in page] == [full[0].message_id]
            assert total == len(full)
            assert {
                full_completeness.truncation_reason,
                page_completeness.truncation_reason,
                probe.truncation_reason,
            } == {"cycle"}
            assert not full_completeness.complete and not page_completeness.complete and not probe.complete
        finally:
            await reader.close()

    asyncio.run(exercise())


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
    write_fixture_index_session(conn, parent, signature_cache=cache)

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
        return write_fixture_index_session(conn, child, signature_cache=cache)

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
    write_fixture_index_session(conn, parent_grown, signature_cache=cache)

    assert _composed(fork_a_id) == ["hello", "hi there", "fork A diverges", "fork A reply"]
    assert _composed(fork_b_id) == ["hello", "hi there", "fork B diverges", "fork B reply"]

    close_fixture_index_connection(conn)


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
    parent_id = write_fixture_index_session(conn, parent)

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
    child_id = write_fixture_index_session(conn, child)
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
    close_fixture_index_connection(conn)

    # Re-open on a fresh connection so the read below starts a clean snapshot,
    # matching how a live reader (CLI/MCP/API) connects independently of the
    # writer/daemon connection.
    reader = _connect(db)
    reader.execute("PRAGMA journal_mode=WAL")

    real_edge_lookup = _write_module._prefix_sharing_edge_sync
    fired = {"count": 0}

    def _hook(
        conn_inner: sqlite3.Connection,
        session_id: str,
        before_input: _write_module.BeforeIndexInput | None = None,
    ) -> tuple[str, str] | None:
        if session_id == child_id and fired["count"] == 0:
            fired["count"] += 1
            assert conn_inner.in_transaction, "composition must already hold a transaction before this hook fires"
            _concurrently_mutate_parent_block_text(db, parent_id)
        return real_edge_lookup(conn_inner, session_id, before_input)

    monkeypatch.setattr(_write_module, "_prefix_sharing_edge_sync", _hook)

    envelope = read_archive_session_envelope(reader, child_id)
    texts = ["".join(block.text or "" for block in message.blocks) for message in envelope.messages]

    assert fired["count"] == 1, "the interleaving hook never fired -- test is not exercising the race"
    # Old-consistent: the reader's held snapshot predates the concurrent edit,
    # so it must see the ORIGINAL parent text, not a torn mix.
    assert texts == ["hello", "hi there", "child diverges"]

    close_fixture_index_connection(reader)

    # The concurrent edit itself did land (proving it wasn't silently a no-op) --
    # a fresh read afterwards sees the new text.
    post = _connect(db)
    post_envelope = read_archive_session_envelope(post, child_id)
    post_texts = ["".join(block.text or "" for block in message.blocks) for message in post_envelope.messages]
    assert post_texts == ["hello", "hi there (concurrently edited)", "child diverges"]
    close_fixture_index_connection(post)


def test_async_composition_holds_one_snapshot_across_concurrent_parent_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async twin of the sync 4ts.4 regression above: get_messages must not
    tear when a concurrent writer edits the parent's shared prefix mid-walk."""
    db = tmp_path / "index.db"
    conn, parent_id, child_id = _setup_interleaving_fixture(db)
    close_fixture_index_connection(conn)

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
    write_fixture_index_session(conn, parent)

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
    child_id = write_fixture_index_session(conn, child)

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

    close_fixture_index_connection(conn)

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

    # polylogue-ppkj: the actual `polylogue read` / HTTP messages surface
    # goes through get_messages_paginated, not get_messages_with_lineage_
    # completeness directly -- prove the signal survives that call too,
    # instead of being dropped by the plain get_messages() wrapper it used
    # to route through for composed (prefix-sharing) sessions.
    async def _run_paginated() -> tuple[list[str | None], int, LineageCompleteness]:
        reader = await aiosqlite.connect(db)
        try:
            reader.row_factory = aiosqlite.Row
            records, total, page_completeness = await get_messages_paginated(reader, child_id, limit=50, offset=0)
            return [r.text for r in records], total, page_completeness
        finally:
            await reader.close()

    page_texts, page_total, page_completeness = asyncio.run(_run_paginated())
    assert page_texts == ["child diverges"]
    assert page_total == 1
    assert page_completeness.complete is False
    assert page_completeness.truncation_reason == "dangling_branch_point"


#: Deeper than every depth cap the lineage path used to carry (the recursive
#: sync reader stopped at 64), so a reintroduced cap truncates this chain.
_DEEP_CHAIN_LEVELS = 70


def test_deep_chain_composes_complete_on_every_reader(tmp_path: Path) -> None:
    """A valid acyclic chain composes whole however deep it is.

    The visited set is the only walk bound. Anti-vacuity: reinstate a depth
    cap below ``_DEEP_CHAIN_LEVELS`` in ``_composed_transcript_plan`` or in
    ``get_messages_with_lineage_completeness`` and the leaf reads incomplete,
    missing the root message.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    provider_session_id = "root"
    write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=provider_session_id,
            title="root",
            messages=[_msg("root-0", Role.USER, "root message", 0)],
        ),
    )
    leaf_id = None
    for level in range(_DEEP_CHAIN_LEVELS):
        child_provider_id = f"level-{level}"
        leaf_id = write_fixture_index_session(
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
    assert envelope.lineage_complete is True
    assert envelope.lineage_truncation_reason is None
    assert envelope.messages[0].blocks[0].text == "root message"
    close_fixture_index_connection(conn)

    async def _run() -> LineageCompleteness:
        reader = await aiosqlite.connect(db)
        try:
            reader.row_factory = aiosqlite.Row
            _records, completeness = await get_messages_with_lineage_completeness(reader, leaf_id)
            return completeness
        finally:
            await reader.close()

    completeness = asyncio.run(_run())
    assert completeness.complete is True
    assert completeness.truncation_reason is None


#: Deeper than the largest cap the lineage walks used to carry (1024).
_VERY_DEEP_CHAIN_LEVELS = 1030


def test_chain_deeper_than_every_former_cap_composes_complete(tmp_path: Path) -> None:
    """polylogue-cpisn: cycle detection alone terminates every lineage walk.

    The chain is seeded as one-message sessions linked by SQL, so its depth is
    cheap to build and every level inherits exactly its parent's composed
    transcript. Anti-vacuity: reinstate a depth cap below
    ``_VERY_DEEP_CHAIN_LEVELS`` in ``_composed_transcript_plan``,
    ``_composed_db_signatures``, ``get_messages_with_lineage_completeness`` or
    ``_CompositionShape.segments`` and the matching assertion below fails.
    """
    from polylogue.storage.derived.lineage.compact import _CompositionShape
    from polylogue.storage.sqlite.archive_tiers.write import _composed_db_signatures

    db = tmp_path / "index.db"
    conn = _connect(db)
    sessions = [
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=f"deep-{level:04d}",
            title=f"deep-{level:04d}",
            messages=[_msg("m", Role.USER, f"level {level}", 0)],
        )
        for level in range(_VERY_DEEP_CHAIN_LEVELS)
    ]
    # These seeds have no parent dependencies. The first real write initializes
    # the archive; all remaining inputs prepare before one owned transaction.
    session_ids = [write_fixture_index_session(conn, sessions[0])]
    with (
        prepared_fixture_index_batch(conn, sessions[1:], archive_root=tmp_path) as (seal, prepared),
        _fixture_writer_admission(conn, "test.fixture.deep-lineage", tmp_path),
        seal.mutation_scope(conn),
    ):
        session_ids.extend(
            write_fixture_index_session(conn, session, prepared_write=carrier)
            for session, carrier in zip(sessions[1:], prepared, strict=True)
        )
    conn.executemany(
        """
        INSERT INTO session_links(
            src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id,
            branch_point_message_id, inheritance, status, confidence, evidence_json, observed_at_ms
        ) VALUES (?, 'codex-session', ?, 'fork', ?, ?, 'prefix-sharing', NULL, 1.0, '[]', 0)
        """,
        (
            (child, parent.split(":", 1)[1], parent, archive_message_id(parent, "m"))
            for parent, child in zip(session_ids, session_ids[1:], strict=False)
        ),
    )
    conn.commit()
    leaf_id = session_ids[-1]
    expected = [f"level {level}" for level in range(_VERY_DEEP_CHAIN_LEVELS)]

    envelope = read_archive_session_envelope(conn, leaf_id)
    assert envelope.lineage_complete is True
    assert [message.blocks[0].text for message in envelope.messages] == expected
    assert len(_composed_db_signatures(conn, leaf_id)) == _VERY_DEEP_CHAIN_LEVELS
    assert _CompositionShape(conn).segments(leaf_id) is not None
    close_fixture_index_connection(conn)

    async def _run() -> tuple[int, LineageCompleteness]:
        reader = await aiosqlite.connect(db)
        try:
            reader.row_factory = aiosqlite.Row
            records, completeness = await get_messages_with_lineage_completeness(reader, leaf_id)
            return len(records), completeness
        finally:
            await reader.close()

    count, completeness = asyncio.run(_run())
    assert completeness.complete is True
    assert count == _VERY_DEEP_CHAIN_LEVELS


def test_writer_composes_beyond_recursive_reader_depth(tmp_path: Path) -> None:
    """A valid branch point beyond the sync reader's stack guard stays valid.

    Writer composition walks the whole chain. If it stopped early, the later
    descendants could not see the root branch point and would become
    spawned-fresh with a duplicated root message.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    provider_session_id = "deep-root"
    root_id = write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=provider_session_id,
            title="deep root",
            messages=[_msg("root-0", Role.USER, "root message", 0)],
        ),
    )

    leaf: ParsedSession | None = None
    leaf_id: str | None = None
    for level in range(_DEEP_CHAIN_LEVELS):
        child_provider_id = f"deep-level-{level}"
        leaf = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=child_provider_id,
            title=child_provider_id,
            parent_session_provider_id=provider_session_id,
            branch_type=BranchType.FORK,
            updated_at=f"2027-01-01T00:{level // 60:02d}:{level % 60:02d}Z",
            messages=[
                _msg("root-0", Role.USER, "root message", 0),
                _msg(f"tail-{level}", Role.ASSISTANT, f"level {level} tail", 1),
            ],
        )
        leaf_id = write_fixture_index_session(conn, leaf)
        provider_session_id = child_provider_id

    assert leaf is not None
    assert leaf_id is not None
    link = conn.execute(
        "SELECT inheritance, branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (leaf_id,),
    ).fetchone()
    assert tuple(link) == ("prefix-sharing", archive_message_id(root_id, "root-0"))
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (leaf_id,)).fetchone()[0] == 1

    # Full replacement/re-ingest exercises writer alignment again rather than
    # merely preserving the link produced on the first write.
    write_fixture_index_session(
        conn,
        leaf.model_copy(update={"updated_at": "2027-01-01T00:59:59Z"}),
    )
    link = conn.execute(
        "SELECT inheritance, branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (leaf_id,),
    ).fetchone()
    assert tuple(link) == ("prefix-sharing", archive_message_id(root_id, "root-0"))
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (leaf_id,)).fetchone()[0] == 1
    close_fixture_index_connection(conn)

    assert asyncio.run(_read_texts(db, leaf_id)) == ["root message", f"level {_DEEP_CHAIN_LEVELS - 1} tail"]


def test_shallow_chain_reports_complete(tmp_path: Path) -> None:
    """Sanity check: a normal, shallow fork reports lineage_complete=True --
    the completeness signal must not be trivially always-false."""
    db = tmp_path / "index.db"
    conn = _connect(db)

    write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="parent",
            title="parent",
            messages=[_msg("p0", Role.USER, "hello", 0)],
        ),
    )
    child_id = write_fixture_index_session(
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
    close_fixture_index_connection(conn)


def _three_generation_sessions() -> tuple[ParsedSession, ParsedSession, ParsedSession]:
    """Grandparent P, parent B (P's prefix + one tail turn), child A (P's first
    two turns + its own tail). Mirrors the executed reproduction on
    polylogue-7xrv5."""
    grandparent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="gp",
        title="grandparent",
        messages=[
            _msg("m0", Role.USER, "m0", 0),
            _msg("m1", Role.ASSISTANT, "m1", 1),
            _msg("m2", Role.USER, "m2", 2),
            _msg("m3", Role.ASSISTANT, "m3", 3),
        ],
    )
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        parent_session_provider_id="gp",
        branch_type=BranchType.FORK,
        messages=[
            _msg("m0", Role.USER, "m0", 0),
            _msg("m1", Role.ASSISTANT, "m1", 1),
            _msg("m2", Role.USER, "m2", 2),
            _msg("m3", Role.ASSISTANT, "m3", 3),
            _msg("m4", Role.USER, "m4", 4),
        ],
    )
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        title="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[
            _msg("m0", Role.USER, "m0", 0),
            _msg("m1", Role.ASSISTANT, "m1", 1),
            _msg("x2", Role.USER, "x2", 2),
        ],
    )
    return grandparent, parent, child


@pytest.mark.parametrize(
    "order",
    [
        ("grandparent", "parent", "child"),
        ("grandparent", "child", "parent"),
        ("child", "parent", "grandparent"),
        ("parent", "child", "grandparent"),
        ("child", "grandparent", "parent"),
        ("parent", "grandparent", "child"),
    ],
)
def test_three_generation_lineage_composes_identically_in_every_visit_order(
    tmp_path: Path, order: tuple[str, str, str]
) -> None:
    """polylogue-7xrv5: a rebuild visits sources in lexicographic key order, so
    the grandparent can land after both descendants. Re-extracting the parent
    deletes exactly the rows the child's branch point names; unless the child is
    added to the in-write repair scope its edge dangles and it composes to its
    own tail only.

    Anti-vacuity: reverting the ``reextract_invalidated_ids`` contribution to
    ``impacted_session_ids`` in ``_resolve_session_graph`` makes every
    grandparent-last ordering read ``['x2']``.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    grandparent, parent, child = _three_generation_sessions()
    by_name = {"grandparent": grandparent, "parent": parent, "child": child}

    written: dict[str, str] = {}
    for name in order:
        written[name] = write_fixture_index_session(conn, by_name[name])
    conn.commit()

    assert [message.blocks[0].text for message in read_archive_session_envelope(conn, written["child"]).messages] == [
        "m0",
        "m1",
        "x2",
    ]
    assert [message.blocks[0].text for message in read_archive_session_envelope(conn, written["parent"]).messages] == [
        "m0",
        "m1",
        "m2",
        "m3",
        "m4",
    ]
    assert count_dangling_prefix_branch_points(conn) == (0, 0)
    close_fixture_index_connection(conn)


def test_dangling_branch_point_census_counts_edges_and_sessions(tmp_path: Path) -> None:
    """polylogue-7xrv5: the archive-wide census is what makes a truncating
    archive measurable, and (polylogue-ga6ib) it is now the *only* archive-wide
    thing there is -- the daemon reports this count at startup and corrects
    nothing. The one corrector is the writer's own scoped, in-transaction call.

    Anti-vacuity: a census that ignored the branch point's existence (or scoped
    itself to one session) would report ``(0, 0)`` for the corrupted state
    below.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    grandparent, parent, child = _three_generation_sessions()
    write_fixture_index_session(conn, grandparent)
    parent_id = write_fixture_index_session(conn, parent)
    child_id = write_fixture_index_session(conn, child)
    conn.commit()
    assert count_dangling_prefix_branch_points(conn) == (0, 0)

    conn.execute(
        "UPDATE session_links SET branch_point_message_id = ? WHERE src_session_id = ?",
        (f"{parent_id}:n:m1", child_id),
    )
    conn.commit()
    assert count_dangling_prefix_branch_points(conn) == (1, 1)
    assert [message.blocks[0].text for message in read_archive_session_envelope(conn, child_id).messages] == ["x2"]

    assert _repair_stale_prefix_branch_points_db(conn, {child_id}) == 1
    conn.commit()
    assert count_dangling_prefix_branch_points(conn) == (0, 0)
    close_fixture_index_connection(conn)


def _hermes_chain_state_db(
    path: Path,
    *,
    links: int,
    messages_per_session: int,
    session_tokens: list[tuple[int, int]] | None = None,
) -> None:
    """A Hermes state.db of ``links`` sessions, each a compression continuation of the last."""
    with sqlite3.connect(path) as source:
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
                input_tokens INTEGER,
                output_tokens INTEGER,
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
            """
        )
        message_id = 0
        for index in range(links):
            session_id = f"s{index:04d}"
            parent = f"s{index - 1:04d}" if index else None
            reported = session_tokens[index] if session_tokens else (None, None)
            source.execute(
                "INSERT INTO sessions VALUES (?, 'cli', '{}', ?, ?, ?, 'compression', ?, ?, ?)",
                (
                    session_id,
                    parent,
                    float(index),
                    float(index) + 0.5,
                    reported[0],
                    reported[1],
                    f"Session {index}",
                ),
            )
            for offset in range(messages_per_session):
                message_id += 1
                source.execute(
                    "INSERT INTO messages VALUES (?, ?, ?, ?, ?, NULL, 0, 1, 0)",
                    (
                        message_id,
                        session_id,
                        "user" if offset % 2 == 0 else "assistant",
                        f"{session_id} message {offset}",
                        float(index) + offset / 1000,
                    ),
                )


def test_hermes_continuation_chain_far_past_the_retired_bound_parses(tmp_path: Path) -> None:
    """A continuation chain longer than any declared bound is ingested, not refused.

    Before polylogue-7y53q the parser composed each continuation child's full
    inherited prefix and the archive writer aligned it straight back off, so a
    chain of ``N`` links holding ``M`` messages each retained ``M*N*(N+1)/2``
    messages. A ~2 MB ``state.db`` reached 1.6M messages and 5.2 GB RSS, and
    the parser answered with a permanent typed refusal
    (``HermesLineageBoundError``) at 128 links. The chain below is 200 links.

    Anti-vacuity: restore the composing pass and this raises instead of
    returning -- 200 exceeds the old depth bound of 128. A parity-only
    assertion on a short chain passes under both implementations, which is why
    the chain is built past the retired bound rather than at three links.
    """
    links = 200
    messages_per_session = 4
    state_db = tmp_path / "state.db"
    _hermes_chain_state_db(state_db, links=links, messages_per_session=messages_per_session)

    parsed = parse_state_db(state_db)

    assert len(parsed) == links
    # Retention is the source's own message count, not its square.
    assert sum(len(session.messages) for session in parsed) == links * messages_per_session

    by_raw_id = {session.provider_session_id.split("@", 1)[0]: session for session in parsed}
    for index in range(links):
        session = by_raw_id[f"s{index:04d}"]
        # Each link carries only its own rows...
        assert [message.text for message in session.messages] == [
            f"s{index:04d} message {offset}" for offset in range(messages_per_session)
        ]
        # ...at the positions they occupy in the composed transcript...
        assert [message.position for message in session.messages] == list(
            range(index * messages_per_session, (index + 1) * messages_per_session)
        )
        # ...and declares where it diverged, rather than replaying the prefix.
        if index == 0:
            assert session.branch_point_provider_message_id is None
        else:
            parent = by_raw_id[f"s{index - 1:04d}"]
            assert session.branch_point_provider_message_id == parent.messages[-1].provider_message_id


def test_hermes_continuation_segments_store_each_message_once_and_read_composed(tmp_path: Path) -> None:
    """The declared divergence still composes the whole chain on read.

    The segmented parse only moves where the prefix is expressed: the archive
    must still hold each message exactly once and recompose the full
    transcript for every link (#2467).
    """
    links = 12
    messages_per_session = 3
    state_db = tmp_path / "state.db"
    _hermes_chain_state_db(state_db, links=links, messages_per_session=messages_per_session)
    conn = _connect(tmp_path / "index.db")

    written = [write_fixture_index_session(conn, session) for session in parse_state_db(state_db)]
    conn.commit()

    stored = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
    assert stored == links * messages_per_session, "storage must hold each message exactly once"

    for index, session_id in enumerate(written):
        envelope = read_archive_session_envelope(conn, session_id)
        assert envelope.lineage_complete is True
        assert [message.blocks[0].text for message in envelope.messages] == [
            f"s{link:04d} message {offset}" for link in range(index + 1) for offset in range(messages_per_session)
        ]
        assert envelope.lineage_inheritance == ("prefix-sharing" if index else "none")

    # The edge carries the staleness witness, so a parent whose branch-point
    # content changes cannot silently keep composing underneath the child.
    witnesses = conn.execute(
        "SELECT branch_point_content_address FROM session_links WHERE inheritance = 'prefix-sharing'"
    ).fetchall()
    assert len(witnesses) == links - 1
    assert all(row[0] is not None for row in witnesses)
    close_fixture_index_connection(conn)


def test_hermes_continuation_child_keeps_its_own_reported_usage(tmp_path: Path) -> None:
    """A continuation child's usage stays the totals its own session row reported.

    Hermes reports cumulative counters per session, not across a continuation
    chain (``docs/cost-model.md``), so a chain's stored totals must track the
    reported 10, 13, 16, ... exactly.

    Anti-vacuity: make the parser emit composed prefixes again and every link
    past the first stores the parent's totals on top of its own, so the
    segment-per-session identity this asserts is lost. polylogue-uoq3x removed
    the writer-side rebase that used to turn the same input into the sawtooth
    10, 3, 13, 6, 16, ...; the un-rebased writer is now pinned for every
    replaying origin by
    ``test_replaying_chain_stores_each_link_reported_cumulative``.
    """
    links = 5
    state_db = tmp_path / "state.db"
    _hermes_chain_state_db(
        state_db,
        links=links,
        messages_per_session=2,
        session_tokens=[(10 + index * 3, 20 + index * 5) for index in range(links)],
    )
    conn = _connect(tmp_path / "index.db")
    written = [write_fixture_index_session(conn, session) for session in parse_state_db(state_db)]
    conn.commit()

    totals = {
        str(row["session_id"]): (int(row["total_input_tokens"]), int(row["total_output_tokens"]))
        for row in conn.execute(
            "SELECT session_id, total_input_tokens, total_output_tokens FROM session_provider_usage_events"
        ).fetchall()
    }

    assert [totals[session_id] for session_id in written] == [(10 + i * 3, 20 + i * 5) for i in range(links)]
    close_fixture_index_connection(conn)


def test_alias_invalidation_records_retryable_convergence_debt(tmp_path: Path) -> None:
    """A provider-session identity contradiction NULLs a child's lineage
    columns, leaving it reading as a complete root with its recomposed prefix
    gone. That loss must be *named* as retryable convergence debt targeting the
    invalidated child, not left silent until someone orders a full rebuild
    (polylogue-e0xan).

    Anti-vacuity: drop the ``_record_identity_invalidation_debt`` call (or
    point it at a different target) and the debt row for ``child_id`` is
    absent, so the assertions below go red.
    """
    db = tmp_path / "index.db"
    initialize_archive_database(tmp_path / "ops.db", ArchiveTier.OPS)
    conn = _connect(db)

    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="alpha",
        title="alpha",
        provider_session_aliases=["shared-stem"],
        messages=[_msg("a0", Role.USER, "hello", 0), _msg("a1", Role.ASSISTANT, "hi there", 1)],
    )
    write_fixture_index_session(conn, parent)

    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        title="child",
        parent_session_provider_id="shared-stem",
        branch_type=BranchType.FORK,
        messages=[
            _msg("c0", Role.USER, "hello", 0),
            _msg("c1", Role.ASSISTANT, "hi there", 1),
            _msg("cx", Role.USER, "child diverges here", 2),
        ],
    )
    child_id = write_fixture_index_session(conn, child)
    resolved_before = conn.execute(
        "SELECT resolved_dst_session_id FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    assert resolved_before["resolved_dst_session_id"] is not None

    ops = sqlite3.connect(tmp_path / "ops.db")
    try:
        assert ops.execute("SELECT COUNT(*) FROM convergence_debt").fetchone()[0] == 0
    finally:
        ops.close()

    # A second session claims the same alias: the claim is now ambiguous and
    # the child's lineage columns are invalidated.
    contender = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="beta",
        title="beta",
        provider_session_aliases=["shared-stem"],
        messages=[_msg("b0", Role.USER, "unrelated", 0)],
    )
    write_fixture_index_session(conn, contender)
    resolved_after = conn.execute(
        "SELECT resolved_dst_session_id, branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    assert resolved_after["resolved_dst_session_id"] is None
    assert resolved_after["branch_point_message_id"] is None

    ops = sqlite3.connect(tmp_path / "ops.db")
    ops.row_factory = sqlite3.Row
    try:
        debt = ops.execute(
            "SELECT stage, target_type, target_id, status, attempts, last_error, next_retry_at FROM convergence_debt"
        ).fetchall()
    finally:
        ops.close()
    assert [(row["target_type"], row["target_id"]) for row in debt] == [("session_id", child_id)]
    assert debt[0]["stage"] == IDENTITY_INVALIDATION_DEBT_STAGE
    assert debt[0]["status"] == "failed"
    assert debt[0]["attempts"] >= 1
    assert "identity contradiction" in debt[0]["last_error"]
    # Retryable, not an inert marker: the row carries a scheduled retry.
    assert debt[0]["next_retry_at"]


def test_only_a_prefix_sharing_edge_may_carry_a_branch_anchor(tmp_path: Path) -> None:
    """The canonical DDL rejects a contradictory inheritance/anchor pair.

    A ``spawned-fresh`` child references its parent without inheriting a
    prefix, and an undecided edge has no evidence for a divergence point, so
    neither may carry ``branch_point_message_id`` /
    ``branch_point_content_address``. Composition reads the anchor as the last
    inherited parent message, so a contradictory row is a lineage the reader
    would silently compose from (polylogue-pkst AC1).

    The law is deliberately one-directional: a prefix-sharing edge whose
    parent message has not landed yet keeps a NULL anchor, which the last two
    positive cases pin -- tightening it to require the anchor would reject
    captured evidence for an unarrived parent.

    Anti-vacuity: this drives the real generated DDL through
    ``initialize_archive_database``, so deleting the table constraint from
    ``SESSION_LINKS_SPEC`` makes both refusals succeed; the positive cases
    fail if the constraint is written as a plain equality, because
    ``inheritance = 'prefix-sharing'`` evaluates to NULL for an undecided
    edge and SQLite lets a NULL CHECK pass.
    """
    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    conn = sqlite3.connect(index_db)
    try:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES ('child', 'codex-session', zeroblob(32))"
        )

        def insert(dst: str, inheritance: str | None, anchor: str | None) -> None:
            conn.execute(
                "INSERT INTO session_links (src_session_id, dst_origin, dst_native_id, link_type, "
                "inheritance, branch_point_message_id, method, confidence, evidence_json, observed_at_ms) "
                "VALUES ('codex-session:child', 'codex-session', ?, 'fork', ?, ?, 'test', 1.0, '[]', 1)",
                (dst, inheritance, anchor),
            )

        anchor = "codex-session:parent:n:m1"
        with pytest.raises(sqlite3.IntegrityError):
            insert("p", "spawned-fresh", anchor)
        with pytest.raises(sqlite3.IntegrityError):
            insert("q", None, anchor)

        # Representable: a decided prefix-sharing edge, and the unresolved
        # shapes the resolver actually writes before a parent arrives.
        insert("r", "prefix-sharing", anchor)
        insert("s", "prefix-sharing", None)
        insert("t", "spawned-fresh", None)
        insert("u", None, None)
        conn.commit()
        stored = {
            str(row[0]): (row[1], row[2])
            for row in conn.execute("SELECT dst_native_id, inheritance, branch_point_message_id FROM session_links")
        }
    finally:
        conn.close()

    assert stored == {
        "r": ("prefix-sharing", anchor),
        "s": ("prefix-sharing", None),
        "t": ("spawned-fresh", None),
        "u": (None, None),
    }


def test_a_content_address_witness_is_also_refused_without_prefix_sharing(tmp_path: Path) -> None:
    """The same law covers the branch point's content-address witness.

    ``branch_point_content_address`` is derived only alongside
    ``branch_point_message_id`` (``archive_tiers/write.py``'s
    ``_message_content_address_for_id``), and a reader treats it as proof the
    anchor still names the message it was bound to. A witness on a
    non-prefix-sharing edge is the same contradiction.

    Anti-vacuity: drop ``branch_point_content_address`` from the constraint
    and this insert succeeds.
    """
    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    conn = sqlite3.connect(index_db)
    try:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES ('child', 'codex-session', zeroblob(32))"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO session_links (src_session_id, dst_origin, dst_native_id, link_type, "
                "inheritance, branch_point_content_address, method, confidence, evidence_json, observed_at_ms) "
                "VALUES ('codex-session:child', 'codex-session', 'p', 'fork', 'spawned-fresh', "
                "zeroblob(32), 'test', 1.0, '[]', 1)"
            )
    finally:
        conn.close()


def _codex_session(session_id: str, texts: list[str], *, parent: str | None = None) -> ParsedSession:
    """One Codex session whose message native ids are its texts."""
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_id,
        title=session_id,
        parent_session_provider_id=parent,
        branch_type=BranchType.FORK if parent is not None else None,
        messages=[
            _msg(text, Role.USER if index % 2 == 0 else Role.ASSISTANT, text, index) for index, text in enumerate(texts)
        ],
    )


def _composed_texts(conn: sqlite3.Connection, session_id: str) -> list[str | None]:
    return [message.blocks[0].text for message in read_archive_session_envelope(conn, session_id).messages]


def _edge_state(conn: sqlite3.Connection, child_id: str) -> tuple[str | None, str | None, str | None]:
    row = conn.execute(
        "SELECT resolved_dst_session_id, inheritance, branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    return (row[0], row[1], row[2])


def test_dropped_branch_point_keeps_the_child_whole(tmp_path: Path) -> None:
    """polylogue-gy2yu: a full replace that SHORTENS the parent never strands a child.

    A full replace is the ordinary route for a re-acquired source. The child's
    inherited prefix is evidence from the child's own bytes, so when the
    replacement drops the branch-point message the same write materializes the
    inherited rows into the child: the child stops inheriting, stays
    topologically linked to its parent, and reads complete. No convergence debt
    is recorded, because nothing was lost.

    Anti-vacuity: skip ``_settle_inherited_prefixes`` in
    ``write_parsed_session_to_archive`` and the child composes to ``["x2"]``
    with ``dangling_branch_point``.
    """
    from polylogue.sources.live.cursor import CursorStore

    db = tmp_path / "index.db"
    cursor = CursorStore(db)
    conn = _connect(db)

    parent_id = write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1", "m2"]))
    child_id = write_fixture_index_session(conn, _codex_session("child", ["m0", "m1", "x2"], parent="parent"))
    conn.commit()
    assert _composed_texts(conn, child_id) == ["m0", "m1", "x2"]
    assert _edge_state(conn, child_id) == (parent_id, "prefix-sharing", f"{parent_id}:n:m1")

    # The re-acquired export no longer carries m1 -- the child's branch point.
    write_fixture_index_session(conn, _codex_session("parent", ["m0", "m2"]))
    conn.commit()

    assert _composed_texts(conn, parent_id) == ["m0", "m2"]
    envelope = read_archive_session_envelope(conn, child_id)
    assert envelope.lineage_complete is True
    assert [message.blocks[0].text for message in envelope.messages] == ["m0", "m1", "x2"]
    assert _edge_state(conn, child_id) == (parent_id, "spawned-fresh", None)
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (child_id,)).fetchone()[0] == 3
    assert count_dangling_prefix_branch_points(conn) == (0, 0)
    assert cursor.list_convergence_debt() == []
    assert conn.execute("SELECT name FROM sqlite_temp_master").fetchall() == []
    close_fixture_index_connection(conn)


def test_intact_parent_rewrite_leaves_the_child_inheriting(tmp_path: Path) -> None:
    """Opposite direction: re-writing a parent whose inherited rows survive
    unchanged copies nothing. Materializing unconditionally would fail here."""
    db = tmp_path / "index.db"
    conn = _connect(db)
    parent_id = write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1", "m2"]))
    child_id = write_fixture_index_session(conn, _codex_session("child", ["m0", "m1", "x2"], parent="parent"))
    write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1", "m2", "m3"]))
    conn.commit()

    assert _composed_texts(conn, child_id) == ["m0", "m1", "x2"]
    assert _edge_state(conn, child_id) == (parent_id, "prefix-sharing", f"{parent_id}:n:m1")
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (child_id,)).fetchone()[0] == 1
    close_fixture_index_connection(conn)


def test_grandchild_follows_rows_its_parent_materialized(tmp_path: Path) -> None:
    """A grandchild that branched inside the child's inherited prefix keeps
    composing after that prefix moves into the child.

    Anti-vacuity: drop the descendant branch-point rewrite in
    ``_materialize_inherited_prefix`` and the grandchild's branch point names
    a deleted parent row, so it reads ``dangling_branch_point``.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    parent_id = write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1", "m2", "m3"]))
    child_id = write_fixture_index_session(conn, _codex_session("child", ["m0", "m1", "m2", "x3"], parent="parent"))
    grandchild_id = write_fixture_index_session(conn, _codex_session("grandchild", ["m0", "m1", "g2"], parent="child"))
    conn.commit()
    assert _composed_texts(conn, grandchild_id) == ["m0", "m1", "g2"]
    assert _edge_state(conn, grandchild_id)[2] == f"{parent_id}:n:m1"

    write_fixture_index_session(conn, _codex_session("parent", ["m0", "m3"]))
    conn.commit()

    assert _composed_texts(conn, child_id) == ["m0", "m1", "m2", "x3"]
    grandchild = read_archive_session_envelope(conn, grandchild_id)
    assert grandchild.lineage_complete is True
    assert [message.blocks[0].text for message in grandchild.messages] == ["m0", "m1", "g2"]
    assert _edge_state(conn, grandchild_id) == (child_id, "prefix-sharing", f"{child_id}:n:m1")
    close_fixture_index_connection(conn)


def test_materialized_prefix_restores_a_swept_attachment(tmp_path: Path) -> None:
    """The dropped branch-point message was the only reference to an
    attachment, so the parent's replace sweeps the attachment row before the
    child's copy needs it.

    Anti-vacuity: skip restoring the snapshotted attachment rows and the
    copied ref fails its foreign key, rolling back the parent's write.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[_msg("p0", Role.USER, "hello", 0), _msg("p1", Role.ASSISTANT, "hi there", 1)],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="prefix-attachment",
                message_provider_id="p1",
                name="prefix.txt",
                path="prefix.txt",
            )
        ],
    )
    parent_id = write_fixture_index_session(conn, parent)
    child_id = write_fixture_index_session(
        conn, _codex_session("child", ["hello", "hi there", "child diverges here"], parent="parent")
    )
    conn.commit()

    write_fixture_index_session(
        conn, parent.model_copy(update={"messages": [_msg("p0", Role.USER, "hello", 0)], "attachments": []})
    )
    conn.commit()

    assert _edge_state(conn, child_id) == (parent_id, "spawned-fresh", None)
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert [tuple(row) for row in conn.execute("SELECT session_id FROM attachment_refs")] == [(child_id,)]
    assert [tuple(row) for row in conn.execute("SELECT ref_count FROM attachments")] == [(1,)]
    close_fixture_index_connection(conn)


def test_materialized_prefix_remaps_child_event_refs_with_fks_suspended(tmp_path: Path) -> None:
    """In a bulk rebuild the parent's delete does not null the child's event
    reference, so the remap must match the captured id as well as NULL.

    Anti-vacuity: restore only ``source_message_id IS NULL`` rows and the
    event keeps naming the deleted parent row.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    child_id = write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=[
                _msg("c0", Role.USER, "hello", 0),
                _msg("c1", Role.ASSISTANT, "hi there", 1),
                _msg("cx", Role.USER, "child diverges here", 2),
            ],
            session_events=[
                ParsedSessionEvent(
                    event_type="capture_gap", source_message_provider_id="c1", payload={"summary": "prefix event"}
                )
            ],
        ),
    )
    parent = _codex_session("parent", ["hello", "hi there"])
    parent_id = write_fixture_index_session(conn, parent)
    conn.commit()
    events = "SELECT source_message_id FROM session_events WHERE session_id = ?"
    assert [tuple(row) for row in conn.execute(events, (child_id,))] == [(f"{parent_id}:n:hi there",)]

    conn.execute("PRAGMA foreign_keys = OFF")
    write_fixture_index_session(conn, _codex_session("parent", ["hello"]))
    conn.commit()

    assert _composed_texts(conn, child_id) == ["hello", "hi there", "child diverges here"]
    assert [tuple(row) for row in conn.execute(events, (child_id,))] == [(f"{child_id}:n:hi there",)]
    close_fixture_index_connection(conn)


def test_materialized_empty_tail_child_gets_an_active_leaf(tmp_path: Path) -> None:
    """A child that replayed its parent completely stored no rows of its own;
    once it owns the prefix, its last message is its active leaf.

    Anti-vacuity: drop ``_keep_an_active_leaf`` and the child's pointer names
    no row.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1"]))
    child_id = write_fixture_index_session(conn, _codex_session("child", ["m0", "m1"], parent="parent"))
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (child_id,)).fetchone()[0] == 0

    write_fixture_index_session(conn, _codex_session("parent", ["m0"]))
    conn.commit()

    assert _composed_texts(conn, child_id) == ["m0", "m1"]
    leaf = archive_message_id(child_id, "m1")
    pointer = conn.execute("SELECT active_leaf_message_id FROM sessions WHERE session_id = ?", (child_id,)).fetchone()
    assert tuple(pointer) == (leaf,)
    leaves = conn.execute("SELECT message_id FROM messages WHERE session_id = ? AND is_active_leaf = 1", (child_id,))
    assert [tuple(row) for row in leaves] == [(leaf,)]
    close_fixture_index_connection(conn)


def test_materialization_evidence_survives_a_default_array_evidence(tmp_path: Path) -> None:
    """An edge still holding the schema-default ``[]`` evidence records why it
    stopped inheriting.

    Anti-vacuity: guard the ``json_set`` with ``json_valid`` alone and the
    array passes unchanged, so the materialized edge is indistinguishable from
    an originally fresh spawn.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1"]))
    child_id = write_fixture_index_session(conn, _codex_session("child", ["m0", "m1", "c"], parent="parent"))
    conn.execute("UPDATE session_links SET evidence_json = '[]' WHERE src_session_id = ?", (child_id,))
    conn.commit()

    write_fixture_index_session(conn, _codex_session("parent", ["m0"]))
    conn.commit()

    link = conn.execute(
        "SELECT inheritance, json_extract(evidence_json, '$.inherited_prefix') FROM session_links "
        "WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    assert tuple(link) == ("spawned-fresh", "materialized-after-parent-rewrite")
    assert _composed_texts(conn, child_id) == ["m0", "m1", "c"]
    close_fixture_index_connection(conn)


def test_relocated_branch_point_repairs_in_write(tmp_path: Path) -> None:
    """The repairable half of polylogue-gy2yu, closed by the producer itself.

    Re-acquiring ``parent`` with a parent claim of its own normalizes it to
    tail-only storage, so ``m1`` moves from ``parent`` to ``gp``. The child's
    branch point still names ``parent:n:m1``. That row is gone, but the message
    survives in the parent's *composed* transcript, so the edge can be
    re-resolved onto ``gp:n:m1`` -- and must be, at write time, with no
    archive-wide sweep involved.

    Anti-vacuity: dropping ``anchored_stranded_ids`` from ``impacted_session_ids``
    makes the child compose ``['x2']`` with ``lineage_complete`` False here.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)

    write_fixture_index_session(conn, _codex_session("gp", ["m0", "m1", "m2", "m3"]))
    write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1", "m2", "m3", "m4"]))
    child_id = write_fixture_index_session(conn, _codex_session("child", ["m0", "m1", "x2"], parent="parent"))
    conn.commit()
    assert _composed_texts(conn, child_id) == ["m0", "m1", "x2"]

    write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1", "m2", "m3", "m4"], parent="gp"))
    conn.commit()

    envelope = read_archive_session_envelope(conn, child_id)
    assert [message.blocks[0].text for message in envelope.messages] == ["m0", "m1", "x2"]
    assert envelope.lineage_complete is True
    assert envelope.lineage_truncation_reason is None
    assert count_dangling_prefix_branch_points(conn) == (0, 0)
    assert (
        conn.execute(
            "SELECT branch_point_message_id FROM session_links WHERE src_session_id = ?", (child_id,)
        ).fetchone()[0]
        == "codex-session:gp:n:m1"
    )
    close_fixture_index_connection(conn)


def test_descendant_anchored_through_an_intermediate_parent_stays_whole(tmp_path: Path) -> None:
    """``A -> B -> C`` with C branching inside B's inherited prefix, at an
    A-owned row. Rewriting A without that row loses a row of B's inherited
    prefix, so B materializes it, and C follows B's copy of its branch point
    although the row it named in A is gone.

    Anti-vacuity: capture only A's direct children and C reads
    ``dangling_branch_point``.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    write_fixture_index_session(conn, _codex_session("a", ["m0", "m1", "m2", "m3"]))
    b_id = write_fixture_index_session(conn, _codex_session("b", ["m0", "m1", "m2", "b3"], parent="a"))
    c_id = write_fixture_index_session(conn, _codex_session("c", ["m0", "m1", "c2"], parent="b"))
    conn.commit()
    assert _edge_state(conn, c_id)[:2] == (b_id, "prefix-sharing")

    write_fixture_index_session(conn, _codex_session("a", ["m0", "m2", "m3"]))
    conn.commit()

    envelope = read_archive_session_envelope(conn, c_id)
    assert envelope.lineage_complete is True
    assert [message.blocks[0].text for message in envelope.messages] == ["m0", "m1", "c2"]
    assert _edge_state(conn, c_id) == (b_id, "prefix-sharing", f"{b_id}:n:m1")
    assert _composed_texts(conn, b_id) == ["m0", "m1", "m2", "b3"]
    assert read_archive_session_envelope(conn, b_id).lineage_complete is True
    close_fixture_index_connection(conn)


def test_reanchored_child_keeps_its_event_reference(tmp_path: Path) -> None:
    """A relocated branch point re-anchors the child; the child's own event
    reference into the relocated row follows it instead of staying NULL.

    Anti-vacuity: restore references without the re-anchor map and the event
    keeps ``source_message_id`` NULL after the parent's delete nulled it.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    write_fixture_index_session(conn, _codex_session("gp", ["m0", "m1", "m2", "m3"]))
    child = _codex_session("child", ["m0", "m1", "x2"], parent="parent").model_copy(
        update={
            "session_events": [
                ParsedSessionEvent(
                    event_type="capture_gap", source_message_provider_id="m1", payload={"summary": "prefix event"}
                )
            ]
        }
    )
    child_id = write_fixture_index_session(conn, child)
    parent_id = write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1", "m2", "m3", "m4"]))
    conn.commit()
    events = "SELECT source_message_id FROM session_events WHERE session_id = ?"
    assert [tuple(row) for row in conn.execute(events, (child_id,))] == [(f"{parent_id}:n:m1",)]

    write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1", "m2", "m3", "m4"], parent="gp"))
    conn.commit()

    assert _edge_state(conn, child_id)[1:] == ("prefix-sharing", "codex-session:gp:n:m1")
    assert [tuple(row) for row in conn.execute(events, (child_id,))] == [("codex-session:gp:n:m1",)]
    close_fixture_index_connection(conn)


def test_materialized_dispatch_block_keeps_its_subagent_edge(tmp_path: Path) -> None:
    """A subagent dispatched from a tool call inside the child's inherited
    prefix keeps its dispatch pointer once that call moves into the child.

    Anti-vacuity: skip ``_restore_dispatch_refs`` and the delete's
    ``ON DELETE SET NULL`` leaves the edge's ``parent_tool_use_block_id`` NULL.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    dispatch = ParsedMessage(
        provider_message_id="m1",
        role=Role.ASSISTANT,
        text="",
        position=1,
        blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, tool_name="Task", tool_id="task-1", tool_input={})],
    )
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[_msg("m0", Role.USER, "go", 0), dispatch],
    )
    parent_id = write_fixture_index_session(conn, parent)
    child_id = write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=[_msg("m0", Role.USER, "go", 0), dispatch, _msg("x2", Role.USER, "tail", 2)],
        ),
    )
    worker_id = write_fixture_index_session(conn, _codex_session("worker", ["w0"]))
    conn.execute(
        """
        INSERT INTO session_links(
            src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id,
            inheritance, status, parent_tool_use_block_id, confidence, evidence_json, observed_at_ms
        ) VALUES (?, 'codex-session', 'child', 'subagent', ?, 'spawned-fresh', NULL, ?, 1.0, '[]', 0)
        """,
        (worker_id, child_id, f"{parent_id}:n:m1:0"),
    )
    conn.commit()

    write_fixture_index_session(conn, parent.model_copy(update={"messages": [_msg("m0", Role.USER, "go", 0)]}))
    conn.commit()

    assert _edge_state(conn, child_id) == (parent_id, "spawned-fresh", None)
    pointer = conn.execute(
        "SELECT parent_tool_use_block_id FROM session_links WHERE src_session_id = ?", (worker_id,)
    ).fetchone()
    assert tuple(pointer) == (f"{child_id}:n:m1:0",)
    close_fixture_index_connection(conn)


def test_materialized_prefix_across_ancestors_keeps_counts_and_references(tmp_path: Path) -> None:
    """``gp -> parent -> child``: materializing the child copies a gp-owned row
    as well as parent-owned ones. The copied gp attachment ref raises that
    attachment's ``ref_count``, and the child's event pointing at the gp row
    follows the row into the child.

    Anti-vacuity: refresh only the snapshotted attachments and ``ref_count``
    stays 1 beside two refs; capture references only to parent-owned rows and
    the event keeps naming the gp row, outside the child's transcript.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    child_id = write_fixture_index_session(
        conn,
        _codex_session("child", ["m0", "m1", "p2", "x3"], parent="parent").model_copy(
            update={
                "session_events": [
                    ParsedSessionEvent(
                        event_type="capture_gap", source_message_provider_id="m1", payload={"summary": "gp event"}
                    )
                ]
            }
        ),
    )
    gp = _codex_session("gp", ["m0", "m1"]).model_copy(
        update={
            "attachments": [
                ParsedAttachment(
                    provider_attachment_id="gp-attachment", message_provider_id="m1", name="gp.txt", path="gp.txt"
                )
            ]
        }
    )
    gp_id = write_fixture_index_session(conn, gp)
    write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1", "p2", "p3"], parent="gp"))
    conn.commit()
    events = "SELECT source_message_id FROM session_events WHERE session_id = ?"
    assert [tuple(row) for row in conn.execute(events, (child_id,))] == [(f"{gp_id}:n:m1",)]

    write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1", "p3"], parent="gp"))
    conn.commit()

    assert _composed_texts(conn, child_id) == ["m0", "m1", "p2", "x3"]
    assert _edge_state(conn, child_id)[1:] == ("spawned-fresh", None)
    assert [tuple(row) for row in conn.execute(events, (child_id,))] == [(f"{child_id}:n:m1",)]
    assert [tuple(row) for row in conn.execute("SELECT COUNT(*) FROM attachment_refs")] == [(2,)]
    assert [tuple(row) for row in conn.execute("SELECT ref_count FROM attachments")] == [(2,)]
    close_fixture_index_connection(conn)


def test_renumbered_tail_moves_its_compaction_boundary(tmp_path: Path) -> None:
    """When the copied prefix cannot fit below the child's own positions, the
    tail shifts, and a compaction boundary addressing the tail shifts with it.

    Anti-vacuity: shift messages without the boundaries and the boundary keeps
    addressing position 0, now a copied prefix row.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[_msg("m0", Role.USER, "m0", 0), _msg("m1", Role.ASSISTANT, "m1", 1)],
    )
    write_fixture_index_session(conn, parent)
    child_id = write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=[
                _msg("m0", Role.USER, "m0", 5),
                _msg("m1", Role.ASSISTANT, "m1", 6),
                _msg("x", Role.USER, "tail", 0),
            ],
            session_events=[
                ParsedSessionEvent(event_type="compaction", boundary_start_position=0, boundary_end_position=0)
            ],
        ),
    )
    conn.commit()
    tail = "SELECT position FROM messages WHERE session_id = ? AND native_id = 'x'"
    assert [tuple(row) for row in conn.execute(tail, (child_id,))] == [(0,)]

    write_fixture_index_session(conn, parent.model_copy(update={"messages": [_msg("m0", Role.USER, "m0", 0)]}))
    conn.commit()

    assert _edge_state(conn, child_id)[1:] == ("spawned-fresh", None)
    assert [tuple(row) for row in conn.execute(tail, (child_id,))] == [(2,)]
    boundary = conn.execute(
        "SELECT boundary_start_position, boundary_end_position FROM session_events WHERE session_id = ?", (child_id,)
    )
    assert [tuple(row) for row in boundary] == [(2, 2)]
    close_fixture_index_connection(conn)


def test_dispatch_pointer_follows_the_dispatchers_own_lineage(tmp_path: Path) -> None:
    """Two children of the rewritten parent materialize the same tool call;
    the subagent dispatched through ``c``'s descendant must point at ``c``'s
    copy, not at the lexicographically first sibling's.

    Anti-vacuity: pick any materialized owner and the pointer lands in
    ``b-sibling``.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    dispatch = ParsedMessage(
        provider_message_id="m1",
        role=Role.ASSISTANT,
        text="",
        position=1,
        blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, tool_name="Task", tool_id="task-1", tool_input={})],
    )

    def session(name: str, parent: str | None, tail: list[tuple[str, str]]) -> ParsedSession:
        messages = [_msg("m0", Role.USER, "go", 0), dispatch]
        messages += [_msg(native, Role.USER, text, index + 2) for index, (native, text) in enumerate(tail)]
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=name,
            title=name,
            parent_session_provider_id=parent,
            branch_type=BranchType.FORK if parent else None,
            messages=messages,
        )

    parent = session("parent", None, [])
    parent_id = write_fixture_index_session(conn, parent)
    write_fixture_index_session(conn, session("b-sibling", "parent", [("b2", "sibling tail")]))
    c_id = write_fixture_index_session(conn, session("c", "parent", [("c2", "c tail")]))
    d_id = write_fixture_index_session(conn, session("d", "c", [("c2", "c tail"), ("d3", "d tail")]))
    worker_id = write_fixture_index_session(conn, _codex_session("worker", ["w0"]))
    conn.execute(
        """
        INSERT INTO session_links(
            src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id,
            inheritance, status, parent_tool_use_block_id, confidence, evidence_json, observed_at_ms
        ) VALUES (?, 'codex-session', 'd', 'subagent', ?, 'spawned-fresh', NULL, ?, 1.0, '[]', 0)
        """,
        (worker_id, d_id, f"{parent_id}:n:m1:0"),
    )
    conn.commit()

    write_fixture_index_session(conn, parent.model_copy(update={"messages": [_msg("m0", Role.USER, "go", 0)]}))
    conn.commit()

    pointer = conn.execute(
        "SELECT parent_tool_use_block_id FROM session_links WHERE src_session_id = ?", (worker_id,)
    ).fetchone()
    assert tuple(pointer) == (f"{c_id}:n:m1:0",)
    close_fixture_index_connection(conn)


def test_ownership_is_by_session_not_by_id_prefix(tmp_path: Path) -> None:
    """A parent whose native id extends the child's (``child`` / ``child:parent``)
    shares the child's message-id text prefix; its rows are still inherited.

    Anti-vacuity: classify ownership by ``startswith`` and the inherited rows
    look child-owned, nothing is materialized, and the write raises.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    parent_id = write_fixture_index_session(conn, _codex_session("child:parent", ["m0", "m1", "m2"]))
    child_id = write_fixture_index_session(conn, _codex_session("child", ["m0", "m1", "x2"], parent="child:parent"))
    conn.commit()
    assert _edge_state(conn, child_id)[:2] == (parent_id, "prefix-sharing")

    write_fixture_index_session(conn, _codex_session("child:parent", ["m0", "m2"]))
    conn.commit()

    assert _composed_texts(conn, child_id) == ["m0", "m1", "x2"]
    assert _edge_state(conn, child_id) == (parent_id, "spawned-fresh", None)
    close_fixture_index_connection(conn)


def test_descendant_through_a_materialized_ancestor_follows_the_copy(tmp_path: Path) -> None:
    """``G -> P -> B -> C -> D`` with D branching at a G-owned row inherited
    through C. Rewriting P strands B, which materializes its whole prefix; D,
    two levels below B and never anchored in P, must follow the copy.

    Anti-vacuity: remap only B's direct children and D reads
    ``dangling_branch_point``.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    base = ["m0", "m1", "m2", "m3", "m4"]
    write_fixture_index_session(conn, _codex_session("g", base))
    write_fixture_index_session(conn, _codex_session("p", [*base, "p5"], parent="g"))
    b_id = write_fixture_index_session(conn, _codex_session("b", [*base, "p5", "b6"], parent="p"))
    c_id = write_fixture_index_session(conn, _codex_session("c", [*base, "p5", "b6", "c7"], parent="b"))
    d_id = write_fixture_index_session(conn, _codex_session("d", ["m0", "m1", "d2"], parent="c"))
    conn.commit()
    assert _edge_state(conn, d_id)[0] == c_id

    write_fixture_index_session(conn, _codex_session("p", [*base, "q5"], parent="g"))
    conn.commit()

    assert _edge_state(conn, b_id)[1:] == ("spawned-fresh", None)
    envelope = read_archive_session_envelope(conn, d_id)
    assert envelope.lineage_complete is True
    assert [message.blocks[0].text for message in envelope.messages] == ["m0", "m1", "d2"]
    assert _edge_state(conn, d_id) == (c_id, "prefix-sharing", f"{b_id}:n:m1")
    close_fixture_index_connection(conn)


def test_prefix_compaction_boundary_stays_on_the_copied_rows(tmp_path: Path) -> None:
    """When only the tail shifts, a boundary over the inherited prefix keeps
    addressing the copied rows while a boundary over the tail moves, and a
    range crossing from the prefix into the tail moves only its end.

    Anti-vacuity: shift every boundary and the prefix boundary lands on a
    different copied message; shift by the start endpoint alone and the
    crossing range stays ``(0, 1)``, leaving the moved tail outside it.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[_msg("m0", Role.USER, "m0", 0), _msg("m1", Role.ASSISTANT, "m1", 1)],
    )
    write_fixture_index_session(conn, parent)
    child_id = write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=[
                _msg("m0", Role.USER, "m0", 0),
                _msg("m1", Role.ASSISTANT, "m1", 1),
                _msg("x", Role.USER, "tail", 1, variant_index=1),
            ],
            session_events=[
                ParsedSessionEvent(event_type="compaction", boundary_start_position=0, boundary_end_position=0),
                ParsedSessionEvent(event_type="compaction", boundary_start_position=1, boundary_end_position=1),
                ParsedSessionEvent(event_type="compaction", boundary_start_position=0, boundary_end_position=1),
            ],
        ),
    )
    conn.commit()

    write_fixture_index_session(conn, parent.model_copy(update={"messages": [_msg("m0", Role.USER, "m0", 0)]}))
    conn.commit()

    assert _edge_state(conn, child_id)[1:] == ("spawned-fresh", None)
    tail = "SELECT position FROM messages WHERE session_id = ? AND native_id = 'x'"
    assert [tuple(row) for row in conn.execute(tail, (child_id,))] == [(2,)]
    boundaries = conn.execute(
        """SELECT boundary_start_position, boundary_end_position FROM session_events
           WHERE session_id = ? ORDER BY boundary_start_position, boundary_end_position""",
        (child_id,),
    )
    assert [tuple(row) for row in boundaries] == [(0, 0), (0, 2), (2, 2)]
    close_fixture_index_connection(conn)


def test_reanchor_across_an_inserted_prefix_row_keeps_refs_and_dispatch(tmp_path: Path) -> None:
    """The parent is re-parsed under a grandparent whose prefix inserts a row
    before the child's unchanged branch point. The child stays composable,
    and both its event and a subagent dispatch pointer into the relocated
    tool call follow it.

    Anti-vacuity: require equal prefix lengths in ``_reanchor_inherited_rows`` and the
    event stays NULL; drop the re-anchor fallback in
    ``_restore_dispatch_refs`` and the pointer stays NULL.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    call = ParsedMessage(
        provider_message_id="b",
        role=Role.ASSISTANT,
        text="",
        position=1,
        blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, tool_name="Task", tool_id="task-1", tool_input={})],
    )

    def moved(message: ParsedMessage, position: int) -> ParsedMessage:
        return message.model_copy(update={"position": position})

    gp_id = write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="gp",
            title="gp",
            messages=[_msg("m0", Role.USER, "m0", 0), _msg("ins", Role.USER, "inserted", 1), moved(call, 2)],
        ),
    )
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[_msg("m0", Role.USER, "m0", 0), call, _msg("m2", Role.USER, "m2", 2)],
    )
    # Child first: its event then resolves onto the parent row at extraction.
    child_id = write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=[_msg("m0", Role.USER, "m0", 0), call, _msg("x", Role.USER, "tail", 2)],
            session_events=[
                ParsedSessionEvent(event_type="capture_gap", source_message_provider_id="b", payload={"summary": "e"})
            ],
        ),
    )
    parent_id = write_fixture_index_session(conn, parent)
    events = "SELECT source_message_id FROM session_events WHERE session_id = ?"
    assert [tuple(row) for row in conn.execute(events, (child_id,))] == [(f"{parent_id}:n:b",)]
    worker_id = write_fixture_index_session(conn, _codex_session("worker", ["w0"]))
    conn.execute(
        """
        INSERT INTO session_links(
            src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id,
            inheritance, status, parent_tool_use_block_id, confidence, evidence_json, observed_at_ms
        ) VALUES (?, 'codex-session', 'child', 'subagent', ?, 'spawned-fresh', NULL, ?, 1.0, '[]', 0)
        """,
        (worker_id, child_id, f"{parent_id}:n:b:0"),
    )
    conn.commit()

    write_fixture_index_session(
        conn,
        parent.model_copy(
            update={
                "parent_session_provider_id": "gp",
                "branch_type": BranchType.FORK,
                "messages": [
                    _msg("m0", Role.USER, "m0", 0),
                    _msg("ins", Role.USER, "inserted", 1),
                    moved(call, 2),
                    _msg("m2", Role.USER, "m2", 3),
                ],
            }
        ),
    )
    conn.commit()

    assert _edge_state(conn, child_id)[1:] == ("prefix-sharing", f"{gp_id}:n:b")
    assert [tuple(row) for row in conn.execute(events, (child_id,))] == [(f"{gp_id}:n:b",)]
    pointer = conn.execute(
        "SELECT parent_tool_use_block_id FROM session_links WHERE src_session_id = ?", (worker_id,)
    ).fetchone()
    assert tuple(pointer) == (f"{gp_id}:n:b:0",)
    close_fixture_index_connection(conn)


def test_reanchor_never_hands_a_removed_duplicate_the_surviving_row(tmp_path: Path) -> None:
    """The prefix holds two identical messages with different native ids;
    the rewrite removes the first and keeps the second. A lost inherited row
    materializes the child's whole pre-write prefix, so each reference follows
    its own copy: the removed one is neither nulled nor redirected onto the
    survivor.

    Anti-vacuity: keep inheriting because the branch point survived and the
    ``a1`` event is nulled; match copies by content alone and both events
    name one row.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)

    def parent(messages: list[ParsedMessage]) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX, provider_session_id="parent", title="parent", messages=messages
        )

    # Child first: its events then resolve onto the parent rows at extraction.
    child_id = write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=[
                _msg("u0", Role.USER, "go", 0),
                _msg("a1", Role.ASSISTANT, "same", 1),
                _msg("a2", Role.ASSISTANT, "same", 2),
                _msg("x3", Role.USER, "tail", 3),
            ],
            session_events=[
                ParsedSessionEvent(event_type="capture_gap", source_message_provider_id=pid, payload={"summary": pid})
                for pid in ("a1", "a2")
            ],
        ),
    )
    parent_id = write_fixture_index_session(
        conn,
        parent(
            [
                _msg("u0", Role.USER, "go", 0),
                _msg("a1", Role.ASSISTANT, "same", 1),
                _msg("a2", Role.ASSISTANT, "same", 2),
                _msg("u3", Role.USER, "next", 3),
            ]
        ),
    )
    conn.commit()
    # The parent's rewrite leaves the child's event rows in place: a1's first, a2's second.
    events = "SELECT source_message_id FROM session_events WHERE session_id = ? ORDER BY rowid"
    assert [tuple(row) for row in conn.execute(events, (child_id,))] == [
        (f"{parent_id}:n:a1",),
        (f"{parent_id}:n:a2",),
    ]

    write_fixture_index_session(
        conn,
        parent(
            [_msg("u0", Role.USER, "go", 0), _msg("a2", Role.ASSISTANT, "same", 1), _msg("u3", Role.USER, "next", 2)]
        ),
    )
    conn.commit()

    assert _edge_state(conn, child_id) == (parent_id, "spawned-fresh", None)
    assert _composed_texts(conn, child_id) == ["go", "same", "same", "tail"]
    referenced = [row[0] for row in conn.execute(events, (child_id,))]
    owned = {str(row[0]) for row in conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (child_id,))}
    assert len(set(referenced)) == 2 and set(referenced) <= owned
    close_fixture_index_connection(conn)


def _dispatch_session(name: str, parent: str | None, tail: list[str]) -> ParsedSession:
    """``m0`` then a ``Task`` tool call ``m1``, then text messages named by ``tail``."""
    call = ParsedMessage(
        provider_message_id="m1",
        role=Role.ASSISTANT,
        text="",
        position=1,
        blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, tool_name="Task", tool_id="task-1", tool_input={})],
    )
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=name,
        title=name,
        parent_session_provider_id=parent,
        branch_type=BranchType.FORK if parent else None,
        messages=[
            _msg("m0", Role.USER, "go", 0),
            call,
            *(_msg(text, Role.USER, text, index + 2) for index, text in enumerate(tail)),
        ],
    )


def test_dispatch_pointer_into_a_live_ancestor_block_follows_the_copy(tmp_path: Path) -> None:
    """``gp -> parent -> child`` with the dispatched tool call owned by ``gp``,
    which the rewrite leaves alive. Materializing the child copies that call;
    a subagent dispatched from the child now points at the child's copy.

    Anti-vacuity: capture dispatch pointers only into the rewritten parent's
    blocks and the edge keeps naming ``gp``'s block, outside the child's
    now self-contained transcript.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    child_id = write_fixture_index_session(conn, _dispatch_session("child", "parent", ["p2", "x3"]))
    gp_id = write_fixture_index_session(conn, _dispatch_session("gp", None, []))
    write_fixture_index_session(conn, _dispatch_session("parent", "gp", ["p2", "p3"]))
    worker_id = write_fixture_index_session(conn, _codex_session("worker", ["w0"]))
    conn.execute(
        """
        INSERT INTO session_links(
            src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id,
            inheritance, status, parent_tool_use_block_id, confidence, evidence_json, observed_at_ms
        ) VALUES (?, 'codex-session', 'child', 'subagent', ?, 'spawned-fresh', NULL, ?, 1.0, '[]', 0)
        """,
        (worker_id, child_id, f"{gp_id}:n:m1:0"),
    )
    conn.commit()

    write_fixture_index_session(conn, _dispatch_session("parent", "gp", ["p3"]))
    conn.commit()

    assert _edge_state(conn, child_id)[1:] == ("spawned-fresh", None)
    pointer = conn.execute(
        "SELECT parent_tool_use_block_id FROM session_links WHERE src_session_id = ?", (worker_id,)
    ).fetchone()
    assert tuple(pointer) == (f"{child_id}:n:m1:0",)
    close_fixture_index_connection(conn)


def test_uncaptured_descendant_references_follow_the_materialized_copy(tmp_path: Path) -> None:
    """``G -> P -> B -> C -> D``: D branches at a G-owned row and its event
    names that row. Neither D's parent nor its branch point is P's, so the
    guard never captured D; once B materializes, D composes through B's copy
    and its event must follow the copy too.

    Anti-vacuity: remap only the descendants' branch points and D's event
    keeps naming ``g``'s row, outside its composed transcript.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    base = ["m0", "m1", "m2", "m3", "m4"]
    g_id = write_fixture_index_session(conn, _codex_session("g", base))
    write_fixture_index_session(conn, _codex_session("p", [*base, "p5"], parent="g"))
    b_id = write_fixture_index_session(conn, _codex_session("b", [*base, "p5", "b6"], parent="p"))
    # D before C: D's event then resolves onto the composed row when C arrives.
    d_id = write_fixture_index_session(
        conn,
        _codex_session("d", ["m0", "m1", "d2"], parent="c").model_copy(
            update={
                "session_events": [
                    ParsedSessionEvent(
                        event_type="capture_gap", source_message_provider_id="m0", payload={"summary": "d event"}
                    )
                ]
            }
        ),
    )
    write_fixture_index_session(conn, _codex_session("c", [*base, "p5", "b6", "c7"], parent="b"))
    conn.commit()
    events = "SELECT source_message_id FROM session_events WHERE session_id = ?"
    assert [tuple(row) for row in conn.execute(events, (d_id,))] == [(f"{g_id}:n:m0",)]

    write_fixture_index_session(conn, _codex_session("p", [*base, "q5"], parent="g"))
    conn.commit()

    assert _edge_state(conn, b_id)[1:] == ("spawned-fresh", None)
    assert _edge_state(conn, d_id)[2] == f"{b_id}:n:m1"
    assert [tuple(row) for row in conn.execute(events, (d_id,))] == [(f"{b_id}:n:m0",)]
    close_fixture_index_connection(conn)


def test_materialized_generated_producer_ref_names_the_copy(tmp_path: Path) -> None:
    """An id-less assistant message owns a model-output attachment, so the
    writer names its producer ``message:<stored id>``. Materializing that
    message into the child re-roots the generated producer onto the copy.

    Anti-vacuity: copy ``producer_ref`` verbatim and the child's attachment
    names the deleted parent message as its producer.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    made = ParsedMessage(
        provider_message_id="",
        role=Role.ASSISTANT,
        text="made a file",
        position=1,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="made a file")],
    )
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[_msg("m0", Role.USER, "go", 0), made],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="generated", message_position=1, message_variant_index=0, name="out.txt"
            )
        ],
    )
    parent_id = write_fixture_index_session(conn, parent)
    refs = "SELECT message_id, producer_ref FROM attachment_refs WHERE session_id = ?"
    [(parent_message, parent_producer)] = [tuple(row) for row in conn.execute(refs, (parent_id,))]
    assert parent_producer == f"message:{parent_message}"
    child_id = write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=[_msg("m0", Role.USER, "go", 0), made, _msg("x2", Role.USER, "tail", 2)],
        ),
    )
    conn.commit()

    write_fixture_index_session(
        conn, parent.model_copy(update={"messages": [_msg("m0", Role.USER, "go", 0)], "attachments": []})
    )
    conn.commit()

    assert _edge_state(conn, child_id)[1:] == ("spawned-fresh", None)
    [(child_message, child_producer)] = [tuple(row) for row in conn.execute(refs, (child_id,))]
    assert child_message.startswith(f"{child_id}:")
    assert child_producer == f"message:{child_message}"
    close_fixture_index_connection(conn)


def test_anchored_branch_point_lookup_uses_the_branch_index(tmp_path: Path) -> None:
    """Every session write runs this lookup, so it must not scan ``session_links``.

    A full-corpus replay pays it once per session. Measured on the live index
    shape (23,496 sessions, 9,497 edges): 0.617 ms per write as a scan -- 14.5 s
    of replay that grows as sessions x edges -- against 0.0053 ms as an indexed
    range probe.

    Anti-vacuity: dropping ``idx_session_links_branch_point`` from the index
    tier DDL, or rewriting the range predicate as ``substr(...) = ?`` /
    ``LIKE``, puts ``SCAN`` back in the plan.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    plan = conn.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT DISTINCT l.src_session_id
        FROM session_links l
        WHERE l.branch_point_message_id >= :low AND l.branch_point_message_id < :high
        """,
        {"low": "codex-session:parent:", "high": "codex-session:parent;"},
    ).fetchall()
    detail = " | ".join(str(row["detail"]) for row in plan)
    assert "idx_session_links_branch_point" in detail, detail
    assert "SCAN" not in detail, detail
    close_fixture_index_connection(conn)


def test_deferred_asserted_branch_point_with_a_lone_surrogate_binds_on_parent_save(tmp_path: Path) -> None:
    """A child saved before its parent binds its surrogate-bearing branch point later.

    Anti-vacuity: read the asserted id back with bare ``json_extract`` and the
    parent's save raises ``Could not decode to UTF-8`` and rolls back.
    """
    state_db = tmp_path / "state.db"
    _hermes_chain_state_db(state_db, links=2, messages_per_session=2)
    parent, child = sorted(parse_state_db(state_db), key=lambda session: session.provider_session_id)
    surrogate_id = "m\ud800"
    last = parent.messages[-1].model_copy(update={"provider_message_id": surrogate_id})
    parent = parent.model_copy(update={"messages": [*parent.messages[:-1], last]})
    child = child.model_copy(update={"branch_point_provider_message_id": surrogate_id})
    conn = _connect(tmp_path / "index.db")

    child_id = write_fixture_index_session(conn, child)
    conn.commit()
    write_fixture_index_session(conn, parent)
    conn.commit()

    bound = conn.execute(
        "SELECT branch_point_message_id FROM session_links WHERE src_session_id = ?", (child_id,)
    ).fetchone()
    assert bound is not None and bound[0] is not None
    close_fixture_index_connection(conn)


def _child_ids(conn: sqlite3.Connection, child_id: str) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute("SELECT message_id FROM messages WHERE session_id = ? ORDER BY position", (child_id,))
    ]


def _materialize_then_replay(
    tmp_path: Path,
    parent_messages: list[ParsedMessage],
    child_messages: list[ParsedMessage],
    rewritten: list[ParsedMessage],
) -> tuple[list[str], list[str], list[str]]:
    """Tail IDs while inheriting, IDs after the parent rewrite, IDs after a child replay."""
    conn = _connect(tmp_path / "index.db")

    def session(name: str, messages: list[ParsedMessage], parent: str | None = None) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=name,
            title=name,
            parent_session_provider_id=parent,
            branch_type=BranchType.FORK if parent is not None else None,
            messages=messages,
        )

    write_fixture_index_session(conn, session("parent", parent_messages))
    child_id = write_fixture_index_session(conn, session("child", child_messages, parent="parent"))
    conn.commit()
    assert _edge_state(conn, child_id)[1] == "prefix-sharing"
    inheriting = _child_ids(conn, child_id)

    write_fixture_index_session(conn, session("parent", rewritten))
    conn.commit()
    assert _edge_state(conn, child_id)[1] == "spawned-fresh"
    materialized = _child_ids(conn, child_id)

    write_fixture_index_session(conn, session("child", child_messages, parent="parent"), force_replace=True)
    conn.commit()
    replayed = _child_ids(conn, child_id)
    # A second replay reads the scope the first one carried forward.
    write_fixture_index_session(conn, session("child", child_messages, parent="parent"), force_replace=True)
    conn.commit()
    assert _child_ids(conn, child_id) == replayed
    close_fixture_index_connection(conn)
    return inheriting, materialized, replayed


def test_a_materialized_child_keeps_its_ids_when_prefix_and_tail_share_a_native_id(tmp_path: Path) -> None:
    """polylogue-5gg3u: a stored message ID never moves across a parent rewrite and a replay.

    The tail repeats the prefix's native id ``m1``. While inheriting, the tail
    is identified over itself, so its row is ``n:m1``; the materialized copy of
    the prefix's ``m1`` takes a content ID. Anti-vacuity: replay without the
    recorded identity scope and the whole transcript treats ``m1`` as a
    duplicate, moving the tail row to a content ID.
    """
    parent = [_msg("m0", Role.USER, "hello", 0), _msg("m1", Role.ASSISTANT, "hi there", 1)]
    child = [*parent, _msg("m1", Role.USER, "child diverges here", 2)]
    rewritten = [_msg("m0", Role.USER, "hello", 0)]
    inheriting, materialized, replayed = _materialize_then_replay(tmp_path, parent, child, rewritten)

    assert inheriting == [archive_message_id("codex-session:child", "m1")]
    assert set(inheriting) <= set(materialized)
    assert replayed == materialized


def test_a_materialized_child_keeps_its_ids_for_id_less_duplicates(tmp_path: Path) -> None:
    """ID-less duplicates across prefix and tail keep their content occurrences.

    Anti-vacuity: number occurrences over the whole transcript on replay and
    the tail's ``hi`` (occurrence 0 while inheriting) becomes occurrence 1.
    """
    parent = [_msg("", Role.USER, "hi", 0), _msg("", Role.ASSISTANT, "answer", 1)]
    child = [*parent, _msg("", Role.USER, "hi", 2)]
    rewritten = [_msg("", Role.USER, "hi", 0), _msg("", Role.ASSISTANT, "another answer", 1)]
    inheriting, materialized, replayed = _materialize_then_replay(tmp_path, parent, child, rewritten)

    assert len(inheriting) == 1
    assert set(inheriting) <= set(materialized)
    assert len(materialized) == 3
    assert replayed == materialized


def test_a_materialized_child_keeps_its_ids_when_a_message_is_prepended(tmp_path: Path) -> None:
    """The copied prefix is found by content, not by its count from the start.

    Anti-vacuity: split the replay at the recorded message count and the two
    identical ID-less messages swap occurrences once a distinct message is
    prepended.
    """
    parent = [_msg("", Role.USER, "hi", 0), _msg("", Role.ASSISTANT, "answer", 1)]
    child = [*parent, _msg("", Role.USER, "hi", 2)]
    rewritten = [_msg("", Role.USER, "hi", 0), _msg("", Role.ASSISTANT, "another answer", 1)]
    _inheriting, materialized, _replayed = _materialize_then_replay(tmp_path, parent, child, rewritten)

    conn = _connect(tmp_path / "index.db")
    prepended = [
        _msg("", Role.USER, "preamble", 0),
        *(message.model_copy(update={"position": (message.position or 0) + 1}) for message in child),
    ]
    write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=prepended,
        ),
        force_replace=True,
    )
    conn.commit()
    assert set(materialized) <= set(_child_ids(conn, "codex-session:child"))
    close_fixture_index_connection(conn)


def test_a_scoped_replay_places_rows_as_materialization_did(tmp_path: Path) -> None:
    """Parsed prefix coordinates that collide with the tail are renumbered on
    replay exactly as materialization renumbered them.

    Anti-vacuity: keep the parser's positions on replay and the tail sorts
    before its copied prefix.
    """
    parent = [_msg("m0", Role.USER, "m0", 0), _msg("m1", Role.ASSISTANT, "m1", 1)]
    child = [_msg("m0", Role.USER, "m0", 5), _msg("m1", Role.ASSISTANT, "m1", 6), _msg("x", Role.USER, "tail", 0)]
    rewritten = [_msg("m0", Role.USER, "m0", 0)]
    _inheriting, materialized, replayed = _materialize_then_replay(tmp_path, parent, child, rewritten)
    assert replayed == materialized
    conn = _connect(tmp_path / "index.db")
    rows = conn.execute(
        "SELECT native_id, position FROM messages WHERE session_id = ? ORDER BY position", ("codex-session:child",)
    ).fetchall()
    assert [row[0] for row in rows] == ["m0", "m1", "x"]
    close_fixture_index_connection(conn)


def _replay_child(tmp_path: Path, messages: list[ParsedMessage]) -> list[str]:
    conn = _connect(tmp_path / "index.db")
    write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=messages,
        ),
        force_replace=True,
    )
    conn.commit()
    ids = _child_ids(conn, "codex-session:child")
    close_fixture_index_connection(conn)
    return ids


def test_a_scoped_replay_places_a_prepended_row_before_a_renumbered_prefix(tmp_path: Path) -> None:
    """Anti-vacuity: leave the prepended row out of the placement and it takes
    the renumbered prefix's position, rolling the replay back on the
    coordinate key."""
    parent = [_msg("m0", Role.USER, "m0", 0), _msg("m1", Role.ASSISTANT, "m1", 1)]
    child = [_msg("m0", Role.USER, "m0", 5), _msg("m1", Role.ASSISTANT, "m1", 6), _msg("x", Role.USER, "tail", 0)]
    _inheriting, materialized, _replayed = _materialize_then_replay(
        tmp_path, parent, child, [_msg("m0", Role.USER, "m0", 0)]
    )
    prepended = [
        _msg("p", Role.USER, "preamble", 0),
        *(m.model_copy(update={"position": (m.position or 0) + 1}) for m in child),
    ]
    assert set(materialized) <= set(_replay_child(tmp_path, prepended))


def test_a_scoped_replay_keeps_a_prefix_native_id_a_new_row_repeats(tmp_path: Path) -> None:
    """Anti-vacuity: compare natives only outside the prefix and the new row
    takes the copied prefix row's ``n:m1`` ID, rolling the replay back."""
    parent = [_msg("m0", Role.USER, "hello", 0), _msg("m1", Role.ASSISTANT, "hi there", 1)]
    child = [*parent, _msg("x", Role.USER, "child diverges here", 2)]
    _inheriting, materialized, _replayed = _materialize_then_replay(
        tmp_path, parent, child, [_msg("m0", Role.USER, "hello", 0)]
    )
    extended = [*child, _msg("m1", Role.ASSISTANT, "a later reply reusing m1", 3)]
    replayed = _replay_child(tmp_path, extended)
    assert set(materialized) <= set(replayed)
    assert len(replayed) == 4


def test_a_scoped_replay_refuses_a_prefix_that_appears_twice(tmp_path: Path) -> None:
    """Anti-vacuity: take the first matching run and a prepended copy of the
    prefix takes the stored native IDs of the materialized one."""
    from polylogue.storage.sqlite.archive_tiers.write import InheritedPrefixMaterializationError

    parent = [_msg("m0", Role.USER, "hello", 0), _msg("m1", Role.ASSISTANT, "hi there", 1)]
    child = [*parent, _msg("x", Role.USER, "child diverges here", 2)]
    _materialize_then_replay(tmp_path, parent, child, [_msg("m0", Role.USER, "hello", 0)])
    doubled = [
        *parent,
        *(message.model_copy(update={"position": (message.position or 0) + 2}) for message in child),
    ]
    with pytest.raises(InheritedPrefixMaterializationError, match="ambiguous"):
        _replay_child(tmp_path, doubled)


def test_an_append_keeps_the_materialized_identity_scope(tmp_path: Path) -> None:
    """Anti-vacuity: skip the scope on an append and the edge it rewrites loses
    it, so the next full replay slices the child against its parent again."""
    parent = [_msg("m0", Role.USER, "hello", 0), _msg("m1", Role.ASSISTANT, "hi there", 1)]
    child = [*parent, _msg("x", Role.USER, "child diverges here", 2)]
    _inheriting, materialized, _replayed = _materialize_then_replay(
        tmp_path, parent, child, [_msg("m0", Role.USER, "hello", 0)]
    )
    conn = _connect(tmp_path / "index.db")
    appended = _msg("y", Role.ASSISTANT, "appended reply", 3)
    write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=[appended],
        ),
        merge_append=True,
    )
    conn.commit()
    close_fixture_index_connection(conn)
    replayed = _replay_child(tmp_path, [*child, appended])
    assert set(materialized) <= set(replayed)


def test_a_materialized_child_keeps_its_ids_after_a_replay_drops_its_parent(tmp_path: Path) -> None:
    """The identity scope belongs to the child, not to its parent edge.

    A revision that no longer declares the parent deletes the edge. Anti-vacuity:
    keep the scope on that edge and the second parentless replay numbers the
    identical ID-less prefix and tail rows over the whole transcript, swapping
    their IDs.
    """
    parent = [_msg("", Role.USER, "hi", 0), _msg("", Role.ASSISTANT, "answer", 1)]
    child = [*parent, _msg("", Role.USER, "hi", 2)]
    rewritten = [_msg("", Role.USER, "hi", 0), _msg("", Role.ASSISTANT, "another answer", 1)]
    _inheriting, materialized, _replayed = _materialize_then_replay(tmp_path, parent, child, rewritten)

    conn = _connect(tmp_path / "index.db")
    orphaned = ParsedSession(source_name=Provider.CODEX, provider_session_id="child", title="child", messages=child)
    for _replay in range(2):
        write_fixture_index_session(conn, orphaned, force_replace=True)
        conn.commit()
        assert not conn.execute(
            "SELECT 1 FROM session_links WHERE src_session_id = ?", ("codex-session:child",)
        ).fetchall()
        assert _child_ids(conn, "codex-session:child") == materialized
    close_fixture_index_connection(conn)


def test_a_lost_row_before_a_surviving_branch_point_materializes_the_prefix(tmp_path: Path) -> None:
    """A surviving branch point does not prove the rows before it survived.

    Anti-vacuity: treat any composed inherited row as an intact prefix and the
    parent rewritten without ``m0`` shortens the child to ``[m1, x2]``.
    """
    conn = _connect(tmp_path / "index.db")
    write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1"]))
    child_id = write_fixture_index_session(conn, _codex_session("child", ["m0", "m1", "x2"], parent="parent"))
    conn.commit()
    assert _edge_state(conn, child_id)[1] == "prefix-sharing"

    write_fixture_index_session(conn, _codex_session("parent", ["m1"]))
    conn.commit()
    assert _composed_texts(conn, child_id) == ["m0", "m1", "x2"]
    assert _edge_state(conn, child_id)[1] == "spawned-fresh"
    close_fixture_index_connection(conn)


def test_a_scoped_replay_refuses_a_prefix_that_no_longer_matches(tmp_path: Path) -> None:
    """Anti-vacuity: fall back to the leading rows and a replay without the
    first copied row makes ``[m1, tail]`` the prefix, swapping the two
    identical ID-less tail rows' stored occurrences."""
    from polylogue.storage.sqlite.archive_tiers.write import InheritedPrefixMaterializationError

    parent = [_msg("m0", Role.USER, "hello", 0), _msg("m1", Role.ASSISTANT, "hi there", 1)]
    tail = [_msg("", Role.USER, "same", 2), _msg("", Role.USER, "same", 3)]
    _materialize_then_replay(tmp_path, parent, [*parent, *tail], [_msg("m0", Role.USER, "hello", 0)])
    with pytest.raises(InheritedPrefixMaterializationError, match="no longer appears"):
        _replay_child(tmp_path, [parent[1], *tail])


def test_a_deep_chain_composes_in_linear_time(tmp_path: Path) -> None:
    """One list is cut and extended down the chain; no level's transcript is kept.

    Covers the writer's signatures, the envelope planner and compact
    accounting of every level. Anti-vacuity: rebuild ``prefix + own`` at every
    level and cache each, and the composition holds every intermediate
    transcript, quadratic in depth.
    """
    from polylogue.storage.sqlite.archive_tiers import write as write_module

    conn = _connect(tmp_path / "index.db")
    depth = 60
    write_fixture_index_session(conn, _codex_session("s0", ["r0"]))
    for level in range(1, depth):
        texts = [f"r{index}" for index in range(level + 1)]
        write_fixture_index_session(conn, _codex_session(f"s{level}", texts, parent=f"s{level - 1}"))
    conn.commit()
    leaf = f"codex-session:s{depth - 1}"
    intermediates: dict[str, list[tuple[str, str]]] = {}
    composed = write_module._composed_db_signatures(conn, leaf, composed_cache=intermediates)
    assert len(composed) == depth
    assert set(intermediates) == {"codex-session:s0", leaf}
    assert _composed_texts(conn, leaf) == [f"r{index}" for index in range(depth)]

    from polylogue.storage.derived.lineage.compact import _CompositionShape

    shape = _CompositionShape(conn)
    for level in range(depth):
        accounting = shape.accounting(f"codex-session:s{level}")
        assert (accounting.unique, accounting.inherited) == (1, level)
    # Every level shares its parent's segments: one new node per level.
    distinct: set[int] = set()
    for node in shape._segments.values():
        while node is not None and id(node) not in distinct:
            distinct.add(id(node))
            node = node.prev
    assert len(distinct) <= 2 * depth
    close_fixture_index_connection(conn)


def test_a_scoped_replay_keeps_an_appended_duplicate_of_a_copy(tmp_path: Path) -> None:
    """A row appended after materialization keeps the occurrence the append gave it.

    The ID-less prefix ``[hi, answer]`` is copied into the child as content
    IDs; an appended ``hi`` then takes the occurrence after the copy.
    Anti-vacuity: count every non-prefix row before the copies on replay and
    the copy and the appended row exchange IDs.
    """
    parent = [_msg("", Role.USER, "hi", 0), _msg("", Role.ASSISTANT, "answer", 1)]
    child = [*parent, _msg("", Role.USER, "tail", 2)]
    rewritten = [_msg("", Role.USER, "hi", 0), _msg("", Role.ASSISTANT, "another answer", 1)]
    _inheriting, materialized, _replayed = _materialize_then_replay(tmp_path, parent, child, rewritten)
    conn = _connect(tmp_path / "index.db")
    appended = _msg("", Role.USER, "hi", 3)
    write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=[appended],
        ),
        merge_append=True,
    )
    conn.commit()
    stored = _child_ids(conn, "codex-session:child")
    close_fixture_index_connection(conn)
    assert len(stored) == 4 and set(materialized) <= set(stored)
    assert _replay_child(tmp_path, [*child, appended]) == stored


@pytest.mark.timeout(0)
def test_the_writer_admits_a_chain_deeper_than_any_walk_budget(tmp_path: Path) -> None:
    """Each level is written through the production writer with its parent claim.

    Anti-vacuity: bound the admission cycle walk (``_would_create_cycle``) by a
    step budget below ``_VERY_DEEP_CHAIN_LEVELS`` and the deeper edges are
    quarantined as indeterminate instead of resolved.
    """
    conn = _connect(tmp_path / "index.db")
    for level in range(_VERY_DEEP_CHAIN_LEVELS):
        write_fixture_index_session(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=f"deep-{level:04d}",
                title=f"deep-{level:04d}",
                parent_session_provider_id=f"deep-{level - 1:04d}" if level else None,
                branch_type=BranchType.FORK if level else None,
                messages=[_msg("m", Role.USER, f"level {level}", 0)],
            ),
        )
    conn.commit()
    quarantined = conn.execute("SELECT COUNT(*) FROM session_links WHERE status = 'quarantined'").fetchone()[0]
    resolved = conn.execute("SELECT COUNT(*) FROM session_links WHERE resolved_dst_session_id IS NOT NULL").fetchone()[
        0
    ]
    assert (quarantined, resolved) == (0, _VERY_DEEP_CHAIN_LEVELS - 1)
    close_fixture_index_connection(conn)


def test_a_scoped_replay_finds_an_edited_native_prefix_row(tmp_path: Path) -> None:
    """A copied row stored under its native ID is found by that ID.

    Anti-vacuity: match the copied prefix by content alone and editing the
    text of ``m1`` refuses the replay as a prefix that no longer appears.
    """
    parent = [_msg("m0", Role.USER, "hello", 0), _msg("m1", Role.ASSISTANT, "hi there", 1)]
    child = [*parent, _msg("x", Role.USER, "child diverges here", 2)]
    _inheriting, materialized, _replayed = _materialize_then_replay(
        tmp_path, parent, child, [_msg("m0", Role.USER, "hello", 0)]
    )
    edited = [child[0], _msg("m1", Role.ASSISTANT, "hi there, edited", 1), child[2]]
    assert _replay_child(tmp_path, edited) == materialized


def test_a_scoped_replay_refuses_a_duplicate_inserted_inside_the_tail(tmp_path: Path) -> None:
    """Anti-vacuity: number the rows after the prefix in their new order and
    the inserted ``hi`` takes the old tail row's ``hi.0``, moving it to ``hi.2``."""
    from polylogue.storage.sqlite.archive_tiers.write import InheritedPrefixMaterializationError

    parent = [_msg("", Role.USER, "hi", 0), _msg("", Role.ASSISTANT, "answer", 1)]
    child = [*parent, _msg("", Role.USER, "hi", 2), _msg("", Role.USER, "tail", 3)]
    rewritten = [_msg("", Role.USER, "hi", 0), _msg("", Role.ASSISTANT, "another answer", 1)]
    _materialize_then_replay(tmp_path, parent, child, rewritten)
    inserted = [*parent, _msg("", Role.USER, "hi", 2), _msg("", Role.USER, "hi", 3), _msg("", Role.USER, "tail", 4)]
    with pytest.raises(InheritedPrefixMaterializationError, match="inside the tail"):
        _replay_child(tmp_path, inserted)


def _child_hashes(conn: sqlite3.Connection, child_id: str) -> dict[str, bytes]:
    return {
        str(row[0]): bytes(row[1])
        for row in conn.execute("SELECT message_id, content_hash FROM messages WHERE session_id = ?", (child_id,))
    }


def test_materialized_rows_carry_the_hash_a_replay_computes(tmp_path: Path) -> None:
    """Copied prefix rows and shifted tail rows are rehashed from stored columns.

    Parsed prefix coordinates collide with the tail, so materialization copies
    the prefix and shifts the tail. Anti-vacuity: give copies a synthetic hash
    or leave the shifted tail's hash on its former position, and the next
    unchanged replay rewrites those hashes.
    """
    parent = [_msg("m0", Role.USER, "m0", 0), _msg("m1", Role.ASSISTANT, "m1", 1)]
    child = [_msg("m0", Role.USER, "m0", 5), _msg("m1", Role.ASSISTANT, "m1", 6), _msg("x", Role.USER, "tail", 0)]
    _materialize_then_replay(tmp_path, parent, child, [_msg("m0", Role.USER, "m0", 0)])
    conn = _connect(tmp_path / "index.db")
    # ``_materialize_then_replay`` already replayed twice; one more replay of
    # the materialized child must not move any stored hash.
    before = _child_hashes(conn, "codex-session:child")
    close_fixture_index_connection(conn)
    _replay_child(tmp_path, child)
    conn = _connect(tmp_path / "index.db")
    assert _child_hashes(conn, "codex-session:child") == before
    close_fixture_index_connection(conn)


def test_materialization_hashes_match_the_first_replay(tmp_path: Path) -> None:
    """Anti-vacuity: keep the synthetic copy hash and the first replay after
    materialization changes every copied row's ``content_hash``."""
    conn = _connect(tmp_path / "index.db")
    parent = [_msg("m0", Role.USER, "m0", 0), _msg("m1", Role.ASSISTANT, "m1", 1)]
    child = [_msg("m0", Role.USER, "m0", 5), _msg("m1", Role.ASSISTANT, "m1", 6), _msg("x", Role.USER, "tail", 0)]

    def session(name: str, messages: list[ParsedMessage], parent_name: str | None = None) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=name,
            title=name,
            parent_session_provider_id=parent_name,
            branch_type=BranchType.FORK if parent_name else None,
            messages=messages,
        )

    write_fixture_index_session(conn, session("parent", parent))
    child_id = write_fixture_index_session(conn, session("child", child, "parent"))
    write_fixture_index_session(conn, session("parent", [_msg("m0", Role.USER, "m0", 0)]))
    conn.commit()
    materialized = _child_hashes(conn, child_id)
    write_fixture_index_session(conn, session("child", child, "parent"), force_replace=True)
    conn.commit()
    assert _child_hashes(conn, child_id) == materialized
    close_fixture_index_connection(conn)


def test_divergent_answered_tool_use_keeps_child_pairing_after_parent_replacement(tmp_path: Path) -> None:
    """The answered child call differs from the parent's unanswered call.

    Their shared prefix ends at m0. Replacing the parent keeps the child-owned
    answered call and its result together, with unchanged canonical hashes.
    """
    conn = _connect(tmp_path / "index.db")
    call = ParsedMessage(
        provider_message_id="m1",
        role=Role.ASSISTANT,
        text="",
        position=1,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name="Bash",
                tool_id="t1",
                tool_input={"cmd": "ls"},
                tool_outcome=ToolOutcome.NO_RESULT,
            )
        ],
    )
    answered = call.model_copy(update={"blocks": [call.blocks[0].model_copy(update={"tool_outcome": ToolOutcome.OK})]})
    result = ParsedMessage(
        provider_message_id="r2",
        role=Role.TOOL,
        text="",
        position=2,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT, tool_id="t1", text="ok", is_error=False, tool_outcome=ToolOutcome.OK
            )
        ],
    )
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[_msg("m0", Role.USER, "go", 0), call],
    )
    write_fixture_index_session(conn, parent)
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        title="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[_msg("m0", Role.USER, "go", 0), answered, result],
    )
    child_id = write_fixture_index_session(conn, child)
    expected_edge = ("codex-session:parent", "prefix-sharing", "codex-session:parent:n:m0")
    assert _edge_state(conn, child_id) == expected_edge
    assert tuple(
        conn.execute(
            "SELECT tool_outcome FROM blocks WHERE session_id=? AND block_type='tool_use'", (child_id,)
        ).fetchone()
    ) == (ToolOutcome.OK.value,)
    write_fixture_index_session(conn, parent.model_copy(update={"messages": [_msg("m0", Role.USER, "go", 0)]}))
    conn.commit()
    assert _edge_state(conn, child_id) == expected_edge
    outcome = conn.execute(
        "SELECT tool_outcome FROM blocks WHERE session_id = ? AND block_type = 'tool_use'", (child_id,)
    ).fetchone()
    assert tuple(outcome) == (ToolOutcome.OK.value,)
    materialized = _child_hashes(conn, child_id)
    write_fixture_index_session(conn, child, force_replace=True)
    conn.commit()
    assert _child_hashes(conn, child_id) == materialized
    close_fixture_index_connection(conn)


def test_the_tail_stays_connected_to_the_copied_prefix(tmp_path: Path) -> None:
    """The child's first tail message names the last inherited one as parent.

    Anti-vacuity: resolve parents only among the written tail and the row's
    ``parent_message_id`` stays NULL; remap only copied rows and it keeps
    naming the parent's deleted row.
    """
    conn = _connect(tmp_path / "index.db")
    write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1"]))
    tail = _msg("x2", Role.USER, "tail", 2).model_copy(update={"parent_message_provider_id": "m1"})
    child = _codex_session("child", ["m0", "m1"], parent="parent")
    child_id = write_fixture_index_session(conn, child.model_copy(update={"messages": [*child.messages, tail]}))
    conn.commit()
    parent_of = "SELECT parent_message_id FROM messages WHERE message_id = ?"
    assert tuple(conn.execute(parent_of, (f"{child_id}:n:x2",)).fetchone()) == ("codex-session:parent:n:m1",)

    write_fixture_index_session(conn, _codex_session("parent", ["m0"]))
    conn.commit()
    assert _edge_state(conn, child_id)[1] == "spawned-fresh"
    assert tuple(conn.execute(parent_of, (f"{child_id}:n:x2",)).fetchone()) == (f"{child_id}:n:m1",)
    close_fixture_index_connection(conn)


def test_a_compaction_boundary_on_the_prefix_follows_its_copied_rows(tmp_path: Path) -> None:
    """Inherited positions 5/6 are renumbered to 0/1 under a tail at 0.

    Anti-vacuity: classify endpoints by the tail's minimum alone and the
    boundary on inherited position 5 is shifted past the prefix instead of
    following its row to 0.
    """
    conn = _connect(tmp_path / "index.db")
    write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="parent",
            title="parent",
            messages=[_msg("m0", Role.USER, "m0", 5), _msg("m1", Role.ASSISTANT, "m1", 6)],
        ),
    )
    child_id = write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=[
                _msg("m0", Role.USER, "m0", 5),
                _msg("m1", Role.ASSISTANT, "m1", 6),
                _msg("x", Role.USER, "t", 0),
            ],
            session_events=[
                ParsedSessionEvent(
                    event_type="compaction", payload={}, boundary_start_position=5, boundary_end_position=6
                )
            ],
        ),
    )
    conn.commit()
    write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="parent",
            title="parent",
            messages=[_msg("m0", Role.USER, "m0", 5)],
        ),
    )
    conn.commit()
    assert _edge_state(conn, child_id)[1] == "spawned-fresh"
    rows = {
        str(row[0]): int(row[1])
        for row in conn.execute("SELECT native_id, position FROM messages WHERE session_id = ?", (child_id,))
    }
    boundary = conn.execute(
        "SELECT boundary_start_position, boundary_end_position FROM session_events WHERE session_id = ?", (child_id,)
    ).fetchone()
    assert tuple(boundary) == (rows["m0"], rows["m1"])
    close_fixture_index_connection(conn)


def test_a_boundary_message_id_follows_the_copied_row(tmp_path: Path) -> None:
    """Anti-vacuity: capture only ``source_message_id`` and the plain-text
    ``boundary_message_id`` keeps naming the parent's deleted row."""
    conn = _connect(tmp_path / "index.db")
    write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1"]))
    child_id = write_fixture_index_session(
        conn,
        _codex_session("child", ["m0", "m1", "x2"], parent="parent").model_copy(
            update={"session_events": [ParsedSessionEvent(event_type="compaction", payload={})]}
        ),
    )
    # The summary the compaction names is an inherited row.
    conn.execute(
        "UPDATE session_events SET boundary_message_id = ? WHERE session_id = ?",
        ("codex-session:parent:n:m1", child_id),
    )
    conn.commit()
    write_fixture_index_session(conn, _codex_session("parent", ["m0"]))
    conn.commit()
    boundary = conn.execute(
        "SELECT boundary_message_id FROM session_events WHERE session_id = ?", (child_id,)
    ).fetchone()
    assert tuple(boundary) == (f"{child_id}:n:m1",)
    close_fixture_index_connection(conn)


def test_an_uncaptured_descendant_reference_to_a_deleted_row_follows_the_copy(tmp_path: Path) -> None:
    """``G -> P -> B -> D``: D branches at a B-owned row, so the guard never
    captures it, but its event names a P-owned row the rewrite deletes.

    Anti-vacuity: remap descendants only by their surviving old ids and the
    foreign key's NULL leaves D's event naming nothing.
    """
    conn = _connect(tmp_path / "index.db")
    base = ["m0", "m1", "m2"]
    write_fixture_index_session(conn, _codex_session("g", base))
    write_fixture_index_session(conn, _codex_session("p", [*base, "p3"], parent="g"))
    # D before B: D's event then resolves onto the composed row when B arrives.
    d_id = write_fixture_index_session(
        conn,
        _codex_session("d", [*base, "p3", "b4", "d5"], parent="b").model_copy(
            update={
                "session_events": [
                    ParsedSessionEvent(event_type="capture_gap", source_message_provider_id="p3", payload={})
                ]
            }
        ),
    )
    b_id = write_fixture_index_session(conn, _codex_session("b", [*base, "p3", "b4"], parent="p"))
    conn.commit()
    events = "SELECT source_message_id FROM session_events WHERE session_id = ?"
    assert [tuple(row) for row in conn.execute(events, (d_id,))] == [("codex-session:p:n:p3",)]

    write_fixture_index_session(conn, _codex_session("p", [*base, "q3"], parent="g"))
    conn.commit()
    assert _edge_state(conn, b_id)[1] == "spawned-fresh"
    assert [tuple(row) for row in conn.execute(events, (d_id,))] == [(f"{b_id}:n:p3",)]
    close_fixture_index_connection(conn)


def test_a_reused_block_id_is_not_taken_for_the_dispatched_call(tmp_path: Path) -> None:
    """The rewrite keeps ``m1`` at the same position but calls a different tool.

    Anti-vacuity: accept any block under the old generated id as the
    survivor and the subagent edge points at the unrelated new call, outside
    the child's materialized transcript.
    """
    conn = _connect(tmp_path / "index.db")

    def call(tool_id: str) -> ParsedMessage:
        return ParsedMessage(
            provider_message_id="m1",
            role=Role.ASSISTANT,
            text="",
            position=1,
            blocks=[
                ParsedContentBlock(
                    type=BlockType.TOOL_USE, tool_name="Task", tool_id=tool_id, tool_input={"t": tool_id}
                )
            ],
        )

    later = _msg("m2", Role.USER, "later", 2)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[_msg("m0", Role.USER, "go", 0), call("task-1"), later],
    )
    parent_id = write_fixture_index_session(conn, parent)
    child_id = write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="child",
            title="child",
            parent_session_provider_id="parent",
            branch_type=BranchType.FORK,
            messages=[_msg("m0", Role.USER, "go", 0), call("task-1"), later, _msg("x3", Role.USER, "tail", 3)],
        ),
    )
    worker_id = write_fixture_index_session(conn, _codex_session("worker", ["w0"]))
    conn.execute(
        """
        INSERT INTO session_links(
            src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id,
            inheritance, status, parent_tool_use_block_id, confidence, evidence_json, observed_at_ms
        ) VALUES (?, 'codex-session', 'child', 'subagent', ?, 'spawned-fresh', NULL, ?, 1.0, '[]', 0)
        """,
        (worker_id, child_id, f"{parent_id}:n:m1:0"),
    )
    conn.commit()
    write_fixture_index_session(
        conn, parent.model_copy(update={"messages": [_msg("m0", Role.USER, "go", 0), call("task-2")]})
    )
    conn.commit()
    assert _edge_state(conn, child_id)[1] == "spawned-fresh"
    pointer = conn.execute(
        "SELECT parent_tool_use_block_id FROM session_links WHERE src_session_id = ?", (worker_id,)
    ).fetchone()
    assert tuple(pointer) == (f"{child_id}:n:m1:0",)
    close_fixture_index_connection(conn)


def test_a_dispatch_pointer_follows_the_copy_its_dispatcher_composes(tmp_path: Path) -> None:
    """``parent -> b -> d``: the call is copied into ``b`` while ``d``, the
    subagent's resolved dispatcher, keeps inheriting through ``b``.

    The parent's rewrite keeps the call itself, so its old block still exists.
    Anti-vacuity: prefer that surviving block over the copy ``d`` now composes
    and the pointer names a block outside ``d``'s transcript.
    """
    conn = _connect(tmp_path / "index.db")
    parent_id = write_fixture_index_session(conn, _dispatch_session("parent", None, ["p2", "p3"]))
    b_id = write_fixture_index_session(conn, _dispatch_session("b", "parent", ["p2", "b3"]))
    d_id = write_fixture_index_session(conn, _dispatch_session("d", "b", ["p2", "b3", "d4"]))
    worker_id = write_fixture_index_session(conn, _codex_session("worker", ["w0"]))
    conn.execute(
        """
        INSERT INTO session_links(
            src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id,
            inheritance, status, parent_tool_use_block_id, confidence, evidence_json, observed_at_ms
        ) VALUES (?, 'codex-session', 'd', 'subagent', ?, 'spawned-fresh', NULL, ?, 1.0, '[]', 0)
        """,
        (worker_id, d_id, f"{parent_id}:n:m1:0"),
    )
    conn.commit()
    write_fixture_index_session(conn, _dispatch_session("parent", None, ["p3"]))
    conn.commit()
    assert _edge_state(conn, b_id)[1] == "spawned-fresh"
    assert _edge_state(conn, d_id)[1] == "prefix-sharing"
    pointer = conn.execute(
        "SELECT parent_tool_use_block_id FROM session_links WHERE src_session_id = ?", (worker_id,)
    ).fetchone()
    assert tuple(pointer) == (f"{b_id}:n:m1:0",)
    close_fixture_index_connection(conn)


def test_a_guard_follows_the_edge_readers_compose(tmp_path: Path) -> None:
    """A child with two resolved prefix-sharing edges composes through the one
    ``link_type, dst_origin, dst_native_id`` orders first.

    Anti-vacuity: keep whichever edge SQLite returns last and settlement acts
    on the other parent's edge, leaving the composing one dangling.
    """
    conn = _connect(tmp_path / "index.db")
    write_fixture_index_session(conn, _codex_session("a-parent", ["m0", "m1"]))
    write_fixture_index_session(conn, _codex_session("z-parent", ["m0", "m1"]))
    child_id = write_fixture_index_session(conn, _codex_session("child", ["m0", "m1", "x2"], parent="a-parent"))
    conn.execute(
        """
        INSERT INTO session_links(
            src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id,
            branch_point_message_id, inheritance, status, confidence, evidence_json, observed_at_ms
        ) VALUES (?, 'codex-session', 'z-parent', 'fork', 'codex-session:z-parent',
                  'codex-session:z-parent:n:m1', 'prefix-sharing', NULL, 1.0, '[]', 0)
        """,
        (child_id,),
    )
    conn.commit()
    write_fixture_index_session(conn, _codex_session("a-parent", ["m0"]))
    conn.commit()
    envelope = read_archive_session_envelope(conn, child_id)
    assert envelope.lineage_complete is True
    assert [message.blocks[0].text for message in envelope.messages] == ["m0", "m1", "x2"]
    close_fixture_index_connection(conn)


def test_a_scoped_replay_keeps_per_segment_prefix_positions(tmp_path: Path) -> None:
    """Two ancestors each contribute a prefix row at ``(0, 0)``.

    Materialization numbers them by source session; the child's own parse
    gives both position 0. Anti-vacuity: map replay positions by position
    alone and both rows land on 0, refusing every unchanged replay.
    """
    conn = _connect(tmp_path / "index.db")
    write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX, provider_session_id="g", title="g", messages=[_msg("g0", Role.USER, "g0", 0)]
        ),
    )
    p_messages = [_msg("g0", Role.USER, "g0", 5), _msg("p1", Role.ASSISTANT, "p1", 0)]
    write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="p",
            title="p",
            parent_session_provider_id="g",
            branch_type=BranchType.FORK,
            messages=p_messages,
        ),
    )
    child_messages = [
        _msg("g0", Role.USER, "g0", 0),
        _msg("p1", Role.ASSISTANT, "p1", 0),
        _msg("c2", Role.USER, "c2", 1),
    ]
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="c",
        title="c",
        parent_session_provider_id="p",
        branch_type=BranchType.FORK,
        messages=child_messages,
    )
    child_id = write_fixture_index_session(conn, child)
    conn.commit()
    write_fixture_index_session(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="p",
            title="p",
            parent_session_provider_id="g",
            branch_type=BranchType.FORK,
            messages=[_msg("g0", Role.USER, "g0", 5)],
        ),
    )
    conn.commit()
    assert _edge_state(conn, child_id)[1] == "spawned-fresh"
    placed = "SELECT message_id, position, variant_index FROM messages WHERE session_id = ? ORDER BY message_id"
    stored = conn.execute(placed, (child_id,)).fetchall()
    write_fixture_index_session(conn, child, force_replace=True)
    conn.commit()
    assert conn.execute(placed, (child_id,)).fetchall() == stored
    close_fixture_index_connection(conn)


def test_settlement_streams_inherited_prefixes_through_the_guard_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent rewrite that materializes several children holds no per-row
    Python copy of their prefixes: the guard and the plan are TEMP tables.

    Anti-vacuity: build the guard or the plan from ``_composed_db_signatures``
    lists and the patched list builder fails the parent's write.
    """
    from polylogue.storage.sqlite.archive_tiers import write as write_module

    conn = _connect(tmp_path / "index.db")
    base = [f"m{index}" for index in range(40)]
    write_fixture_index_session(conn, _codex_session("parent", base))
    children = [
        write_fixture_index_session(conn, _codex_session(f"child{n}", [*base, f"x{n}"], parent="parent"))
        for n in range(3)
    ]
    conn.commit()

    def no_lists(*_args: object, **_kwargs: object) -> list[tuple[str, str]]:
        raise AssertionError("settlement composed a transcript into a Python list")

    def list_free(function: Any) -> Any:
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with monkeypatch.context() as scoped:
                scoped.setattr(write_module, "_composed_db_signatures", no_lists)
                return function(*args, **kwargs)

        return guarded

    with monkeypatch.context() as patched:
        for name in ("_capture_inherited_prefixes", "_settle_inherited_prefixes"):
            patched.setattr(write_module, name, list_free(getattr(write_module, name)))
        write_fixture_index_session(conn, _codex_session("parent", base[:10]))
    conn.commit()
    for n, child_id in enumerate(children):
        assert _edge_state(conn, child_id)[1] == "spawned-fresh"
        assert _composed_texts(conn, child_id) == [*base, f"x{n}"]
    close_fixture_index_connection(conn)


# --- Attachments owned by an inherited prefix message -----------------------
#
# A message signature is its role and blocks; attachments ride beside it. A
# prefix-sharing child's replay can therefore carry an attachment on a message
# its parent physically owns. The parent row is that attachment's owner when it
# references it; when it does not, the message is not shared and stays in the
# child's tail with the child's own reference. Both write orders converge.


def _prefix_attachment(message_provider_id: str, **extra: Any) -> ParsedAttachment:
    return ParsedAttachment(
        provider_attachment_id="shared-file",
        message_provider_id=message_provider_id,
        name="shared.txt",
        mime_type="text/plain",
        path="shared.txt",
        direction="user_input",
        **extra,
    )


def _composed_attachments(conn: sqlite3.Connection, session_id: str) -> list[tuple[str | None, str, str | None]]:
    """``(owning session, message text, attachment name)`` across the composed transcript."""
    return [
        (message.source_session_id, str(message.blocks[0].text), attachment.display_name)
        for message in read_archive_session_envelope(conn, session_id).messages
        for attachment in message.attachments
    ]


def test_inherited_attachment_the_parent_references_is_owned_by_the_parent_row(tmp_path: Path) -> None:
    """The child's replayed copy adds no reference, no orphan and no false diagnosis.

    Its inline bytes still complete the shared attachment row the parent
    recorded without them. Anti-vacuity: stop resolving tail-unowned
    attachments against ``inherited_prefix_message_ids`` and the write reports
    ``provider_never_linked`` and leaves the shared row unfetched.
    """
    conn = _connect(tmp_path / "index.db")
    parent_id = write_fixture_index_session(
        conn,
        _codex_session("parent", ["m0", "m1"]).model_copy(update={"attachments": [_prefix_attachment("m1")]}),
    )
    payload = b"bytes only the child's replay carried"
    child_attachment = _prefix_attachment("m1", inline_bytes=payload, size_bytes=None)
    child = _codex_session("child", ["m0", "m1", "x2"], parent="parent").model_copy(
        update={"attachments": [child_attachment]}
    )
    digest = hashlib.sha256(payload).digest()
    outcomes: list[Any] = []
    child_id = write_fixture_index_session(
        conn,
        child,
        preacquired_attachment_blobs={child_attachment.acquisition_key: (digest, len(payload), "acquired")},
        write_outcome=outcomes,
    )
    conn.commit()

    assert outcomes[-1].unresolved_attachment_owners == ()
    assert _edge_state(conn, child_id) == (parent_id, "prefix-sharing", f"{parent_id}:n:m1")
    assert _composed_attachments(conn, child_id) == [(parent_id, "m1", "shared.txt")]
    assert conn.execute("SELECT COUNT(*) FROM attachment_refs WHERE session_id = ?", (child_id,)).fetchone()[0] == 0
    [(ref_count, acquisition_status, blob_hash)] = [
        tuple(row) for row in conn.execute("SELECT ref_count, acquisition_status, blob_hash FROM attachments")
    ]
    assert (ref_count, acquisition_status, bytes(blob_hash)) == (1, "acquired", digest)
    close_fixture_index_connection(conn)


def test_inherited_message_whose_attachment_the_parent_lacks_stays_in_the_child_tail(tmp_path: Path) -> None:
    """The recomposed child shows the attachment its own replay carried.

    Anti-vacuity: drop ``_attachment_shared_prefix_limit`` from
    ``_extract_prefix_tail`` and ``m1`` is inherited from the parent row, the
    attachment has no owner and the composed child shows none.
    """
    conn = _connect(tmp_path / "index.db")
    parent_id = write_fixture_index_session(conn, _codex_session("parent", ["m0", "m1"]))
    child = _codex_session("child", ["m0", "m1", "x2"], parent="parent").model_copy(
        update={"attachments": [_prefix_attachment("m1")]}
    )
    outcomes: list[Any] = []
    child_id = write_fixture_index_session(conn, child, write_outcome=outcomes)
    conn.commit()

    assert outcomes[-1].unresolved_attachment_owners == ()
    assert _edge_state(conn, child_id) == (parent_id, "prefix-sharing", f"{parent_id}:n:m0")
    assert _composed_texts(conn, child_id) == ["m0", "m1", "x2"]
    assert _composed_attachments(conn, child_id) == [(child_id, "m1", "shared.txt")]
    assert conn.execute("SELECT COUNT(*) FROM attachments WHERE ref_count <= 0").fetchone()[0] == 0
    close_fixture_index_connection(conn)


def test_late_parent_keeps_the_childs_attachment_bearing_message(tmp_path: Path) -> None:
    """Child-first converges to the parent-first shape instead of deleting evidence.

    The child owns its whole transcript until the parent arrives; extracting
    the shared prefix then deletes the child's rows, and with them any
    attachment reference the parent's copy does not carry. Anti-vacuity: drop
    ``_stored_attachment_shared_prefix_limit`` from late-parent resolution and
    ``m1``'s row -- and the attachment's only reference -- is deleted.
    """
    parent_root = tmp_path / "parent-first"
    parent_root.mkdir()
    parent_first = _connect(parent_root / "index.db")
    write_fixture_index_session(parent_first, _codex_session("parent", ["m0", "m1"]))
    child = _codex_session("child", ["m0", "m1", "x2"], parent="parent").model_copy(
        update={"attachments": [_prefix_attachment("m1")]}
    )
    expected_id = write_fixture_index_session(parent_first, child)
    parent_first.commit()

    child_root = tmp_path / "child-first"
    child_root.mkdir()
    child_first = _connect(child_root / "index.db")
    child_id = write_fixture_index_session(child_first, child)
    parent_id = write_fixture_index_session(child_first, _codex_session("parent", ["m0", "m1"]))
    child_first.commit()

    assert child_id == expected_id
    assert _edge_state(child_first, child_id) == (parent_id, "prefix-sharing", f"{parent_id}:n:m0")
    assert _edge_state(child_first, child_id) == _edge_state(parent_first, expected_id)
    assert _composed_texts(child_first, child_id) == ["m0", "m1", "x2"]
    assert _composed_attachments(child_first, child_id) == _composed_attachments(parent_first, expected_id)
    assert _composed_attachments(child_first, child_id) == [(child_id, "m1", "shared.txt")]
    close_fixture_index_connection(parent_first)
    close_fixture_index_connection(child_first)


def test_prepared_child_write_refuses_once_the_parent_drops_an_inherited_attachment(tmp_path: Path) -> None:
    """A prefix boundary chosen against parent references is revalidated at commit.

    The child was prepared while the parent row referenced the attachment, so
    ``m1`` was to be inherited. The parent is replaced without it before the
    commit; inheriting ``m1`` now would leave the attachment unowned, so the
    prepared write is refused as retryable and a re-preparation keeps ``m1``.
    Anti-vacuity: drop the attachment revalidation from the prepared branch and
    the commit succeeds with the attachment reported
    ``inherited_owner_unreferenced`` and absent from the composed child.
    """
    conn = _connect(tmp_path / "index.db")
    parent = _codex_session("parent", ["m0", "m1"])
    write_fixture_index_session(conn, parent.model_copy(update={"attachments": [_prefix_attachment("m1")]}))
    child = _codex_session("child", ["m0", "m1", "x2"], parent="parent").model_copy(
        update={"attachments": [_prefix_attachment("m1")]}
    )
    prepared = _write_module.prepare_session_write(conn, child, merge_append=False)
    write_fixture_index_session(conn, parent)
    conn.commit()

    try:
        with (
            pytest.raises(_write_module.PreparedSessionWriteRefusedError, match="attachments changed"),
            fixture_index_mutation_scope(conn),
        ):
            write_fixture_index_session(
                conn, child, content_hash=str(session_content_hash(child)), prepared_write=prepared
            )
    finally:
        prepared.close()

    child_id = write_fixture_index_session(conn, child)
    conn.commit()
    assert _composed_attachments(conn, child_id) == [(child_id, "m1", "shared.txt")]
    close_fixture_index_connection(conn)


def test_late_prefix_attachment_boundary_keeps_only_one_sql_batch(tmp_path: Path) -> None:
    from collections.abc import Sequence
    from contextlib import closing
    from typing import overload

    from polylogue.storage.sqlite.archive_tiers.write import _stored_attachment_shared_prefix_limit

    class LazyPrefix(Sequence[tuple[str, str]]):
        def __init__(self, prefix: str) -> None:
            self.prefix = prefix
            self.reads = 0

        def __len__(self) -> int:
            return 100_000

        @overload
        def __getitem__(self, index: int) -> tuple[str, str]: ...

        @overload
        def __getitem__(self, index: slice) -> list[tuple[str, str]]: ...

        def __getitem__(self, index: int | slice) -> tuple[str, str] | list[tuple[str, str]]:
            if isinstance(index, slice):
                raise AssertionError("must not materialize a prefix slice")
            if not 0 <= index < len(self):
                raise IndexError(index)
            self.reads += 1
            return f"{self.prefix}-{index}", "signature"

    child, parent = LazyPrefix("child"), LazyPrefix("parent")
    with closing(sqlite3.connect(tmp_path / "prefix.db")) as conn:
        conn.execute(
            "CREATE TABLE attachment_refs (message_id TEXT, attachment_id TEXT, PRIMARY KEY(message_id, attachment_id))"
        )
        conn.execute("INSERT INTO attachment_refs VALUES ('child-0', 'unique-file')")
        assert _stored_attachment_shared_prefix_limit(conn, child, parent, len(child)) == 0
    assert child.reads <= 500
    assert parent.reads == 1


@pytest.mark.parametrize("prepared", [False, True], ids=["direct", "prepared"])
@pytest.mark.parametrize("grandchild", [False, True], ids=["child", "nested"])
def test_parent_replacement_preserves_already_inherited_attachments(
    tmp_path: Path, prepared: bool, grandchild: bool
) -> None:
    from contextlib import ExitStack

    with ExitStack() as cleanup:
        conn = _connect(tmp_path / "index.db")
        cleanup.callback(close_fixture_index_connection, conn)
        parent = _codex_session("parent", ["m0", "m1"])
        parent_id = write_fixture_index_session(
            conn, parent.model_copy(update={"attachments": [_prefix_attachment("m1")]})
        )
        child = _codex_session("child", ["m0", "m1", "x2"], parent="parent").model_copy(
            update={"attachments": [_prefix_attachment("m1")]}
        )
        child_id = write_fixture_index_session(conn, child)
        ids = [child_id]
        if grandchild:
            grand = _codex_session("grand", ["m0", "m1", "y2"], parent="child").model_copy(
                update={"attachments": [_prefix_attachment("m1")]}
            )
            ids.append(write_fixture_index_session(conn, grand))
        assert _composed_attachments(conn, child_id) == [(parent_id, "m1", "shared.txt")]
        if prepared:
            carrier = _write_module.prepare_session_write(conn, parent, merge_append=False)
            try:
                with fixture_index_mutation_scope(conn):
                    write_fixture_index_session(
                        conn, parent, content_hash=str(session_content_hash(parent)), prepared_write=carrier
                    )
            finally:
                carrier.close()
        else:
            write_fixture_index_session(conn, parent)
        assert _composed_attachments(conn, parent_id) == []
        assert _composed_texts(conn, child_id) == ["m0", "m1", "x2"]
        for session_id in ids:
            attachments = _composed_attachments(conn, session_id)
            assert [(message, name) for _, message, name in attachments] == [("m1", "shared.txt")]
            assert all(owner != parent_id for owner, _, _ in attachments)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("SELECT COUNT(*) FROM attachments WHERE ref_count <= 0").fetchone()[0] == 0


def test_inherited_stream_hydrates_only_message_owned_attachments_in_one_snapshot(tmp_path: Path) -> None:
    from collections.abc import AsyncIterator
    from contextlib import aclosing, asynccontextmanager, closing

    from polylogue.storage.sqlite.query_store import SQLiteQueryStore

    db = tmp_path / "index.db"
    conn = _connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    parent = _codex_session("parent", ["m0", "m1", "m2"]).model_copy(
        update={
            "attachments": [
                _prefix_attachment("m1", caption="original"),
                _prefix_attachment("m2").model_copy(
                    update={"provider_attachment_id": "outside", "name": "outside.txt"}
                ),
            ]
        }
    )
    parent_id = write_fixture_index_session(conn, parent)
    child = _codex_session("child", ["m0", "m1", "x2"], parent="parent").model_copy(
        update={
            "attachments": [
                _prefix_attachment("m1", caption="original"),
                _prefix_attachment("x2", caption="original tail").model_copy(
                    update={"provider_attachment_id": "tail", "name": "tail.txt"}
                ),
            ]
        }
    )
    child_id = write_fixture_index_session(conn, child)
    conn.commit()
    conn.close()

    async def exercise() -> None:
        held: list[aiosqlite.Connection] = []

        @asynccontextmanager
        async def connection() -> AsyncIterator[aiosqlite.Connection]:
            async with aiosqlite.connect(db) as reader:
                reader.row_factory = aiosqlite.Row
                held.append(reader)
                yield reader

        queries = SQLiteQueryStore(connection_factory=connection)
        async with aclosing(queries.iter_messages(child_id, chunk_size=2)) as stream:
            first = await anext(stream)
            assert first.provider_message_id == "m0"
            assert held[-1].in_transaction
            with closing(_connect(db)) as writer:
                writer.execute("UPDATE attachment_refs SET caption = 'changed'")
                writer.commit()
            rest = [record async for record in stream]
            assert [record.provider_message_id for record in rest] == ["m1", "x2"]
            assert [(a.session_id, a.display_name, a.caption) for a in rest[0].attachments] == [
                (parent_id, "shared.txt", "original")
            ]
            assert not first.attachments
            assert [(a.session_id, a.display_name, a.caption) for a in rest[1].attachments] == [
                (child_id, "tail.txt", "original tail")
            ]
            assert all(a.display_name != "outside.txt" for record in rest for a in record.attachments)
        eager = await queries.get_messages(child_id)
        page, total, completeness = await queries.get_messages_paginated(child_id, limit=3)
        grouped = await queries.get_messages_batch([child_id])
        assert total == 3 and completeness.complete
        for records in (eager, page, grouped[child_id]):
            assert [record.provider_message_id for record in records] == ["m0", "m1", "x2"]
            assert [(a.session_id, a.display_name, a.caption) for a in records[1].attachments] == [
                (parent_id, "shared.txt", "changed")
            ]

    asyncio.run(exercise())
