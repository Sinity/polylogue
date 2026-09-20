"""Lineage normalization (#2467): a prefix-sharing child (fork / resume /
spawned subagent / auto-compaction copy) copies the parent's leading context.
The archive must store only the child's divergent tail plus a lineage edge with
a branch point, and reads must compose the parent prefix back in.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers import hermes_state
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
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION, LineageCompleteness
from polylogue.storage.sqlite.archive_tiers import write as _write_module
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    _MAX_LINEAGE_DEPTH,
    IDENTITY_INVALIDATION_DEBT_STAGE,
    _provider_usage_cumulative_baseline,
    count_dangling_prefix_branch_points,
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
from tests.infra.identity import archive_message_id


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
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


def _seed_fresh_session_products(conn: sqlite3.Connection, session_id: str, *, message_count: int) -> None:
    """Materialize the derived partition a converger would have written for
    ``session_id`` as it stands, stamping the value-complete input binding so
    inspection reports it current."""
    binding = session_input_bindings(conn, (session_id,))[session_id]
    conn.execute(
        """
        INSERT INTO session_profiles (
            session_id, materializer_version, materialized_at, source_updated_at,
            source_sort_key, input_content_hash, input_row_count, source_name,
            message_count
        )
        SELECT session_id, ?, '', datetime(updated_at_ms / 1000, 'unixepoch'),
               CAST(sort_key_ms AS REAL) / 1000.0, ?, ?, origin, ?
        FROM sessions
        WHERE session_id = ?
        """,
        (SESSION_INSIGHT_MATERIALIZER_VERSION, binding, message_count, message_count, session_id),
    )
    conn.execute(
        "INSERT INTO session_latency_profiles (session_id, materializer_version, materialized_at, source_name)"
        " VALUES (?, ?, '', '')",
        (session_id, SESSION_INSIGHT_MATERIALIZER_VERSION),
    )


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
        SELECT input_tokens, output_tokens, cache_read_tokens,
               CASE WHEN provider_cost_usd IS NOT NULL THEN 'origin_reported'
                    WHEN catalog_cost_usd IS NOT NULL THEN 'priced' END AS cost_provenance
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
        "cost_provenance": "priced",
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
            "total_input_tokens": 60,
            "total_cached_input_tokens": 10,
            "total_output_tokens": 15,
            "total_tokens": 75,
        },
        {
            "source_message_id": archive_message_id(child_id, "cy"),
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
    branch_point = archive_message_id(ancestor_id, "a1")

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
    parent_id = write_parsed_session_to_archive(conn, parent)

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
    child_id = write_parsed_session_to_archive(conn, child)

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
    fresh_id = write_parsed_session_to_archive(conn, fresh)
    fresh_link = conn.execute("SELECT inheritance FROM session_links WHERE src_session_id = ?", (fresh_id,)).fetchone()
    assert fresh_link["inheritance"] == "spawned-fresh"
    assert [
        row[0]
        for row in conn.execute(
            "SELECT position FROM messages WHERE session_id = ? ORDER BY position", (fresh_id,)
        ).fetchall()
    ] == [0, 1]

    conn.close()

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
    parent_id = write_parsed_session_to_archive(conn, parent)
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
    child_id = write_parsed_session_to_archive(conn, child)
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

    conn.close()
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
    conn.close()

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
    conn.close()

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
    ``_reextract_prefix_tail_db`` leaves the stale rows in place and the child
    out of the repair-candidate set.
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
    child_id = write_parsed_session_to_archive(conn, child)
    _seed_fresh_session_products(conn, child_id, message_count=4)
    assert _nonvalid_partitions(conn) == []

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
    write_parsed_session_to_archive(conn, parent)

    stored = conn.execute(
        "SELECT position FROM messages WHERE session_id = ? ORDER BY position", (child_id,)
    ).fetchall()
    assert [row[0] for row in stored] == [2, 3]
    for relation in ("session_profiles", "session_latency_profiles"):
        retained = conn.execute(f"SELECT COUNT(*) FROM {relation} WHERE session_id = ?", (child_id,)).fetchone()[0]
        assert retained == 0, f"{relation} retained the pre-extraction projection"
    assert child_id in _nonvalid_partitions(conn)

    conn.close()


@pytest.mark.parametrize("child_first", [False, True], ids=["parent-first", "child-first"])
def test_variant_prefix_lineage_converges_across_order_and_parent_replacement(
    tmp_path: Path, child_first: bool
) -> None:
    """The production write/read routes agree on every sibling variant.

    This is the deterministic reduction of the state-machine order failure:
    parent-first and child-first ingestion must leave the same child tail,
    resolved link, and composed transcript.  Replacing the parent then proves
    the child composes the latest accepted sibling values rather than retaining
    a duplicate child-owned variant.
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
        child_id = write_parsed_session_to_archive(conn, child)
        parent_id = write_parsed_session_to_archive(conn, parent)
    else:
        parent_id = write_parsed_session_to_archive(conn, parent)
        child_id = write_parsed_session_to_archive(conn, child)

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
    write_parsed_session_to_archive(conn, replacement)

    assert asyncio.run(_read_texts(db, child_id)) == ["root", "primary v2", "sibling v2", "child tail"]
    assert (
        conn.execute(
            "SELECT native_id, position, variant_index FROM messages WHERE session_id = ? ORDER BY position, variant_index",
            (child_id,),
        ).fetchall()
        == physical
    )
    assert (
        conn.execute(
            "SELECT resolved_dst_session_id, branch_point_message_id, inheritance, status "
            "FROM session_links WHERE src_session_id = ?",
            (child_id,),
        ).fetchone()
        == link
    )
    conn.close()


def test_missing_variant_branch_point_keeps_only_owned_child_tail(tmp_path: Path) -> None:
    """A full parent replacement may not substitute a shorter sibling prefix."""
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
    parent_id = write_parsed_session_to_archive(conn, parent)
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
    child_id = write_parsed_session_to_archive(conn, child)

    write_parsed_session_to_archive(
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
    assert tuple(link) == (archive_message_id(parent_id, "p1-alt"), "prefix-sharing", None)
    envelope = read_archive_session_envelope(conn, child_id)
    assert [message.blocks[0].text for message in envelope.messages] == ["child tail"]
    assert envelope.lineage_complete is False
    assert envelope.lineage_truncation_reason == "dangling_branch_point"
    conn.close()


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
    root_id = write_parsed_session_to_archive(conn, root)
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
    parent_id = write_parsed_session_to_archive(conn, parent)
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
    child_id = write_parsed_session_to_archive(conn, child)

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
    conn.close()


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
    root_id = write_parsed_session_to_archive(conn, root)
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
    parent_id = write_parsed_session_to_archive(conn, parent)
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
    child_id = write_parsed_session_to_archive(conn, child)

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
    conn.close()


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
    parent_id = write_parsed_session_to_archive(conn, parent)
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
    child_id = write_parsed_session_to_archive(conn, child)

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
        )
        fired["count"] += 1
        raise RuntimeError("fault after graph resolution")

    monkeypatch.setattr(_write_module, "_resolve_session_graph", _fail_after_graph_resolution)
    with pytest.raises(RuntimeError, match="fault after graph resolution"):
        write_parsed_session_to_archive(conn, replacement)
    assert fired["count"] == 1, "fault injection did not reach graph resolution"
    assert _state() == before

    monkeypatch.setattr(_write_module, "_resolve_session_graph", real_resolve)
    write_parsed_session_to_archive(conn, replacement)
    after_retry = _state()
    assert after_retry[2:] == (["root", "parent v2", "child tail"], True, None)
    write_parsed_session_to_archive(conn, replacement)
    assert _state() == after_retry
    conn.close()


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

    repaired = repair_stale_prefix_branch_points(conn)
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
    assert branch_point == archive_message_id(ancestor_id, "msg-10")
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
        attachments=[
            ParsedAttachment(
                provider_attachment_id="prefix-attachment",
                message_provider_id="c1",
                name="prefix.txt",
                path="prefix.txt",
            )
        ],
    )
    child_id = write_parsed_session_to_archive(conn, child)
    assert conn.execute("SELECT COUNT(*) FROM blocks WHERE session_id = ?", (child_id,)).fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM attachment_native_ids").fetchone()[0] == 1

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
    assert conn.execute("SELECT COUNT(*) FROM attachment_native_ids").fetchone()[0] == 0
    # The prefix-anchored attachment lost its only ref; the row must be swept,
    # not left acquired-but-unreachable for archive verification.
    assert conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0] == 0
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
        attachments=[
            ParsedAttachment(
                provider_attachment_id="empty-tail-attachment",
                message_provider_id="c1",
                name="empty-tail.txt",
                path="empty-tail.txt",
            )
        ],
    )
    child_id = write_parsed_session_to_archive(conn, child)
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
    assert conn.execute("SELECT COUNT(*) FROM attachment_native_ids").fetchone()[0] == 0
    # Same sweep requirement as the partial-tail path: no ref-less rows survive.
    assert conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0] == 0
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
        SELECT input_tokens, output_tokens, cache_read_tokens,
               CASE WHEN provider_cost_usd IS NOT NULL THEN 'origin_reported'
                    WHEN catalog_cost_usd IS NOT NULL THEN 'priced' END AS cost_provenance
        FROM session_model_usage
        WHERE session_id = ? AND model_name = 'gpt-5-codex'
        """,
        (child_id,),
    ).fetchone()
    assert dict(usage) == {
        "input_tokens": 50,
        "output_tokens": 15,
        "cache_read_tokens": 10,
        "cost_provenance": "priced",
    }
    # polylogue-664l: session_provider_usage_events dropped its 8 Hermes
    # billing-provenance columns (index v61, zero production readers). The
    # third session_event above carries only billing evidence (no token
    # counts), so `_provider_usage_event_row_has_evidence` -- now gated
    # purely on the token counters -- correctly writes no row for it. Of the
    # two remaining events, the "c1" one is deleted by the prefix-tail
    # reextraction (its source message is in the shared parent prefix); only
    # the divergent-tail "cy" event survives, with baseline subtraction
    # applied against the parent's cumulative totals.
    remaining = conn.execute(
        "SELECT total_input_tokens, total_tokens FROM session_provider_usage_events WHERE session_id = ?",
        (child_id,),
    ).fetchall()
    assert len(remaining) == 1
    assert dict(remaining[0]) == {"total_input_tokens": 60, "total_tokens": 75}


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


def test_writer_composes_beyond_recursive_reader_depth(tmp_path: Path) -> None:
    """A valid branch point beyond the sync reader's stack guard stays valid.

    The writer and async reader are iterative and share a larger runaway limit.
    If writer composition accidentally reuses the sync reader's 64-level guard,
    the later descendants cannot see the root branch point and become
    spawned-fresh with a duplicated root message.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    provider_session_id = "deep-root"
    root_id = write_parsed_session_to_archive(
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
    for level in range(_MAX_LINEAGE_DEPTH + 2):
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
        leaf_id = write_parsed_session_to_archive(conn, leaf)
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
    write_parsed_session_to_archive(
        conn,
        leaf.model_copy(update={"updated_at": "2027-01-01T00:59:59Z"}),
    )
    link = conn.execute(
        "SELECT inheritance, branch_point_message_id FROM session_links WHERE src_session_id = ?",
        (leaf_id,),
    ).fetchone()
    assert tuple(link) == ("prefix-sharing", archive_message_id(root_id, "root-0"))
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (leaf_id,)).fetchone()[0] == 1
    conn.close()

    assert asyncio.run(_read_texts(db, leaf_id)) == ["root message", f"level {_MAX_LINEAGE_DEPTH + 1} tail"]


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
        written[name] = write_parsed_session_to_archive(conn, by_name[name])
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
    conn.close()


def test_dangling_branch_point_census_counts_edges_and_sessions(tmp_path: Path) -> None:
    """polylogue-7xrv5: the archive-wide census is what makes a truncating
    archive measurable after a rebuild, before the next daemon start runs the
    repair.

    Anti-vacuity: a census that ignored the branch point's existence (or scoped
    itself to one session) would report ``(0, 0)`` for the corrupted state
    below.
    """
    db = tmp_path / "index.db"
    conn = _connect(db)
    grandparent, parent, child = _three_generation_sessions()
    write_parsed_session_to_archive(conn, grandparent)
    parent_id = write_parsed_session_to_archive(conn, parent)
    child_id = write_parsed_session_to_archive(conn, child)
    conn.commit()
    assert count_dangling_prefix_branch_points(conn) == (0, 0)

    conn.execute(
        "UPDATE session_links SET branch_point_message_id = ? WHERE src_session_id = ?",
        (f"{parent_id}:n:m1", child_id),
    )
    conn.commit()
    assert count_dangling_prefix_branch_points(conn) == (1, 1)
    assert [message.blocks[0].text for message in read_archive_session_envelope(conn, child_id).messages] == ["x2"]

    assert repair_stale_prefix_branch_points(conn) == 1
    conn.commit()
    assert count_dangling_prefix_branch_points(conn) == (0, 0)
    conn.close()


def _hermes_chain_state_db(path: Path, *, links: int, messages_per_session: int) -> None:
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
            source.execute(
                "INSERT INTO sessions VALUES (?, 'cli', '{}', ?, ?, ?, 'compression', ?)",
                (session_id, parent, float(index), float(index) + 0.5, f"Session {index}"),
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


def test_hermes_continuation_hydration_refuses_past_its_declared_composition_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound is the fix: a chain past it refuses, typed, before composing it.

    Anti-vacuity: with the bound removed (or raised past the chain), this same
    input composes ``links * (links + 1) / 2 * messages_per_session`` messages
    and the call returns instead of raising -- exactly the unbounded behaviour
    this test exists to keep out. The composed-message counter also proves the
    refusal is *pre-allocation*: a bound checked after composing would leave the
    counter at the full quadratic total.
    """
    links = 40
    messages_per_session = 10
    limit = 600
    monkeypatch.setattr(hermes_state, "HERMES_MAX_COMPOSED_MESSAGES", limit)

    composed_messages = 0
    original_copy = ParsedMessage.model_copy

    def counting_copy(self: ParsedMessage, **kwargs: Any) -> ParsedMessage:
        nonlocal composed_messages
        composed_messages += 1
        return original_copy(self, **kwargs)

    monkeypatch.setattr(ParsedMessage, "model_copy", counting_copy)

    state_db = tmp_path / "state.db"
    _hermes_chain_state_db(state_db, links=links, messages_per_session=messages_per_session)

    with pytest.raises(hermes_state.HermesLineageBoundError) as refusal:
        parse_state_db(state_db)

    assert refusal.value.bound == "composed_messages"
    assert refusal.value.limit == limit
    assert refusal.value.observed > limit
    # Named counts, not a generic message: the operator can see what was refused.
    assert str(refusal.value.observed) in str(refusal.value)
    assert str(limit) in str(refusal.value)

    unbounded_total = messages_per_session * links * (links + 1) // 2
    assert unbounded_total == 8200
    # Composition stops at the bound. The slack covers the per-message copies
    # made while building the individual sessions, which are linear in the source.
    assert composed_messages < limit + links * messages_per_session
    assert composed_messages < unbounded_total // 4


def test_hermes_continuation_hydration_copies_are_shallow_and_share_blocks(tmp_path: Path) -> None:
    """Within the bound, recomposition costs one shallow copy per composed message.

    Anti-vacuity: restoring ``model_copy(deep=True)`` makes the block-identity
    assertion red, because a deep copy duplicates the whole block subtree once
    per link -- the term that turned a 2 MB state.db into gigabytes.
    """
    links = 6
    messages_per_session = 3
    state_db = tmp_path / "state.db"
    _hermes_chain_state_db(state_db, links=links, messages_per_session=messages_per_session)

    parsed = parse_state_db(state_db)
    by_raw_id = {session.provider_session_id.split("@", 1)[0]: session for session in parsed}

    # Recomposition semantics are preserved: each child reads as parent prefix + tail.
    for index in range(links):
        session = by_raw_id[f"s{index:04d}"]
        assert [message.text for message in session.messages] == [
            f"s{link:04d} message {offset}" for link in range(index + 1) for offset in range(messages_per_session)
        ]
        assert [message.position for message in session.messages] == list(range((index + 1) * messages_per_session))
        assert session.messages[-1].is_active_leaf is True
        assert not any(message.is_active_leaf for message in session.messages[:-1])

    root_block = by_raw_id["s0000"].messages[0].blocks[0]
    for index in range(1, links):
        composed_block = by_raw_id[f"s{index:04d}"].messages[0].blocks[0]
        assert composed_block is root_block, "recomposition must share blocks, never deep-copy them"


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
    write_parsed_session_to_archive(conn, parent)

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
    child_id = write_parsed_session_to_archive(conn, child)
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
    write_parsed_session_to_archive(conn, contender)
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
