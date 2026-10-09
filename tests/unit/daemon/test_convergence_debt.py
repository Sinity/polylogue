"""The ``lineage_prefix_recompose`` debt stage has an owner that drains it.

``storage/sqlite/archive_tiers/write.py`` records convergence debt whenever a
provider-session identity contradiction truncates a child's recomposed lineage
prefix. Until ``make_default_convergence_stages`` registered an implementation
for that stage name, the drain classified every such row as unimplemented, so
the rows accumulated and nothing ever re-derived the lost prefix
(polylogue-ia88n).

These tests drive the real production route end to end: a real archive, the
real writer, the real ``_drain_convergence_debt_once``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.replay_lineage import LineageGraph, LineageNode, codex_lineage_payload, seed_lineage_graph
from tests.infra.retained_replay import replay_retained_components

CHILD = "codex-session:s01"
PARENT = "codex-session:s00"


def _index(root: Path) -> sqlite3.Connection:
    # Index capture binds its seal to the connection's original measured creator.
    conn = connect_measured(root / "index.db", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _child_edge(root: Path) -> tuple[str | None, str | None]:
    """The child's ``(resolved parent, branch anchor)`` as the index holds them."""
    with _index(root) as conn:
        row = conn.execute(
            """SELECT resolved_dst_session_id, branch_point_message_id
               FROM session_links WHERE src_session_id = ?""",
            (CHILD,),
        ).fetchone()
    assert row is not None, "the child's parent claim must always be recorded"
    return (None if row[0] is None else str(row[0]), None if row[1] is None else str(row[1]))


def _debt(root: Path) -> list[sqlite3.Row]:
    with sqlite3.connect(root / "ops.db") as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT stage, target_type, target_id, last_error FROM convergence_debt").fetchall()


def _make_retry_due(root: Path) -> None:
    with sqlite3.connect(root / "ops.db") as conn:
        conn.execute("UPDATE convergence_debt SET next_retry_at = '1970-01-01T00:00:00+00:00'")
        conn.commit()


def _contender(*aliases: str) -> ParsedSession:
    """A second session claiming the parent's provider-session value."""
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="contender",
        title="contender",
        provider_session_aliases=list(aliases),
        messages=[
            ParsedMessage(
                provider_message_id="x0",
                role=Role.USER,
                text="unrelated",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="unrelated")],
            )
        ],
    )


def _write(root: Path, session: ParsedSession) -> None:
    conn = _index(root)
    try:
        write_fixture_index_session(conn, session)
        conn.commit()
    finally:
        conn.close()


def _drain_on_admitted_owner(root: Path) -> int:
    """Run one debt page as the daemon does: on its compute creator, admitted.

    ``_run_convergence_debt_pass`` submits the drain to the daemon's compute
    adapter with the stage write admission bound; the recompose stage's
    retained-raw replay prepares only on that admitted creator.
    """
    import asyncio

    from polylogue.daemon import cli as daemon_cli
    from tests.infra.archive_templates import run_off_event_loop
    from tests.infra.live_ingest import prepared_live_convergence_owner

    async def run() -> int:
        async with prepared_live_convergence_owner(root) as owner:
            return await owner.run_convergence_sync(
                "test.convergence-debt.drain",
                lambda: daemon_cli._drain_convergence_debt_once(
                    root / "index.db", compute_adapter=owner._compute_adapter
                ),
            )

    return run_off_event_loop(lambda: asyncio.run(run()))


@pytest.fixture
def truncated_child(tmp_path: Path) -> Path:
    """An archive whose child lost its recomposed prefix to an alias collision."""
    root = tmp_path / "archive"
    seed_lineage_graph(
        root,
        LineageGraph(
            nodes=(
                LineageNode(native_id="s00", parent_native_id=None, tail_length=2),
                LineageNode(native_id="s01", parent_native_id="s00", tail_length=2),
            ),
            write_order=(0, 1),
        ),
    )
    replay_retained_components(root)
    assert _child_edge(root)[0] == PARENT, "the fixture must start with a recomposed prefix"

    _write(root, _contender("s00"))
    assert _child_edge(root)[0] is None, "the alias collision must truncate the child"
    assert [(row["stage"], row["target_id"]) for row in _debt(root)] == [("lineage_prefix_recompose", CHILD)]
    return root


def test_lineage_prefix_debt_is_drained_by_its_stage(truncated_child: Path) -> None:
    """One drain pass re-derives the child's prefix and clears the row.

    Anti-vacuity: drop ``make_lineage_prefix_recompose_stage`` from
    ``make_default_convergence_stages`` and the drain reports the row as an
    unimplemented stage, leaving the edge unresolved and the row in place --
    both assertions below go red.
    """
    root = truncated_child
    # The contender re-parses without the contested alias: the parent's claim
    # is unambiguous again, so retained evidence can settle the edge.
    _write(root, _contender())
    assert _child_edge(root)[0] is None, "no ordinary write re-resolves the child"

    _make_retry_due(root)
    assert _drain_on_admitted_owner(root) == 1

    parent, anchor = _child_edge(root)
    assert parent == PARENT
    assert anchor is not None
    assert _debt(root) == []
    assert _composed_texts(root, CHILD) == ["s00-tail-0", "s00-tail-1", "s01-tail-0", "s01-tail-1"]
    from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope

    conn = _index(root)
    try:
        assert read_archive_session_envelope(conn, CHILD).lineage_complete is True
    finally:
        conn.close()


def test_parent_rewrite_cannot_discharge_recorded_prefix_loss(truncated_child: Path) -> None:
    """Resolving a stored tail is not proof that the child's raw prefix returned."""
    import asyncio

    import aiosqlite

    from polylogue.operations.lineage_prefix_recompose import unrecomposed_prefix_reason
    from polylogue.storage.derived.lineage.compact import derive_compact_lineage
    from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope
    from polylogue.storage.sqlite.queries.message_query_reads import (
        get_lineage_completeness,
        get_messages_with_lineage_completeness,
    )

    root = truncated_child
    _write(root, _contender())
    rewritten = codex_lineage_payload("s00", ["s00-tail-0", "s00-tail-1", "new-parent-turn"])
    (parent,) = parse_payload(
        Provider.CODEX, [json.loads(line) for line in rewritten.splitlines()], "s00", source_path="s00.jsonl"
    )
    _write(root, parent)
    assert _child_edge(root)[0] == PARENT
    assert _composed_texts(root, CHILD) == ["s01-tail-0", "s01-tail-1"]
    conn = _index(root)
    try:
        envelope = read_archive_session_envelope(conn, CHILD)
        assert envelope.lineage_complete is False
        assert envelope.lineage_truncation_reason == "dangling_branch_point"
        assert unrecomposed_prefix_reason(conn, CHILD) == "recorded_prefix_loss"
        graph = derive_compact_lineage(conn, CHILD)
        assert graph is not None and graph.seed_node().accounting.status == "unknown"
        evidence = json.loads(
            conn.execute("SELECT evidence_json FROM session_links WHERE src_session_id = ?", (CHILD,)).fetchone()[0]
        )
        assert evidence["invalidated_prefix"]["branch_point_message_id"] == "codex-session:s00:n:m1"
        assert len(evidence["invalidated_prefix"]["branch_point_content_address"]) == 64
    finally:
        conn.close()

    async def async_completeness() -> None:
        async with aiosqlite.connect(f"{(root / 'index.db').as_uri()}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            _, full = await get_messages_with_lineage_completeness(conn, CHILD)
            bounded = await get_lineage_completeness(conn, CHILD)
            assert full.complete is False and bounded.complete is False
            assert full.truncation_reason == bounded.truncation_reason == "dangling_branch_point"

    asyncio.run(async_completeness())
    assert len(_debt(root)) == 1
    _make_retry_due(root)
    assert _drain_on_admitted_owner(root) == 1
    assert _debt(root) == []
    assert _composed_texts(root, CHILD) == ["s00-tail-0", "s00-tail-1", "s01-tail-0", "s01-tail-1"]


def test_cancelled_recompose_keeps_prefix_loss(
    truncated_child: Path, bounded_compute_adapter: BoundedComputeAdapter
) -> None:
    import threading

    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.core.compute_cancel import compute_cancel
    from polylogue.operations.lineage_prefix_recompose import recompose_session_prefix, unrecomposed_prefix_reason

    root = truncated_child
    _write(root, _contender())
    before = _debt(root)
    cancelled = threading.Event()
    cancelled.set()
    token = compute_cancel.set(cancelled)
    try:
        with pytest.raises(DaemonOperationCancelled):
            recompose_session_prefix(root, root / "index.db", CHILD, compute_adapter=bounded_compute_adapter)
    finally:
        compute_cancel.reset(token)
    assert [tuple(row) for row in _debt(root)] == [tuple(row) for row in before]
    conn = _index(root)
    try:
        assert unrecomposed_prefix_reason(conn, CHILD) == "recorded_prefix_loss"
    finally:
        conn.close()


def test_child_append_keeps_recorded_prefix_loss(truncated_child: Path) -> None:
    from polylogue.archive.session.branch_type import BranchType
    from polylogue.operations.lineage_prefix_recompose import unrecomposed_prefix_reason
    from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope

    root = truncated_child
    appended = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="s01",
        parent_session_provider_id="s00",
        branch_type=BranchType.FORK,
        messages=[ParsedMessage(provider_message_id="appended", role=Role.USER, text="new-child-turn")],
    )
    conn = _index(root)
    try:
        write_fixture_index_session(conn, appended, merge_append=True)
        conn.commit()
        assert unrecomposed_prefix_reason(conn, CHILD) == "recorded_prefix_loss"
        assert read_archive_session_envelope(conn, CHILD).lineage_complete is False
    finally:
        conn.close()


def test_lineage_prefix_debt_survives_live_contradiction(truncated_child: Path) -> None:
    """A row is never cleared while the contradiction still blocks recompose.

    Anti-vacuity: make the stage report convergence without proving the prefix
    came back (return ``True`` from ``execute_sessions``, or drop the
    post-replay recheck in ``recompose_session_prefix``) and the drain clears a
    row whose child is still truncated, so the surviving-row assertion goes
    red. The error assertion is the other half: a stage that refused by
    returning ``False`` would overwrite the writer's diagnostic with the
    engine's generic "returned False".
    """
    root = truncated_child
    _make_retry_due(root)
    assert _drain_on_admitted_owner(root) == 1

    assert _child_edge(root)[0] is None
    rows = _debt(root)
    assert [(row["stage"], row["target_id"]) for row in rows] == [("lineage_prefix_recompose", CHILD)]
    error = str(rows[0]["last_error"])
    assert "identity contradiction" in error
    assert "'s00'" in error
    assert "returned False" not in error


def _composed_texts(root: Path, session_id: str) -> list[str | None]:
    from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope

    with _index(root) as conn:
        envelope = read_archive_session_envelope(conn, session_id)
        return [message.blocks[0].text if message.blocks else None for message in envelope.messages]


def test_parent_rewrite_on_the_replay_route_never_strands_a_child(tmp_path: Path) -> None:
    """polylogue-gy2yu on a replayed archive: a parent re-parse that rewrites
    the child's branch-point message leaves the child whole in the same write.

    The child's inherited prefix is evidence from its own bytes, so the write
    materializes it into the child instead of recording a lineage debt for a
    later re-derivation.

    Anti-vacuity: skip ``_settle_inherited_prefixes`` in the writer and the
    child composes short while no debt names it.
    """
    root = tmp_path / "archive"
    seed_lineage_graph(
        root,
        LineageGraph(
            nodes=(
                LineageNode(native_id="s00", parent_native_id=None, tail_length=3),
                LineageNode(native_id="s01", parent_native_id="s00", tail_length=2),
            ),
            write_order=(0, 1),
        ),
    )
    replay_retained_components(root)
    complete = ["s00-tail-0", "s00-tail-1", "s00-tail-2", "s01-tail-0", "s01-tail-1"]
    assert _composed_texts(root, CHILD) == complete

    # The parent is re-acquired with the message the child branched after
    # rewritten under a new native id.
    rewritten = codex_lineage_payload("s00", ["s00-tail-0", "s00-tail-1", "s00-tail-2-rewritten"])
    rewritten = rewritten.replace(b'"id":"m2"', b'"id":"m2-rewritten"')
    (parent_session,) = parse_payload(
        Provider.CODEX, [json.loads(line) for line in rewritten.splitlines()], "s00", source_path="s00.jsonl"
    )
    _write(root, parent_session)
    assert _composed_texts(root, PARENT) == ["s00-tail-0", "s00-tail-1", "s00-tail-2-rewritten"]
    assert _composed_texts(root, CHILD) == complete
    assert _debt(root) == []


def test_hook_paste_debt_is_retried_for_its_session_and_cleared(
    tmp_path: Path, bounded_compute_adapter: BoundedComputeAdapter
) -> None:
    """The registered session callback applies retained hook evidence.

    Anti-vacuity: omit the hook-paste stage or leave its session callbacks
    absent and the due debt is not acted on, so the message remains unmarked.
    """
    from polylogue.daemon import cli as daemon_cli

    root = tmp_path / "archive"
    root.mkdir()
    index_db = root / "index.db"
    source_db = root / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_runtime_source_fixture(source_db)

    hook_time_ms = int(datetime(2026, 5, 7, 12, 0, tzinfo=UTC).timestamp() * 1000)
    session_id = "codex-session:hook-native"
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """INSERT INTO sessions (
                   native_id, origin, content_hash, created_at_ms, updated_at_ms
               ) VALUES (?, 'codex-session', ?, ?, ?)""",
            ("hook-native", b"s" * 32, hook_time_ms, hook_time_ms),
        )
        conn.execute(
            """INSERT INTO messages (
                   session_id, native_id, position, role, content_hash, occurred_at_ms
               ) VALUES (?, 'm1', 0, 'user', ?, ?)""",
            (session_id, b"m" * 32, hook_time_ms + 100),
        )

    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """INSERT INTO raw_hook_events (
                   hook_event_id, origin, native_id, session_native_id,
                   source_path, event_type, payload_json, observed_at_ms
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "hook:e1",
                "codex-session",
                "hook-native:UserPromptSubmit:e1",
                "hook-native",
                "/spool/pending/e1.json",
                "UserPromptSubmit",
                '{"event_type":"UserPromptSubmit","timestamp":"2026-05-07T12:00:00Z",'
                '"payload":{"session_id":"hook-native","prompt":"Inspect [Pasted text #1]"}}',
                hook_time_ms,
            ),
        )

    cursor = CursorStore(index_db)
    cursor.record_convergence_debt(
        stage="hook_paste_enrichment",
        subject_type="session_id",
        subject_id=session_id,
        error="initial hook paste enrichment failed",
    )
    _make_retry_due(root)

    assert daemon_cli._drain_convergence_debt_once(index_db, compute_adapter=bounded_compute_adapter) == 1

    with sqlite3.connect(index_db) as conn:
        message = conn.execute(
            "SELECT has_paste FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    assert message == (1,)
    assert cursor.list_convergence_debt() == []
