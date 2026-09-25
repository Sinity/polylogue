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

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.infra.replay_lineage import LineageGraph, LineageNode, seed_lineage_graph

CHILD = "codex-session:s01"
PARENT = "codex-session:s00"


def _index(root: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(root / "index.db")
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
        write_parsed_session_to_archive(conn, session)
        conn.commit()
    finally:
        conn.close()


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
    backfill_historical_revision_evidence(root)
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
    from polylogue.daemon import cli as daemon_cli

    root = truncated_child
    # The contender re-parses without the contested alias: the parent's claim
    # is unambiguous again, so retained evidence can settle the edge.
    _write(root, _contender())
    assert _child_edge(root)[0] is None, "no ordinary write re-resolves the child"

    _make_retry_due(root)
    assert daemon_cli._drain_convergence_debt_once(root / "index.db") == 1

    parent, anchor = _child_edge(root)
    assert parent == PARENT
    assert anchor is not None
    assert _debt(root) == []


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
    from polylogue.daemon import cli as daemon_cli

    root = truncated_child
    _make_retry_due(root)
    assert daemon_cli._drain_convergence_debt_once(root / "index.db") == 1

    assert _child_edge(root)[0] is None
    rows = _debt(root)
    assert [(row["stage"], row["target_id"]) for row in rows] == [("lineage_prefix_recompose", CHILD)]
    error = str(rows[0]["last_error"])
    assert "identity contradiction" in error
    assert "'s00'" in error
    assert "returned False" not in error


def test_hook_paste_debt_is_retried_for_its_session_and_cleared(tmp_path: Path) -> None:
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
    initialize_archive_database(source_db, ArchiveTier.SOURCE)

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

    assert daemon_cli._drain_convergence_debt_once(index_db) == 1

    with sqlite3.connect(index_db) as conn:
        message = conn.execute(
            "SELECT has_paste FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    assert message == (1,)
    assert cursor.list_convergence_debt() == []
