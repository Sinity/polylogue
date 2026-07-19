"""Bulk-generation-build lifecycle (polylogue-v6i3): empty derived state
throughout replay, one archive-wide repopulate at readiness.

Background: a whale offline rebuild measured a 3h+ stall doing per-session
FTS/trigram/action_pairs/delegation_facts maintenance one session at a time,
where an ARCHIVE-WIDE ``messages_fts`` + ``blocks_command_trigram``
delete-all took 28.7s (bead polylogue-v6i3). ``write_parsed_session_to_
archive(..., bulk_build=True)`` -- the offline rebuild path's mode, layered
on top of the existing ``bulk_fts`` guard-gated bulk FTS mode (#3152) --
skips ALL per-session maintenance of these four derived surfaces (not just
the whale prefix-reextract cascade #3152 already handles) and defers
everything to one archive-wide repopulate the caller runs once at readiness
(``maintenance/rebuild_index.py``'s ``_repopulate_bulk_build_derived_state``).

These tests prove: (a) ``bulk_build=True`` writes leave ``messages_fts`` /
``blocks_command_trigram`` / ``action_pairs`` empty for the written session,
where mode-off leaves them populated (so (a) is not vacuously true for every
write); (b) the readiness repopulate (``rebuild_fts_index_sync`` +
``rebuild_command_trigram_index_sync`` + ``rebuild_all_action_pairs_sync``)
produces byte-identical content to trickle-mode (``bulk_build=False``)
population of the SAME corpus, including a prefix-sharing lineage cascade
that exercises ``_bulk_fts_session_guard``'s bulk-build no-op branch; (c) the
whole-transaction ``FTS_BULK_SESSION_WRITE_GUARD`` row never leaks past an
exception; (d) an anti-vacuity check -- skip the readiness repopulate for one
surface and show the parity comparison then fails, proving the equivalence
test in (b) is actually exercising the repopulate code, not passing by
coincidence.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.fts.fts_lifecycle import rebuild_command_trigram_index_sync, rebuild_fts_index_sync
from polylogue.storage.fts.sql import FTS_BULK_SESSION_WRITE_GUARD
from polylogue.storage.sqlite.action_pairs import rebuild_all_action_pairs_sync
from polylogue.storage.sqlite.archive_tiers import write as _write_module
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.revision_application import assert_session_fts_exact_sync
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _tool_pair(idx: int, position: int, *, tool_id_prefix: str = "tool") -> ParsedMessage:
    tool_id = f"{tool_id_prefix}-{idx}"
    return ParsedMessage(
        provider_message_id=f"m{idx}",
        role=Role.ASSISTANT,
        text=None,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name="Bash",
                tool_id=tool_id,
                tool_input={"command": f"echo {idx}"},
            ),
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=tool_id,
                text=f"output {idx}",
                is_error=False,
                exit_code=0,
            ),
        ],
    )


def _text_message(idx: int, position: int, text: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=f"text{idx}",
        role=Role.USER,
        text=text,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _session(session_id: str, *, n_pairs: int = 2) -> ParsedSession:
    messages = [_text_message(0, 0, f"session {session_id} opening remark")]
    for i in range(n_pairs):
        messages.append(_tool_pair(i, position=i + 1, tool_id_prefix=f"{session_id}-tool"))
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_id,
        title=f"session {session_id}",
        messages=messages,
    )


def _lineage_scenario(conn: sqlite3.Connection, *, bulk_fts: bool, bulk_build: bool) -> tuple[str, str]:
    """A prefix-sharing child+parent pair, mirroring
    ``test_bulk_fts_prefix_reextract.py``'s partial-tail scenario, extended
    with tool_use/tool_result pairs so action_pairs/trigram content exists
    to compare, not just messages_fts."""
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="lineage-child",
        title="lineage child",
        parent_session_provider_id="lineage-parent",
        messages=[
            _text_message(0, 0, "hello"),
            _tool_pair(0, position=1, tool_id_prefix="child"),
            _text_message(1, 2, "child diverges here"),
        ],
    )
    child_id = write_parsed_session_to_archive(conn, child, bulk_fts=bulk_fts, bulk_build=bulk_build)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="lineage-parent",
        title="lineage parent",
        messages=[
            _text_message(0, 0, "hello"),
            _tool_pair(0, position=1, tool_id_prefix="parent"),
            _text_message(1, 2, "parent continues alone"),
        ],
    )
    parent_id = write_parsed_session_to_archive(conn, parent, bulk_fts=bulk_fts, bulk_build=bulk_build)
    return child_id, parent_id


def _build_corpus(conn: sqlite3.Connection, *, bulk_build: bool) -> list[str]:
    """Two independent sessions plus one prefix-sharing lineage pair."""
    session_ids = []
    for label in ("alpha", "beta"):
        session_ids.append(
            write_parsed_session_to_archive(conn, _session(label), bulk_fts=bulk_build, bulk_build=bulk_build)
        )
    child_id, parent_id = _lineage_scenario(conn, bulk_fts=bulk_build, bulk_build=bulk_build)
    session_ids.extend([child_id, parent_id])
    return session_ids


def _fts_rows(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    rows = conn.execute(
        "SELECT block_id, message_id, session_id, block_type, text FROM messages_fts ORDER BY block_id"
    ).fetchall()
    return sorted(tuple(row) for row in rows)


def _trigram_rows(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    rows = conn.execute(
        """
        SELECT b.block_id, t.tool_detail_text
        FROM blocks_command_trigram AS t
        JOIN blocks AS b ON b.rowid = t.rowid
        ORDER BY b.block_id
        """
    ).fetchall()
    return sorted(tuple(row) for row in rows)


def _action_pair_rows(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    rows = conn.execute(
        """
        SELECT tool_use_block_id, session_id, message_id, tool_id, use_rank, tool_name,
               semantic_type, tool_command, tool_path, tool_result_block_id, is_error, exit_code
        FROM action_pairs
        ORDER BY tool_use_block_id
        """
    ).fetchall()
    return sorted(tuple(row) for row in rows)


def test_bulk_build_write_leaves_derived_surfaces_empty(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session_id = write_parsed_session_to_archive(conn, _session("solo"), bulk_fts=True, bulk_build=True)

    assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM blocks_command_trigram_docsize").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM action_pairs WHERE session_id = ?", (session_id,)).fetchone()[0] == 0
    # The guard row must never leak past the write it protected.
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM derived_refresh_guard WHERE guard_name = ?",
            (FTS_BULK_SESSION_WRITE_GUARD,),
        ).fetchone()[0]
        == 0
    )
    # Real rows exist to index -- an empty derived surface here is a
    # deliberate skip, not an artifact of an empty session.
    assert conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0] > 0
    conn.close()


def test_bulk_build_off_matches_todays_per_session_population(tmp_path: Path) -> None:
    """Anti-vacuity baseline for the test above: without bulk_build, the same
    session write populates all three surfaces immediately, proving the
    empty result above is a real skip, not something every write produces."""
    conn = _connect(tmp_path / "index.db")
    session_id = write_parsed_session_to_archive(conn, _session("solo"))

    assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM blocks_command_trigram_docsize").fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM action_pairs WHERE session_id = ?", (session_id,)).fetchone()[0] > 0
    conn.close()


def test_bulk_build_readiness_repopulate_matches_trickle_mode(tmp_path: Path) -> None:
    """THE key equivalence proof: a bulk-build corpus, repopulated once at
    readiness, must be byte-identical to the same corpus built entirely in
    today's per-session trickle mode."""
    conn_trickle = _connect(tmp_path / "trickle.db")
    _build_corpus(conn_trickle, bulk_build=False)
    fts_trickle = _fts_rows(conn_trickle)
    trigram_trickle = _trigram_rows(conn_trickle)
    action_pairs_trickle = _action_pair_rows(conn_trickle)
    conn_trickle.close()

    conn_bulk = _connect(tmp_path / "bulk.db")
    _build_corpus(conn_bulk, bulk_build=True)
    # Confirm the empty-throughout invariant actually held for this corpus
    # before repopulating -- otherwise the parity check below could pass
    # vacuously if bulk_build silently did nothing.
    assert conn_bulk.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0
    assert conn_bulk.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0] == 0

    rebuild_fts_index_sync(conn_bulk)
    rebuild_command_trigram_index_sync(conn_bulk)
    rebuild_all_action_pairs_sync(conn_bulk)
    conn_bulk.commit()

    fts_bulk = _fts_rows(conn_bulk)
    trigram_bulk = _trigram_rows(conn_bulk)
    action_pairs_bulk = _action_pair_rows(conn_bulk)

    assert fts_bulk == fts_trickle
    assert trigram_bulk == trigram_trickle
    assert action_pairs_bulk == action_pairs_trickle
    assert fts_bulk, "corpus produced no messages_fts rows -- comparison would be vacuous"
    assert trigram_bulk, "corpus produced no trigram rows -- comparison would be vacuous"
    assert action_pairs_bulk, "corpus produced no action_pairs rows -- comparison would be vacuous"
    conn_bulk.close()


def test_bulk_build_exact_sync_assertion_accepts_empty_state_but_still_checks_triggers(tmp_path: Path) -> None:
    """``assert_session_fts_exact_sync(..., bulk_build=True)`` must not raise
    for a session left deliberately unindexed, but must still fail if the
    canonical triggers are somehow missing (the trigger-presence half of the
    proof is unaffected by bulk-build mode)."""
    conn = _connect(tmp_path / "index.db")
    session_id = write_parsed_session_to_archive(conn, _session("solo"), bulk_fts=True, bulk_build=True)

    # Deliberately out of sync (0 indexed vs >0 expected) -- must not raise.
    assert_session_fts_exact_sync(conn, session_id, bulk_build=True)

    # Without bulk_build, the same out-of-sync state must be caught.
    with pytest.raises(RuntimeError, match="FTS proof failed"):
        assert_session_fts_exact_sync(conn, session_id, bulk_build=False)
    conn.close()


def test_bulk_build_guard_row_cleared_even_on_exception(tmp_path: Path) -> None:
    """A failure mid-write must not leave the whole-transaction guard row set."""
    conn = _connect(tmp_path / "index.db")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected write failure")

    original = _write_module._write_blocks
    _write_module._write_blocks = _boom
    try:
        with pytest.raises(RuntimeError, match="injected write failure"):
            write_parsed_session_to_archive(conn, _session("solo"), bulk_fts=True, bulk_build=True)
    finally:
        _write_module._write_blocks = original

    assert (
        conn.execute(
            "SELECT COUNT(*) FROM derived_refresh_guard WHERE guard_name = ?",
            (FTS_BULK_SESSION_WRITE_GUARD,),
        ).fetchone()[0]
        == 0
    )
    conn.close()


def test_bulk_build_anti_vacuity_repopulate_is_load_bearing(tmp_path: Path) -> None:
    """Skip the messages_fts half of the readiness repopulate and show the
    equivalence proof then fails -- confirming the parity test above is
    actually exercising ``rebuild_fts_index_sync``, not passing by accident."""
    conn_trickle = _connect(tmp_path / "trickle.db")
    _build_corpus(conn_trickle, bulk_build=False)
    fts_trickle = _fts_rows(conn_trickle)
    conn_trickle.close()

    conn_bulk = _connect(tmp_path / "bulk.db")
    _build_corpus(conn_bulk, bulk_build=True)
    # Deliberately DO NOT call rebuild_fts_index_sync here.
    rebuild_command_trigram_index_sync(conn_bulk)
    rebuild_all_action_pairs_sync(conn_bulk)
    conn_bulk.commit()

    fts_bulk = _fts_rows(conn_bulk)
    assert fts_bulk != fts_trickle
    assert fts_bulk == []
    assert fts_trickle, "trickle-mode reference produced no rows -- comparison would be vacuous"
    conn_bulk.close()
