"""Guard-gated bulk FTS mode for whale prefix-sharing lineage cascades (crd8).

Background: a prefix-sharing child (fork/resume/auto-compaction) ingested
before its parent is stored whole; once the parent arrives,
``_resolve_session_graph`` -> ``_reextract_prefix_tail_db`` deletes the now
-inherited prefix rows from the child. For a whale lineage session with a
huge prefix, deleting those ``blocks`` rows one at a time fires
``messages_fts_ad`` per row -- measured as a 25+ minute single-DELETE stall
on the live rebuild (bead polylogue-crd8).

``bulk_fts=True`` (threaded from the offline replay path only; ordinary
daemon writes default to ``False`` and are therefore byte-for-byte
unchanged) makes ``write_parsed_session_to_archive`` set a *dedicated*
``derived_refresh_guard`` row ('fts-bulk-session-write') around the
dependent-delete call, gating only the ``messages_fts_{ai,ad,au}`` trigger
BODIES (the triggers stay present in ``sqlite_master`` throughout -- see
``_bulk_fts_session_guard`` in ``write.py``), and performs one explicit
session-scoped FTS delete-then-reinsert bracketing the mutation instead.

These tests prove: (a) mode-off keeps today's per-row-trigger-maintained FTS
content: (b) mode-on produces byte-identical FTS content for the same
session; (c) the guard row never leaks past an exception; (d) the apply-side
``assert_session_fts_exact_sync`` parity+trigger-presence proof still passes
after a bulk-mode apply; (e) the explicit bulk re-insert is load-bearing, not
vacuous -- removing it makes the parity proof fail.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.fts.sql import FTS_BULK_SESSION_WRITE_GUARD
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


def _fts_rows_for_session(conn: sqlite3.Connection, session_id: str) -> list[tuple[object, ...]]:
    """Session-scoped ``messages_fts`` content, independent of literal rowid."""
    rows = conn.execute(
        """
        SELECT block_id, message_id, session_id, block_type, text
        FROM messages_fts
        WHERE rowid IN (SELECT rowid FROM blocks WHERE session_id = ?)
        ORDER BY block_id
        """,
        (session_id,),
    ).fetchall()
    return sorted(tuple(row) for row in rows)


def _write_partial_tail_scenario(conn: sqlite3.Connection, *, bulk_fts: bool) -> str:
    """Child stores a divergent tail after a shared prefix (`_delete_prefix_message_dependents`)."""
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
    write_parsed_session_to_archive(conn, parent, bulk_fts=bulk_fts)
    return child_id


_Scenario = Callable[..., str]


def _write_full_tail_scenario(conn: sqlite3.Connection, *, bulk_fts: bool) -> str:
    """Child is entirely inherited (`_delete_all_session_message_dependents`)."""
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
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _msg("p0", Role.USER, "hello", 0),
            _msg("p1", Role.ASSISTANT, "hi there", 1),
        ],
    )
    write_parsed_session_to_archive(conn, parent, bulk_fts=bulk_fts)
    return child_id


@pytest.mark.parametrize("scenario", [_write_partial_tail_scenario, _write_full_tail_scenario])
def test_bulk_fts_off_reextract_matches_trigger_maintained_fts(tmp_path: Path, scenario: _Scenario) -> None:
    """Mode OFF (today's default): per-row triggers keep FTS in exact sync."""
    conn = _connect(tmp_path / "index.db")
    child_id = scenario(conn, bulk_fts=False)
    assert_session_fts_exact_sync(conn, child_id)
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM derived_refresh_guard WHERE guard_name = ?",
            (FTS_BULK_SESSION_WRITE_GUARD,),
        ).fetchone()[0]
        == 0
    )
    conn.close()


@pytest.mark.parametrize("scenario", [_write_partial_tail_scenario, _write_full_tail_scenario])
def test_bulk_fts_on_produces_identical_fts_rows_as_off(tmp_path: Path, scenario: _Scenario) -> None:
    """THE key equivalence proof: bulk mode must not change FTS content, only how it gets there."""
    conn_off = _connect(tmp_path / "off.db")
    child_id_off = scenario(conn_off, bulk_fts=False)
    rows_off = _fts_rows_for_session(conn_off, child_id_off)
    assert_session_fts_exact_sync(conn_off, child_id_off)
    conn_off.close()

    conn_on = _connect(tmp_path / "on.db")
    child_id_on = scenario(conn_on, bulk_fts=True)
    rows_on = _fts_rows_for_session(conn_on, child_id_on)
    assert_session_fts_exact_sync(conn_on, child_id_on)
    # The guard row must never leak past the write it protected.
    assert (
        conn_on.execute(
            "SELECT COUNT(*) FROM derived_refresh_guard WHERE guard_name = ?",
            (FTS_BULK_SESSION_WRITE_GUARD,),
        ).fetchone()[0]
        == 0
    )
    conn_on.close()

    assert child_id_off == child_id_on
    assert rows_on == rows_off
    if scenario is _write_partial_tail_scenario:
        # The full-tail scenario legitimately ends with zero surviving blocks
        # (the whole child was inherited) -- only the partial-tail scenario's
        # surviving divergent-tail rows prove this isn't a vacuous empty==empty
        # comparison.
        assert rows_off, "scenario produced no FTS rows -- test would pass vacuously"


def test_bulk_fts_guard_row_cleared_even_on_exception(tmp_path: Path) -> None:
    """A failure mid-guard must not leave the dedicated guard row set."""
    conn = _connect(tmp_path / "index.db")
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
        ],
    )
    write_parsed_session_to_archive(conn, child)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected dependent-delete failure")

    original = _write_module._delete_prefix_message_dependents
    _write_module._delete_prefix_message_dependents = _boom
    try:
        parent = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="parent",
            title="parent",
            messages=[
                _msg("p0", Role.USER, "hello", 0),
                _msg("p1", Role.ASSISTANT, "hi there", 1),
            ],
        )
        with pytest.raises(RuntimeError, match="injected dependent-delete failure"):
            write_parsed_session_to_archive(conn, parent, bulk_fts=True)
    finally:
        _write_module._delete_prefix_message_dependents = original

    # write_parsed_session_to_archive runs in its own `with conn:` transaction,
    # so the failed write rolled back entirely -- including the guard-row
    # insert. Confirm no leaked guard row survives (transaction-rollback state,
    # not merely in-process cleanup).
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM derived_refresh_guard WHERE guard_name = ?",
            (FTS_BULK_SESSION_WRITE_GUARD,),
        ).fetchone()[0]
        == 0
    )
    conn.close()


def test_bulk_fts_explicit_reinsert_is_load_bearing_not_vacuous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: removing the explicit bulk re-insert must break the parity proof.

    If this test passed with the mutation applied, the equivalence tests above
    would not actually be exercising the new bulk-insert code path.
    """
    conn = _connect(tmp_path / "index.db")

    def _no_op_insert_sql(_chunk_size: int) -> str:
        # Same single ``?`` binding shape as the real insert_session_rows_sql(1)
        # call site, but never touches messages_fts.
        return "SELECT 1 WHERE ? IS NOT NULL"

    monkeypatch.setattr(_write_module, "insert_session_rows_sql", _no_op_insert_sql)

    child_id = _write_partial_tail_scenario(conn, bulk_fts=True)
    with pytest.raises(RuntimeError, match="FTS proof failed"):
        assert_session_fts_exact_sync(conn, child_id)
    conn.close()
