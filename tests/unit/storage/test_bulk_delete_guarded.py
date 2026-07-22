"""Regression: ``ArchiveStore.delete_sessions`` must not detonate per-row
derived-refresh triggers (polylogue-meoz).

Background: a plain per-session ``DELETE FROM sessions`` fires
``blocks_action_pairs_ad`` once per deleted ``blocks`` row (both the explicit
cascade and the FK ``ON DELETE CASCADE`` invoke AFTER DELETE triggers per
row); each firing deletes+rebuilds the *entire session's* ``action_pairs`` via
two window-function scans and re-derives ``delegation_facts``. Live incident
2026-07-21: deleting 91 sessions ran 3h, 375GB reads, zero commit, before
being killed.

The fix wraps the delete in both ``derived_refresh_guard`` rows
(``'session-write'``, ``'fts-bulk-session-write'``) and performs the derived
maintenance explicitly, one pass, mirroring ``_bulk_fts_session_guard``'s
delete-then-guard-then-mutate shape (``write.py``).

These tests prove: (a) FTS/identity/trigram coherence after a bulk delete
through the PRODUCT delete API; (b) ``action_pairs``/``delegation_facts`` rows
for the deleted sessions are gone; (c) anti-vacuity -- a *canary* trigger
carrying the exact same ``WHEN NOT EXISTS (... derived_refresh_guard ...)``
condition as the real ``blocks_action_pairs_ad``/``blocks_command_trigram_ad``
triggers never fires during the delete. ``sqlite3.Connection.set_trace_callback``
was tried first and rejected: it only reports the outer statement text
(``DELETE FROM sessions ...``), never the SQL text executed *inside* a
trigger body, so it cannot distinguish a guarded delete from an unguarded one
-- verified empirically against a minimal repro before choosing the canary
approach.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

# Mirrors the exact WHEN-clause guard names the real triggers gate on (see
# ``blocks_action_pairs_ad`` and ``blocks_command_trigram_ad`` in
# ``storage/sqlite/archive_tiers/index.py``).
_SESSION_WRITE_GUARD = "session-write"
_FTS_BULK_GUARD = "fts-bulk-session-write"


def _install_canary_triggers(conn: sqlite3.Connection) -> None:
    """Install AFTER-DELETE canary triggers on ``blocks`` carrying the exact
    same guard WHEN-clauses as the real ``blocks_action_pairs_ad`` and
    ``blocks_command_trigram_ad`` production triggers.

    A canary fires precisely when the corresponding expensive real trigger
    body would also have fired for that row -- so a nonzero canary count
    after ``delete_sessions`` directly proves the per-row rebuild machinery
    was NOT suppressed, without needing to inspect trigger-body SQL text
    (which SQLite's trace callback does not expose).
    """
    conn.execute("CREATE TABLE IF NOT EXISTS _canary_counts (name TEXT PRIMARY KEY, n INTEGER NOT NULL)")
    conn.execute("INSERT OR REPLACE INTO _canary_counts (name, n) VALUES ('session_write', 0), ('fts_bulk', 0)")
    conn.execute(
        f"""
        CREATE TRIGGER IF NOT EXISTS _canary_action_pairs_ad
        AFTER DELETE ON blocks
        WHEN NOT EXISTS (SELECT 1 FROM derived_refresh_guard WHERE guard_name = '{_SESSION_WRITE_GUARD}')
        BEGIN
            UPDATE _canary_counts SET n = n + 1 WHERE name = 'session_write';
        END;
        """
    )
    conn.execute(
        f"""
        CREATE TRIGGER IF NOT EXISTS _canary_trigram_ad
        AFTER DELETE ON blocks
        WHEN old.block_type = 'tool_use' AND old.tool_detail_text != ' '
         AND NOT EXISTS (SELECT 1 FROM derived_refresh_guard WHERE guard_name = '{_FTS_BULK_GUARD}')
        BEGIN
            UPDATE _canary_counts SET n = n + 1 WHERE name = 'fts_bulk';
        END;
        """
    )
    conn.commit()


def _canary_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return dict(conn.execute("SELECT name, n FROM _canary_counts").fetchall())


def _trigram_ghost_posting_count(conn: sqlite3.Connection) -> int:
    """Indexed trigram rows whose content-table (``blocks``) row is gone."""
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM blocks_command_trigram_docsize AS d
            LEFT JOIN blocks AS b ON b.rowid = d.id
            WHERE b.rowid IS NULL
            """
        ).fetchone()[0]
    )


def _fts_identity_orphan_count(conn: sqlite3.Connection) -> int:
    """``messages_fts_identity`` ledger rows whose ``blocks`` row is gone."""
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM messages_fts_identity AS i
            LEFT JOIN blocks AS b ON b.rowid = i.rowid
            WHERE b.rowid IS NULL
            """
        ).fetchone()[0]
    )


def _fts_docsize_ghost_count(conn: sqlite3.Connection) -> int:
    """``messages_fts`` (contentless) docsize rows whose ``blocks`` row is gone."""
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM messages_fts_docsize AS d
            LEFT JOIN blocks AS b ON b.rowid = d.id
            WHERE b.rowid IS NULL
            """
        ).fetchone()[0]
    )


def _tool_session(provider_session_id: str, *, n_pairs: int) -> ParsedSession:
    """A session with ``n_pairs`` tool_use/tool_result pairs across distinct
    messages -- enough indexable blocks that a per-row trigger regression
    would fire (and be traced) multiple times, not just once."""
    messages: list[ParsedMessage] = [
        ParsedMessage(
            provider_message_id="m0",
            role=Role.USER,
            text="please run some commands",
            position=0,
            variant_index=0,
            is_active_path=True,
            is_active_leaf=False,
            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="please run some commands")],
        )
    ]
    for i in range(n_pairs):
        tool_id = f"tool-{provider_session_id}-{i}"
        messages.append(
            ParsedMessage(
                provider_message_id=f"use-{i}",
                role=Role.ASSISTANT,
                text=f"running command {i}",
                position=2 * i + 1,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=False,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Bash",
                        tool_id=tool_id,
                        tool_input={"command": f"echo pair-{i}"},
                    )
                ],
            )
        )
        messages.append(
            ParsedMessage(
                provider_message_id=f"result-{i}",
                role=Role.TOOL,
                text=f"pair-{i}",
                position=2 * i + 2,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=False,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        tool_id=tool_id,
                        text=f"pair-{i}",
                        is_error=False,
                        exit_code=0,
                    )
                ],
            )
        )
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=provider_session_id,
        title=provider_session_id,
        messages=messages,
    )


def test_delete_sessions_bulk_leaves_fts_trigram_and_action_pairs_coherent(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    session_ids: list[str] = []
    with ArchiveStore(root) as facade:
        for i in range(3):
            session_ids.append(facade.write_parsed(_tool_session(f"bulk-delete-{i}", n_pairs=4)))

    index_db_path = root / "index.db"
    conn = sqlite3.connect(index_db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Sanity: the sessions actually produced action_pairs, indexable FTS
        # rows, and trigram-indexed tool_use blocks before the delete -- a
        # test that starts from an already-empty state would prove nothing.
        action_pairs_before = conn.execute(
            "SELECT COUNT(*) FROM action_pairs WHERE session_id IN ({})".format(", ".join("?" for _ in session_ids)),
            session_ids,
        ).fetchone()[0]
        assert action_pairs_before == 3 * 4  # one action_pairs row per tool_use/tool_result pair
        fts_rows_before = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
        assert fts_rows_before > 0
        trigram_rows_before = conn.execute("SELECT COUNT(*) FROM blocks_command_trigram_docsize").fetchone()[0]
        assert trigram_rows_before == 3 * 4  # one indexed tool_use block per pair

        # Manually seed a delegation_facts row so the explicit
        # ``DELETE FROM delegation_facts WHERE parent_session_id = ?``
        # ``delete_sessions`` now issues has something real to remove --
        # exercising a genuine production statement, not a mock.
        conn.execute(
            """
            INSERT INTO delegation_facts (
                delegation_id, parent_session_id, mapping_state, result_status, parent_origin
            ) VALUES (?, ?, 'mapped', 'ok', 'codex-session')
            """,
            (f"deleg-{session_ids[0]}", session_ids[0]),
        )
        conn.commit()
    finally:
        conn.close()

    with ArchiveStore(root) as facade:
        deleted = facade.delete_sessions(tuple(session_ids))

    assert deleted == 3

    conn = sqlite3.connect(index_db_path)
    conn.row_factory = sqlite3.Row
    try:
        remaining_sessions = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id IN ({})".format(", ".join("?" for _ in session_ids)),
            session_ids,
        ).fetchone()[0]
        assert remaining_sessions == 0

        # (a) FTS coherence: no dangling docsize/identity/trigram postings.
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 0
        assert _fts_docsize_ghost_count(conn) == 0
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_identity").fetchone()[0] == 0
        assert _fts_identity_orphan_count(conn) == 0
        assert conn.execute("SELECT COUNT(*) FROM blocks_command_trigram_docsize").fetchone()[0] == 0
        assert _trigram_ghost_posting_count(conn) == 0

        # (b) action_pairs / delegation_facts rows for the deleted sessions
        # are gone.
        remaining_action_pairs = conn.execute(
            "SELECT COUNT(*) FROM action_pairs WHERE session_id IN ({})".format(", ".join("?" for _ in session_ids)),
            session_ids,
        ).fetchone()[0]
        assert remaining_action_pairs == 0
        remaining_delegation_facts = conn.execute(
            "SELECT COUNT(*) FROM delegation_facts WHERE parent_session_id IN ({})".format(
                ", ".join("?" for _ in session_ids)
            ),
            session_ids,
        ).fetchone()[0]
        assert remaining_delegation_facts == 0

        # Guard rows must never leak past the transaction they protected.
        assert conn.execute("SELECT COUNT(*) FROM derived_refresh_guard").fetchone()[0] == 0
    finally:
        conn.close()


def test_delete_sessions_bulk_never_fires_unguarded_per_row_canary(tmp_path: Path) -> None:
    """Anti-vacuity: install canary triggers carrying the exact WHEN-clause
    guard conditions of the real ``blocks_action_pairs_ad`` /
    ``blocks_command_trigram_ad`` triggers, then prove neither one fires
    during a bulk ``delete_sessions`` call through the PRODUCT API.

    What would make this fail: removing (or narrowing the scope of) either
    ``INSERT OR REPLACE INTO derived_refresh_guard(...)`` call this PR adds to
    ``ArchiveStore.delete_sessions``. With the ``'session-write'`` guard
    absent, ``blocks_action_pairs_ad`` (and this test's mirroring canary)
    fires once per deleted ``blocks`` row; with ``'fts-bulk-session-write'``
    absent, ``blocks_command_trigram_ad`` (and its canary) fires once per
    deleted tool_use block. Either regression flips the corresponding canary
    count from 0 to a positive number equal to the number of blocks deleted
    while that guard was unset.
    """
    root = tmp_path / "archive"
    session_ids: list[str] = []
    with ArchiveStore(root) as facade:
        for i in range(2):
            session_ids.append(facade.write_parsed(_tool_session(f"canary-delete-{i}", n_pairs=3)))

    index_db_path = root / "index.db"
    setup_conn = sqlite3.connect(index_db_path)
    try:
        _install_canary_triggers(setup_conn)
    finally:
        setup_conn.close()

    with ArchiveStore(root) as facade:
        deleted = facade.delete_sessions(tuple(session_ids))
    assert deleted == 2

    verify_conn = sqlite3.connect(index_db_path)
    try:
        counts = _canary_counts(verify_conn)
    finally:
        verify_conn.close()
    assert counts == {"session_write": 0, "fts_bulk": 0}, (
        f"unguarded per-row trigger canary fired during bulk delete: {counts}"
    )
