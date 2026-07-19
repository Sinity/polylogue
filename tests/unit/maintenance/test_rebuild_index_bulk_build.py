"""``maintenance/rebuild_index.py``'s bulk-build derived-state lifecycle
(polylogue-v6i3): the two direct-file helpers the offline rebuild orchestrator
calls around a bulk-build replay -- ``_clear_bulk_build_derived_stores`` (once
per resumed operation, defensive idempotent bookkeeping) and
``_repopulate_bulk_build_derived_state`` (once at readiness, the real
archive-wide repopulate that retires the manual pre-promote recovery script).

Tested at the direct-function level (mirroring
``test_planner_statistics_seed.py``'s ``_refresh_generation_planner_
statistics`` pattern) rather than through the full ``RebuildLease``/generation
orchestration, which is exercised elsewhere and is not what these two
functions' correctness depends on.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.maintenance.archive_verification import verify_archive
from polylogue.maintenance.rebuild_index import (
    _clear_bulk_build_derived_stores,
    _repopulate_bulk_build_derived_state,
)
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _session(session_id: str) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_id,
        title=f"session {session_id}",
        messages=[
            ParsedMessage(
                provider_message_id="m0",
                role=Role.USER,
                text="hello searchable text",
                position=0,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=False,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hello searchable text")],
            ),
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                text=None,
                position=1,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=False,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Bash",
                        tool_id=f"{session_id}-tool",
                        tool_input={"command": "echo hi"},
                    ),
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        tool_id=f"{session_id}-tool",
                        text="hi",
                        is_error=False,
                        exit_code=0,
                    ),
                ],
            ),
        ],
    )


def test_clear_bulk_build_derived_stores_empties_populated_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    conn = _connect(db_path)
    write_parsed_session_to_archive(conn, _session("s1"))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM blocks_command_trigram_docsize").fetchone()[0] > 0
    action_pairs_before = conn.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0]
    assert action_pairs_before > 0
    blocks_before = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
    conn.close()

    _clear_bulk_build_derived_stores(db_path)

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM blocks_command_trigram_docsize").fetchone()[0] == 0
    # Only the two named surfaces clear here; action_pairs/blocks are a
    # readiness-time concern, not a resume-clear concern.
    assert conn.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0] == action_pairs_before
    assert conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == blocks_before
    conn.close()


def test_clear_bulk_build_derived_stores_idempotent_on_already_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    conn = _connect(db_path)
    conn.commit()
    conn.close()

    _clear_bulk_build_derived_stores(db_path)
    _clear_bulk_build_derived_stores(db_path)  # must not raise on an already-empty table

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0
    conn.close()


def test_repopulate_bulk_build_derived_state_produces_exact_parity(tmp_path: Path) -> None:
    """A corpus written entirely in bulk_build mode (derived surfaces
    deliberately empty) must reach exact archive-wide parity after one
    repopulate call -- the same check ``rebuild_index_from_source`` runs
    right before readiness."""
    db_path = tmp_path / "index.db"
    conn = _connect(db_path)
    for label in ("alpha", "beta", "gamma"):
        write_parsed_session_to_archive(conn, _session(label), bulk_fts=True, bulk_build=True)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0] == 0
    conn.close()

    _repopulate_bulk_build_derived_state(db_path)

    report = verify_archive(tmp_path, checks=["fts-parity"])
    assert not report.blocking, [check.summary for check in report.checks]

    conn = sqlite3.connect(db_path)
    text_block_count = conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert text_block_count > 0
    assert fts_count == text_block_count

    tool_use_count = conn.execute(
        "SELECT COUNT(*) FROM blocks WHERE block_type = 'tool_use' AND tool_detail_text != ' '"
    ).fetchone()[0]
    trigram_count = conn.execute("SELECT COUNT(*) FROM blocks_command_trigram_docsize").fetchone()[0]
    assert tool_use_count > 0
    assert trigram_count == tool_use_count

    action_pairs_count = conn.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0]
    assert action_pairs_count == tool_use_count
    conn.close()
