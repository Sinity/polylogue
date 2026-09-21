"""FTS readiness reports the current authoritative input/output relation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.daemon.fts_status import fts_readiness_info
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def test_exact_coverage_counts_tool_blocks_as_indexable(tmp_path: Path) -> None:
    """Exact coverage must use the FTS-population predicate (search_text != '').

    A tool_use block carries a derived search_text but a NULL display text. The
    FTS index includes it, so the exact source count must include it too —
    otherwise indexed/source exceeds 100%.
    """
    db = tmp_path / "index.db"
    initialize_archive_database(db, ArchiveTier.INDEX)
    conn = sqlite3.connect(db)
    try:
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="cov-tool-1",
            title="tool coverage",
            messages=[
                ParsedMessage(
                    provider_message_id="u1",
                    role=Role.USER,
                    text="run the build",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="run the build")],
                ),
                ParsedMessage(
                    provider_message_id="a1",
                    role=Role.ASSISTANT,
                    text=None,
                    position=1,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="exec_command",
                            tool_id="t1",
                            tool_input={"command": "make build"},
                        ),
                    ],
                ),
            ],
        )
        write_parsed_session_to_archive(conn, session)
        conn.commit()
        text_blocks = int(conn.execute("SELECT COUNT(*) FROM blocks WHERE text IS NOT NULL").fetchone()[0])
        search_blocks = int(conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0])
    finally:
        conn.close()

    # The tool block makes the two predicates diverge; the exact coverage must
    # use search_text (the FTS predicate), not text.
    assert search_blocks > text_blocks

    fts = fts_readiness_info(db, exact=True)
    assert fts["coverage_pct"] == 100.0
    assert fts["messages_ready"] is True


def test_genuinely_empty_archive_reports_coverage_as_unmeasured_not_exact(tmp_path: Path) -> None:
    """polylogue-oitx: a freshly initialized, genuinely empty archive has a
    zero-denominator coverage_pct (0 indexable rows). ``invariant_ready``
    only proves triggers/tables exist -- it is not evidence of measured
    coverage, so this must report ``None`` (unmeasured), never a fabricated
    ``100.0``/``0.0`` derived from ``invariant_ready`` alone.
    """
    db = tmp_path / "index.db"
    initialize_archive_database(db, ArchiveTier.INDEX)

    fts = fts_readiness_info(db, exact=False)

    assert fts["message_indexable_count"] == 0
    assert fts["message_indexed_count"] == 0
    assert fts["coverage_pct"] is None


def test_genuinely_empty_archive_reports_coverage_as_unmeasured_exact(tmp_path: Path) -> None:
    """Same as above, through the exact (invariant-snapshot) path."""
    db = tmp_path / "index.db"
    initialize_archive_database(db, ArchiveTier.INDEX)

    fts = fts_readiness_info(db, exact=True)

    assert fts["message_indexable_count"] == 0
    assert fts["message_indexed_count"] == 0
    assert fts["coverage_pct"] is None


def test_unreadable_archive_index_reports_nothing_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """polylogue-bu47u: a failed readiness query certifies nothing.

    A readiness flag used to return ``True`` from inside the
    ``except sqlite3.Error`` handler while every sibling key returned the
    not-ready value -- a positive readiness claim emitted by an error path.

    Anti-vacuity: restore ``"messages_ready": True`` (or the fabricated
    ``coverage_pct: 0.0``) in that handler and this fails.
    """
    from polylogue.daemon import fts_status

    index = tmp_path / "index.db"
    initialize_archive_database(index, ArchiveTier.INDEX)

    def explode(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.Error("simulated readiness query failure")

    monkeypatch.setattr(fts_status, "open_readonly_connection", explode)

    payload = fts_status._archive_readiness_info(index, exact=False)

    assert payload is not None
    assert payload["messages_ready"] is False
    assert payload["invariant_ready"] is False
    assert payload["coverage_pct"] is None
    assert payload["coverage_exact"] is False
