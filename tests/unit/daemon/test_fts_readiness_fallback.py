"""FTS readiness recomputes exact coverage when the freshness cache is poisoned.

On a fresh archive, daemon startup legitimately records ``messages_fts|ready|0|0``
over an empty index. Trigger-maintained ingest then populates the index without
ever triggering a rebuild that refreshes those counts, leaving the durable
freshness row as the untrusted ``ready|0|0`` shape. Read surfaces (``status``,
``/healthz``, ``/metrics``) funnel through ``fts_readiness_info`` and must report
the real coverage instead of dividing 0/0 and showing 0% over a populated index.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.daemon.fts_status import fts_readiness_info
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.fts.freshness import (
    record_fts_surface_stale_preserving_counts_sync,
    record_fts_surface_state_sync,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _populated_index(db: Path) -> None:
    initialize_archive_database(db, ArchiveTier.INDEX)
    conn = sqlite3.connect(db)
    try:
        session = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="fts-cov-1",
            title="coverage probe",
            messages=[
                ParsedMessage(
                    provider_message_id="u1",
                    role=Role.USER,
                    text="searchable content here",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="searchable content here")],
                ),
            ],
        )
        write_parsed_session_to_archive(conn, session)
        conn.commit()
    finally:
        conn.close()


def test_readiness_falls_back_to_exact_when_freshness_poisoned(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    _populated_index(db)

    # Poison the durable freshness cache with the empty-archive ready|0|0 shape.
    conn = sqlite3.connect(db)
    try:
        record_fts_surface_state_sync(conn, surface="messages_fts", state="ready", source_rows=0, indexed_rows=0)
        conn.commit()
    finally:
        conn.close()

    fts = fts_readiness_info(db, exact=False)

    indexable = int(cast(int, fts["message_indexable_count"]))
    indexed = int(cast(int, fts["message_indexed_count"]))
    assert indexable > 0
    assert indexed == indexable
    assert fts["coverage_pct"] == 100.0
    assert fts["messages_ready"] is True
    # The fallback recomputed exact counts rather than trusting the cache.
    assert fts["coverage_exact"] is True


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


def test_readiness_trusts_a_healthy_freshness_record_without_recompute(tmp_path: Path) -> None:
    """A trusted ready|N|N record is used directly (fast path), not recomputed."""
    db = tmp_path / "index.db"
    _populated_index(db)

    conn = sqlite3.connect(db)
    try:
        indexed = int(conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0])
        record_fts_surface_state_sync(
            conn,
            surface="messages_fts",
            state="ready",
            source_rows=indexed,
            indexed_rows=indexed,
        )
        conn.commit()
    finally:
        conn.close()

    fts = fts_readiness_info(db, exact=False)

    assert fts["messages_ready"] is True
    assert fts["coverage_pct"] == 100.0
    assert fts["coverage_exact"] is False


def test_single_session_defer_does_not_falsely_zero_archive_wide_coverage(tmp_path: Path) -> None:
    """polylogue-5eyy: a single deferred session repair must not report 0% archive-wide.

    ``archive.py``'s raw-revision-authoritative write path marks ``messages_fts``
    STALE (without touching the DB rows themselves) whenever a single session's
    FTS repair is deferred during a live write. Before the fix,
    ``record_fts_surface_state_sync`` was called with no ``source_rows``/
    ``indexed_rows`` -- its defaults are 0 -- which unconditionally clobbered the
    *entire* surface's durable counts to 0/0 even though the archive-wide FTS
    index remains fully populated and only one session's repair was deferred.
    Downstream this made every status/readiness consumer (CLI ops status,
    daemon status API, MCP status tool) report coverage_pct=0.0 and
    state=missing for the whole archive.
    """
    db = tmp_path / "index.db"
    _populated_index(db)

    conn = sqlite3.connect(db)
    try:
        indexed = int(conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0])
        assert indexed > 0
        # Real, healthy archive-wide coverage recorded (e.g. by a prior
        # invariant snapshot / bounded repair).
        record_fts_surface_state_sync(
            conn,
            surface="messages_fts",
            state="ready",
            source_rows=indexed,
            indexed_rows=indexed,
        )
        conn.commit()

        # Simulate a single session's FTS repair being deferred during a live
        # authoritative write (archive.py's ``defer_fts`` branch) -- the real
        # archive-wide FTS rows are untouched, only the freshness ledger is
        # marked stale to force a later targeted recheck.
        record_fts_surface_stale_preserving_counts_sync(
            conn,
            surface="messages_fts",
            detail="live authoritative replay deferred targeted session FTS repair",
        )
        conn.commit()
    finally:
        conn.close()

    fts = fts_readiness_info(db, exact=False)

    # The pending single-session repair legitimately means the surface isn't
    # trusted as fully "ready", but the real, still-populated archive-wide
    # counts must survive -- not be falsely reported as 0/missing.
    surfaces = fts["surfaces"]
    assert isinstance(surfaces, dict)
    messages_fts_surface = surfaces["messages_fts"]
    assert isinstance(messages_fts_surface, dict)
    assert messages_fts_surface["freshness_recorded_state"] == "stale"
    assert fts["message_indexed_count"] == indexed
    assert fts["message_indexable_count"] == indexed
    assert fts["coverage_pct"] == 100.0


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
